import numpy as np
import pandas as pd


class DataPreprocessor:

    BASE_FEATURES = [
        "Return_1d", "Return_5d", "Return_10d",
        "LogReturn_1d",
        "Volatility_5d", "Volatility_10d", "Volatility_20d",
        "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Position", "BB_Width",
        "ATR_14_Norm",
        "OBV_Norm",
        "Volume_Ratio",
        "Close_SMA10_Ratio", "Close_SMA50_Ratio",
        "SMA10_SMA50_Ratio",
        "Momentum_10d",
        "Stoch_K", "Stoch_D",
    ]

    # Extra features when market context is provided
    MARKET_FEATURES = [
        "SPY_Return_1d", "SPY_Return_5d",
        "VIX_Level", "VIX_Change_1d", "VIX_MA20_Ratio",
        "Excess_Return_1d", "RelStrength_20d",
        "Beta_60d",
    ]

    def __init__(self, data):
        self.raw_data = data.copy()
        self.feature_df = None
        self.market_data = None
        self.feature_columns = list(self.BASE_FEATURES)

    # ---------- public API ----------

    def set_market_context(self, market_data):
        """market_data: dict with keys 'spy' and 'vix' (DataFrames with Close)."""
        self.market_data = market_data
        return self

    # ---------- indicator helpers ----------

    @staticmethod
    def _rsi(close, period=14):
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(high, low, close, period=14):
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def _obv(close, volume):
        direction = np.sign(close.diff()).fillna(0)
        return (direction * volume).cumsum()

    @staticmethod
    def _stochastic(high, low, close, k_period=14, d_period=3):
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        k = 100 * (close - low_min) / (high_max - low_min).replace(0, np.nan)
        d = k.rolling(window=d_period).mean()
        return k, d

    # ---------- main pipeline ----------

    def add_technical_indicators(self):
        df = self.raw_data.copy()

        if "Volume" not in df.columns:
            df["Volume"] = 0

        close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

        df["Return_1d"] = close.pct_change()
        df["Return_5d"] = close.pct_change(5)
        df["Return_10d"] = close.pct_change(10)
        df["LogReturn_1d"] = np.log(close / close.shift(1))

        df["Volatility_5d"] = df["Return_1d"].rolling(5).std()
        df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
        df["Volatility_20d"] = df["Return_1d"].rolling(20).std()

        df["RSI_14"] = self._rsi(close, 14)

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        upper = sma20 + 2 * std20
        lower = sma20 - 2 * std20
        df["BB_Position"] = (close - lower) / (upper - lower).replace(0, np.nan)
        df["BB_Width"] = (upper - lower) / sma20

        df["ATR_14_Norm"] = self._atr(high, low, close, 14) / close

        obv = self._obv(close, volume)
        df["OBV_Norm"] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std()

        df["Volume_Ratio"] = volume / volume.rolling(20).mean()

        sma10 = close.rolling(10).mean()
        sma50 = close.rolling(50).mean()
        df["Close_SMA10_Ratio"] = close / sma10
        df["Close_SMA50_Ratio"] = close / sma50
        df["SMA10_SMA50_Ratio"] = sma10 / sma50

        df["Momentum_10d"] = close / close.shift(10) - 1

        df["Stoch_K"], df["Stoch_D"] = self._stochastic(high, low, close)

        # ---- market context (optional) ----
        if self.market_data is not None:
            df = self._add_market_features(df, close)
            self.feature_columns = list(self.BASE_FEATURES) + list(self.MARKET_FEATURES)
        else:
            self.feature_columns = list(self.BASE_FEATURES)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        self.feature_df = df
        return df

    def _add_market_features(self, df, close):
        spy = self.market_data["spy"]["Close"].copy()
        vix = self.market_data["vix"]["Close"].copy()
        spy.name = "_SPY_Close"
        vix.name = "_VIX_Close"

        # Align on date index; ffill small gaps (e.g. half-day closures)
        merged = df.join(spy, how="left").join(vix, how="left")
        merged["_SPY_Close"] = merged["_SPY_Close"].ffill()
        merged["_VIX_Close"] = merged["_VIX_Close"].ffill()

        spy_close = merged["_SPY_Close"]
        vix_close = merged["_VIX_Close"]

        merged["SPY_Return_1d"] = spy_close.pct_change()
        merged["SPY_Return_5d"] = spy_close.pct_change(5)

        merged["VIX_Level"] = vix_close
        merged["VIX_Change_1d"] = vix_close.pct_change()
        merged["VIX_MA20_Ratio"] = vix_close / vix_close.rolling(20).mean()

        ticker_ret = close.pct_change()
        merged["Excess_Return_1d"] = ticker_ret - merged["SPY_Return_1d"]

        # Relative strength: 20-day cumulative ticker return vs SPY
        merged["RelStrength_20d"] = (
            (1 + ticker_ret).rolling(20).apply(np.prod, raw=True)
            / (1 + merged["SPY_Return_1d"]).rolling(20).apply(np.prod, raw=True)
        )

        # Rolling 60-day beta of ticker vs SPY
        window = 60
        cov = ticker_ret.rolling(window).cov(merged["SPY_Return_1d"])
        var = merged["SPY_Return_1d"].rolling(window).var()
        merged["Beta_60d"] = cov / var.replace(0, np.nan)

        # Drop helper cols
        return merged.drop(columns=["_SPY_Close", "_VIX_Close"])

    # ---------- targets ----------

    def prepare_data_for_training(self):
        """Classification target: next-day direction."""
        if self.feature_df is None:
            self.add_technical_indicators()

        df = self.feature_df.copy()
        df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
        # NaN > x is False, not NaN, so the last row would otherwise keep a
        # fabricated "down" label even though its next-day close is unknown.
        df = df.iloc[:-1]
        df.dropna(inplace=True)

        X = df[self.feature_columns].copy()
        y = df["Target"].copy()
        return X, y

    def prepare_data_for_regression(self):
        """Regression target: next-day log return."""
        if self.feature_df is None:
            self.add_technical_indicators()

        df = self.feature_df.copy()
        next_close = df["Close"].shift(-1)
        df["TargetReturn"] = np.log(next_close / df["Close"])
        df.dropna(inplace=True)

        X = df[self.feature_columns].copy()
        y = df["TargetReturn"].copy()
        return X, y

    # ---------- CV ----------

    @staticmethod
    def walk_forward_splits(n_samples, n_splits=5, min_train_frac=0.5):
        start = int(n_samples * min_train_frac)
        remaining = n_samples - start
        fold_size = remaining // n_splits
        if fold_size < 1:
            raise ValueError("Not enough data for the requested number of splits.")
        for i in range(n_splits):
            train_end = start + i * fold_size
            test_end = train_end + fold_size if i < n_splits - 1 else n_samples
            yield np.arange(0, train_end), np.arange(train_end, test_end)
