import os
import pandas as pd
import yfinance as yf


class DataLoader:

    def __init__(self, ticker, start_date, end_date, data_dir="data"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir
        # Sanitize ticker for filesystem (^VIX → _VIX)
        safe_ticker = ticker.replace("^", "_").replace("/", "_")
        self.file_path = os.path.join(
            data_dir, f"{safe_ticker}_{start_date}_{end_date}.csv"
        )
        os.makedirs(data_dir, exist_ok=True)

    def download_data(self):
        print(f"Downloading {self.ticker}...")
        try:
            df = yf.download(
                self.ticker, start=self.start_date, end=self.end_date,
                progress=False, auto_adjust=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            return df
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_data(self):
        if os.path.exists(self.file_path):
            return pd.read_csv(self.file_path, index_col=0, parse_dates=True)
        df = self.download_data()
        if df is not None and not df.empty:
            df.to_csv(self.file_path)
        return df


class MarketContextLoader:


    def __init__(self, start_date, end_date, data_dir="data"):
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = data_dir

    def get_market_data(self):
        spy = DataLoader("SPY", self.start_date, self.end_date, self.data_dir).get_data()
        vix = DataLoader("^VIX", self.start_date, self.end_date, self.data_dir).get_data()
        if spy is None or vix is None or spy.empty or vix.empty:
            return None
        return {"spy": spy, "vix": vix}
