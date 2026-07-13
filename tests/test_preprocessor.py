"""Feature engineering, target construction and walk-forward splits.

These tests back up the thesis claims directly: NFR4 (no look-ahead
leakage) and the 22/30-feature design from Chapter 4.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import DataPreprocessor
from tests.conftest import make_ohlcv


class TestFeatureEngineering:
    def test_all_base_features_present(self, feature_data):
        _, feature_df, _, _ = feature_data
        for col in DataPreprocessor.BASE_FEATURES:
            assert col in feature_df.columns, f"missing feature {col}"

    def test_feature_count_without_market_context(self, feature_data):
        proc, _, X, _ = feature_data
        assert len(proc.feature_columns) == 22
        assert X.shape[1] == 22

    def test_no_nan_or_inf_in_features(self, feature_data):
        _, feature_df, _, _ = feature_data
        values = feature_df[DataPreprocessor.BASE_FEATURES].values
        assert np.isfinite(values).all()

    def test_rsi_within_bounds(self, feature_data):
        _, feature_df, _, _ = feature_data
        assert feature_df["RSI_14"].between(0, 100).all()

    def test_stochastic_within_bounds(self, feature_data):
        _, feature_df, _, _ = feature_data
        assert feature_df["Stoch_K"].between(0, 100).all()
        assert feature_df["Stoch_D"].between(0, 100).all()

    def test_bb_position_mostly_inside_bands(self, feature_data):
        # By construction ~95% of closes sit inside 2-sigma bands
        _, feature_df, _, _ = feature_data
        inside = feature_df["BB_Position"].between(-0.5, 1.5).mean()
        assert inside > 0.9

    def test_market_features_added_with_context(self, ohlcv):
        spy = make_ohlcv(n=len(ohlcv), seed=7)
        vix = make_ohlcv(n=len(ohlcv), seed=8)
        spy.index = ohlcv.index
        vix.index = ohlcv.index
        proc = DataPreprocessor(ohlcv)
        proc.set_market_context({"spy": spy, "vix": vix})
        feature_df = proc.add_technical_indicators()
        assert len(proc.feature_columns) == 30
        for col in DataPreprocessor.MARKET_FEATURES:
            assert col in feature_df.columns, f"missing market feature {col}"
        values = feature_df[proc.feature_columns].values
        assert np.isfinite(values).all()


class TestTargets:
    def test_classification_target_is_next_day_direction(self, feature_data):
        _, feature_df, X, y = feature_data
        close = feature_df["Close"]
        expected = (close.shift(-1) > close).astype(int).iloc[:-1]
        assert (y.values == expected.loc[y.index].values).all()

    def test_classification_drops_unlabelable_last_row(self, feature_data):
        _, feature_df, X, _ = feature_data
        # The very last feature row has no next-day close, so it must not
        # appear in the training set.
        assert feature_df.index[-1] not in X.index

    def test_regression_target_is_log_return(self, ohlcv, regression_data):
        X, y = regression_data
        proc = DataPreprocessor(ohlcv)
        df = proc.add_technical_indicators()
        close = df["Close"]
        expected = np.log(close.shift(-1) / close).dropna()
        assert np.allclose(y.values, expected.loc[y.index].values)

    def test_classification_and_regression_same_length(self, feature_data,
                                                       regression_data):
        _, _, Xc, _ = feature_data
        Xr, _ = regression_data
        assert len(Xc) == len(Xr)


class TestWalkForwardSplits:
    def test_no_look_ahead_leakage(self):
        for tr, te in DataPreprocessor.walk_forward_splits(500, n_splits=5):
            assert tr.max() < te.min(), "training window overlaps the future"

    def test_windows_expand(self):
        train_sizes = [len(tr) for tr, _ in
                       DataPreprocessor.walk_forward_splits(500, n_splits=5)]
        assert train_sizes == sorted(train_sizes)
        assert len(set(train_sizes)) == len(train_sizes)

    def test_test_folds_cover_second_half_without_overlap(self):
        n = 500
        seen = []
        for _, te in DataPreprocessor.walk_forward_splits(n, n_splits=5):
            seen.extend(te.tolist())
        assert seen == list(range(250, n)), "test folds must tile the 2nd half"

    def test_fold_count(self):
        folds = list(DataPreprocessor.walk_forward_splits(500, n_splits=5))
        assert len(folds) == 5

    def test_raises_when_not_enough_data(self):
        with pytest.raises(ValueError):
            list(DataPreprocessor.walk_forward_splits(8, n_splits=5))
