"""DataLoader caching behaviour (FR1) — no network access needed."""

import pandas as pd

from src.data_loader import DataLoader
from tests.conftest import make_ohlcv


def test_cached_csv_is_used_instead_of_download(tmp_path):
    df = make_ohlcv(n=50)
    loader = DataLoader("FAKE", "2022-01-01", "2022-12-31", data_dir=str(tmp_path))
    df.to_csv(loader.file_path)

    # Poison download_data: if the loader tries the network, the test fails.
    loader.download_data = lambda: (_ for _ in ()).throw(
        AssertionError("tried to download despite cache"))

    got = loader.get_data()
    assert len(got) == len(df)
    assert list(got.columns) == list(df.columns)
    assert isinstance(got.index, pd.DatetimeIndex)


def test_ticker_sanitized_for_filesystem(tmp_path):
    loader = DataLoader("^VIX", "2022-01-01", "2022-12-31", data_dir=str(tmp_path))
    assert "^" not in loader.file_path
    assert "_VIX" in loader.file_path
