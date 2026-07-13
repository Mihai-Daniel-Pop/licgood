"""GUI smoke tests, run offscreen (no display needed)."""

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

import gui_qt


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    gui_qt.apply_app_theme(app, dark=False)
    return app


@pytest.fixture()
def window(app):
    win = gui_qt.MainWindow()
    yield win
    win.close()


def fake_results(mode="Classification"):
    idx = pd.date_range("2024-01-01", periods=120, freq="B")
    feature_df = pd.DataFrame({"Close": np.linspace(100, 130, 120)}, index=idx)
    models = {}
    for name in ["Random Forest", "XGBoost", "LSTM", "NEAT", "Jordan", "Ensemble"]:
        if mode == "Regression":
            models[name] = {"pred_return": 0.004, "importance": None}
        else:
            models[name] = {"pred": 1, "proba": np.array([0.45, 0.55]),
                            "importance": None}
    models["Random Forest"]["importance"] = np.linspace(0.01, 0.2, 22)
    wf = {name: [0.5, 0.52, 0.54] for name in gui_qt.MODEL_COLORS}
    return {
        "ticker": "TEST", "feature_df": feature_df, "raw_df": feature_df,
        "X": None, "y": None, "mode": mode,
        "feature_columns": list(gui_qt.DataPreprocessor.BASE_FEATURES),
        "models": models, "walk_forward": wf,
    }


def test_window_has_all_tabs(window):
    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert any("Feature Wiki" in t for t in tabs)
    assert any("Walk-Forward" in t for t in tabs)
    assert len(tabs) == 6


def test_all_models_have_cards_and_combo_entries(window):
    for name in ["Random Forest", "XGBoost", "LSTM", "NEAT", "Jordan", "Ensemble"]:
        assert name in window.cards
    combo = [window.model_combo.itemText(i)
             for i in range(window.model_combo.count())]
    assert "NEAT" in combo and "Jordan" in combo


def test_classification_results_render(window):
    window._on_finished(fake_results("Classification"))
    assert "UP" in window.cards["NEAT"].value_lbl.text()


def test_regression_results_render(window):
    window._on_finished(fake_results("Regression"))
    assert "%" in window.cards["Jordan"].value_lbl.text()


def test_theme_toggle_switches_and_rerenders(window):
    window._on_finished(fake_results())
    window.theme_btn.setChecked(True)
    assert window.dark_mode is True
    assert window.colors == gui_qt.DARK_COLORS
    window.theme_btn.setChecked(False)
    assert window.dark_mode is False
    assert window.colors == gui_qt.LIGHT_COLORS


def test_export_charts_writes_pngs_and_csv(window, tmp_path):
    window._on_finished(fake_results())
    saved = window._export_charts(str(tmp_path))
    names = [os.path.basename(p) for p in saved]
    for expected in ["TEST_prediction.png", "TEST_price_chart.png",
                     "TEST_walk_forward.png", "TEST_feature_importance.png",
                     "TEST_walk_forward_scores.csv"]:
        assert expected in names
    for p in saved:
        assert os.path.getsize(p) > 0


def test_export_without_results_writes_nothing(window, tmp_path):
    window.last_results = None
    assert window._export_charts(str(tmp_path)) == []


def test_wiki_documents_every_feature(window):
    html = window.wiki_browser.toHtml()
    all_features = (list(gui_qt.DataPreprocessor.BASE_FEATURES)
                    + list(gui_qt.DataPreprocessor.MARKET_FEATURES))
    missing = [f for f in all_features if f not in html]
    assert not missing, f"wiki is missing features: {missing}"
    for model in ["Random Forest", "XGBoost", "LSTM", "NEAT", "Jordan", "Ensemble"]:
        assert model in html
