
import os
import string
import sys
import traceback
from datetime import date

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import QDate, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDateEdit, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QStatusBar, QTabWidget,
    QTextBrowser, QTextEdit, QVBoxLayout, QWidget,
)

from src.data_loader import DataLoader, MarketContextLoader
from src.models import CLASSIFIERS, REGRESSORS, Ensemble, SkModel
from src.preprocessor import DataPreprocessor

# ---------- Theme ----------

LIGHT_COLORS = {
    "window": "#f4f5f7", "surface": "#ffffff", "text": "#1f2430",
    "muted": "#5a6472", "border": "#dfe3ea", "field_border": "#d6dae1",
    "selection": "#cfe0ff", "accent": "#2f6fed", "btn_bg": "#ffffff",
    "hover": "#eef1f6", "pressed": "#e2e7ef", "disabled_bg": "#f0f1f4",
    "disabled_text": "#a8b0bd", "check_border": "#c2c8d2",
    "tab_bg": "#e9ecf2", "tab_hover": "#f0f2f6", "status_bg": "#eceef2",
    "progress_bg": "#eef1f6", "card_border": "#e2e6ed",
    "up": "#1f9d57", "down": "#d83a4a",
    "grid": "#e5e8ee", "axis_edge": "#c9ced6", "legend_edge": "#d6dae1",
}

DARK_COLORS = {
    "window": "#15181e", "surface": "#1e222b", "text": "#e6e9ef",
    "muted": "#9aa3b2", "border": "#2c313c", "field_border": "#343b48",
    "selection": "#2b4a8f", "accent": "#5b8cff", "btn_bg": "#262b36",
    "hover": "#2e3441", "pressed": "#232833", "disabled_bg": "#1c2029",
    "disabled_text": "#5a6472", "check_border": "#3a4150",
    "tab_bg": "#1a1e26", "tab_hover": "#20242e", "status_bg": "#1a1e26",
    "progress_bg": "#1a1e26", "card_border": "#2c313c",
    "up": "#2fbf71", "down": "#ff5d6c",
    "grid": "#2a2f3a", "axis_edge": "#3a4150", "legend_edge": "#3a4150",
}

# Per-model chart colors, readable on both themes
MODEL_COLORS = {
    "Random Forest": "#1f9d57",
    "XGBoost": "#2f6fed",
    "LSTM": "#7c5cdc",
    "NEAT": "#d9820a",
    "Jordan": "#0ea5b7",
}

QSS = string.Template("""
* { font-family: 'Segoe UI', 'Helvetica Neue', sans-serif; font-size: 11pt; }

QMainWindow, QWidget { background-color: $window; color: $text; }

QGroupBox {
    border: 1px solid $border;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 8px;
    background-color: $surface;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: $accent; }

QLineEdit, QDateEdit, QComboBox, QTextEdit {
    background-color: $surface;
    border: 1px solid $field_border;
    border-radius: 6px;
    padding: 6px 10px;
    color: $text;
    selection-background-color: $selection;
    selection-color: $text;
}
QLineEdit:focus, QDateEdit:focus, QComboBox:focus, QTextEdit:focus {
    border: 1px solid $accent;
}

QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: $surface;
    border: 1px solid $field_border;
    selection-background-color: $selection;
    selection-color: $text;
    color: $text;
}

QPushButton {
    background-color: $btn_bg;
    border: 1px solid $field_border;
    border-radius: 6px;
    padding: 8px 16px;
    color: $text;
    font-weight: 600;
}
QPushButton:hover { background-color: $hover; }
QPushButton:pressed { background-color: $pressed; }
QPushButton:disabled { background-color: $disabled_bg; color: $disabled_text; }

QPushButton#primary { background-color: #2f6fed; color: #ffffff; border: 1px solid #2f6fed; }
QPushButton#primary:hover { background-color: #1f5fe0; }
QPushButton#primary:pressed { background-color: #1a52c8; }

QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px; border: 1px solid $check_border; background: $surface; }
QCheckBox::indicator:checked { background-color: $accent; border: 1px solid $accent; }

QTabWidget::pane { border: 1px solid $border; border-radius: 8px; top: -1px; background: $surface; }
QTabBar::tab {
    background: $tab_bg;
    color: $muted;
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected { background: $surface; color: $accent; border-bottom: 2px solid $accent; }
QTabBar::tab:hover:!selected { background: $tab_hover; }

QProgressBar {
    border: 1px solid $field_border;
    border-radius: 6px;
    background-color: $progress_bg;
    text-align: center;
    color: $text;
}
QProgressBar::chunk { background-color: #2f6fed; border-radius: 5px; }

QStatusBar { background-color: $status_bg; color: $muted; border-top: 1px solid $border; }

QLabel#cardTitle { color: $muted; font-size: 10pt; font-weight: 600; letter-spacing: 1px; }
QLabel#cardValueUp { color: $up; font-size: 22pt; font-weight: 700; }
QLabel#cardValueDown { color: $down; font-size: 22pt; font-weight: 700; }
QLabel#cardValueNeutral { color: $text; font-size: 22pt; font-weight: 700; }
QLabel#cardSub { color: $muted; font-size: 10pt; }

QLabel#bigHeading { color: $text; font-size: 18pt; font-weight: 700; }
QLabel#subHeading { color: $accent; font-size: 11pt; font-weight: 600; }

QFrame#card {
    background-color: $surface;
    border: 1px solid $card_border;
    border-radius: 10px;
    padding: 10px;
}

QTextBrowser {
    background-color: $surface;
    border: 1px solid $field_border;
    border-radius: 6px;
    color: $text;
}
""")


def apply_mpl_theme(colors):
    plt.rcParams.update({
        "figure.facecolor": colors["surface"],
        "axes.facecolor": colors["surface"],
        "axes.edgecolor": colors["axis_edge"],
        "axes.labelcolor": colors["text"],
        "axes.titlecolor": colors["text"],
        "xtick.color": colors["muted"],
        "ytick.color": colors["muted"],
        "grid.color": colors["grid"],
        "grid.alpha": 0.9,
        "text.color": colors["text"],
        "legend.facecolor": colors["surface"],
        "legend.edgecolor": colors["legend_edge"],
        "legend.labelcolor": colors["text"],
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def apply_app_theme(app, dark):
    colors = DARK_COLORS if dark else LIGHT_COLORS
    palette = QPalette()
    for role, key in [("Window", "window"), ("WindowText", "text"),
                      ("Base", "surface"), ("AlternateBase", "window"),
                      ("Text", "text"), ("Button", "surface"),
                      ("ButtonText", "text"), ("ToolTipBase", "surface"),
                      ("ToolTipText", "text"), ("Highlight", "accent")]:
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(colors[key]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)
    app.setStyleSheet(QSS.substitute(colors))
    apply_mpl_theme(colors)


# ---------- Feature wiki ----------

FEATURE_WIKI = [
    ("Returns & momentum",
     "How much the price has moved recently. Positive values mean the stock "
     "went up over that window, negative values mean it went down.",
     [
         ("Return_1d",
          "Percentage change of the closing price versus the previous trading "
          "day. +0.02 means the stock closed 2% higher than yesterday."),
         ("Return_5d",
          "Percentage change of the close versus 5 trading days ago (about one "
          "week). Captures short-term direction."),
         ("Return_10d",
          "Percentage change of the close versus 10 trading days ago (about "
          "two weeks). Captures medium-term direction."),
         ("LogReturn_1d",
          "The natural logarithm of today's close divided by yesterday's. "
          "Almost identical to Return_1d for small moves, but mathematically "
          "nicer: log returns can be added across days. This is also the "
          "quantity the regression models predict."),
         ("Momentum_10d",
          "Today's close divided by the close 10 days ago, minus 1. Another "
          "view of two-week momentum: positive = uptrend, negative = downtrend."),
     ]),
    ("Volatility — how nervous the stock is",
     "Volatility measures how large the daily swings have been, regardless of "
     "direction. High volatility means the price is jumping around a lot.",
     [
         ("Volatility_5d",
          "Standard deviation of the daily returns over the last 5 days. "
          "A very twitchy week produces a high value."),
         ("Volatility_10d",
          "Same, over the last 10 days."),
         ("Volatility_20d",
          "Same, over the last 20 days (about one month)."),
         ("ATR_14_Norm",
          "Average True Range over 14 days, divided by the closing price. The "
          "true range is the full span a stock traded in a day (including gaps "
          "from the previous close). 0.03 means the stock moves about 3% of "
          "its price per day on average."),
         ("BB_Width",
          "Width of the Bollinger Bands relative to the 20-day average price. "
          "The bands sit 2 standard deviations above and below the 20-day "
          "average, so a wide band = volatile month, a narrow band = calm "
          "month (often before a breakout)."),
     ]),
    ("Oscillators — overbought / oversold gauges",
     "Oscillators are bounded indicators that flag when a stock looks "
     "stretched compared to its own recent behaviour.",
     [
         ("RSI_14",
          "Relative Strength Index over 14 days, from 0 to 100. It compares "
          "the size of recent gains to recent losses. Above ~70 the stock is "
          "traditionally considered overbought (due for a pause), below ~30 "
          "oversold (due for a bounce), around 50 is neutral."),
         ("Stoch_K",
          "Stochastic oscillator %K, from 0 to 100: where today's close sits "
          "inside the highest-high / lowest-low range of the last 14 days. "
          "100 = closing at the very top of the recent range, 0 = at the "
          "bottom."),
         ("Stoch_D",
          "A 3-day moving average of Stoch_K. Smoother and slower; crossovers "
          "of %K over %D are a classic trading signal."),
         ("BB_Position",
          "Where today's close sits inside the Bollinger Bands. 0 = touching "
          "the lower band (unusually cheap versus the last 20 days), 0.5 = "
          "exactly in the middle, 1 = touching the upper band (unusually "
          "expensive). Values below 0 or above 1 mean the price has broken "
          "out of the bands — a strong move."),
     ]),
    ("Trend",
     "Trend indicators compare fast and slow views of the price to tell "
     "whether the stock is in an uptrend or a downtrend.",
     [
         ("MACD",
          "Moving Average Convergence Divergence: the 12-day exponential "
          "moving average minus the 26-day one. Positive = short-term average "
          "above long-term average = uptrend; negative = downtrend."),
         ("MACD_Signal",
          "A 9-day exponential moving average of the MACD line itself. Acts "
          "as the slower 'confirmation' line."),
         ("MACD_Hist",
          "MACD minus MACD_Signal. Positive and growing = the uptrend is "
          "accelerating; shrinking towards zero = the trend is losing steam. "
          "The sign flip is the classic MACD crossover signal."),
         ("Close_SMA10_Ratio",
          "Today's close divided by its own 10-day simple moving average. "
          "Above 1 = trading above the short-term average (bullish), below "
          "1 = under it (bearish)."),
         ("Close_SMA50_Ratio",
          "Same, against the 50-day moving average — the medium-term trend."),
         ("SMA10_SMA50_Ratio",
          "The 10-day average divided by the 50-day average. Above 1 means "
          "the short-term trend is above the long-term one (a 'golden cross' "
          "state); below 1 the opposite ('death cross')."),
     ]),
    ("Volume",
     "Volume tells you how much conviction is behind a move: a rally on huge "
     "volume is more meaningful than one on thin trading.",
     [
         ("OBV_Norm",
          "On-Balance Volume, standardized. OBV adds the day's volume when "
          "the price closes up and subtracts it when it closes down, so it "
          "accumulates buying/selling pressure. The value shown is a z-score "
          "versus its own last 20 days: +2 = unusually strong buying "
          "pressure, -2 = unusually strong selling pressure."),
         ("Volume_Ratio",
          "Today's traded volume divided by the 20-day average volume. 1 = "
          "a normal day, 3 = three times the usual activity (something is "
          "happening), 0.5 = a very quiet day."),
     ]),
    ("Market context (only when 'Include market context' is enabled)",
     "Individual stocks rarely move independently of the overall market. "
     "These features describe what the S&P 500 (via the SPY ETF) and the "
     "VIX 'fear index' are doing.",
     [
         ("SPY_Return_1d",
          "Yesterday-to-today return of SPY, the S&P 500 ETF — a proxy for "
          "the whole U.S. market."),
         ("SPY_Return_5d",
          "One-week return of SPY."),
         ("VIX_Level",
          "The CBOE Volatility Index: the market's expectation of volatility "
          "over the next 30 days, implied by option prices. Roughly: below "
          "15 = calm, 15–25 = normal, above 30 = fear/panic."),
         ("VIX_Change_1d",
          "Daily percentage change of the VIX. A sharp rise means fear is "
          "spiking right now."),
         ("VIX_MA20_Ratio",
          "VIX divided by its own 20-day average. Above 1 = fear is elevated "
          "versus the past month, below 1 = calmer than usual."),
         ("Excess_Return_1d",
          "The stock's daily return minus SPY's daily return: how much the "
          "stock beat (or lagged) the market today."),
         ("RelStrength_20d",
          "The stock's cumulative 20-day return divided by SPY's. Above 1 = "
          "the stock has outperformed the market over the last month."),
         ("Beta_60d",
          "Rolling 60-day beta versus SPY: how strongly the stock moves with "
          "the market. 1 = moves one-for-one with the market, 2 = twice as "
          "hard in both directions, 0 = independent, negative = moves against "
          "the market."),
     ]),
    ("The models",
     "Every algorithm receives exactly the same features above; they differ "
     "in how they turn them into a prediction.",
     [
         ("Random Forest",
          "Hundreds of decision trees, each trained on a random slice of the "
          "data and features, that vote on the outcome. Robust and hard to "
          "overfit; also provides the feature importance rankings."),
         ("XGBoost",
          "Gradient-boosted trees: trees are built one after another, each "
          "one correcting the errors of the previous ones. Often the "
          "strongest tabular-data model."),
         ("LSTM",
          "A recurrent neural network with Long Short-Term Memory cells. "
          "Unlike the trees, it reads a sequence of the last 30 days and "
          "keeps a learned internal memory, so it can pick up temporal "
          "patterns."),
         ("NEAT",
          "NeuroEvolution of Augmenting Topologies: instead of training a "
          "fixed network with gradient descent, a population of small neural "
          "networks is evolved — mutation and crossover change both the "
          "weights and the structure — and the fittest network survives. "
          "Note: its confidence values tend to be extreme (near 0% or 100%) "
          "because evolved networks are not calibrated the way "
          "gradient-trained ones are."),
         ("Jordan",
          "A classic Jordan recurrent network (1986): it also reads the last "
          "30 days, but its memory is its own previous output, fed back into "
          "the hidden layer through a decaying context unit. A simpler, "
          "more transparent form of recurrence than the LSTM."),
         ("Ensemble",
          "Combines Random Forest, XGBoost and LSTM. In classification mode "
          "it averages their probabilities (soft voting); in regression mode "
          "it averages their predicted returns. Averaging models with "
          "different blind spots usually cancels some noise."),
     ]),
]


def wiki_html(colors):
    parts = [
        f"""
        <h1 style="color:{colors['text']};">Feature wiki</h1>
        <p style="color:{colors['muted']};">
        The models never see the raw price. Every day is described by the
        <i>features</i> (also called indicators or influences) listed below,
        and the models learn patterns in these numbers. This page explains
        what each one means and how to read its value. The same names appear
        in the <b>Feature Importance</b> tab.
        </p>
        """
    ]
    for section, blurb, entries in FEATURE_WIKI:
        parts.append(
            f"<h2 style='color:{colors['accent']};'>{section}</h2>"
            f"<p style='color:{colors['muted']};'>{blurb}</p>"
            "<table cellspacing='0' cellpadding='6' width='100%'>"
        )
        for name, desc in entries:
            parts.append(
                f"<tr><td width='170' style='color:{colors['accent']};'>"
                f"<b><code>{name}</code></b></td>"
                f"<td style='color:{colors['text']};'>{desc}</td></tr>"
            )
        parts.append("</table>")
    return "".join(parts)


# ---------- Worker ----------

class AnalysisWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, ticker, start_date, end_date, model_choice,
                 force_retrain, tune, run_walk_forward,
                 mode="Classification", include_market=False):
        super().__init__()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model_choice = model_choice
        self.force_retrain = force_retrain
        self.tune = tune
        self.run_walk_forward = run_walk_forward
        self.mode = mode
        self.include_market = include_market

    def run(self):
        try:
            self.progress.emit(f"Fetching data for {self.ticker}...")
            raw_df = DataLoader(self.ticker, self.start_date, self.end_date).get_data()
            if raw_df is None or raw_df.empty:
                self.failed.emit(f"No data found for {self.ticker}")
                return

            processor = DataPreprocessor(raw_df)
            if self.include_market:
                self.progress.emit("Fetching market context (SPY + VIX)...")
                market = MarketContextLoader(self.start_date, self.end_date).get_market_data()
                if market is not None:
                    processor.set_market_context(market)
                else:
                    self.progress.emit("Market data unavailable — continuing without context.")

            self.progress.emit("Engineering features...")
            feature_df = processor.add_technical_indicators()
            regression = self.mode == "Regression"
            if regression:
                X, y = processor.prepare_data_for_regression()
            else:
                X, y = processor.prepare_data_for_training()
            if len(X) < 100:
                self.failed.emit(f"Not enough data after feature engineering "
                                 f"(got {len(X)} rows). Use a longer date range.")
                return

            registry = REGRESSORS if regression else CLASSIFIERS
            feats = processor.feature_df[processor.feature_columns]
            results = {"ticker": self.ticker, "feature_df": feature_df,
                       "mode": self.mode, "feature_columns": processor.feature_columns,
                       "models": {}, "walk_forward": {}}

            if self.model_choice == "All":
                wanted = list(registry)
            elif self.model_choice == "Ensemble":
                wanted = list(Ensemble.BASES)
            else:
                wanted = [self.model_choice]

            for name in wanted:
                model = registry[name]()
                if self.force_retrain or not os.path.exists(model.path):
                    self.progress.emit(f"Training {name}...")
                    if isinstance(model, SkModel):
                        model.train(X, y, tune=self.tune, verbose=False)
                    else:
                        model.train(X, y, verbose=False)
                out = model.predict(feats)
                if out is None:
                    continue
                entry = {"importance": model.feature_importance()}
                if regression:
                    entry["pred_return"] = out
                else:
                    entry["pred"] = out[0]
                    entry["proba"] = np.asarray(out[1], dtype=float)
                results["models"][name] = entry

            if self.model_choice in ("All", "Ensemble") and results["models"]:
                self.progress.emit("Running ensemble...")
                task = "regressor" if regression else "classifier"
                ens_out = Ensemble(task).predict(feats)
                if ens_out is not None:
                    if regression:
                        results["models"]["Ensemble"] = {
                            "pred_return": ens_out, "importance": None}
                    else:
                        results["models"]["Ensemble"] = {
                            "pred": ens_out[0], "proba": ens_out[1],
                            "importance": None}

            if self.run_walk_forward:
                for name, factory in registry.items():
                    self.progress.emit(f"Walk-forward CV ({name})...")
                    results["walk_forward"][name] = factory().walk_forward_score(
                        X, y, n_splits=5, verbose=False)

            self.finished.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ---------- UI components ----------

class PredictionCard(QFrame):
    """A single colored card showing one model's prediction."""

    def __init__(self, name):
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("card")  # styled by the active theme QSS
        self.setMinimumWidth(160)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.setContentsMargins(14, 14, 14, 14)
        self.title_lbl = QLabel(name.upper())
        self.title_lbl.setObjectName("cardTitle")
        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("cardValueNeutral")
        self.conf_lbl = QLabel("confidence —")
        self.conf_lbl.setObjectName("cardSub")
        layout.addWidget(self.title_lbl)
        layout.addWidget(self.value_lbl)
        layout.addWidget(self.conf_lbl)
        layout.addStretch()

    def _set_value(self, text, style):
        self.value_lbl.setText(text)
        self.value_lbl.setObjectName(style)
        self.value_lbl.style().unpolish(self.value_lbl)
        self.value_lbl.style().polish(self.value_lbl)

    def update_result(self, pred, proba):
        self._set_value("UP ▲" if pred == 1 else "DOWN ▼",
                        "cardValueUp" if pred == 1 else "cardValueDown")
        self.conf_lbl.setText(f"confidence {float(proba[pred]) * 100:.2f}%")

    def update_return(self, log_return):
        pct = (np.exp(log_return) - 1) * 100
        self._set_value(f"{pct:+.2f}% {'▲' if pct >= 0 else '▼'}",
                        "cardValueUp" if pct >= 0 else "cardValueDown")
        self.conf_lbl.setText(f"log-return {log_return:+.5f}")

    def reset(self):
        self._set_value("—", "cardValueNeutral")
        self.conf_lbl.setText("—")


class MplWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure()  # facecolor comes from the active mpl theme
        self.canvas = FigureCanvas(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def apply_theme(self, colors):
        self.figure.set_facecolor(colors["surface"])
        self.canvas.draw_idle()


# ---------- Main window ----------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Prediction System")
        self.resize(1400, 900)
        self.worker = None
        self.thread = None
        self.last_results = None
        self.dark_mode = False
        self.colors = LIGHT_COLORS
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # ---- Sidebar ----
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sb_layout = QVBoxLayout(sidebar)
        sb_layout.setSpacing(12)

        title = QLabel("📈  Stock Predictor")
        title.setObjectName("bigHeading")
        sub = QLabel("Multi-model thesis prototype")
        sub.setObjectName("subHeading")
        sb_layout.addWidget(title)
        sb_layout.addWidget(sub)

        self.theme_btn = QPushButton("🌙  Dark mode")
        self.theme_btn.setCheckable(True)
        self.theme_btn.setToolTip("Switch between the light and dark theme.")
        self.theme_btn.toggled.connect(self._on_theme_toggled)
        sb_layout.addWidget(self.theme_btn)

        cfg_box = QGroupBox("Configuration")
        cfg_layout = QGridLayout(cfg_box)
        cfg_layout.setVerticalSpacing(10)

        cfg_layout.addWidget(QLabel("Ticker:"), 0, 0)
        self.ticker_input = QLineEdit("NVDA")
        cfg_layout.addWidget(self.ticker_input, 0, 1)

        cfg_layout.addWidget(QLabel("Start:"), 1, 0)
        self.start_date = QDateEdit(QDate(2023, 1, 1))
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        cfg_layout.addWidget(self.start_date, 1, 1)

        cfg_layout.addWidget(QLabel("End:"), 2, 0)
        today = date.today()
        self.end_date = QDateEdit(QDate(today.year, today.month, today.day))
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        cfg_layout.addWidget(self.end_date, 2, 1)

        cfg_layout.addWidget(QLabel("Model:"), 3, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["All", "Ensemble"] + list(MODEL_COLORS))
        cfg_layout.addWidget(self.model_combo, 3, 1)

        cfg_layout.addWidget(QLabel("Mode:"), 4, 0)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Classification", "Regression"])
        self.mode_combo.setToolTip(
            "Classification: predict up/down direction.\n"
            "Regression: predict next-day log-return magnitude."
        )
        cfg_layout.addWidget(self.mode_combo, 4, 1)

        sb_layout.addWidget(cfg_box)

        opts_box = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_box)
        self.market_cb = QCheckBox("Include market context (SPY + VIX)")
        self.market_cb.setChecked(True)
        self.market_cb.setToolTip("Adds SPY return, VIX level/change, beta, "
                                  "relative strength, excess return.")
        self.force_retrain_cb = QCheckBox("Force retrain")
        self.tune_cb = QCheckBox("Hyperparameter tuning (Optuna)")
        self.tune_cb.setToolTip("Slower. Tunes RF and XGBoost via walk-forward CV.")
        self.walkforward_cb = QCheckBox("Run walk-forward CV after analysis")
        for cb in (self.market_cb, self.force_retrain_cb, self.tune_cb,
                   self.walkforward_cb):
            opts_layout.addWidget(cb)
        sb_layout.addWidget(opts_box)

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._on_run)
        sb_layout.addWidget(self.run_btn)

        self.export_btn = QPushButton("💾  Export charts")
        self.export_btn.setToolTip("Save every chart as a PNG (plus the "
                                   "walk-forward scores as CSV) — e.g. for "
                                   "thesis figures.")
        self.export_btn.clicked.connect(self._on_export)
        sb_layout.addWidget(self.export_btn)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        sb_layout.addWidget(self.progress)

        sb_layout.addStretch()

        self.tips_lbl = QLabel()
        self.tips_lbl.setWordWrap(True)
        sb_layout.addWidget(self.tips_lbl)
        self._update_tips()

        root.addWidget(sidebar)

        # ---- Tabs ----
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_prediction_tab(), "🎯 Prediction")
        self.tabs.addTab(self._build_chart_tab(), "📊 Price Chart")
        self.tabs.addTab(self._build_walkforward_tab(), "🔄 Walk-Forward")
        self.tabs.addTab(self._build_importance_tab(), "🧠 Feature Importance")
        self.tabs.addTab(self._build_wiki_tab(), "📖 Feature Wiki")
        self.tabs.addTab(self._build_log_tab(), "📜 Log")
        root.addWidget(self.tabs, stretch=1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready.")

    def _build_prediction_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(14)
        header = QLabel("Latest Prediction (next trading day)")
        header.setObjectName("bigHeading")
        layout.addWidget(header)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self.cards = {}
        for name in list(MODEL_COLORS) + ["Ensemble"]:
            self.cards[name] = PredictionCard(name)
            cards_row.addWidget(self.cards[name])
        layout.addLayout(cards_row)

        self.proba_widget = MplWidget()
        self.proba_box = QGroupBox("Per-model probabilities")
        proba_layout = QVBoxLayout(self.proba_box)
        proba_layout.addWidget(self.proba_widget)
        layout.addWidget(self.proba_box, stretch=1)
        return w

    def _build_chart_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.price_widget = MplWidget()
        layout.addWidget(self.price_widget)
        return w

    def _build_walkforward_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        info = QLabel(
            "Walk-forward CV trains on an expanding window and tests on the next chunk, "
            "for all five algorithms (Random Forest, XGBoost, LSTM, NEAT, Jordan). "
            "Enable in the sidebar and click Run Analysis."
        )
        info.setObjectName("subHeading")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.wf_widget = MplWidget()
        layout.addWidget(self.wf_widget, stretch=1)
        return w

    def _build_importance_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.fi_widget = MplWidget()
        layout.addWidget(self.fi_widget)
        return w

    def _build_wiki_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.wiki_browser = QTextBrowser()
        layout.addWidget(self.wiki_browser)
        self.wiki_browser.setHtml(wiki_html(self.colors))
        return w

    def _build_log_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 10pt; }"
        )
        layout.addWidget(self.log_text)
        return w

    def _update_tips(self):
        self.tips_lbl.setText(
            f"<small style='color:{self.colors['muted']}'>"
            "<b>Tips</b><br>"
            "• Use 1y+ of data for stable training.<br>"
            "• 'Ensemble' soft-votes RF, XGBoost and LSTM.<br>"
            "• 'All' also trains NEAT and Jordan.<br>"
            "• Walk-forward CV compares all five algorithms.<br>"
            "• The Feature Wiki tab explains every indicator."
            "</small>"
        )

    # ---------- Run analysis ----------

    def _log(self, msg):
        self.log_text.append(msg)
        self.status.showMessage(msg)

    def _on_run(self):
        ticker = self.ticker_input.text().strip().upper()
        if not ticker:
            QMessageBox.warning(self, "Missing ticker", "Please enter a stock ticker.")
            return

        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.log_text.clear()
        for card in self.cards.values():
            card.reset()

        self.worker = AnalysisWorker(
            ticker=ticker,
            start_date=self.start_date.date().toString("yyyy-MM-dd"),
            end_date=self.end_date.date().toString("yyyy-MM-dd"),
            model_choice=self.model_combo.currentText(),
            force_retrain=self.force_retrain_cb.isChecked(),
            tune=self.tune_cb.isChecked(),
            run_walk_forward=self.walkforward_cb.isChecked(),
            mode=self.mode_combo.currentText(),
            include_market=self.market_cb.isChecked(),
        )
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._log)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_finished(self, results):
        self.last_results = results
        mode = results.get("mode", "Classification")
        self._log(f"Analysis complete. mode={mode}  "
                  f"features={len(results.get('feature_columns', []))}")
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)

        for name, card in self.cards.items():
            if name in results["models"]:
                m = results["models"][name]
                if mode == "Regression":
                    card.update_return(m["pred_return"])
                else:
                    card.update_result(m["pred"], m["proba"])
        self._render_results(results)

    def _render_results(self, results):
        """Draw every chart from a results dict (also used on theme change)."""
        mode = results.get("mode", "Classification")
        if mode == "Regression":
            self.proba_box.setTitle("Per-model predicted returns")
            self._plot_returns(results)
        else:
            self.proba_box.setTitle("Per-model probabilities")
            self._plot_probabilities(results)
        self._plot_price_chart(results["ticker"], results["feature_df"])
        if results["walk_forward"]:
            self._plot_walk_forward(results["walk_forward"], mode)
        self._plot_feature_importance(results["models"], results.get("feature_columns"))

    def _on_failed(self, err_msg):
        self.progress.setVisible(False)
        self.run_btn.setEnabled(True)
        self._log(f"ERROR:\n{err_msg}")
        QMessageBox.critical(self, "Analysis failed", err_msg.splitlines()[-1])

    # ---------- Plots ----------

    def _plot_probabilities(self, results):
        fig = self.proba_widget.figure
        fig.clear()
        names = list(results["models"].keys())
        if names:
            ax = fig.add_subplot(111)
            up_probs = [results["models"][n]["proba"][1] * 100 for n in names]
            colors = [self.colors["up"] if p >= 50 else self.colors["down"]
                      for p in up_probs]
            bars = ax.barh(names, up_probs, color=colors,
                           edgecolor=self.colors["surface"])
            ax.axvline(50, color=self.colors["accent"], linestyle="--",
                       linewidth=1, alpha=0.7)
            ax.set_xlim(0, 100)
            ax.set_xlabel("P(UP) %")
            ax.set_title("Probability of upward move — per model")
            for bar, p in zip(bars, up_probs):
                ax.text(min(p + 2, 96), bar.get_y() + bar.get_height() / 2,
                        f"{p:.1f}%", va="center", color=self.colors["text"],
                        fontweight="bold")
            fig.tight_layout()
        self.proba_widget.canvas.draw()

    def _plot_returns(self, results):
        fig = self.proba_widget.figure
        fig.clear()
        names = list(results["models"].keys())
        if names:
            ax = fig.add_subplot(111)
            pcts = [(np.exp(results["models"][n]["pred_return"]) - 1) * 100
                    for n in names]
            colors = [self.colors["up"] if p >= 0 else self.colors["down"]
                      for p in pcts]
            bars = ax.barh(names, pcts, color=colors,
                           edgecolor=self.colors["surface"])
            ax.axvline(0, color=self.colors["accent"], linestyle="--",
                       linewidth=1, alpha=0.7)
            ax.set_xlabel("Predicted next-day return (%)")
            ax.set_title("Predicted next-day return — per model")
            for bar, p in zip(bars, pcts):
                ax.text(p + (0.05 if p >= 0 else -0.05),
                        bar.get_y() + bar.get_height() / 2,
                        f"{p:+.2f}%", va="center",
                        ha="left" if p >= 0 else "right",
                        color=self.colors["text"], fontweight="bold")
            fig.tight_layout()
        self.proba_widget.canvas.draw()

    def _plot_price_chart(self, ticker, df):
        fig = self.price_widget.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.plot(df.index, df["Close"], color=self.colors["accent"],
                linewidth=1.5, label="Close")
        ax.plot(df.index, df["Close"].rolling(20).mean(), color="#d9820a",
                linewidth=1, alpha=0.9, label="SMA 20")
        ax.plot(df.index, df["Close"].rolling(50).mean(), color="#7c5cdc",
                linewidth=1, alpha=0.9, label="SMA 50")
        ax.set_title(f"{ticker}  —  Price & moving averages")
        ax.set_ylabel("Price")
        ax.legend(loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        self.price_widget.canvas.draw()

    def _plot_walk_forward(self, wf, mode="Classification"):
        fig = self.wf_widget.figure
        fig.clear()
        ax = fig.add_subplot(111)
        models = list(wf.keys())
        n_folds = max(len(v) for v in wf.values())
        x = np.arange(n_folds)
        width = 0.8 / max(len(models), 1)

        for i, name in enumerate(models):
            scores = wf[name]
            ax.bar(x + (i - len(models) / 2 + 0.5) * width, scores, width,
                   label=f"{name} (mean {np.mean(scores):.3f})",
                   color=MODEL_COLORS.get(name, "#8a93a3"),
                   edgecolor=self.colors["surface"])

        ax.axhline(0.5, color=self.colors["down"], linestyle="--",
                   linewidth=1, label="Random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(n_folds)])
        ax.set_ylabel("Directional accuracy" if mode == "Regression" else "Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Walk-forward CV — directional accuracy (regression)"
                     if mode == "Regression" else "Walk-forward cross-validation")
        ax.legend(loc="lower right", ncols=2, fontsize=9)
        fig.tight_layout()
        self.wf_widget.canvas.draw()

    def _plot_feature_importance(self, models, feature_names=None):
        fig = self.fi_widget.figure
        fig.clear()
        if not feature_names:
            feature_names = DataPreprocessor.BASE_FEATURES

        with_imp = {n: m["importance"] for n, m in models.items()
                    if m.get("importance") is not None}
        if not with_imp:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No feature importances available.\n"
                    "Train Random Forest or XGBoost first.",
                    ha="center", va="center", color=self.colors["muted"])
            ax.set_axis_off()
        else:
            for idx, (name, imp) in enumerate(with_imp.items(), start=1):
                ax = fig.add_subplot(1, len(with_imp), idx)
                top = np.argsort(imp)[::-1][:12]
                ax.barh([feature_names[i] for i in top][::-1],
                        [imp[i] for i in top][::-1],
                        color=self.colors["accent"],
                        edgecolor=self.colors["surface"])
                ax.set_title(f"{name} — top 12")
            fig.tight_layout()
        self.fi_widget.canvas.draw()

    # ---------- Export ----------

    def _on_export(self):
        if not self.last_results:
            QMessageBox.information(self, "Nothing to export",
                                    "Run an analysis first.")
            return
        default = os.path.join(os.getcwd(), "exports")
        os.makedirs(default, exist_ok=True)
        folder = QFileDialog.getExistingDirectory(self, "Choose export folder",
                                                  default)
        if folder:
            saved = self._export_charts(folder)
            self._log(f"Exported {len(saved)} file(s) to {folder}")

    def _export_charts(self, folder):
        """Save every populated chart as PNG (150 dpi) and the walk-forward
        scores as CSV. Returns the list of written paths."""
        os.makedirs(folder, exist_ok=True)
        results = self.last_results or {}
        ticker = results.get("ticker", "charts")
        saved = []
        widgets = {"prediction": self.proba_widget,
                   "price_chart": self.price_widget,
                   "walk_forward": self.wf_widget,
                   "feature_importance": self.fi_widget}
        for name, widget in widgets.items():
            if not widget.figure.get_axes():
                continue
            path = os.path.join(folder, f"{ticker}_{name}.png")
            widget.figure.savefig(path, dpi=150,
                                  facecolor=widget.figure.get_facecolor())
            saved.append(path)
        if results.get("walk_forward"):
            df = pd.DataFrame(results["walk_forward"])
            df.index = [f"Fold {i+1}" for i in range(len(df))]
            path = os.path.join(folder, f"{ticker}_walk_forward_scores.csv")
            df.to_csv(path)
            saved.append(path)
        return saved

    # ---------- Theme ----------

    def _on_theme_toggled(self, checked):
        self.dark_mode = checked
        self.theme_btn.setText("☀️  Light mode" if checked else "🌙  Dark mode")
        self.colors = DARK_COLORS if checked else LIGHT_COLORS
        apply_app_theme(QApplication.instance(), checked)
        self._update_tips()
        self.wiki_browser.setHtml(wiki_html(self.colors))
        for widget in (self.proba_widget, self.price_widget,
                       self.wf_widget, self.fi_widget):
            widget.apply_theme(self.colors)
        if self.last_results:
            self._render_results(self.last_results)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_app_theme(app, dark=False)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
