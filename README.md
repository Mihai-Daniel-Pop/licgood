# Stock Predictor — Intelligent Systems based on Stock Market Analysis

## Models
Random Forest, XGBoost, LSTM, NEAT, Jordan, Ensemble
Each model can do 2 things in terms of work: 
"classification" (will tomorrow closer higher or lower) and "regression" (tomorrow's log return)

## How  to  setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## How to run
```bash
python gui_qt.py                # desktop GUI (light/dark theme toggle inside)
python main.py --ticker NVDA --walk-forward   # CLI training run
python run_experiments.py      # batch experiments -> results/ (Chapter 7 data)
python -m pytest               # test suite (offline, ~15 s)
```

## GUI tour

- **Prediction** — per-model display of result.
- **Price Chart** — close price with 20 an 50-day moving averages.
- **Walk-Forward** — per-fold accuracy with expanding-window cross validation.
- **Feature Importance** — which factors influence the model.
- **Feature Wiki** — plain-language explanation of all 22 base +
  8 market-context features and all models.
- **Log** — live progress and errors.

## Design guarantees (backed by tests)

- **No look-ahead leakage** — scalers fit on the training slice only.
- **Reproducibility** — seeded RNGs; cached data makes re-runs offline.
- **Modularity** — a new model only needs the three-method interface.
