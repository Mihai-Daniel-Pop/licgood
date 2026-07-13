"""One test file for all five algorithms x both tasks: the uniform
train / predict / walk_forward_score contract, persistence round-trips,
and model-specific guarantees."""

import os

import numpy as np
import pytest

from src.models import CLASSIFIERS, REGRESSORS, LSTMModel, NEATModel

# Fast settings per algorithm so the whole suite stays quick
FAST = {"LSTM": {"epochs": 2}, "Jordan": {"epochs": 2},
        "NEAT": {"pop_size": 15, "generations": 2}}
FAST_WF = {"LSTM": {"epochs": 1}, "Jordan": {"epochs": 1},
           "NEAT": {"pop_size": 15, "generations": 2}}


@pytest.fixture
def feats(feature_data):
    proc, _, _, _ = feature_data
    return proc.feature_df[proc.feature_columns]


@pytest.mark.parametrize("name", list(CLASSIFIERS))
def test_classifier_train_predict_persist(name, feature_data, feats, tmp_path):
    _, _, X, y = feature_data
    model = redirect(CLASSIFIERS[name](), tmp_path)
    metrics = model.train(X, y, verbose=False, **FAST.get(name, {}))
    assert 0.0 <= metrics["accuracy"] <= 1.0

    pred, proba = model.predict(feats)
    assert pred in (0, 1)
    assert len(proba) == 2 and np.isclose(sum(proba), 1.0)
    assert 0.0 <= proba[1] <= 1.0

    # a brand-new instance must reload from disk and agree
    fresh = redirect(CLASSIFIERS[name](), tmp_path)
    pred2, proba2 = fresh.predict(feats)
    assert pred2 == pred and np.allclose(proba2, proba, atol=1e-6)


@pytest.mark.parametrize("name", list(REGRESSORS))
def test_regressor_train_predict_persist(name, feature_data, feats,
                                         regression_data, tmp_path):
    X, y = regression_data
    model = redirect(REGRESSORS[name](), tmp_path)
    metrics = model.train(X, y, verbose=False, **FAST.get(name, {}))
    assert metrics["mae"] >= 0.0

    pred = model.predict(feats)
    assert isinstance(pred, float) and np.isfinite(pred)

    fresh = redirect(REGRESSORS[name](), tmp_path)
    assert np.isclose(fresh.predict(feats), pred, atol=1e-6)


@pytest.mark.parametrize("name", list(CLASSIFIERS))
def test_classifier_walk_forward(name, feature_data, tmp_path):
    _, _, X, y = feature_data
    model = redirect(CLASSIFIERS[name](), tmp_path)
    scores = model.walk_forward_score(X, y, n_splits=3, verbose=False,
                                      **FAST_WF.get(name, {}))
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.parametrize("name", list(REGRESSORS))
def test_regressor_walk_forward(name, regression_data, tmp_path):
    X, y = regression_data
    model = redirect(REGRESSORS[name](), tmp_path)
    scores = model.walk_forward_score(X, y, n_splits=3, verbose=False,
                                      **FAST_WF.get(name, {}))
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_tree_feature_importance(feature_data, tmp_path):
    _, _, X, y = feature_data
    for name in ("Random Forest", "XGBoost"):
        model = redirect(CLASSIFIERS[name](), tmp_path)
        model.train(X, y, verbose=False)
        imp = model.feature_importance()
        assert imp is not None and len(imp) == X.shape[1]
    lstm = redirect(CLASSIFIERS["LSTM"](), tmp_path)
    assert lstm.feature_importance() is None


def test_predict_returns_none_without_saved_model(feats, tmp_path):
    for name, factory in CLASSIFIERS.items():
        assert redirect(factory(), tmp_path, name).predict(feats) is None


def test_default_paths_are_distinct():
    paths = [factory().path
             for registry in (CLASSIFIERS, REGRESSORS)
             for factory in registry.values()]
    assert len(set(paths)) == len(paths), "two models share a save file"


def test_walk_forward_does_not_touch_saved_model(feature_data, tmp_path):
    _, _, X, y = feature_data
    model = redirect(LSTMModel("classifier"), tmp_path)
    model.train(X, y, epochs=1, verbose=False)
    mtime = os.path.getmtime(model.path)
    model.walk_forward_score(X, y, n_splits=3, epochs=1, verbose=False)
    assert os.path.getmtime(model.path) == mtime


def test_neat_training_is_reproducible(feature_data, feats, tmp_path):
    """NFR2: same inputs + same seed => same prediction."""
    _, _, X, y = feature_data
    a = redirect(NEATModel("classifier"), tmp_path, "a")
    a.train(X, y, pop_size=15, generations=2, verbose=False)
    b = redirect(NEATModel("classifier"), tmp_path, "b")
    b.train(X, y, pop_size=15, generations=2, verbose=False)
    assert np.allclose(a.predict(feats)[1], b.predict(feats)[1])


def redirect(model, folder, name="model"):
    from tests.conftest import redirect_paths
    return redirect_paths(model, folder, name)
