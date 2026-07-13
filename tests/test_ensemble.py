"""Ensemble voting math, tested with stub base models (no disk, no training)."""

import numpy as np

from src.models import Ensemble


class Stub:
    def __init__(self, out):
        self.out = out

    def predict(self, _):
        return self.out


def test_soft_voting_weighted_average():
    ens = Ensemble("classifier",
                   weights={"Random Forest": 1.0, "XGBoost": 1.0, "LSTM": 2.0})
    ens.parts = {"Random Forest": Stub((1, [0.4, 0.6])),
                 "XGBoost": Stub((0, [0.7, 0.3])),
                 "LSTM": Stub((1, [0.2, 0.8]))}

    pred, proba = ens.predict(None)
    expected = (np.array([0.4, 0.6]) + np.array([0.7, 0.3])
                + 2.0 * np.array([0.2, 0.8])) / 4.0
    assert np.allclose(proba, expected)
    assert pred == int(np.argmax(expected))


def test_soft_voting_skips_missing_models():
    ens = Ensemble("classifier")
    ens.parts = {"Random Forest": Stub((1, [0.3, 0.7])),
                 "XGBoost": Stub(None), "LSTM": Stub(None)}
    pred, proba = ens.predict(None)
    assert np.allclose(proba, [0.3, 0.7])
    assert pred == 1


def test_returns_none_when_no_models():
    ens = Ensemble("classifier")
    ens.parts = {n: Stub(None) for n in Ensemble.BASES}
    assert ens.predict(None) is None


def test_accuracy_based_weights_are_clamped():
    ens = Ensemble("classifier")
    ens.set_weights({"Random Forest": 0.60, "XGBoost": 0.40, "LSTM": 0.55})
    assert np.isclose(ens.weights["Random Forest"], 0.10)
    assert np.isclose(ens.weights["XGBoost"], 0.05), \
        "below-chance model must be clamped"
    assert np.isclose(ens.weights["LSTM"], 0.05)


def test_regression_weighted_average():
    ens = Ensemble("regressor",
                   weights={"Random Forest": 1.0, "XGBoost": 3.0, "LSTM": 0.0})
    ens.parts = {"Random Forest": Stub(0.01), "XGBoost": Stub(-0.02),
                 "LSTM": Stub(0.10)}  # weight 0: must not influence the result
    expected = (1.0 * 0.01 + 3.0 * -0.02 + 0.0 * 0.10) / 4.0
    assert np.isclose(ens.predict(None), expected)
