
import json
import os
import random

import joblib
import neat
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.preprocessor import DataPreprocessor

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dir_acc(y_true, y_pred):
    """Fraction of predictions with the correct sign."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(((y_true >= 0) == (y_pred >= 0)).mean())


class BaseModel:

    algo = ""

    def __init__(self, task="classifier"):
        self.task = task
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = list(DataPreprocessor.BASE_FEATURES)
        self.path = os.path.join(MODELS_DIR, f"{self.algo}_{task}.pkl")

    def _matrix(self, X):
        self.feature_columns = list(X.columns)
        return np.asarray(X, dtype=float), None

    def _save(self, extra=None):
        bundle = {"model": self.model, "scaler": self.scaler,
                  "features": self.feature_columns}
        bundle.update(extra or {})
        joblib.dump(bundle, self.path)

    def _load(self):
        if self.model is not None:
            return True
        if not os.path.exists(self.path):
            return False
        bundle = joblib.load(self.path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]
        self.feature_columns = bundle["features"]
        self._loaded(bundle)
        return True

    def _loaded(self, bundle):
        pass

    def feature_importance(self):
        return None



class SkModel(BaseModel):
    """Random Forest and XGBoost, classification and regression."""

    def __init__(self, algo="rf", task="classifier"):
        self.algo = algo
        super().__init__(task)

    def _new(self, y_train=None, **params):
        if self.algo == "rf":
            base = dict(n_estimators=400, max_depth=12, min_samples_split=10,
                        min_samples_leaf=3, max_features="sqrt",
                        n_jobs=-1, random_state=42)
            base.update(params)
            if self.task == "classifier":
                return RandomForestClassifier(class_weight="balanced", **base)
            return RandomForestRegressor(**base)

        base = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                    subsample=0.85, colsample_bytree=0.85,
                    reg_alpha=0.0, reg_lambda=1.0, min_child_weight=2,
                    random_state=42, tree_method="hist", n_jobs=-1)
        base.update(params)
        if self.task == "classifier":
            pos = max(int(np.sum(y_train)), 1) if y_train is not None else 1
            neg = max(len(y_train) - pos, 1) if y_train is not None else 1
            return xgb.XGBClassifier(scale_pos_weight=neg / pos,
                                     eval_metric="logloss", **base)
        return xgb.XGBRegressor(eval_metric="rmse", **base)

    def train(self, X, y, tune=False, verbose=True):
        X_arr, _ = self._matrix(X)
        y_arr = np.asarray(y)
        split = int(len(X_arr) * 0.8)
        X_train = self.scaler.fit_transform(X_arr[:split])
        X_test = self.scaler.transform(X_arr[split:])
        y_train, y_test = y_arr[:split], y_arr[split:]

        params = self._tune(X_train, y_train, verbose) if tune else {}
        self.model = self._new(y_train, **params)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        if self.task == "classifier":
            metrics = {"accuracy": float(accuracy_score(y_test, preds))}
        else:
            metrics = {"mae": float(np.mean(np.abs(y_test - preds))),
                       "rmse": float(np.sqrt(np.mean((y_test - preds) ** 2))),
                       "directional_accuracy": dir_acc(y_test, preds)}
        if verbose:
            print(f"{self.algo} {self.task}: {metrics}")
        self._save()
        metrics.update(n_train=len(X_train), n_test=len(X_test))
        return metrics

    def _tune(self, X_train, y_train, verbose, n_trials=None):
        try:
            import optuna
        except ImportError:
            return {}
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = n_trials or (25 if self.algo == "rf" else 30)

        def objective(trial):
            if self.algo == "rf":
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 200, 600, step=100),
                    max_depth=trial.suggest_int("max_depth", 4, 20),
                    min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                    max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                )
            else:
                params = dict(
                    n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
                    max_depth=trial.suggest_int("max_depth", 3, 10),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                    subsample=trial.suggest_float("subsample", 0.6, 1.0),
                    colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                    reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                    min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                )
            scores = []
            for tr, te in DataPreprocessor.walk_forward_splits(
                    len(X_train), n_splits=3, min_train_frac=0.6):
                m = self._new(y_train[tr], **params)
                m.fit(X_train[tr], y_train[tr])
                p = m.predict(X_train[te])
                scores.append(accuracy_score(y_train[te], p)
                              if self.task == "classifier" else dir_acc(y_train[te], p))
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        if verbose:
            print(f"  best {self.algo} params: {study.best_params}")
        return study.best_params

    def walk_forward_score(self, X, y, n_splits=5, verbose=True):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)
        scores = []
        for fold, (tr, te) in enumerate(
                DataPreprocessor.walk_forward_splits(len(X_arr), n_splits)):
            scaler = StandardScaler()
            m = self._new(y_arr[tr])
            m.fit(scaler.fit_transform(X_arr[tr]), y_arr[tr])
            preds = m.predict(scaler.transform(X_arr[te]))
            score = (accuracy_score(y_arr[te], preds)
                     if self.task == "classifier" else dir_acc(y_arr[te], preds))
            scores.append(float(score))
            if verbose:
                print(f"  {self.algo} fold {fold+1}: {score:.4f}")
        return scores

    def predict(self, features):
        if not self._load():
            return None
        row = self.scaler.transform(
            np.asarray(features[self.feature_columns].iloc[-1:], dtype=float))
        if self.task == "classifier":
            return int(self.model.predict(row)[0]), self.model.predict_proba(row)[0]
        return float(self.model.predict(row)[0])

    def feature_importance(self):
        if not self._load():
            return None
        return self.model.feature_importances_



class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden=96, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True,
                            dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class JordanNet(nn.Module):
    """Jordan (1986) recurrence: the previous output is the memory.
    h_t = tanh(W [x_t, c_t]); y_t = W_o h_t; c_{t+1} = 0.5 c_t + tanh(y_t)"""

    def __init__(self, input_size, hidden=64, decay=0.5, dropout=0.25):
        super().__init__()
        self.decay = decay
        self.in_layer = nn.Linear(input_size + 1, hidden)
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden, 1)

    def forward(self, x):
        context = torch.zeros(x.shape[0], 1, device=x.device)
        out = context
        for t in range(x.shape[1]):
            h = torch.tanh(self.in_layer(torch.cat([x[:, t, :], context], dim=1)))
            out = self.out_layer(self.dropout(h))
            context = self.decay * context + torch.tanh(out)
        return out


class SeqModel(BaseModel):
    """Common training/eval code for LSTM and Jordan (both tasks)."""

    net_cls = None
    train_epochs = 80
    train_patience = 12
    wf_patience = None  
    seq_len = 30

    def _sequences(self, X, y, indices=None):

        idx = indices if indices is not None else range(len(X))
        seqs, targets = [], []
        for i in idx:
            if i < self.seq_len:
                continue
            seqs.append(X[i - self.seq_len:i])
            targets.append(y[i])
        return np.array(seqs), np.array(targets)

    def _criterion(self, y_train):
        if self.task == "classifier":
            pos = max(y_train.sum(), 1)
            neg = max(len(y_train) - pos, 1)
            w = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
            return nn.BCEWithLogitsLoss(pos_weight=w)
        return nn.MSELoss()

    def _fit(self, Xtr, ytr, Xval, yval, epochs, patience):

        torch.manual_seed(42)
        net = self.net_cls(Xtr.shape[2]).to(DEVICE)
        criterion = self._criterion(ytr)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

        Xtr_t = torch.FloatTensor(Xtr).to(DEVICE)
        ytr_t = torch.FloatTensor(ytr).unsqueeze(1).to(DEVICE)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr_t, ytr_t), batch_size=32, shuffle=True)

        if patience is None:
            for _ in range(epochs):
                net.train()
                for bx, by in loader:
                    optimizer.zero_grad()
                    loss = criterion(net(bx), by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()
            net.eval()
            return net

        Xval_t = torch.FloatTensor(Xval).to(DEVICE)
        yval_t = torch.FloatTensor(yval).unsqueeze(1).to(DEVICE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)
        best, best_state, waited = float("inf"), None, 0
        for _ in range(epochs):
            net.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(net(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
            net.eval()
            with torch.no_grad():
                val = criterion(net(Xval_t), yval_t).item()
            scheduler.step(val)
            if val < best:
                best, waited = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                if waited >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        return net

    def _eval(self, net, X_seq):
        with torch.no_grad():
            out = net(torch.FloatTensor(X_seq).to(DEVICE)).cpu().numpy().ravel()
        return out

    def train(self, X, y, epochs=None, verbose=True):
        X_arr, _ = self._matrix(X)
        y_arr = np.asarray(y, dtype=float)
        split = int(len(X_arr) * 0.8)
        X_train = self.scaler.fit_transform(X_arr[:split])
        X_test = self.scaler.transform(X_arr[split:])
        Xtr, ytr = self._sequences(X_train, y_arr[:split])
        Xte, yte = self._sequences(X_test, y_arr[split:])
        if len(Xtr) == 0 or len(Xte) == 0:
            raise ValueError(f"Not enough data for seq_len={self.seq_len}")

        self.model = self._fit(Xtr, ytr, Xte, yte,
                               epochs or self.train_epochs, self.train_patience)
        out = self._eval(self.model, Xte)
        if self.task == "classifier":
            preds = (1 / (1 + np.exp(-out)) >= 0.5).astype(int)
            metrics = {"accuracy": float((preds == yte.astype(int)).mean())}
        else:
            metrics = {"mae": float(np.mean(np.abs(yte - out))),
                       "directional_accuracy": dir_acc(yte, out)}
        if verbose:
            print(f"{self.algo} {self.task}: {metrics}")
        self._save()
        metrics.update(n_train=len(Xtr), n_test=len(Xte))
        return metrics

    def _save(self, extra=None):
        joblib.dump({"model": self.model.state_dict(), "scaler": self.scaler,
                     "features": self.feature_columns}, self.path)

    def _loaded(self, bundle):
        state = self.model
        self.model = self.net_cls(len(self.feature_columns)).to(DEVICE)
        self.model.load_state_dict(state)
        self.model.eval()

    def walk_forward_score(self, X, y, n_splits=5, epochs=25, verbose=True):
        """Fresh in-memory net per fold; never touches the saved model."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        scores = []
        for fold, (tr, te) in enumerate(
                DataPreprocessor.walk_forward_splits(len(X_arr), n_splits)):
            scaler = StandardScaler().fit(X_arr[tr])
            X_scaled = scaler.transform(X_arr)
            Xtr, ytr = self._sequences(X_scaled, y_arr, tr)
            Xte, yte = self._sequences(X_scaled, y_arr, te)
            net = self._fit(Xtr, ytr, Xte, yte, epochs, self.wf_patience)
            out = self._eval(net, Xte)
            if self.task == "classifier":
                probs = 1 / (1 + np.exp(-out))
                score = float(((probs >= 0.5).astype(int) == yte.astype(int)).mean())
            else:
                score = dir_acc(yte, out)
            scores.append(score)
            if verbose:
                print(f"  {self.algo} fold {fold+1}: {score:.4f}")
        return scores

    def predict(self, features):
        if not self._load():
            return None
        data = features[self.feature_columns].tail(self.seq_len)
        if len(data) < self.seq_len:
            return None
        x = torch.FloatTensor(self.scaler.transform(data.values)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = float(self.model(x).cpu().numpy().ravel()[0])
        if self.task == "classifier":
            p_up = 1 / (1 + np.exp(-out))
            return (1 if p_up >= 0.5 else 0), [1 - p_up, p_up]
        return out


class LSTMModel(SeqModel):
    algo = "lstm"
    net_cls = LSTMNet
    train_epochs, train_patience, wf_patience = 80, 12, None


class JordanModel(SeqModel):
    algo = "jordan"
    net_cls = JordanNet
    train_epochs, train_patience, wf_patience = 60, 10, 8



NEAT_CONFIG = """
[NEAT]
fitness_criterion      = max
fitness_threshold      = 1000.0
no_fitness_termination = True
pop_size               = {pop_size}
reset_on_extinction    = True

[DefaultGenome]
activation_default      = {activation}
activation_mutate_rate  = 0.0
activation_options      = {activation}
aggregation_default     = sum
aggregation_mutate_rate = 0.0
aggregation_options     = sum
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob           = 0.5
conn_delete_prob        = 0.3
enabled_default         = True
enabled_mutate_rate     = 0.05
feed_forward            = True
initial_connection      = full_direct
node_add_prob           = 0.3
node_delete_prob        = 0.2
num_hidden              = 0
num_inputs              = {num_inputs}
num_outputs             = 1
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 12
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""


NEAT_ROWS_TRAIN, NEAT_ROWS_WF = 800, 400


class NEATModel(BaseModel):


    algo = "neat"

    def __init__(self, task="classifier"):
        super().__init__(task)
        self.config = None
        self.scale = 1.0
        self.config_path = os.path.join(MODELS_DIR, f"neat_{task}_config.ini")

    def _make_config(self, num_inputs, pop_size):
        activation = "sigmoid" if self.task == "classifier" else "tanh"
        with open(self.config_path, "w") as fh:
            fh.write(NEAT_CONFIG.format(num_inputs=num_inputs,
                                        pop_size=pop_size, activation=activation))
        return neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           self.config_path)

    def _fitness(self, outputs, targets):
        if self.task == "classifier":
            return 1.0 - float(np.mean((outputs - targets) ** 2))
        scaled = np.clip(targets / self.scale, -1.0, 1.0)
        return -float(np.mean((outputs - scaled) ** 2))

    def _evolve(self, config, rows, targets, generations, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        def eval_genomes(genomes, cfg):
            for _, genome in genomes:
                net = neat.nn.FeedForwardNetwork.create(genome, cfg)
                out = np.array([net.activate(r)[0] for r in rows])
                genome.fitness = self._fitness(out, targets)

        return neat.Population(config).run(eval_genomes, generations)

    def _outputs(self, genome, config, rows):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        return np.array([net.activate(r)[0] for r in rows])

    def train(self, X, y, pop_size=90, generations=25, verbose=True):
        X_arr, _ = self._matrix(X)
        y_arr = np.asarray(y, dtype=float)
        split = int(len(X_arr) * 0.8)
        X_train = self.scaler.fit_transform(X_arr[:split])
        X_test = self.scaler.transform(X_arr[split:])
        y_train, y_test = y_arr[:split], y_arr[split:]
        if self.task == "regressor":
            self.scale = max(float(np.std(y_train)) * 3.0, 1e-6)

        self.config = self._make_config(X_arr.shape[1], pop_size)
        self.model = self._evolve(self.config, X_train[-NEAT_ROWS_TRAIN:],
                                  y_train[-NEAT_ROWS_TRAIN:], generations)

        out = self._outputs(self.model, self.config, X_test)
        if self.task == "classifier":
            metrics = {"accuracy": float(((out >= 0.5).astype(int)
                                          == y_test.astype(int)).mean())}
        else:
            preds = out * self.scale
            metrics = {"mae": float(np.mean(np.abs(y_test - preds))),
                       "directional_accuracy": dir_acc(y_test, preds)}
        if verbose:
            print(f"neat {self.task}: {metrics}  "
                  f"({len(self.model.connections)} connections)")
        self._save({"config": self.config, "scale": self.scale})
        metrics.update(n_train=len(X_train), n_test=len(X_test))
        return metrics

    def _loaded(self, bundle):
        self.config = bundle["config"]
        self.scale = bundle["scale"]

    def walk_forward_score(self, X, y, n_splits=5, pop_size=60,
                           generations=12, verbose=True):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        scores = []
        for fold, (tr, te) in enumerate(
                DataPreprocessor.walk_forward_splits(len(X_arr), n_splits)):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_arr[tr])
            Xte = scaler.transform(X_arr[te])
            if self.task == "regressor":
                self.scale = max(float(np.std(y_arr[tr])) * 3.0, 1e-6)
            config = self._make_config(X_arr.shape[1], pop_size)
            genome = self._evolve(config, Xtr[-NEAT_ROWS_WF:],
                                  y_arr[tr][-NEAT_ROWS_WF:],
                                  generations, seed=42 + fold)
            out = self._outputs(genome, config, Xte)
            if self.task == "classifier":
                score = float(((out >= 0.5).astype(int)
                               == y_arr[te].astype(int)).mean())
            else:
                score = dir_acc(y_arr[te], out * self.scale)
            scores.append(score)
            if verbose:
                print(f"  neat fold {fold+1}: {score:.4f}")
        return scores

    def predict(self, features):
        if not self._load():
            return None
        row = self.scaler.transform(
            np.asarray(features[self.feature_columns].iloc[-1:], dtype=float))
        out = self._outputs(self.model, self.config, row)[0]
        if self.task == "classifier":
            p_up = float(np.clip(out, 0.0, 1.0))
            return (1 if p_up >= 0.5 else 0), [1 - p_up, p_up]
        return float(out * self.scale)



CLASSIFIERS = {
    "Random Forest": lambda: SkModel("rf", "classifier"),
    "XGBoost": lambda: SkModel("xgb", "classifier"),
    "LSTM": lambda: LSTMModel("classifier"),
    "NEAT": lambda: NEATModel("classifier"),
    "Jordan": lambda: JordanModel("classifier"),
}

REGRESSORS = {
    "Random Forest": lambda: SkModel("rf", "regressor"),
    "XGBoost": lambda: SkModel("xgb", "regressor"),
    "LSTM": lambda: LSTMModel("regressor"),
    "NEAT": lambda: NEATModel("regressor"),
    "Jordan": lambda: JordanModel("regressor"),
}


class Ensemble:
    """Soft-votes RF + XGBoost + LSTM (classification) or averages their
    predicted returns (regression), with per-model weights."""

    BASES = ("Random Forest", "XGBoost", "LSTM")

    def __init__(self, task="classifier", weights=None):
        self.task = task
        registry = CLASSIFIERS if task == "classifier" else REGRESSORS
        self.parts = {name: registry[name]() for name in self.BASES}
        self.weights = weights or {name: 1.0 for name in self.BASES}

    def set_weights(self, scores):
        """Weights proportional to (score - 0.5), clamped to >= 0.05."""
        self.weights = {n: max(s - 0.5, 0.05) for n, s in scores.items()}

    def predict(self, features):
        outs = {n: m.predict(features) for n, m in self.parts.items()}
        outs = {n: o for n, o in outs.items() if o is not None}
        if not outs:
            return None
        total = sum(self.weights.get(n, 1.0) for n in outs)
        if self.task == "classifier":
            proba = sum(self.weights.get(n, 1.0) * np.asarray(o[1], dtype=float)
                        for n, o in outs.items()) / total
            return int(np.argmax(proba)), proba
        return sum(self.weights.get(n, 1.0) * o for n, o in outs.items()) / total
