"""CatBoost (Prokhorenkova et al. 2018): symmetrische Bäume (`cb_tree.py`) plus zwei Maßnahmen gegen **Prediction Shift** - die Verzerrung, die entsteht, wenn der Pseudo-Gradient einer Zeile
von einem Modell stammt, das diese Zeile schon gesehen hat (ihr Rest wirkt dann systematisch zu klein, siehe `cb_evaluation.prediction_shift_bias`). `fit_standard` ist die GEWÖHNLICHE
(verzerrte) Variante - jede Runde nutzt die volle, aktuelle Vorhersage aller Zeilen, so wie gradient-boosting-demo/xgboost-demo/lightgbm-demo es tun. `fit_ordered` ist CatBoosts Antwort:
**Geordnetes Boosting** - eine zufällige Permutation in `n_blocks` Blöcke geteilt; der Baum, der Block k aktualisiert, wird NUR mit Gradienten aus den Blöcken VOR k trainiert. Block 1 hat keine
Vorgänger und bleibt bei der Startvorhersage stehen (derselbe Preis, den CatBoosts eigene O(log n)-Stützmodelle für die ersten Zeilen der Permutation zahlen)."""

from dataclasses import dataclass

import numpy as np

import cb_tree as T

EPS = 1e-12


@dataclass(frozen=True)
class Ensemble:
    trees: tuple                  # gewöhnlich: ein Baum je Runde. geordnet: ein Baum je (Runde, Block>=1)
    f0: float
    learning_rate: float
    task: str
    ordered: bool


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def grad_hess(y, F, task):
    if task == "reg":
        return F - y, np.ones_like(F)
    p = _sigmoid(F)
    return p - y, p * (1.0 - p)


def init_value(y, task):
    if task == "reg":
        return float(np.mean(y))
    p = np.clip(float(np.mean(y)), 1e-6, 1.0 - 1e-6)
    return float(np.log(p / (1.0 - p)))


# --- Gewöhnliches (verzerrtes) Boosting - Vergleichsmaßstab -------------------------------------------------------------------------------------------

def fit_standard(X, y, task, depth=4, lam=1.0, max_bin=63, n_rounds=60, learning_rate=0.2, seed=0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    edges = T.build_bin_edges(X, max_bin)
    f0 = init_value(y, task)
    F = np.full(n, f0)
    trees = []
    for _ in range(n_rounds):
        g, h = grad_hess(y, F, task)
        tree = T.grow(X, g, h, edges, depth, lam)
        F = F + learning_rate * T.predict_value(tree, X)
        trees.append(tree)
    return Ensemble(tuple(trees), f0, learning_rate, task, False)


# --- Geordnetes Boosting -----------------------------------------------------------------------------------------------------------------------------

def fit_ordered(X, y, task, depth=4, lam=1.0, max_bin=63, n_rounds=60, learning_rate=0.2, n_blocks=8, seed=0):
    """`n_blocks` Blöcke einer zufälligen Permutation nähern CatBoosts O(log n) Stützmodelle an (hier: feste, gleich große Blöcke statt geometrisch wachsender - eine bewusste
    Vereinfachung, siehe README). Jede Runde entstehen `n_blocks` Bäume: Baum_k (k>=1) wird NUR mit Zeilen aus den Blöcken 0..k-1 trainiert und aktualisiert NUR die Vorhersage der
    Zeilen in Block k - Block k's eigene Zeilen fließen nie in den Baum ein, der ihre eigene Vorhersage verändert. Block 0 hat keinen Vorgänger; er bekommt einen eigenen, gewöhnlichen
    (selbstbezogenen) Baum - sonst würde seine Vorhersage für immer beim Startwert stehen bleiben, und jeder Baum, der (über die Kette der Blöcke) von ihm abhängt, würde runde für Runde
    dieselbe Korrektur addieren, statt sie mit fortschreitendem Fit schrumpfen zu lassen (führte zu explodierenden Vorhersagen - gemessen und behoben, siehe README). Dieser eine Block
    trägt deshalb denselben (kleinen, auf 1/`n_blocks` der Zeilen begrenzten) Rest an Selbstbezug wie gewöhnliches Boosting - der Preis, den auch CatBoosts eigene Stützmodelle für die
    ersten Zeilen der Permutation zahlen."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    edges = T.build_bin_edges(X, max_bin)
    f0 = init_value(y, task)
    F = np.full(n, f0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    blocks = np.array_split(perm, n_blocks)
    rounds = []
    for _ in range(n_rounds):
        round_trees = {}
        g0, h0 = grad_hess(y[blocks[0]], F[blocks[0]], task)
        tree0 = T.grow(X[blocks[0]], g0, h0, edges, depth, lam)
        F[blocks[0]] = F[blocks[0]] + learning_rate * T.predict_value(tree0, X[blocks[0]])
        round_trees[0] = tree0
        seen = blocks[0]
        for k in range(1, n_blocks):
            g_all, h_all = grad_hess(y[seen], F[seen], task)
            tree = T.grow(X[seen], g_all, h_all, edges, depth, lam)
            F[blocks[k]] = F[blocks[k]] + learning_rate * T.predict_value(tree, X[blocks[k]])
            round_trees[k] = tree
            seen = np.concatenate([seen, blocks[k]])
        rounds.append(round_trees)
    return Ensemble(tuple(rounds), f0, learning_rate, task, True), blocks


def predict_raw(ensemble, X, upto=None):
    """Vorhersage für NEUE Zeilen (nie Teil einer Permutation) - immer die volle Summe aller bis dahin gebauten Bäume, ob geordnet trainiert oder nicht (die Reihenfolge war nur eine
    Trainings-Vorkehrung; einmal fertig, ist jeder Baum ein normaler additiver Beitrag)."""
    if ensemble.ordered:
        rounds = ensemble.trees[:upto] if upto else ensemble.trees
        F = np.full(len(X), ensemble.f0)
        for round_trees in rounds:
            k_last = max(round_trees)
            F = F + ensemble.learning_rate * T.predict_value(round_trees[k_last], X)
        return F
    trees = ensemble.trees[:upto] if upto else ensemble.trees
    F = np.full(len(X), ensemble.f0)
    for tree in trees:
        F = F + ensemble.learning_rate * T.predict_value(tree, X)
    return F


def predict_value(ensemble, X, upto=None):
    F = predict_raw(ensemble, X, upto)
    return F if ensemble.task == "reg" else _sigmoid(F)


def predict(ensemble, X, upto=None):
    v = predict_value(ensemble, X, upto)
    return (v > 0.5).astype(int) if ensemble.task == "class" else v


def n_rounds_of(ensemble):
    return len(ensemble.trees)
