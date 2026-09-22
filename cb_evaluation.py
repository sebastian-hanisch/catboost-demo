"""Messungen an CatBoost: Prediction-Shift-Verzerrung (in-sample gegen frisch gezogene Werte, auf reinem Rauschen - kein Signal, nur die Verzerrung selbst), Leckage der naiven
Ziel-Mittelwert-Kodierung gegen geordnete Ziel-Statistik (Trainings-/Testabstand der Korrelation mit dem Ziel), Wirkung der Kardinalität (Zahl der Depot-Stufen)."""

from dataclasses import dataclass

import numpy as np

import cb_algorithm as cbm
import cb_constants as C
import cb_encoding as E
import cb_scenario as S
import cb_tree as T


def baseline_error(ds, task):
    _, ytr, _, yte = S.split(ds, task)
    if task == "class":
        return float(np.mean(yte != int(ytr.mean() > 0.5)))
    return float(np.sqrt(np.mean((yte - ytr.mean()) ** 2)))


def _metrics(ensemble, X, y, task):
    if task == "class":
        return {"error": float(np.mean(cbm.predict(ensemble, X) != y))}
    v = cbm.predict_value(ensemble, X)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def primary(metrics, task):
    return metrics["error"] if task == "class" else metrics["rmse"]


def ensemble_importances(ensemble, n_features):
    trees = ensemble.trees if not ensemble.ordered else [rt[max(rt)] for rt in ensemble.trees]
    imps = [T.importances(t) for t in trees]
    return np.mean(imps, axis=0) if imps else np.zeros(n_features)


@dataclass
class Analysis:
    ds: object
    task: str
    encoding: str
    depth: int
    lam: float
    max_bin: int
    n_rounds: int
    lr: float
    ordered: bool
    n_blocks: int
    ensemble: object
    train: dict
    test: dict
    baseline: float
    verdict: str
    imp: np.ndarray
    encoders: dict


def analyse(task, encoding, depth, lam, max_bin, n_rounds, lr, ordered, n_blocks, n, n_noise, label_noise, n_depots, seed):
    ds = S.generate_dataset(n, n_noise, label_noise if task == "class" else 0, seed, n_depots=n_depots)
    Xtr, ytr, Xte, yte = S.split(ds, task)
    Xtr_e, encoders = E.encode_dataset(Xtr, C.CAT_FEATURES, ytr.astype(float), encoding, seed)
    Xte_e = E.apply_encoding(Xte, C.CAT_FEATURES, encoders)
    if ordered:
        ensemble, _ = cbm.fit_ordered(Xtr_e, ytr.astype(float), task, depth=depth, lam=lam, max_bin=max_bin, n_rounds=n_rounds, learning_rate=lr, n_blocks=n_blocks, seed=0)
    else:
        ensemble = cbm.fit_standard(Xtr_e, ytr.astype(float), task, depth=depth, lam=lam, max_bin=max_bin, n_rounds=n_rounds, learning_rate=lr, seed=0)
    train = _metrics(ensemble, Xtr_e, ytr, task)
    test = _metrics(ensemble, Xte_e, yte, task)
    baseline = baseline_error(ds, task)
    a = Analysis(ds, task, encoding, depth, lam, max_bin, n_rounds, lr, ordered, n_blocks, ensemble, train, test, baseline, "", ensemble_importances(ensemble, Xtr_e.shape[1]), encoders)
    a.verdict = verdict(a)
    return a


def verdict(a):
    tr, te = primary(a.train, a.task), primary(a.test, a.task)
    if a.n_rounds <= 1:
        return "stump"
    over = (te - tr > C.OVERFIT_GAP_CLASS) if a.task == "class" else (te > C.OVERFIT_RATIO_REG * max(tr, 1e-9))
    if over:
        return "overfit"
    return "underfit" if te > C.UNDERFIT_SHARE * a.baseline else "ok"


def round_rows(a, ks=None):
    ks = ks or sorted(set(np.unique(np.round(np.geomspace(1, a.n_rounds, min(20, a.n_rounds))).astype(int))))
    Xtr, ytr, Xte, yte = S.split(a.ds, a.task)
    Xtr_e = E.apply_encoding(Xtr, C.CAT_FEATURES, a.encoders)
    Xte_e = E.apply_encoding(Xte, C.CAT_FEATURES, a.encoders)
    rows = []
    for k in ks:
        tr = primary(_metrics_upto(a.ensemble, Xtr_e, ytr, a.task, k), a.task)
        te = primary(_metrics_upto(a.ensemble, Xte_e, yte, a.task, k), a.task)
        rows.append({"k": int(k), "train": tr, "test": te})
    return rows


def _metrics_upto(ensemble, X, y, task, upto):
    if task == "class":
        return {"error": float(np.mean(cbm.predict(ensemble, X, upto=upto) != y))}
    v = cbm.predict_value(ensemble, X, upto=upto)
    return {"rmse": float(np.sqrt(np.mean((v - y) ** 2))), "mae": float(np.mean(np.abs(v - y)))}


def best_round(rows):
    return min(rows, key=lambda r: (r["test"], r["k"]))


# --- Prediction Shift: in-sample gegen frisch gezogenes Rauschen (kein echtes Signal) ------------------------------------------------------------------

DEPTH_GRID = (1, 2, 3, 4, 5, 6)


def prediction_shift_bias(n, sigma, depth, max_bin, seed):
    """Ein Baum wächst auf reinem Rauschen (X ohne jeden Bezug zu y) - `in_sample` misst den Rest gegen die eigenen Trainingswerte (die den Baum mitbestimmt haben), `fresh` gegen einen
    unabhängig gezogenen Satz Werte derselben Verteilung an denselben Blättern (nie vom Baum gesehen). `true_var` = sigma² ist bekannt, weil wir die Daten selbst erzeugen."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = rng.normal(0.0, sigma, n)
    edges = T.build_bin_edges(X, max_bin)
    tree = T.grow(X, -y, np.ones(n), edges, depth, lam=0.0)
    pred = T.predict_value(tree, X)
    resid_insample = y - pred
    y_fresh = np.random.default_rng([int(seed), 999]).normal(0.0, sigma, n)
    resid_fresh = y_fresh - pred
    return {"in_sample": float(np.mean(resid_insample ** 2)), "fresh": float(np.mean(resid_fresh ** 2)), "true": sigma ** 2}


def prediction_shift_rows(n, sigma, max_bin, seed, grid=DEPTH_GRID):
    return [{"depth": d, **prediction_shift_bias(n, sigma, d, max_bin, seed)} for d in grid]


# --- Leckage der naiven Kodierung gegen geordnete Ziel-Statistik --------------------------------------------------------------------------------------

def encoding_r2(cat_col, n, n_depots, seed):
    ds = S.generate_dataset(n=n, seed=seed, n_depots=n_depots)
    Xtr, ytr, Xte, yte = S.split(ds, "reg")
    depot_tr = Xtr[:, cat_col].astype(int)
    depot_te = Xte[:, cat_col].astype(int)
    naive = E.naive_target_mean(depot_tr, ytr)
    ordered = E.ordered_target_stat(depot_tr, ytr, seed=0)
    lookup = E._category_lookup(depot_tr, ytr, 1.0)
    naive_test = lookup[np.clip(depot_te, 0, len(lookup) - 1)]
    r_naive_tr = float(np.corrcoef(ytr, naive)[0, 1] ** 2)
    r_test = float(np.corrcoef(yte, naive_test)[0, 1] ** 2)                    # dieselbe Vorhersage-Kodierung für naiv UND geordnet nach dem Training (siehe README)
    r_ord_tr = float(np.corrcoef(ytr, ordered)[0, 1] ** 2)
    return {"naive_train": r_naive_tr, "test": r_test, "ordered_train": r_ord_tr}


def encoding_leakage_rows(n, n_depots=24, cat_col=8, seeds=C.SWEEP_SEEDS):
    rows = [encoding_r2(cat_col, n, n_depots, sd) for sd in seeds]
    return {"naive_train": float(np.mean([r["naive_train"] for r in rows])), "test": float(np.mean([r["test"] for r in rows])), "ordered_train": float(np.mean([r["ordered_train"] for r in rows]))}


# --- Wirkung der Kardinalität --------------------------------------------------------------------------------------------------------------------------

CARDINALITY_GRID = (4, 8, 16, 24, 48)


def cardinality_rows(n, cat_col=8, grid=CARDINALITY_GRID, seeds=C.SWEEP_SEEDS):
    rows = []
    for n_depots in grid:
        r = [encoding_r2(cat_col, n, n_depots, sd) for sd in seeds]
        naive_gap = float(np.mean([x["naive_train"] - x["test"] for x in r]))
        ordered_gap = float(np.mean([x["ordered_train"] - x["test"] for x in r]))
        rows.append({"n_depots": n_depots, "naive_gap": naive_gap, "ordered_gap": ordered_gap, "rows_per_depot": round(n * 0.7 / n_depots)})
    return rows
