"""Unabhängige Orakel für die CatBoost-Demo.

1. Symmetrischer Baum gegen Brute-Force (Tiefe 1-3, jede Ebene: alle Merkmale x alle Bin-Grenzen, Gain je Gruppe mit Masken statt Histogrammen).
2. Regression gegen die echte `catboost`-Bibliothek (score_function="L2", Newton, ohne Zufall, Plain): mit ganzzahligen Merkmalen (jede Kante zwischen zwei Werten ist ein Kandidat) stimmen
   Vorhersagen über mehrere Runden überein. Klassifikation: die Bibliothek wählt Splits mit einem reinen Gradienten-L2-Score (keine zweite Ableitung), die Demo mit dem Gain 1/2*[...] - deshalb
   werden dort nur Bäume verglichen, bei denen beide denselben Split gewählt haben (dann müssen die Newton-Blattwerte gleich sein).
3. Geordnete Target Statistics gegen eine O(n^2)-Referenz (Mittel der Zeilen derselben Kategorie, die in der Permutation VOR der Zeile stehen).
4. Geordnetes Boosting gegen eine Referenz mit Brute-Force-Stümpfen (Tiefe 1), gleiche Permutation und Blockstruktur.
5. Regressionstest: die Trainingskurve (round_rows) muss am Ende den Trainingsfehler der Kennzahl treffen (früher wurde sie mit der Kodierung nachgeschlagen, nicht mit der Trainingskodierung).
"""

import json
import os
import tempfile

import numpy as np
import pytest

import cb_algorithm as cbm
import cb_encoding as E
import cb_evaluation as ev
import cb_tree as T


def _brute_force_tree(X, g, h, edges, depth, lam):
    """Referenz: pro Ebene alle (Merkmal, Kante)-Kandidaten, Gain je aktueller Gruppe mit Masken; Gleichstand: kleinstes Merkmal, dann kleinste Kante (wie die Demo)."""
    group = np.zeros(len(g), dtype=int)
    feats, thrs = [], []
    for level in range(depth):
        best = None
        for f in range(X.shape[1]):
            for thr in edges[f]:
                total = 0.0
                for gi in range(1 << level):
                    m = group == gi
                    left = m & (X[:, f] <= thr)
                    right = m & (X[:, f] > thr)
                    Hl, Hr = h[left].sum(), h[right].sum()
                    if Hl > 1e-12 and Hr > 1e-12:
                        Gl, Gr, G, H = g[left].sum(), g[right].sum(), g[m].sum(), h[m].sum()
                        total += 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam) - G ** 2 / (H + lam))
                if best is None or total > best[0] + 1e-12:
                    best = (total, f, float(thr))
        feats.append(best[1]), thrs.append(best[2])
        group = group * 2 + (X[:, best[1]] > best[2]).astype(int)
    leaves = np.array([-g[group == k].sum() / (h[group == k].sum() + lam) if (group == k).any() else 0.0 for k in range(1 << depth)])
    return feats, thrs, leaves


def test_symmetric_tree_matches_brute_force_on_random_instances():
    rng = np.random.default_rng(11)
    for it in range(60):
        n, d, depth = int(rng.integers(30, 90)), int(rng.integers(1, 4)), int(rng.integers(1, 4))
        X = rng.normal(size=(n, d))
        if it % 4 == 0:
            X = np.round(X)                                                    # viele gleiche Werte
        g = rng.normal(size=n)
        h = rng.uniform(0.3, 1.5, n)
        lam = float(rng.choice([0.0, 1.0, 5.0]))
        edges = T.build_bin_edges(X, 12)
        tree = T.grow(X, g, h, edges, depth, lam)
        feats, thrs, leaves = _brute_force_tree(X, g, h, edges, depth, lam)
        assert list(tree.features) == feats and np.allclose(tree.thresholds, thrs), it
        assert np.allclose(tree.values, leaves, atol=1e-9), it


def test_regression_matches_catboost_library_on_integer_features():
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(2)
    compared = 0
    for it in range(60):
        n, d, levels = int(rng.integers(300, 600)), int(rng.integers(1, 5)), int(rng.integers(2, 8))
        X = rng.integers(0, levels, size=(n, d)).astype(float)
        y = 1.5 * X[:, 0] - 2.0 * (X[:, -1] > 2) + rng.normal(size=n)
        depth, lam, rounds, lr = int(rng.integers(1, 5)), float(rng.choice([0.0, 0.5, 1.0, 3.0])), int(rng.integers(1, 5)), float(rng.choice([0.1, 0.3, 1.0]))
        ens = cbm.fit_standard(X, y, "reg", depth=depth, lam=lam, n_rounds=rounds, learning_rate=lr)
        m = catboost.CatBoostRegressor(iterations=rounds, depth=depth, learning_rate=lr, l2_leaf_reg=lam, score_function="L2", random_strength=0, bootstrap_type="No", boosting_type="Plain",
                                       leaf_estimation_iterations=1, leaf_estimation_method="Newton", border_count=254, verbose=False, thread_count=1, random_seed=0, boost_from_average=True).fit(X, y)
        Xt = rng.integers(0, levels, size=(80, d)).astype(float)
        assert np.max(np.abs(cbm.predict_value(ens, Xt) - m.predict(Xt))) < 1e-4, (it, depth, lam, rounds, lr)
        compared += 1
    assert compared == 60


def test_classification_leaf_values_match_catboost_when_the_split_is_the_same():
    catboost = pytest.importorskip("catboost")
    rng = np.random.default_rng(5)
    same = 0
    for it in range(40):
        n, d, levels = int(rng.integers(300, 600)), int(rng.integers(1, 4)), int(rng.integers(2, 8))
        X = rng.integers(0, levels, size=(n, d)).astype(float)
        y = (X[:, 0] + rng.normal(0, 1.5, n) > levels / 2).astype(float)
        if y.min() == y.max():
            continue
        lam, lr = float(rng.choice([0.5, 1.0, 3.0])), float(rng.choice([0.3, 1.0]))
        ens = cbm.fit_standard(X, y, "class", depth=1, lam=lam, n_rounds=1, learning_rate=lr)
        m = catboost.CatBoostClassifier(iterations=1, depth=1, learning_rate=lr, l2_leaf_reg=lam, score_function="L2", random_strength=0, bootstrap_type="No", boosting_type="Plain",
                                        leaf_estimation_iterations=1, leaf_estimation_method="Newton", border_count=254, verbose=False, thread_count=1, random_seed=0, boost_from_average=True,
                                        loss_function="Logloss").fit(X, y.astype(int))
        path = tempfile.mktemp(suffix=".json")
        try:
            m.save_model(path, format="json")
            with open(path, encoding="utf-8") as fh:
                tree = json.load(fh)["oblivious_trees"][0]
        finally:
            if os.path.exists(path):
                os.remove(path)
        split = tree["splits"][0]
        t = ens.trees[0]
        if split["float_feature_index"] == int(t.features[0]) and abs(split["border"] - t.thresholds[0]) < 0.6:
            assert np.allclose(tree["leaf_values"], t.values * lr, atol=1e-4)
            same += 1
    assert same >= 25


def test_ordered_target_statistic_matches_quadratic_reference():
    rng = np.random.default_rng(3)
    for it in range(60):
        n, k = int(rng.integers(5, 80)), int(rng.integers(1, 7))
        cats = rng.integers(0, k, n)
        y = rng.normal(size=n)
        w, seed = float(rng.choice([0.5, 1.0, 2.0])), int(rng.integers(0, 100))
        ts = E.ordered_target_stat(cats, y, seed, prior_weight=w)
        pos = np.empty(n, dtype=int)
        pos[np.random.default_rng(seed).permutation(n)] = np.arange(n)
        prior = y.mean()
        ref = np.empty(n)
        for i in range(n):
            before = [j for j in range(n) if cats[j] == cats[i] and pos[j] < pos[i]]
            ref[i] = (y[before].sum() + w * prior) / (len(before) + w)
        assert np.allclose(ts, ref), it


def test_naive_target_mean_by_hand():
    cats = np.array([0, 0, 1])
    y = np.array([1.0, 3.0, 5.0])
    # Prior = 3, Gewicht 1: Kategorie 0: (4 + 3) / 3, Kategorie 1: (5 + 3) / 2
    assert np.allclose(E.naive_target_mean(cats, y), [7 / 3, 7 / 3, 4.0])


def _stump(X, g, h, edges, lam):
    best = None
    for f in range(X.shape[1]):
        for thr in edges[f]:
            left = X[:, f] <= thr
            Hl, Hr = h[left].sum(), h[~left].sum()
            if Hl > 1e-12 and Hr > 1e-12:
                Gl, Gr, G, H = g[left].sum(), g[~left].sum(), g.sum(), h.sum()
                gain = 0.5 * (Gl ** 2 / (Hl + lam) + Gr ** 2 / (Hr + lam) - G ** 2 / (H + lam))
                if best is None or gain > best[0] + 1e-12:
                    best = (gain, f, float(thr))
    _, f, thr = best
    left = X[:, f] <= thr
    return f, thr, -g[left].sum() / (h[left].sum() + lam), -g[~left].sum() / (h[~left].sum() + lam)


def test_ordered_boosting_matches_stump_reference():
    rng = np.random.default_rng(8)
    for it in range(15):
        n, d, nb, rounds = int(rng.integers(60, 120)), 3, int(rng.integers(3, 6)), 3
        X = rng.normal(size=(n, d))
        y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(0, 0.3, n)
        lam, lr, seed = float(rng.choice([1.0, 3.0])), float(rng.choice([0.3, 1.0])), int(rng.integers(0, 50))
        ens, blocks = cbm.fit_ordered(X, y, "reg", depth=1, lam=lam, max_bin=15, n_rounds=rounds, learning_rate=lr, n_blocks=nb, seed=seed)
        # Referenz: dieselbe Permutation/Blöcke, Stümpfe per Brute-Force
        ref_blocks = np.array_split(np.random.default_rng(seed).permutation(n), nb)
        assert all(np.array_equal(a, b) for a, b in zip(blocks, ref_blocks))
        edges = T.build_bin_edges(X, 15)
        F = np.full(n, y.mean())
        last = []
        for _ in range(rounds):
            f, thr, vl, vr = _stump(X[ref_blocks[0]], F[ref_blocks[0]] - y[ref_blocks[0]], np.ones(len(ref_blocks[0])), edges, lam)
            F[ref_blocks[0]] += lr * np.where(X[ref_blocks[0], f] <= thr, vl, vr)
            seen = ref_blocks[0]
            for k in range(1, nb):
                f, thr, vl, vr = _stump(X[seen], F[seen] - y[seen], np.ones(len(seen)), edges, lam)
                F[ref_blocks[k]] += lr * np.where(X[ref_blocks[k], f] <= thr, vl, vr)
                seen = np.concatenate([seen, ref_blocks[k]])
            last.append((f, thr, vl, vr))                                      # der Baum des letzten Blocks bestimmt die Vorhersage für neue Zeilen
        Xt = rng.normal(size=(40, d))
        pred = np.full(40, y.mean())
        for f, thr, vl, vr in last:
            pred += lr * np.where(Xt[:, f] <= thr, vl, vr)
        assert np.allclose(cbm.predict_value(ens, Xt), pred, atol=1e-9), it


@pytest.mark.parametrize("encoding", ["ordered", "naive", "onehot"])
def test_training_curve_ends_at_the_reported_training_error(encoding):
    a = ev.analyse("class", encoding, 3, 5.0, 63, 12, 0.1, False, 8, 600, 3, 0, 24, 7)
    last = ev.round_rows(a, ks=[12])[0]
    assert last["train"] == pytest.approx(ev.primary(a.train, "class"))
    assert last["test"] == pytest.approx(ev.primary(a.test, "class"))
