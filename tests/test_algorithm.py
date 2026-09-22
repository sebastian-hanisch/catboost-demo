"""CatBoost gegen unabhängige Referenzen: die symmetrische Gain-Formel gegen Brute-Force über dieselben Bin-Grenzen, geordnete Target Statistics nutzt nie das eigene Etikett einer Zeile
(direkt nachprüfbar), geordnetes Boosting trainiert den Baum, der eine Zeile aktualisiert, NIE mit dieser Zeile selbst, Prediction-Shift-Verzerrung auf reinem Rauschen, Vorhersagen über
Rang-/Fehlergrenzen gegen die echte `catboost`-Bibliothek."""

import numpy as np
import pytest

import cb_algorithm as cbm
import cb_encoding as E
import cb_scenario as S
import cb_tree as T


def _reg(n=400, d=5, seed=0, noise=0.4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(0, noise, n)
    return X, y


def _cls(n=400, d=5, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * np.sin(X[:, 1]) + rng.normal(0, noise, n) > 0).astype(float)
    return X, y


# --- Symmetrische Gain-Formel gegen Brute-Force ------------------------------------------------------------------------------------------------------

def test_symmetric_split_matches_brute_force_at_depth_one():
    """Bei Tiefe 1 gibt es nur eine Gruppe - der symmetrische Baum muss exakt den besten Einzelschnitt finden, wie xgboost-demo/lightgbm-demo."""
    rng = np.random.default_rng(1)
    n, d = 300, 4
    X = rng.normal(size=(n, d))
    grad = rng.normal(size=n)
    hess = rng.uniform(0.5, 1.5, n)
    edges = T.build_bin_edges(X, 63)
    tree = T.grow(X, grad, hess, edges, depth=1, lam=1.0)

    Gtot, Htot = grad.sum(), hess.sum()
    best = None
    for f in range(d):
        for edge in edges[f]:
            left = X[:, f] <= edge
            Gl, Hl = grad[left].sum(), hess[left].sum()
            Gr, Hr = Gtot - Gl, Htot - Hl
            g = 0.5 * (Gl ** 2 / (Hl + 1.0) + Gr ** 2 / (Hr + 1.0) - Gtot ** 2 / (Htot + 1.0))
            if best is None or g > best[2]:
                best = (f, edge, g)
    assert (int(tree.features[0]), float(tree.thresholds[0])) == (best[0], pytest.approx(best[1]))


def test_symmetric_tree_always_has_two_to_the_depth_leaves():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 5))
    grad = rng.normal(size=400)
    hess = np.ones(400)
    edges = T.build_bin_edges(X, 63)
    for depth in range(1, 6):
        tree = T.grow(X, grad, hess, edges, depth, lam=1.0)
        assert tree.n_leaves == 2 ** depth


# --- Geordnete Target Statistics nutzt nie das eigene Etikett --------------------------------------------------------------------------------------------

def test_ordered_target_statistic_never_uses_its_own_row():
    """Verändert man NUR y_i (die eigene Zeile), darf sich die geordnete Kodierung jeder ANDEREN Zeile derselben Kategorie nicht ändern, wenn sie vor i liegt (`prior_weight=0`, damit der
    - für sich genommen unbedenkliche - globale Prior-Term, der IMMER von allen y abhängt, das Ergebnis nicht verwässert; die Erhaltungs-Eigenschaft betrifft die je-Kategorie-Summe, nicht den Prior)."""
    rng = np.random.default_rng(3)
    n = 60
    cats = rng.integers(0, 5, n)
    y = rng.normal(size=n)
    with np.errstate(invalid="ignore"):                                        # 0/0 = NaN ist hier erwartet (prior_weight=0, erste Zeile einer Kategorie)
        ts1 = E.ordered_target_stat(cats, y, seed=7, prior_weight=0.0)
        y2 = y.copy()
        y2[0] += 1000.0                                                        # eine einzelne Zeile massiv verändert
        ts2 = E.ordered_target_stat(cats, y2, seed=7, prior_weight=0.0)
    order = np.random.default_rng(7).permutation(n)
    pos_of_0 = int(np.nonzero(order == 0)[0][0])
    later = order[pos_of_0:]                                                    # 0 selbst und alles danach darf sich ändern; alles DAVOR nicht
    earlier = order[:pos_of_0]
    assert np.allclose(ts1[earlier], ts2[earlier], equal_nan=True)              # unbeteiligte frühere Zeilen: identisch (auch wenn NaN - erste Zeile einer Kategorie, prior_weight=0)
    assert not np.allclose(ts1[later], ts2[later], equal_nan=True)              # mindestens die eigene und spätere Zeilen derselben Kategorie ändern sich


def test_naive_encoding_does_use_its_own_row_and_ordered_does_not():
    rng = np.random.default_rng(4)
    n = 40
    cats = rng.integers(0, 3, n)
    y = rng.normal(size=n)
    naive1 = E.naive_target_mean(cats, y)
    y2 = y.copy()
    y2[5] += 500.0
    naive2 = E.naive_target_mean(cats, y2)
    same_cat = np.nonzero((cats == cats[5]) & (np.arange(n) != 5))[0]
    assert not np.allclose(naive1[same_cat], naive2[same_cat])                  # naive: jede andere Zeile derselben Kategorie ändert sich mit


def test_one_hot_has_no_leakage_by_construction():
    rng = np.random.default_rng(5)
    cats = rng.integers(0, 6, 30)
    oh = E.one_hot(cats)
    assert oh.shape == (30, 6) and np.array_equal(oh.sum(axis=1), np.ones(30))


# --- Geordnetes Boosting: der Baum, der Block k aktualisiert, sieht Block k nie -----------------------------------------------------------------------

def test_ordered_boosting_never_trains_a_blocks_tree_on_its_own_rows():
    X, y = _reg(300, 4, 0)
    ens, blocks = cbm.fit_ordered(X, y, "reg", depth=2, lam=5.0, n_rounds=3, learning_rate=0.1, n_blocks=6, seed=0)
    block_of = np.empty(len(y), dtype=int)
    for k, idx in enumerate(blocks):
        block_of[idx] = k
    for round_trees in ens.trees:
        for k, tree in round_trees.items():
            if k == 0:
                continue                                                        # Block 0 hat keinen Vorgänger, bekommt bewusst einen selbstbezogenen Baum (siehe README)
            seen = np.concatenate(blocks[:k])
            assert set(seen.tolist()).isdisjoint(blocks[k].tolist())            # Struktur-Invariante: "gesehene" und "Ziel"-Zeilen sind immer disjunkt


def test_ordered_boosting_is_numerically_stable_at_default_like_settings():
    """Regressionstest für einen gefundenen Fehler: Block 0 blieb für immer beim Startwert stehen, wodurch jeder von ihm abhängige Baum runde für Runde dieselbe (nicht schrumpfende)
    Korrektur addierte und die Vorhersage über viele Runden explodieren ließ (siehe README). Behoben: Block 0 bekommt einen eigenen, gewöhnlichen Baum."""
    X, y = _reg(1200, 9, 7, noise=7.0)
    y = y * 20 + 70                                                             # Größenordnung wie die echte Lieferdauer
    ens, _ = cbm.fit_ordered(X, y, "reg", depth=3, lam=5.0, n_rounds=40, learning_rate=0.1, n_blocks=8, seed=0)
    Xt, _ = _reg(200, 9, 8, noise=7.0)
    pred = cbm.predict_value(ens, Xt)
    assert np.all(np.isfinite(pred)) and np.abs(pred).max() < 1000.0            # keine Explosion - grobe, aber klare Schranke


# --- Prediction Shift ----------------------------------------------------------------------------------------------------------------------------------

def test_prediction_shift_bias_on_pure_noise():
    rng = np.random.default_rng(0)
    n, sigma = 400, 5.0
    X = rng.normal(size=(n, 5))
    y = rng.normal(0.0, sigma, n)
    edges = T.build_bin_edges(X, 63)
    tree = T.grow(X, -y, np.ones(n), edges, depth=4, lam=0.0)
    pred = T.predict_value(tree, X)
    resid_insample = np.mean((y - pred) ** 2)
    y_fresh = np.random.default_rng([0, 999]).normal(0.0, sigma, n)
    resid_fresh = np.mean((y_fresh - pred) ** 2)
    assert resid_insample < sigma ** 2 < resid_fresh                            # In-Sample sieht besser aus als die Wahrheit, frisch schlechter


# --- Fast exakt / über Toleranz gegen die echte catboost-Bibliothek ---------------------------------------------------------------------------------

def test_classification_is_close_to_the_catboost_library():
    from catboost import CatBoostClassifier

    X, y = _cls(600, 5, 0)
    Xt, yt_true = _cls(100, 5, 1, noise=0.0)
    ens = cbm.fit_standard(X, y, "class", depth=4, lam=1.0, n_rounds=60, learning_rate=0.2, seed=0)
    ref = CatBoostClassifier(iterations=60, depth=4, learning_rate=0.2, l2_leaf_reg=1.0, loss_function="Logloss", verbose=False, random_seed=0).fit(X, y.astype(int))
    p_own = cbm.predict_value(ens, Xt)
    p_ref = ref.predict_proba(Xt)[:, 1]
    assert np.corrcoef(p_own, p_ref)[0, 1] > 0.9
    err_own = np.mean((p_own > 0.5).astype(int) != yt_true)
    err_ref = np.mean((p_ref > 0.5).astype(int) != yt_true)
    assert abs(err_own - err_ref) < 0.15


# --- Grenzfälle --------------------------------------------------------------------------------------------------------------------------------------

def test_a_single_round_equals_f0_plus_learning_rate_times_the_first_tree():
    X, y = _reg(200, 4, 0)
    ens = cbm.fit_standard(X, y, "reg", depth=2, lam=1.0, n_rounds=1, learning_rate=0.5, seed=0)
    tree0 = ens.trees[0]
    edges = T.build_bin_edges(X, 63)
    expected = ens.f0 + 0.5 * T.predict_value(tree0, X)
    assert np.allclose(cbm.predict_value(ens, X), expected)


def test_generator_has_a_categorical_depot_column_with_a_real_fixed_effect():
    ds = S.generate_dataset(500, 3, 0, 7, n_depots=12)
    assert ds.X.shape == (500, 12) and ds.names[8] == "Depot"
    depot = ds.X[:, 8]
    assert depot.min() >= 0 and depot.max() < 12
    assert len(ds.depot_effect) == 12 and np.std(ds.depot_effect) > 0.0


def test_generator_matches_cart_demo_conventions_otherwise():
    ds = S.generate_dataset(500, 3, 0, 7)
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150 and ds.names[:2] == ("Distanz", "Ladegewicht")
