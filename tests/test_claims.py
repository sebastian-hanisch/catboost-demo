"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-24 nach der Bin-Korrektur, Toleranzen fangen Rundung ab). `analyse()` und die Experiment-Funktionen sind deterministisch (kein Zufall
außer im Datenerzeuger, der - festen, mit `seed` reproduzierbaren - Permutation des geordneten Boostings/der Kodierung und der Teilstichprobe)."""

import functools

import pytest

import cb_constants as C
import cb_evaluation as ev
import cb_scenario as S

PRESET = {"standard": "🌳 Standard", "stump": "🪓 Ein Schritt (kein Boosting)", "naive": "🎯 Naives Target Encoding", "ordered": "⚖️ Geordnete Target Statistics", "reg": "📈 Regression Standard"}


@functools.lru_cache(maxsize=None)
def _preset(key):
    p = C.PRESETS[PRESET[key]]
    return ev.analyse(p["task"], p["encoding"], p["depth"], p["lam"], 63, p["n_rounds"], p["lr"], p["ordered"], p["n_blocks"], p["n"], p["n_noise"], p["label_noise"], p["n_depots"], p["seed"])


def _help(key, *needles):
    text = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in text, (key, n)


# --- Preset-Hilfen --------------------------------------------------------------------------------------------------------------------------------

def test_standard_preset():
    a = _preset("standard")
    assert (a.train["error"], a.test["error"], a.baseline) == pytest.approx((0.2071, 0.2083, 0.4889), abs=0.0015)
    _help("standard", "20.7 %", "20.8 %", "48.9 %")


def test_single_step_preset_beats_guessing_but_not_by_much():
    a = _preset("stump")
    assert len(a.ensemble.trees) == 1 and a.verdict == "stump"
    assert a.test["error"] == pytest.approx(0.2889, abs=0.0015)
    _help("stump", "28.9 %")


def test_naive_vs_ordered_encoding_end_to_end_preset_numbers():
    naive = _preset("naive")
    ordered = _preset("ordered")
    assert (naive.train["error"], naive.test["error"]) == pytest.approx((0.1524, 0.1944), abs=0.0015)
    assert (ordered.train["error"], ordered.test["error"]) == pytest.approx((0.1738, 0.2000), abs=0.0015)
    _help("naive", "15.2 %", "19.4 %")
    _help("ordered", "17.4 %", "20.0 %")
    assert ordered.test["error"] > naive.test["error"]                                       # geordnet in diesem Durchlauf etwas schlechter als naiv (Text der Hilfe)


def test_regression_standard_preset():
    a = _preset("reg")
    assert a.task == "reg"
    assert a.test["rmse"] == pytest.approx(13.645, abs=0.02)
    _help("reg", "13.6")


def test_every_preset_is_a_valid_setting():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["encoding"] in C.ENCODING_OPTIONS
        assert C.DEPTH_MIN <= p["depth"] <= C.DEPTH_MAX and C.N_ROUNDS_MIN <= p["n_rounds"] <= C.N_ROUNDS_MAX
        assert C.LR_MIN <= p["lr"] <= C.LR_MAX and C.LAM_MIN <= p["lam"] <= C.LAM_MAX
        assert C.N_BLOCKS_MIN <= p["n_blocks"] <= C.N_BLOCKS_MAX and p["n_depots"] >= 2
        assert C.N_MIN <= p["n"] <= C.N_MAX and name in C.PRESET_HELP


# --- Prediction Shift ----------------------------------------------------------------------------------------------------------------------------------

def test_prediction_shift_experiment_numbers():
    rows = ev.prediction_shift_rows(400, 7.0, 63, C.DEFAULT_SEED)
    d1, d6 = rows[0], rows[-1]
    assert (d1["in_sample"], d1["fresh"]) == pytest.approx((49.04, 51.82), abs=0.05)
    assert (d6["in_sample"], d6["fresh"]) == pytest.approx((39.07, 60.17), abs=0.05)
    assert all(r["true"] == pytest.approx(49.0) for r in rows)
    in_sample = [r["in_sample"] for r in rows]
    fresh = [r["fresh"] for r in rows]
    assert all(in_sample[i] >= in_sample[i + 1] for i in range(len(in_sample) - 1))         # In-Sample sinkt monoton mit der Tiefe
    assert all(fresh[i] <= fresh[i + 1] for i in range(len(fresh) - 1))                     # Frischer Rest wächst monoton mit der Tiefe
    assert in_sample[-1] < 49.0 < fresh[-1]                                                 # beide Enden liegen jenseits der wahren Varianz


# --- Leckage der Kodierung --------------------------------------------------------------------------------------------------------------------------

def test_encoding_leakage_experiment_numbers():
    lk = ev.encoding_leakage_rows(1200, 24)
    assert (lk["naive_train"], lk["test"], lk["ordered_train"]) == pytest.approx((0.0559, 0.0342, 0.0102), abs=0.005)
    assert lk["naive_train"] - lk["test"] > lk["ordered_train"] - lk["test"]                # naive leckt klar mehr als geordnet


# --- Wirkung der Kardinalität ---------------------------------------------------------------------------------------------------------------------------

def test_cardinality_experiment_numbers():
    rows = ev.cardinality_rows(1200)
    r4, r48 = rows[0], rows[-1]
    assert r4["n_depots"] == 4 and r48["n_depots"] == 48
    assert r4["naive_gap"] == pytest.approx(-0.002, abs=0.01)
    assert r48["naive_gap"] == pytest.approx(0.0849, abs=0.01)
    assert r48["naive_gap"] > r4["naive_gap"]                                               # die Lücke wächst in der Tendenz mit der Kardinalität
    assert all(abs(r["ordered_gap"]) < 0.03 for r in rows)                                  # geordnet bleibt über alle Stufenzahlen nah bei 0


# --- Erzeuger --------------------------------------------------------------------------------------------------------------------------------------

def test_generator_has_depot_and_matches_cart_demo_conventions():
    ds = S.generate_dataset(500, 3, 0, 7)
    assert ds.X.shape == (500, 12) and ds.names[:2] == ("Distanz", "Ladegewicht") and ds.names[8] == "Depot"
    Xtr, ytr, Xte, yte = S.split(ds, "class")
    assert len(Xtr) == 350 and len(Xte) == 150
