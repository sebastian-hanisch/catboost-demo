"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Aufgaben- und Kodierungswechsel, Abspielen, Permalink, alle drei Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cb_constants as C
from cb_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _play(at):
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()


def test_default_renders_without_exception_and_gives_one_verdict():
    at = _run()
    assert any("CatBoost in Aktion" in m.value for m in at.markdown) and not at.error


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert not at.error and not at.exception


def test_a_single_step_shows_no_play_button_and_says_so():
    at = _run(lambda a: _apply(a, C.PRESETS["🪓 Ein Schritt (kein Boosting)"]))
    assert not any(b.label == "▶️ Abspielen" for b in at.button)
    assert any("Nur eine Runde eingestellt" in i.value for i in at.info)


def test_switching_task_hides_label_noise_for_regression():
    at = _run()
    labels = {w.label for w in at.sidebar.slider}
    assert "Falsche Etiketten im Training [%]" in labels
    at.session_state["task_select"] = "reg"
    at.run()
    assert not at.exception
    labels = {w.label for w in at.sidebar.slider}
    assert "Falsche Etiketten im Training [%]" not in labels


def test_unchecking_ordered_boosting_hides_the_blocks_slider():
    at = _run()
    labels = {w.label for w in at.sidebar.slider}
    assert "Stützblöcke" in labels
    at.session_state["ordered_checkbox"] = False
    at.run()
    assert not at.exception
    labels = {w.label for w in at.sidebar.slider}
    assert "Stützblöcke" not in labels


def test_a_kept_slider_survives_a_round_trip_through_the_other_task():
    at = _run()
    at.session_state["label_noise_slider"] = 12
    at.run()
    at.session_state["task_select"] = "reg"
    at.run()
    at.session_state["task_select"] = "class"
    at.run()
    assert not at.exception and at.slider(key="label_noise_slider").value == 12


def test_extreme_settings_render():
    def small(at):
        at.session_state["n_slider"] = C.N_MIN
        at.session_state["depth_slider"] = C.DEPTH_MIN
        at.session_state["n_rounds_slider"] = C.N_ROUNDS_MIN
        at.session_state["n_noise_slider"] = 0
        at.session_state["n_depots_slider"] = 2

    def big(at):
        at.session_state["task_select"] = "reg"
        at.session_state["encoding_select"] = "onehot"
        at.session_state["n_slider"] = C.N_MAX
        at.session_state["depth_slider"] = C.DEPTH_MAX
        at.session_state["n_rounds_slider"] = 40
        at.session_state["n_noise_slider"] = C.NOISE_MAX
        at.session_state["n_depots_slider"] = 48
    for setup in (small, big):
        at = _run(setup)
        assert not at.exception


def test_step_slider_returns_to_the_last_round_when_settings_change():
    at = _run()
    at.slider(key="cb_step").set_value(5)
    at.run()
    assert at.slider(key="cb_step").value == 5
    at.session_state["n_rounds_slider"] = 20
    at.run()
    assert not at.exception and at.slider(key="cb_step").value == 20


def test_every_round_of_a_small_ensemble_renders():
    at = _run(lambda a: a.session_state.__setitem__("n_rounds_slider", 4))
    for k in range(1, int(at.slider(key="cb_step").max) + 1):
        at.slider(key="cb_step").set_value(k)
        at.run()
        assert not at.exception, k


def test_play_renders_several_frames_without_duplicate_keys(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    at = _run(lambda a: a.session_state.__setitem__("n_rounds_slider", 6))
    _play(at)
    assert not at.exception, [e.value for e in at.exception]


def test_permalink_parameters_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["nr"] = "99999"
    at.query_params["depth"] = "-4"
    at.run()
    assert not at.exception
    assert at.slider(key="n_rounds_slider").value == C.N_ROUNDS_MAX and at.slider(key="depth_slider").value == C.DEPTH_MIN


def test_the_address_bar_mirrors_the_settings():
    at = _run(lambda a: _apply(a, C.PRESETS["🎯 Naive Kodierung"]))
    assert str(at.query_params["enc"]) in ("naive", "['naive']")


def test_prediction_shift_experiment_runs_on_demand():
    at = _run()
    assert not any("genau die Verzerrung" in c.value for c in at.caption)
    at.button(key="shift_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "genau die Verzerrung" in text


def test_leakage_experiment_runs_on_demand():
    at = _run()
    at.button(key="leak_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "nie das eigene Etikett" in text


def test_cardinality_experiment_runs_on_demand():
    at = _run()
    at.button(key="card_start").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(c.value for c in at.caption)
    assert "je weniger Zeilen" in text


def _calls(src, name):
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert sorted(set(keys)) == sorted(["tree_chart", "round_chart", "importance_chart", "shift_chart", "leakage_chart", "cardinality_chart"]), keys
    looped = [c for c in calls if 'key=f"' in c]
    assert len(looped) == 1 and '_{current}"' in looped[0]
    viz = (ROOT / "cb_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("lock_axes(fig") >= 5


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))
