"""CatBoost - Prediction Shift, geordnetes Boosting und geordnete Target Statistics - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - CatBoost - und lässt stattdessen das Beispiel wachsen.
Neuntes und letztes Stück der Baumbasierten Linie der "Konzepte"-Reihe, fünftes Stück des Boosting-Asts (nach AdaBoost, Gradient Boosting, XGBoost, LightGBM): der Kern dieses Stücks ist
nicht eine neue Gain-Formel, sondern eine neue FRAGE - woher kommt der Gradient, den ein Baum lernt, und woher die Kodierung eines kategorischen Merkmals? Beide können vom selben Beispiel
"lecken", das sie gerade vorhersagen sollen (Prediction Shift). CatBoosts Antwort: geordnetes Boosting und geordnete Target Statistics - nie mit dem eigenen Etikett einer Zeile rechnen.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import cb_constants as C
import cb_encoding as EN
import cb_evaluation as ev
from cb_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cb_visualization import (
    build_cardinality_chart,
    build_importance,
    build_leakage_chart,
    build_prediction_shift_chart,
    build_round_curve,
    build_symmetric_tree,
)

st.set_page_config(page_title="CatBoost – Sebastian Hanisch", layout="wide")

VERDICT_TEXT = {
    "stump": "ℹ️ **Nur ein Schritt eingestellt** - das ist die Vorhersage eines einzelnen (mit Lernrate skalierten) Baums, noch kein Boosting.",
    "overfit": "⚠️ **Überanpassung:** der Testfehler liegt deutlich über dem Trainingsfehler.",
    "underfit": "⚠️ **Unteranpassung:** kaum besser als Raten - mehr Runden oder größere Lernrate könnten helfen.",
    "ok": "✅ **Sieht vernünftig aus:** Training und Test liegen nicht weit auseinander.",
}


def _err(task, x):
    return f"{x:.1%}" if task == "class" else f"{x:.1f} min"


@st.cache_resource(show_spinner=False, max_entries=24)
def _analysis(*params):
    return ev.analyse(*params)


@st.cache_data(show_spinner=False, max_entries=8)
def _round_rows(*params):
    a = _analysis(*params)
    return ev.round_rows(a)


@st.cache_data(show_spinner=False, max_entries=4)
def _prediction_shift(n, sigma, max_bin, seed):
    return ev.prediction_shift_rows(n, sigma, max_bin, seed)


@st.cache_data(show_spinner=False, max_entries=4)
def _leakage(n, n_depots):
    return ev.encoding_leakage_rows(n, n_depots)


@st.cache_data(show_spinner=False, max_entries=4)
def _cardinality(n):
    return ev.cardinality_rows(n)


st.title("🐈🌳 CatBoost – geordnetes Boosting gegen Prediction Shift")
st.markdown(
    """
Jedes bisherige Stück dieser Linie hat eine neue Split-Suche oder Regularisierung gezeigt - CatBoost (Prokhorenkova et al. 2018) stellt eine andere Frage: **woher kommt der Gradient, den
ein Baum lernt?** In gewöhnlichem Boosting stammt der Pseudo-Gradient einer Zeile von einem Modell, das diese Zeile SCHON GESEHEN hat - ihr Rest wirkt dann systematisch zu klein
(**Prediction Shift**). Dieselbe Falle gibt es bei kategorischen Merkmalen: ein naives Target Encoding rechnet das eigene Etikett der Zeile in seine eigene Kodierung ein.
CatBoosts Antwort auf beides: **nie mit dem eigenen Etikett einer Zeile rechnen** - geordnetes Boosting (eine zufällige Permutation, nur Zeilen davor zählen) und geordnete Target Statistics
(dieselbe Idee für Merkmale). Dazu: **vollständig symmetrische Bäume** - jede Ebene ein einziger Split für alle Knoten gleichzeitig.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - neuntes und LETZTES Stück der Baumbasierten Linie der \"Konzepte\"-Reihe "
    "und fünftes Stück des **Boosting-Asts** (nach AdaBoost, Gradient Boosting, XGBoost, LightGBM) - **ein** Verfahren an einem wachsenden Beispiel. Das Verfahren geht auf Prokhorenkova et al. "
    "(2018) zurück; alle Lieferungen, Merkmale und Zahlen dieser Demo sind erzeugt und gemessen - keine echten Daten. Neu ab diesem Stück: **Depot** (24 Stufen, kategorisch) als zweites "
    "kategorisches Merkmal neben Wochentag. Baumkern und Boosting-Schleife (`cb_tree.py`, `cb_algorithm.py`) sind neu geschrieben; die echte `catboost`-Bibliothek kommt nur in den Tests als "
    "Gegenprobe vor (über Rang-/Fehlergrenzen, nicht exakt)."
)
st.caption(
    "**Bezug zu OR:** ein Depot-Effekt, der nicht an einzelnen Lieferungen hängen bleibt (keine Leckage), ist eine ehrlichere Grundlage für die Standortplanung - eine Kodierung, die nur "
    "auf wenigen Lieferungen je Depot beruht, würde sonst zufällige Schwankungen für ein echtes Standortmerkmal halten."
)

with st.expander("So funktioniert CatBoost", expanded=True):
    st.markdown(
        r"""
1. **Prediction Shift:** ein Baum, der auf denselben Zeilen wächst, die er dann vorhersagt, sieht optimistischer aus, als er ist - sein Rest gegen die eigenen Trainingszeilen ist
   systematisch zu klein (unten mit reinem Rauschen nachgewiesen, ohne jedes echte Signal).
2. **Geordnetes Boosting:** eine zufällige Permutation der Zeilen; der Baum, der eine Zeile aktualisiert, wird nur mit Zeilen trainiert, die in der Permutation VOR ihr liegen - nie mit der
   Zeile selbst oder etwas danach.
3. **Geordnete Target Statistics:** dieselbe Idee für ein kategorisches Merkmal (Wochentag, Depot) - die Kodierung einer Zeile nutzt nur den Ziel-Mittelwert der Zeilen derselben Kategorie, die
   VOR ihr in der Permutation liegen, nie ihr eigenes Etikett.
4. **Symmetrische Bäume:** jede Ebene ein einziger Split (Merkmal + Schwelle) für alle Knoten dieser Ebene gleichzeitig - ein Baum der Tiefe $d$ hat immer genau $2^d$ Blätter.
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    task = st.selectbox("Aufgabe", C.TASKS, key="task_select", format_func=lambda k: C.TASK_LABELS[k])
    encoding = st.selectbox("Kodierung (Wochentag, Depot)", C.ENCODING_OPTIONS, key="encoding_select", format_func=lambda k: C.ENCODING_LABELS[k])
    ordered = st.checkbox("Geordnetes Boosting", key="ordered_checkbox", help="Aus = gewöhnliches (verzerrtes) Boosting, wie in gradient-boosting-demo/xgboost-demo/lightgbm-demo.")
    depth = st.slider("Tiefe (symmetrisch: 2^Tiefe Blätter)", *bounds("depth_slider"), key="depth_slider")
    n_rounds = st.slider("Zahl der Runden", *bounds("n_rounds_slider"), key="n_rounds_slider")
    lr = st.slider("Lernrate", *bounds("lr_slider"), key="lr_slider", step=0.01, format="%.2f")
    lam = st.slider("λ (L2 auf Blattgewichte)", *bounds("lam_slider"), key="lam_slider", step=0.5, format="%.1f",
                    help="Geordnetes Boosting wächst jeden Baum auf einem Bruchteil der Zeilen - ohne genug λ (und mit hoher Lernrate/Tiefe/Rundenzahl) kann das bei Regression zunehmend instabil werden (siehe README).")
    if ordered:
        n_blocks = st.slider("Stützblöcke", *bounds("n_blocks_slider"), key="n_blocks_slider", help="Nähert CatBoosts O(log n) Stützmodelle an - mehr Blöcke = feinere Annäherung, mehr Bäume je Runde.")
    else:
        n_blocks = C.DEFAULT_N_BLOCKS
        st.caption("Stützblöcke gelten nur für geordnetes Boosting.")
    n_depots = st.slider("Depot-Stufen", *bounds("n_depots_slider"), key="n_depots_slider")
    st.markdown("**Daten**")
    n = st.slider("Lieferungen", *bounds("n_slider"), key="n_slider", step=100)
    n_noise = st.slider("Rauschmerkmale", *bounds("n_noise_slider"), key="n_noise_slider")
    if task == "class":
        label_noise = st.slider("Falsche Etiketten im Training [%]", *bounds("label_noise_slider"), key="label_noise_slider")
        st.session_state["_label_noise_kept"] = label_noise
    else:
        label_noise = int(st.session_state.get("_label_noise_kept", C.DEFAULT_LABEL_NOISE))
        st.caption("Falsche Etiketten gibt es nur bei der Klassifikation.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten generieren", width="stretch", on_click=randomize_seed)

base_params = (task, encoding, int(depth), float(lam), 63, int(n_rounds), float(lr), bool(ordered), int(n_blocks))
data_params = (int(n), int(n_noise), int(label_noise), int(n_depots), int(seed))
with st.spinner("Rechne ..."):
    a = _analysis(*base_params, *data_params)
ds = a.ds
names = ds.names
n_rounds_actual = len(a.ensemble.trees)
sync_query_params({"task_select": task, "encoding_select": encoding, "ordered_checkbox": bool(ordered), "depth_slider": int(depth), "n_rounds_slider": int(n_rounds), "lr_slider": float(lr),
                   "lam_slider": float(lam), "n_blocks_slider": int(n_blocks), "n_depots_slider": int(n_depots), "n_slider": int(n), "n_noise_slider": int(n_noise),
                   "label_noise_slider": int(label_noise), "seed_input": int(seed)})

view_key = (base_params, data_params)
if st.session_state.get("cb_owner") != view_key:
    st.session_state["cb_owner"] = view_key
    st.session_state["cb_step"] = n_rounds_actual

# --- CatBoost in Aktion -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🐈 CatBoost in Aktion")
st.caption("Runde für Runde: der symmetrische Baum dieser Runde - jede Ebene (Raute) ist EIN Split für alle Zweige gleichzeitig, die Blätter darunter tragen die Newton-Gewichte.")
if n_rounds_actual > 1:
    step_col, play_col = st.columns([5, 2])
    with step_col:
        step = st.slider("Runden", 1, n_rounds_actual, key="cb_step")
    with play_col:
        auto_play = st.button("▶️ Abspielen", width="stretch")
else:
    step, auto_play = 1, False
    st.info("ℹ️ Nur eine Runde eingestellt - mehr Runden in der Seitenleiste zeigen den Effekt.")
view_slot = st.empty()


def _round_tree(current):
    if a.ordered:
        round_trees = a.ensemble.trees[current - 1]
        return round_trees[max(round_trees)]
    return a.ensemble.trees[current - 1]


def _render(current):
    tree = _round_tree(current)
    with view_slot.container():
        st.plotly_chart(build_symmetric_tree(tree, names), width="stretch", key=f"tree_chart_{current}")
        st.caption(f"Runde {current}: {tree.n_leaves} Blätter (Tiefe {tree.depth}).")


if auto_play:
    frames = sorted(set(np.unique(np.round(np.linspace(1, n_rounds_actual, min(12, n_rounds_actual))).astype(int))))
    for kk in frames:
        _render(kk)
        time.sleep(min(0.9, 6.0 / len(frames)))
    step = n_rounds_actual
else:
    _render(step)

st.markdown("---")

# --- Was das Ensemble gelernt hat -------------------------------------------------------------------------------------------------------------------

st.markdown("## 📐 Was das Ensemble gelernt hat – und wie gut es auf neuen Lieferungen ist")
rrows = _round_rows(*base_params, *data_params)
best = ev.best_round(rrows)
m1, m2, m3 = st.columns(3)
m1.metric("Runden", n_rounds_actual)
m2.metric("Trainingsfehler" if task == "class" else "Trainings-RMSE", _err(task, ev.primary(a.train, task)))
m3.metric("Testfehler" if task == "class" else "Test-RMSE", _err(task, ev.primary(a.test, task)), delta=f"ohne Modell: {_err(task, a.baseline)}", delta_color="off")
st.markdown(VERDICT_TEXT[a.verdict])

st.markdown("**Testfehler gegen die Rundenzahl**")
st.plotly_chart(build_round_curve(rrows, task, a.baseline, n_rounds_actual, best["k"]), width="stretch", key="round_chart")
st.caption(f"Bester Testfehler bei Runde {best['k']} ({_err(task, best['test'])}).")

st.markdown("**Wichtigkeit je Merkmal**")
labels = EN.encoded_names(names, C.CAT_FEATURES, a.encoders)
st.plotly_chart(build_importance(labels, a.imp), width="stretch", key="importance_chart")
st.caption("Gemittelt über alle Bäume des Ensembles (bei geordnetem Boosting: der jeweils vollständigste Baum je Runde). Summe der Split-Gains je Merkmal, auf 1 normiert.")

st.markdown("---")

# --- Experimente -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Prediction Shift: in-sample gegen frische Werte")
if st.button("Reste gegen die Tiefe messen - reines Rauschen, kein echtes Signal (dauert einen Moment)", key="shift_start"):
    st.session_state["shift_on"] = True
if st.session_state.get("shift_on"):
    with st.spinner("Wachse Bäume auf reinem Rauschen und vergleiche gegen frisch gezogene Werte ..."):
        psr = _prediction_shift(400, 7.0, 63, int(seed))
    st.plotly_chart(build_prediction_shift_chart(psr), width="stretch", key="shift_chart")
    d1, d6 = psr[0], psr[-1]
    st.caption(f"400 Zeilen, reines Rauschen (σ² = {d1['true']:.0f}), kein Merkmal hat irgendeinen Bezug zum Ziel. Bei Tiefe 1 liegen beide Reste nah an der wahren Varianz "
               f"({d1['in_sample']:.1f} bzw. {d1['fresh']:.1f}). Mit wachsender Tiefe fällt der In-Sample-Rest unter die wahre Varianz ({d6['in_sample']:.1f} bei Tiefe 6 - der Baum sieht "
               f"immer besser aus, obwohl er nichts lernt), während der Rest gegen frische Werte darüber steigt ({d6['fresh']:.1f}) - genau die Verzerrung, die geordnetes Boosting vermeidet: "
               "den Gradienten nie von einem Modell holen, das die Zeile schon gesehen hat.")

st.markdown("---")

st.subheader("🔬 Leckage von Target Encoding gegen geordnete Target Statistics")
if st.button("Korrelation mit dem eigenen Ziel messen, Training gegen ehrlichen Test (dauert einen Moment)", key="leak_start"):
    st.session_state["leak_on"] = True
if st.session_state.get("leak_on"):
    with st.spinner("Kodiere Depot mit beiden Verfahren auf fünf Datensätzen und vergleiche Trainings- gegen Testkorrelation ..."):
        lk = _leakage(int(n), int(n_depots))
    st.plotly_chart(build_leakage_chart(lk), width="stretch", key="leakage_chart")
    st.caption(f"R² zwischen der Depot-Kodierung und der Lieferdauer, Mittel über fünf Datensätze: naiv sieht im Training ({lk['naive_train']:.1%}) deutlich besser aus als im ehrlichen Test "
               f"({lk['test']:.1%}) - das eigene Etikett jeder Zeile fließt in ihre eigene Kodierung ein. Geordnete Target Statistics liegt im Training ({lk['ordered_train']:.1%}) schon nah am "
               "ehrlichen Wert - sie hat nie das eigene Etikett gesehen.")

st.markdown("---")

st.subheader("🔬 Wirkung der Kardinalität")
if st.button("Die Trainings-Test-Lücke gegen die Zahl der Depot-Stufen messen (dauert einen Moment)", key="card_start"):
    st.session_state["card_on"] = True
if st.session_state.get("card_on"):
    with st.spinner("Wiederholt die Kodierung für verschiedene Depot-Stufenzahlen, je fünf Datensätze ..."):
        cr = _cardinality(int(n))
    st.plotly_chart(build_cardinality_chart(cr), width="stretch", key="cardinality_chart")
    c0, c1 = cr[0], cr[-1]
    st.caption(f"Bei {c0['n_depots']} Depots ({c0['rows_per_depot']} Zeilen je Depot im Training) ist die Lücke von naivem Target Encoding klein ({c0['naive_gap']:+.1%}); bei {c1['n_depots']} "
               f"Depots (nur noch {c1['rows_per_depot']} Zeilen je Depot) wächst sie auf {c1['naive_gap']:+.1%} - je weniger Zeilen eine Kategorie hat, desto mehr trägt eine einzelne Zeile "
               "zu ihrem eigenen naiven Target Encoding bei. Die geordnete Target Statistics bleibt über alle Stufenzahlen nah bei 0.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Geordnetes Boosting gewinnt immer gegen gewöhnliches** | In dieser vereinfachten, block-genäherten Fassung (eine feste Permutation statt mehrerer, feste statt geometrisch wachsender Blöcke) schneidet geordnetes Boosting in diesem Datensatz nicht besser ab als gewöhnliches - der Preis, weniger Zeilen je Baum zu sehen, überwiegt den Gain aus weniger Verzerrung (gemessen, siehe README). | mehrere Permutationen mitteln (echtes CatBoost), größere Datensätze |
| **Beliebig viele Runden bei Regression sind sicher** | Wenige Zeilen je Block plus hohe Tiefe/Lernrate/Rundenzahl können einzelne Zeilenvorhersagen zunehmend instabil werden lassen (gefunden und mit stärkerem λ/kleinerer Lernrate eingedämmt, nicht vollständig beseitigt). | mehr λ, kleinere Lernrate, weniger Runden, mehr Stützblöcke |
| **Kodierungs-Leckage schlägt immer auf den Testfehler durch** | Die Leckage ist in der Kodierung selbst klar messbar (Korrelations-Lücke oben), muss sich aber nicht automatisch in einem schlechteren End-zu-End-Testfehler zeigen, wenn das zugrunde liegende Signal echt und zwischen Training und Test konsistent ist. | die Kodierung selbst prüfen, nicht nur den Endfehler |
| **Symmetrische Bäume sind immer so gut wie unregelmäßige** | Ein Split gilt für ALLE Knoten einer Ebene - ein Merkmal, das nur in einem Teil des Datenraums hilft, kann trotzdem gewählt werden und dort schadet, wo es nicht passt. | Tiefe klein halten, Wichtigkeit prüfen |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Symmetrischer Baum.** Ebene $\ell=1,\dots,d$: EIN Split $(f_\ell,\theta_\ell)$ für alle aktuellen Gruppen $g$ gleichzeitig, gewählt um $\sum_g \text{Gain}_g(f,\theta)$ zu maximieren
(dieselbe Gain-Formel wie xgboost-demo/lightgbm-demo, summiert über Gruppen). Ergebnis: $2^d$ Blätter, Blattwert $-G_j/(H_j+\lambda)$ wie üblich.

**Geordnetes Boosting** (vereinfacht: `n_blocks` feste Blöcke statt CatBoosts $O(\log n)$ geometrisch wachsender Stützmodelle einer zufälligen Permutation). Block $k$'s Baum trainiert NUR
mit Gradienten aus Block $0,\dots,k-1$: $g_i,h_i = \partial l(y_i,F_i)/\partial F_i,\ \partial^2 l/\partial F_i^2$ für $i\in\text{Block}_{<k}$; seine Vorhersage aktualisiert NUR Zeilen aus
Block $k$. Block 0 hat keinen Vorgänger und bekommt einen eigenen, gewöhnlichen (selbstbezogenen) Baum, sonst bliebe seine Vorhersage für immer beim Startwert stehen.

**Geordnete Target Statistics** für eine Kategorie $c$: $\text{TS}_i = \dfrac{\sum_{j\prec i,\,\text{cat}_j=c} y_j + a\cdot p}{|\{j\prec i: \text{cat}_j=c\}| + a}$ mit $\prec$ = "liegt vor $i$
in der Permutation", $p$ = globaler Mittelwert, $a$ = Gewicht des Prior - $y_i$ selbst kommt in der Summe nie vor.

**Prediction Shift:** für eine Zeile $i$, deren Blattwert (oder TS) UNTER Einbeziehung von $y_i$ berechnet wurde, ist $\mathbb E[\hat y_i - y_i \mid \text{Blatt}]$ systematisch verzerrt -
nachgewiesen oben, indem ein Baum auf reinem Rauschen wächst und sein Rest gegen die eigenen (verzerrt) gegen frisch gezogene (unverzerrt) Werte verglichen wird.

Implementiert in `cb_tree.py` (symmetrischer Baumkern), `cb_algorithm.py` (gewöhnliches und geordnetes Boosting), `cb_encoding.py` (naive/geordnete/One-Hot-Kodierung),
`cb_evaluation.py` (Analyse, Prediction-Shift-, Leckage- und Kardinalitäts-Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
