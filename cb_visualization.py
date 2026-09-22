"""Plotly-Darstellungen: der symmetrische Baum einer Runde (jede Ebene EIN Split für alle Knoten), Fehlerkurve gegen Runden, Prediction-Shift-Verzerrung gegen die Tiefe, Leckage-Vergleich
(naiv gegen geordnet), Wirkung der Kardinalität, Wichtigkeit. Alle Achsen sind gesperrt (Touch-Scrollen).

Kein 2D-Entscheidungsgrenzen-Karte wie in den Vorgänger-Stücken: die kodierten Merkmale haben je nach Kodierung unterschiedlich viele Spalten (One-Hot spreizt eine Kategorie in viele Spalten
auf) - eine Karte über zwei feste Merkmal-Indizes würde bei jedem Kodierungswechsel etwas anderes bedeuten. Der symmetrische Baum selbst zeigt das Besondere dieses Stücks besser."""

import numpy as np
import plotly.graph_objects as go

import cb_constants as C

CLASS_SCALE = [[0.0, "#2ca02c"], [0.5, "#f2e394"], [1.0, "#d62728"]]


def lock_axes(fig, height=None, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=height, dragmode=False, **layout)
    return fig


def feature_label(names, f):
    unit = dict(C.FEATURES).get(names[f], "") if f < len(names) else ""
    return f"{names[f]} [{unit}]" if unit else (names[f] if f < len(names) else f"Merkmal {f}")


# --- Symmetrischer Baum ------------------------------------------------------------------------------------------------------------------------------

def build_symmetric_tree(tree, names, height=340):
    """Jede Ebene ist EIN Knoten mit EINEM Split (gilt für alle Zweige gleichzeitig) - gezeichnet als ein Diamant je Ebene, darunter die `2**depth` Blätter in fester Reihenfolge."""
    depth = tree.depth
    n_leaves = tree.n_leaves
    fig = go.Figure()
    # Ebenen-Knoten (ein Diamant je Ebene, mittig)
    level_x = [n_leaves / 2.0] * depth
    level_y = [-float(d) for d in range(depth)]
    labels = [f"{feature_label(names, int(tree.features[d]))} > {tree.thresholds[d]:.3g}?" for d in range(depth)]
    # Verbindungslinien: jede Ebene mit der nächsten, und die letzte Ebene mit jedem Blatt
    ex, ey = [], []
    for d in range(depth - 1):
        ex += [level_x[d], level_x[d + 1], None]
        ey += [level_y[d], level_y[d + 1], None]
    leaf_x = list(range(n_leaves))
    leaf_y = [-float(depth)] * n_leaves
    if depth > 0:
        for lx in leaf_x:
            ex += [level_x[-1], lx, None]
            ey += [level_y[-1], -float(depth), None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#9aa0a6", width=1), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=level_x, y=level_y, mode="markers+text", text=labels, textposition="middle right", textfont=dict(size=10),
                             marker=dict(symbol="diamond", size=20, color="#ffffff", line=dict(color="#555555", width=1.5)), hoverinfo="skip", showlegend=False))
    vmax = float(np.max(np.abs(tree.values))) if n_leaves else 1.0
    fig.add_trace(go.Scatter(x=leaf_x, y=leaf_y, mode="markers+text", text=[f"{v:+.2g}" for v in tree.values], textposition="bottom center", textfont=dict(size=9),
                             marker=dict(size=16, color=tree.values, colorscale="RdBu_r", cmin=-vmax, cmax=vmax, line=dict(color="#111111", width=1)),
                             hovertext=[f"Blatt {i}: Gewicht {v:+.4g}" for i, v in enumerate(tree.values)], hoverinfo="text", showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return lock_axes(fig, height, plot_bgcolor="rgba(0,0,0,0)")


# --- Fehlerkurve gegen Runden --------------------------------------------------------------------------------------------------------------------------

def _error_axis(fig, task):
    fig.update_yaxes(title="Fehlerquote" if task == "class" else "RMSE [min]", rangemode="tozero", **({"tickformat": ".0%"} if task == "class" else {}))


def build_round_curve(rows, task, baseline, current_k, best_k=None, height=340):
    k = [r["k"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=k, y=[r["train"] for r in rows], mode="lines+markers", name="Training", line=dict(color=C.COLORS["train"])))
    fig.add_trace(go.Scatter(x=k, y=[r["test"] for r in rows], mode="lines+markers", name="Test", line=dict(color=C.COLORS["test"])))
    fig.add_hline(y=baseline, line=dict(color="#888888", dash="dash"), annotation_text="ohne Modell (Raten)", annotation_position="top right")
    fig.add_vline(x=current_k, line=dict(color="#111111", dash="dot"))
    if best_k is not None and best_k != current_k:
        fig.add_vline(x=best_k, line=dict(color="#2ca02c", dash="dot"), annotation_text="bester Testfehler", annotation_position="bottom left")
    fig.update_xaxes(title="Runden", type="log" if k[-1] > 30 else "linear")
    _error_axis(fig, task)
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Prediction Shift ----------------------------------------------------------------------------------------------------------------------------------

def build_prediction_shift_chart(rows, height=340):
    depths = [r["depth"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=depths, y=[r["in_sample"] for r in rows], mode="lines+markers", name="In-Sample-Rest", line=dict(color=C.COLORS["insample"])))
    fig.add_trace(go.Scatter(x=depths, y=[r["fresh"] for r in rows], mode="lines+markers", name="Frischer Rest", line=dict(color=C.COLORS["loo"])))
    fig.add_hline(y=rows[0]["true"], line=dict(color="#888888", dash="dash"), annotation_text="wahre Varianz σ²", annotation_position="bottom right")
    fig.update_xaxes(title="Tiefe des Baums")
    fig.update_yaxes(title="Mittlerer quadratischer Rest", rangemode="tozero")
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Leckage der Kodierung -------------------------------------------------------------------------------------------------------------------------------

def build_leakage_chart(leakage, height=320):
    labels = ["Naives Target Encoding\n(Training)", "Geordnete Target Statistics\n(Training)", "beide\n(ehrlicher Test)"]
    vals = [leakage["naive_train"], leakage["ordered_train"], leakage["test"]]
    colors = [C.COLORS["naive"], C.COLORS["ordered"], "#888888"]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colors, text=[f"{v:.1%}" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="R² mit dem Ziel", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False)


def build_cardinality_chart(rows, height=340):
    n = [r["n_depots"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=[r["naive_gap"] for r in rows], mode="lines+markers", name="Naives Target Encoding", line=dict(color=C.COLORS["naive"])))
    fig.add_trace(go.Scatter(x=n, y=[r["ordered_gap"] for r in rows], mode="lines+markers", name="Geordnete Target Statistics", line=dict(color=C.COLORS["ordered"])))
    fig.add_hline(y=0.0, line=dict(color="#888888", dash="dash"))
    fig.update_xaxes(title="Zahl der Depot-Stufen")
    fig.update_yaxes(title="R²-Lücke (Training minus ehrlicher Test)", tickformat=".0%")
    return lock_axes(fig, height, legend=dict(orientation="h", y=1.12))


# --- Wichtigkeit ---------------------------------------------------------------------------------------------------------------------------------------

def build_importance(labels, imp, height=330):
    """`labels`: Merkmalsnamen NACH der Kodierung, gleiche Reihenfolge/Länge wie `imp` (siehe `cb_encoding.encoded_names`)."""
    order = np.argsort(-imp, kind="stable")
    fig = go.Figure(go.Bar(x=imp[order], y=[labels[f] for f in order], orientation="h", marker_color="#1f77b4",
                           text=[f"{imp[f]:.1%}" for f in order], textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Wichtigkeit (Anteil am Gesamt-Gain)", tickformat=".0%", rangemode="tozero")
    return lock_axes(fig, height, showlegend=False).update_layout(margin=dict(l=10, r=60, t=30, b=10))
