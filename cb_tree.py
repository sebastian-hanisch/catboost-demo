"""Der CatBoost-Baumkern: **vollständig symmetrische (oblivious) Bäume** - anders als jeder Baum in dieser Linie (unregelmäßige, individuell gewachsene Knoten) benutzt JEDE Ebene EINEN einzigen
Split (Merkmal + Schwelle) für ALLE Knoten dieser Ebene gleichzeitig. Ein Baum der Tiefe `depth` hat deshalb immer genau `2**depth` Blätter, und lässt sich als `depth` Splits (statt eines
Baums aus Knoten) darstellen - Vorhersage ist ein paar Vergleiche, kein Baumdurchlauf. Die Split-Suche wiederverwendet die Histogramm-Bausteine aus lightgbm-demo (Bins einmal berechnen, je
Gruppe ein Histogramm), sucht aber je Ebene den Split, der den GAIN ÜBER ALLE AKTUELLEN GRUPPEN SUMMIERT maximiert - eine echt neue Split-Suche, kein wiederverwendeter Kern."""

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class Tree:
    features: np.ndarray         # (depth,) Merkmal je Ebene
    thresholds: np.ndarray       # (depth,) Schwelle je Ebene
    values: np.ndarray           # (2**depth,) Blattwert je Blattindex
    gains: np.ndarray            # (depth,) über alle Gruppen summierter Gain je Ebene
    depth: int
    lam: float
    n_features: int

    @property
    def n_leaves(self):
        return len(self.values)


# --- Bins (wortgleiches Vorgehen wie lightgbm-demo) --------------------------------------------------------------------------------------------------

def build_bin_edges(X, max_bin):
    edges = []
    for f in range(X.shape[1]):
        q = np.linspace(0.0, 1.0, max_bin + 1)[1:-1]
        e = np.unique(np.quantile(X[:, f], q))
        edges.append(e)
    return edges


def digitize(X, edges):
    m, d = X.shape
    bins = np.empty((m, d), dtype=np.int32)
    for f in range(d):
        bins[:, f] = np.searchsorted(edges[f], X[:, f], side="left")
    return bins


def histogram(bins_f, grad, hess, n_bins):
    g = np.bincount(bins_f, weights=grad, minlength=n_bins)
    h = np.bincount(bins_f, weights=hess, minlength=n_bins)
    return g, h


# --- Symmetrisches Wachsen: EIN Split je Ebene, über alle Gruppen hinweg ----------------------------------------------------------------------------

def grow(X, grad, hess, edges, depth, lam=1.0):
    """Wächst `depth` Ebenen; jede Ebene wählt den (Merkmal, Schwelle), der den Gain summiert über ALLE aktuellen Gruppen maximiert, und wendet ihn auf jede Gruppe gleichzeitig an.
    Endet immer mit genau `2**depth` Blättern (keine Vorwärts-Beschneidung wie gamma/min_child_weight in xgboost-demo/lightgbm-demo - das ist hier bewusst nicht der Untersuchungsgegenstand)."""
    X = np.asarray(X, dtype=float)
    grad = np.asarray(grad, dtype=float)
    hess = np.asarray(hess, dtype=float)
    n, d = X.shape
    bins = digitize(X, edges)
    n_bins = [len(e) + 1 for e in edges]
    group = np.zeros(n, dtype=np.int32)

    chosen_f, chosen_thr, chosen_gain = [], [], []
    for level in range(depth):
        n_groups = 1 << level
        group_idx = [np.nonzero(group == g)[0] for g in range(n_groups)]
        best = None                                                          # (total_gain, feature, bin_k, threshold)
        for f in range(d):
            n_k = n_bins[f] - 1
            if n_k < 1:
                continue
            total_gain = np.zeros(n_k)
            for idx_g in group_idx:
                if len(idx_g) < 2:
                    continue
                gg, hh = histogram(bins[idx_g, f], grad[idx_g], hess[idx_g], n_bins[f])
                cg, ch = np.cumsum(gg)[:-1], np.cumsum(hh)[:-1]
                Gtot, Htot = float(gg.sum()), float(hh.sum())
                Gr, Hr = Gtot - cg, Htot - ch
                ok = (ch > EPS) & (Hr > EPS)                                  # beide Kinder brauchen echtes Hesse-Gewicht in DIESER Gruppe, sonst 0/0 bzw. x/0
                with np.errstate(divide="ignore", invalid="ignore"):
                    gain_g = 0.5 * (cg ** 2 / (ch + lam) + Gr ** 2 / (Hr + lam) - Gtot ** 2 / (Htot + lam))
                total_gain += np.where(ok, gain_g, 0.0)                       # diese Schwelle trennt in dieser Gruppe nichts - trägt 0 zur Summe bei, disqualifiziert die Schwelle nicht insgesamt
            k = int(total_gain.argmax())
            if best is None or total_gain[k] > best[0]:
                best = (float(total_gain[k]), f, k)
        _, f, k = best
        thr = float(edges[f][k])
        chosen_f.append(f), chosen_thr.append(thr), chosen_gain.append(best[0])
        group = group * 2 + (X[:, f] > thr).astype(np.int32)

    n_leaves = 1 << depth
    g_sum = np.bincount(group, weights=grad, minlength=n_leaves)
    h_sum = np.bincount(group, weights=hess, minlength=n_leaves)
    values = -g_sum / (h_sum + lam + EPS)                                    # +EPS: eine leere Gruppe (kann bei sehr feiner Tiefe/wenig Daten vorkommen) bekommt Blattwert 0 statt NaN
    return Tree(np.array(chosen_f), np.array(chosen_thr), values, np.array(chosen_gain), depth, lam, d)


# --- Anwenden --------------------------------------------------------------------------------------------------------------------------------------

def leaf_index(tree, X):
    X = np.asarray(X, dtype=float)
    idx = np.zeros(len(X), dtype=np.int64)
    for d in range(tree.depth):
        f, thr = tree.features[d], tree.thresholds[d]
        idx = idx * 2 + (X[:, f] > thr).astype(np.int64)
    return idx


def predict_value(tree, X):
    return tree.values[leaf_index(tree, X)]


def importances(tree):
    imp = np.zeros(tree.n_features)
    for d in range(tree.depth):
        imp[tree.features[d]] += tree.gains[d]
    s = imp.sum()
    return imp / s if s > 0 else imp
