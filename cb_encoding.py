"""Kodierung kategorischer Merkmale (Wochentag, Depot): geordnete Target Statistics (CatBoosts eigener Beitrag) gegen zwei Vergleichsmaßstäbe - naives Target Encoding (leckt: das eigene
Etikett der Zeile fließt in seine eigene Kodierung ein) und One-Hot (leckt nicht, aber bei vielen Stufen dünn besetzt und hochdimensional)."""

import numpy as np


def naive_target_mean(cat_values, y, prior_weight=1.0):
    """Jede Zeile bekommt den Ziel-Mittelwert ALLER Zeilen ihrer Kategorie - EINSCHLIESSLICH der eigenen Zeile. Das ist die Leckage: eine Kategorie mit einer einzigen Zeile bekommt exakt deren
    eigenes Etikett als "Vorhersage" zurück."""
    cat_values = np.asarray(cat_values).astype(int)
    y = np.asarray(y, dtype=float)
    prior = float(y.mean())
    n_cat = int(cat_values.max()) + 1
    sums = np.bincount(cat_values, weights=y, minlength=n_cat)
    counts = np.bincount(cat_values, minlength=n_cat)
    means = (sums + prior_weight * prior) / (counts + prior_weight)
    return means[cat_values]


def ordered_target_stat(cat_values, y, seed, prior_weight=1.0):
    """Jede Zeile bekommt den Ziel-Mittelwert NUR der Zeilen derselben Kategorie, die in einer zufälligen Permutation VOR ihr liegen (ihre eigene Zeile zählt nie mit) - dieselbe Idee wie
    Ordered Boosting, hier auf die Merkmalskodierung angewandt. Ein Anteil `prior_weight` des globalen Mittels wird immer mit eingerechnet (verhindert Division durch 0 für die erste Zeile
    einer Kategorie und dämpft sehr kleine Kategorien in Richtung des globalen Mittels - dieselbe Glättung wie beim naiven Target Encoding, damit der Vergleich fair bleibt)."""
    cat_values = np.asarray(cat_values).astype(int)
    y = np.asarray(y, dtype=float)
    n = len(y)
    prior = float(y.mean())
    n_cat = int(cat_values.max()) + 1
    order = np.random.default_rng(seed).permutation(n)
    running_sum = np.zeros(n_cat)
    running_count = np.zeros(n_cat)
    ts = np.empty(n)
    for pos in order:
        c = cat_values[pos]
        ts[pos] = (running_sum[c] + prior_weight * prior) / (running_count[c] + prior_weight)
        running_sum[c] += y[pos]
        running_count[c] += 1.0
    return ts


def one_hot(cat_values, n_cat=None):
    """Eine Spalte je Kategorie (1 = diese Kategorie, sonst 0) - keine Leckage, aber `n_cat` zusätzliche, meist dünn besetzte Spalten."""
    cat_values = np.asarray(cat_values).astype(int)
    n_cat = n_cat or int(cat_values.max()) + 1
    out = np.zeros((len(cat_values), n_cat))
    out[np.arange(len(cat_values)), cat_values] = 1.0
    return out


def encode_dataset(X, cat_cols, y, encoding, seed, prior_weight=1.0):
    """Ersetzt die Spalten `cat_cols` (kategorisch) durch ihre Kodierung; alle anderen Spalten bleiben unverändert. `y` und die Kodierung kommen NUR aus den TRAININGSDATEN - für neue
    (Test-)Zeilen wird dieselbe, schon berechnete Zuordnung nachgeschlagen (siehe `apply_encoding_map`), nie neu aus den Testdaten berechnet."""
    cat_cols = list(cat_cols)
    keep_cols = [f for f in range(X.shape[1]) if f not in cat_cols]
    parts = [X[:, keep_cols]]
    encoders = {}
    for f in cat_cols:
        vals = X[:, f]
        if encoding == "onehot":
            n_cat = int(vals.max()) + 1
            parts.append(one_hot(vals, n_cat))
            encoders[f] = ("onehot", n_cat)
        else:
            if encoding == "naive":
                enc = naive_target_mean(vals, y, prior_weight)
                lookup = _category_lookup(vals, y, prior_weight)
            else:
                enc = ordered_target_stat(vals, y, seed, prior_weight)
                lookup = _category_lookup(vals, y, prior_weight)               # zur Anwendung auf neue Zeilen: das VOLLE Trainingsmittel je Kategorie (die Ordnung gilt nur beim Training)
            parts.append(enc.reshape(-1, 1))
            encoders[f] = ("target", lookup)
    return np.column_stack(parts), encoders


def _category_lookup(cat_values, y, prior_weight):
    cat_values = np.asarray(cat_values).astype(int)
    y = np.asarray(y, dtype=float)
    prior = float(y.mean())
    n_cat = int(cat_values.max()) + 1
    sums = np.bincount(cat_values, weights=y, minlength=n_cat)
    counts = np.bincount(cat_values, minlength=n_cat)
    return (sums + prior_weight * prior) / (counts + prior_weight)


def encoded_names(names, cat_cols, encoders):
    """Merkmalsnamen NACH `encode_dataset`, in derselben Spaltenreihenfolge (nicht-kategorische Spalten zuerst, dann die kodierten) - für Wichtigkeits-Diagramme."""
    cat_cols = list(cat_cols)
    keep = [names[f] for f in range(len(names)) if f not in cat_cols]
    out = list(keep)
    for f in cat_cols:
        kind, payload = encoders[f]
        if kind == "onehot":
            out += [f"{names[f]} = {c}" for c in range(payload)]
        else:
            out.append(names[f])
    return tuple(out)


def apply_encoding(X, cat_cols, encoders):
    """Wendet eine mit `encode_dataset` (auf dem Training) gelernte Kodierung auf NEUE Zeilen (Test) an - nie neu aus den Testdaten berechnet, sonst würde der Test die Kodierung mitbestimmen."""
    cat_cols = list(cat_cols)
    keep_cols = [f for f in range(X.shape[1]) if f not in cat_cols]
    parts = [X[:, keep_cols]]
    for f in cat_cols:
        kind, payload = encoders[f]
        vals = X[:, f].astype(int)
        if kind == "onehot":
            n_cat = payload
            oh = np.zeros((len(vals), n_cat))
            valid = vals < n_cat
            oh[np.arange(len(vals))[valid], vals[valid]] = 1.0
            parts.append(oh)
        else:
            lookup = payload
            safe = np.clip(vals, 0, len(lookup) - 1)
            parts.append(lookup[safe].reshape(-1, 1))
    return np.column_stack(parts)
