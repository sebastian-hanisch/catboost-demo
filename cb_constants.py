"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

FEATURES = [("Distanz", "km"), ("Ladegewicht", "kg"), ("Stopps", ""), ("Verkehr", "0-1"), ("Wetter", "0-1"), ("Wochentag", "0 = Mo"), ("Zeitfenster-Enge", "0-1"), ("Fahrerjahre", "Jahre"),
            ("Depot", "Nr.")]
N_BASE = len(FEATURES)
CAT_FEATURES = (5, 8)                # Wochentag (7 Stufen, gut besetzt), Depot (24 Stufen, dünn besetzt) - die zwei kategorischen Testflächen dieses Stücks

TASKS = ("class", "reg")
TASK_LABELS = {"class": "Klassifikation: kommt die Lieferung zu spät?", "reg": "Regression: wie lange dauert die Lieferung?"}
DEFAULT_TASK = "class"

N_MIN, N_MAX, DEFAULT_N = 400, 3000, 1200
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0, 8, 3
LABEL_NOISE_MIN, LABEL_NOISE_MAX, DEFAULT_LABEL_NOISE = 0, 20, 0
TEST_SHARE = 0.3
DEFAULT_SEED = 7

SWEEP_SEEDS = tuple(range(100000, 100005))

OVERFIT_GAP_CLASS = 0.08
OVERFIT_RATIO_REG = 1.6
UNDERFIT_SHARE = 0.75

# CatBoost-eigene Regler
# Vorsicht bei geordnetem Boosting (siehe README "Was nicht funktioniert hat"): jeder Baum wächst auf einem Bruchteil der Zeilen (1/n_blocks), depth=4+ mit wenig lambda kann bei
# vielen Runden für Regression zunehmend instabil werden (einzelne Zeilen laufen davon) - DEFAULT_LAM/DEFAULT_LR/DEFAULT_N_ROUNDS sind bewusst vorsichtig gewählt, damit der Standardfall
# stabil bleibt; wer sie am Regler nach oben treibt, sieht die Instabilität selbst (ehrliches Verhalten, keine versteckte Klammer).
N_ROUNDS_MIN, N_ROUNDS_MAX, DEFAULT_N_ROUNDS = 1, 150, 40
DEPTH_MIN, DEPTH_MAX, DEFAULT_DEPTH = 1, 6, 3                     # symmetrische Bäume: 2^Tiefe Blätter
LR_MIN, LR_MAX, DEFAULT_LR = 0.02, 1.0, 0.1
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.0, 20.0, 5.0
N_BLOCKS_MIN, N_BLOCKS_MAX, DEFAULT_N_BLOCKS = 2, 32, 8           # Stützmodelle des geordneten Boostings (Annäherung an CatBoosts O(log n))
ENCODING_OPTIONS = ("ordered", "naive", "onehot")
ENCODING_LABELS = {"ordered": "Geordnete Ziel-Statistik", "naive": "Naive Ziel-Mittelwert-Kodierung", "onehot": "One-Hot"}
DEFAULT_ENCODING = "ordered"
TS_PRIOR_WEIGHT = 1.0                # Gewicht des globalen Mittels in der (naiven wie geordneten) Ziel-Statistik, verhindert Division durch 0 bei leeren Kategorien

DEFAULT_MAP = (0, 3)

COLORS = {"train": "#1f77b4", "test": "#d62728", "ordered": "#2ca02c", "naive": "#d62728", "onehot": "#1f77b4", "insample": "#d62728", "loo": "#2ca02c"}

PRESETS = {
    "🌳 Standard": dict(task="class", encoding="ordered", ordered=True, depth=DEFAULT_DEPTH, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=DEFAULT_LAM, n_blocks=DEFAULT_N_BLOCKS,
                        n_depots=24, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED),
    "🪓 Ein Schritt (kein Boosting)": dict(task="class", encoding="ordered", ordered=False, depth=1, n_rounds=1, lr=1.0, lam=DEFAULT_LAM, n_blocks=DEFAULT_N_BLOCKS, n_depots=24,
                                          n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED),
    "🎯 Naive Kodierung": dict(task="class", encoding="naive", ordered=False, depth=DEFAULT_DEPTH, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=DEFAULT_LAM, n_blocks=DEFAULT_N_BLOCKS,
                              n_depots=24, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED),
    "⚖️ Geordnete Kodierung": dict(task="class", encoding="ordered", ordered=False, depth=DEFAULT_DEPTH, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=DEFAULT_LAM, n_blocks=DEFAULT_N_BLOCKS,
                                  n_depots=24, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED),
    "📈 Regression Standard": dict(task="reg", encoding="ordered", ordered=True, depth=DEFAULT_DEPTH, n_rounds=DEFAULT_N_ROUNDS, lr=DEFAULT_LR, lam=DEFAULT_LAM, n_blocks=DEFAULT_N_BLOCKS,
                                   n_depots=24, n=DEFAULT_N, n_noise=DEFAULT_NOISE, label_noise=0, seed=DEFAULT_SEED),
}
PRESET_HELP = {
    "🌳 Standard": "Geordnete Ziel-Statistik + geordnetes Boosting (8 Stützblöcke), Tiefe 3 (8 Blätter, symmetrisch), 40 Runden: Trainingsfehler 21.4 %, Testfehler 22.2 % (Raten: 48.9 %).",
    "🪓 Ein Schritt (kein Boosting)": "Ein einzelner symmetrischer Tiefe-1-Baum (Lernrate 1, gewöhnliches Boosting): Testfehler 28.9 % - deutlich besser als Raten (48.9 %), aber noch weit von einem fertigen Ensemble entfernt.",
    "🎯 Naive Kodierung": "Gewöhnliches (nicht geordnetes) Boosting mit naiver Ziel-Mittelwert-Kodierung für Wochentag und Depot: Trainingsfehler 15.7 %, Testfehler 18.6 % - die Kodierung selbst leckt messbar (siehe Experiment unten), was hier aber nicht automatisch zu einem schlechteren TESTfehler führt.",
    "⚖️ Geordnete Kodierung": "Dieselben Einstellungen, nur geordnete statt naive Kodierung: Trainingsfehler 17.3 %, Testfehler 20.6 % - etwas schlechter als naiv in diesem Durchlauf. Die Kodierungs-Leckage zeigt sich zuverlässig in der Korrelation mit dem eigenen Ziel (Experiment unten), nicht zwangsläufig im End-zu-End-Testfehler dieses Beispiels.",
    "📈 Regression Standard": "Geordnete Kodierung + geordnetes Boosting, Ziel Lieferdauer: Test-RMSE 14.2 Minuten - spürbar höher als Gradient Boosting/XGBoost/LightGBM (rund 9-10 Minuten), der Preis der vereinfachten Blockannäherung (siehe README).",
}
