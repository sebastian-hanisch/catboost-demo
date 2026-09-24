# CatBoost – geordnetes Boosting gegen Prediction Shift – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-catboost-demo.streamlit.app/)**

Neuntes und **letztes** Stück der **Baumbasierten Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", fünftes Stück des
**Boosting-Asts** (nach AdaBoost, Gradient Boosting, XGBoost, LightGBM): anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo
**ein** Verfahren – **CatBoost** (Prokhorenkova et al. 2018) – an einem wachsenden Beispiel.
Vehikel: dieselben **Lieferungen** wie in cart-demo/.../lightgbm-demo, plus ein neuntes Merkmal - **Depot** (24 Stufen, kategorisch), eigens für dieses Stück zurückgehalten.
Alle Daten sind erzeugt, alle Zahlen gemessen und in `tests/test_claims.py` festgehalten – keine echten Daten, `catboost` nur in den Tests als Gegenprobe.

**Bezug zu OR:** ein Depot-Effekt, der nicht an einzelnen Lieferungen hängen bleibt (keine Leckage), ist eine ehrlichere Grundlage für die Standortplanung – eine Kodierung, die nur auf
wenigen Lieferungen je Depot beruht, würde sonst zufällige Schwankungen für ein echtes Standortmerkmal halten.

**Einordnung in die Reihe:** jedes vorige Stück hat die Split-Suche oder Regularisierung verändert (Varianz → regularisierter Gain → Histogramme/blattweise) - CatBoost stellt eine andere
Frage: **woher kommt der Gradient, den ein Baum lernt?** In gewöhnlichem Boosting von einem Modell, das die Zeile schon gesehen hat - ihr Rest wirkt dann systematisch zu klein
(**Prediction Shift**). Dieselbe Falle gibt es bei kategorischen Merkmalen (naives Target Encoding). CatBoosts Antwort: nie mit dem eigenen Etikett einer Zeile rechnen -
**geordnetes Boosting** und **geordnete Target Statistics**. Dazu: **vollständig symmetrische Bäume**.

```
CART → Bagging → Random Forest → Extra Trees                                          (Bagging-Ast, fertig)
CART → AdaBoost → Gradient Boosting → XGBoost → LightGBM → CatBoost (dieses Stück)      (Boosting-Ast, fertig - LINIE VOLLSTÄNDIG)
```

| Frage | Ergebnis (1200 Lieferungen, 3 Rauschmerkmale, 24 Depots, 70 % Training / 30 % Test, Seed 7, sofern nicht anders angegeben) |
|---|---|
| **Symmetrische Gain-Formel gegen Brute-Force** | ✅ Bei Tiefe 1 (eine Gruppe) findet der symmetrische Baumkern exakt denselben besten Split wie eine Brute-Force-Suche über dieselben Bin-Grenzen. |
| **Geordnete Target Statistics nutzt nie das eigene Etikett** | ✅ Direkt nachgewiesen: verändert man eine einzelne Zeile massiv, ändert sich die Kodierung aller FRÜHEREN Zeilen derselben Kategorie nicht - bei naivem Target Encoding ändert sich jede andere Zeile derselben Kategorie sofort. |
| **Geordnetes Boosting: Block k nie mit den eigenen Zeilen trainiert** | ✅ Struktur-Invariante direkt geprüft: die Zeilen, mit denen der Baum für Block k trainiert wird, und die Zeilen, deren Vorhersage er aktualisiert, sind immer disjunkt. |
| **Prediction Shift auf reinem Rauschen** (kein echtes Signal, 400 Zeilen, σ² = 49) | ✅ Bei Tiefe 6: In-Sample-Rest 39,1 (unter der wahren Varianz - sieht besser aus, als er ist), Rest gegen frische Werte 60,2 (darüber) - beide Enden weichen mit wachsender Tiefe weiter von σ² ab. |
| **Leckage: Target Encoding gegen geordnete Target Statistics** (Mittel über fünf Datensätze) | ✅ Naiv: R² mit dem Ziel 5,6 % im Training gegen 3,4 % im ehrlichen Test (Leckage). Geordnet: 1,0 % im Training - schon nah am ehrlichen Wert. |
| **Wirkung der Kardinalität** (4 bis 48 Depot-Stufen) | ✅ Naives Leck wächst in der Tendenz mit der Kardinalität (−0,2 % bei 4 Depots auf +8,5 % bei 48) - weniger Zeilen je Kategorie heißt mehr Gewicht der eigenen Zeile. Geordnet bleibt über alle Stufenzahlen nah bei 0. |
| **Kreuzprobe mit der echten `catboost`-Bibliothek** | ⚠️ Nur über Rang-/Fehlergrenzen (Korrelation > 0,9) - die eigene, vereinfachte Blockannäherung des geordneten Boostings unterscheidet sich von CatBoosts echtem $O(\log n)$-Schema, kein exakter Abgleich. |
| **Geordnetes Boosting gegen gewöhnliches, End-zu-End** | ⚠️ **Ehrlich negativ:** in dieser vereinfachten Fassung (eine feste Permutation, feste statt geometrisch wachsender Blöcke) schneidet geordnetes Boosting NICHT besser ab als gewöhnliches - der Preis, weniger Zeilen je Baum zu sehen, überwiegt den Gain aus weniger Verzerrung (siehe "Was nicht funktioniert hat"). |

## Was die Demo zeigt

- **CatBoost in Aktion:** Runde für Runde mit Schritt-Regler und Abspielen - der symmetrische Baum dieser Runde (jede Ebene eine Raute mit EINEM Split für alle Zweige, Blätter mit
  Newton-Gewichten darunter).
- **Was das Ensemble gelernt hat:** Trainings- und Testfehler gegen die Rundenzahl mit dem besten Testpunkt markiert, Wichtigkeit je Merkmal (nach Kodierung).
- **Regler:** Aufgabe, Kodierung (geordnet | naiv | One-Hot) für Wochentag und Depot, geordnetes Boosting an/aus, Tiefe (symmetrisch: 2^Tiefe Blätter), Rundenzahl, Lernrate, λ, Stützblöcke,
  Depot-Stufen, Rauschmerkmale, falsche Etiketten, Lieferungen, Seed.
- **Drei Experimente auf Knopfdruck:** Prediction Shift (in-sample gegen frische Werte, reines Rauschen), Leckage der Kodierung (Korrelation mit dem Ziel, Training gegen ehrlichen Test),
  Wirkung der Kardinalität (Depot-Stufenzahl).

## Modell und Verfahren

- **Baumkern** (`cb_tree.py`, neu geschrieben): **vollständig symmetrische (oblivious) Bäume** - jede Ebene EIN Split (Merkmal + Schwelle) für ALLE aktuellen Knoten gleichzeitig, gewählt
  um den über alle Gruppen summierten Gain zu maximieren (dieselbe Gain-Formel wie xgboost-demo/lightgbm-demo). Ein Baum der Tiefe $d$ hat immer $2^d$ Blätter, dargestellt als $d$ Splits
  statt eines Baums aus Knoten.
- **Geordnetes Boosting** (`cb_algorithm.py`, vereinfacht: feste Blöcke statt CatBoosts $O(\log n)$ geometrisch wachsender Stützmodelle einer zufälligen Permutation): Block $k$'s Baum
  trainiert nur mit Zeilen aus Blöcken $0,\dots,k-1$ und aktualisiert nur Block $k$'s Vorhersage. Block 0 hat keinen Vorgänger und bekommt einen eigenen, gewöhnlichen Baum (siehe unten).
- **Geordnete Target Statistics** (`cb_encoding.py`): dieselbe Idee für ein kategorisches Merkmal - die Kodierung einer Zeile nutzt nur den Ziel-Mittelwert der Zeilen derselben Kategorie,
  die VOR ihr in der Permutation liegen (plus einen geglätteten Anteil des globalen Mittels), nie ihr eigenes Etikett. Naives Target Encoding (Vergleichsmaßstab) und One-Hot ebenfalls implementiert.

## Was nicht funktioniert hat / gefundene Fehler

- **Gefundener und behobener Fehler (Regression explodierte nach ~30 Runden):** der erste Entwurf ließ Block 0 (kein Vorgänger) für immer bei der Startvorhersage stehen. Jeder Baum, der
  über die Blockkette von Block 0 abhing, fitte deshalb runde für Runde DIESELBE (nicht schrumpfende) Korrektur - statt zu konvergieren, addierte sich diese Korrektur unbegrenzt auf
  (Vorhersagen liefen bis in den vierstelligen Bereich davon, bei einer echten Dauer von 15-200 Minuten). Behoben: Block 0 bekommt einen eigenen, gewöhnlichen (selbstbezogenen) Baum -
  der einzige Rest an "ungeordneter" Leckage in diesem vereinfachten Schema, begrenzt auf 1/`n_blocks` der Zeilen, derselbe Preis, den auch CatBoosts eigene Stützmodelle für die ersten
  Zeilen der Permutation zahlen. Ein Regressionstest hält das fest (`test_ordered_boosting_is_numerically_stable_at_default_like_settings`).
- **Geordnetes Boosting gewinnt in dieser vereinfachten Fassung nicht gegen gewöhnliches:** die ursprüngliche Erwartung war, dass geordnetes Boosting durchgehend besser generalisiert
  (weniger Prediction Shift). Gemessen über 16 Kombinationen aus Tiefe (2, 3, 4, 6), Blockzahl (4, 8) und Datensatzgröße (600, 1200; Seed 7) zeigt sich eher das Gegenteil: geordnet hat in 11 Kombinationen den höheren Testfehler,
  in 5 den niedrigeren (einzelne Testfehler, keine Mittelung über Datensätze). Grund, so weit nachvollzogen: jeder Baum sieht nur einen Bruchteil der Zeilen (höchstens `(n_blocks-1)/n_blocks`), UND es wird nur EINE feste Permutation verwendet (kein
  Mitteln über mehrere, wie es das echte CatBoost tut) - der Effizienzverlust überwiegt den Verzerrungsgewinn auf diesem Datensatz. Ehrlich als offene Grenze berichtet, nicht wegoptimiert.
- **Kodierungs-Leckage zeigt sich klar in der Kodierung selbst, aber nicht zuverlässig im End-zu-End-Testfehler:** die R²-Lücke (Training gegen ehrlichen Test) der naiven Kodierung ist
  eindeutig und wächst mit der Kardinalität - ein direkter, sauberer Nachweis der Leckage. Im vollen Boosting-Modell (Preset "Naives Target Encoding" gegen "Geordnete Target Statistics") schneidet naive
  Kodierung beim Testfehler in diesem Beispiel sogar leicht BESSER ab, weil der zugrunde liegende Depot-Effekt echt und zwischen Training und Test konsistent ist. Beide Befunde stehen
  nebeneinander in der App, statt den zweiten zu unterschlagen, nur weil er der ursprünglichen Erwartung widerspricht.
- **Keine 2D-Entscheidungsgrenzen-Karte** wie in den Vorgänger-Stücken: die kodierten Merkmale haben je nach Kodierung unterschiedlich viele Spalten (One-Hot spreizt eine Kategorie in
  viele Spalten auf), eine feste Karten-Achse würde bei jedem Kodierungswechsel etwas anderes bedeuten. Der symmetrische Baum selbst zeigt das Besondere dieses Stücks ohnehin besser.

- **Bin-Fehler (korrigiert 2026-09-24):** die Bin-Zuordnung (`x == Kante` fiel in den rechten Bin) passte nicht zur Regel "`x > Schwelle` geht nach rechts"; bei Kanten, die mit Datenwerten zusammenfallen (seltene 0/1-Merkmale, ganzzahlige Werte, Wiederholungen), war der Split wirkungslos. Jetzt gilt `Kante[k-1] < x <= Kante[k]` (`side="left"`), wie im LightGBM-Stück.
  **Folgen für die Zahlen:** Standard-Preset Trainings-/Testfehler 21,4/22,2 % → 20,7/20,8 %; Naives Target Encoding 15,7/18,6 % → 15,2/19,4 %; Geordnete Target Statistics 17,3/20,6 % → 17,4/20,0 %; Regression Test-RMSE 14,2 → 13,6 Minuten; Prediction Shift bei Tiefe 6: 40,6/59,1 → 39,1/60,2. Leckage- und Kardinalitäts-Experiment unverändert.

## Verifikation

`tests/test_algorithm.py` (14 Tests): symmetrische Gain-Formel exakt gegen Brute-Force bei Tiefe 1; jeder Baum hat exakt $2^d$ Blätter; geordnete Target Statistics nutzt nachweislich nie
das eigene Etikett (naives Target Encoding tut es); geordnetes Boosting trainiert den Baum für Block $k$ nachweislich nie mit Block $k$'s eigenen Zeilen; Regressionstest für den behobenen
Explosions-Fehler; Prediction-Shift-Verzerrung auf reinem Rauschen; Vorhersagen über Rang-/Fehlergrenzen gegen die echte `catboost`-Bibliothek; Grenzfälle (eine Runde, Depot-Erzeugung).
`tests/test_claims.py` (9 Tests) hält **jede Zahl** aus App und README fest. `tests/test_app.py` (21 Tests) prüft die Oberfläche per AppTest (jedes Preset, Aufgaben- und Kodierungswechsel,
Abspielen mit rundenspezifischen Diagramm-Schlüsseln, Permalink, alle drei Experimente).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `cb_tree.py` | Symmetrischer (oblivious) Baumkern (neu geschrieben) |
| `cb_algorithm.py` | Gewöhnliches und geordnetes Boosting |
| `cb_encoding.py` | Naives Target Encoding / geordnete Target Statistics, One-Hot |
| `cb_scenario.py` | Lieferdaten (wie xgboost-demo, plus Depot mit festem Kategorien-Effekt) |
| `cb_evaluation.py` | Analyse, Rundenkurve, Prediction-Shift-, Leckage- und Kardinalitäts-Experimente |
| `cb_visualization.py` | Symmetrischer Baum, Kurven-, Balken- und Wichtigkeitsdiagramme |
| `cb_presets.py`, `cb_constants.py` | Regler, Permalink, Schnellstart-Beispiele, Grenzen |
| `tests/` | Algorithmus-, Claims- und App-Tests |

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\python -m pip install -r requirements.txt
venv\Scripts\python -m streamlit run app.py
```

## Tests ausführen

```bash
venv\Scripts\python -m pip install -r requirements-dev.txt
venv\Scripts\python -m pytest tests -q
```

---

Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
