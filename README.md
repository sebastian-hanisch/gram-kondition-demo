# Bandbreite und Kondition: wo die Rechnung still falsch wird – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-gram-kondition-demo.streamlit.app/)** -->

Sechste Demo der „Konzepte“-Reihe **Kernmethoden: Skalarprodukte tauschen** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie folgt auf die Demo zu Kernel Ridge und dem Mittel eines Gauß-Prozesses (`kernel-ridge-demo`), die voraussetzt, dass die Gleichung (K + λI)a = y exakt gelöst wird und die Rechenwege übereinstimmen, und hebt diese Annahme auf: bei kleinem λ und großer Bandbreite weichen die Rechenwege von der exakten Lösung ab, ohne dass LU oder Eigenzerlegung es melden (Cholesky meldet nur, wenn die Zerlegung scheitert). Die Reihe geht danach mit der Gram-Näherung weiter (`gram-naeherung-demo`). Der Hook der Reihe („der Löser meldet nichts“) steht hier in seiner stärksten Form: **Hauptgestalt Rechenfehler**, die Cholesky-Zerlegung gelingt, der Rückwärtsfehler ist winzig, die Koeffizienten sind falsch. Getrennt davon steht der **Modellfehler** (die Interpolation λ → 0 ist selbst bei exakter Rechnung schlecht) als Beleg, dass der Löser nicht der Grund ist.

Die Daten sind **ausgedacht**: Punkte gleichverteilt in einem Quadrat bzw. auf einer Strecke, eine bekannte glatte Zielfunktion und Rauschen. Es gibt keine reale Datenquelle; in den gelesenen Quellen gab es keine Messung, ob ein Kernmodell in Logistik, Lager oder Fertigung bessere Entscheidungen liefert (belegt sind nur Kriging als Simulations-Metamodell und die One-Class SVM bei Telemetrie). Gezählt werden Punktmengen, Zufalls-Seeds und Eigenwerte; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet.

## Warum dieses Problem

Eine Gauß-Gram-Matrix verliert mit wachsender Bandbreite σ ihre kleinen Eigenwerte an die Rundung. LU und Eigenzerlegung geben trotzdem eine Zahl zurück, Cholesky nur, wenn die Zerlegung gelingt (σ = 5, λ = 0: 156 von 300 Punktmengen); in diesen Fällen ist das Residuum klein. Die Kernfragen der Seite: (1) Wie viele Eigenwerte hat die Gram-Matrix noch? (2) Gelingt die Rechnung, und stimmt sie? (3) Was kauft ein Nugget? (4) Wovon hängt „Interpolation“ ab? Jede Antwort ist eine **Zählung oder ein Band über viele Punktmengen bzw. Zufalls-Seeds**, bei der Rechnung mit exakter Referenz (die Gleitkomma-Matrix in ganzen Zahlen gelöst), nie eine Einzelinstanz allein, außer den exakt gerechneten Handbeispielen; nur das Flat Limit (Experiment 4) und diese Handbeispiele sind exakt gerechnet und brauchen keine Stichprobe.

## Modell

**Gauß-Kern** k(x, x') = exp(−‖x − x'‖²/(2σ²)) mit Bandbreite σ (Längenmaß der Eingabe), γ = 1/(2σ²); d = 1 oder 2, n ≤ 240. **λ** wird zur Diagonale addiert, K + λI, in zwei Rollen: als **Nugget** (Jitter, Rechenhilfe) und, wie in `kernel-ridge-demo`, als Rauschvarianz des GP-Mittels (Modell); die App sagt bei jeder Zahl, welche Rolle gemeint ist. K ist positiv semidefinit (Mercer); für verschiedene Punkte ist sie mathematisch positiv definit, bei doppelten Punkten singulär, und in Gleitkomma bei großem σ oft nicht mehr positiv semidefinit. cond = λ_max/λ_min der symmetrischen Matrix, für K + λI gleich (λ_max + λ)/(λ_min + λ); liegt der kleinste Eigenwert unter der Rundungsgrenze, ist cond nur eine **Untergrenze** (1/eps = 4,5e15). **Exakte Lösung** heißt immer: die exakte Lösung des Gleitkomma-Systems (Matrixeinträge als Dualbrüche, gelöst in ganzen Zahlen nach Bareiss), nicht die wahre Funktion. Fehler = ‖a − a_exakt‖/‖a_exakt‖; η = ‖Aa − y‖/(‖A‖‖a‖ + ‖y‖) mit exakt gerechnetem Residuum. Formeln, Symboltabelle und Quellen stehen im 📐-Expander der App. Den Aufbau der Matrix (Differenzen der Punkte, dann exp) legt `gkd_kernel.py` fest; die Messungen gelten für diesen Aufbau.

## Methodik

Fünf Experimente, jedes mit Verdikt (reine Funktion in `gkd_evaluation.py`):

1. **Die Fläche σ × λ ist ein Tal.** Mittlerer Testfehler gegen die Wahrheit über 7 Bandbreiten und 6 Werte von λ (42 Zellen), Beispiel-Regression in 2D, Leave-one-out je Zufalls-Seed.
2. **Cholesky gelingt, Lösung falsch (der Hook).** n ≤ 40 Punkte im Einheitsquadrat, bis zu 30 Punktmengen, drei Gleitkomma-Wege gegen die exakte Lösung, exakt gerechneter Rückwärtsfehler η, Vorhersagefehler an 100 Testpunkten (exakt in Brüchen). Dazu: ändert der Löser den Testfehler gegen die Wahrheit?
3. **Das Nugget als Schranke aus λ allein:** cond(K + λI) ≤ 1 + n/λ (Satz, weil λ_max ≤ Spur K = n und λ_min(K + λI) ≥ λ) und die beobachtete Faustregel R(λ) = 16 − log10(1 + n/λ) für die richtigen Dezimalstellen (kein Satz), gegen die beobachteten Stellen der LU. Gegenbefund: ein Nugget ändert das Modell.
4. **Flat Limit: die Eigenwertleiter.** Gauß-Kern auf ganzzahligen Punkten mit q = exp(−1/(2σ²)) = 1 − 10^−k als Bruch (so bleibt K rational): exakte Eigenwerte (Jacobi-Verfahren in 120 Dezimalstellen) gegen die geschlossene Leiter λ_m ≈ c_m (1/σ)^(2(m−1)), c_m = R_mm²/(m−1)!, gegen die auf Doppelte gerundete Matrix und gegen `numpy.linalg.eigvalsh`.
5. **Interpolation ist ein Objekt der Abschneidegrenze.** Testfehler der Interpolation (Eigenwerte unter τ · λ_max abgeschnitten) gegen die per Leave-one-out gewählte Vorhersage und die konstante Vorhersage, immer Mittel, Median und Maximum; vorgerechnet: 300 frische Sätze zu je 30 Zufalls-Seeds, die n-Reihe, die τ-Reihe und die Gegenprobe „exakt gegen Gleitkomma“.

**Schnellstart (Presets in der Seitenleiste):** „Cholesky gelingt, Lösung falsch (n = 20, σ = 5, λ = 0)“ (Standard), „Gut konditioniert (σ = 0,5, λ = 0,01)“, „Mit Nugget 1e-6“, „Interpolation (σ = 1, λ = 0)“, „Leave-one-out gewählt“ und „Flat Limit (n = 5, 1/σ → 0)“. Das aktive Preset ist hervorgehoben; ein Preset ohne Effekt meldet keinen Effekt („Gut konditioniert“ meldet keine Warnung, `tests/test_presets.py`).

**Zufall:** alle Erzeuger ziehen aus SplitMix64 (`gkd_rng.py`, wörtliche Kopie der Vorlage mit Kopier-Test; `NormalDist().inv_cdf(random())` für Normalzahlen), kein `numpy.random`. Eine frühere Messung dieser Seite lief mit `numpy.random.default_rng`; **alle Zahlen dieser README sind mit den Zufalls-Strömen der App neu gemessen**, als Band über frische Zufalls-Seeds (`tools/gkd_messung.py`, Ausgabe `tools/out_gkd_messung.txt`; vorgerechnete Zahlen `gkd_frozen.json`, `tools/gkd_make_frozen.py` mit `--check`). Laufzeit-Abhängigkeiten: Streamlit, numpy, Plotly und scipy (nur `scipy.linalg.solve_triangular` für die beiden Dreieckssysteme der Cholesky-Lösung; die Zerlegung selbst ist `numpy.linalg.cholesky`); scikit-learn kommt nur in den Tests vor (Fremdreferenz `rbf_kernel`, `KernelRidge`). Der Zufalls-Strom der App und die LAPACK-Bibliothek bestimmen die Zählung „Cholesky gelingt“: sie ist ein **Rundungslotto** und eine **Messung dieser Plattform** (Windows, numpy 2.5.3, Python 3.14); ein Linux-Lauf steht aus.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für den Gauß-Kern, d ≤ 2, Zähleinheiten Punktmengen (je eine eigene y-Ziehung), Zufalls-Seeds und Eigenwerte. „Satz“ = 30 Punktmengen bzw. 30 Zufalls-Seeds. Die Spalte „Test“ nennt die Testfunktion in `tests/test_claims.py` (Bänder, kleinere Fassung der Messung) oder in der genannten Datei; „Messung dieser README, nicht im Test“ heißt, dass die Zahl aus `tools/out_gkd_messung.txt` stammt.

| Frage | Befund | Test |
|---|---|---|
| Cholesky gelingt, n = 20, σ = 5, λ = 0, y ~ N(0, I) | 156 von 300 Punktmengen (10 Sätze zu 30; je Satz 12 bis 19, Median 16); davon liegt die Lösung in 137 (88 %) um mehr als 10 % daneben; Median des Fehlers 0,45, Stichprobenmaximum je Satz 1,6 bis 32 (schwere Tails, kein Befund) | `test_falle_sigma_5_lambda_0_cholesky_succeeds_in_a_minority_to_half_and_the_solution_is_mostly_wrong` (weites Band), Zahlen: Messung dieser README |
| Rückwärtsfehler in diesen Fällen | exakt gerechnetes η höchstens 1,0e-16 in jeder der 1944 gelungenen Zerlegungen der sieben Zellen von n = 20 (η wird nur für gelungene Zerlegungen gerechnet), also unter eps = 2,2e-16: das Residuum verrät den Fehler nicht | gleicher Test (η unter eps) |
| Vorhersage | bei σ = 5, λ = 0 weicht die Vorhersage an 100 Testpunkten in 156 von 156 gelungenen Zerlegungen um mehr als 1e-3 (relativ) von der exakten ab; über alle sieben Zellen in 486 von 1944; die Hoffnung „falsche Koeffizienten, richtige Vorhersage“ bestätigt sich nicht | gleicher Test (`pred_bad == ok`), Gesamtzahl: Messung dieser README |
| Kleinere Bandbreiten, λ = 0 | σ = 3: 290 von 300, Median des Fehlers 1,3e-2, 40 über 10 %; σ = 2: 298 von 300, 2,7e-4; σ = 1: 300 von 300, 1,8e-7 | `test_falle_smaller_bandwidths_fail_less_and_err_less` |
| Mit Nugget, σ = 5, 300 Punktmengen je Zelle | λ = 1e-10 / 1e-6 / 1e-2: Cholesky gelingt in 300 von 300; Median des Fehlers 1,6e-6 / 2,1e-10 / 2,4e-14, Stichprobenmaximum je Satz höchstens 3,2e-6 / 3,9e-10 / 4,5e-14 | `test_falle_with_a_nugget_all_succeed_with_small_errors` |
| Nugget-Schranke | n = 20, 5 Bandbreiten × 3 Werte von λ × 30 Punktmengen = 450 Instanzen: cond über 1 + n/λ in 0, beobachtete Stellen der LU unter R(λ) in 0; kleinste Beobachtung 5,43 / 9,44 / 13,39 Stellen gegen R = 4,70 / 8,70 / 12,70 bei λ = 1e-10 / 1e-6 / 1e-2 | `test_nugget_bound_and_digits_rule_hold_in_every_instance_of_a_reduced_run` (90 Instanzen), Handwerte in `test_kernel.py` |
| Ein Nugget ändert das Modell | n = 40, σ = 5, 10 Sätze zu 30 Zufalls-Seeds, Testfehler gegen die Wahrheit, Satz-Mittel bei λ = 1e-8 / 1e-4 / 1e-2 / 0,1: 0,30 bis 0,69 (Median der Sätze 0,45) / 0,16 bis 0,20 (0,18) / 0,29 bis 0,32 (0,29) / 0,34 bis 0,37 (0,34); konstante Vorhersage 0,78 bis 0,80 | `test_nugget_changes_the_model_best_at_a_middle_lambda` |
| Rechenfehler gegen Modellfehler | n = 40, σ = 5, 10 Zufalls-Seeds, 5 Sätze: bei λ = 1e-8 und 1e-4 ist der Testfehler der exakten Lösung und der LU in den ersten drei Stellen gleich; bei λ = 0 liegen beide im Mittel zwischen 1e4 und 2,2e5 (Median 5e2 bis 5,5e4) gegen 0,78 der Konstanten | `test_solver_does_not_change_the_test_error_but_lambda_zero_is_far_above_the_constant` |
| Fläche σ × λ, n = 60, Rauschen 0,3, 20 Sätze zu 30 Zufalls-Seeds | beste Zelle σ = 1,6, λ = 0,01 in 19 von 20 Sätzen (sonst σ = 0,8, λ = 0,1), Testfehler 0,046 bis 0,065 (Median der Sätze 0,052); Leave-one-out je Zufalls-Seed gewählt 0,055 bis 0,094 (0,069); im Rückblick je Zufalls-Seed beste Zelle 0,042 bis 0,055; konstante Vorhersage 0,78; Zelle σ = 0,8, λ = 1e-6 beim 18,5- bis 40fachen der Konstanten (Median 28); σ = 6,4: 0,076 bis 0,107 bei λ = 1e-6, 0,68 bei λ = 10 | `test_surface_valley_best_cell_and_loo_choice` |
| Interpolation, σ = 1, n = 60, τ = 1e-13 | 60 Sätze: über der konstanten Vorhersage (0,78) in 1800 von 1800 Instanzen; Satz-Median 150 bis 2892 (Median der Sätze 449), Satz-Mittel 496 bis 1,6e5; Testfehler bei per Leave-one-out gewähltem λ: Satz-Median 0,045 bis 0,063 | `test_interpolation_sigma_one_is_far_above_the_constant_in_every_instance_and_loo_is_far_below` |
| Streuung über frische Sätze (vorgerechnet) | 300 Sätze zu 30 (9000 Instanzen): Satz-Median 197 bis 1232 (95-%-Band, Median der Sätze 457), Satz-Mittel 1080 bis 8,8e4 (ebenfalls 95-%-Band, Median 5398), Satz-Maximum im Median 7,0e4, größter Einzelwert 1,6e7; über der konstanten Vorhersage in 9000 von 9000 | `test_frozen_streuung_bands_and_the_set_is_not_the_number` |
| σ = 0,3 und σ = 3, n = 60, τ = 1e-13 | σ = 0,3 (60 Sätze): Satz-Median 0,29 bis 0,50 gegen 0,78, über der Konstanten in 288 von 1800; σ = 3: Satz-Median 4,4 bis 25, über der Konstanten in 1750 von 1800 | `test_sigma_small_is_not_catastrophic_and_sigma_three_is` |
| Die Wahl durch Leave-one-out | σ = 1, 60 Sätze: gewähltes λ = 0,1 in 1486, 0,01 in 304, 1 in 7, 1e-4 in 3 von 1800 Instanzen | `test_loo_choice_is_a_small_lambda_not_the_interpolation` (Zählung: Messung dieser README) |
| Abschneidegrenze (vorgerechnet) | σ = 1, n = 60: Satz-Mittel (Median der Satz-Mediane) bei τ = 1e-15 / 1e-13 / 1e-11 / 1e-9 / 1e-7 / 1e-5: 5394 (460) / 5394 (460) / 2242 (438) / 208 (95) / 8,8 (5,5) / 0,48 (0,37); bei τ = 1e-13 werden bei σ = 1 im Median 0 von 60 Eigenwerten abgeschnitten, bei σ = 3 im Median 17 | `test_frozen_tau_series_sigma_one_falls_with_tau_and_sigma_small_does_not_depend_on_it`, `test_the_truncation_threshold_decides_the_number` |
| n-Reihe (vorgerechnet) | σ = 1, τ = 1e-13, je n 60 Sätze: Median der Satz-Mediane 1,8 / 31 / 460 / 3130 / 26 bei n = 20 / 40 / 60 / 120 / 240, nicht monoton; bei n = 240 werden im Median 90,5 von 240 Eigenwerten abgeschnitten (n = 120: 6,5); der Abfall bei n = 240 fällt mit dieser Abschneidung zusammen (Deutung, nicht getrennt gemessen) | `test_frozen_n_series_is_not_monotone_and_the_cut_matches_the_drop` |
| Ist es die Rundung? | n = 60, σ = 1: die exakte Interpolation (Bareiss, Vorhersage in Brüchen) und die Eigenzerlegung bei τ = 1e-13 geben denselben Testfehler (Verhältnis 0,9998 bis 1,0001 in 200 von 200 frischen Zufalls-Seeds, darunter 5 mit ‖a‖ über 1e10); Norm der exakten Koeffizienten im Median 6,5e7 (Etiketten-Norm 7,1). Im Median und in Zufalls-Seeds mit ‖a‖ unter 1e10 ist die Rundung nicht die Ursache, sondern die Konditionszahl (das Rauschen wird mit 1/λ_klein verstärkt) | `test_frozen_exact_interpolation_equals_floating_point_except_possibly_in_the_extreme_tail` |
| Eigenwertleiter (exakt) | x = (0, 1, 2, 3, 4): c = (5, 10, 7, 12/5, 12/35); Verhältnis exakter Eigenwert zu Leiter 1,000000 bei k = 8 (alle fünf), 0,999993 bis 0,999999 bei k = 6, 0,93 bis 0,99 bei k = 2 (σ = 7,05); Steigungen 0 / 2 / 4 / 6 / 8 auf 1e-3; cond = 9,1146e31 bei k = 8 | `test_flat.py`, `test_literatur_kondition.py`, `test_flat_limit_ratio_and_condition_bands_of_the_readme` |
| Eingaberundung, n = 6, x = 0 bis 5 | bei k = 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 / 12 sind in `eigvalsh` 6 / 6 / 5 / 4 / 3 / 3 / 3 / 2 / 2 / 2 Eigenwerte echt (Abweichung gegen die rationale Matrix unter 10 %), in der auf Doppelte gerundeten Matrix 6 / 6 / 5 / 4 / 3 / 3 / 3 / 3 / 2 / 2: in 9 von 10 Zellen gleich, die Grenze liegt innerhalb eines Eigenwerts; die Leiter sagt 6 / 6 / 4 / 4 / 3 / 3 / 2 / 2 / 2 / 2 voraus (innerhalb eines Eigenwerts in 10 von 10); der kleinste Eigenwert der gerundeten Matrix ist in 7 von 8 Zellen (k = 4 bis 12) negativ, der von `eigvalsh` in 8 von 8: die gerundete Gram-Matrix ist nicht mehr positiv semidefinit | `test_flat.py::test_n_equals_six_matches_the_measured_counts_to_within_one_eigenvalue` (Band ±1), `test_the_rounded_gram_matrix_is_not_positive_semidefinite_any_more` |
| Zahl echter Eigenwerte (Schwelle 1e-16 · λ_max), je 3 Punktmengen | 1D auf [0, 1], n = 20 / 50 / 100: σ = 0,05: 20 / 45 bis 49 / 60; σ = 1: 8 bis 9 / 12 bis 13 / 15 bis 16; σ = 3: 6 bis 7 / 9 bis 12 / 13. 2D im Einheitsquadrat: σ = 0,2: 20 / 50 / 100; σ = 1: 20 / 43 bis 44 / 51 bis 53; σ = 3: 19 / 27 bis 28 / 28 bis 30. Mehr Punkte kaufen bei fester Bandbreite nur unterproportional mehr Rang: bei σ ≥ 1 wächst die Zahl echter Eigenwerte von n = 20 auf n = 100 auf etwa das 1,5- bis 2,7fache; bei sehr kleinem σ (0,05 bzw. 0,2) wächst sie mit n | `test_real_eigenvalue_count_does_not_rise_with_sigma_and_more_points_buy_little_rank` (Richtung), Zahlen: Messung dieser README |
| Leave-one-out in Gleitkomma | n = 20, 8 Werte von λ, 2 × 30 Zufalls-Seeds: das per Eigenzerlegung gewählte λ ist dasselbe wie bei exakter Rechnung in 60 von 60 Fällen, obwohl der gerechnete mittlere quadratische Leave-one-out-Fehler um bis zu 2,4 % von dem der exakten Rechnung abweicht; gemessen für σ = 1 und σ = 3 | `test_loo_in_floating_point_chooses_the_same_lambda_as_the_exact_computation` (10 Fälle) |
| Cholesky-Grenze der Mini-Instanz | drei Punkte x = (0, 1, 2): der dritte Pivot ≈ 2/σ⁴ erreicht eps bei (2/eps)^(1/4) = 9,74e3 (= 2^(53/4)). Bei Einträgen als Potenzen K_ij = q^((x_i − x_j)²) gelingt `numpy.linalg.cholesky` bis zur Vorhersage in jedem Rasterwert und scheitert ab dem 1,4fachen in jedem Rasterwert (Raster mit 1501 Werten); bei Einträgen exp(−d²/(2σ²)), dem Aufbau der App, scheitert sie erstmals schon beim 0,8fachen und gelingt noch beim 10fachen (38 % der Rasterwerte über dem Doppelten): **die Grenze hängt am Aufbau der Matrix, ein Rundungslotto** | `test_handbeispiel.py::test_8_*` (Bänder) |

Die Handwerte der Mini-Instanz (det K = 135/256, Pivots 1, 3/4, 45/64, Eigenwerte 15/16 und (33 ± √513)/32, cond 5,376, Leave-one-out geschlossen gleich echtem Herausnehmen in Brüchen, cond ≈ 9σ⁴ im Flat Limit, cond 199 bei zwei fast gleichen Punkten) stehen exakt in `tests/test_handbeispiel.py`, `tests/test_fit.py` und `tests/test_flat.py`.

## Befunde und Korrekturen

Eine frühere Messung (anderer Zufallsgenerator, `numpy.random.default_rng`, eine Punktmenge bzw. ein Satz von 30 Zufalls-Seeds) nannte andere Zahlen; sie sind durch die Messung dieser README ersetzt, und die folgenden Aussagen sind dabei geschärft oder zurückgenommen worden (frühere Messung, nicht im Test):

- **„Cholesky gelingt bei n = 20, σ = 5, die Lösung hat Fehler 1,0“** war eine Einzelinstanz. Über 30 Punktmengen gelang die Zerlegung früher in 17, hier in 12 bis 19 je Satz (156 von 300): ein Rundungslotto, und die App zeigt immer die Zählung.
- **„Ridgeless 3308 gegen 0,06“** war das Mittel eines atypischen Satzes von 30 Zufalls-Seeds (damals Median 911,5, Maximum 33 250). Über 300 frische Sätze liegt der Satz-Median im 95-%-Band zwischen 197 und 1232 (Spanne aller 300 Sätze 134 bis 1765) und das Satz-Mittel im 95-%-Band zwischen 1080 und 8,8e4; **Mittel nie ohne Median und Maximum**.
- **„Rundung wirkt als Regularisierung“** ist für diese Matrix im Median widerlegt: die exakte Interpolation gibt denselben Testfehler wie die Gleitkommarechnung. Die frühere Prüfung fand in einem von 24 frischen Zufalls-Seeds mit ‖a‖ ≈ 1e11 einen Ausreißer, in dem die Gleitkommarechnung um das 80fache unter der exakten lag; in den 200 Zufalls-Seeds dieser README (5 davon mit ‖a‖ über 1e10) zeigt sich das nicht (Verhältnis 0,9998 bis 1,0001). Die Aussage gilt deshalb für den Median und für Zufalls-Seeds mit ‖a‖ unter 1e10.
- **„Der 19fache Fehler“** gehört zum **Modell**, nicht zum Löser: die Zelle σ = 0,8, λ = 1e-6 liegt beim 18,5- bis 40fachen der konstanten Vorhersage, und der Löser ändert den Testfehler bei λ = 1e-8 und 1e-4 (n = 40, σ = 5) nicht.
- **Cholesky-Grenze „erstes Scheitern beim 1,19fachen, ab dem 1,32fachen immer“** gilt nur für Einträge als Potenzen eines gerundeten q, nicht für exp(−d²/(2σ²)) (siehe Tabelle).
- **„Satz-Mittel bei λ = 0 bis 2,3e5“** (frühere Fassung dieser README) war falsch gerundet: das größte Satz-Mittel liegt bei 2,2e5 (Grund-Zufalls-Seed 3101, exakte Lösung, siehe `tools/out_gkd_messung.txt`).
- **Eigenzerlegung „verliert nichts“** gilt in 9 von 10 Zellen; bei k = 9 liegt die Grenze innerhalb eines Eigenwerts.

## Ehrliche Grenzen

Negative Befunde stehen gleichberechtigt: Das Gelingen der Cholesky-Zerlegung ist ein Rundungslotto (σ = 3: 290 von 300, σ = 5: 156 von 300); der Fehler der Lösung ändert bei λ = 1e-8 und 1e-4 (n = 40, σ = 5) den Testfehler gegen die Wahrheit nicht messbar, bei λ = 0 ändert er ihn um einen Faktor von bis zu etwa 6 (Satz-Mediane) bzw. 3 (Satz-Mittel), aber nicht die Aussage: beide Lösungen liegen um Größenordnungen über der konstanten Vorhersage (der Modellfehler der Interpolation dominiert); „große Bandbreite heißt Unteranpassung“ gilt nur bei großem λ; „die Interpolation ist nicht gutartig“ gilt für σ = 1 und 3 (n = 60, d = 2); bei σ = 0,3 ist die Interpolation nicht katastrophal, aber nicht durchweg besser: 288 von 1800 Instanzen über der Konstanten; ein Nugget ändert das Modell; die Wahl von λ durch Leave-one-out wird in diesem Test von der Konditionszahl nicht getäuscht (60 von 60; n = 20, σ = 1 und 3, Raster von 8 Werten von λ). Was die Zahlen nicht zeigen, steht hier:

**Annahmen.** Gauß-Kern mit absoluter Bandbreite σ (Längenmaß der Eingabe), γ = 1/(2σ²); Punkte gleichverteilt im Einheitsquadrat bzw. auf einer Strecke, y ausgedacht (Standardnormal bzw. eine glatte Zielfunktion mit Rauschen); „exakt“ heißt die exakte Lösung des **Gleitkomma-Systems**, nicht die wahre Funktion.

**Grenzen.** Nur Gauß-Kern, d ≤ 2, n ≤ 240; exakte Elimination live bis n = 40; Flat Limit nur d = 1; nur Zählungen über Punktmengen, Zufalls-Seeds und Eigenwerte; die Zählung „Cholesky gelingt“ ist ein Rundungslotto und eine Messung dieser Plattform; kein Nutzen von Kernmodellen im Betrieb belegt.

**Bekannte Probleme und Unsicherheiten.** Fehlerziffern hängen an der LAPACK-Bibliothek und werden nur als Bänder gezählt; Stichprobenmaxima haben schwere Tails und sind kein Befund; die Messungen der README nutzen den Zufalls-Strom dieser App (SplitMix64) und sind als Band über frische Zufalls-Seeds angegeben; ein Linux-Lauf steht aus; der Mausklick im Browser ist nur über AppTest und Stichproben im Browser geprüft.

| Annahme oder Reichweite | Was gemessen ist, und wo es endet | Wer setzt an |
|---|---|---|
| **Gauß-Kern** | Alle Zahlen gelten für den Gauß-Kern; die Matérn-Eigenwertabfälle sind in **kernwahl-demo** gemessen, hier nicht | kernwahl-demo |
| **d ≤ 2, n ≤ 240** | Flat Limit nur für d = 1 (ganzzahlige Punkte) exakt gerechnet; die exakte Elimination läuft live bis n = 40, die Interpolation bis n = 240 | - |
| **Plattform** | Die Zählung „Cholesky gelingt in k von m“ ist ein Rundungslotto und hängt an LAPACK, Plattform und Zufalls-Strom; die README nennt sie als Messung dieser Plattform | Linux-Prüfung vor dem Push |
| **Matrix-Aufbau** | Die Rundungsgrenze der Cholesky-Zerlegung hängt davon ab, wie K gebaut wird (Differenzen mit exp(−d²/(2σ²)) gegen Potenzen eines gerundeten q); die App baut mit exp | - |
| **„Interpolation ist nicht gutartig“** | Gemessen für d = 2, n = 60 und σ = 1 (und 3); bei σ = 0,3 nicht katastrophal, aber nicht durchweg besser (288 von 1800 Instanzen über der Konstanten); gutartiges Überanpassen hat nach der Literatur Voraussetzungen (hohe Dimension, Eigenwertabfall), die Messung hier gilt für d = 2 und widerspricht ihr deshalb nicht | - |
| **Kein Nutzen im Betrieb belegt** | Gezählt werden Punktmengen, Zufalls-Seeds und Eigenwerte; ob ein Kernmodell in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, wird hier nicht behauptet | - |
| **Distanzkonzentration** | Mit wachsender Dimension rücken Abstände zusammen; in dieser Demo nicht gemessen | - |
| **Nicht angeboten, nicht gemessen** | Stabilisierte Flat-Limit-Rechnung (RBF-QR u. a.), die Matérn-Kondition, der Rundungsboden in 2D als Leiter, alles ohne Zählung über Punktmengen | - |

**Quellen (gelesene Teile).** Barthelmé, Usevich, *Spectral properties of kernel matrices in the flat limit*, arXiv 1910.14067 v3 (Abstract, Abschnitt 1 samt den drei Abbildungen darin): im Flat Limit wird die Gram-Matrix singulär, die Eigenwerte haben für d = 1 die Größenordnung ε^(2(k−1)). Der geschlossene Koeffizient c_m = R_mm²/(m−1)! ist in den gelesenen Teilen nicht zu finden und hier eigene Rechnung. Terenin u. a., JMLR 25 (2024), arXiv 2210.07893 (Abschnitt 2, Unterabschnitte 3 und 4): hinreichende Bedingung für das Gelingen der Cholesky-Zerlegung; hier nicht nachgerechnet. Rasmussen, Williams, *Gaussian Processes for Machine Learning*, Kapitel 3, Fußnote 10: der Zuschlag „Jitter“. Die Aussagen „Cholesky scheitert ab Bandbreite ...“ stehen in keiner dieser Quellen, sie sind eigene Rechnung. Neu ist nichts davon: Kondition, Nugget-Schranke und Flat Limit sind Lehrbuchwissen; neu ist nur die Vorführbarkeit mit Zähleinheit und exakter Referenz.

## Verifikation

- **Exakte Referenz:** Bareiss gegen Gauß-Elimination in Brüchen (Gleichheit der Brüche, 25 Instanzen), exaktes Residuum gegen Bruchrechnung, exakte Vorhersage gegen Brüche, Jacobi-Eigenwerte gegen das exakte charakteristische Polynom (Faddeev-LeVerrier, Newton in 120 Stellen) und `numpy.linalg.eigvalsh`.
- **Handbeispiel:** drei Punkte, q = 1/2: Determinante, Pivots, Eigenwerte, Lösung auf fünf Wegen (Brüche, `numpy.linalg.solve`, `scipy.linalg.cho_solve`, Eigenzerlegung, `KernelRidge`), Leave-one-out in Brüchen, Flat Limit, Cholesky-Grenze für beide Aufbauten (`tests/test_handbeispiel.py`).
- **Orakel:** `tests/test_oracle_gram_kondition.py` (exakte Löser, Eigenwerte auf drei Wegen, cond-Formel gegen `numpy.linalg.cond`, Leave-one-out gegen echtes Herausnehmen, Vorhersage und η gegen Bruchrechnung, Nugget-Zeilen gegen Neuzählung), `tests/test_literatur_kondition.py` (Größenordnung ε^(2(k−1)), Stufe W).
- **Eigenschaften:** cond fällt in λ und liegt unter 1 + n/λ, Spur K = n, positiv semidefinit bis auf Rundung, Eigenwerte unabhängig von Umordnung, Verschiebung und gemeinsamer Skalierung von Punkten und σ, doppelter Punkt macht K singulär, bei λ > 0 genau eine Lösung mit exakt null Rest (`tests/test_properties.py`).
- **Randfälle:** n = 1, n = 2 mit gleichen Punkten (exakt singulär, ZeroDivisionError, aus der Zählung ausgeschlossen), λ = 0 und λ = 1e-300, σ sehr klein und sehr groß, Regler an den Rändern (`tests/test_edge_cases.py`).
- **Verdikte:** reine Funktionen, je Zweig und je Zahl eine Aussage (`tests/test_verdicts.py`); je Preset die erwartete Verdikt-Zeile (`tests/test_presets.py`: „Gut konditioniert“ meldet keine Warnung).
- **Zufall:** Kopier-Test gegen die Vorlage, SplitMix64-Anker, Normalzahlen, kein `numpy.random` (`tests/test_rng.py`), Erzeuger (`tests/test_scenario.py`).
- **Zahl gegen Test:** Tabelle oben, `tests/test_claims.py`; Texte (`tests/test_texts.py`); vorgerechnete Zahlen mit `--check`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an beiden Rändern, jede Option), Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf) und Footer; Laufzeit-Importe (`tests/test_runtime_imports.py`).
- **Zusammen:** **359 Tests**. Nicht geprüft: Linux (CI und `demo_linux_check.py` stehen aus), Browser-Klicks außerhalb von AppTest.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2): Ergebnis, Kernfragen, fünf Experimente |
| `gkd_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `gkd_rng.py` | SplitMix64 (wörtliche Kopie der Vorlage, Kopier-Test) |
| `gkd_scenario.py` | Erzeuger: Punkte im Quadrat und auf der Strecke, Zielvektoren, Beispiel-Regression |
| `gkd_kernel.py` | Gauß-Gram-Matrix, Eigenwerte, cond, Nugget-Schranke, Pivots |
| `gkd_fit.py` | Rechenwege (LU, Cholesky, Eigenzerlegung), Abschneiden, Leave-one-out, Wahlregel |
| `gkd_exact.py` | exakte Rechnung: Bareiss, Rückwärtsfehler, Vorhersage, Jacobi-Eigenwerte, Pivots in Brüchen |
| `gkd_flat.py` | Flat Limit: rationale Matrix, Leiter, Gram-Schmidt, Zählung echter Eigenwerte |
| `gkd_experiments.py` | Rechnungen der fünf Experimente |
| `gkd_evaluation.py` | Zusammenfassungen, Verdikte als reine Funktionen, Tabellen |
| `gkd_format.py` | Zahlen für Anzeigetexte (Dezimalkomma) |
| `gkd_texts.py` | Anzeigetexte, Annahmen, Formeln |
| `gkd_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `gkd_presets.py`, `gkd_layout.py` | Permalink, Presets, Zufalls-Seed-Button; fester Kopf |
| `gkd_frozen.json` | vorgerechnete Zahlen (Experiment 5) |
| `tools/gkd_make_frozen.py` | rechnet `gkd_frozen.json` und prüft sie (`--check`) |
| `tools/gkd_messung.py`, `tools/out_gkd_messung.txt` | Messskript der README und seine Ausgabe |
| `requirements-dev.txt`, `requirements.txt` | Laufzeit plus pytest und scikit-learn (nur Tests); Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Handbeispiel, exakte Rechnung, Orakel, Eigenschaften, Randfälle, Verdikte, Zahl gegen Test, Presets, Layout, AppTest-Smoke-Test, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

Stabilisierte Rechnung im Flat Limit (RBF-QR, Driscoll/Fornberg, Wathen/Zhu: nur als Verweis, nicht als Befund), die Matérn-Kondition und die geschlossene Kondition der Ornstein-Uhlenbeck-Matrix, die Bedingung von Terenin u. a. als Test, ein Rundungsboden in 2D als Leiter, ein Gitter-Regler für die Fläche (feste 7 × 6 Zellen), ein PDF-Export und alles ohne Zählung über Punktmengen. Kein Vergleich von Normalgleichung gegen QR (dazu `ols-projektion-demo`), kein Nachbau von Diskrepanzprinzip, GCV und L-Kurve (`parameterwahl-demo`), kein Nachbau von SMO und KKT (`ocsvm-demo`).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html).
