"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zusammenfassungen, nicht Zufalls-Seeds. Je Zweig und je Zahl eine Aussage."""

import gkd_constants as C
import gkd_evaluation as V
import gkd_experiments as X
from gkd_flat import FlatResult
from gkd_format import de, lam_text, sci


def _summary(**kw):
    base = dict(m=30, excluded=0, ok=17, bad_10=14, bad_3=17, median_err=0.46, max_err=7.3, max_eta=6.1e-17, median_err_lu=0.5, pred_bad=17, cond_lower=30, median_cond=4.5e15, median_real=9.0, n=20)
    base.update(kw)
    return V.FalleSummary(**base)


# ---------------------------------------------------------------- Kernfrage 1
def test_eigenvalue_verdict_all_real_and_lost():
    kind, text = V.verdict_eigenwerte(_summary(median_real=20.0), 0.5)
    assert kind == "success" and "Alle 20 Eigenwerte" in text and "σ = 0,5" in text and "30 Punktmengen" in text and "kein Eigenwert unter der Rundungsgrenze" in text
    kind, text = V.verdict_eigenwerte(_summary(median_real=9.0), 5.0)
    assert kind == "warning" and "nur noch 9 über 1e-16" in text and "die übrigen 11 sind" in text and "Von 20 Eigenwerten" in text and "σ = 5" in text
    assert V.verdict_eigenwerte(_summary(median_real=19.5), 1.0)[0] == "warning"        # 19,5 wird auf 19 abgerundet: ein Eigenwert verloren


# ---------------------------------------------------------------- Kernfrage 2
def test_trap_verdict_warns_with_counts_median_maximum_and_eta():
    kind, text = V.verdict_falle(_summary(), 5.0, 0.0)
    assert kind == "warning"
    for part in ("n = 20 Punkte", "σ = 5", "λ = 0", "30 Punktmengen", "Cholesky gelingt in 17 von 30", "in 14 davon", "mehr als 10 %", "Median des Fehlers 0,460", "Stichprobenmaximum 7,30",
                 "höchstens 6,1e-17", "das Residuum verrät den Fehler nicht"):
        assert part in text, part


def test_trap_verdict_thresholds_and_branches():
    assert V.verdict_falle(_summary(bad_10=1), 5.0, 0.0)[0] == "warning"                # eine Punktmenge über 10 % genügt
    kind, text = V.verdict_falle(_summary(bad_10=0, bad_3=4, median_err=2e-3, max_err=0.05), 3.0, 0.0)
    assert kind == "info" and "in 4 davon weicht die Lösung um mehr als 1e-3 ab, in keiner um mehr als 10 %" in text and "2,0e-3" in text and "5,0e-2" in text
    kind, text = V.verdict_falle(_summary(ok=30, bad_10=0, bad_3=0, median_err=2e-10, max_err=3e-10, max_eta=5e-17), 5.0, 1e-6)
    assert kind == "success" and "30 von 30" in text and "höchstens 3,0e-10" in text and "λ = 1e-6" in text
    kind, text = V.verdict_falle(_summary(ok=0, bad_10=0, bad_3=0, median_err=float("nan"), max_err=float("nan"), max_eta=float("nan")), 20.0, 0.0)
    assert kind == "info" and "scheitert in allen 30 Punktmengen" in text and "meldet das selbst" in text
    kind, text = V.verdict_falle(_summary(m=0, ok=0, bad_10=0, bad_3=0), 5.0, 0.0)
    assert kind == "warning" and "exakt singulär" in text
    big_eta = V.verdict_falle(_summary(max_eta=1e-9), 5.0, 0.0)[1]
    assert "verrät den Fehler nicht" not in big_eta and "1,0e-9" in big_eta                # ein großes η wird nicht als „unauffällig“ verkauft


def test_prediction_verdict():
    assert V.verdict_vorhersage(_summary(ok=0))[0] == "info"
    kind, text = V.verdict_vorhersage(_summary(pred_bad=0, ok=30))
    assert kind == "success" and "In keiner der 30 gelungenen Zerlegungen" in text and f"{C.N_TEST_PRED} Testpunkten" in text
    kind, text = V.verdict_vorhersage(_summary(pred_bad=17, ok=17))
    assert kind == "warning" and "In 17 von 17 gelungenen Zerlegungen" in text and "bestätigt sich hier nicht" in text


# ---------------------------------------------------------------- Kernfrage 3
def _rows(violations=0, digits=0):
    return (X.NuggetRow(1e-10, 2e11 + 1, 4.699, 30, violations, digits, 5.43, 5.6, 1.9e11),
            X.NuggetRow(1e-6, 2e7 + 1, 8.699, 30, 0, 0, 9.44, 9.6, 2e7), X.NuggetRow(1e-2, 2001.0, 12.699, 30, 0, 0, 13.39, 13.6, 2000.0))


def test_nugget_verdict_success_and_warning():
    kind, text = V.verdict_nugget(_rows(), 20, 5.0)
    assert kind == "success" and "n = 20, σ = 5, 90 Instanzen (3 Werte von λ)" in text and "in keiner Instanz verletzt" in text
    assert "λ = 1e-10: mindestens 5,43 Stellen (R = 4,70)" in text and "λ = 1e-6: mindestens 9,44 Stellen (R = 8,70)" in text and "λ = 0,01: mindestens 13,4 Stellen (R = 12,7)" in text
    kind, text = V.verdict_nugget(_rows(violations=2, digits=1), 20, 5.0)
    assert kind == "warning" and "in 2 Instanzen verletzt" in text and "in 1 Instanzen unter R(λ)" in text
    assert V.verdict_nugget(_rows(digits=1), 20, 5.0)[0] == "warning"


def test_nugget_model_verdict_names_every_lambda_and_the_constant():
    rows = (X.ModelRow(1e-8, 0.64, 0.30, 8.8), X.ModelRow(1e-4, 0.18, 0.16, 0.42), X.ModelRow(1e-2, 0.29, 0.27, 0.42), X.ModelRow(0.1, 0.34, 0.32, 0.47))
    kind, text = V.verdict_nugget_modell(rows, 0.78, 40, 5.0, 30)
    assert kind == "info" and "bei λ = 1e-8 / 1e-4 / 0,01 / 0,1 bei 0,640 (0,300) / 0,180 (0,160) / 0,290 (0,270) / 0,340 (0,320)" in text and "konstante Vorhersage liegt bei 0,780" in text
    assert "σ = 5, n = 40" in text and "30 Zufalls-Seeds" in text


def test_solver_vs_model_verdict():
    rows = (X.SolverVsModel(0.0, 3.5e4, 1.5e3, 3.6e5, 6.7e3), X.SolverVsModel(1e-8, 1.68, 0.21, 1.68, 0.21), X.SolverVsModel(1e-4, 0.204, 0.134, 0.204, 0.134))
    kind, text = V.verdict_solver_vs_model(rows, 0.78, 40, 5.0, 10)
    assert kind == "info" and "bei λ = 1e-8 und 1e-4 ändert der Löser den Mittelwert des Testfehlers nicht" in text
    assert "bei λ = 0 liegen exakte Lösung und LU im Mittel bei 3,5e4 und 3,6e5 (Median 1500 und 6700), beide über der konstanten Vorhersage (0,780)" in text
    rows2 = (X.SolverVsModel(0.0, 0.5, 0.5, 0.6, 0.6), X.SolverVsModel(1e-8, 1.0, 1.0, 2.0, 2.0))
    text2 = V.verdict_solver_vs_model(rows2, 0.78, 40, 5.0, 10)[1]
    assert "unterscheiden sich exakte Lösung und LU im Testfehler" in text2 and "nicht beide über" in text2


# ---------------------------------------------------------------- Experiment 4
def _flat(real_float=3, real_rounded=3, predicted=2, neg=False, ratios=(1.0, 1.0, 1.0)):
    return FlatResult(xs=(0, 1, 2), k=8, sigma=7071.07, eps2=2e-8, exact=(3.0, 4e-8, 1e-16), rounded=(3.0, 4e-8, 1e-16), floating=(3.0, 4e-8, 1e-16), ladder=(3.0, 4e-8, 1e-16),
                      coefficients=(3, 2, 1), ratio_exact_ladder=ratios, real_rounded=real_rounded, real_float=real_float, predicted=predicted, smallest_rounded_negative=neg,
                      smallest_float_negative=neg, pivots_float=(1.0, 0.1, 0.01), pivot_negative_at=None, cholesky_ok=True, cond_exact=3e16, cond_ladder=3e16)


def test_flat_verdict_counts_and_negative_remark():
    kind, text = V.verdict_flat(_flat())
    assert kind == "success" and "x = (0, 1, 2)" in text and "q = 1 − 1e-8" in text and "σ ≈ 7071" in text and "3 echt" in text and "gerundeten Matrix 3" in text and "sagt 2 voraus" in text
    kind, text = V.verdict_flat(_flat(real_float=2, neg=True, ratios=(1.0, 1.0, 1.0 + 3e-6)))
    assert kind == "warning" and "2 echt" in text and "größte Abweichung des Verhältnisses von 1 gleich 3,0e-6" in text and "nicht mehr positiv semidefinit" in text
    assert "nicht mehr positiv semidefinit" not in V.verdict_flat(_flat())[1]


def test_flat_hint_equal_and_different_counts():
    assert "verliert gegenüber der gerundeten Matrix nichts zusätzlich (3 echte Eigenwerte in beiden)" in V.verdict_flat_hinweis(_flat())[1]
    text = V.verdict_flat_hinweis(_flat(real_float=2, real_rounded=3))[1]
    assert "Gleitkomma-Eigenzerlegung 2 echte Eigenwerte, gerundete Matrix 3" in text and "um 1 Eigenwert(e)" in text


# ---------------------------------------------------------------- Experiment 5
def _ridge(mean=3000.0, median=900.0, mx=3e4, const=0.8, above=30, loo=0.05):
    return V.RidgeSummary(const_mean=const, above_const=above, count=30, cut_median=0.0, lambda_counts=((0.01, 5), (0.1, 25)), interp_mean=mean, interp_median=median, interp_max=mx,
                          loo_mean=loo, loo_median=loo / 2, loo_max=loo * 4)


def test_ridge_verdict_interpolation_branches():
    kind, text = V.verdict_ridge(_ridge(), 1.0, 60, 1e-13, C.WAHLEN[0])
    assert kind == "warning"
    for part in ("n = 60 Punkte", "σ = 1", "ein Satz von 30 Zufalls-Seeds", "τ = 1e-13", "im Median 0 von 60", "Mittel 3000, Median 900, Maximum 3e4", "gegen 0,800 der konstanten Vorhersage",
                 "über der konstanten Vorhersage in 30 von 30 Zufalls-Seeds", "Median bei 0,0250"):
        assert part in text, part
    kind, text = V.verdict_ridge(_ridge(mean=0.6, median=0.36, mx=2.0, above=4), 0.3, 60, 1e-13, C.WAHLEN[0])
    assert kind == "info" and "nicht schlechter als die Konstante" in text and "4 von 30" in text and "σ = 0,3" in text


def test_ridge_verdict_loo_names_the_chosen_lambda_counts():
    kind, text = V.verdict_ridge(_ridge(), 1.0, 60, 1e-13, C.WAHLEN[1])
    assert kind == "info" and "gewählt: 0,01 in 5, 0,1 in 25" in text and "Mittel 0,0500, Median 0,0250, Maximum 0,200" in text and "konstante Vorhersage 0,800" in text
    assert "Interpolation derselben Instanzen (τ = 1e-13) liegt bei Mittel 3000" in text


def test_streuung_and_exakt_gleit_verdicts_use_the_frozen_numbers():
    z = {"sets": 300, "per_set": 30, "median_lo": 197.0, "median_hi": 1232.0, "median_med": 457.0, "mean_lo": 1080.0, "mean_hi": 87650.0, "above_const": 9000, "loo_median_med": 0.0533}
    kind, text = V.verdict_streuung({"streuung": z})
    assert kind == "info" and "300 frische Sätze zu je 30 Zufalls-Seeds (9000 Instanzen" in text and "197 bis 1230 (95-%-Band, Median der Sätze 457)" in text
    assert "Satz-Mittel 1080 bis 8,76e4" in text and "in 9000 von 9000 Instanzen" in text and "Satz-Median bei 0,053" in text
    assert "fehlt" in V.verdict_streuung({})[1] and "fehlt" in V.verdict_exakt_gleit({})[1]
    g = {"seeds": 40, "ratio_close": 39, "ratio_min": 0.99, "ratio_max": 40.0, "norm_median": 1.1e8, "norm_max": 2e11, "label_norm_median": 7.2, "large_norm": 1, "small_norm_close": 39}
    text = V.verdict_exakt_gleit({"exakt_gleit": g})[1]
    assert "in 39 von 40 Zufalls-Seeds denselben Testfehler" in text and "im Median bei 1,1e8" in text and "Norm der Etiketten etwa 7,2" in text
    assert "In 1 von 1 Zufalls-Seeds mit ‖a‖ ≥ 1e10 (größte Norm 2,0e11) weichen exakte und Gleitkomma-Rechnung ab (Verhältnis 0,9900 bis 40,0000 über alle 40 Zufalls-Seeds)" in text and "Regularisierung" in text
    g_same = dict(g, ratio_close=40, small_norm_close=39, ratio_min=0.9998, ratio_max=1.0001)
    text = V.verdict_exakt_gleit({"exakt_gleit": g_same})[1]
    assert "Auch in den 1 Zufalls-Seed(s) mit ‖a‖ ≥ 1e10" in text and "bleibt der Testfehler gleich (Verhältnis 0,9998 bis 1,0001 über alle 40 Zufalls-Seeds)" in text and "Regularisierung" not in text
    g2 = dict(g_same, large_norm=0, small_norm_close=40)
    assert "keinen Zufalls-Seed mit ‖a‖ ≥ 1e10" in V.verdict_exakt_gleit({"exakt_gleit": g2})[1]


# ---------------------------------------------------------------- Experiment 1
def test_surface_verdict_names_best_worst_and_loo():
    sig, lam = C.FLAECHE_SIGMAS, C.FLAECHE_LAMS
    mean = [[0.7] * len(lam) for _ in sig]
    mean[4][2] = 0.05                       # beste Zelle: σ = 1,6, λ = 0,01
    mean[3][0] = 20.0                       # schlechteste: σ = 0,8, λ = 1e-6
    mean[6][0], mean[6][5] = 0.08, 0.68
    sf = X.SurfaceResult(sig, lam, tuple(tuple(r) for r in mean), tuple(tuple(r) for r in mean), 0.78, (4, 2), 0.069, 0.046, 30)
    kind, text = V.verdict_flaeche(sf, 60, 0.3)
    assert kind == "info"
    for part in ("n = 60, Rauschen 0,30, 30 Zufalls-Seeds", "beste Zelle σ = 1,6, λ = 0,01 (Testfehler 0,0500)", "schlechteste Zelle σ = 0,8, λ = 1e-6 mit 20,0, dem 25,6fachen der konstanten Vorhersage (0,780)",
                 "Leave-one-out je Zufalls-Seed gewählt 0,0690", "im Rückblick beste Zelle 0,0460", "σ = 6,4 liegt bei λ = 1e-6 bei 0,0800, bei λ = 10 bei 0,680"):
        assert part in text, part


# ---------------------------------------------------------------- Zahlenformat
def test_number_formats():
    assert de(0.0469) == "0,0469" and de(3308) == "3310" and de(3308, 4) == "3308" and de(911.5, 4) == "911,5" and de(1.2e7) == "1,2e7" and de(87650) == "8,76e4" and de(1230) == "1230" and de(float("nan")) == "–"
    assert sci(6.1e-17) == "6,1e-17" and sci(4.5e15, 1) == "4,5e15" and sci(0.0125, 2) == "1,25e-2"
    assert lam_text(0.0) == "0" and lam_text(1e-10) == "1e-10" and lam_text(0.01) == "0,01" and lam_text(1.0) == "1"
