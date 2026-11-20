"""Rechnungen der Experimente (gkd_experiments.py): Aufbau, Zählungen, Determinismus, Zusammenhang mit der Zusammenfassung; kleine Einstellungen."""

import math

import numpy as np
import pytest

import gkd_constants as C
import gkd_evaluation as V
import gkd_experiments as X


def test_falle_instance_is_deterministic_and_the_seeds_differ_per_index():
    a = X.falle_instance(10, 1.0, 0.0, 35, 0)
    b = X.falle_instance(10, 1.0, 0.0, 35, 0)
    c = X.falle_instance(10, 1.0, 0.0, 35, 1)
    assert a == b and a != c
    assert X.falle_seeds(35, 0) != X.falle_seeds(35, 1) and X.falle_seeds(35, 0) != X.falle_seeds(36, 0)


def test_falle_zaehlung_has_m_instances_and_a_well_conditioned_case_succeeds_everywhere():
    ins = X.falle_zaehlung(12, 0.5, 0.01, 8, 35)
    s = V.summarize_falle(ins, 12)
    assert len(ins) == 8 and s.m == 8 and s.ok == 8 and s.bad_10 == 0 and s.bad_3 == 0
    assert s.max_err < 1e-10 and s.max_eta < 1e-14 and s.pred_bad == 0 and s.cond_lower == 0
    assert all(i.cholesky_ok and not i.cond_lower_bound for i in ins)


def test_falle_in_the_trap_region_has_failures_or_large_errors_and_small_eta():
    ins = X.falle_zaehlung(20, 5.0, 0.0, 12, 35)
    s = V.summarize_falle(ins, 20)
    assert s.cond_lower == s.m                                      # bei σ = 5 liegt der kleinste Eigenwert unter der Rundungsgrenze
    assert s.ok < s.m or s.bad_10 > 0                               # entweder scheitert die Zerlegung, oder die Lösung liegt daneben
    assert s.max_eta < 1e-14                                        # exakter Rückwärtsfehler unter der Rundungsgenauigkeit
    for i in ins:
        if not i.cholesky_ok:
            assert math.isnan(i.err_chol) and math.isnan(i.eta_chol) and math.isnan(i.pred_chol)


def test_summary_counts_follow_the_thresholds_on_handset_instances():
    def inst(ok, err, eta=1e-17, pred=0.0, singular=False, lower=False, n_real=10):
        return X.FalleInstance(singular, ok, err if ok else math.nan, 1.0, 1.0, eta if ok else math.nan, 1.0, pred if ok else math.nan, 1.0, 1e5, lower, n_real, (1.0,))
    s = V.summarize_falle([inst(True, 0.5, pred=0.4), inst(True, 0.1), inst(True, 0.0999), inst(True, 1e-3), inst(True, 1e-4), inst(False, 0.0), inst(True, 0.2, singular=True)], 10)
    assert (s.m, s.excluded, s.ok) == (6, 1, 5)
    assert s.bad_10 == 1 and s.bad_3 == 3                          # nur > 0,1 und > 1e-3 zählen (0,1 und 1e-3 selbst nicht)
    assert s.pred_bad == 1 and s.median_err == pytest.approx(0.0999)


def test_nugget_reihe_structure():
    rows = X.nugget_reihe(10, 2.0, 4, 35)
    assert [r.lam for r in rows] == list(C.NUGGET_LAMS)
    for r in rows:
        assert r.count == 4 and r.cond_violations == 0 and r.digits_violations == 0 and r.digits_min <= r.digits_median
        assert r.bound == pytest.approx(1 + 10 / r.lam) and r.rule == pytest.approx(16 - math.log10(1 + 10 / r.lam))


def test_nugget_testfehler_and_solver_vs_model_shapes():
    rows, const = X.nugget_testfehler(15, 2.0, 35, seeds=4)
    assert [r.lam for r in rows] == list(C.NUGGET_TEST_LAMS) and const > 0.5
    assert all(r.median <= r.maximum and r.mean <= r.maximum for r in rows)
    srows, sconst = X.solver_vs_model(12, 2.0, 35, seeds=3, n_test=30)
    assert [r.lam for r in srows] == [0.0, 1e-8, 1e-4] and sconst > 0.5
    for r in srows[1:]:
        assert r.exact_mean == pytest.approx(r.lu_mean, rel=1e-3)   # bei λ = 1e-8 und 1e-4 ändert der Löser den Testfehler nicht (hier n = 12, σ = 2)


def test_ridge_set_has_thirty_instances_and_is_a_set_not_the_number():
    rs = X.ridge_set(1.0, 20, 1e-13, 35, 0)
    s = V.summarize_ridge(rs)
    assert len(rs.interp) == len(rs.loo) == len(rs.const) == len(rs.cut) == len(rs.chosen_lambda) == C.RIDGE_INSTANZEN == s.count
    assert set(rs.chosen_lambda) <= set(C.LOO_GRID)
    assert X.ridge_set(1.0, 20, 1e-13, 35, 0) == rs and X.ridge_set(1.0, 20, 1e-13, 35, 1) != rs      # anderer Satz, andere Zahlen
    assert X.ridge_instance_seed(35, 0, 0) != X.ridge_instance_seed(35, 1, 0)


def test_ridge_loo_beats_interpolation_and_sigma_small_is_not_catastrophic():
    s1 = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-13, 35, 0))
    assert s1.interp_mean > 100 * s1.const_mean and s1.above_const == s1.count and s1.loo_mean < 0.2
    s03 = V.summarize_ridge(X.ridge_set(0.3, 60, 1e-13, 35, 0))
    assert s03.interp_median < s03.const_mean < s03.interp_mean * 3                                      # σ = 0,3: Median unter der Konstanten
    assert s03.above_const < s03.count


def test_truncation_threshold_changes_the_ridgeless_result():
    a = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-13, 35, 0))
    b = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-5, 35, 0))
    assert b.interp_mean < a.interp_mean / 100 and b.cut_median > a.cut_median                           # die Zahl ist ein Objekt der Abschneidegrenze


def test_surface_shapes_best_cell_and_constants():
    sf = X.surface(60, 0.3, 6, 35)
    assert len(sf.test_mean) == len(C.FLAECHE_SIGMAS) and all(len(r) == len(C.FLAECHE_LAMS) for r in sf.test_mean) and sf.seeds == 6
    bi, bj = sf.best_cell
    assert sf.test_mean[bi][bj] == min(min(r) for r in sf.test_mean)
    assert sf.hindsight_mean <= sf.loo_mean and sf.hindsight_mean <= sf.test_mean[bi][bj]
    assert sf.const_mean == pytest.approx(0.78, abs=0.08) and np.all(np.isfinite(np.array(sf.test_mean)))
    flat = [v for row in sf.test_mean for v in row]
    assert max(flat) > 5 * sf.const_mean                              # das Tal hat einen Rand weit über der konstanten Vorhersage
    assert sf.test_mean[-1][0] < sf.const_mean / 4                    # große Bandbreite bei kleinem λ ist keine Unteranpassung


def test_surface_without_noise_is_better_than_with_noise_at_the_best_cell():
    quiet, noisy = X.surface(60, 0.0, 4, 35), X.surface(60, 0.3, 4, 35)
    assert quiet.test_mean[quiet.best_cell[0]][quiet.best_cell[1]] < noisy.test_mean[noisy.best_cell[0]][noisy.best_cell[1]]
