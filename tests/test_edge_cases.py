"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1 und n = 2, doppelte Punkte, λ = 0 und λ = 1e-300, σ sehr klein und sehr groß,
exakt singuläre Matrix, Regler an den Rändern, leere Zusammenfassungen."""

import math

import numpy as np
import pytest

import gkd_constants as C
import gkd_evaluation as V
import gkd_exact as E
import gkd_experiments as X
import gkd_fit as Fit
import gkd_flat as Fl
import gkd_kernel as Kn
import gkd_scenario as S


def test_exactly_one_point_k_is_the_one_by_one_matrix_with_cond_one():
    K = Kn.gauss_gram(np.array([[0.3, 0.4]]), np.array([[0.3, 0.4]]), 1.0)
    assert K.shape == (1, 1) and K[0, 0] == 1.0
    cond, lower = Kn.condition(Kn.eigenvalues(K))
    assert cond == 1.0 and not lower
    assert E.exact_solve_fractions(K, [2.5]) == [2.5]
    ins = X.falle_instance(1, 1.0, 0.0, 35, 0)
    assert ins.cholesky_ok and ins.err_chol == 0.0 and ins.n_real == 1


def test_exactly_two_points_the_smallest_case_with_something_to_compare():
    x = np.array([[0.0, 0.0], [1.0, 0.0]])
    K = Kn.gauss_gram(x, x, 1.0)
    a = E.exact_solve(K, np.array([1.0, -1.0]))
    assert K @ a == pytest.approx([1.0, -1.0], abs=1e-15)
    assert Fit.relative_error(Fit.solve_cholesky(K, np.array([1.0, -1.0])), a) < 1e-14


def test_two_equal_points_give_an_exactly_singular_matrix_and_the_exact_solver_says_so():
    x = np.array([[0.3, 0.3], [0.3, 0.3]])
    K = Kn.gauss_gram(x, x, 1.0)
    with pytest.raises(ZeroDivisionError):
        E.exact_solve_fractions(K, np.array([1.0, 2.0]))
    assert np.linalg.eigvalsh(K)[0] == pytest.approx(0.0, abs=1e-15)


def test_singular_exact_matrix_is_excluded_from_the_count_not_counted_as_an_error():
    ins = X.falle_instance(10, 1e9, 0.0, 35, 0)                                 # σ = 1e9: alle Einträge sind 1,0 (exakt), also eine Eins-Matrix, exakt singulär
    assert ins.exact_singular and math.isnan(ins.err_chol) and math.isnan(ins.eta_chol)
    s = V.summarize_falle([ins, ins], 10)
    assert (s.m, s.excluded, s.ok) == (0, 2, 0)
    assert V.verdict_falle(s, 1e9, 0.0)[0] == "warning" and V.verdict_eigenwerte(s, 1e9)[0] == "info" and V.verdict_vorhersage(s)[0] == "info"
    assert V.paths_table([ins])[0]["Median des Fehlers"] == "kein Ergebnis"


def test_empty_zaehlung_gives_an_empty_summary_without_error():
    s = V.summarize_falle([], 20)
    assert s.m == 0 and s.ok == 0 and math.isnan(s.median_err)
    assert V.verdict_falle(s, 5.0, 0.0)[0] == "warning"


@pytest.mark.parametrize("lam", [0.0, 1e-300])
def test_lambda_zero_and_lambda_1e_minus_300_change_nothing_numerically(lam):
    X_ = S.points_square(8, 3)
    K = Kn.gauss_gram(X_, X_, 0.7)
    assert np.array_equal(K + lam * np.eye(8), K)                               # 1e-300 verschwindet neben den Einträgen
    cond, lower = Kn.condition(Kn.eigenvalues(K), lam)
    assert cond > 1 and not lower
    assert V.summarize_falle([X.falle_instance(8, 0.7, lam, 35, 0)], 8).ok == 1


def test_sigma_very_small_gives_the_identity_and_sigma_very_large_does_not_crash():
    X_ = S.points_square(10, 4)
    K = Kn.gauss_gram(X_, X_, 1e-3)
    assert np.array_equal(K, np.eye(10))
    assert Kn.condition(Kn.eigenvalues(K)) == (1.0, False)
    ins = X.falle_instance(10, 1e-3, 0.0, 35, 0)
    assert ins.cholesky_ok and ins.err_chol == 0.0 and ins.cond == 1.0 and ins.n_real == 10
    big = X.falle_instance(10, 20.0, 0.0, 35, 0)                                # größte Bandbreite des Reglers: kein Absturz, Untergrenze der Konditionszahl
    assert big.cond_lower_bound and big.n_real < 10 and not big.exact_singular


def test_nugget_helps_at_the_largest_bandwidth():
    ins = X.falle_instance(20, 20.0, 1e-6, 35, 0)
    assert ins.cholesky_ok and ins.err_chol < 1e-8 and not ins.cond_lower_bound


def test_regler_corners_of_every_experiment_compute():
    for n, sg, lam in ((C.N_FALLE_MIN, C.SIGMA_MIN, C.LAMS[0]), (C.N_FALLE_MAX, C.SIGMA_MAX, C.LAMS[-1]), (C.N_FALLE_MIN, C.SIGMA_MAX, C.LAMS[1])):
        ins = X.falle_zaehlung(n, sg, lam, 2, 35)
        assert len(ins) == 2 and V.summarize_falle(ins, n).m + V.summarize_falle(ins, n).excluded == 2
    for sg in (C.SIGMAS_RIDGE[0], C.SIGMAS_RIDGE[-1]):
        for n in (C.NS_RIDGE[0], C.NS_RIDGE[-1]):
            for tau in (C.TAUS[0], C.TAUS[-1]):
                rs = X.ridge_set(sg, n, tau, 35, 0, count=2)
                assert len(rs.interp) == 2 and all(math.isfinite(v) for v in rs.interp + rs.loo)
    sf = X.surface(C.N_FLAECHE_MIN, C.NOISE_MIN, C.S_FLAECHE_MIN, 35)
    assert sf.seeds == C.S_FLAECHE_MIN and sf.test_mean[sf.best_cell[0]][sf.best_cell[1]] < 0.2
    for k in (C.K_FLAT_MIN, C.K_FLAT_MAX):
        for n in (C.N_FLAT_MIN, C.N_FLAT_MAX):
            assert Fl.analyse_flat(tuple(range(n)), k).cond_exact > 1


def test_tables_work_without_frozen_numbers_and_with_a_missing_file():
    assert V.tau_table({}) == []
    assert "fehlt" in V.verdict_streuung({})[1] and "fehlt" in V.verdict_exakt_gleit({})[1]
