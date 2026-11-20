"""Rechenwege (gkd_fit.py): Wege gegen die Bruchlösung der Mini-Instanz, Abschneiden, Leave-one-out geschlossen gegen echtes Herausnehmen, Wahlregel."""

from fractions import Fraction

import numpy as np
import pytest

import gkd_exact as E
import gkd_fit as Fit
import gkd_kernel as Kn
import gkd_scenario as S
from oracle_helpers import solve_fractions


def _mini():
    h = Fraction(1, 2)
    K = [[Fraction(1), h, h ** 4], [h, Fraction(1), h], [h ** 4, h, Fraction(1)]]
    lam = Fraction(1, 10)
    A = [[K[i][j] + (lam if i == j else 0) for j in range(3)] for i in range(3)]
    y = [Fraction(1), Fraction(0), Fraction(-1, 2)]
    return K, A, y


def test_three_float_paths_agree_with_the_exact_fraction_solution_on_the_mini_instance():
    K, A, y = _mini()
    a_ex = np.array([float(v) for v in solve_fractions(A, y)])
    An = np.array([[float(v) for v in r] for r in A])
    yn = np.array([float(v) for v in y])
    for name, a in (("LU", Fit.solve_lu(An, yn)), ("Cholesky", Fit.solve_cholesky(An, yn)), ("Eigen", Fit.solve_eigen(An, yn))):
        assert np.max(np.abs(a - a_ex)) < 1e-14, name


def test_paths_report_failure_by_none_not_by_a_wrong_number():
    A = np.array([[1.0, 2.0], [2.0, 1.0]])                       # nicht positiv definit: Cholesky meldet es
    assert Fit.solve_cholesky(A, np.array([1.0, 1.0])) is None
    assert Fit.solve_lu(np.zeros((2, 2)), np.ones(2)) is None    # exakt singulär: LU meldet es
    assert Fit.solve_eigen(np.zeros((2, 2)), np.ones(2)) is None  # Eigenwert null: nicht endlich
    assert Fit.relative_error(None, np.ones(2)) == float("inf")


def test_relative_error_by_hand():
    assert Fit.relative_error(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0
    assert Fit.relative_error(np.array([3.0, 4.0]), np.array([0.0, 4.0])) == pytest.approx(0.75)


def test_truncation_drops_exactly_the_eigenvalues_under_tau_lambda_max():
    K = np.diag([1.0, 1e-4, 1e-8, 1e-12])
    y = np.ones(4)
    a, cut = Fit.interpolate_truncated(K, y, 1e-9)
    assert cut == 1 and a == pytest.approx([1.0, 1e4, 1e8, 0.0])
    a, cut = Fit.interpolate_truncated(K, y, 1e-5)
    assert cut == 2 and a == pytest.approx([1.0, 1e4, 0.0, 0.0])
    a, cut = Fit.interpolate_truncated(K, y, 1e-15)
    assert cut == 0 and a == pytest.approx([1.0, 1e4, 1e8, 1e12])


def test_interpolation_is_the_exact_solution_when_nothing_is_cut():
    X = S.points_square(12, 3)
    K = Kn.gauss_gram(X, X, 0.4)
    y = S.normal_targets(12, 4)
    a, cut = Fit.interpolate_truncated(K, y, 1e-15)
    assert cut == 0 and np.max(np.abs(K @ a - y)) < 1e-9


def test_loo_closed_form_equals_leaving_out_for_real_in_floating_point():
    X, y = S.regression_data(14, 0.3, 17)
    K = Kn.gauss_gram(X, X, 1.0)
    for lam in (1e-2, 1.0):
        e = Fit.loo_residuals(K, y, lam)
        for i in range(14):
            idx = [j for j in range(14) if j != i]
            a = np.linalg.solve(K[np.ix_(idx, idx)] + lam * np.eye(13), y[idx])
            assert e[i] == pytest.approx(y[i] - K[i, idx] @ a, rel=1e-8, abs=1e-10)


def test_loo_closed_form_equals_leaving_out_exactly_in_fractions_on_the_mini_instance():
    K, A, y = _mini()
    cols = [solve_fractions(A, [Fraction(int(i == j)) for i in range(3)]) for j in range(3)]
    Ainv = [[cols[j][i] for j in range(3)] for i in range(3)]
    Ainv_y = [sum(Ainv[i][j] * y[j] for j in range(3)) for i in range(3)]
    closed = [Ainv_y[i] / Ainv[i][i] for i in range(3)]
    real = []
    for i in range(3):
        idx = [j for j in range(3) if j != i]
        a_m = solve_fractions([[A[r][c] for c in idx] for r in idx], [y[r] for r in idx])
        real.append(y[i] - sum(K[i][idx[t]] * a_m[t] for t in range(2)))
    assert closed == real                                         # Gleichheit in Brüchen, exakt
    An = np.array([[float(v) for v in r] for r in K])
    e = Fit.loo_residuals(An, np.array([float(v) for v in y]), 0.1)
    assert e == pytest.approx([float(v) for v in real], abs=1e-13)


def test_hat_diagonal_is_trace_n_over_one_plus_lambda_for_the_identity():
    a, h = Fit.ridge_eigen(np.eye(5), np.arange(5.0), 1.0)
    assert h == pytest.approx([0.5] * 5) and a == pytest.approx(np.arange(5.0) / 2)


def test_lambda_choice_equals_the_argmin_of_real_leave_one_out_in_floating_point():
    grid = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
    for seed in range(6):
        X, y = S.regression_data(16, 0.3, 200 + seed)
        yc = y - y.mean()
        K = Kn.gauss_gram(X, X, 1.0)
        real = []
        for lam in grid:
            err = 0.0
            for i in range(16):
                idx = [j for j in range(16) if j != i]
                a = np.linalg.solve(K[np.ix_(idx, idx)] + lam * np.eye(15), yc[idx])
                err += (yc[i] - K[i, idx] @ a) ** 2
            real.append(err / 16)
        lam, a = Fit.choose_lambda_loo(K, yc, grid)
        assert lam == grid[int(np.argmin(real))], seed
        assert np.max(np.abs((K + lam * np.eye(16)) @ a - yc)) < 1e-8


def test_lambda_choice_takes_the_smaller_lambda_on_a_tie():
    K, y = np.eye(4), np.array([1.0, -1.0, 1.0, -1.0])           # für K = I ist das LOO-Residuum e = y unabhängig von λ: Gleichstand
    lam, a = Fit.choose_lambda_loo(K, y, (1e-6, 1e-2, 1.0))
    assert lam == 1e-6 and a == pytest.approx(y / (1 + 1e-6))


def test_kernel_ridge_equals_scikit_learn_on_random_instances():
    KR = pytest.importorskip("sklearn.kernel_ridge").KernelRidge
    for seed in range(5):
        X, y = S.regression_data(20, 0.3, 100 + seed)
        Xt, _ = S.reg_test_points(15)
        for sigma, lam in ((1.0, 0.1), (0.5, 1e-3)):
            K = Kn.gauss_gram(X, X, sigma)
            a = Fit.solve_lu(K + lam * np.eye(20), y)
            ref = KR(alpha=lam, kernel="rbf", gamma=Kn.gamma_of_sigma(sigma)).fit(X, y).predict(Xt)
            assert np.max(np.abs(Kn.gauss_gram(Xt, X, sigma) @ a - ref)) < 1e-8 * max(1.0, np.max(np.abs(ref)))


def test_exact_solution_is_what_the_float_paths_approximate_at_a_moderate_nugget():
    X = S.points_square(15, 2)
    y = S.normal_targets(15, 3)
    A = Kn.gauss_gram(X, X, 1.0) + 0.01 * np.eye(15)
    a_ex = E.exact_solve(A, y)
    assert Fit.relative_error(Fit.solve_cholesky(A, y), a_ex) < 1e-12
    assert Fit.relative_error(Fit.solve_lu(A, y), a_ex) < 1e-12
    assert Fit.relative_error(Fit.solve_eigen(A, y), a_ex) < 1e-12
