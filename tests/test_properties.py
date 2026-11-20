"""Eigenschaftstests (je 20 bis 40 Zufallsinstanzen, feste Zufalls-Seeds; nur belegte Aussagen, ORACLE-PLAYBOOK §5).

Nicht als Eigenschaft behauptet: „cond wächst in σ“ (für feste Punkte unbewiesen; nur als Messung in der README)."""

import numpy as np
import pytest

import gkd_exact as E
import gkd_kernel as Kn
import gkd_scenario as S

CASES = [(6 + s % 14, (0.3, 0.8, 2.0)[s % 3], s) for s in range(30)]


@pytest.mark.parametrize("n,sigma,seed", CASES)
def test_cond_falls_in_lambda_and_stays_under_one_plus_n_over_lambda(n, sigma, seed):
    X = S.points_square(n, 1000 + seed)
    ev = Kn.eigenvalues(Kn.gauss_gram(X, X, sigma))
    prev = None
    for lam in (1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        c, _ = Kn.condition(ev, lam)
        assert c <= Kn.nugget_cond_bound(n, lam) * (1 + 1e-9)
        assert prev is None or c <= prev * (1 + 1e-12)                                   # monoton fallend in λ (folgt aus (a + λ)/(b + λ))
        prev = c


@pytest.mark.parametrize("n,sigma,seed", CASES)
def test_trace_is_n_and_the_matrix_is_positive_semidefinite_up_to_rounding(n, sigma, seed):
    X = S.points_square(n, 2000 + seed)
    K = Kn.gauss_gram(X, X, sigma)
    assert np.trace(K) == n and np.array_equal(K, K.T)
    assert Kn.eigenvalues(K)[-1] >= -n * Kn.EPS * 1e2


@pytest.mark.parametrize("n,sigma,seed", CASES[:20])
def test_eigenvalues_are_invariant_under_reordering_shift_and_joint_scaling(n, sigma, seed):
    X = S.points_square(n, 3000 + seed)
    ev = Kn.eigenvalues(Kn.gauss_gram(X, X, sigma))
    perm = np.array(sorted(range(n), key=lambda i: (i * 7919 + seed) % n))                 # feste Umordnung ohne Zufallsquelle
    assert np.allclose(ev, Kn.eigenvalues(Kn.gauss_gram(X[perm], X[perm], sigma)), rtol=1e-9, atol=1e-12 * ev[0])
    Y = X + np.array([3.0, -2.0])                                                        # Verschiebung aller Punkte: gleiche Abstände
    assert np.allclose(ev, Kn.eigenvalues(Kn.gauss_gram(Y, Y, sigma)), rtol=1e-9, atol=1e-12 * ev[0])
    Z = 2.5 * X                                                                          # Skalierung von Punkten UND σ mit demselben Faktor
    assert np.allclose(ev, Kn.eigenvalues(Kn.gauss_gram(Z, Z, 2.5 * sigma)), rtol=1e-9, atol=1e-12 * ev[0])


@pytest.mark.parametrize("seed", range(10))
def test_a_doubled_point_makes_the_matrix_singular(seed):
    X = S.points_square(8, 4000 + seed)
    X = np.vstack([X, X[3]])
    ev = Kn.eigenvalues(Kn.gauss_gram(X, X, 0.7))
    assert ev[-1] <= len(X) * Kn.EPS * ev[0] * 10
    with pytest.raises(ZeroDivisionError):
        E.exact_solve_fractions(Kn.gauss_gram(X, X, 0.7), np.ones(len(X)))              # exakt singulär: die exakte Elimination meldet es


@pytest.mark.parametrize("n,sigma,seed", CASES[:15])
def test_with_a_nugget_the_solution_is_unique_and_the_exact_solution_has_zero_residual(n, sigma, seed):
    X = S.points_square(n, 5000 + seed)
    y = S.normal_targets(n, 6000 + seed)
    A = Kn.gauss_gram(X, X, sigma) + 0.1 * np.eye(n)
    assert Kn.eigenvalues(A)[-1] >= 0.1 * (1 - 1e-9)                                     # K + λI ist positiv definit: genau eine Lösung
    sol = E.exact_solve_fractions(A, y)
    from fractions import Fraction
    for i in range(n):
        assert sum(Fraction(float(A[i, j])) * sol[j] for j in range(n)) == Fraction(float(y[i]))   # Rest exakt null (Brüche)
