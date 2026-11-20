"""Gram-Matrix, Eigenwerte, Konditionszahl, Pivots und Nugget-Schranke (gkd_kernel.py): Handrechnung, Fremdreferenz (scikit-learn, numpy) und Eigenschaften."""

import math

import numpy as np
import pytest

import gkd_kernel as Kn
import gkd_scenario as S


def _points(n, d, seed):
    return S.points_square(n, seed)[:, :d] if d <= 2 else np.hstack([S.points_square(n, seed), S.points_square(n, seed + 1)[:, :1]])


def test_gram_matrix_of_the_mini_instance_by_hand():
    sigma = 1.0 / math.sqrt(2.0 * math.log(2.0))                  # q = exp(-1/(2σ²)) = 1/2
    K = Kn.gauss_gram(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]), sigma)
    expected = np.array([[1, 0.5, 1 / 16], [0.5, 1, 0.5], [1 / 16, 0.5, 1]])
    assert np.max(np.abs(K - expected)) < 1e-15
    assert Kn.gamma_of_sigma(sigma) == pytest.approx(math.log(2.0), rel=1e-14)


@pytest.mark.parametrize("d", [1, 2, 3])
def test_gram_matrix_equals_scikit_learn_rbf_kernel(d):
    metrics = pytest.importorskip("sklearn.metrics.pairwise")
    X = _points(12, d, 5)
    Y = _points(7, d, 9)
    for sigma in (0.3, 1.0, 5.0):
        ref = metrics.rbf_kernel(X, Y, gamma=Kn.gamma_of_sigma(sigma))
        assert np.max(np.abs(Kn.gauss_gram(X, Y, sigma) - ref)) < 1e-13


def test_gram_matrix_accepts_vectors_as_one_dimensional_points():
    x = np.array([0.0, 1.0, 3.0])
    assert np.allclose(Kn.gauss_gram(x, x, 2.0), Kn.gauss_gram(x[:, None], x[:, None], 2.0), atol=0, rtol=0)


def test_sigma_must_be_positive():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            Kn.gauss_gram(np.zeros((2, 1)), np.zeros((2, 1)), bad)
        with pytest.raises(ValueError):
            Kn.gauss_from_sqdist(np.zeros((2, 2)), bad)


def test_gauss_from_sqdist_is_the_same_matrix_bit_for_bit():
    X = _points(9, 2, 3)
    assert np.array_equal(Kn.gauss_gram(X, X, 0.7), Kn.gauss_from_sqdist(Kn.sqdist(X, X), 0.7))


def test_condition_of_the_mini_instance_by_hand():
    ev = np.array([(33 + math.sqrt(513)) / 32, 15 / 16, (33 - math.sqrt(513)) / 32])
    cond, lower = Kn.condition(ev)
    assert cond == pytest.approx(5.376, abs=1e-3) and not lower
    cond2, _ = Kn.condition(ev, 0.1)
    assert cond2 == pytest.approx((ev[0] + 0.1) / (ev[-1] + 0.1), rel=1e-15)


@pytest.mark.parametrize("sigma,lam", [(0.2, 0.0), (0.5, 0.01), (1.0, 1e-4), (0.3, 1.0)])
def test_condition_from_eigenvalues_equals_numpy_cond(sigma, lam):
    X = _points(30, 2, 21)
    K = Kn.gauss_gram(X, X, sigma)
    cond, lower = Kn.condition(Kn.eigenvalues(K), lam)
    assert not lower
    assert cond == pytest.approx(np.linalg.cond(K + lam * np.eye(30)), rel=1e-9)


def test_condition_is_only_a_lower_bound_below_the_rounding_floor():
    X = _points(20, 2, 4)
    cond, lower = Kn.condition(Kn.eigenvalues(Kn.gauss_gram(X, X, 5.0)))
    assert lower and cond == pytest.approx(Kn.INV_EPS, rel=1e-12)          # (λ_max)/(eps λ_max) = 1/eps = 4,5e15
    assert Kn.INV_EPS == pytest.approx(4.5e15, rel=0.01)
    c2, l2 = Kn.condition(np.array([1.0, 1e-16]))                          # genau im Bereich der Rundung
    assert l2 and c2 == pytest.approx(Kn.INV_EPS)
    c3, l3 = Kn.condition(np.array([1.0, 1e-10]), 0.0)
    assert not l3 and c3 == pytest.approx(1e10)


def test_nugget_bound_and_digits_rule_by_hand():
    assert Kn.nugget_cond_bound(20, 1e-10) == pytest.approx(2e11 + 1, rel=1e-15)
    assert Kn.digits_rule(20, 1e-10) == pytest.approx(16 - math.log10(2e11 + 1))
    assert round(Kn.digits_rule(20, 1e-10), 2) == 4.70 and round(Kn.digits_rule(20, 1e-6), 2) == 8.70 and round(Kn.digits_rule(20, 1e-2), 2) == 12.70
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            Kn.nugget_cond_bound(20, bad)


def test_ldl_pivots_equal_the_squared_cholesky_diagonal(mini_gram):
    K = np.array([[float(v) for v in row] for row in mini_gram])
    assert Kn.ldl_pivots(K) == pytest.approx([1.0, 0.75, 45 / 64], abs=1e-15)
    assert np.max(np.abs(Kn.ldl_pivots(K) - np.diag(np.linalg.cholesky(K)) ** 2)) < 1e-15
    X = _points(15, 2, 8)
    Kg = Kn.gauss_gram(X, X, 0.6)
    assert np.max(np.abs(Kn.ldl_pivots(Kg) - np.diag(np.linalg.cholesky(Kg)) ** 2)) < 1e-13


def test_ldl_pivots_show_where_a_matrix_stops_being_positive_definite():
    piv = Kn.ldl_pivots(np.array([[1.0, 2.0], [2.0, 1.0]]))
    assert piv[0] == 1.0 and piv[1] == -3.0
    assert Kn.cholesky_or_none(np.array([[1.0, 2.0], [2.0, 1.0]])) is None
    zero = Kn.ldl_pivots(np.array([[0.0, 1.0], [1.0, 0.0]]))
    assert zero[0] == 0.0 and np.isnan(zero[1])


def test_cholesky_or_none_returns_the_factor_when_it_works(mini_gram):
    K = np.array([[float(v) for v in row] for row in mini_gram])
    L = Kn.cholesky_or_none(K)
    assert np.max(np.abs(L @ L.T - K)) < 1e-15
