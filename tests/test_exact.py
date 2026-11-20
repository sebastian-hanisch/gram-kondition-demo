"""Exakte Rechnung (gkd_exact.py) gegen Gauß-Elimination in Brüchen, Handrechnung und ein zweites Verfahren: je Einheit eine Mini-Instanz (DEMO-PLAYBOOK §1b)."""

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

import gkd_exact as E
from gkd_rng import SplitMix64
from oracle_helpers import newton_eigenvalues, solve_fractions


def _spd(n, seed):
    """Symmetrisch positiv definite Gleitkomma-Matrix B Bᵀ + I und eine rechte Seite, aus SplitMix64."""
    rng = SplitMix64(seed)
    B = np.array([[rng.random() - 0.5 for _ in range(n)] for _ in range(n)])
    return B @ B.T + np.eye(n), np.array([rng.random() * 4 - 2 for _ in range(n)])


def test_scaled_ints_roundtrip():
    ints, shift = E.scaled_ints([0.5, 3.0, 0.0625, 0.0])
    assert shift == 4 and ints == [8, 48, 1, 0]
    ints, shift = E.scaled_ints([5.0, -2.0])
    assert shift == 0 and ints == [5, -2]


@pytest.mark.parametrize("seed", range(25))
def test_bareiss_equals_fraction_elimination_exactly(seed):
    n = 2 + seed % 5
    A, b = _spd(n, seed)
    got = E.exact_solve_fractions(A, b)
    ref = solve_fractions([[Fraction(float(v)) for v in row] for row in A], [Fraction(float(v)) for v in b])
    assert got == ref                                   # Brüche gleich, nicht nur nahe


def test_exact_solution_satisfies_the_system_exactly_and_the_float_solution_is_close():
    A, b = _spd(5, 99)
    sol = E.exact_solve_fractions(A, b)
    for i in range(5):
        assert sum(Fraction(float(A[i, j])) * sol[j] for j in range(5)) == Fraction(float(b[i]))      # Rest exakt null
    assert np.allclose(E.exact_solve(A, b), np.linalg.solve(A, b), rtol=1e-12)


def test_multi_rhs_equals_separate_solves():
    A, b = _spd(4, 7)
    B = np.column_stack([b, np.eye(4)[:, 0], np.eye(4)[:, 3]])
    multi = E.exact_solve_multi(A, B)
    for c in range(3):
        assert multi[c] == E.exact_solve_fractions(A, B[:, c])


def test_pivot_swap_and_singular_matrix():
    assert E.exact_solve_fractions([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0]) == [Fraction(3), Fraction(2)]     # Nullpivot: Zeilen vertauschen
    with pytest.raises(ZeroDivisionError):
        E.exact_solve_fractions([[1.0, 1.0], [1.0, 1.0]], [1.0, 2.0])                                     # exakt singulär
    with pytest.raises(ZeroDivisionError):
        E.exact_solve_fractions([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0])


def test_one_by_one_system():
    assert E.exact_solve_fractions([[4.0]], [1.0]) == [Fraction(1, 4)]


def test_exact_residual_norm_against_fractions():
    A, b = _spd(4, 3)
    a = np.linalg.solve(A, b)
    res = [sum(Fraction(float(A[i, j])) * Fraction(float(a[j])) for j in range(4)) - Fraction(float(b[i])) for i in range(4)]
    assert E.exact_residual_norm(A, a, b) == pytest.approx(math.sqrt(sum(float(r) ** 2 for r in res)), rel=1e-12)
    # Mini-Instanz von Hand: A = [[2, 0], [0, 1]], a = (1, 1), y = (2, 0): Residuum (0, 1), Norm 1
    assert E.exact_residual_norm(np.array([[2.0, 0.0], [0.0, 1.0]]), np.array([1.0, 1.0]), np.array([2.0, 0.0])) == pytest.approx(1.0, abs=0)


def test_backward_error_is_tiny_for_the_float_solution_and_large_for_a_wrong_one():
    A, b = _spd(5, 11)
    a = np.linalg.solve(A, b)
    assert E.backward_error(A, a, b) < 1e-15
    assert E.backward_error(A, a + 0.1, b) > 1e-3


def test_exact_predict_equals_fraction_dot_product():
    A, b = _spd(4, 5)
    sol = E.exact_solve_fractions(A, b)
    Kt = np.array([[0.1, 0.25, 0.5, 0.75], [1.0, 0.0, 0.3, 0.2]])
    got = E.exact_predict(Kt, sol)
    for t in range(2):
        assert got[t] == float(sum(Fraction(float(Kt[t, j])) * sol[j] for j in range(4)))               # gleich auf das letzte Bit (korrekt gerundet)


def test_jacobi_eigenvalues_of_the_mini_gram_matrix_are_the_closed_form(mini_gram):
    ev = E.jacobi_eigenvalues(mini_gram)
    sq = Decimal(513).sqrt() / 16
    expected = [(Decimal(33) / 16 + sq) / 2, Decimal(15) / 16, (Decimal(33) / 16 - sq) / 2]
    for got, exp in zip(ev, expected):
        assert abs(got - exp) < Decimal(10) ** -100


@pytest.mark.parametrize("seed", range(6))
def test_jacobi_equals_newton_on_the_characteristic_polynomial(seed):
    n = 3 + seed % 3
    A, _ = _spd(n, 40 + seed)
    R = [[Fraction(float(v)) for v in row] for row in A]
    j, nw = E.jacobi_eigenvalues(R), newton_eigenvalues(R)
    assert all(abs(a / b - 1) < Decimal(10) ** -60 for a, b in zip(j, nw))
    assert np.allclose([float(v) for v in j], np.linalg.eigvalsh(A)[::-1], rtol=1e-12)


def test_jacobi_handles_matrices_that_are_not_positive_semidefinite():
    ev = E.jacobi_eigenvalues([[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]])
    assert [float(v) for v in ev] == pytest.approx([1.0, -1.0], abs=1e-100)


def test_exact_pivots_of_the_mini_gram_matrix(mini_gram):
    assert E.exact_pivots([[float(v) for v in row] for row in mini_gram]) == [1, Fraction(3, 4), Fraction(45, 64)]


def test_exact_pivots_detect_a_matrix_that_is_not_positive_definite():
    piv = E.exact_pivots([[1.0, 2.0], [2.0, 1.0]])
    assert piv == [1, -3]
    with pytest.raises(ZeroDivisionError):
        E.exact_pivots([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
