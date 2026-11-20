"""Flat Limit (gkd_flat.py): Koeffizienten der Leiter von Hand und in Brüchen, exakte Eigenwerte gegen die Leiter, Zählung echter Eigenwerte, Randfälle.

Handwerte (Mini-Instanz x = (0, 1, 2), q = 1/2): Gram-Schmidt der Monome 1, x, x² gibt R² = (3, 2, 2/3), also c = (3, 2, 1/3); λ ≈ (3, 2/σ², 1/(3σ⁴)), cond ≈ 9σ⁴."""

import math
from fractions import Fraction

import numpy as np
import pytest

import gkd_exact as E
import gkd_flat as Fl
from oracle_helpers import gs_squares_from_gram_determinants, newton_eigenvalues


def test_ladder_coefficients_of_the_mini_instance_by_hand():
    assert Fl.gram_schmidt_squares((0, 1, 2)) == [3, 2, Fraction(2, 3)]
    assert Fl.ladder_coefficients((0, 1, 2)) == [3, 2, Fraction(1, 3)]


@pytest.mark.parametrize("xs,expected", [
    ((0, 1, 2, 3, 4), [5, 10, 7, Fraction(12, 5), Fraction(12, 35)]),
    ((0, 1, 3, 4, 7), [5, 30, 72, Fraction(174, 5), Fraction(1764, 145)]),
])
def test_ladder_coefficients_of_the_planned_point_sets(xs, expected):
    assert Fl.ladder_coefficients(xs) == expected


@pytest.mark.parametrize("xs", [(0, 1, 2), (0, 1, 2, 3), (0, 1, 3, 4, 7), (-2, 0, 1, 5), (0, 2, 3, 7, 11, 12)])
def test_gram_schmidt_equals_the_quotient_of_gram_determinants(xs):
    assert Fl.gram_schmidt_squares(xs) == gs_squares_from_gram_determinants(xs)          # zweiter Rechenweg, exakt gleich


def test_rational_gram_and_q_of_k():
    assert Fl.q_of_k(4) == Fraction(9999, 10000)
    K = Fl.rational_gram((0, 1, 2), Fraction(1, 2))
    assert K == [[1, Fraction(1, 2), Fraction(1, 16)], [Fraction(1, 2), 1, Fraction(1, 2)], [Fraction(1, 16), Fraction(1, 2), 1]]


def test_eps2_of_q_is_minus_two_log_q():
    e2 = Fl.eps2_of_q(Fraction(1, 2))
    assert float(e2) == pytest.approx(2 * math.log(2), rel=1e-15)
    assert float(Fl.eps2_of_q(Fl.q_of_k(8))) == pytest.approx(2e-8, rel=1e-7)


@pytest.mark.parametrize("xs", [(0, 1, 2), (0, 1, 2, 3, 4), (0, 1, 3, 4, 7)])
def test_exact_eigenvalues_follow_the_ladder_in_the_flat_limit(xs):
    n = len(xs)
    r8 = Fl.analyse_flat(xs, 8)
    r6 = Fl.analyse_flat(xs, 6)
    assert all(abs(v - 1) < 40 * r8.eps2 for v in r8.ratio_exact_ladder)          # 1,000000 bei k = 8
    assert all(abs(v - 1) < 40 * r6.eps2 for v in r6.ratio_exact_ladder)
    slopes = [math.log(a / b) / math.log(math.sqrt(r6.eps2) / math.sqrt(r8.eps2)) for a, b in zip(r6.exact, r8.exact)]
    assert slopes == pytest.approx([2 * m for m in range(n)], abs=1e-3)          # Steigungen 0 / 2 / 4 / ...
    assert r8.cond_exact == pytest.approx(r8.cond_ladder, rel=1e-6)


def test_mini_instance_flat_limit_values_by_hand():
    for k in (4, 6, 8):
        q = Fl.q_of_k(k)
        e2 = float(Fl.eps2_of_q(q))
        lam = E.jacobi_eigenvalues(Fl.rational_gram((0, 1, 2), q))
        assert float(lam[0]) == pytest.approx(3, rel=40 * e2)
        assert float(lam[1]) == pytest.approx(2 * e2, rel=40 * e2)
        assert float(lam[2]) == pytest.approx(e2 ** 2 / 3, rel=40 * e2)
        det = float(lam[0] * lam[1] * lam[2])
        assert det == pytest.approx(2 * e2 ** 3, rel=40 * e2)                     # det K = 2/σ⁶
        assert float(lam[0] / lam[2]) == pytest.approx(9 / e2 ** 2, rel=40 * e2)  # cond = 9 σ⁴


def test_determinant_closed_form_matches_the_exact_determinant():
    for xs in ((0, 1, 2, 3, 4), (0, 1, 3, 4, 7)):
        q = Fl.q_of_k(6)
        R = Fl.rational_gram(xs, q)
        ev = E.jacobi_eigenvalues(R)
        det = math.prod(float(v) for v in ev)
        e2 = Fl.eps2_of_q(q)
        assert det == pytest.approx(float(Fl.determinant_closed_form(xs, e2)), rel=40 * float(e2))


def test_jacobi_on_the_flat_limit_matrix_equals_newton_on_the_characteristic_polynomial():
    R = Fl.rational_gram((0, 1, 2, 3, 4), Fl.q_of_k(8))
    j, nw = E.jacobi_eigenvalues(R), newton_eigenvalues(R)
    assert all(abs(a / b - 1) < 1e-50 for a, b in zip(j, nw))                     # Eigenwerte über 31 Größenordnungen, relativ auf 50 Stellen gleich


def test_count_real_counts_eigenvalues_within_ten_percent():
    assert Fl.count_real([1.0, 0.5, 0.01], [1.05, 0.5, 1e-9]) == 2
    assert Fl.count_real([0.91], [1.0]) == 1 and Fl.count_real([1.09], [1.0]) == 1          # Abweichung 9 %: echt
    assert Fl.count_real([0.89], [1.0]) == 0 and Fl.count_real([1.11], [1.0]) == 0          # Abweichung 11 %: nicht mehr echt


def test_n_equals_six_matches_the_measured_counts_to_within_one_eigenvalue():
    xs = tuple(range(6))
    ks = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
    res = [Fl.analyse_flat(xs, k) for k in ks]
    measured = [6, 6, 5, 4, 3, 3, 3, 2, 2, 2]                                      # Messung dieser Plattform (README); auf einer anderen LAPACK-Bibliothek um einen Eigenwert verschiebbar
    assert all(abs(r.real_float - m) <= 1 for r, m in zip(res, measured)) and res[0].real_float == 6 and res[-1].real_float <= 3
    assert all(a >= b for a, b in zip([r.real_float for r in res], [r.real_float for r in res][1:]))
    assert all(abs(r.predicted - r.real_float) <= 1 for r in res)
    assert all(abs(r.real_rounded - r.real_float) <= 1 for r in res)
    assert sum(1 for r in res if r.real_rounded == r.real_float) >= 9              # in 9 von 10 Zellen gleich (Eingaberundung, nicht der Löser)


def test_the_rounded_gram_matrix_is_not_positive_semidefinite_any_more():
    xs = tuple(range(6))
    res = {k: Fl.analyse_flat(xs, k) for k in (2, 4, 8, 12)}
    assert not res[2].smallest_rounded_negative and not res[2].smallest_float_negative
    for k in (4, 8, 12):
        assert res[k].smallest_float_negative
    assert sum(1 for k in (4, 5, 6, 7, 8, 9, 10, 12) if Fl.analyse_flat(xs, k).smallest_float_negative) == 8


def test_pivots_and_cholesky_status_follow_the_rounding():
    ok = Fl.analyse_flat((0, 1, 2, 3, 4), 4)
    assert ok.cholesky_ok and ok.pivot_negative_at is None and all(p > 0 for p in ok.pivots_float)
    bad = Fl.analyse_flat((0, 1, 2, 3, 4, 5), 12)
    assert not bad.cholesky_ok and bad.pivot_negative_at is not None and bad.pivot_negative_at <= 6


def test_analyse_flat_returns_consistent_shapes_for_all_regler_corners():
    for n in (3, 8):
        for k in (2, 12):
            for xs in ((tuple(range(n))), (0, 1, 3, 4, 7, 9, 12, 16)[:n]):
                r = Fl.analyse_flat(xs, k)
                assert len(r.exact) == len(r.rounded) == len(r.floating) == len(r.ladder) == len(r.ratio_exact_ladder) == len(r.pivots_float) == n
                assert r.sigma == pytest.approx(1 / math.sqrt(r.eps2)) and r.cond_exact > 0 and np.all(np.diff(r.exact) <= 0)
