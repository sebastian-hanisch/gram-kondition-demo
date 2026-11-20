"""Szenario (gkd_scenario.py): Erzeuger deterministisch, im Gebiet, richtige Form; die Zielfunktion von Hand."""

import math

import numpy as np
import pytest

import gkd_constants as C
import gkd_scenario as S


def test_point_generators_are_deterministic_in_range_and_shaped():
    P = S.points_square(30, 5)
    assert P.shape == (30, 2) and np.array_equal(P, S.points_square(30, 5)) and not np.array_equal(P, S.points_square(30, 6))
    assert P.min() >= 0.0 and P.max() < 1.0
    L = S.points_line(30, 5)
    assert L.shape == (30, 1) and np.all(np.diff(L[:, 0]) >= 0) and L.min() >= 0.0 and L.max() < 1.0


def test_points_are_uniform():
    P = S.points_square(4000, 9)
    for j in range(2):
        counts = np.histogram(P[:, j], bins=10, range=(0, 1))[0]
        assert counts.min() > 4000 / 10 - 5 * math.sqrt(4000 * 0.1 * 0.9)
        assert counts.max() < 4000 / 10 + 5 * math.sqrt(4000 * 0.1 * 0.9)


def test_target_function_by_hand():
    X = np.array([[0.0, 0.0], [math.pi / 3, 0.0], [0.0, 2.0], [1.0, -1.0]])
    expected = [0.5, math.sin(1.5 * math.pi / 3) + 0.5, 0.5 * math.cos(5.0), math.sin(1.5) + 0.5 * math.cos(-2.5) - 0.3]
    assert S.target_2d(X) == pytest.approx(expected, abs=1e-14)


def test_regression_data_is_in_the_region_with_the_stated_noise():
    X, y = S.regression_data(4000, 0.3, 11)
    assert X.shape == (4000, 2) and X.min() >= C.REG_LO and X.max() < C.REG_HI
    r = y - S.target_2d(X)
    assert abs(r.mean()) < 4 * 0.3 / 63 and abs(r.std() - 0.3) < 0.02
    X0, y0 = S.regression_data(50, 0.0, 11)
    assert np.array_equal(y0, S.target_2d(X0))                                # Rauschen null: genau die Zielfunktion


def test_test_points_are_fixed_and_noise_free():
    X1, f1 = S.reg_test_points()
    X2, f2 = S.reg_test_points()
    assert X1.shape == (C.N_TEST_REG, 2) and np.array_equal(X1, X2) and np.array_equal(f1, S.target_2d(X1))
    assert S.reg_test_points(17)[0].shape == (17, 2) and S.square_test_points().shape == (C.N_TEST_PRED, 2)


def test_integer_point_sets_and_sigma_of_q():
    assert S.equispaced_points(4) == (0, 1, 2, 3)
    assert S.uneven_points(5) == (0, 1, 3, 4, 7)
    for bad in (0, 9):
        with pytest.raises(ValueError):
            S.uneven_points(bad)
    # q = 1/2 <=> k mit 1 - 10^-k = 1/2 gibt es nicht ganzzahlig; Kontrolle bei k = 2: q = 0,99, σ² = 1/(-2 ln 0,99)
    assert S.sigma_from_q_exponent(2) == pytest.approx(1 / math.sqrt(-2 * math.log(0.99)), rel=1e-14)
    assert S.sigma_from_q_exponent(8) == pytest.approx(7071.07, rel=1e-5)
    assert S.sigma_from_q_exponent(12) > S.sigma_from_q_exponent(8)
