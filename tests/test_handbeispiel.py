"""Das Handbeispiel (drei Punkte x = (0, 1, 2), q = exp(-1/(2σ²)) = 1/2, also σ = 0,8493, γ = ln 2) als Tests, mit denselben engen Toleranzen: Brüche, Handrechnung,
numpy, scipy (cho_solve) und scikit-learn (rbf_kernel, KernelRidge) als Fremdreferenz. Die Cholesky-Grenze (Handbeispiel 8) wird für beide Aufbauten der Matrix geprüft:
Potenzen eines gerundeten q und exp(-d²/(2σ²)), der Aufbau der App (Befund der zweiten Prüfung)."""

import math
from fractions import Fraction

import numpy as np
import pytest

import gkd_exact as E
import gkd_flat as Fl
import gkd_kernel as Kn
from oracle_helpers import det_fractions, solve_fractions

XS = (0, 1, 2)
SIGMA = 1 / math.sqrt(2 * math.log(2))
EPS = float(np.finfo(float).eps)


def _kn(K):
    return np.array([[float(v) for v in row] for row in K])


def test_1_matrix_equals_the_fraction_matrix_and_scikit_learn(mini_gram):
    metrics = pytest.importorskip("sklearn.metrics.pairwise")
    assert abs(SIGMA - 0.8493218) < 1e-6
    assert Kn.gamma_of_sigma(SIGMA) == pytest.approx(math.log(2), abs=1e-12)
    Ks = metrics.rbf_kernel(np.array(XS, float)[:, None], gamma=math.log(2))
    assert np.max(np.abs(Ks - _kn(mini_gram))) < 1e-15
    assert np.max(np.abs(Kn.gauss_gram(np.array(XS, float), np.array(XS, float), SIGMA) - _kn(mini_gram))) < 1e-15


def test_2_determinant_on_three_ways(mini_gram):
    q = Fraction(1, 2)
    closed = (1 - q ** 4) * (1 - q ** 2) ** 2
    assert closed == det_fractions(mini_gram) == Fraction(135, 256)
    Kn_ = _kn(mini_gram)
    assert abs(np.linalg.det(Kn_) - 135 / 256) < 1e-15
    assert abs(np.prod(np.linalg.eigvalsh(Kn_)) - 135 / 256) < 1e-14


def test_3_eigenvalues_antisymmetric_exact_and_symmetric_from_a_quadratic(mini_gram):
    K = mini_gram
    v = [1, 0, -1]
    assert [sum(K[i][j] * v[j] for j in range(3)) for i in range(3)] == [Fraction(15, 16) * c for c in v]
    q = Fraction(1, 2)
    tr, dt = 2 + q ** 4, (1 - q ** 2) ** 2
    assert (tr, dt) == (Fraction(33, 16), Fraction(9, 16)) and tr * tr - 4 * dt == Fraction(513, 256)
    lam = sorted([(33 + math.sqrt(513)) / 32, 15 / 16, (33 - math.sqrt(513)) / 32])
    ev = np.linalg.eigvalsh(_kn(K))
    assert np.max(np.abs(np.array(lam) - ev)) < 1e-14 and abs(sum(lam) - 3) < 1e-14
    cond = lam[-1] / lam[0]
    assert abs(cond - np.linalg.cond(_kn(K))) / cond < 1e-12
    assert abs(lam[-1] - 1.7390) < 1e-3 and abs(lam[0] - 0.3235) < 1e-3 and abs(cond - 5.376) < 1e-2


def test_4_cholesky_pivots_by_hand(mini_gram):
    K = mini_gram
    d1 = K[0][0]
    d2 = K[1][1] - K[1][0] ** 2 / d1
    l31, l32 = K[2][0] / d1, (K[2][1] - K[1][0] * K[2][0] / d1) / d2
    d3 = K[2][2] - l31 ** 2 * d1 - l32 ** 2 * d2
    assert (d1, d2, d3) == (1, Fraction(3, 4), Fraction(45, 64))
    q = Fraction(1, 2)
    assert d1 * d2 * d3 == det_fractions(K) and d3 == (1 - q ** 4) * (1 - q ** 2) and d2 == 1 - q ** 2
    assert np.max(np.abs(np.diag(np.linalg.cholesky(_kn(K))) ** 2 - np.array([float(d1), float(d2), float(d3)]))) < 1e-15


def test_5_solve_on_five_paths_and_kernel_ridge(mini_gram):
    cho = pytest.importorskip("scipy.linalg")
    KR = pytest.importorskip("sklearn.kernel_ridge").KernelRidge
    lam = Fraction(1, 10)
    A = [[mini_gram[i][j] + (lam if i == j else 0) for j in range(3)] for i in range(3)]
    y = [Fraction(1), Fraction(0), Fraction(-1, 2)]
    a_fr = solve_fractions(A, y)
    assert all(sum(A[i][j] * a_fr[j] for j in range(3)) == y[i] for i in range(3))
    An, yn = _kn(A), _kn([y])[0]
    a_f = np.array([float(v) for v in a_fr])
    L, V = np.linalg.eigh(An)
    for a in (np.linalg.solve(An, yn), cho.cho_solve(cho.cho_factor(An), yn), V @ ((V.T @ yn) / L), E.exact_solve(An, yn)):
        assert np.max(np.abs(a - a_f)) < 1e-14
    kr = KR(alpha=0.1, kernel="rbf", gamma=math.log(2)).fit(np.array(XS, float)[:, None], yn)
    xt = np.array([[0.5], [1.5], [3.0]])
    k_t = np.exp(-math.log(2) * (xt - np.array(XS, float)[None, :]) ** 2)
    assert np.max(np.abs(kr.predict(xt) - k_t @ a_f)) < 1e-13
    ev_a = np.linalg.eigvalsh(An)
    assert ev_a[-1] / ev_a[0] <= 1 + 3 / 0.1                                      # cond(K + λI) ≤ 1 + n/λ = 31
    lam_hi, lam_lo = (33 + math.sqrt(513)) / 32, (33 - math.sqrt(513)) / 32
    assert abs(ev_a[-1] / ev_a[0] - (lam_hi + 0.1) / (lam_lo + 0.1)) < 1e-12


def test_7_flat_limit_of_the_mini_instance_in_floating_point_at_sigma_10():
    sg = 10.0
    q = math.exp(-1 / (2 * sg * sg))
    K = np.array([[q ** ((a - b) ** 2) for b in XS] for a in XS])
    cm = [float(c) for c in Fl.ladder_coefficients(XS)]
    assert cm == pytest.approx([3, 2, 1 / 3])
    for m, lam in enumerate(sorted(np.linalg.eigvalsh(K), reverse=True)):
        assert abs(lam / (cm[m] * (1 / sg ** 2) ** m) - 1) < 0.2                    # auf 20 % gleich c_m (1/σ)^(2(m-1))
    assert abs(np.linalg.cond(K) / (9 * sg ** 4) - 1) < 0.2


# ---------------------------------------------------------------- 8. Wo bricht Cholesky? Pivot d3 = (1-q^4)(1-q^2) ~ 2/σ⁴ erreicht eps bei σ = (2/eps)^(1/4)
PRED = (2 / EPS) ** 0.25


def _gram_power(sg):
    q = math.exp(-1 / (2 * sg * sg))
    return np.array([[q ** ((a - b) ** 2) for b in XS] for a in XS])


def _gram_exp(sg):
    x = np.array(XS, float)
    return np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * sg * sg))


def _chol_ok(K):
    return Kn.cholesky_or_none(K) is not None


def _exact_pd(K):
    try:
        return all(p > 0 for p in E.exact_pivots(K))
    except ZeroDivisionError:
        return False


def test_8_prediction_of_the_limit_is_two_to_the_53_over_4():
    assert PRED == pytest.approx(9.742e3, abs=10)                                 # nicht 5,5e3
    assert PRED == pytest.approx(2 ** 13.25, abs=1e-6)


def test_8_power_construction_cholesky_fails_first_above_the_prediction_and_always_from_1_4_times():
    ratios = np.geomspace(0.5, 2.0, 1501)
    okn = np.array([_chol_ok(_gram_power(r * PRED)) for r in ratios])
    okx = np.array([_exact_pd(_gram_power(r * PRED)) for r in ratios])
    assert okn[ratios <= 1.0].all()                                               # Erfolg bis zur Vorhersage
    assert 1.0 < ratios[~okn][0] < 1.5                                            # erstes Scheitern zwischen dem 1,0- und 1,5fachen
    assert not okn[ratios >= 1.4].any()                                           # ab dem 1,4fachen in jedem Rasterwert
    assert okx[ratios <= 1.25].all() and not okx[ratios >= 1.33].any()            # gerundete Matrix exakt positiv definit bis 1,25, ab 1,33 nicht mehr
    assert not (okn & ~okx).any()                                                 # numpy gelingt nirgends, wo die gerundete Matrix nicht positiv definit ist
    assert (~okn & okx).any()                                                     # aber es gibt Rasterwerte mit Scheitern trotz exakt positiv definiter Matrix (Lotto)


def test_8_exp_construction_the_aufbau_of_the_app_fails_earlier_and_succeeds_much_further():
    raster = np.linspace(2e3, 1e5, 2001)
    ratios = raster / PRED
    oke = np.array([_chol_ok(_gram_exp(s)) for s in raster])
    okx = np.array([_exact_pd(_gram_exp(s)) for s in raster])
    assert 0.5 < ratios[~oke][0] < 1.3                                            # erstes Scheitern früh (Messung 0,8fach), nicht erst beim 1,19fachen
    assert ratios[oke][-1] > 5.0                                                  # Erfolge noch weit über der Vorhersage (Messung: bis zum 10fachen)
    assert 0.25 < oke[ratios > 2.0].mean() < 0.55                                 # über dem Doppelten gelingt Cholesky in etwa 38 % der Rasterwerte
    assert 0.25 < okx[ratios > 2.0].mean() < 0.55
    assert (oke & ~okx).any()                                                     # Rundung in der Zerlegung: Erfolg trotz exakt nicht positiv definiter Matrix
    assert oke[ratios <= 0.5].mean() > 0.97                                       # weit unter der Vorhersage gelingt sie fast immer


def test_8_both_constructions_work_at_sigma_10_and_100_and_fail_at_1e7():
    for sg in (10.0, 100.0):
        assert _chol_ok(_gram_power(sg)) and _chol_ok(_gram_exp(sg))
    assert not _chol_ok(_gram_power(1e7))
    assert not _chol_ok(_gram_exp(1e7))


def test_9_two_nearly_equal_points_by_eigenvectors_in_fractions():
    r = Fraction(99, 100)
    K2 = [[1, r], [r, 1]]
    for vv, ew in (((1, 1), 1 + r), ((1, -1), 1 - r)):
        assert [sum(K2[i][j] * vv[j] for j in range(2)) for i in range(2)] == [ew * c for c in vv]
    assert K2[0][0] * K2[1][1] - K2[0][1] * K2[1][0] == (1 + r) * (1 - r)
    assert (1 + r) / (1 - r) == 199
    assert abs(np.linalg.cond(np.array([[1, 0.99], [0.99, 1]])) - 199) < 1e-9 * 199
