"""Orakel-Test gram-kondition-demo / gram_kondition: Kernroutinen gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, DEMO-PLAYBOOK §12).

Orakel (jeweils anderer Rechenweg als der der App):
(a) exakte Lösung der App (Bareiss, ganze Zahlen) gegen Gauß-Elimination in Brüchen, numpy, `scipy.linalg.cho_solve` und `sklearn.kernel_ridge.KernelRidge`;
(b) Eigenwerte: Jacobi-Verfahren der App gegen das exakte charakteristische Polynom (Faddeev-LeVerrier, Newton mit Abspalten) gegen `numpy.linalg.eigvalsh`; im Flat Limit
    gegen die geschlossene Form (Gram-Schmidt der Monome, zweiter Weg über Gram-Determinanten);
(c) Konditionszahl (λ_max + λ)/(λ_min + λ) gegen `numpy.linalg.cond`;
(d) Leave-one-out geschlossen gegen echtes Herausnehmen (Gleitkomma und Brüche);
(e) exakte Vorhersage (Brüche) gegen Fraction-Skalarprodukt, und der exakt gerechnete Rückwärtsfehler gegen eine Rechnung in Brüchen.
Auch die angezeigten Kennzahlen (Fehler, η, cond, Zahl echter Eigenwerte) werden geprüft, nicht nur das Hauptergebnis. Kleine feste Fassung unter 10 s; der große Lauf (200 Zufallsinstanzen)
steht im Messskript tools/gkd_messung.py (Abschnitt B, G)."""

import math
from fractions import Fraction

import numpy as np
import pytest

import gkd_exact as E
import gkd_experiments as X
import gkd_fit as Fit
import gkd_flat as Fl
import gkd_kernel as Kn
import gkd_scenario as S
from oracle_helpers import gs_squares_from_gram_determinants, newton_eigenvalues, solve_fractions


def _fr(A):
    return [[Fraction(float(v)) for v in row] for row in A]


@pytest.mark.parametrize("seed", range(12))
def test_a_exact_solution_equals_fraction_elimination_and_the_float_paths_at_moderate_nugget(seed):
    n = 4 + seed % 8
    X_ = S.points_square(n, 300 + seed)
    y = S.normal_targets(n, 400 + seed)
    A = Kn.gauss_gram(X_, X_, 0.8) + 0.05 * np.eye(n)
    sol = E.exact_solve_fractions(A, y)
    assert sol == solve_fractions(_fr(A), [Fraction(float(v)) for v in y])          # exakt gleich
    a_ex = np.array([float(v) for v in sol])
    cho = pytest.importorskip("scipy.linalg")
    assert np.allclose(a_ex, np.linalg.solve(A, y), rtol=1e-10, atol=1e-12)
    assert np.allclose(a_ex, cho.cho_solve(cho.cho_factor(A), y), rtol=1e-10, atol=1e-12)
    assert np.allclose(a_ex, Fit.solve_cholesky(A, y), rtol=1e-10, atol=1e-12)


def test_a_predictions_equal_scikit_learn_kernel_ridge():
    KR = pytest.importorskip("sklearn.kernel_ridge").KernelRidge
    for seed in range(4):
        X_, y = S.regression_data(25, 0.3, 500 + seed)
        Xt, _ = S.reg_test_points(20)
        K = Kn.gauss_gram(X_, X_, 1.2)
        A = K + 0.01 * np.eye(25)
        sol = E.exact_solve_fractions(A, y)
        pred = E.exact_predict(Kn.gauss_gram(Xt, X_, 1.2), sol)
        ref = KR(alpha=0.01, kernel="rbf", gamma=Kn.gamma_of_sigma(1.2)).fit(X_, y).predict(Xt)
        assert np.max(np.abs(pred - ref)) < 1e-9 * max(1.0, np.max(np.abs(ref)))


@pytest.mark.parametrize("xs,k", [((0, 1, 2, 3, 4), 4), ((0, 1, 2, 3, 4), 8), ((0, 1, 3, 4, 7), 6), ((0, 1, 2), 2)])
def test_b_eigenvalues_three_ways_and_the_closed_form(xs, k):
    q = Fl.q_of_k(k)
    R = Fl.rational_gram(xs, q)
    jac, newt = E.jacobi_eigenvalues(R), newton_eigenvalues(R)
    assert all(abs(a / b - 1) < 1e-50 for a, b in zip(jac, newt))
    float_ev = np.linalg.eigvalsh(np.array([[float(v) for v in row] for row in R]))[::-1]
    top = float(jac[0])
    for a, b in zip(jac, float_ev):                                                 # eigvalsh nur dort, wo die Eigenwerte über der Rundungsgrenze liegen
        if float(a) > 1e-12 * top:
            assert float(a) == pytest.approx(float(b), rel=1e-6)
    assert Fl.gram_schmidt_squares(xs) == gs_squares_from_gram_determinants(xs)      # R_mm² auf zwei Wegen
    e2 = Fl.eps2_of_q(q)
    if k >= 6:
        for m, lam in enumerate(jac):
            assert abs(float(lam / Fl.ladder_prediction(xs, e2)[m]) - 1) < 40 * float(e2)


def test_c_condition_formula_equals_numpy_cond_on_random_instances():
    for seed in range(15):
        n = 6 + seed
        X_ = S.points_square(n, 600 + seed)
        sg = (0.2, 0.5, 1.0)[seed % 3]
        K = Kn.gauss_gram(X_, X_, sg)
        for lam in (0.0, 1e-4, 0.1, 1.0):
            c, lower = Kn.condition(Kn.eigenvalues(K), lam)
            if not lower:
                assert c == pytest.approx(np.linalg.cond(K + lam * np.eye(n)), rel=max(1e-9, 1e-15 * c))      # Rundungsfehler beider Verfahren wächst mit cond
            assert c <= (Kn.nugget_cond_bound(n, lam) * (1 + 1e-9) if lam > 0 else Kn.INV_EPS * (1 + 1e-9))


def test_d_loo_closed_form_equals_leaving_one_out_for_real_exactly_in_fractions():
    X_, y = S.regression_data(7, 0.3, 700)
    yc = y - y.mean()
    K = Kn.gauss_gram(X_, X_, 1.0)
    for lam in (0.05, 1.0):
        A = K + lam * np.eye(7)
        sol = E.exact_solve_multi(A, np.column_stack([yc, np.eye(7)]))
        closed = [sol[0][i] / sol[1 + i][i] for i in range(7)]
        real = []
        for i in range(7):
            idx = [j for j in range(7) if j != i]
            a_m = E.exact_solve_fractions(A[np.ix_(idx, idx)], yc[idx])
            real.append(Fraction(float(yc[i])) - sum(Fraction(float(K[i, idx[t]])) * a_m[t] for t in range(6)))
        assert closed == real                                                        # Gleichheit in Brüchen
        assert Fit.loo_residuals(K, yc, lam) == pytest.approx([float(v) for v in real], rel=1e-8, abs=1e-10)


def test_e_displayed_quantities_error_eta_condition_and_count_of_real_eigenvalues():
    ins = X.falle_instance(12, 1.5, 0.0, 800, 0)
    X_ = S.points_square(12, X.falle_seeds(800, 0)[0])
    y = S.normal_targets(12, X.falle_seeds(800, 0)[1])
    K = Kn.gauss_gram(X_, X_, 1.5)
    sol = E.exact_solve_fractions(K, y)
    a_ex = np.array([float(v) for v in sol])
    a_lu = np.linalg.solve(K, y)
    assert ins.err_lu == pytest.approx(np.linalg.norm(a_lu - a_ex) / np.linalg.norm(a_ex), rel=1e-12)
    res = [sum(Fraction(float(K[i, j])) * Fraction(float(a_lu[j])) for j in range(12)) - Fraction(float(y[i])) for i in range(12)]
    eta = math.sqrt(sum(float(r) ** 2 for r in res)) / (np.linalg.norm(K, 2) * np.linalg.norm(a_lu) + np.linalg.norm(y))
    assert ins.eta_lu == pytest.approx(eta, rel=1e-9)                               # exakt gerechnetes Residuum gegen Bruchrechnung
    ev = np.linalg.eigvalsh(K)[::-1]
    assert ins.n_real == int((ev > 1e-16 * ev[0]).sum()) and ins.eigenvalues == pytest.approx(tuple(ev), rel=1e-12)
    assert ins.cond == pytest.approx(np.linalg.cond(K), rel=1e-6) or ins.cond_lower_bound
    pred_ref = [float(sum(Fraction(float(v)) * s for v, s in zip(row, sol))) for row in Kn.gauss_gram(S.square_test_points(), X_, 1.5)]
    p_lu = Kn.gauss_gram(S.square_test_points(), X_, 1.5) @ a_lu
    assert ins.pred_lu == pytest.approx(np.linalg.norm(p_lu - np.array(pred_ref)) / np.linalg.norm(pred_ref), rel=1e-6)


def test_e_nugget_row_equals_a_direct_recount():
    n, sigma, lam = 10, 2.0, 1e-6
    row = X.nugget_reihe(n, sigma, 5, 900, (lam,))[0]
    assert row.bound == pytest.approx(1 + n / lam) and row.rule == pytest.approx(16 - math.log10(1 + n / lam))
    digits = []
    for i in range(5):
        sp, sy = X.falle_seeds(900, i)
        X_, y = S.points_square(n, sp), S.normal_targets(n, sy)
        A = Kn.gauss_gram(X_, X_, sigma) + lam * np.eye(n)
        a_ex = np.array([float(v) for v in solve_fractions(_fr(A), [Fraction(float(v)) for v in y])])
        digits.append(-math.log10(np.linalg.norm(np.linalg.solve(A, y) - a_ex) / np.linalg.norm(a_ex)))
    assert row.digits_min == pytest.approx(min(digits), rel=1e-6) and row.digits_median == pytest.approx(float(np.median(digits)), rel=1e-6)
    assert row.digits_violations == sum(1 for d in digits if d < row.rule) and row.count == 5
