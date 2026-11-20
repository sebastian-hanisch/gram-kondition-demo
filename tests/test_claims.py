"""Zahl → Test: die Zahlen der README als Bänder (nicht als Einzelwerte), gerechnet mit denselben Funktionen wie die App.

Die README nennt Messungen über 10 Sätze zu je 30 Punktmengen (tools/gkd_messung.py, Ausgabe tools/out_gkd_messung.txt) und vorgerechnete Sätze (gkd_frozen.json). Diese Tests rechnen
kleinere Fassungen neu und prüfen, dass die Aussage im Band der README liegt; die Zählung „Cholesky gelingt in k von 30“ ist ein Rundungslotto (Plattform, LAPACK, Zufalls-Strom) und
steht im Test nur als weites Band. Eine Zahl ohne Test steht in der README ausdrücklich als „Messung dieser README, nicht im Test“.
Unterschied zu den Handwerten: die stehen exakt in test_handbeispiel.py, test_flat.py und test_oracle_gram_kondition.py."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import gkd_constants as C
import gkd_evaluation as V
import gkd_experiments as X
import gkd_flat as Fl
import gkd_kernel as Kn
import gkd_scenario as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FROZEN = json.loads((ROOT / C.FROZEN_FILE).read_text(encoding="utf-8"))
BASES = (1000, 1001, 1002)          # drei Sätze zu je 30 Punktmengen (die README misst zehn: 1000 bis 1009)


def _sets(n, sigma, lam, bases=BASES):
    return [V.summarize_falle(X.falle_zaehlung(n, sigma, lam, 30, b), n) for b in bases]


# ---------------------------------------------------------------- A: Cholesky gelingt (Hook)
def test_falle_sigma_5_lambda_0_cholesky_succeeds_in_a_minority_to_half_and_the_solution_is_mostly_wrong():
    sets = _sets(20, 5.0, 0.0)
    for s in sets:
        assert 6 <= s.ok <= 24                                                   # README: 12 bis 19 von 30 je Satz (Rundungslotto, weites Band)
        assert s.max_eta < 2.2e-16                                               # exakter Rückwärtsfehler unter der Rundungsgenauigkeit (README: höchstens 1,0e-16)
    ok, bad = sum(s.ok for s in sets), sum(s.bad_10 for s in sets)
    assert bad / ok > 0.6                                                        # README: 137 von 156 (88 %) der gelungenen Zerlegungen liegen um mehr als 10 % daneben
    assert 0.2 < float(np.median([s.median_err for s in sets])) < 0.9            # README: Median des Fehlers 0,45
    assert all(s.pred_bad == s.ok for s in sets)                                 # README: in 156 von 156 gelungenen Zerlegungen weicht die Vorhersage um mehr als 1e-3 ab
    assert all(s.cond_lower == s.m for s in sets)                                # cond ist hier nur Untergrenze (4,5e15)


def test_falle_smaller_bandwidths_fail_less_and_err_less():
    med = {}
    for sg in (1.0, 2.0, 3.0):
        sets = _sets(20, sg, 0.0)
        med[sg] = float(np.median([s.median_err for s in sets]))
        assert all(s.ok >= 22 for s in sets)                                     # README: σ = 3: 290 von 300; σ = 2: 298 von 300; σ = 1: 300 von 300
    assert 3e-8 < med[1.0] < 1e-6 and 5e-5 < med[2.0] < 2e-3 and 3e-3 < med[3.0] < 6e-2          # README: Median des Fehlers 1,8e-7 / 2,7e-4 / 1,3e-2
    assert med[1.0] < med[2.0] < med[3.0]


@pytest.mark.parametrize("lam,max_err", [(1e-10, 1e-5), (1e-6, 1e-9), (1e-2, 1e-13)])
def test_falle_with_a_nugget_all_succeed_with_small_errors(lam, max_err):
    for s in _sets(20, 5.0, lam, BASES[:2]):
        assert s.ok == s.m == 30 and s.bad_3 == 0 and s.max_err < max_err and s.max_eta < 2.2e-16     # README: Fehler höchstens 3,2e-6 / 3,9e-10 / 4,5e-14 in 300 Punktmengen


# ---------------------------------------------------------------- B: Nugget
def test_nugget_bound_and_digits_rule_hold_in_every_instance_of_a_reduced_run():
    total = 0
    for sg in (0.5, 2.0, 5.0):
        for r in X.nugget_reihe(20, sg, 10, 2000):
            total += r.count
            assert r.cond_violations == 0 and r.digits_violations == 0 and r.digits_min > r.rule        # README: 450 von 450 Instanzen; kleinste Stellen 5,43 / 9,44 / 13,39 gegen R = 4,70 / 8,70 / 12,70
    assert total == 90


def test_nugget_changes_the_model_best_at_a_middle_lambda():
    rows, const = X.nugget_testfehler(40, 5.0, 3000, seeds=C.RIDGE_INSTANZEN)
    mean = {r.lam: r.mean for r in rows}
    assert mean[1e-4] < mean[1e-2] < mean[0.1] < const                           # README: Satz-Mittel 0,18 / 0,29 / 0,34 gegen 0,78
    assert mean[1e-8] > mean[1e-4] and 0.7 < const < 0.85


def test_solver_does_not_change_the_test_error_but_lambda_zero_is_far_above_the_constant():
    rows, const = X.solver_vs_model(40, 5.0, 3100, seeds=4, n_test=100)
    by = {r.lam: r for r in rows}
    for lam in (1e-8, 1e-4):
        assert by[lam].exact_mean == pytest.approx(by[lam].lu_mean, rel=1e-3)    # README: bei λ = 1e-8 und 1e-4 gleich in den ersten drei Stellen
    assert by[0.0].exact_mean > 100 * const and by[0.0].lu_mean > 100 * const    # README: bei λ = 0 beide mit Mittel 1e4 bis 2e5 gegen 0,78


# ---------------------------------------------------------------- C: Fläche
def test_surface_valley_best_cell_and_loo_choice():
    hits, best, loo, worst = 0, [], [], []
    for base in (4000, 4001, 4002):
        sf = X.surface(60, 0.3, 30, base)
        hits += int(sf.best_cell == (C.FLAECHE_SIGMAS.index(1.6), C.FLAECHE_LAMS.index(1e-2)))
        best.append(sf.test_mean[sf.best_cell[0]][sf.best_cell[1]])
        loo.append(sf.loo_mean)
        worst.append(sf.test_mean[3][0] / sf.const_mean)
        assert sf.test_mean[6][0] < 0.12 and sf.test_mean[6][-1] > 0.6          # README: σ = 6,4: 0,0795 bei λ = 1e-6, 0,682 bei λ = 10
    assert hits >= 2                                                             # README: beste Zelle (1,6; 0,01) in 19 von 20 Sätzen
    assert all(0.04 < b < 0.07 for b in best) and all(b < l < 0.11 for b, l in zip(best, loo))   # README: beste Zelle 0,0455 bis 0,0653; LOO-gewählt 0,055 bis 0,094
    assert all(10 < w < 50 for w in worst)                                       # README: Zelle (0,8; 1e-6) beim 18,5- bis 40fachen der Konstanten


# ---------------------------------------------------------------- D: Interpolation, Leave-one-out
def test_interpolation_sigma_one_is_far_above_the_constant_in_every_instance_and_loo_is_far_below():
    for j in range(4):
        s = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-13, 5000, j))
        assert s.above_const == s.count == 30                                    # README: 1800 von 1800 (60 Sätze)
        assert 50 < s.interp_median < 4000 and s.interp_mean > s.interp_median   # README: Satz-Median 150 bis 2892 (Median der Sätze 449)
        assert s.loo_median < 0.1 and s.cut_median == 0                          # README: Satz-Median der LOO-Wahl 0,045 bis 0,063; kein Eigenwert abgeschnitten


def test_sigma_small_is_not_catastrophic_and_sigma_three_is():
    s03 = [V.summarize_ridge(X.ridge_set(0.3, 60, 1e-13, 5000, j)) for j in range(4)]
    assert all(s.interp_median < s.const_mean for s in s03)                      # README: Satz-Median 0,29 bis 0,50 gegen 0,78
    assert sum(s.above_const for s in s03) < 0.4 * 4 * 30                        # README: über der Konstanten in 288 von 1800 (16 %)
    s3 = [V.summarize_ridge(X.ridge_set(3.0, 60, 1e-13, 5000, j)) for j in range(4)]
    assert all(s.interp_median > 3 * s.const_mean for s in s3)                   # README: Satz-Median 4,4 bis 25
    assert sum(s.above_const for s in s3) >= 0.9 * 4 * 30                        # README: 1750 von 1800


def test_loo_choice_is_a_small_lambda_not_the_interpolation():
    s = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-13, 5000, 0))
    chosen = dict(s.lambda_counts)
    assert sum(chosen.values()) == 30 and chosen.get(0.1, 0) + chosen.get(0.01, 0) >= 25      # README: σ = 1: 0,1 in 1486 und 0,01 in 304 von 1800


def test_the_truncation_threshold_decides_the_number():
    lo = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-13, 5000, 0))
    hi = V.summarize_ridge(X.ridge_set(1.0, 60, 1e-5, 5000, 0))
    assert hi.interp_mean < 1.5 and lo.interp_mean > 100                          # README: τ = 1e-5: Satz-Mittel 0,48; τ = 1e-13: 5394


# ---------------------------------------------------------------- vorgerechnete Zahlen (gkd_frozen.json)
def test_frozen_streuung_bands_and_the_set_is_not_the_number():
    z = FROZEN["streuung"]
    assert z["sets"] == 300 and z["per_set"] == 30 and z["instances"] == 9000
    assert z["above_const"] == 9000                                              # README: in 9000 von 9000 Instanzen über der konstanten Vorhersage
    assert 150 < z["median_lo"] < 250 and 1000 < z["median_hi"] < 1500 and 350 < z["median_med"] < 600      # README: Satz-Median 197 bis 1232 (Median der Sätze 457)
    assert 800 < z["mean_lo"] < 1500 and 5e4 < z["mean_hi"] < 1.5e5 and 3500 < z["mean_med"] < 8000          # README: Satz-Mittel 1080 bis 87652 (Median 5398)
    assert 0.04 < z["loo_median_lo"] < z["loo_median_med"] < z["loo_median_hi"] < 0.07                     # README: LOO Satz-Median 0,045 bis 0,063 (0,053)
    assert z["mean_med"] > 5 * z["median_med"]                                   # das Mittel hängt am Tail: nie ohne Median und Maximum nennen
    assert z["max_all"] > 10 * z["max_med"]                                      # größter Einzelwert weit über dem Median der Satz-Maxima


def test_frozen_n_series_is_not_monotone_and_the_cut_matches_the_drop():
    rows = FROZEN["n_reihe"]
    med = {int(n): r["median_med"] for n, r in rows.items()}
    assert med[20] < med[40] < med[60] < med[120] and med[240] < med[120] / 10      # README: 1,8 / 31 / 460 / 3130 / 26: steigt bis n = 120, fällt bei n = 240
    assert rows["240"]["cut_median"] > 50 and rows["120"]["cut_median"] < 20 and rows["60"]["cut_median"] == 0     # README: abgeschnitten 90 von 240 bei n = 240, 6,5 bei n = 120


def test_frozen_tau_series_sigma_one_falls_with_tau_and_sigma_small_does_not_depend_on_it():
    z = FROZEN["tau_reihe"]
    mean1 = [z["1.0"]["tau"][f"{t:g}"]["mean_med"] for t in C.TAUS]
    assert all(a >= b for a, b in zip(mean1, mean1[1:])) and mean1[0] > 1000 and mean1[-1] < 1.0       # README: 5394 bis 0,48
    mean03 = {z["0.3"]["tau"][f"{t:g}"]["mean_med"] for t in C.TAUS[:-1]}
    assert len(mean03) == 1                                                       # σ = 0,3: bis τ = 1e-7 wird nichts abgeschnitten, gleiche Zahl
    assert z["3.0"]["cut_1e-13"] > 10 and z["1.0"]["cut_1e-13"] == 0              # README: σ = 3 schneidet bei 1e-13 17 von 60 Eigenwerten ab, σ = 1 keinen


def test_frozen_exact_interpolation_equals_floating_point_except_possibly_in_the_extreme_tail():
    g = FROZEN["exakt_gleit"]
    assert g["seeds"] >= 40 and g["ratio_close"] >= g["seeds"] - 2                # README: exakte und Gleitkomma-Interpolation im Testfehler gleich (Verhältnis 0,9 bis 1,1)
    assert 1e7 < g["norm_median"] < 1e9 and 6 < g["label_norm_median"] < 8        # README: ‖a_exakt‖ im Median 6,5e7 gegen Etiketten-Norm 7,1
    assert g["median_exact"] == pytest.approx(g["median_float"], rel=1e-2)


def test_frozen_file_check_values_are_reproduced_by_the_tool():
    sys.path.insert(0, str(ROOT / "tools"))
    import gkd_make_frozen as G
    stored, now = FROZEN["check"], G.check_values()
    assert set(stored) == set(now)
    for key in stored:
        a, b = np.atleast_1d(stored[key]), np.atleast_1d(now[key])
        assert a == pytest.approx(b, rel=1e-6), key                               # gleiche Zufalls-Ströme, gleiche Rechnung: bis auf BLAS-Rundung gleich


# ---------------------------------------------------------------- E: Rundungsboden (Zählung echter Eigenwerte)
def test_real_eigenvalue_count_does_not_rise_with_sigma_and_more_points_buy_little_rank():
    def count(maker, n, sg, b=6000):
        X_ = maker(n, S.sub_seed(C.KIND_ROUND, b, n))
        ev = Kn.eigenvalues(Kn.gauss_gram(X_, X_, sg))
        return int((ev > 1e-16 * ev[0]).sum())
    for maker in (S.points_line, S.points_square):
        for n in (20, 50):
            counts = [count(maker, n, sg) for sg in (0.05, 0.2, 1.0, 3.0)]
            assert all(a >= b for a, b in zip(counts, counts[1:])), (maker.__name__, n, counts)        # nicht steigend in σ
    c20, c100 = count(S.points_line, 20, 3.0), count(S.points_line, 100, 3.0)
    assert c20 <= c100 <= c20 + 12                                                # README: 1D, σ = 3: 6 bis 7 bei n = 20, 13 bei n = 100


# ---------------------------------------------------------------- F: Leave-one-out in Gleitkomma gegen exakt
def test_loo_in_floating_point_chooses_the_same_lambda_as_the_exact_computation():
    import gkd_exact as E
    import gkd_fit as Fit
    grid = [1e-10, 1e-6, 1e-2, 1.0]
    same = total = 0
    for sg in (1.0, 3.0):
        for s in range(5):
            Xs, y = S.regression_data(12, 0.3, S.sub_seed(C.KIND_ROUND, 7000, 100 * int(sg) + s))
            yc, K = y - y.mean(), Kn.gauss_gram(Xs, Xs, sg)
            ex, fl = [], []
            for lam in grid:
                sol = E.exact_solve_multi(K + lam * np.eye(12), np.column_stack([yc, np.eye(12)]))
                ex.append(float(np.mean([float(sol[0][i] / sol[1 + i][i]) ** 2 for i in range(12)])))
                fl.append(Fit.loo_mse_from_eig(K, *np.linalg.eigh(K), yc, lam))
            total += 1
            same += int(int(np.argmin(ex)) == int(np.argmin(fl)))
    assert same == total == 10                                                    # README: 60 von 60 (n = 20, 8 Werte von λ)


# ---------------------------------------------------------------- Flat Limit: exakte Zahlen stehen in test_flat.py; hier ihre Bänder der README
def test_flat_limit_ratio_and_condition_bands_of_the_readme():
    r8 = Fl.analyse_flat((0, 1, 2, 3, 4), 8)
    assert all(abs(v - 1) < 1e-6 for v in r8.ratio_exact_ladder) and r8.cond_exact == pytest.approx(9.1146e31, rel=1e-4)        # README: Verhältnis 1,000000, cond 9,1146e31
    r2 = Fl.analyse_flat((0, 1, 2, 3, 4), 2)
    assert all(0.9 < v < 1.0 for v in r2.ratio_exact_ladder) and r2.sigma == pytest.approx(7.05, abs=0.005)                   # README: 0,93 bis 0,99 bei k = 2 (σ = 7,05)


# ---------------------------------------------------------------- Testzahl der README
def test_the_test_count_in_the_readme_equals_the_collected_tests():
    if os.environ.get("GKD_COUNTING"):
        pytest.skip("läuft innerhalb der Zählung")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, GKD_COUNTING="1"), timeout=300)
    m = re.search(r"(\d+) tests? collected", res.stdout)
    assert m, res.stdout[-300:]
    assert f"**{m.group(1)} Tests**" in README, f"README nennt nicht {m.group(1)} Tests"
