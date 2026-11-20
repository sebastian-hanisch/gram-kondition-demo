"""Messskript der README: alle Zahlen mit dem Zufalls-Strom dieser App (SplitMix64), als Band über frische Zufalls-Seeds. Schreibt tools/out_gkd_messung.txt.

    OMP_NUM_THREADS=1 python tools/gkd_messung.py [Abschnitt ...]     # A B C D E F G (Standard: alle)

Zähleinheiten: Punktmengen (Instanzen; je eine eigene y-Ziehung), Sätze zu je 30 Zufalls-Seeds, Eigenwerte. Messung dieser Plattform (LAPACK von numpy), keine Sekunden als Befund.
Die Tests (tests/test_claims.py) prüfen Bänder, die aus dieser Ausgabe stammen; eine Zahl ohne Test steht in der README als „Messung dieser README, nicht im Test“."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gkd_constants as C  # noqa: E402
import gkd_exact as E  # noqa: E402
import gkd_experiments as X  # noqa: E402
import gkd_fit as Fit  # noqa: E402
import gkd_flat as Fl  # noqa: E402
import gkd_kernel as Kn  # noqa: E402
import gkd_scenario as S  # noqa: E402
from gkd_evaluation import summarize_falle  # noqa: E402

LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def band(v):
    v = list(v)
    return f"{min(v):g} bis {max(v):g} (Median {float(np.median(v)):g})"


def sec_a():
    out("A. Cholesky gelingt (n = 20, Punkte im Einheitsquadrat, y ~ N(0, I)): 10 Sätze zu je 30 Punktmengen = 300 Punktmengen je Zelle; Grund-Zufalls-Seeds 1000 bis 1009")
    out("   sigma | lambda | gelingt (Summe von 300) | je Satz von 30: Band | davon Fehler > 0,1 (Summe) | > 1e-3 | Median Fehler | Stichprobenmaximum je Satz (Band) | größtes eta | Vorhersage > 1e-3 (Summe) | cond nur Untergrenze (Summe)")
    for sg, lam in ((1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (5.0, 0.0), (5.0, 1e-10), (5.0, 1e-6), (5.0, 1e-2)):
        oks, b10, b3, errs, maxs, etas, pred, low = [], 0, 0, [], [], [], 0, 0
        for base in range(1000, 1010):
            ins = X.falle_zaehlung(20, sg, lam, 30, base)
            s = summarize_falle(ins, 20)
            oks.append(s.ok)
            b10 += s.bad_10
            b3 += s.bad_3
            pred += s.pred_bad
            low += s.cond_lower
            e = [i.err_chol for i in ins if i.cholesky_ok]
            errs += e
            maxs.append(max(e) if e else float("nan"))
            etas.append(s.max_eta)
        out(f"   {sg:g} | {lam:g} | {sum(oks)} | {band(oks)} | {b10} | {b3} | {float(np.median(errs)):.2e} | {min(maxs):.2g} bis {max(maxs):.2g} | {np.nanmax(etas):.1e} | {pred} | {low}")
    out("   Kontrolle der Fehlerverteilung bei sigma = 5, lambda = 0: Anteil der Cholesky-gelungenen Punktmengen mit Fehler > 0,1")
    allok = allbad = 0
    for base in range(1000, 1010):
        s = summarize_falle(X.falle_zaehlung(20, 5.0, 0.0, 30, base), 20)
        allok += s.ok
        allbad += s.bad_10
    out(f"   {allbad} von {allok}")


def sec_b():
    out("\nB. Nugget (n = 20): 5 Bandbreiten sigma = 0,5 / 1 / 2 / 3 / 5, je 30 Punktmengen, drei Werte von lambda = 450 Instanzen; Grund-Zufalls-Seed 2000")
    tot = vc = vd = 0
    mins = {lam: [] for lam in C.NUGGET_LAMS}
    for sg in (0.5, 1.0, 2.0, 3.0, 5.0):
        for r in X.nugget_reihe(20, sg, 30, 2000):
            tot += r.count
            vc += r.cond_violations
            vd += r.digits_violations
            mins[r.lam].append(r.digits_min)
    out(f"   Instanzen {tot}; cond über der Schranke 1 + n/lambda: {vc}; beobachtete Stellen unter R(lambda): {vd}")
    for lam in C.NUGGET_LAMS:
        out(f"   lambda = {lam:g}: R = {Kn.digits_rule(20, lam):.2f}; kleinste beobachtete Stellen über die 5 Bandbreiten: {min(mins[lam]):.2f}")
    out("   Gegenbefund (Beispiel-Regression, 30 Zufalls-Seeds, Testfehler gegen die Wahrheit), n = 40, sigma = 5, lambda = 1e-8 / 1e-4 / 1e-2 / 0,1: Mittel (Median) / Maximum; je 10 Grund-Zufalls-Seeds 3000 bis 3009, Band der Mittel")
    rows = {lam: [] for lam in C.NUGGET_TEST_LAMS}
    meds = {lam: [] for lam in C.NUGGET_TEST_LAMS}
    consts = []
    for base in range(3000, 3010):
        rr, const = X.nugget_testfehler(40, 5.0, base)
        consts.append(const)
        for r in rr:
            rows[r.lam].append(r.mean)
            meds[r.lam].append(r.median)
    for lam in C.NUGGET_TEST_LAMS:
        out(f"   lambda = {lam:g}: Satz-Mittel {band(rows[lam])}; Satz-Median {band(meds[lam])}")
    out(f"   konstante Vorhersage (Satz-Mittel) {band(consts)}")
    out("   Rechenfehler gegen Modellfehler (n = 40, sigma = 5, 10 Zufalls-Seeds, 200 Testpunkte): exakt / LU, Mittel (Median), 5 Grund-Zufalls-Seeds 3100 bis 3104")
    for base in range(3100, 3105):
        rr, const = X.solver_vs_model(40, 5.0, base)
        out("   base " + str(base) + ": " + "; ".join(f"lambda = {r.lam:g}: exakt {r.exact_mean:.4g} ({r.exact_median:.4g}) / LU {r.lu_mean:.4g} ({r.lu_median:.4g})" for r in rr) + f"; Konstante {const:.4f}")


def sec_c():
    out("\nC. Fläche sigma x lambda (n = 60, Rauschen 0,3, 30 Zufalls-Seeds), 20 Grund-Zufalls-Seeds 4000 bis 4019")
    best, bv, loo, hind, const, c08, c64, c64big, worst = [], [], [], [], [], [], [], [], []
    for base in range(4000, 4020):
        sf = X.surface(60, 0.3, 30, base)
        bi, bj = sf.best_cell
        best.append((sf.sigmas[bi], sf.lams[bj]))
        bv.append(sf.test_mean[bi][bj])
        loo.append(sf.loo_mean)
        hind.append(sf.hindsight_mean)
        const.append(sf.const_mean)
        c08.append(sf.test_mean[3][0] / sf.const_mean)
        c64.append(sf.test_mean[6][0])
        c64big.append(sf.test_mean[6][-1])
        worst.append(max(max(r) for r in sf.test_mean) / sf.const_mean)
    from collections import Counter
    out(f"   beste Zelle (Zählung über 20 Sätze): {dict(Counter(best))}")
    out(f"   Testfehler der besten Zelle: {band(bv)}; LOO-gewählt: {band(loo)}; im Rückblick je Zufalls-Seed beste Zelle: {band(hind)}; konstante Vorhersage {band(const)}")
    out(f"   Zelle (0,8; 1e-6) als Vielfaches der Konstanten: {band(c08)}; schlechteste Zelle als Vielfaches der Konstanten: {band(worst)}")
    out(f"   sigma = 6,4: lambda = 1e-6: {band(c64)}; lambda = 10: {band(c64big)}")


def sec_d():
    out("\nD. Ridgeless (Interpolation) und Leave-one-out, Beispiel-Regression, n = 60: 60 Sätze zu je 30 Zufalls-Seeds je sigma, Grund-Zufalls-Seed 5000")
    for sg in (0.3, 1.0, 3.0):
        meds, means, loo_meds, loo_means, above, tot, lam_counts, consts = [], [], [], [], 0, 0, {}, []
        for j in range(60):
            rs = X.ridge_set(sg, 60, 1e-13, 5000, j)
            ip, lo = np.array(rs.interp), np.array(rs.loo)
            meds.append(float(np.median(ip)))
            means.append(float(ip.mean()))
            loo_meds.append(float(np.median(lo)))
            loo_means.append(float(lo.mean()))
            consts.append(float(np.mean(rs.const)))
            above += int((ip > np.array(rs.const)).sum())
            tot += len(ip)
            for l in rs.chosen_lambda:
                lam_counts[l] = lam_counts.get(l, 0) + 1
        out(f"   sigma = {sg:g}: Satz-Median {band(meds)}; Satz-Mittel {band(means)}; über der Konstanten in {above} von {tot} Instanzen; Satz-Median der LOO-Wahl {band(loo_meds)}; Konstante {float(np.median(consts)):.3f}; "
            f"Sätze mit Satz-Mittel unter der Konstanten: {sum(1 for m, c in zip(means, consts) if m < c)} von 60; gewähltes lambda: {dict(sorted(lam_counts.items()))}")
    out("   Abhängigkeit von tau (sigma = 1, n = 60): siehe gkd_frozen.json (tau_reihe) und n-Reihe (n_reihe)")


def sec_e():
    out("\nE. Flat Limit exakt (deterministisch, kein Zufall)")
    xs = tuple(range(5))
    out(f"   x = {xs}: c_m = {[str(c) for c in Fl.ladder_coefficients(xs)]}")
    for k in (2, 6, 8):
        r = Fl.analyse_flat(xs, k)
        out(f"   k = {k}: Verhältnis exakt / Leiter {[round(v, 6) for v in r.ratio_exact_ladder]}; cond exakt {r.cond_exact:.5e}, Leiter {r.cond_ladder:.5e}")
    xs = (0, 1, 3, 4, 7)
    out(f"   x = {xs}: c_m = {[str(c) for c in Fl.ladder_coefficients(xs)]}")
    xs = tuple(range(6))
    out("   n = 6, x = 0..5, k = 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9 / 10 / 12:")
    res = [Fl.analyse_flat(xs, k) for k in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)]
    out(f"   echt in Gleitkomma: {[r.real_float for r in res]}; gerundete Matrix: {[r.real_rounded for r in res]}; Vorhersage: {[r.predicted for r in res]}")
    out(f"   kleinster Eigenwert der gerundeten Matrix negativ: {[r.smallest_rounded_negative for r in res]}; Gleitkomma negativ: {[r.smallest_float_negative for r in res]}")
    out(f"   Cholesky gelingt: {[r.cholesky_ok for r in res]}")


def sec_f():
    out("\nF. Zahl echter Eigenwerte (Schwelle 1e-16 · lambda_max), je drei Punktmengen (Grund-Zufalls-Seeds 6000 bis 6002), Gauß-Kern")
    for dim, maker in (("1D, Punkte gleichverteilt auf [0, 1]", S.points_line), ("2D, Punkte gleichverteilt im Einheitsquadrat", S.points_square)):
        out(f"   {dim}: n = 20 / 50 / 100 (Bereich über die 3 Punktmengen)")
        for sg in (0.05, 0.2, 1.0, 3.0):
            cells = []
            for n in (20, 50, 100):
                vals = []
                for b in range(6000, 6003):
                    X_ = maker(n, S.sub_seed(C.KIND_ROUND, b, n))
                    ev = Kn.eigenvalues(Kn.gauss_gram(X_, X_, sg))
                    vals.append(int((ev > 1e-16 * ev[0]).sum()))
                cells.append(f"{min(vals)} bis {max(vals)}" if min(vals) != max(vals) else str(vals[0]))
            out(f"      sigma = {sg:g}: " + " / ".join(cells))


def sec_g():
    out("\nG. Leave-one-out in Gleitkomma gegen exakt (n = 20, Beispiel-Regression, 30 Zufalls-Seeds je sigma, 8 Werte von lambda 1e-10 bis 10)")
    grid = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    same = tot = 0
    worst = 0.0
    for sg in (1.0, 3.0):
        for s in range(30):
            Xs, y = S.regression_data(20, 0.3, S.sub_seed(C.KIND_ROUND, 7000, 1000 * int(sg * 10) + s))
            yc = y - y.mean()
            K = Kn.gauss_gram(Xs, Xs, sg)
            ex, fl = [], []
            for lam in grid:
                A = K + lam * np.eye(20)
                sol = E.exact_solve_multi(A, np.column_stack([yc, np.eye(20)]))
                ainv_y = np.array([float(v) for v in sol[0]])
                diag = np.array([float(sol[1 + i][i]) for i in range(20)])
                ex.append(float(np.mean((ainv_y / diag) ** 2)))
                fl.append(Fit.loo_mse_from_eig(K, *np.linalg.eigh(K), yc, lam))
            tot += 1
            same += int(int(np.argmin(ex)) == int(np.argmin(fl)))
            worst = max(worst, max(abs(a - b) / a for a, b in zip(ex, fl)))
    out(f"   gewähltes lambda gleich (Eigenzerlegung gegen exakt): {same} von {tot}; größter relativer Fehler der LOO-MSE {worst:.1e}")


SECTIONS = {"A": sec_a, "B": sec_b, "C": sec_c, "D": sec_d, "E": sec_e, "F": sec_f, "G": sec_g}


def main():
    names = [a.upper() for a in sys.argv[1:]] or list(SECTIONS)
    out(f"gkd_messung: numpy {np.__version__}; Zufalls-Strom SplitMix64 (gkd_rng.py); Zähleinheiten: Punktmengen, Sätze zu 30 Zufalls-Seeds, Eigenwerte")
    for nm in names:
        SECTIONS[nm]()
    target = ROOT / "tools" / ("out_gkd_messung.txt" if len(names) == len(SECTIONS) else f"out_gkd_messung_{''.join(names)}.txt")
    target.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    out(f"geschrieben: {target.name}")


if __name__ == "__main__":
    main()
