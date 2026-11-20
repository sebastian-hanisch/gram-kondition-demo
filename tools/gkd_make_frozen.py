"""Rechnet die vorgerechneten Zahlen der App (gkd_frozen.json) und prüft sie.

    python tools/gkd_make_frozen.py            # rechnet alles neu und schreibt gkd_frozen.json (einige Minuten)
    python tools/gkd_make_frozen.py --check    # rechnet eine kleine Teilmenge neu und vergleicht mit den Prüfwerten in der Datei (rel. 1e-6, Band wegen BLAS/LAPACK)

Zähleinheiten: Sätze zu je 30 Zufalls-Seeds (Instanzen), Eigenwerte. Alle Zufallszahlen aus SplitMix64 (gkd_rng.py). `OMP_NUM_THREADS=1` setzen.
Keine Sekunden als Befund: die Laufzeit steht nirgends in der Datei."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gkd_constants as C  # noqa: E402
import gkd_exact as E  # noqa: E402
import gkd_experiments as X  # noqa: E402
import gkd_fit as Fit  # noqa: E402
import gkd_kernel as Kn  # noqa: E402
import gkd_scenario as S  # noqa: E402

OUT = ROOT / C.FROZEN_FILE
FROZEN_BASE = 777            # Grund-Zufalls-Seed der vorgerechneten Sätze (nicht der Standard der Seitenleiste)
CHECK_SETS = 6               # Sätze der Prüfung


def pct(v, q):
    return float(np.percentile(np.asarray(v), q))


def streuung(sets: int) -> dict:
    """σ = 1, n = 60, τ = 1e-13: Kennzahl je Satz (Median, Mittel, Maximum der Interpolation) über `sets` frische Sätze zu je 30 Zufalls-Seeds."""
    med, mean, mx, loo_med, above, tot = [], [], [], [], 0, 0
    for j in range(sets):
        rs = X.ridge_set(1.0, 60, 1e-13, FROZEN_BASE, j)
        ip = np.array(rs.interp)
        med.append(float(np.median(ip)))
        mean.append(float(ip.mean()))
        mx.append(float(ip.max()))
        loo_med.append(float(np.median(rs.loo)))
        above += int((ip > np.array(rs.const)).sum())
        tot += len(ip)
    return {
        "sets": sets, "per_set": C.RIDGE_INSTANZEN, "median_lo": pct(med, 2.5), "median_hi": pct(med, 97.5), "median_med": float(np.median(med)),
        "median_min": min(med), "median_max": max(med),
        "mean_lo": pct(mean, 2.5), "mean_hi": pct(mean, 97.5), "mean_med": float(np.median(mean)), "max_med": float(np.median(mx)), "max_all": max(mx),
        "loo_median_med": float(np.median(loo_med)), "loo_median_lo": pct(loo_med, 2.5), "loo_median_hi": pct(loo_med, 97.5),
        "above_const": above, "instances": tot,
    }


def n_reihe(sets: int) -> dict:
    """σ = 1, τ = 1e-13: je n der Median der Satz-Mediane (und 95-%-Band), der Median der Satz-Mittel, abgeschnittene Eigenwerte (Median)."""
    rows = {}
    for n in C.NS_RIDGE:
        med, mean, cut = [], [], []
        for j in range(sets):
            rs = X.ridge_set(1.0, n, 1e-13, FROZEN_BASE + n, j)
            ip = np.array(rs.interp)
            med.append(float(np.median(ip)))
            mean.append(float(ip.mean()))
            cut.append(float(np.median(rs.cut)))
        rows[str(n)] = {"sets": sets, "median_med": float(np.median(med)), "median_lo": pct(med, 2.5), "median_hi": pct(med, 97.5), "mean_med": float(np.median(mean)),
                        "cut_median": float(np.median(cut))}
    return rows


def tau_reihe(sets: int) -> dict:
    """n = 60: je σ und τ der Median der Satz-Mittel und der Satz-Mediane über `sets` Sätze, dazu die konstante Vorhersage (Median der Satz-Mittel)."""
    Xte, fte = S.reg_test_points()
    out = {}
    for sg in C.SIGMAS_RIDGE:
        per_tau_mean = {t: [] for t in C.TAUS}
        per_tau_med = {t: [] for t in C.TAUS}
        cuts, consts = [], []
        for j in range(sets):
            errs = {t: [] for t in C.TAUS}
            cs, cc = [], []
            for s in range(C.RIDGE_INSTANZEN):
                Xs, y = S.regression_data(60, C.RIDGE_NOISE, X.ridge_instance_seed(FROZEN_BASE + 60, j, s))
                ym = float(y.mean())
                K, Kt = Kn.gauss_gram(Xs, Xs, sg), Kn.gauss_gram(Xte, Xs, sg)
                ev, V = np.linalg.eigh(K)
                for t in C.TAUS:
                    a, nc = Fit.interpolate_from_eig(ev, V, y - ym, t)
                    errs[t].append(float(np.mean((Kt @ a + ym - fte) ** 2)))
                    if t == 1e-13:
                        cs.append(nc)
                cc.append(float(np.mean((ym - fte) ** 2)))
            for t in C.TAUS:
                per_tau_mean[t].append(float(np.mean(errs[t])))
                per_tau_med[t].append(float(np.median(errs[t])))
            cuts.append(float(np.median(cs)))
            consts.append(float(np.mean(cc)))
        out[str(sg)] = {"const": float(np.median(consts)), "cut_1e-13": float(np.median(cuts)),
                        "tau": {f"{t:g}": {"mean_med": float(np.median(per_tau_mean[t])), "median_med": float(np.median(per_tau_med[t]))} for t in C.TAUS}, "sets": sets}
    return out


def exakt_gleit(seeds: int) -> dict:
    """Ist die Rundung die Ursache? n = 60, σ = 1: Testfehler der EXAKTEN Interpolation (Bareiss, Vorhersage in Brüchen) gegen die Eigenzerlegung mit τ = 1e-13, je Zufalls-Seed."""
    Xte, fte = S.reg_test_points()
    ratios, norms, ex, fl, labels = [], [], [], [], []
    for s in range(seeds):
        Xs, y = S.regression_data(60, C.RIDGE_NOISE, X.ridge_instance_seed(FROZEN_BASE + 9, 9000, s))
        ym = float(y.mean())
        K, Kt = Kn.gauss_gram(Xs, Xs, 1.0), Kn.gauss_gram(Xte, Xs, 1.0)
        sol = E.exact_solve_fractions(K, y - ym)
        e_ex = float(np.mean((E.exact_predict(Kt, sol) + ym - fte) ** 2))
        a, _ = Fit.interpolate_truncated(K, y - ym, 1e-13)
        e_fl = float(np.mean((Kt @ a + ym - fte) ** 2))
        ratios.append(e_ex / e_fl)
        labels.append(float(np.linalg.norm(y - ym)))
        norms.append(math.sqrt(sum(float(v) ** 2 for v in sol)))
        ex.append(e_ex)
        fl.append(e_fl)
    r, nrm = np.array(ratios), np.array(norms)
    close = (r > 0.9) & (r < 1.1)
    small = nrm < 1e10
    return {
        "seeds": seeds, "ratio_close": int(close.sum()), "ratio_min": float(r.min()), "ratio_max": float(r.max()), "norm_median": float(np.median(nrm)), "norm_max": float(nrm.max()),
        "small_norm": int(small.sum()), "small_norm_close": int((close & small).sum()), "large_norm": int((~small).sum()),
        "label_norm_median": float(np.median(labels)), "median_exact": float(np.median(ex)), "median_float": float(np.median(fl)), "mean_exact": float(np.mean(ex)), "mean_float": float(np.mean(fl)),
    }


def check_values() -> dict:
    """Kleine, schnelle Teilmenge, die `--check` neu rechnet."""
    rs = [X.ridge_set(1.0, 60, 1e-13, FROZEN_BASE, j) for j in range(CHECK_SETS)]
    ip = np.array([np.median(r.interp) for r in rs])
    r20 = X.ridge_set(1.0, 20, 1e-13, FROZEN_BASE + 20, 0)
    eg = exakt_gleit(3)
    return {"streuung_set_medians": [float(v) for v in ip], "n20_median": float(np.median(r20.interp)), "exakt_gleit_ratios": [eg["ratio_min"], eg["ratio_max"]]}


def build(full: bool = True) -> dict:
    data = {
        "meta": {"zweck": "vorgerechnete Zahlen der App (Experiment 5); erzeugt von tools/gkd_make_frozen.py; Zufallsquelle SplitMix64; Gauß-Kern, Beispiel-Regression, n und σ wie angegeben",
                 "grund_seed": FROZEN_BASE},
        "streuung": streuung(300 if full else 12),
        "n_reihe": n_reihe(60 if full else 4),
        "tau_reihe": tau_reihe(60 if full else 4),
        "exakt_gleit": exakt_gleit(200 if full else 4),
        "check": check_values(),
    }
    return data


def close(a, b, rel=1e-6):
    return all(math.isclose(x, y, rel_tol=rel, abs_tol=1e-12) for x, y in zip(a, b)) and len(a) == len(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Prüfwerte neu rechnen und mit der Datei vergleichen")
    ap.add_argument("--quick", action="store_true", help="nur zum Ausprobieren: kleine Sätze, schreibt in eine andere Datei")
    ap.add_argument("--only", choices=["streuung", "n_reihe", "tau_reihe", "exakt_gleit"], help="nur diesen Teil neu rechnen und in die bestehende Datei einsetzen")
    args = ap.parse_args()
    if args.check:
        stored = json.loads(OUT.read_text(encoding="utf-8"))["check"]
        now = check_values()
        ok = all(close(stored[k], now[k]) if isinstance(stored[k], list) else math.isclose(stored[k], now[k], rel_tol=1e-6) for k in stored)
        print("gkd_frozen.json: Prüfwerte " + ("stimmen" if ok else "WEICHEN AB") + f" ({len(stored)} Größen)")
        sys.exit(0 if ok else 1)
    if args.only:
        data = json.loads(OUT.read_text(encoding="utf-8"))
        data[args.only] = {"streuung": lambda: streuung(300), "n_reihe": lambda: n_reihe(60), "tau_reihe": lambda: tau_reihe(60), "exakt_gleit": lambda: exakt_gleit(200)}[args.only]()
        data["check"] = check_values()
        OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"ersetzt: {args.only}")
        return
    data = build(full=not args.quick)
    target = OUT if not args.quick else OUT.with_name("gkd_frozen_quick.json")
    target.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"geschrieben: {target.name}")


if __name__ == "__main__":
    main()
