"""Kennzahlen aus den Rechnungen und die Verdikte der App als reine Funktionen: `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`,
`tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten. Jede Aussage nennt ihre Voraussetzung (Kern, d, n, Zähleinheit)."""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import gkd_constants as C
from gkd_format import de, lam_text, sci, sigma_text, tau_text

FROZEN_PATH = Path(__file__).resolve().parent / C.FROZEN_FILE


def load_frozen() -> dict:
    """Die vorgerechneten Zahlen (tools/gkd_make_frozen.py); leeres Wörterbuch, wenn die Datei fehlt."""
    try:
        return json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


# =============================================================================================================================================
# Experiment 2: Zusammenfassung der Zählung
# =============================================================================================================================================
@dataclass(frozen=True)
class FalleSummary:
    m: int                      # Punktmengen mit exakter Referenz
    excluded: int               # Punktmengen, in denen die exakte Matrix singulär war
    ok: int                     # Cholesky gelingt
    bad_10: int                 # davon Koeffizientenfehler über 10 %
    bad_3: int                  # davon Fehler über 1e-3
    median_err: float           # Median des Fehlers der Cholesky-Lösung (nan ohne gelungene Zerlegung)
    max_err: float              # Stichprobenmaximum
    max_eta: float              # größter exakter Rückwärtsfehler der Cholesky-Lösungen
    median_err_lu: float
    pred_bad: int               # gelungene Zerlegungen, deren Vorhersage um mehr als 1e-3 von der exakten abweicht
    cond_lower: int             # Punktmengen, in denen cond nur Untergrenze ist
    median_cond: float
    median_real: float          # Eigenwerte über 1e-16 · λ_max (Median über die Punktmengen)
    n: int


def summarize_falle(instances, n: int) -> FalleSummary:
    used = [i for i in instances if not i.exact_singular]
    ok = [i for i in used if i.cholesky_ok]
    errs = np.array([i.err_chol for i in ok])
    etas = np.array([i.eta_chol for i in ok])
    return FalleSummary(
        m=len(used), excluded=len(instances) - len(used), ok=len(ok),
        bad_10=int((errs > C.ERR_BAD).sum()), bad_3=int((errs > C.ERR_SMALL).sum()),
        median_err=float(np.median(errs)) if len(errs) else math.nan, max_err=float(errs.max()) if len(errs) else math.nan,
        max_eta=float(etas.max()) if len(etas) else math.nan,
        median_err_lu=float(np.median([i.err_lu for i in used])) if used else math.nan,
        pred_bad=sum(1 for i in ok if i.pred_chol > C.ERR_SMALL),
        cond_lower=sum(1 for i in used if i.cond_lower_bound),
        median_cond=float(np.median([i.cond for i in used])) if used else math.nan,
        median_real=float(np.median([i.n_real for i in used])) if used else math.nan,
        n=n,
    )


def verdict_eigenwerte(s: FalleSummary, sigma: float) -> tuple[str, str]:
    """Kernfrage 1: Wie viele Eigenwerte hat die Gram-Matrix noch? (Median über die Punktmengen der Zählung.)"""
    if s.m == 0 or math.isnan(s.median_real):
        return "info", "Ohne Punktmenge mit exakter Referenz gibt es hier keine Eigenwerte zu zählen."
    r = int(s.median_real)
    if r >= s.n:
        return "success", (f"Alle {s.n} Eigenwerte der Gram-Matrix (σ = {sigma_text(sigma)}, Punkte im Einheitsquadrat, Median über {s.m} Punktmengen) liegen über 1e-16 · λ_max: "
                           "bei dieser Bandbreite liegt im Median kein Eigenwert unter der Rundungsgrenze.")
    return "warning", (f"Von {s.n} Eigenwerten der Gram-Matrix (σ = {sigma_text(sigma)}, Punkte im Einheitsquadrat) liegen im Median über {s.m} Punktmengen nur noch {r} über 1e-16 · λ_max; "
                       f"die übrigen {s.n - r} sind an die Rundung verloren.")


def verdict_falle(s: FalleSummary, sigma: float, lam: float) -> tuple[str, str]:
    """Kernfrage 2: Gelingt die Rechnung, und stimmt sie? Zählung über alle Punktmengen, nie eine Einzelinstanz allein."""
    role = "ohne Nugget" if lam == 0 else "als Nugget, Rechenhilfe"
    head = f"n = {s.n} Punkte im Einheitsquadrat, σ = {sigma_text(sigma)}, λ = {lam_text(lam)} ({role}), {s.m} Punktmengen"
    if s.m == 0:
        return "warning", f"{head}: in jeder Punktmenge war die Matrix exakt singulär, es gibt keine exakte Referenz."
    if s.ok == 0:
        return "info", (f"{head}: Cholesky scheitert in allen {s.m} Punktmengen und meldet das selbst (LinAlgError): hier gibt es keinen stillen Fehler, die Matrix ist nach der Rundung nicht "
                        "mehr positiv definit.")
    if s.bad_10 > 0:
        eta = (f" Der exakt gerechnete Rückwärtsfehler ist in jeder gelungenen Zerlegung höchstens {sci(s.max_eta)}, also unter der Rundungsgenauigkeit: das Residuum verrät den Fehler nicht."
               if s.max_eta < 1e-14 else f" Der größte exakt gerechnete Rückwärtsfehler ist {sci(s.max_eta)}.")
        return "warning", (f"{head}: Cholesky gelingt in {s.ok} von {s.m}; in {s.bad_10} davon liegt die Lösung um mehr als 10 % von der exakten Lösung des Gleitkomma-Systems entfernt "
                           f"(Median des Fehlers {de(s.median_err)}, Stichprobenmaximum {de(s.max_err)}).{eta}")
    if s.bad_3 > 0:
        return "info", (f"{head}: Cholesky gelingt in {s.ok} von {s.m}; in {s.bad_3} davon weicht die Lösung um mehr als 1e-3 ab, in keiner um mehr als 10 % "
                        f"(Median des Fehlers {sci(s.median_err)}, Stichprobenmaximum {sci(s.max_err)}).")
    return "success", (f"{head}: Cholesky gelingt in {s.ok} von {s.m}; der Fehler gegen die exakte Lösung des Gleitkomma-Systems ist in jeder gelungenen Zerlegung höchstens {sci(s.max_err)} "
                       f"(Median {sci(s.median_err)}).")


def verdict_vorhersage(s: FalleSummary) -> tuple[str, str]:
    """Die Koeffizientenfehler wirken auf die Vorhersage: Zählung der gelungenen Zerlegungen mit Vorhersagefehler über 1e-3 (relativ, 100 Testpunkte)."""
    if s.ok == 0:
        return "info", "Ohne gelungene Cholesky-Zerlegung gibt es hier keine Vorhersage zu vergleichen."
    if s.pred_bad == 0:
        return "success", f"In keiner der {s.ok} gelungenen Zerlegungen weicht die Vorhersage an {C.N_TEST_PRED} Testpunkten um mehr als 1e-3 (relativ) von der exakten Vorhersage ab."
    return "warning", (f"In {s.pred_bad} von {s.ok} gelungenen Zerlegungen weicht die Vorhersage an {C.N_TEST_PRED} Testpunkten um mehr als 1e-3 (relativ) von der exakten Vorhersage ab: "
                       "falsche Koeffizienten bei richtiger Vorhersage, diese Hoffnung bestätigt sich hier nicht.")


# =============================================================================================================================================
# Experiment 3: Nugget
# =============================================================================================================================================
def verdict_nugget(rows, n: int, sigma: float) -> tuple[str, str]:
    """Kernfrage 3: Was kauft ein Nugget? Schranke cond(K + λI) ≤ 1 + n/λ und beobachtete Stellen der LU gegen R(λ) = 16 − log10(1 + n/λ)."""
    total = sum(r.count for r in rows)
    vc, vd = sum(r.cond_violations for r in rows), sum(r.digits_violations for r in rows)
    per = "; ".join(f"λ = {lam_text(r.lam)}: mindestens {de(r.digits_min, 3)} Stellen (R = {de(r.rule, 3)})" for r in rows)
    head = f"n = {n}, σ = {sigma_text(sigma)}, {total} Instanzen ({len(rows)} Werte von λ)"
    if vc == 0 and vd == 0:
        return "success", f"{head}: die Schranke cond(K + λI) ≤ 1 + n/λ wird in keiner Instanz verletzt, und die beobachteten Stellen der LU liegen in keiner unter R(λ). {per}."
    return "warning", (f"{head}: die Schranke wird in {vc} Instanzen verletzt, die beobachteten Stellen liegen in {vd} Instanzen unter R(λ). {per}.")


def verdict_nugget_modell(rows, const: float, n: int, sigma: float, seeds: int) -> tuple[str, str]:
    """Gegenbefund: ein Nugget ändert das Modell (Testfehler gegen die Wahrheit, Beispiel-Regression)."""
    parts = " / ".join(f"{de(r.mean, 3)} ({de(r.median, 3)})" for r in rows)
    lams = " / ".join(lam_text(r.lam) for r in rows)
    return "info", (f"Ein Nugget ändert das Modell: bei σ = {sigma_text(sigma)}, n = {n} (Beispiel-Regression, {seeds} Zufalls-Seeds, Testfehler gegen die Wahrheit) liegt der Mittelwert (Median) "
                    f"bei λ = {lams} bei {parts}; die konstante Vorhersage liegt bei {de(const, 3)}.")


def verdict_solver_vs_model(rows, const: float, n: int, sigma: float, seeds: int) -> tuple[str, str]:
    """Rechenfehler gegen Modellfehler: ändert der Löser den Testfehler gegen die Wahrheit?"""
    pos = [r for r in rows if r.lam > 0]
    zero = [r for r in rows if r.lam == 0]
    same = bool(pos) and all(math.isclose(r.exact_mean, r.lu_mean, rel_tol=1e-3) for r in pos)
    lams = " und ".join(lam_text(r.lam) for r in pos)
    text = f"Beispiel-Regression, n = {n}, σ = {sigma_text(sigma)}, {seeds} Zufalls-Seeds: "
    if same:
        text += f"bei λ = {lams} ändert der Löser den Mittelwert des Testfehlers nicht (exakte Lösung gegen LU in den ersten drei Stellen gleich)"
    else:
        text += f"bei λ = {lams} unterscheiden sich exakte Lösung und LU im Testfehler"
    if zero:
        z = zero[0]
        both = z.exact_mean > const and z.lu_mean > const
        text += (f"; bei λ = 0 liegen exakte Lösung und LU im Mittel bei {de(z.exact_mean, 2)} und {de(z.lu_mean, 2)} (Median {de(z.exact_median, 2)} und {de(z.lu_median, 2)}), "
                 f"{'beide über' if both else 'nicht beide über'} der konstanten Vorhersage ({de(const, 3)})")
    return "info", text + ". Der Modellfehler der Interpolation ist von der Rechnung getrennt zu lesen."


# =============================================================================================================================================
# Experiment 4: Flat Limit
# =============================================================================================================================================
def verdict_flat(r) -> tuple[str, str]:
    """Flat Limit: Zahl echter Eigenwerte (Zähleinheit: Eigenwerte) in der Gleitkomma-Zerlegung und in der gerundeten Matrix; Leiter als Vorhersage."""
    n = len(r.xs)
    dev = max(abs(v - 1.0) for v in r.ratio_exact_ladder)
    head = (f"{n} ganzzahlige Punkte x = {tuple(r.xs)}, q = 1 − 1e-{r.k} (σ ≈ {de(r.sigma, 4)}, Gauß-Kern, d = 1): von {n} Eigenwerten sind in der Gleitkomma-Eigenzerlegung "
            f"{r.real_float} echt (Abweichung gegen die exakte rationale Matrix unter 10 %), in der auf Doppelte gerundeten Matrix {r.real_rounded}; die Leiter sagt {r.predicted} voraus. ")
    tail = f"Exakter Eigenwert gegen die Leiter c_m (1/σ)^(2(m−1)): größte Abweichung des Verhältnisses von 1 gleich {sci(dev)}."
    neg = " Der kleinste Eigenwert der gerundeten Matrix ist negativ: die gerundete Gram-Matrix ist nicht mehr positiv semidefinit." if r.smallest_rounded_negative else ""
    kind = "success" if r.real_float == n else "warning"
    return kind, head + tail + neg


def verdict_flat_hinweis(r) -> tuple[str, str]:
    """Eingaberundung statt Löser: die Zahl echter Eigenwerte in F und D im Vergleich."""
    if r.real_float == r.real_rounded:
        return "info", (f"Die Eigenzerlegung verliert gegenüber der gerundeten Matrix nichts zusätzlich ({r.real_float} echte Eigenwerte in beiden): die Grenze setzt das Runden der Matrixeinträge, "
                        "bevor ein Löser läuft; sie liegt innerhalb eines Eigenwerts und hängt an Schwelle und Plattform.")
    return "info", (f"Gleitkomma-Eigenzerlegung {r.real_float} echte Eigenwerte, gerundete Matrix {r.real_rounded}: bei diesen Werten weicht die Eigenzerlegung um "
                    f"{abs(r.real_float - r.real_rounded)} Eigenwert(e) von der gerundeten Matrix ab (die Grenze liegt innerhalb eines Eigenwerts und hängt an der 10-%-Schwelle und der Plattform).")


# =============================================================================================================================================
# Experiment 5: Interpolation und Abschneidegrenze
# =============================================================================================================================================
@dataclass(frozen=True)
class RidgeSummary:
    const_mean: float
    above_const: int            # Instanzen, in denen die Interpolation über der konstanten Vorhersage liegt
    count: int
    cut_median: float           # abgeschnittene Eigenwerte (Median über die Instanzen)
    lambda_counts: tuple        # ((λ, Zahl), ...) der per Leave-one-out gewählten λ
    interp_mean: float
    interp_median: float
    interp_max: float
    loo_mean: float
    loo_median: float
    loo_max: float


def summarize_ridge(rs) -> RidgeSummary:
    ip, lo, cn = np.array(rs.interp), np.array(rs.loo), np.array(rs.const)
    counts = Counter(rs.chosen_lambda)
    return RidgeSummary(
        const_mean=float(cn.mean()), above_const=int((ip > cn).sum()), count=len(ip),
        cut_median=float(np.median(rs.cut)), lambda_counts=tuple(sorted(counts.items())),
        interp_mean=float(ip.mean()), interp_median=float(np.median(ip)), interp_max=float(ip.max()),
        loo_mean=float(lo.mean()), loo_median=float(np.median(lo)), loo_max=float(lo.max()),
    )


def verdict_ridge(s: RidgeSummary, sigma: float, n: int, tau: float, wahl: str) -> tuple[str, str]:
    """Kernfrage 4: Wovon hängt „Interpolation“ ab? Immer Mittel, Median und Maximum; der Satz von 30 Zufalls-Seeds ist ein Satz, nicht „die“ Zahl."""
    base = f"Beispiel-Regression, n = {n} Punkte, σ = {sigma_text(sigma)}, ein Satz von {s.count} Zufalls-Seeds"
    if wahl == C.WAHLEN[1]:
        lam = ", ".join(f"{lam_text(l)} in {c}" for l, c in s.lambda_counts)
        return "info", (f"{base}: mit per Leave-one-out gewähltem λ (Raster 1e-6 bis 10; gewählt: {lam}) liegt der Testfehler gegen die Wahrheit bei Mittel {de(s.loo_mean)}, Median {de(s.loo_median)}, "
                        f"Maximum {de(s.loo_max)} (konstante Vorhersage {de(s.const_mean)}); die Interpolation derselben Instanzen (τ = {tau_text(tau)}) liegt bei Mittel {de(s.interp_mean)}, "
                        f"Median {de(s.interp_median)}, Maximum {de(s.interp_max)}.")
    over = f"über der konstanten Vorhersage in {s.above_const} von {s.count} Zufalls-Seeds"
    text = (f"{base}: die Interpolation (Eigenwerte unter τ = {tau_text(tau)} · λ_max abgeschnitten, im Median {int(s.cut_median)} von {n}) hat einen Testfehler gegen die Wahrheit von "
            f"Mittel {de(s.interp_mean)}, Median {de(s.interp_median)}, Maximum {de(s.interp_max)} gegen {de(s.const_mean)} der konstanten Vorhersage; {over}.")
    if s.interp_mean > s.const_mean:
        return "warning", text + f" Mit Leave-one-out gewähltem λ liegt der Median bei {de(s.loo_median)}."
    return "info", text + f" Hier ist die Interpolation im Mittel nicht schlechter als die Konstante; mit Leave-one-out gewähltem λ liegt der Median bei {de(s.loo_median)}."


def verdict_streuung(frozen: dict) -> tuple[str, str]:
    """Die vorgerechnete Streuung über frische Sätze (σ = 1, n = 60, τ = 1e-13): die Aussage ist ein Band, nie die Zahl eines Satzes."""
    z = frozen.get("streuung")
    if not z:
        return "info", "Die vorgerechnete Streuung über frische Sätze liegt nicht vor (Datei gkd_frozen.json fehlt)."
    return "info", (f"Vorgerechnet, {z['sets']} frische Sätze zu je {z['per_set']} Zufalls-Seeds ({z['sets'] * z['per_set']} Instanzen; σ = 1, n = 60, τ = 1e-13): Satz-Median der Interpolation "
                    f"{de(z['median_lo'], 3)} bis {de(z['median_hi'], 3)} (95-%-Band, Median der Sätze {de(z['median_med'], 3)}), Satz-Mittel {de(z['mean_lo'], 3)} bis {de(z['mean_hi'], 3)}; "
                    f"über der konstanten Vorhersage in {z['above_const']} von {z['sets'] * z['per_set']} Instanzen. Mit Leave-one-out gewähltem λ liegt der Satz-Median bei {de(z['loo_median_med'], 2)}.")


# =============================================================================================================================================
# Experiment 1: die Fläche
# =============================================================================================================================================
def verdict_flaeche(sf, n: int, noise: float) -> tuple[str, str]:
    """Die Fläche σ × λ ist ein Tal: beste Zelle, schlechteste Zelle gegen die konstante Vorhersage, Leave-one-out-Wahl; Negativbefund zur großen Bandbreite."""
    bi, bj = sf.best_cell
    best = sf.test_mean[bi][bj]
    flat = [(sf.test_mean[i][j], i, j) for i in range(len(sf.sigmas)) for j in range(len(sf.lams))]
    worst, wi, wj = max(flat)
    big_small_lam = sf.test_mean[len(sf.sigmas) - 1][0]
    big_big_lam = sf.test_mean[len(sf.sigmas) - 1][-1]
    return "info", (f"Beispiel-Regression, n = {n}, Rauschen {de(noise, 2)}, {sf.seeds} Zufalls-Seeds: beste Zelle σ = {sigma_text(sf.sigmas[bi])}, λ = {lam_text(sf.lams[bj])} (Testfehler {de(best)}); "
                    f"schlechteste Zelle σ = {sigma_text(sf.sigmas[wi])}, λ = {lam_text(sf.lams[wj])} mit {de(worst)}, dem {de(worst / sf.const_mean, 3)}fachen der konstanten Vorhersage ({de(sf.const_mean)}); "
                    f"mit Leave-one-out je Zufalls-Seed gewählt {de(sf.loo_mean)} (je Zufalls-Seed im Rückblick beste Zelle {de(sf.hindsight_mean)}). Große Bandbreite heißt nicht Unteranpassung: "
                    f"σ = {sigma_text(sf.sigmas[-1])} liegt bei λ = {lam_text(sf.lams[0])} bei {de(big_small_lam)}, bei λ = {lam_text(sf.lams[-1])} bei {de(big_big_lam)}.")


def verdict_exakt_gleit(frozen: dict) -> tuple[str, str]:
    """Ist es die Rundung? Exakte Interpolation (Bareiss, Vorhersage in Brüchen) gegen die Eigenzerlegung bei τ = 1e-13: im Median nein, in der Spitze der Verteilung ja."""
    z = frozen.get("exakt_gleit")
    if not z:
        return "info", "Die vorgerechnete Gegenprobe exakt gegen Gleitkomma liegt nicht vor (Datei gkd_frozen.json fehlt)."
    text = (f"Ist es die Rundung? Vorgerechnet, n = 60, σ = 1, {z['seeds']} frische Zufalls-Seeds: die exakte Interpolation liefert in {z['ratio_close']} von {z['seeds']} Zufalls-Seeds denselben Testfehler "
            f"wie die Eigenzerlegung bei τ = 1e-13 (Verhältnis zwischen 0,9 und 1,1); die Norm der exakten Koeffizienten liegt im Median bei {sci(z['norm_median'], 1)} "
            f"(Norm der Labels etwa {de(z['label_norm_median'], 2)}). Die Rundung ist im Median und in Zufalls-Seeds mit ‖a‖ unter 1e10 nicht die Ursache; die Ursache ist die Konditionszahl "
            "(das Rauschen in y wird mit 1/λ_klein verstärkt). ")
    large_close = z["ratio_close"] - z["small_norm_close"]                       # Zufalls-Seeds mit ‖a‖ ≥ 1e10, in denen der Testfehler trotzdem gleich bleibt
    span = f"Verhältnis {z['ratio_min']:.4f} bis {z['ratio_max']:.4f} über alle {z['seeds']} Zufalls-Seeds".replace(".", ",")
    if z["large_norm"] and large_close < z["large_norm"]:
        text += (f"In {z['large_norm'] - large_close} von {z['large_norm']} Zufalls-Seeds mit ‖a‖ ≥ 1e10 (größte Norm {sci(z['norm_max'], 1)}) weichen exakte und Gleitkomma-Rechnung ab ({span}): "
                 "in der Spitze der Verteilung wirkt das Abschneiden bei 1e-13 als Regularisierung.")
    elif z["large_norm"]:
        text += f"Auch in den {z['large_norm']} Zufalls-Seed(s) mit ‖a‖ ≥ 1e10 (größte Norm {sci(z['norm_max'], 1)}) bleibt der Testfehler gleich ({span})."
    else:
        text += f"In dieser Messung gab es keinen Zufalls-Seed mit ‖a‖ ≥ 1e10 (größte Norm {sci(z['norm_max'], 1)})."
    return "info", text


# =============================================================================================================================================
# Tabellen für die App (Zeilen aus fertigen Zeichenketten: Dezimalkomma, keine Locale-Abhängigkeit)
# =============================================================================================================================================
def paths_table(instances) -> list[dict]:
    """Fehler der drei Gleitkomma-Wege gegen die exakte Lösung, über die Punktmengen (nur Punktmengen mit exakter Referenz)."""
    used = [i for i in instances if not i.exact_singular]
    rows = []
    for name, key in (("Cholesky", "err_chol"), ("LU (numpy.linalg.solve)", "err_lu"), ("Eigenzerlegung ohne Abschneiden", "err_eigen")):
        vals = np.array([getattr(i, key) for i in used], float)
        fin = vals[np.isfinite(vals)]
        rows.append({"Weg": name, "Ergebnis in": f"{len(fin)} von {len(used)}", "Median des Fehlers": sci(float(np.median(fin))) if len(fin) else "kein Ergebnis",
                     "Fehler über 1e-3": f"{int((fin > C.ERR_SMALL).sum())} von {len(fin)}", "Fehler über 10 %": f"{int((fin > C.ERR_BAD).sum())} von {len(fin)}"})
    return rows


def nugget_table(rows) -> list[dict]:
    return [{"λ": lam_text(r.lam), "Schranke 1 + n/λ": sci(r.bound), "größtes cond": sci(r.cond_max), "R(λ) = 16 − log10(1 + n/λ)": de(r.rule, 3),
             "Stellen (kleinster Wert / Median)": f"{de(r.digits_min, 3)} / {de(r.digits_median, 3)}", "Schranke verletzt": f"{r.cond_violations} von {r.count}",
             "Stellen unter R": f"{r.digits_violations} von {r.count}"} for r in rows]


def model_table(rows) -> list[dict]:
    return [{"λ": lam_text(r.lam), "Testfehler, Mittel": de(r.mean, 3), "Median": de(r.median, 3), "Maximum": de(r.maximum, 3)} for r in rows]


def flat_table(r) -> list[dict]:
    out = []
    for m in range(len(r.xs)):
        out.append({"Nr. m": m + 1, "exakt (rationale Matrix)": sci(r.exact[m], 3), "Leiter c_m (1/σ)^(2(m−1))": sci(r.ladder[m], 3),
                    "Verhältnis exakt / Leiter": de(r.ratio_exact_ladder[m], 6), "gerundete Matrix (exakt gerechnet)": sci(r.rounded[m], 3),
                    "Gleitkomma-Eigenzerlegung": sci(r.floating[m], 3),
                    "echt in Gleitkomma": "ja" if abs(r.floating[m] / r.exact[m] - 1.0) < C.ECHT_SCHWELLE else "nein"})
    return out


def tau_table(frozen: dict) -> list[dict]:
    z = frozen.get("tau_reihe")
    if not z:
        return []
    rows = []
    for sg, d in z.items():
        row = {"σ": sigma_text(float(sg))}
        for t in C.TAUS:
            cell = d["tau"][f"{t:g}"]
            row[f"τ = {tau_text(t)}"] = f"{de(cell['mean_med'], 3)} ({de(cell['median_med'], 3)})"
        row["abgeschnitten bei 1e-13"] = de(d["cut_1e-13"], 3)
        row["konstante Vorhersage"] = de(d["const"], 3)
        rows.append(row)
    return rows
