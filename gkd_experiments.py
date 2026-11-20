"""Rechnungen der fünf Experimente (die App ruft sie, mit st.cache_data, auf; die Tests und die Messskripte rufen sie ebenso).

Zähleinheiten: Punktmengen (Instanzen, je eine eigene y-Ziehung), Zufalls-Seeds, Eigenwerte. Alle Zufallszahlen aus SplitMix64 (gkd_scenario.py).
„Exakt“ heißt: exakte Lösung des Gleitkomma-Systems (gkd_exact.py, Bareiss in ganzen Zahlen)."""

import math
from dataclasses import dataclass

import numpy as np

import gkd_constants as C
import gkd_exact as E
import gkd_fit as Fit
import gkd_kernel as Kn
import gkd_scenario as S


# =============================================================================================================================================
# Experiment 2: die Falle (Cholesky gelingt, Lösung falsch)
# =============================================================================================================================================
@dataclass(frozen=True)
class FalleInstance:
    exact_singular: bool            # die exakte Elimination fand eine exakt singuläre Matrix: kein Vergleich möglich
    cholesky_ok: bool
    err_chol: float                 # relativer Koeffizientenfehler gegen die exakte Lösung (inf, wenn der Weg nichts liefert)
    err_lu: float
    err_eigen: float
    eta_chol: float                 # exakter normweiser Rückwärtsfehler (nan, wenn Cholesky scheitert)
    eta_lu: float
    pred_chol: float                # relativer Vorhersagefehler gegen die exakte Vorhersage an N_TEST_PRED Testpunkten (nan, wenn Cholesky scheitert)
    pred_lu: float
    cond: float
    cond_lower_bound: bool          # cond nur Untergrenze (kleinster Eigenwert unter der Rundungsgrenze)
    n_real: int                     # Eigenwerte über 1e-16 · λ_max
    eigenvalues: tuple              # absteigend (für die Anzeige)


def falle_seeds(base: int, index: int) -> tuple[int, int]:
    """Zufalls-Seeds (Punkte, y) der Punktmenge Nummer index."""
    return S.sub_seed(C.KIND_FALLE, base, 2 * index), S.sub_seed(C.KIND_FALLE, base, 2 * index + 1)


def falle_instance(n: int, sigma: float, lam: float, base: int, index: int) -> FalleInstance:
    sp, sy = falle_seeds(base, index)
    X = S.points_square(n, sp)
    y = S.normal_targets(n, sy)
    K = Kn.gauss_gram(X, X, sigma)
    A = K + lam * np.eye(n)
    eigs = Kn.eigenvalues(K)
    cond, lower = Kn.condition(eigs, lam)
    n_real = int((eigs > 1e-16 * eigs[0]).sum())
    try:
        sol = E.exact_solve_fractions(A, y)
    except ZeroDivisionError:
        return FalleInstance(True, False, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan, cond, lower, n_real, tuple(float(v) for v in eigs))
    a_ex = np.array([float(v) for v in sol])
    Kt = Kn.gauss_gram(S.square_test_points(), X, sigma)
    p_ex = E.exact_predict(Kt, sol)
    nrm_p = float(np.linalg.norm(p_ex))
    a_lu, a_ch, a_eg = Fit.solve_lu(A, y), Fit.solve_cholesky(A, y), Fit.solve_eigen(A, y)

    def pred_err(a):
        return math.nan if a is None or nrm_p == 0 else float(np.linalg.norm(Kt @ a - p_ex) / nrm_p)

    return FalleInstance(
        exact_singular=False, cholesky_ok=a_ch is not None,
        err_chol=Fit.relative_error(a_ch, a_ex) if a_ch is not None else math.nan,
        err_lu=Fit.relative_error(a_lu, a_ex), err_eigen=Fit.relative_error(a_eg, a_ex),
        eta_chol=E.backward_error(A, a_ch, y) if a_ch is not None else math.nan,
        eta_lu=E.backward_error(A, a_lu, y) if a_lu is not None else math.nan,
        pred_chol=pred_err(a_ch), pred_lu=pred_err(a_lu),
        cond=cond, cond_lower_bound=lower, n_real=n_real, eigenvalues=tuple(float(v) for v in eigs),
    )


def falle_zaehlung(n: int, sigma: float, lam: float, m: int, base: int) -> tuple[FalleInstance, ...]:
    """Die Zählung: m Punktmengen (Nummer 0 bis m-1) mit je eigener y-Ziehung."""
    return tuple(falle_instance(n, sigma, lam, base, i) for i in range(m))


# =============================================================================================================================================
# Experiment 3: das Nugget als Schranke
# =============================================================================================================================================
@dataclass(frozen=True)
class NuggetRow:
    lam: float
    bound: float                    # 1 + n/λ
    rule: float                     # R(λ) = 16 - log10(1 + n/λ)
    count: int                      # Instanzen mit exakter Referenz
    cond_violations: int            # cond(K + λ I) über der Schranke
    digits_violations: int          # beobachtete Stellen (LU) unter R(λ)
    digits_min: float
    digits_median: float
    cond_max: float


def nugget_reihe(n: int, sigma: float, m: int, base: int, lams=C.NUGGET_LAMS) -> tuple[NuggetRow, ...]:
    """Je λ der Reihe: Schranke und beobachtete Stellen der LU gegen die exakte Lösung über m Punktmengen (dieselben Punktmengen wie Experiment 2)."""
    rows = []
    for lam in lams:
        bound, rule = Kn.nugget_cond_bound(n, lam), Kn.digits_rule(n, lam)
        digits, conds = [], []
        viol = 0
        for i in range(m):
            sp, sy = falle_seeds(base, i)
            X, y = S.points_square(n, sp), S.normal_targets(n, sy)
            K = Kn.gauss_gram(X, X, sigma)
            A = K + lam * np.eye(n)
            cond, _ = Kn.condition(Kn.eigenvalues(K), lam)
            conds.append(cond)
            viol += int(cond > bound * (1 + 1e-9))
            try:
                a_ex = E.exact_solve(A, y)
            except ZeroDivisionError:
                continue
            a_lu = Fit.solve_lu(A, y)
            digits.append(-math.log10(max(Fit.relative_error(a_lu, a_ex), 1e-17)))
        d = np.array(digits)
        rows.append(NuggetRow(lam, bound, rule, len(d), viol, int((d < rule).sum()), float(d.min()), float(np.median(d)), float(max(conds))))
    return tuple(rows)


@dataclass(frozen=True)
class ModelRow:
    lam: float
    mean: float
    median: float
    maximum: float


def nugget_testfehler(n: int, sigma: float, base: int, seeds: int = C.RIDGE_INSTANZEN, lams=C.NUGGET_TEST_LAMS) -> tuple[tuple[ModelRow, ...], float]:
    """Gegenbefund: ein Nugget ändert das Modell. Beispiel-Regression (Rauschen sd 0,3, Etiketten zentriert), Testfehler gegen die Wahrheit auf 1000 rauschfreien Testpunkten;
    Mittel, Median und Maximum über `seeds` Zufalls-Seeds je λ und der Fehler der konstanten Vorhersage (Mittel)."""
    Xte, fte = S.reg_test_points()
    errs = {lam: [] for lam in lams}
    const = []
    for s in range(seeds):
        X, y = S.regression_data(n, C.RIDGE_NOISE, S.sub_seed(C.KIND_NUGGET, base, s))
        ym = float(y.mean())
        K = Kn.gauss_gram(X, X, sigma)
        Kt = Kn.gauss_gram(Xte, X, sigma)
        const.append(float(np.mean((ym - fte) ** 2)))
        for lam in lams:
            a = Fit.solve_lu(K + lam * np.eye(n), y - ym)
            errs[lam].append(float(np.mean((Kt @ a + ym - fte) ** 2)) if a is not None else math.nan)
    rows = tuple(ModelRow(lam, float(np.nanmean(v)), float(np.nanmedian(v)), float(np.nanmax(v))) for lam, v in errs.items())
    return rows, float(np.mean(const))


@dataclass(frozen=True)
class SolverVsModel:
    lam: float
    exact_mean: float
    exact_median: float
    lu_mean: float
    lu_median: float


def solver_vs_model(n: int, sigma: float, base: int, seeds: int = 10, n_test: int = 200, lams=(0.0, 1e-8, 1e-4)) -> tuple[tuple[SolverVsModel, ...], float]:
    """Rechenfehler gegen Modellfehler: Testfehler gegen die Wahrheit der EXAKTEN Lösung und der LU (Beispiel-Regression, `seeds` Zufalls-Seeds, `n_test` Testpunkte,
    Etiketten zentriert). Ändert der Löser den Testfehler nicht, dominiert der Modellfehler. Gibt auch den Fehler der konstanten Vorhersage (Mittel)."""
    Xte, fte = S.reg_test_points(n_test)
    ex = {lam: [] for lam in lams}
    lu = {lam: [] for lam in lams}
    const = []
    for s in range(seeds):
        X, y = S.regression_data(n, C.RIDGE_NOISE, S.sub_seed(C.KIND_NUGGET, base, 1000 + s))
        ym = float(y.mean())
        K = Kn.gauss_gram(X, X, sigma)
        Kt = Kn.gauss_gram(Xte, X, sigma)
        const.append(float(np.mean((ym - fte) ** 2)))
        for lam in lams:
            A = K + lam * np.eye(n)
            try:
                sol = E.exact_solve_fractions(A, y - ym)
            except ZeroDivisionError:
                continue
            ex[lam].append(float(np.mean((E.exact_predict(Kt, sol) + ym - fte) ** 2)))
            a = Fit.solve_lu(A, y - ym)
            lu[lam].append(float(np.mean((Kt @ a + ym - fte) ** 2)) if a is not None else math.nan)
    rows = tuple(SolverVsModel(lam, float(np.mean(ex[lam])), float(np.median(ex[lam])), float(np.nanmean(lu[lam])), float(np.nanmedian(lu[lam]))) for lam in lams)
    return rows, float(np.mean(const))


# =============================================================================================================================================
# Experiment 5: Interpolation ist ein Objekt der Abschneidegrenze (Modellfehler)
# =============================================================================================================================================
@dataclass(frozen=True)
class RidgeSet:
    """Ein Satz von RIDGE_INSTANZEN Instanzen (Zufalls-Seeds); alle Testfehler gegen die Wahrheit (Mittel über 1000 rauschfreie Testpunkte)."""
    interp: tuple
    loo: tuple
    const: tuple
    cut: tuple                  # abgeschnittene Eigenwerte je Instanz bei τ
    chosen_lambda: tuple        # per Leave-one-out gewähltes λ je Instanz


def ridge_instance_seed(base: int, set_index: int, s: int) -> int:
    return S.sub_seed(C.KIND_RIDGE, base, set_index * C.RIDGE_INSTANZEN + s)


def ridge_set(sigma: float, n: int, tau: float, base: int, set_index: int = 0, count: int = C.RIDGE_INSTANZEN) -> RidgeSet:
    Xte, fte = S.reg_test_points()
    interp, loo, const, cut, chosen = [], [], [], [], []
    for s in range(count):
        X, y = S.regression_data(n, C.RIDGE_NOISE, ridge_instance_seed(base, set_index, s))
        ym = float(y.mean())
        yc = y - ym
        K = Kn.gauss_gram(X, X, sigma)
        Kt = Kn.gauss_gram(Xte, X, sigma)
        ev, V = np.linalg.eigh(K)
        a, nc = Fit.interpolate_from_eig(ev, V, yc, tau)
        interp.append(float(np.mean((Kt @ a + ym - fte) ** 2)))
        cut.append(nc)
        lam, a2 = Fit.choose_lambda_from_eig(K, ev, V, yc, C.LOO_GRID)
        loo.append(float(np.mean((Kt @ a2 + ym - fte) ** 2)))
        chosen.append(lam)
        const.append(float(np.mean((ym - fte) ** 2)))
    return RidgeSet(tuple(interp), tuple(loo), tuple(const), tuple(cut), tuple(chosen))


# =============================================================================================================================================
# Experiment 1: die Fläche σ × λ
# =============================================================================================================================================
@dataclass(frozen=True)
class SurfaceResult:
    sigmas: tuple
    lams: tuple
    test_mean: tuple            # [i][j] Mittel des Testfehlers über die Zufalls-Seeds, Zeile = σ, Spalte = λ
    test_median: tuple
    const_mean: float           # Fehler der konstanten Vorhersage (Mittel über die Zufalls-Seeds)
    best_cell: tuple            # (i, j) der Zelle mit dem kleinsten mittleren Testfehler
    loo_mean: float             # Mittel des Testfehlers der je Zufalls-Seed per Leave-one-out gewählten Zelle
    hindsight_mean: float       # Mittel des Testfehlers der je Zufalls-Seed im Rückblick besten Zelle
    seeds: int


def surface(n: int, noise: float, seeds: int, base: int, sigmas=C.FLAECHE_SIGMAS, lams=C.FLAECHE_LAMS) -> SurfaceResult:
    Xte, fte = S.reg_test_points()
    ns, nl = len(sigmas), len(lams)
    test = np.zeros((seeds, ns, nl))
    loo_err = np.zeros((seeds, ns, nl))
    const = []
    for s in range(seeds):
        X, y = S.regression_data(n, noise, S.sub_seed(C.KIND_FLAECHE, base, s))
        ym = float(y.mean())
        yc = y - ym
        const.append(float(np.mean((ym - fte) ** 2)))
        D2, D2t = Kn.sqdist(X, X), Kn.sqdist(Xte, X)
        for i, sg in enumerate(sigmas):
            K, Kt = Kn.gauss_from_sqdist(D2, sg), Kn.gauss_from_sqdist(D2t, sg)
            ev, V = np.linalg.eigh(K)
            for j, lam in enumerate(lams):
                a, h = Fit.ridge_from_eig(ev, V, yc, lam)
                test[s, i, j] = float(np.mean((Kt @ a + ym - fte) ** 2))
                loo_err[s, i, j] = float(np.mean(((yc - K @ a) / (1.0 - h)) ** 2))
    loo_err = np.where(np.isfinite(loo_err), loo_err, np.inf)
    mean = test.mean(0)
    bi, bj = np.unravel_index(int(np.argmin(mean)), mean.shape)
    loo_pick = [np.unravel_index(int(np.argmin(loo_err[s])), (ns, nl)) for s in range(seeds)]
    loo_test = [test[s][loo_pick[s]] for s in range(seeds)]
    best = [test[s].min() for s in range(seeds)]
    return SurfaceResult(
        tuple(sigmas), tuple(lams), tuple(tuple(float(v) for v in row) for row in mean), tuple(tuple(float(v) for v in row) for row in np.median(test, 0)),
        float(np.mean(const)), (int(bi), int(bj)), float(np.mean(loo_test)), float(np.mean(best)), seeds)
