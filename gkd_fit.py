"""Rechenwege für (K + λ I) a = y und die Leave-one-out-Wahl von λ (numpy, scipy.linalg.solve_triangular).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b): die drei Gleitkomma-Wege `solve_lu`, `solve_cholesky`, `solve_eigen`,
das Abschneiden der Eigenwerte (`interpolate_truncated`), die geschlossene Leave-one-out-Formel (`loo_residuals`) und die Wahlregel (`choose_lambda_loo`).
Die exakte Referenz steht in gkd_exact.py; die Funktionen hier kennen sie nicht. Die `*_from_eig`-Varianten nehmen eine fertige Eigenzerlegung
(`numpy.linalg.eigh(K)`), damit eine Matrix für viele λ und τ nur einmal zerlegt wird."""

import numpy as np
from scipy.linalg import solve_triangular

import gkd_kernel as Kn


def solve_lu(A, y):
    """numpy.linalg.solve (LU mit Zeilenvertauschung). None, wenn numpy die Matrix als singulär meldet."""
    try:
        return np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        return None


def solve_cholesky(A, y):
    """Cholesky-Zerlegung A = L Lᵀ, dann zwei Dreieckslösungen. None, wenn die Zerlegung scheitert (die Rechnung meldet es dann selbst)."""
    L = Kn.cholesky_or_none(A)
    if L is None:
        return None
    return solve_triangular(L.T, solve_triangular(L, y, lower=True), lower=False)


def solve_eigen(A, y):
    """Eigenzerlegung A = V diag(l) Vᵀ, a = V diag(1/l) Vᵀ y ohne Abschneiden. None, wenn ein Eigenwert null ist (Ergebnis nicht endlich)."""
    lam, V = np.linalg.eigh(A)
    with np.errstate(all="ignore"):
        a = V @ ((V.T @ y) / lam)
    return a if np.all(np.isfinite(a)) else None


def interpolate_from_eig(ev, V, y, tau: float):
    """Interpolation (λ → 0): Eigenwerte unter tau · λ_max werden abgeschnitten (Pseudoinverse auf dem oberen Teil). Gibt (Koeffizienten, Zahl abgeschnittener Eigenwerte).
    tau ist die Abschneidegrenze des Verfahrens, keine Naturgröße. `ev` aufsteigend wie von numpy.linalg.eigh."""
    keep = ev > tau * ev[-1]
    a = V[:, keep] @ ((V[:, keep].T @ y) / ev[keep])
    return a, int((~keep).sum())


def interpolate_truncated(K, y, tau: float):
    ev, V = np.linalg.eigh(K)
    return interpolate_from_eig(ev, V, y, tau)


def ridge_from_eig(ev, V, y, lam: float):
    """Kernel Ridge für ein λ: (Koeffizienten, Diagonale der Hut-Matrix H = K (K + λ I)⁻¹)."""
    ev = np.maximum(ev, 0.0)
    a = V @ ((V.T @ y) / (ev + lam))
    h = np.einsum("ij,ij,j->i", V, V, ev / (ev + lam))
    return a, h


def ridge_eigen(K, y, lam: float):
    ev, V = np.linalg.eigh(K)
    return ridge_from_eig(ev, V, y, lam)


def loo_residuals(K, y, lam: float) -> np.ndarray:
    """Leave-one-out-Residuen e_i = (y_i - ŷ_i)/(1 - H_ii) (geschlossene Formel, ohne n-mal neu anzupassen); gleich [A⁻¹y]_i / [A⁻¹]_ii mit A = K + λ I."""
    a, h = ridge_eigen(K, y, lam)
    return (y - K @ a) / (1.0 - h)


def loo_mse_from_eig(K, ev, V, y, lam: float) -> float:
    a, h = ridge_from_eig(ev, V, y, lam)
    return float(np.mean(((y - K @ a) / (1.0 - h)) ** 2))


def choose_lambda_from_eig(K, ev, V, y, grid):
    """Wahlregel: das λ aus dem Raster mit dem kleinsten Leave-one-out-Fehler (bei Gleichstand das kleinere). Gibt (λ, Koeffizienten)."""
    best = None
    for lam in grid:
        e = loo_mse_from_eig(K, ev, V, y, lam)
        if best is None or e < best[0]:
            best = (e, lam)
    lam = best[1]
    a, _ = ridge_from_eig(ev, V, y, lam)
    return lam, a


def choose_lambda_loo(K, y, grid):
    ev, V = np.linalg.eigh(K)
    return choose_lambda_from_eig(K, ev, V, y, grid)


def relative_error(a, a_exact) -> float:
    """||a - a_exakt|| / ||a_exakt|| (unendlich, wenn der Weg keinen Koeffizientenvektor liefert)."""
    if a is None:
        return float("inf")
    return float(np.linalg.norm(np.asarray(a) - np.asarray(a_exact)) / np.linalg.norm(a_exact))
