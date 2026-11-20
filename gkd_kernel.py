"""Gauß-Gram-Matrix, Eigenwerte, Konditionszahl, Pivots und die Nugget-Schranke (nur numpy).

Konvention (Linie Kernmethoden): Gauß-Kern k(x, x') = exp(-||x - x'||² / (2 σ²)) mit der Bandbreite σ (Längenmaß der Eingabe); γ = 1/(2 σ²).
Die Matrix wird aus **Differenzen** der Punkte gebaut (keine Auslöschung durch ||x||² + ||x'||² - 2 x·x'); das ist der Aufbau, an dem die Messungen dieser
Demo gelten (andere Aufbauten, etwa als Potenzen eines gerundeten q, haben ihre Rundungsgrenze an anderer Stelle: README)."""

import math

import numpy as np

EPS = float(np.finfo(float).eps)        # 2,2e-16
INV_EPS = 1.0 / EPS                     # 4,5e15: obere Grenze für eine Konditionszahl, die nur als Untergrenze bekannt ist


def sqdist(X, Y) -> np.ndarray:
    """Quadrierte euklidische Abstände aus Differenzen; X: m x d, Y: n x d (oder Vektoren der Länge m und n bei d = 1)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    D = X[:, None, :] - Y[None, :, :]
    return (D ** 2).sum(-1)


def gauss_gram(X, Y, sigma: float) -> np.ndarray:
    """exp(-d²/(2 σ²)); σ muss positiv sein."""
    if not sigma > 0:
        raise ValueError(f"Bandbreite σ muss positiv sein, nicht {sigma}")
    return np.exp(-sqdist(X, Y) / (2.0 * sigma ** 2))


def gauss_from_sqdist(D2, sigma: float) -> np.ndarray:
    """exp(-D2/(2 σ²)) aus vorberechneten quadrierten Abständen (dieselbe Matrix wie gauss_gram, wenn D2 = sqdist)."""
    if not sigma > 0:
        raise ValueError(f"Bandbreite σ muss positiv sein, nicht {sigma}")
    return np.exp(-np.asarray(D2, float) / (2.0 * sigma ** 2))


def gamma_of_sigma(sigma: float) -> float:
    return 1.0 / (2.0 * sigma ** 2)


def eigenvalues(K) -> np.ndarray:
    """Eigenwerte der symmetrischen Matrix, absteigend (numpy.linalg.eigvalsh)."""
    return np.linalg.eigvalsh(np.asarray(K, float))[::-1]


def condition(eigs, lam: float = 0.0) -> tuple[float, bool]:
    """Konditionszahl von K + λ I aus den Eigenwerten (absteigend): (λ_max + λ)/(λ_min + λ).
    Liegt λ_min + λ unter der Rundungsgrenze eps · λ_max, ist der Wert nur eine **Untergrenze** (zweiter Wert True)."""
    eigs = np.asarray(eigs, float)
    top = float(eigs[0])
    low = float(eigs[-1]) + lam
    floor = EPS * top
    lower_bound = not low > floor
    return (top + lam) / max(low, floor), lower_bound


def nugget_cond_bound(n: int, lam: float) -> float:
    """cond(K + λ I) <= 1 + n/λ: gilt exakt, weil λ_max(K) <= Spur K = n (k(x, x) = 1) und λ_min(K + λ I) >= λ (K ist positiv semidefinit)."""
    if not lam > 0:
        raise ValueError("die Schranke braucht λ > 0")
    return 1.0 + n / lam


def digits_rule(n: int, lam: float) -> float:
    """Faustregel R(λ) = 16 - log10(1 + n/λ): so viele Dezimalstellen bleiben mindestens richtig (Eigenschaft der beobachteten Stellen, kein Satz)."""
    return 16.0 - math.log10(nugget_cond_bound(n, lam))


def ldl_pivots(A) -> np.ndarray:
    """Pivots der LDL^T-Zerlegung ohne Vertauschung in Gleitkomma (Quadrate der Diagonale der Cholesky-Zerlegung, wenn sie gelingt). Ein Pivot <= 0 heißt: die Matrix
    ist (nach der Rundung) nicht positiv definit; die Rechnung läuft trotzdem weiter, damit man sieht, wo es kippt."""
    M = np.array(A, float)
    n = M.shape[0]
    piv = np.empty(n)
    for k in range(n):
        piv[k] = M[k, k]
        if piv[k] == 0.0 or not np.isfinite(piv[k]):
            piv[k + 1:] = np.nan
            break
        if k + 1 < n:
            f = M[k + 1:, k] / piv[k]
            M[k + 1:, k + 1:] -= np.outer(f, M[k, k + 1:])
    return piv


def cholesky_or_none(A):
    """Cholesky-Faktor L (A = L L^T) oder None, wenn numpy.linalg.cholesky scheitert (nicht positiv definit nach der Rundung)."""
    try:
        return np.linalg.cholesky(np.asarray(A, float))
    except np.linalg.LinAlgError:
        return None
