"""Exakte Rechnung: die Referenz der Demo (ganze Zahlen, Brüche, Dezimalzahlen mit 120 Stellen; nur Standardbibliothek und numpy).

„Exakt“ heißt hier immer: exakt für das **Gleitkomma-System** (die Matrixeinträge sind Gleitkommazahlen, also Dualbrüche, und werden
als solche gelöst), nicht für die „wahre“ Funktion. Löser: fraktionsfreie Elimination nach Bareiss (ganze Zahlen), Rücksubstitution in Brüchen.
Eigenwerte einer Matrix aus Brüchen: zyklisches Jacobi-Verfahren in `decimal` (120 Stellen). Jede Funktion ist eine Einheit mit expliziten
Ein- und Ausgaben (DEMO-PLAYBOOK §1b); die Tests prüfen sie an Mini-Instanzen gegen `fractions` und gegen ein zweites Verfahren."""

import math
from decimal import Decimal, getcontext
from fractions import Fraction

import numpy as np

getcontext().prec = 120


# ---------------------------------------------------------------- ganze Zahlen aus Gleitkommazahlen
def scaled_ints(values) -> tuple[list[int], int]:
    """Gleitkommazahlen (beliebig geschachtelt als flache Liste) als ganze Zahlen mit gemeinsamem Zweierpotenz-Nenner: Wert = ganze Zahl / 2**shift."""
    pairs = [float(v).as_integer_ratio() for v in values]
    shift = max(d.bit_length() - 1 for _, d in pairs)          # Nenner sind Zweierpotenzen
    return [num << (shift - (d.bit_length() - 1)) for num, d in pairs], shift


# ---------------------------------------------------------------- Löser nach Bareiss
def exact_solve_multi(A, B) -> list[list[Fraction]]:
    """Löst A x = b für jede Spalte b von B exakt (Bareiss, ganze Zahlen). A: n x n, B: n x r (Gleitkomma). Gibt r Lösungsvektoren aus Brüchen.
    Wirft ZeroDivisionError bei exakt singulärem A."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if B.ndim == 1:
        B = B[:, None]
    n, r = A.shape[0], B.shape[1]
    flat, shift = scaled_ints(list(A.ravel()) + list(B.ravel()))
    M = [flat[i * n:(i + 1) * n] + flat[n * n + i * r:n * n + (i + 1) * r] for i in range(n)]   # A-Zeile gefolgt von den r rechten Seiten (gleicher Nenner)
    prev = 1
    for k in range(n - 1):
        if M[k][k] == 0:
            sw = next((i for i in range(k + 1, n) if M[i][k] != 0), None)
            if sw is None:
                raise ZeroDivisionError("exakt singulär")
            M[k], M[sw] = M[sw], M[k]
        pk, rk = M[k][k], M[k]
        for i in range(k + 1, n):
            ri = M[i]
            f = ri[k]
            for j in range(k + 1, n + r):
                ri[j] = (ri[j] * pk - f * rk[j]) // prev
            ri[k] = 0
        prev = pk
    if M[n - 1][n - 1] == 0:
        raise ZeroDivisionError("exakt singulär")
    sols = []
    for c in range(r):
        x = [Fraction(0)] * n
        for i in range(n - 1, -1, -1):
            s = Fraction(M[i][n + c])
            for j in range(i + 1, n):
                s -= M[i][j] * x[j]
            x[i] = s / M[i][i]
        sols.append(x)
    return sols


def exact_solve_fractions(A, b) -> list[Fraction]:
    """Exakte Lösung von A x = b als Brüche."""
    return exact_solve_multi(A, np.asarray(b, float)[:, None])[0]


def exact_solve(A, b) -> np.ndarray:
    """Exakte Lösung, auf Gleitkomma gerundet."""
    return np.array([float(v) for v in exact_solve_fractions(A, b)])


# ---------------------------------------------------------------- Rückwärtsfehler und Vorhersage, exakt
def exact_residual_norm(A, a, y) -> float:
    """Euklidische Norm des Residuums A a - y, das Residuum selbst exakt in ganzen Zahlen gerechnet (kein Rundungsfehler im Residuum)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    ia, sa = scaled_ints(list(A.ravel()))
    xa, sx = scaled_ints(list(a))
    yy, sy = scaled_ints(list(y))
    total = 0.0
    for i in range(n):
        s = sum(ia[i * n + j] * xa[j] for j in range(n))          # Nenner 2**(sa + sx)
        diff = Fraction(s, 1 << (sa + sx)) - Fraction(yy[i], 1 << sy)
        total += float(diff) ** 2
    return math.sqrt(total)


def backward_error(A, a, y) -> float:
    """Normweiser Rückwärtsfehler eta = ||A a - y|| / (||A||_2 ||a|| + ||y||) mit exakt gerechnetem Residuum."""
    A = np.asarray(A, float)
    return exact_residual_norm(A, a, y) / (np.linalg.norm(A, 2) * np.linalg.norm(a) + np.linalg.norm(y))


def exact_predict(Kt, coef_fr) -> np.ndarray:
    """Vorhersage Kt @ a exakt (a als Brüche), auf Gleitkomma gerundet; Kt Gleitkomma = exakte Dualbrüche."""
    Kt = np.asarray(Kt, float)
    m, n = Kt.shape
    den = 1
    for v in coef_fr:
        den = math.lcm(den, v.denominator)
    num = [v.numerator * (den // v.denominator) for v in coef_fr]
    ik, sk = scaled_ints(list(Kt.ravel()))
    out = np.empty(m)
    for t in range(m):
        s = sum(ik[t * n + j] * num[j] for j in range(n))
        out[t] = s / (den << sk)                                   # ganzzahlige Division, korrekt gerundet auch bei riesigen Zahlen
    return out


# ---------------------------------------------------------------- Eigenwerte einer Matrix aus Brüchen (Jacobi in Dezimalzahlen)
def dec(v) -> Decimal:
    """Bruch (oder Gleitkommazahl) als Decimal mit 120 Stellen."""
    f = v if isinstance(v, Fraction) else Fraction(float(v))
    return Decimal(f.numerator) / Decimal(f.denominator)


def jacobi_eigenvalues(A) -> list[Decimal]:
    """Alle Eigenwerte einer symmetrischen Matrix aus Brüchen, zyklisches Jacobi-Verfahren in `decimal` (120 Stellen), absteigend sortiert.
    Gilt auch für Matrizen, die nur bis auf Rundung positiv semidefinit sind (Eigenwerte um null, auch negative)."""
    n = len(A)
    M = [[dec(v) for v in row] for row in A]
    limit = Decimal(10) ** -230
    for _ in range(80):
        off = sum(M[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < limit:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if M[p][q] == 0:
                    continue
                th = (M[q][q] - M[p][p]) / (2 * M[p][q])
                t = (1 if th >= 0 else -1) / (abs(th) + (th * th + 1).sqrt())
                c = 1 / (t * t + 1).sqrt()
                s = t * c
                for i in range(n):
                    a_, b_ = M[i][p], M[i][q]
                    M[i][p] = c * a_ - s * b_
                    M[i][q] = s * a_ + c * b_
                for i in range(n):
                    a_, b_ = M[p][i], M[q][i]
                    M[p][i] = c * a_ - s * b_
                    M[q][i] = s * a_ + c * b_
    return sorted((M[i][i] for i in range(n)), reverse=True)


def exact_pivots(A) -> list[Fraction]:
    """Pivots der LDL^T-Zerlegung ohne Vertauschung in Brüchen (Gleitkomma-Matrix als exakte Dualbrüche): für positiv definite Matrizen alle > 0.
    Ist ein Pivot null, bricht die Zerlegung ab (ZeroDivisionError)."""
    n = len(A)
    M = [[Fraction(float(v)) for v in row] for row in A]
    piv = []
    for k in range(n):
        d = M[k][k]
        piv.append(d)
        if d == 0:
            if k < n - 1:
                raise ZeroDivisionError("Pivot null")
            break
        for i in range(k + 1, n):
            f = M[i][k] / d
            for j in range(k + 1, n):
                M[i][j] -= f * M[k][j]
    return piv
