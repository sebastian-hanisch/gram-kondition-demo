"""Testhilfen für die Orakel: andere Rechenwege als die der App (Brüche, charakteristisches Polynom, Gram-Determinanten).

Faddeev-LeVerrier ist aus dem Messskript kn_v15_flat.py der Planung abgeschrieben; die Wurzelsuche (Newton mit Abspalten) ist für diese Tests neu geschrieben, die App benutzt
statt dessen das Jacobi-Verfahren (gkd_exact.jacobi_eigenvalues). Die Gauß-Elimination in Brüchen und die Gram-Determinanten sind für diese Tests geschrieben."""

from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 120


def solve_fractions(A, b):
    """Gauß-Elimination in Brüchen mit Pivotsuche (A, b: Zahlen oder Brüche); wirft ZeroDivisionError bei singulärem A."""
    n = len(A)
    M = [[Fraction(v) for v in A[i]] + [Fraction(b[i])] for i in range(n)]
    for k in range(n):
        p = next((r for r in range(k, n) if M[r][k] != 0), None)
        if p is None:
            raise ZeroDivisionError("singulär")
        M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n + 1):
                M[i][j] -= f * M[k][j]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def det_fractions(A):
    """Determinante in Brüchen (Entwicklung durch Elimination)."""
    n = len(A)
    M = [[Fraction(v) for v in row] for row in A]
    det = Fraction(1)
    for k in range(n):
        p = next((r for r in range(k, n) if M[r][k] != 0), None)
        if p is None:
            return Fraction(0)
        if p != k:
            M[k], M[p] = M[p], M[k]
            det = -det
        det *= M[k][k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= f * M[k][j]
    return det


def matmul(A, B):
    return [[sum(A[i][t] * B[t][j] for t in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def charpoly(A):
    """Faddeev-LeVerrier in Brüchen; p(x) = sum c[i] x^i, c[n] = 1."""
    n = len(A)
    c = [Fraction(0)] * (n + 1)
    c[n] = Fraction(1)
    M = [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        AM = matmul(A, M)
        M = [[AM[i][j] + (c[n - k + 1] if i == j else 0) for j in range(n)] for i in range(n)]
        AMk = matmul(A, M)
        c[n - k] = -sum(AMk[i][i] for i in range(n)) / k
    return c


def dec(fr):
    return Decimal(fr.numerator) / Decimal(fr.denominator)


def _horner(coeffs, x):
    """Wert und Ableitung eines Polynoms (aufsteigende Koeffizienten) an der Stelle x."""
    p = Decimal(0)
    dp = Decimal(0)
    for c in reversed(coeffs):
        dp = dp * x + p
        p = p * x + c
    return p, dp


def newton_eigenvalues(A):
    """Alle Eigenwerte der rationalen symmetrisch positiv semidefiniten Matrix A (absteigend) als Decimal aus dem exakten charakteristischen Polynom.
    Newton von oben (Spur als Startwert) findet bei lauter reellen Nullstellen die größte; danach wird das Polynom abgespalten, und jede Nullstelle wird am
    ursprünglichen Polynom nachpoliert (120 Stellen). Die Start-Heuristik des Messskripts (e_m/e_(m-1)) fand für einzelne Matrizen dieselbe Wurzel doppelt."""
    n = len(A)
    cd = [dec(v) for v in charpoly(A)]
    top = sum(dec(A[i][i]) for i in range(n)) + 1
    poly = list(cd)
    roots = []
    for _ in range(n):
        x = top
        for _ in range(2000):
            p, dp = _horner(poly, x)
            new = x - p / dp
            if abs(new - x) <= abs(new) * Decimal(10) ** -100:
                x = new
                break
            x = new
        for _ in range(6):
            p, dp = _horner(cd, x)
            x = x - p / dp
        roots.append(x)
        d = len(poly) - 1                       # poly durch (t - x) teilen
        b = [Decimal(0)] * d
        b[d - 1] = poly[d]
        for k in range(d - 1, 0, -1):
            b[k - 1] = poly[k] + x * b[k]
        poly = b
    return sorted(roots, reverse=True)


def monomial_gram(xs, count):
    """Gram-Matrix der Monome 1, x, ..., x^(count-1) auf den Punkten xs (Brüche): G_ab = sum_i x_i^(a+b)."""
    return [[sum(Fraction(x) ** (a + b) for x in xs) for b in range(count)] for a in range(count)]


def gs_squares_from_gram_determinants(xs):
    """R_mm² als Quotient der Gram-Determinanten der Monome (anderer Weg als Gram-Schmidt): R_mm² = det G_(m+1) / det G_m."""
    dets = [Fraction(1)] + [det_fractions(monomial_gram(xs, c)) for c in range(1, len(xs) + 1)]
    return [dets[m + 1] / dets[m] for m in range(len(xs))]
