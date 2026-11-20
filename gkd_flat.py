"""Flat Limit der Gauß-Gram-Matrix: Eigenwerte im Grenzfall wachsender Bandbreite, exakt (Brüche, Dezimalzahlen mit 120 Stellen) gegen Gleitkomma.

Gauß-Kern auf **ganzzahligen** Punkten, q = exp(-1/(2σ²)) = 1 - 10^-k als Bruch: K_ij = q^((x_i - x_j)²) ist dann rational und alle Referenzgrößen sind exakt.
Geschlossene Form (eigene Rechnung, nicht aus einer gelesenen Quelle): λ_m ≈ c_m (1/σ)^(2(m-1)) mit c_m = R_mm²/(m-1)!, R_mm² = Quadrat des Abstands des Monoms
x^(m-1) vom Span der niedrigeren Monome auf den Punkten (Gram-Schmidt in Brüchen; unabhängig von K). Drei Matrizen werden unterschieden: R (rational, exakt),
D (dieselbe Matrix, Einträge auf Doppelte gerundet, exakt gerechnet) und F (die Gleitkomma-Eigenzerlegung `numpy.linalg.eigvalsh` von D)."""

import math
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

import numpy as np

import gkd_constants as C
import gkd_exact as E
import gkd_kernel as Kn


def q_of_k(k: int) -> Fraction:
    """q = 1 - 10^-k als Bruch."""
    return Fraction(1) - Fraction(1, 10 ** k)


def rational_gram(xs, q: Fraction) -> list[list[Fraction]]:
    """K_ij = q^((x_i - x_j)²) für ganzzahlige Punkte, exakt."""
    return [[q ** ((a - b) ** 2) for b in xs] for a in xs]


def eps2_of_q(q: Fraction) -> Decimal:
    """ε² = 1/σ² = -2 ln q in 120 Stellen."""
    return -2 * (Decimal(q.numerator) / Decimal(q.denominator)).ln()


def gram_schmidt_squares(xs) -> list[Fraction]:
    """R_jj² für j = 0..n-1: Quadrat des Abstands des Monoms x^j vom Span der niedrigeren Monome auf den Punkten (Gram-Schmidt in Brüchen)."""
    cols = [[Fraction(x) ** j for x in xs] for j in range(len(xs))]
    ortho, res = [], []
    for v in cols:
        w = list(v)
        for u, uu in ortho:
            coef = sum(a * b for a, b in zip(w, u)) / uu
            w = [a - coef * b for a, b in zip(w, u)]
        nn = sum(a * a for a in w)
        ortho.append((w, nn))
        res.append(nn)
    return res


def ladder_coefficients(xs) -> list[Fraction]:
    """c_m = R_mm²/(m-1)! (m = 1..n; Index 0 ist m = 1): die Koeffizienten der Eigenwertleiter."""
    return [r / math.factorial(m) for m, r in enumerate(gram_schmidt_squares(xs))]


def ladder_prediction(xs, eps2: Decimal) -> list[Decimal]:
    """Vorhergesagte Eigenwerte c_m (ε²)^(m-1)."""
    return [E.dec(c) * eps2 ** m for m, c in enumerate(ladder_coefficients(xs))]


def determinant_closed_form(xs, eps2: Decimal) -> Decimal:
    """det K ≈ (ε²)^(n(n-1)/2) · ∏_{i<j}(x_j - x_i)² / ∏_{j<n} j!."""
    n = len(xs)
    vander = Fraction(1)
    for i in range(n):
        for j in range(i + 1, n):
            vander *= Fraction(xs[j] - xs[i]) ** 2
    fact = math.prod(math.factorial(j) for j in range(n))
    return eps2 ** (n * (n - 1) // 2) * E.dec(vander) / Decimal(fact)


def count_real(approx, exact, threshold: float = C.ECHT_SCHWELLE) -> int:
    """Zahl der Eigenwerte, die gegen den exakten Wert der rationalen Matrix um weniger als `threshold` (relativ) abweichen (Zähleinheit: Eigenwerte)."""
    return sum(1 for a, b in zip(approx, exact) if abs(float(a) / float(b) - 1.0) < threshold)


def predicted_real_count(xs, eps2: float) -> int:
    """Vorhersage der Zahl echter Eigenwerte aus der Leiter: c_m ε^(2(m-1)) > n · 2,2e-16 · λ_1 (geschlossene Form gegen die Rundungsgrenze)."""
    cm = [float(c) for c in ladder_coefficients(xs)]
    n = len(xs)
    return sum(1 for m in range(n) if cm[m] * eps2 ** m > n * Kn.EPS * cm[0])


@dataclass(frozen=True)
class FlatResult:
    xs: tuple
    k: int
    sigma: float                  # abgeleitet aus q
    eps2: float                   # 1/σ²
    exact: tuple                  # Eigenwerte der rationalen Matrix R (absteigend, Gleitkomma-Darstellung der 120-stelligen Werte)
    rounded: tuple                # Eigenwerte der gerundeten Matrix D (exakt gerechnet)
    floating: tuple               # numpy.linalg.eigvalsh von D
    ladder: tuple                 # c_m ε^(2(m-1))
    coefficients: tuple           # c_m als Brüche
    ratio_exact_ladder: tuple     # exakter Eigenwert / Leiter
    real_rounded: int             # echte Eigenwerte in D gegen R
    real_float: int               # echte Eigenwerte in F gegen R
    predicted: int                # Vorhersage aus der Leiter
    smallest_rounded_negative: bool
    smallest_float_negative: bool
    pivots_float: tuple           # LDL-Pivots der Gleitkomma-Matrix
    pivot_negative_at: int | None  # erste Stelle (ab 1), an der ein Pivot <= 0 ist; None: alle Pivots positiv
    cholesky_ok: bool
    cond_exact: float             # λ_1/λ_n der rationalen Matrix
    cond_ladder: float            # c_1/c_n / (ε²)^(n-1)


def analyse_flat(xs, k: int) -> FlatResult:
    """Alle Größen des Flat-Limit-Experiments für ganzzahlige Punkte xs und q = 1 - 10^-k."""
    xs = tuple(int(x) for x in xs)
    n = len(xs)
    q = q_of_k(k)
    R = rational_gram(xs, q)
    D = [[Fraction(float(v)) for v in row] for row in R]
    lam_r = E.jacobi_eigenvalues(R)
    lam_d = E.jacobi_eigenvalues(D)
    Kf = np.array([[float(v) for v in row] for row in D])
    lam_f = np.sort(np.linalg.eigvalsh(Kf))[::-1]
    e2 = eps2_of_q(q)
    lad = ladder_prediction(xs, e2)
    piv = Kn.ldl_pivots(Kf)
    bad = [i + 1 for i, p in enumerate(piv) if not p > 0]
    return FlatResult(
        xs=xs, k=k, sigma=1.0 / math.sqrt(float(e2)), eps2=float(e2),
        exact=tuple(float(v) for v in lam_r), rounded=tuple(float(v) for v in lam_d), floating=tuple(float(v) for v in lam_f),
        ladder=tuple(float(v) for v in lad), coefficients=tuple(ladder_coefficients(xs)),
        ratio_exact_ladder=tuple(float(a / b) for a, b in zip(lam_r, lad)),
        real_rounded=count_real(lam_d, lam_r), real_float=count_real(lam_f, lam_r), predicted=predicted_real_count(xs, float(e2)),
        smallest_rounded_negative=bool(lam_d[-1] < 0), smallest_float_negative=bool(lam_f[-1] < 0),
        pivots_float=tuple(float(p) for p in piv), pivot_negative_at=(bad[0] if bad else None),
        cholesky_ok=Kn.cholesky_or_none(Kf) is not None,
        cond_exact=float(lam_r[0] / lam_r[-1]),
        cond_ladder=float(E.dec(ladder_coefficients(xs)[0] / ladder_coefficients(xs)[-1]) / e2 ** (n - 1)),
    )
