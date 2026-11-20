"""Szenario: ausgedachte Punktmengen und Zielfunktionen (Modell, keine Betriebsdaten). Alle Zufallszahlen kommen aus SplitMix64 (gkd_rng.py, plattformgleich).

Erzeuger: Punkte gleichverteilt im Einheitsquadrat und auf der Strecke [0, 1] (die Konditionsfrage hängt an der Punktlage, nicht an der Zielfunktion); die
Beispiel-Regression (2D, ausgedachte Zielfunktion auf [-2, 2]² mit Rauschen); Standardnormal-Zielvektoren. Normalzahlen: `NormalDist().inv_cdf(random())`
(schnell genug für die Live-App; `SplitMix64.normal()` kostet 60 bis 130 µs je Zahl)."""

import math
from statistics import NormalDist

import numpy as np

import gkd_constants as C
from gkd_rng import SplitMix64

_INV_CDF = NormalDist().inv_cdf
_TINY = 2.0 ** -53


def sub_seed(kind: int, base: int, index: int) -> int:
    """Eigener Zufalls-Seed je Erzeuger (kind), Grund-Zufalls-Seed der Seitenleiste (base) und laufender Nummer (index): feste, getrennte Ströme."""
    return (kind * 1_000_003 + base) * 100_003 + index


def uniform_block(rng: SplitMix64, count: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    return lo + (hi - lo) * np.array([rng.random() for _ in range(count)])


def normal_block(rng: SplitMix64, count: int) -> np.ndarray:
    """Standardnormal per Inversion; u = 0 wird auf 2^-53 angehoben (inv_cdf braucht u in (0, 1))."""
    out = np.empty(count)
    for i in range(count):
        u = rng.random()
        out[i] = _INV_CDF(u if u > 0.0 else _TINY)
    return out


def points_square(n: int, seed: int) -> np.ndarray:
    """n Punkte gleichverteilt im Einheitsquadrat [0, 1]²."""
    rng = SplitMix64(seed)
    return uniform_block(rng, 2 * n).reshape(n, 2)


def points_line(n: int, seed: int) -> np.ndarray:
    """n Punkte gleichverteilt auf [0, 1], aufsteigend sortiert (Spalte, d = 1)."""
    rng = SplitMix64(seed)
    return np.sort(uniform_block(rng, n))[:, None]


def normal_targets(n: int, seed: int) -> np.ndarray:
    """Zielvektor y ~ N(0, I) (die Konditionsfrage braucht keine Zielfunktion)."""
    return normal_block(SplitMix64(seed), n)


def target_2d(X) -> np.ndarray:
    """Ausgedachte Zielfunktion der Beispiel-Regression: sin(1,5 x1) + 0,5 cos(2,5 x2) + 0,3 x1 x2 auf [-2, 2]²."""
    X = np.asarray(X, float)
    return np.sin(1.5 * X[:, 0]) + 0.5 * np.cos(2.5 * X[:, 1]) + 0.3 * X[:, 0] * X[:, 1]


def regression_data(n: int, noise: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Beispiel-Regression: n Punkte gleichverteilt auf [-2, 2]², y = Zielfunktion + Rauschen (Standardabweichung noise)."""
    rng = SplitMix64(seed)
    X = uniform_block(rng, 2 * n, C.REG_LO, C.REG_HI).reshape(n, 2)
    y = target_2d(X) + noise * normal_block(rng, n)
    return X, y


def reg_test_points(count: int = C.N_TEST_REG) -> tuple[np.ndarray, np.ndarray]:
    """Feste, rauschfreie Testpunkte der Beispiel-Regression (Zufalls-Seed fest) samt Wahrheit f(x)."""
    rng = SplitMix64(C.SEED_TEST_REG)
    X = uniform_block(rng, 2 * count, C.REG_LO, C.REG_HI).reshape(count, 2)
    return X, target_2d(X)


def square_test_points(count: int = C.N_TEST_PRED) -> np.ndarray:
    """Feste Testpunkte im Einheitsquadrat für den Vorhersagefehler der Falle."""
    return points_square(count, C.SEED_TEST_SQUARE)


def equispaced_points(n: int) -> tuple[int, ...]:
    return tuple(range(n))


def uneven_points(n: int) -> tuple[int, ...]:
    """Ganzzahlige, ungleich verteilte Punkte der Eigenwertleiter (die ersten n von C.UNEVEN)."""
    if not 1 <= n <= len(C.UNEVEN):
        raise ValueError(f"n muss zwischen 1 und {len(C.UNEVEN)} liegen")
    return C.UNEVEN[:n]


def sigma_from_q_exponent(k: int) -> float:
    """σ zur Wahl q = exp(-1/(2σ²)) = 1 - 10^-k: σ = 1/sqrt(-2 ln q). Nur zur Anzeige; die exakte Rechnung benutzt q als Bruch."""
    return 1.0 / math.sqrt(-2.0 * math.log1p(-(10.0 ** (-k))))
