"""Zahlen für Anzeigetexte: Dezimalkomma wie auf der Website, wissenschaftliche Schreibweise mit kurzem Exponenten."""

import math


def de(x: float, nd: int = 3) -> str:
    """Zahl mit `nd` gültigen Stellen und Dezimalkomma (0,0469; 3308; 1,2e4 bleibt dem sci vorbehalten)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if math.isinf(x):
        return "unendlich"
    if x == 0:
        return "0"
    mag = math.floor(math.log10(abs(x)))
    if -4 <= mag < 4:
        decimals = nd - 1 - mag
        return f"{round(x, decimals):.{max(decimals, 0)}f}".replace(".", ",")
    mant, exp = f"{x:.{max(nd - 1, 0)}e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"{mant.replace('.', ',')}e{int(exp)}"


def sci(x: float, nd: int = 1) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma und ohne führende Null im Exponenten: 1,2e-16, 4,5e15."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if math.isinf(x):
        return "unendlich"
    mant, exp = f"{x:.{nd}e}".split("e")
    return f"{mant.replace('.', ',')}e{int(exp)}"


def lam_text(lam: float) -> str:
    """λ für Texte: 0, 1e-10, 0,01, 1."""
    if lam == 0:
        return "0"
    if lam >= 1e-2:
        return f"{lam:g}".replace(".", ",")
    return f"1e{int(round(math.log10(lam)))}" if abs(lam - 10 ** round(math.log10(lam))) < 1e-9 * lam else sci(lam, 1)


def sigma_text(sigma: float) -> str:
    return f"{sigma:g}".replace(".", ",")


def tau_text(tau: float) -> str:
    return f"1e{int(round(math.log10(tau)))}"


def zaehl(k: int, m: int) -> str:
    """„k von m“."""
    return f"{k} von {m}"
