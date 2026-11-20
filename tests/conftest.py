"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Mini-Instanz: drei Punkte x = (0, 1, 2) auf der Geraden und q = exp(-1/(2σ²)) = 1/2, also K_ij = q^((x_i - x_j)²) (σ = 0,8493, γ = ln 2):
K = [[1, 1/2, 1/16], [1/2, 1, 1/2], [1/16, 1/2, 1]]; det K = 135/256; Pivots 1, 3/4, 45/64; Eigenwerte 15/16 und (33 ± √513)/32 (Handrechnung)."""

import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture
def mini_points():
    return (0, 1, 2)


@pytest.fixture
def mini_gram():
    h = Fraction(1, 2)
    return [[Fraction(1), h, h ** 4], [h, Fraction(1), h], [h ** 4, h, Fraction(1)]]
