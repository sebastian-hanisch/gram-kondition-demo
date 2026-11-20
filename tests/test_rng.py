"""Zufall: gkd_rng.py ist eine wörtliche Kopie der Vorlage (Kopier-Test), SplitMix64-Anker, Normalzahlen per Inversion, und kein numpy.random in den App-Modulen."""

import re
import statistics
from pathlib import Path

import numpy as np
import pytest

import gkd_scenario as S
from gkd_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
SOURCES = {"gkd_rng.py": ("sebastianhanisch-website/tools/templates", "rng_splitmix64_exact.py")}


@pytest.mark.parametrize("copy", list(SOURCES))
def test_copy_is_verbatim_apart_from_the_header(copy):
    folder, original = SOURCES[copy]
    source = ROOT.parent / folder / original
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    mine = (ROOT / copy).read_bytes().decode("utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    assert mine[1:] == source.read_bytes().decode("utf-8").splitlines(), "die Kopie weicht von der Vorlage ab: Änderungen gehören in die Vorlage, danach neu kopieren"


def test_the_copy_runs_on_its_own_and_matches_known_outputs():
    assert [SplitMix64(0).next_u64() for _ in range(1)] == [0xE220A8397B1DCDAF]
    r = SplitMix64(0)
    assert [r.next_u64() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    assert [SplitMix64(42).random().hex()] == ["0x1.7bae644c5fd6dp-1"]
    assert SplitMix64(1).below(10) in range(10)


def test_no_numpy_random_in_the_app_modules():
    for path in ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"default_rng|np\.random|numpy\.random|RandomState", text), path.name


def test_normal_block_is_standard_normal_and_deterministic():
    z = S.normal_block(SplitMix64(7), 20000)
    assert np.array_equal(z, S.normal_block(SplitMix64(7), 20000)) and not np.array_equal(z, S.normal_block(SplitMix64(8), 20000))
    assert abs(z.mean()) < 4 / np.sqrt(z.size) and abs(z.std() - 1) < 4 / np.sqrt(2 * z.size)
    nd = statistics.NormalDist()
    for q in (0.025, 0.16, 0.5, 0.84, 0.975):
        assert np.mean(z < nd.inv_cdf(q)) == pytest.approx(q, abs=4 * np.sqrt(q * (1 - q) / z.size))
    stats = pytest.importorskip("scipy.stats")
    assert stats.kstest(z, "norm").pvalue > 0.001


def test_normal_block_survives_a_zero_draw():
    class Zero:
        def random(self):
            return 0.0
    z = S.normal_block(Zero(), 2)
    assert np.all(np.isfinite(z)) and z[0] < -8


def test_sub_seeds_separate_the_streams():
    seeds = {S.sub_seed(k, b, i) for k in (1, 2, 3) for b in (0, 35, 999999) for i in (0, 1, 59)}
    assert len(seeds) == 27
