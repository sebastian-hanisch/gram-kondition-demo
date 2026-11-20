"""Literaturwerte (Stufe W: nur aus gelesenen Teilen der Quelle; DEMO-PLAYBOOK §12, Plan: „Quelle nicht eingesehen heißt: kein Test“).

Barthelmé, Usevich, „Spectral properties of kernel matrices in the flat limit“, arXiv 1910.14067 v3 (Abstract, Abschnitt 1 samt den drei Abbildungen darin; Stufe W): im Flat Limit (feste Punkte,
Bandbreite gegen unendlich) wird die Gram-Matrix des Gauß-Kerns singulär, die Eigenwerte haben für d = 1 die Größenordnung ε^(2(k-1)) (k = 1, 2, ...), und der Interpolant bleibt
wohldefiniert. Hier geprüft: die Größenordnung (Steigung 2(k-1) im doppeltlogarithmischen Bild, exakt gerechnet) und die Singularität im Grenzfall (Determinante geht gegen null).
Der Koeffizient c_m = R_mm²/(m-1)! ist **nicht** aus der Quelle (eigene Rechnung, in den gelesenen Teilen nicht gefunden) und steht deshalb in test_flat.py, nicht hier.

Nicht geprüft, weil nicht nachgerechnet: die hinreichende Bedingung für die Gleitkomma-Cholesky aus Terenin u. a., JMLR 25 (2024), Abschnitt 2 (Unterabschnitt 3, Formelanteil nur aus dem Zusammenhang
gelesen) und die Matérn-Eigenwerte (gehören zu kernwahl-demo)."""

import math

import pytest

import gkd_flat as Fl


@pytest.mark.parametrize("xs", [(0, 1, 2, 3, 4), (0, 1, 3, 4, 7)])
def test_eigenvalues_scale_like_eps_to_the_two_times_k_minus_one(xs):
    low, high = Fl.analyse_flat(xs, 6), Fl.analyse_flat(xs, 8)
    for k, (a, b) in enumerate(zip(low.exact, high.exact)):                       # k = 0, 1, ...: Exponent 2k in ε
        slope = math.log(a / b) / math.log(math.sqrt(low.eps2) / math.sqrt(high.eps2))
        assert slope == pytest.approx(2 * k, abs=1e-3)


def test_the_gram_matrix_becomes_singular_in_the_flat_limit_but_stays_regular_for_any_finite_sigma():
    dets = []
    for k in (2, 4, 6, 8):
        r = Fl.analyse_flat((0, 1, 2, 3), k)
        dets.append(math.prod(r.exact))
        assert min(r.exact) > 0                                                   # für endliches σ positiv definit (Mercer, verschiedene Punkte)
    assert all(a > b for a, b in zip(dets, dets[1:])) and dets[-1] < 1e-20 * dets[0]    # die Determinante fällt auf null zu
