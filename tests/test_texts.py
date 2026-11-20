"""Texte: App und README sagen dasselbe (DEMO-PLAYBOOK §5: zwei Fassungen desselben Textes werden von einem Test verglichen, nicht von Hand synchron gehalten)
und halten die Schreibregeln der Linie ein (kein „trägt/tragen“, kein Gedankenstrich im Fließtext, „Zufalls-Seed“, keine Begriffe der Planung)."""

import re
from pathlib import Path

import gkd_constants as C
import gkd_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
PROSE = ("TITLE", "INTRO", "HOW_IT_WORKS", "FALLE_CAPTION", "NUGGET_CAPTION", "FLAT_CAPTION", "RIDGE_CAPTION", "FLAECHE_CAPTION", "GRENZEN_TABLE", "ASSUMPTIONS_MARKDOWN", "MATH_MARKDOWN")


def test_assumptions_text_of_the_app_is_in_the_readme_verbatim():
    assert T.ASSUMPTIONS_MARKDOWN.strip() in README


def test_the_grenzen_table_of_the_app_is_in_the_readme_verbatim():
    assert T.GRENZEN_TABLE.strip() in README


def test_the_preset_names_are_in_the_readme():
    for name in C.PRESETS:
        assert name in README, name


def test_forbidden_words_do_not_appear_in_app_texts_or_readme():
    values = [getattr(T, n) for n in PROSE] + [README] + list(C.PRESET_HELP.values())
    for value in values:
        assert not re.search(r"\bträgt\b|\btragen\b|\btragende|Saatwert|\bSeeds?\b(?<!Zufalls-Seed)(?<!Zufalls-Seeds)", value.replace("Zufalls-Seed", "ZS")), value[:60]


def test_no_dash_in_running_text():
    for name in PROSE:
        value = getattr(T, name)
        assert "–" not in value and "—" not in value, name
    for text in C.PRESET_HELP.values():
        assert "–" not in text and "—" not in text
    body = README.rsplit("\n---\n", 1)[0].split("\n", 1)[1]                        # ohne Überschrift und Footer-Absatz (beide dürfen den Gedankenstrich des Portfolio-Titels haben)
    body = body.replace("„Sebastian Hanisch – Operations Research und Machine Learning“", "")     # der Titel der Website
    assert "–" not in body and "—" not in body


def test_the_planning_words_stay_out_of_the_visible_texts():
    values = [getattr(T, n) for n in PROSE] + list(C.PRESET_HELP.values())
    for value in values:
        assert not re.search(r"\bPhase 0\b|\bStück\b|\bVormessung|\bPlan-|\bPlan\b", value), value[:80]


def test_no_ascii_umlaut_spellings_in_the_app_texts():
    words = ("fuer", "ueber", "koennen", "muessen", "waehlen", "Loesung", "Loeser", "Genauigkeit_", "Bandbreite_", "Abweichung_")
    for name in PROSE:
        text = getattr(T, name)
        for w in words:
            assert not re.search(rf"\b{w}", text), (name, w)
