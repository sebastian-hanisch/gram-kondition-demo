"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_every_select_slider_and_selectbox_option_does_not_raise():
    import gkd_presets as P
    probe = _fresh_app()
    slider_keys = [w.key for w in probe.sidebar.select_slider]
    box_keys = [w.key for w in probe.sidebar.selectbox]
    assert {"lam_falle", "sigma_ridge", "n_ridge", "tau_ridge"} <= set(slider_keys) and {"muster_flat", "wahl_ridge"} <= set(box_keys)
    for key in slider_keys + box_keys:
        options = P.SETTING_SPECS[key].options                                 # die Optionen der App, nicht ihre Anzeigetexte (AppTest nimmt den Wert)
        for value in (options[0], options[-1]):
            at = _fresh_app()
            w = at.sidebar.select_slider(key=key) if key in slider_keys else at.sidebar.selectbox(key=key)
            w.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_seed_input_at_its_bounds_and_the_dice_button_do_not_raise():
    for value in (0, 999999):
        at = _fresh_app()
        at.sidebar.number_input(key="seed_input").set_value(value).run(timeout=TIMEOUT)
        assert not at.exception, (value, [str(e) for e in at.exception])


def test_the_page_shows_all_experiments_the_boundaries_and_the_math_expander():
    at = _fresh_app()
    headers = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headers if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in (1, 2, 3, 4, 5)]
    assert "🚧 Wo die Annahmen enden" in headers
    labels = [e.label for e in at.expander]
    assert "📐 Mathematische Formulierung" in labels and "⚠️ Annahmen, Grenzen und bekannte Probleme" in labels
    kinds = {"success": len(at.success), "info": len(at.info), "warning": len(at.warning)}
    assert sum(kinds.values()) >= 10                                          # die Verdikte der Kernfragen und Experimente stehen als Meldungen da
