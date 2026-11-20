"""Permalink (Begrenzen, Einrasten auf Optionen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import gkd_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # Regler mit fester Auswahl (select_slider): nur diese Werte sind gültig


SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "n_falle": SettingSpec("nf", int, C.DEFAULT_N_FALLE, C.N_FALLE_MIN, C.N_FALLE_MAX),
    "sigma_falle": SettingSpec("sf", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "lam_falle": SettingSpec("lf", float, C.DEFAULT_LAM, options=C.LAMS),
    "m_falle": SettingSpec("mf", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "m_nugget": SettingSpec("mn", int, C.M_NUGGET_DEFAULT, C.M_NUGGET_MIN, C.M_NUGGET_MAX),
    "n_flat": SettingSpec("nl", int, C.DEFAULT_N_FLAT, C.N_FLAT_MIN, C.N_FLAT_MAX),
    "muster_flat": SettingSpec("pl", str, C.MUSTER[0], options=C.MUSTER),
    "k_flat": SettingSpec("kl", int, C.DEFAULT_K_FLAT, C.K_FLAT_MIN, C.K_FLAT_MAX),
    "sigma_ridge": SettingSpec("sr", float, C.DEFAULT_SIGMA_RIDGE, options=C.SIGMAS_RIDGE),
    "n_ridge": SettingSpec("nr", int, C.DEFAULT_N_RIDGE, options=C.NS_RIDGE),
    "tau_ridge": SettingSpec("tr", float, C.DEFAULT_TAU, options=C.TAUS),
    "wahl_ridge": SettingSpec("wr", str, C.WAHLEN[0], options=C.WAHLEN),
    "n_flaeche": SettingSpec("na", int, C.DEFAULT_N_FLAECHE, C.N_FLAECHE_MIN, C.N_FLAECHE_MAX),
    "noise_flaeche": SettingSpec("ra", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "seeds_flaeche": SettingSpec("sa", int, C.DEFAULT_S_FLAECHE, C.S_FLAECHE_MIN, C.S_FLAECHE_MAX),
}
PRESET_KEYS = {key: key for key in SETTING_SPECS}   # ein Preset lädt das ganze Szenario: Regler unter ihrem Schlüssel, der Zufalls-Seed fällt auf seinen Standard zurück


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def options(state_key):
    return SETTING_SPECS[state_key].options


def _snap(spec: SettingSpec, value):
    """Wert aus dem Permalink auf eine gültige Option einrasten (None, wenn keine passt)."""
    for opt in spec.options:
        if opt == value or (isinstance(opt, float) and isinstance(value, float) and math.isclose(opt, value, rel_tol=1e-9, abs_tol=0.0)):
            return opt
    return None


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf Optionen eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None:
                value = _snap(spec, value)
                if value is None:
                    continue
            else:
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for state_key in PRESET_KEYS.values():
        st.session_state[state_key] = C.PRESETS[name].get(state_key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    for name, preset in C.PRESETS.items():
        if all(st.session_state.get(state_key, spec.default) == preset.get(state_key, spec.default) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
