"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink; je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a Punkt 7)."""


import gkd_constants as C
import gkd_evaluation as V
import gkd_experiments as X
import gkd_presets as P


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_valid_for_their_regler():
    for preset in C.PRESETS.values():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[key]
            if spec.options is not None:
                assert value in spec.options, key
            else:
                assert spec.lo <= value <= spec.hi, key
            assert isinstance(value, spec.caster) or (spec.caster is float and isinstance(value, (int, float)))


def test_default_preset_is_the_first_and_equals_the_default_settings():
    assert next(iter(C.PRESETS)) == C.DEFAULT_PRESET and C.PRESETS[C.DEFAULT_PRESET] == {}
    assert P.SETTING_SPECS["sigma_falle"].default == 5.0 and P.SETTING_SPECS["lam_falle"].default == 0.0 and P.SETTING_SPECS["n_falle"].default == 20


def test_all_presets_are_distinct_settings():
    keys = list(P.PRESET_KEYS.values())
    effective = [tuple(p.get(k, P.SETTING_SPECS[k].default) for k in keys) for p in C.PRESETS.values()]
    assert len(set(effective)) == len(effective)


def test_setting_specs_cover_every_preset_key_and_the_seed():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and "seed_input" in P.PRESET_KEYS
    for spec in P.SETTING_SPECS.values():
        if spec.options is not None:
            assert spec.default in spec.options
        else:
            assert spec.lo <= spec.default <= spec.hi


def test_url_params_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(set(params)) == len(params)


def test_permalink_snap_accepts_options_and_rejects_foreign_values():
    spec = P.SETTING_SPECS["lam_falle"]
    assert P._snap(spec, 1e-06) == 1e-6 and P._snap(spec, 0.0) == 0.0
    assert P._snap(spec, 3e-5) is None
    assert P._snap(P.SETTING_SPECS["muster_flat"], "ungleich") == "ungleich" and P._snap(P.SETTING_SPECS["muster_flat"], "x") is None


# ---------------------------------------------------------------- je Preset die erwartete Verdikt-Zeile (kleine Punktmengen-Zahl, feste Bandbreite)
def _falle_verdict(preset, m=12):
    p = C.PRESETS[preset]
    n = p.get("n_falle", C.DEFAULT_N_FALLE)
    sg, lam = p.get("sigma_falle", C.DEFAULT_SIGMA), p.get("lam_falle", C.DEFAULT_LAM)
    return V.verdict_falle(V.summarize_falle(X.falle_zaehlung(n, sg, lam, m, C.DEFAULT_SEED), n), sg, lam)


def test_preset_gut_konditioniert_reports_no_warning():
    kind, text = _falle_verdict("Gut konditioniert (σ = 0,5, λ = 0,01)")
    assert kind == "success" and "Cholesky gelingt in 12 von 12" in text and "σ = 0,5" in text and "λ = 0,01" in text


def test_preset_mit_nugget_reports_success_with_small_errors():
    kind, text = _falle_verdict("Mit Nugget 1e-6")
    assert kind == "success" and "12 von 12" in text and "λ = 1e-6" in text


def test_preset_falle_reports_a_warning_or_a_visible_failure():
    kind, text = _falle_verdict(C.DEFAULT_PRESET)
    assert kind in ("warning", "info") and "σ = 5" in text and "λ = 0" in text
    assert kind == "warning" or "scheitert in allen" in text


def test_preset_interpolation_sigma_1_reports_small_errors_for_the_solver_and_a_warning_for_the_model():
    kind, text = _falle_verdict("Interpolation (σ = 1, λ = 0)")
    assert kind in ("success", "info") and "σ = 1" in text                    # bei σ = 1 ist das Lösen nicht die Falle (Fehler unter 10 %)
    p = C.PRESETS["Interpolation (σ = 1, λ = 0)"]
    rs = V.summarize_ridge(X.ridge_set(p["sigma_ridge"], C.DEFAULT_N_RIDGE, C.DEFAULT_TAU, C.DEFAULT_SEED))
    assert V.verdict_ridge(rs, p["sigma_ridge"], C.DEFAULT_N_RIDGE, C.DEFAULT_TAU, p["wahl_ridge"])[0] == "warning"      # Modellfehler der Interpolation (Mittel über der Konstanten)


def test_preset_leave_one_out_reports_the_loo_variant_without_a_warning():
    p = C.PRESETS["Leave-one-out gewählt"]
    rs = V.summarize_ridge(X.ridge_set(p["sigma_ridge"], C.DEFAULT_N_RIDGE, C.DEFAULT_TAU, C.DEFAULT_SEED))
    kind, text = V.verdict_ridge(rs, p["sigma_ridge"], C.DEFAULT_N_RIDGE, C.DEFAULT_TAU, p["wahl_ridge"])
    assert kind == "info" and "per Leave-one-out gewähltem λ" in text and rs.loo_mean < 0.2


def test_preset_flat_limit_reports_the_exact_ladder_and_all_eigenvalues_real_at_five_points():
    from gkd_flat import analyse_flat
    p = C.PRESETS["Flat Limit (n = 5, 1/σ → 0)"]
    r = analyse_flat(tuple(range(p["n_flat"])), p["k_flat"])
    kind, text = V.verdict_flat(r)
    assert all(abs(v - 1) < 1e-6 for v in r.ratio_exact_ladder) and r.real_float >= 2
    assert kind == ("success" if r.real_float == 5 else "warning") and "x = (0, 1, 2, 3, 4)" in text and "q = 1 − 1e-8" in text
