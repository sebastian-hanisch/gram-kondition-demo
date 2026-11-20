"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Ein Parameter λ hat in dieser Demo zwei Rollen: in Kernel Ridge (Stück 5) die Rauschvarianz des GP-Mittels (Modell), hier zusätzlich das **Nugget**
(englisch Jitter) als Rechenhilfe, das zur Diagonale addiert wird (K + λ I). Die App sagt bei jeder Zahl, welche Rolle gemeint ist. σ ist die absolute
Bandbreite (Längenmaß der Eingabe); der Skalenfaktor f der Median-Heuristik (Stück 4) kommt hier nicht vor."""

# ---- Zufall und feste Testmengen -------------------------------------------------------------------------------------------------------------
SEED_MAX, DEFAULT_SEED = 999999, 35
SEED_TEST_REG = 6401                 # feste Testpunkte der Beispiel-Regression
SEED_TEST_SQUARE = 15400             # feste Testpunkte der Falle im Einheitsquadrat
N_TEST_REG = 1000                    # rauschfreie Testpunkte der Beispiel-Regression
N_TEST_PRED = 100                    # Testpunkte der Falle (exakte Vorhersage in Brüchen)
REG_LO, REG_HI = -2.0, 2.0           # Gebiet der Beispiel-Regression: [-2, 2]²
KIND_FALLE, KIND_NUGGET, KIND_FLAECHE, KIND_RIDGE, KIND_ROUND = 1, 2, 3, 4, 5   # getrennte Zufallsströme je Erzeuger (gkd_scenario.sub_seed)

# ---- Experiment 2 und 3: die Falle und das Nugget ---------------------------------------------------------------------------------------------
N_FALLE_MIN, N_FALLE_MAX, DEFAULT_N_FALLE = 10, 40, 20
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = 0.1, 20.0, 5.0
LAMS = (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)   # 0 = reine Interpolation (λ = 0)
DEFAULT_LAM = 0.0
M_MIN, M_MAX, DEFAULT_M = 1, 30, 30                              # Punktmengen (Instanzen) je Zählung
M_NUGGET_MIN, M_NUGGET_MAX, M_NUGGET_DEFAULT = 1, 15, 10                  # Punktmengen je Wert von λ in der Nugget-Reihe (drei Werte; die exakte Elimination begrenzt die Obergrenze)
NUGGET_LAMS = (1e-10, 1e-6, 1e-2)                                # Nugget-Reihe der Schranke
NUGGET_TEST_LAMS = (1e-8, 1e-4, 1e-2, 1e-1)                      # Nugget-Reihe des Testfehlers gegen die Wahrheit
ERR_BAD = 0.1                                                    # „Lösung um mehr als 10 % daneben“
ERR_SMALL = 1e-3                                                 # Schwelle der Abweichung in der Vorhersage und der kleineren Fehlerklasse
ERR_TINY = 1e-6

# ---- Experiment 4: Flat Limit ---------------------------------------------------------------------------------------------------------------
N_FLAT_MIN, N_FLAT_MAX, DEFAULT_N_FLAT = 3, 8, 6
K_FLAT_MIN, K_FLAT_MAX, DEFAULT_K_FLAT = 2, 12, 6                # q = exp(-1/(2σ²)) = 1 - 10^-k
MUSTER = ("gleichabständig", "ungleich")
UNEVEN = (0, 1, 3, 4, 7, 9, 12, 16)                              # ganzzahlige, ungleich verteilte Punkte
ECHT_SCHWELLE = 0.1                                              # Eigenwert „echt“: relative Abweichung gegen die rationale Matrix unter 10 %

# ---- Experiment 5: Interpolation und Abschneidegrenze -------------------------------------------------------------------------------------------
SIGMAS_RIDGE = (0.3, 1.0, 3.0)
NS_RIDGE = (20, 40, 60, 120, 240)
TAUS = (1e-15, 1e-13, 1e-11, 1e-9, 1e-7, 1e-5)
DEFAULT_SIGMA_RIDGE, DEFAULT_N_RIDGE, DEFAULT_TAU = 1.0, 60, 1e-13
WAHLEN = ("Interpolation (Eigenwerte abschneiden)", "λ per Leave-one-out gewählt")
RIDGE_INSTANZEN = 30                                             # Zufalls-Seeds je Satz
RIDGE_NOISE = 0.3
LOO_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)

# ---- Experiment 1: die Fläche σ × λ -------------------------------------------------------------------------------------------------------------
FLAECHE_SIGMAS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4)
FLAECHE_LAMS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)
N_FLAECHE_MIN, N_FLAECHE_MAX, DEFAULT_N_FLAECHE = 30, 100, 60
NOISE_MIN, NOISE_MAX, DEFAULT_NOISE = 0.0, 0.5, 0.3
S_FLAECHE_MIN, S_FLAECHE_MAX, DEFAULT_S_FLAECHE = 10, 30, 30

# ---- Farben --------------------------------------------------------------------------------------------------------------------------------------
COLOR_EXACT = "#2F6B65"
COLOR_FLOAT = "#d95f02"
COLOR_ROUND = "#7570b3"
COLOR_BAD = "#c0392b"
COLOR_OK = "#2F6B65"
COLOR_LINE = "#4c78a8"
COLOR_GREY = "rgba(150,150,150,0.8)"

# ---- Presets (Name -> Werte; Schlüssel wie in gkd_presets.PRESET_KEYS; nicht genannte Regler stehen auf ihrem Standardwert) -----------------------
DEFAULT_PRESET = "Cholesky gelingt, Lösung falsch (n = 20, σ = 5, λ = 0)"
PRESETS = {
    DEFAULT_PRESET: {},
    "Gut konditioniert (σ = 0,5, λ = 0,01)": {"sigma_falle": 0.5, "lam_falle": 1e-2},
    "Mit Nugget 1e-6": {"lam_falle": 1e-6},
    "Interpolation (σ = 1, λ = 0)": {"sigma_falle": 1.0, "wahl_ridge": WAHLEN[0], "sigma_ridge": 1.0},
    "Leave-one-out gewählt": {"sigma_falle": 1.0, "wahl_ridge": WAHLEN[1], "sigma_ridge": 1.0},
    "Flat Limit (n = 5, 1/σ → 0)": {"n_flat": 5, "muster_flat": MUSTER[0], "k_flat": 8},
}
PRESET_HELP = {
    DEFAULT_PRESET: "n = 20 Punkte im Einheitsquadrat, σ = 5, λ = 0: Experiment 2 zählt über 30 Punktmengen, in wie vielen die Cholesky-Zerlegung gelingt und die Lösung trotzdem "
                    "um mehr als 10 % von der exakten abweicht (Messung dieser Plattform, siehe README).",
    "Gut konditioniert (σ = 0,5, λ = 0,01)": "Kleine Bandbreite und ein Nugget λ = 0,01: die Gram-Matrix ist gut konditioniert, alle Wege treffen die exakte Lösung auf viele Stellen.",
    "Mit Nugget 1e-6": "Dieselbe Falle (σ = 5), aber mit Nugget λ = 1e-6 auf der Diagonale: cond(K + λI) ist höchstens 1 + n/λ, die Fehler bleiben klein.",
    "Interpolation (σ = 1, λ = 0)": "σ = 1, λ = 0: Experiment 2 und die Interpolation von Experiment 5 (Beispiel-Regression, n = 60, Eigenwerte unter τ abgeschnitten).",
    "Leave-one-out gewählt": "Wie das Preset davor, aber Experiment 5 zeigt die Vorhersage mit per Leave-one-out gewähltem λ statt der Interpolation.",
    "Flat Limit (n = 5, 1/σ → 0)": "Fünf Punkte x = (0, 1, 2, 3, 4) und q = 1 - 1e-8, also σ ≈ 7·10³: die exakten Eigenwerte folgen der Leiter c_m (1/σ)^(2(m-1)).",
}

# ---- Datei der eingefrorenen (vorgerechneten) Zahlen ---------------------------------------------------------------------------------------------
FROZEN_FILE = "gkd_frozen.json"
