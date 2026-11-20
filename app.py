"""Bandbreite und Kondition: wo die Rechnung still falsch wird - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünf Experimente zur Gauß-Gram-Matrix: die Fläche σ × λ, die Cholesky-Falle (gegen die exakte Lösung in ganzen Zahlen), das Nugget als Schranke aus λ allein,
die Eigenwertleiter im Flat Limit und die Interpolation als Objekt der Abschneidegrenze. Aufbau nach DEMO-PLAYBOOK §2.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import gkd_constants as C
import gkd_evaluation as V
import gkd_experiments as X
import gkd_flat as Fl
import gkd_scenario as S
import gkd_texts as T
import gkd_visualization as Viz
from gkd_format import de, lam_text, sci, sigma_text, tau_text, zaehl
from gkd_layout import dunkles_design, kopf_css
from gkd_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Bandbreite und Kondition – Sebastian Hanisch", layout="wide")


# ---- Rechnungen (zwischengespeichert; die teuren Teile sind exakte Elimination und die Fläche) ------------------------------------------------
@st.cache_data(show_spinner=False)
def _falle(n, sigma, lam, m, seed):
    return X.falle_zaehlung(n, sigma, lam, m, seed)


@st.cache_data(show_spinner=False)
def _nugget(n, sigma, m, seed):
    return X.nugget_reihe(n, sigma, m, seed)


@st.cache_data(show_spinner=False)
def _nugget_modell(n, sigma, seed):
    return X.nugget_testfehler(n, sigma, seed)


@st.cache_data(show_spinner=False)
def _solver_vs_model(n, sigma, seed):
    return X.solver_vs_model(n, sigma, seed)


@st.cache_data(show_spinner=False)
def _flat(xs, k):
    return Fl.analyse_flat(xs, k)


@st.cache_data(show_spinner=False)
def _ridge(sigma, n, tau, seed):
    return X.ridge_set(sigma, n, tau, seed)


@st.cache_data(show_spinner=False)
def _surface(n, noise, seeds, seed):
    return X.surface(n, noise, seeds, seed)


@st.cache_data(show_spinner=False)
def _frozen():
    return V.load_frozen()


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


# ---- Kopf, Einleitung, Verfahren ---------------------------------------------------------------------------------------------------------------
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title(T.TITLE)
st.markdown(T.INTRO)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/gram-kondition-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich zwei fast parallele Linien auf einem Plan vor. Ihr Kreuzungspunkt ist die Antwort, aber schon ein winziger Rundungsfehler schiebt ihn weit weg. Genau das passiert hier: Die **Gram-Matrix** ist eine Tabelle der Ähnlichkeiten aller Punkte. Reicht die Ähnlichkeit weit (große **Bandbreite**), sind sich alle Punkte fast gleich ähnlich, und die Gleichungen unterscheiden sich kaum. Der Computer rechnet mit Rundung, und die feinen Unterschiede der Tabelle (ihre kleinen **Eigenwerte**) gehen darin verloren.

Die Demo zählt über viele zufällig gewählte Punktmengen, wie oft ein Rechenverfahren ohne Fehlermeldung durchläuft und trotzdem falsch liegt, obwohl die Gegenprobe (das **Residuum**) unauffällig aussieht. Ein kleiner Zuschlag **λ** auf der Diagonale hilft der Rechnung, ändert aber die Aufgabe.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/gram-kondition-erklaert.html)
        """
    )
# erklaerseite:einfach:end

with st.expander("So funktioniert die Rechnung", expanded=True):
    st.markdown(T.HOW_IT_WORKS)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed der Punktmengen", *bounds("seed_input"), key="seed_input", step=1, help="Aus diesem Zufalls-Seed entstehen alle Punktmengen und Zielvektoren (SplitMix64).")
    st.button("🎲 Neue Punktmengen generieren", width="stretch", on_click=randomize_seed)

    st.markdown("**Experiment 2 und 3: die Falle und das Nugget**")
    n_f = st.slider("Punkte n", *bounds("n_falle"), key="n_falle", help="Punkte im Einheitsquadrat. Die exakte Elimination (Bareiss) läuft live; ab n = 40 dauert die Zählung einige Sekunden.")
    sigma_f = st.slider("Bandbreite σ", *bounds("sigma_falle"), key="sigma_falle", step=0.1, help="Längenmaß der Eingabe im Gauß-Kern exp(−d²/(2σ²)); γ = 1/(2σ²).")
    lam_f = st.select_slider("λ (Nugget, Rechenhilfe)", options=options("lam_falle"), key="lam_falle", format_func=lam_text,
                             help="Wird zur Diagonale addiert: K + λI. λ = 0 ist die reine Interpolation.")
    m_f = st.slider("Punktmengen (Zählung)", *bounds("m_falle"), key="m_falle", help="So viele Punktmengen (je eine eigene y-Ziehung) gehen in die Zählung ein.")
    m_n = st.slider("Punktmengen der Nugget-Reihe", *bounds("m_nugget"), key="m_nugget", help="Experiment 3: Punktmengen je Wert von λ (drei Werte).")

    st.markdown("**Experiment 4: Flat Limit**")
    n_l = st.slider("Punkte n (ganzzahlig)", *bounds("n_flat"), key="n_flat", help="Ganzzahlige Punkte auf einer Geraden (d = 1).")
    muster = st.selectbox("Punktmuster", options("muster_flat"), key="muster_flat", help="gleichabständig: 0, 1, 2, …; ungleich: 0, 1, 3, 4, 7, 9, 12, 16 (die ersten n).")
    k_l = st.slider("k mit q = 1 − 10^−k", *bounds("k_flat"), key="k_flat",
                    help="q = exp(−1/(2σ²)) bestimmt σ (angezeigt). Ein größeres k heißt größeres σ, also 1/σ → 0. So bleibt die Matrix rational.")

    st.markdown("**Experiment 5: Interpolation und Abschneidegrenze**")
    sigma_r = st.select_slider("Bandbreite σ", options=options("sigma_ridge"), key="sigma_ridge", format_func=sigma_text, help="Beispiel-Regression in 2D.")
    n_r = st.select_slider("Punkte n", options=options("n_ridge"), key="n_ridge", help="Trainingspunkte der Beispiel-Regression.")
    tau_r = st.select_slider("Abschneidegrenze τ", options=options("tau_ridge"), key="tau_ridge", format_func=tau_text,
                             help="Eigenwerte unter τ · λ_max werden bei der Interpolation weggelassen.")
    wahl_r = st.selectbox("Verdikt zeigt", options("wahl_ridge"), key="wahl_ridge", help="Welche Vorhersage das Verdikt der Kernfrage 4 beschreibt; das Diagramm zeigt immer beide.")

    st.markdown("**Experiment 1: Fläche σ × λ**")
    n_a = st.slider("Punkte n", *bounds("n_flaeche"), key="n_flaeche", step=10, help="Trainingspunkte der Beispiel-Regression.")
    noise_a = st.slider("Rauschen (Standardabweichung)", *bounds("noise_flaeche"), key="noise_flaeche", step=0.05, help="Rauschen der Etiketten.")
    seeds_a = st.slider("Zufalls-Seeds", *bounds("seeds_flaeche"), key="seeds_flaeche", help="Über so viele Zufalls-Seeds wird der Testfehler je Zelle gemittelt.")

sync_query_params({
    "seed_input": seed, "n_falle": n_f, "sigma_falle": sigma_f, "lam_falle": lam_f, "m_falle": m_f, "m_nugget": m_n, "n_flat": n_l, "muster_flat": muster, "k_flat": k_l,
    "sigma_ridge": sigma_r, "n_ridge": n_r, "tau_ridge": tau_r, "wahl_ridge": wahl_r, "n_flaeche": n_a, "noise_flaeche": noise_a, "seeds_flaeche": seeds_a,
})

seed, n_f, m_f, m_n, n_l, k_l, n_r, n_a, seeds_a = int(seed), int(n_f), int(m_f), int(m_n), int(n_l), int(k_l), int(n_r), int(n_a), int(seeds_a)

# ---- Rechnungen ausführen -----------------------------------------------------------------------------------------------------------------------
with st.spinner(f"Exakte Elimination in ganzen Zahlen für {m_f} Punktmengen (n = {n_f}); bei n = 40 dauert das einige Sekunden …"):
    falle = _falle(n_f, float(sigma_f), float(lam_f), m_f, seed)
summary = V.summarize_falle(falle, n_f)

# ---- Ergebnis (Hauptgestalt des Hooks: Rechenfehler) --------------------------------------------------------------------------------------------
st.markdown("## 🎯 Ergebnis: Cholesky gelingt, und die Lösung?")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cholesky gelingt", zaehl(summary.ok, summary.m), delta="Punktmengen", delta_color="off")
c2.metric("davon Fehler über 10 %", zaehl(summary.bad_10, summary.ok), delta="gegen die exakte Lösung", delta_color="off")
c3.metric("Median des Fehlers (Cholesky)", sci(summary.median_err) if summary.ok else "–", delta="relativ, Koeffizienten", delta_color="off")
c4.metric("größter exakter Rückwärtsfehler η", sci(summary.max_eta) if summary.ok else "–", delta="unter eps = 2,2e-16" if summary.ok and summary.max_eta < 2.2e-16 else "Residuum", delta_color="off")
cond_prefix = "≥ " if summary.cond_lower > summary.m / 2 else ""
c5.metric("Median der Konditionszahl", f"{cond_prefix}{sci(summary.median_cond)}" if summary.m else "–", delta="Untergrenze (Rundung)" if cond_prefix else "cond(K + λI)", delta_color="off")
st.plotly_chart(Viz.build_errors(falle), width="stretch", key="chart_errors")
st.caption("Je Balken eine Punktmenge: Fehler der Cholesky-Lösung gegen die exakte Lösung (rot: über 10 %, orange: über 1e-3). Schwarze Rauten: der exakt gerechnete Rückwärtsfehler η. "
           "Graue Kreuze: die Zerlegung scheitert und meldet es selbst.")
st.caption(T.FALLE_CAPTION)

# ---- Kernfragen -----------------------------------------------------------------------------------------------------------------------------------
st.markdown(T.KERNFRAGEN_HEADER)
st.markdown(T.KERNFRAGE_TITLES[0])
show(V.verdict_eigenwerte(summary, float(sigma_f)))
st.markdown(T.KERNFRAGE_TITLES[1])
show(V.verdict_falle(summary, float(sigma_f), float(lam_f)))
show(V.verdict_vorhersage(summary))

with st.spinner("Nugget-Reihe: exakte Elimination für drei Werte von λ …"):
    nug = _nugget(n_f, float(sigma_f), m_n, seed)
st.markdown(T.KERNFRAGE_TITLES[2])
show(V.verdict_nugget(nug, n_f, float(sigma_f)))

ridge = _ridge(float(sigma_r), n_r, float(tau_r), seed)
rsum = V.summarize_ridge(ridge)
st.markdown(T.KERNFRAGE_TITLES[3])
show(V.verdict_ridge(rsum, float(sigma_r), n_r, float(tau_r), wahl_r))

st.markdown("---")

# ---- Experiment 1 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 1: Die Fläche σ × λ ist ein Tal")
st.caption(T.FLAECHE_CAPTION)
with st.spinner(f"Fläche aus {len(C.FLAECHE_SIGMAS) * len(C.FLAECHE_LAMS)} Zellen über {seeds_a} Zufalls-Seeds …"):
    surf = _surface(n_a, float(noise_a), seeds_a, seed)
st.plotly_chart(Viz.build_surface(surf), width="stretch", key="chart_surface")
st.caption(f"Weißer Rahmen: beste Zelle. Konstante Vorhersage (Mittelwert der Etiketten): Testfehler {de(surf.const_mean)}.")
show(V.verdict_flaeche(surf, n_a, float(noise_a)))

st.markdown("---")

# ---- Experiment 2 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 2: Cholesky gelingt, Lösung falsch (der Hook)")
st.caption("Die Zählung steht oben im Ergebnis. Hier: die Eigenwerte der ersten Punktmenge, die drei Rechenwege im Vergleich und die Frage, ob der Rechenfehler den Testfehler bestimmt.")
first = falle[0] if falle else None
if first is not None:
    st.plotly_chart(Viz.build_eigenvalues(first.eigenvalues, float(lam_f), n_f), width="stretch", key="chart_eigs")
    st.caption("Eigenwerte der Gram-Matrix K der ersten Punktmenge (Betrag; Kreuz = Eigenwert ≤ 0). Unter der roten Linie (eps · λ_max) ist nur noch Rundung.")
st.dataframe(V.paths_table(falle), hide_index=True, width="stretch")
st.caption("Fehler gegen die exakte Lösung der Gleitkomma-Matrix, über die Punktmengen der Zählung. „Kein Ergebnis“ heißt: der Weg liefert keinen endlichen Koeffizientenvektor.")
with st.spinner("Rechenfehler gegen Modellfehler: Beispiel-Regression, exakte Lösung gegen LU …"):
    svm_rows, svm_const = _solver_vs_model(n_f, float(sigma_f), seed)
show(V.verdict_solver_vs_model(svm_rows, svm_const, n_f, float(sigma_f), 10))
st.caption("Negativbefund: bei λ = 1e-8 und 1e-4 (n = 40, σ = 5) ändert der Löser den Testfehler gegen die Wahrheit nicht messbar; bei λ = 0 ändert er ihn, aber beide Lösungen liegen weit über der konstanten Vorhersage, der Modellfehler der Interpolation dominiert (Experiment 5).")

st.markdown("---")

# ---- Experiment 3 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 3: Das Nugget, eine Schranke aus λ allein")
st.caption(T.NUGGET_CAPTION)
st.plotly_chart(Viz.build_nugget(nug), width="stretch", key="chart_nugget")
st.dataframe(V.nugget_table(nug), hide_index=True, width="stretch")
nug_model, nug_const = _nugget_modell(n_f, float(sigma_f), seed)
show(V.verdict_nugget_modell(nug_model, nug_const, n_f, float(sigma_f), C.RIDGE_INSTANZEN))
st.dataframe(V.model_table(nug_model), hide_index=True, width="stretch")

st.markdown("---")

# ---- Experiment 4 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 4: Flat Limit, die Eigenwertleiter")
st.caption(T.FLAT_CAPTION)
xs = S.equispaced_points(n_l) if muster == C.MUSTER[0] else S.uneven_points(n_l)
with st.spinner("Eigenwerte der rationalen Matrix (Jacobi-Verfahren in 120 Stellen) …"):
    flat = _flat(xs, k_l)
st.plotly_chart(Viz.build_ladder(flat), width="stretch", key="chart_ladder")
show(V.verdict_flat(flat))
show(V.verdict_flat_hinweis(flat))
st.dataframe(V.flat_table(flat), hide_index=True, width="stretch")
st.caption(f"Pivots der Cholesky-Zerlegung in Gleitkomma: {', '.join(sci(p, 2) for p in flat.pivots_float)}; "
           f"{'ein Pivot ist ab Nr. ' + str(flat.pivot_negative_at) + ' nicht positiv: die Zerlegung scheitert.' if flat.pivot_negative_at else 'alle Pivots sind positiv: die Zerlegung gelingt.'} "
           f"Konditionszahl der rationalen Matrix {sci(flat.cond_exact)}, aus der Leiter {sci(flat.cond_ladder)}.")

st.markdown("---")

# ---- Experiment 5 --------------------------------------------------------------------------------------------------------------------------------
st.subheader("🔬 Experiment 5: Interpolation ist ein Objekt der Abschneidegrenze")
st.caption(T.RIDGE_CAPTION)
st.plotly_chart(Viz.build_ridge(ridge, rsum, float(tau_r)), width="stretch", key="chart_ridge")
frozen = _frozen()
show(V.verdict_streuung(frozen))
show(V.verdict_exakt_gleit(frozen))
fig_n = Viz.build_n_reihe(frozen)
if fig_n is not None:
    st.plotly_chart(fig_n, width="stretch", key="chart_n_reihe")
    st.caption("Vorgerechnet (tools/gkd_make_frozen.py): Median der Satz-Mediane über n, σ = 1, τ = 1e-13, je n 60 Sätze zu 30 Zufalls-Seeds; die Kurve ist nicht monoton.")
st.dataframe(V.tau_table(frozen), hide_index=True, width="stretch")
st.caption("Vorgerechnet: Median der Satz-Mittel (Median der Satz-Mediane) je σ und τ, n = 60, 60 Sätze zu 30 Zufalls-Seeds; letzte Spalte: Eigenwerte, die bei τ = 1e-13 abgeschnitten werden (Median von 60).")

st.markdown("---")

# ---- Grenzen -------------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.GRENZEN_TABLE)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(T.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/gram-kondition-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Punkte $x_i \in \mathbb{R}^d$ ($d = 1$ oder $2$), Ziele $y_i$, eine Bandbreite $\sigma > 0$ und ein $\lambda \ge 0$. Gesucht sind Koeffizienten $a$ mit $(K + \lambda I)\,a = y$ und die Frage, wie weit die berechnete Lösung von der exakten abweicht.

| Symbol | Bedeutung |
|---|---|
| $K$ | Gram-Matrix, $K_{ij} = k(x_i, x_j)$: Tabelle der Ähnlichkeiten aller Punktpaare; positiv semidefinit (Mercer), in Gleitkomma bei großem $\sigma$ oft nicht mehr |
| $\sigma$, $\gamma$ | Bandbreite (Längenmaß der Eingabe) und $\gamma = 1/(2\sigma^2)$ |
| $\lambda$ | wird zur Diagonale addiert, $K + \lambda I$: Nugget (Rechenhilfe) oder Rauschvarianz des GP-Mittels (Modell) |
| $\operatorname{cond}$ | Konditionszahl $\lambda_{\max}/\lambda_{\min}$ der symmetrischen Matrix (2-Norm); unter der Rundungsgrenze nur eine Untergrenze |
| $a_{\text{exakt}}$ | exakte Lösung des Gleitkomma-Systems (Matrixeinträge als Dualbrüche, in ganzen Zahlen gelöst) |
| $\eta$ | normweiser Rückwärtsfehler $\lVert Aa - y\rVert / (\lVert A\rVert\lVert a\rVert + \lVert y\rVert)$ mit exakt gerechnetem Residuum, $A = K + \lambda I$ |
| $q$, $k$ | $q = \exp(-1/(2\sigma^2)) = 1 - 10^{-k}$ (Flat Limit, ganzzahlige Punkte) |
| $c_m$, $R_{mm}^2$ | Koeffizienten der Eigenwertleiter; Quadrat des Abstands des Monoms $x^{m-1}$ vom Span der niedrigeren Monome auf den Punkten |
| $\tau$ | Abschneidegrenze der Interpolation: Eigenwerte unter $\tau\lambda_{\max}$ werden weggelassen |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/gram-kondition-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(T.MATH_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Kernmethoden: Skalarprodukte tauschen](https://sebastianhanisch.net/konzepte-kernmethoden.html)."
)
