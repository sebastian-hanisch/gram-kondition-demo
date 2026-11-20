"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Logarithmische Achsen bekommen feste Bereiche; Werte unter dem Bereich liegen am unteren Rand (Markierung „scheitert“ oder Annotation)."""

import math

import numpy as np
import plotly.graph_objects as go

import gkd_constants as C
import gkd_kernel as Kn
from gkd_format import de, lam_text, sigma_text, tau_text


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _log_floor(values, floor):
    return [max(abs(v), floor) for v in values]


def build_eigenvalues(eigs, lam: float, n: int):
    """Eigenwerte der Gram-Matrix K (absteigend, Betrag; negative als Kreuz) mit dem Rundungsboden eps · λ_max und, bei λ > 0, dem Nugget als waagerechte Linie."""
    idx = list(range(1, len(eigs) + 1))
    top = float(eigs[0])
    neg = [i for i, v in zip(idx, eigs) if v <= 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(eigs, 1e-30), mode="lines+markers", line=dict(color=C.COLOR_LINE, width=2), marker=dict(size=7), name="Eigenwerte von K",
                             hovertemplate="Nr. %{x}: %{y:.2e}<extra></extra>"))
    if neg:
        fig.add_trace(go.Scatter(x=neg, y=_log_floor([eigs[i - 1] for i in neg], 1e-30), mode="markers", marker=dict(symbol="x", size=10, color=C.COLOR_BAD), name="Eigenwert ≤ 0 (als Betrag)"))
    fig.add_hline(y=Kn.EPS * top, line=dict(color=C.COLOR_BAD, dash="dash"), annotation_text="Rundungsboden eps · λ_max", annotation_position="bottom right")
    if lam > 0:
        fig.add_hline(y=lam, line=dict(color=C.COLOR_OK, dash="dot"), annotation_text=f"Nugget λ = {lam_text(lam)}", annotation_position="top right")
    fig.update_yaxes(type="log", range=[-32, 2], title_text="Eigenwert (Betrag)")
    fig.update_xaxes(title_text="Nummer des Eigenwerts (absteigend)", dtick=max(1, n // 10))
    return _base(fig)


def build_errors(instances):
    """Fehler der Cholesky-Lösung gegen die exakte Lösung je Punktmenge (log); scheitert die Zerlegung, steht ein graues Kreuz am unteren Rand; dazu der exakte Rückwärtsfehler η."""
    idx = [i + 1 for i in range(len(instances))]
    ok = [i for i in instances]
    cols = []
    ys = []
    for ins in ok:
        if not ins.cholesky_ok:
            ys.append(None)
            cols.append(C.COLOR_GREY)
        else:
            ys.append(max(ins.err_chol, 1e-17))
            cols.append(C.COLOR_BAD if ins.err_chol > C.ERR_BAD else (C.COLOR_FLOAT if ins.err_chol > C.ERR_SMALL else C.COLOR_OK))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(i) for i, y in zip(idx, ys) if y is not None], y=[y for y in ys if y is not None], marker_color=[c for c, y in zip(cols, ys) if y is not None],
                         name="Fehler der Cholesky-Lösung", hovertemplate="Punktmenge %{x}: %{y:.2e}<extra></extra>"))
    fail = [str(i) for i, ins in zip(idx, ok) if not ins.cholesky_ok]
    if fail:
        fig.add_trace(go.Scatter(x=fail, y=[1e-17] * len(fail), mode="markers", marker=dict(symbol="x", size=10, color=C.COLOR_GREY), name="Cholesky scheitert (die Rechnung meldet es)"))
    eta = [(str(i), ins.eta_chol) for i, ins in zip(idx, ok) if ins.cholesky_ok and not math.isnan(ins.eta_chol)]
    if eta:
        fig.add_trace(go.Scatter(x=[a for a, _ in eta], y=[max(b, 1e-17) for _, b in eta], mode="markers", marker=dict(symbol="diamond", size=7, color="black"), name="exakter Rückwärtsfehler η"))
    fig.add_hline(y=C.ERR_BAD, line=dict(color=C.COLOR_BAD, dash="dot"), annotation_text="10 %", annotation_position="top left")
    fig.update_yaxes(type="log", range=[-17.5, 1.5], title_text="Fehler / η")
    fig.update_xaxes(title_text="Punktmenge", type="category", categoryorder="array", categoryarray=[str(i) for i in idx])
    return _base(fig)


def build_nugget(rows):
    """Beobachtete Stellen der LU (kleinster Wert und Median) gegen die Regel R(λ) = 16 − log10(1 + n/λ) je λ der Reihe."""
    labels = [f"λ = {lam_text(r.lam)}" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r.digits_min for r in rows], name="beobachtete Stellen (kleinster Wert)", marker_color=C.COLOR_OK))
    fig.add_trace(go.Bar(x=labels, y=[r.digits_median for r in rows], name="beobachtete Stellen (Median)", marker_color=C.COLOR_LINE))
    fig.add_trace(go.Scatter(x=labels, y=[r.rule for r in rows], mode="markers+lines", marker=dict(symbol="diamond", size=11, color=C.COLOR_BAD), line=dict(dash="dot", color=C.COLOR_BAD),
                             name="R(λ) = 16 − log10(1 + n/λ)"))
    fig.update_yaxes(range=[0, 17], title_text="richtige Dezimalstellen")
    fig.update_layout(barmode="group")
    return _base(fig, 340)


def build_ladder(res):
    """Eigenwertleiter im Flat Limit: exakte Eigenwerte der rationalen Matrix, die geschlossene Leiter, die gerundete Matrix und die Gleitkomma-Eigenzerlegung (Betrag; negative als Kreuz)."""
    n = len(res.xs)
    idx = list(range(1, n + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(res.exact, 1e-300), mode="lines+markers", line=dict(color=C.COLOR_EXACT, width=3), marker=dict(size=9), name="exakt (rationale Matrix)"))
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(res.ladder, 1e-300), mode="lines", line=dict(color="black", dash="dash"), name="Leiter c_m (1/σ)^(2(m−1))"))
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(res.rounded, 1e-300), mode="markers", marker=dict(symbol="square-open", size=11, color=C.COLOR_ROUND), name="auf Doppelte gerundete Matrix (exakt gerechnet)"))
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(res.floating, 1e-300), mode="markers", marker=dict(symbol="circle-open", size=14, color=C.COLOR_FLOAT), name="Gleitkomma-Eigenzerlegung"))
    neg = [(i, v) for i, v in zip(idx, res.floating) if v <= 0]
    if neg:
        fig.add_trace(go.Scatter(x=[i for i, _ in neg], y=[max(abs(v), 1e-300) for _, v in neg], mode="markers", marker=dict(symbol="x", size=12, color=C.COLOR_BAD), name="Eigenwert ≤ 0 (Betrag)"))
    fig.add_hline(y=Kn.EPS * res.exact[0], line=dict(color=C.COLOR_BAD, dash="dot"), annotation_text="eps · λ_1", annotation_position="bottom right")
    lo = min(math.log10(max(abs(v), 1e-300)) for v in list(res.exact) + list(res.ladder))
    fig.update_yaxes(type="log", range=[min(lo - 1, -20), 2], title_text="Eigenwert")
    fig.update_xaxes(title_text="Nummer m des Eigenwerts", dtick=1)
    return _base(fig, 420)


def build_ridge(rs, summary, tau: float):
    """Testfehler gegen die Wahrheit je Zufalls-Seed (log): Interpolation und per Leave-one-out gewähltes λ, dazu die konstante Vorhersage."""
    idx = [str(i + 1) for i in range(len(rs.interp))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(rs.interp, 1e-6), mode="markers", marker=dict(size=9, color=C.COLOR_BAD), name=f"Interpolation (τ = {tau_text(tau)})",
                             hovertemplate="Zufalls-Seed %{x}: %{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=idx, y=_log_floor(rs.loo, 1e-6), mode="markers", marker=dict(size=9, color=C.COLOR_OK, symbol="diamond"), name="λ per Leave-one-out gewählt",
                             hovertemplate="Zufalls-Seed %{x}: %{y:.3g}<extra></extra>"))
    fig.add_hline(y=summary.const_mean, line=dict(color="black", dash="dash"), annotation_text=f"konstante Vorhersage {de(summary.const_mean, 2)}", annotation_position="top left")
    fig.update_yaxes(type="log", range=[-3, 8], title_text="Testfehler gegen die Wahrheit")
    fig.update_xaxes(title_text="Zufalls-Seed im Satz (30)", type="category")
    return _base(fig)


def build_n_reihe(frozen: dict):
    """Vorgerechnet: Median der Satz-Mediane der Interpolation über n (95-%-Band der Satz-Mediane), σ = 1, τ = 1e-13."""
    rows = frozen.get("n_reihe")
    if not rows:
        return None
    ns = [int(k) for k in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns + ns[::-1], y=[rows[str(n)]["median_hi"] for n in ns] + [rows[str(n)]["median_lo"] for n in ns][::-1], fill="toself", fillcolor="rgba(76,120,168,0.18)",
                             line=dict(width=0), hoverinfo="skip", name="95-%-Band der Satz-Mediane"))
    fig.add_trace(go.Scatter(x=ns, y=[rows[str(n)]["median_med"] for n in ns], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), name="Median der Satz-Mediane",
                             hovertemplate="n = %{x}: %{y:.3g}<extra></extra>"))
    fig.update_xaxes(type="log", title_text="Punkte n (log)", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(type="log", title_text="Testfehler der Interpolation (Satz-Median)")
    return _base(fig, 340)


def build_surface(sf):
    """Heatmap des mittleren Testfehlers (log10) über σ (Zeilen) und λ (Spalten); die beste Zelle ist markiert."""
    z = np.log10(np.array(sf.test_mean))
    text = [[de(v, 3) for v in row] for row in sf.test_mean]
    xs = [lam_text(l) for l in sf.lams]
    ys = [sigma_text(s) for s in sf.sigmas]
    fig = go.Figure(go.Heatmap(z=z, x=xs, y=ys, text=text, texttemplate="%{text}", colorscale="Viridis_r", colorbar=dict(title="log10 Testfehler", thickness=12),
                               hovertemplate="σ = %{y}, λ = %{x}: %{text}<extra></extra>"))
    bi, bj = sf.best_cell
    fig.add_shape(type="rect", x0=bj - 0.5, x1=bj + 0.5, y0=bi - 0.5, y1=bi + 0.5, line=dict(color="white", width=3))
    fig.update_xaxes(title_text="λ (Nugget bzw. Rauschvarianz)", type="category")
    fig.update_yaxes(title_text="Bandbreite σ", type="category")
    return _base(fig, 400)
