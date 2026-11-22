"""Anzeigetexte der App (und derselbe Annahmentext wie in der README: tests/test_texts.py vergleicht beide)."""

TITLE = "🧮 Bandbreite und Kondition: wo die Rechnung still falsch wird"

INTRO = """
Die **Gram-Matrix** eines Gauß-Kerns verliert mit wachsender **Bandbreite σ** ihre kleinen Eigenwerte an die Rundung, und LU und Eigenzerlegung melden das nicht; Cholesky meldet nur, wenn die Zerlegung scheitert.
Diese Seite **zählt über viele Punktmengen**, in wie vielen die **Cholesky-Zerlegung gelingt** und die Lösung trotzdem falsch ist (gemessen an der **exakten Lösung** derselben Gleitkomma-Matrix),
was ein **Nugget λ** (auch Jitter genannt, hier eine Rechenhilfe) dagegen kauft und wie die Eigenwerte im **Flat Limit** abfallen. Getrennt davon zeigt sie, dass die Interpolation (λ → 0)
auch bei exakter Rechnung ein **Modellfehler** bleibt, der von der Abschneidegrenze abhängt. Die Daten sind **ausgedacht** (Modell, keine Betriebsdaten); die Zahlen gelten für den
Gauß-Kern, d ≤ 2, n ≤ 240 und diese Plattform. Weiter unten stehen die Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

HOW_IT_WORKS = """
1. **Gram-Matrix bilden:** K_ij = exp(−‖x_i − x_j‖² / (2σ²)) für n Punkte; σ ist die Bandbreite (ein Längenmaß der Eingabe), γ = 1/(2σ²). Große σ machen alle Einträge fast gleich 1.
2. **System lösen:** (K + λI) a = y per Cholesky-Zerlegung, LU oder Eigenzerlegung. λ ist hier das **Nugget** (Rechenhilfe auf der Diagonale); in Kernel Ridge ist dasselbe λ die Rauschvarianz des GP-Mittels.
3. **Gegen die exakte Lösung prüfen:** dieselbe Gleitkomma-Matrix wird in ganzen Zahlen gelöst (Bareiss). Daraus kommen der Fehler der Koeffizienten und der exakt gerechnete Rückwärtsfehler η.
4. **Warum es kippt:** cond = λ_max/λ_min; Eigenwerte unter eps · λ_max (eps = 2,2e-16) sind nur noch Rundung. Dann ist die Matrix nahe an singulär, und ein kleines η (kleines Residuum) schließt falsche Koeffizienten nicht aus (Experiment 2 zählt das).
"""

KERNFRAGEN_HEADER = "## ❓ Kernfragen: wie viele Eigenwerte bleiben, gelingt die Rechnung, was kauft ein Nugget, wovon hängt „Interpolation“ ab?"
KERNFRAGE_TITLES = (
    "**1. Wie viele Eigenwerte hat die Gram-Matrix noch?**",
    "**2. Gelingt die Rechnung, und stimmt sie?**",
    "**3. Was kauft ein Nugget?**",
    "**4. Wovon hängt „Interpolation“ ab?**",
)

FALLE_CAPTION = (
    "Punkte gleichverteilt im Einheitsquadrat, y ~ N(0, I) je Punktmenge neu gezogen. **Die App zeigt immer die Zählung über alle Punktmengen, nie eine Einzelinstanz allein (außer den exakt gerechneten Handbeispielen):** "
    "ob die Cholesky-Zerlegung gelingt, ist ein Rundungslotto, das an Plattform und Zufalls-Strom hängt. Fehler = ‖a − a_exakt‖/‖a_exakt‖ gegen die exakte Lösung der Gleitkomma-Matrix (Bareiss, ganze Zahlen); "
    "η = exakt gerechneter normweiser Rückwärtsfehler. Live bis n = 40."
)
NUGGET_CAPTION = (
    "Das Nugget λ ist hier eine **Rechenhilfe**: cond(K + λI) ≤ 1 + n/λ gilt exakt (λ_max ≤ Spur K = n und λ_min(K + λI) ≥ λ), unabhängig von σ und der Lage der Punkte. Die Faustregel R(λ) = 16 − log10(1 + n/λ) "
    "ist beobachtet, kein Satz, und braucht keine Eigenwerte; gezählt werden Punktmengen, beobachtet werden die Stellen der LU gegen die exakte Lösung. Ein Nugget ändert aber das Modell (unten)."
)
FLAT_CAPTION = (
    "Gauß-Kern auf **ganzzahligen** Punkten, q = exp(−1/(2σ²)) = 1 − 10^−k als Bruch: so bleibt K rational und die Referenz exakt (Eigenwerte per Jacobi-Verfahren in 120 Stellen, Zähleinheit: Eigenwerte). "
    "In der Anzeige steht 1/σ, nicht ε (ε heißt in anderen Stücken Röhrenbreite und Toleranz). Gilt für den Gauß-Kern, d = 1; die Zahl echter Eigenwerte ist ein Befund der **Eingaberundung**, nicht des Verfahrens."
)
RIDGE_CAPTION = (
    "Beispiel-Regression (ausgedachte Zielfunktion auf [−2, 2]², Rauschen 0,3, 1000 rauschfreie Testpunkte). **Interpolation** = Eigenzerlegung, Eigenwerte unter τ · λ_max abgeschnitten (eine Wahl des Verfahrens). "
    "Ein Satz = 30 Zufalls-Seeds; Mittel, Median und Maximum stehen immer zusammen, weil das Mittel an einzelnen Zufalls-Seeds hängt. Der Satz ist ein Satz, nicht „die“ Zahl."
)
FLAECHE_CAPTION = (
    "Mittlerer Testfehler gegen die Wahrheit über 7 Bandbreiten σ und 6 Werte λ (42 Zellen; hier ist λ die Rauschvarianz des GP-Mittels, also Modell, nicht Rechenhilfe). "
    "Die beste Zelle ist die im Rückblick beste; „Leave-one-out“ wählt je Zufalls-Seed eine Zelle, ohne den Testfehler zu kennen."
)

GRENZEN_TABLE = """
| Annahme oder Reichweite | Was gemessen ist, und wo es endet | Wer setzt an |
|---|---|---|
| **Gauß-Kern** | Alle Zahlen gelten für den Gauß-Kern; die Matérn-Eigenwertabfälle sind in **kernwahl-demo** gemessen, hier nicht | kernwahl-demo |
| **d ≤ 2, n ≤ 240** | Flat Limit nur für d = 1 (ganzzahlige Punkte) exakt gerechnet; die exakte Elimination läuft live bis n = 40, die Interpolation bis n = 240 | - |
| **Plattform** | Die Zählung „Cholesky gelingt in k von m“ ist ein Rundungslotto und hängt an LAPACK, Plattform und Zufalls-Strom; die README nennt sie als Messung dieser Plattform | Linux-Prüfung vor dem Push |
| **Matrix-Aufbau** | Die Rundungsgrenze der Cholesky-Zerlegung hängt davon ab, wie K gebaut wird (Differenzen mit exp(−d²/(2σ²)) gegen Potenzen eines gerundeten q); die App baut mit exp | - |
| **„Interpolation ist nicht gutartig“** | Gemessen für d = 2, n = 60 und σ = 1 (und 3); bei σ = 0,3 nicht katastrophal, aber nicht durchweg besser (288 von 1800 Instanzen über der Konstanten); Benign Overfitting hat nach der Literatur Voraussetzungen (hohe Dimension, Eigenwertabfall), die Messung hier gilt für d = 2 und widerspricht ihr deshalb nicht | - |
| **Kein Nutzen im Betrieb belegt** | Gezählt werden Punktmengen, Zufalls-Seeds und Eigenwerte; ob ein Kernmodell in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, wird hier nicht behauptet | - |
| **Distanzkonzentration** | Mit wachsender Dimension rücken Abstände zusammen; in dieser Demo nicht gemessen | - |
| **Nicht angeboten, nicht gemessen** | Stabilisierte Flat-Limit-Rechnung (RBF-QR u. a.), die Matérn-Kondition, der Rundungsboden in 2D als Leiter, alles ohne Zählung über Punktmengen | - |
"""

ASSUMPTIONS_MARKDOWN = """
**Annahmen.** Gauß-Kern mit absoluter Bandbreite σ (Längenmaß der Eingabe), γ = 1/(2σ²); Punkte gleichverteilt im Einheitsquadrat bzw. auf einer Strecke, y ausgedacht (Standardnormal bzw. eine glatte Zielfunktion mit Rauschen); „exakt“ heißt die exakte Lösung des **Gleitkomma-Systems**, nicht die wahre Funktion.

**Grenzen.** Nur Gauß-Kern, d ≤ 2, n ≤ 240; exakte Elimination live bis n = 40; Flat Limit nur d = 1; nur Zählungen über Punktmengen, Zufalls-Seeds und Eigenwerte; die Zählung „Cholesky gelingt“ ist ein Rundungslotto und eine Messung dieser Plattform; kein Nutzen von Kernmodellen im Betrieb belegt.

**Bekannte Probleme und Unsicherheiten.** Fehlerziffern hängen an der LAPACK-Bibliothek und werden nur als Bänder gezählt; Stichprobenmaxima haben schwere Tails und sind kein Befund; die Messungen der README nutzen den Zufalls-Strom dieser App (SplitMix64) und sind als Band über frische Zufalls-Seeds angegeben; ein Linux-Lauf steht aus; der Mausklick im Browser ist nur über AppTest und Stichproben im Browser geprüft.
"""

MATH_MARKDOWN = r"""
**Lesehilfe.** *Gegeben:* n Punkte $x_i \in \mathbb{R}^d$ ($d = 1$ oder $2$), Ziele $y_i$, eine Bandbreite $\sigma > 0$ und ein $\lambda \ge 0$. *Gesucht:* Koeffizienten $a$ mit $(K + \lambda I)\,a = y$ und die Frage, wie weit die berechnete Lösung von der exakten abweicht.

| Symbol | Bedeutung |
|---|---|
| $K$ | Gram-Matrix, $K_{ij} = k(x_i, x_j)$; positiv semidefinit (Mercer); für verschiedene Punkte mathematisch positiv definit, bei doppelten Punkten singulär, in Gleitkomma bei großem σ oft nicht mehr positiv semidefinit |
| $\sigma$, $\gamma$ | Bandbreite (Längenmaß der Eingabe) und $\gamma = 1/(2\sigma^2)$; in der Anzeige des Flat Limit steht $1/\sigma$ |
| $\lambda$ | wird zur Diagonale addiert: $K + \lambda I$. Zwei Rollen: Nugget (Rechenhilfe, auch Jitter genannt) oder Rauschvarianz des GP-Mittels (Modell) |
| $\operatorname{cond}$ | $\lambda_{\max}/\lambda_{\min}$ der symmetrischen Matrix (2-Norm); liegt $\lambda_{\min}$ unter $\mathrm{eps}\cdot\lambda_{\max}$, ist der Wert nur eine Untergrenze |
| $a_{\text{exakt}}$ | exakte Lösung des Gleitkomma-Systems (Matrixeinträge als Dualbrüche, gelöst in ganzen Zahlen) |
| $\eta$ | normweiser Rückwärtsfehler $\lVert Aa - y\rVert / (\lVert A\rVert\lVert a\rVert + \lVert y\rVert)$ mit exakt gerechnetem Residuum, $A = K + \lambda I$ |
| $q$, $k$ | $q = \exp(-1/(2\sigma^2)) = 1 - 10^{-k}$ (Flat Limit, ganzzahlige Punkte) |
| $c_m$, $R_{mm}^2$ | Koeffizienten der Eigenwertleiter, Quadrat des Abstands des Monoms $x^{m-1}$ vom Span der niedrigeren Monome auf den Punkten |
| $\tau$ | Abschneidegrenze der Interpolation: Eigenwerte unter $\tau\lambda_{\max}$ werden weggelassen |

**Gram-Matrix und System.** Der Gauß-Kern und das lineare System:

$$K_{ij} = \exp\!\Big(-\frac{\lVert x_i - x_j\rVert^2}{2\sigma^2}\Big), \qquad (K + \lambda I)\,a = y .$$

**Kondition und Nugget.** Wegen $k(x,x) = 1$ ist $\lambda_{\max}(K) \le \operatorname{tr}K = n$, und $\lambda_{\min}(K + \lambda I) \ge \lambda$, also

$$\operatorname{cond}(K + \lambda I) = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda} \le 1 + \frac{n}{\lambda}, \qquad R(\lambda) = 16 - \log_{10}\!\Big(1 + \frac{n}{\lambda}\Big),$$

wobei $R(\lambda)$ eine **Faustregel** für die mindestens richtigen Dezimalstellen ist (beobachtet, kein Satz).

**Fehler und Rückwärtsfehler.**

$$\text{Fehler} = \frac{\lVert a - a_{\text{exakt}}\rVert}{\lVert a_{\text{exakt}}\rVert}, \qquad \eta = \frac{\lVert Aa - y\rVert}{\lVert A\rVert\,\lVert a\rVert + \lVert y\rVert} .$$

**Flat Limit** (Gauß-Kern, $d = 1$, feste Punkte, $\sigma \to \infty$): die Eigenwerte fallen in Stufen, mit der Konstanten $c_m$ aus dem Gram-Schmidt-Verfahren der Monome (eigene Rechnung, in dieser Form nicht aus einer gelesenen Quelle übernommen):

$$\lambda_m \approx c_m\,\sigma^{-2(m-1)}, \qquad c_m = \frac{R_{mm}^2}{(m-1)!} .$$

**Leave-one-out** (geschlossen, $A = K + \lambda I$): $e_i = [A^{-1}y]_i / [A^{-1}]_{ii}$ ist das Residuum, wenn Punkt $i$ nicht angepasst wird.

*Der Formelblock ist ein vollständiges Modell für die Rechnungen dieser Seite (Gram-Matrix, System, Kondition, Fehlermaße, Flat-Limit-Leiter, Leave-one-out); nicht enthalten sind die Herleitungen und die Zielfunktion der Beispiel-Regression ($\sin(1{,}5x_1) + 0{,}5\cos(2{,}5x_2) + 0{,}3x_1x_2$ auf $[-2,2]^2$ mit normalverteiltem Rauschen, Standardabweichung 0,3).*

**Literatur.** Barthelmé, Usevich: *Spectral properties of kernel matrices in the flat limit*, arXiv 1910.14067 v3 (Abstract, Abschnitt 1 samt den drei Abbildungen darin; gelesen, nicht der ganze Text). Terenin u. a.: *Numerically Stable Sparse Gaussian Processes via Minimum Separation using Cover Trees*, JMLR 25 (2024), arXiv 2210.07893 v4 (Abschnitt 2, Unterabschnitte 3 und 4; die dortige hinreichende Bedingung für das Gelingen der Cholesky-Zerlegung wird hier nicht nachgerechnet). Rasmussen, Williams: *Gaussian Processes for Machine Learning*, MIT Press 2006, Kapitel 3, Fußnote 10 (der Zuschlag „Jitter“). Die Aussagen „Cholesky scheitert ab Bandbreite ...“ stehen in keiner dieser Quellen: sie sind eigene Rechnung dieser Demo.
"""
