# Portfoliooptimierung unter Parameterunsicherheit

**Eine Untersuchung klassischer, robuster und szenariobasierter Portfoliooptimierung unter realistischen Handelsrestriktionen**

> **Forschungsfrage:** Können fortgeschrittene Portfoliooptimierungsverfahren bei unsicheren Renditeerwartungen und realistischen Handelsrestriktionen eine bessere realisierte risikoadjustierte Performance erzielen als eine einfache Gleichgewichtung?

Dieses Repository enthält eine reproduzierbare Forschungsumgebung mit **12 ETFs**, **5 Allokationsverfahren**, einem monatlichen **Walk-Forward-Backtest über 45 Monate (Januar 2023 bis September 2026)** und einer **Sensitivitätsanalyse mit 49 Parameterkonfigurationen**.

## Inhaltsverzeichnis

1. [Forschungsziel und Einordnung](#1-forschungsziel-und-einordnung)
2. [ETF-Universum](#2-etf-universum)
3. [Projektarchitektur](#3-projektarchitektur)
4. [Daten, Prognosen und Schätzmethoden](#4-daten-prognosen-und-schätzmethoden)
5. [Mathematische Optimierungsmodelle](#5-mathematische-optimierungsmodelle)
6. [Restriktionen und Transaktionskosten](#6-restriktionen-und-transaktionskosten)
7. [Walk-Forward-Backtest und Ergebnisse](#7-walk-forward-backtest-und-ergebnisse)
8. [Robustheits- und Sensitivitätsanalyse](#8-robustheits--und-sensitivitätsanalyse)
9. [Definition der Performancekennzahlen](#9-definition-der-performancekennzahlen)
10. [Installation und Ausführung](#10-installation-und-ausführung)
11. [Methodische Grenzen](#11-methodische-grenzen)
12. [Fazit](#12-fazit)

---

## 1. Forschungsziel und Einordnung

Die klassische Portfoliotheorie optimiert anhand erwarteter Renditen und geschätzter Risiken. In der Praxis sind diese Größen jedoch nicht bekannt, sondern müssen aus begrenzten historischen Stichproben geschätzt werden. Gerade die erwarteten Renditen sind häufig mit erheblichen Schätzfehlern behaftet. Mathematisch anspruchsvollere Optimierung führt deshalb nicht automatisch zu besseren Anlageergebnissen.

Untersucht werden fünf Verfahren, die unterschiedliche Antworten auf diese Unsicherheit geben: einfache Diversifikation, klassische Mittelwert-Varianz-Optimierung, zwei robuste Ansätze und eine szenariobasierte Optimierung des Verlustrisikos. Die Verfahren werden unter vergleichbaren Handelsbedingungen bewertet.

**Untersuchungscharakter:** empirisch und explorativ. Die Sensitivitätsanalyse ist **keine** historische Parameterkalibrierung. Die nachträglich erfolgreichste Parametereinstellung darf nicht als unabhängig validierte Anlagestrategie interpretiert werden.

## 2. ETF-Universum

| ETF | Anlageexposure | ETF | Anlageexposure |
|:---|:---|:---|:---|
| **SPY** | US-amerikanische Standardaktien | **EFA** | Aktien entwickelter Märkte außerhalb der USA |
| **EEM** | Schwellenländeraktien | **TLT** | Langlaufende US-Staatsanleihen |
| **IEF** | US-Staatsanleihen mittlerer Laufzeit | **GLD** | Gold |
| **VNQ** | Börsennotierte US-Immobilien | **LQD** | US-Unternehmensanleihen mit Investment Grade |
| **HYG** | US-Hochzins-Unternehmensanleihen | **XLF** | US-Finanzsektoraktien |
| **XLV** | US-Gesundheitssektoraktien | **XLE** | US-Energiesektoraktien |

Das ETF-Universum umfasst zwölf Instrumente aus unterschiedlichen Anlageklassen und Aktienmarktsegmenten. Die risikobasierten Optimierungsverfahren berücksichtigen statistische Abhängigkeiten zwischen den ETF-Renditen durch Kovarianzschätzungen beziehungsweise gemeinsame historische Renditeszenarien. Dadurch fließen Diversifikationseffekte unmittelbar in die Allokationsentscheidungen ein. Ergänzend begrenzen einheitliche Positionsobergrenzen die Konzentration auf einzelne ETFs. Wirtschaftliche Überschneidungen zwischen breiten Aktienindizes und Sektor-ETFs werden dabei nicht durch gesonderte Sektorlimits begrenzt.
## 3. Projektarchitektur

### 3.1 Aufbau der fünf Module

| Modul | Python-Datei | Aufgabe |
|:---:|:---|:---|
| **01** | `01_data_pipeline.py` | Bereinigung und Aggregation adjustierter Kursdaten; Prüfung von Datenabdeckung und zeitlicher Konsistenz |
| **02** | `02_forecasting_model_selection.py` | Vergleich historischer, statistischer und ML-basierter Renditeprognosen |
| **03** | `03_portfolio_optimization.py` | Schätzung der Optimierungseingaben und Lösung von EQ, MV, ELL, BS und STOCH |
| **04** | `04_walk_forward_backtest.py` | Einheitlicher monatlicher Backtest mit separaten, selbstfinanzierenden Portfolios |
| **05** | `05_robustness_analysis.py` | 49 Sensitivitätsfälle, KPI- und Rangmatrizen, Diagnostik und mathematische Grenzfallprüfungen |

*Hinweis zur Reproduzierbarkeit:* Die Dateien 03–05 sind die zugehörigen Implementierungen. Die Module 01 und 02 sind entsprechend den tatsächlichen Dateinamen im Repository aufgeführt; die Datenschnittstellen müssen zu den Folgemodulen passen.

### 3.2 Datenfluss

```text
Adjustierte tägliche ETF-Kurse (Yahoo Finance)
                    │
                    ▼
     01 · Datenaufbereitung und Validierung
                    │
                    ▼
          data/prices_monthly.csv
                    │
                    ├────► 02 · Prognosevergleich (Forschungsdiagnostik)
                    │
                    ▼
     03 · Portfoliooptimierung (HIST, Σ, HAC)
                    │
                    ▼
        04 · Walk-Forward-Backtest
                    │
                    ▼
             output/module4/
                    │
                    ▼
      05 · Robustheit und Sensitivität
                    │
                    ▼
             output/module5/
```

**Renditeinput der Optimierung:** Modul 02 vergleicht HIST, OLS, Random Forest und XGBoost. Für die anschließende Portfoliokonstruktion wird der **expandierende historische arithmetische Renditemittelwert (HIST)** einheitlich verwendet. Die übrigen Verfahren dienen der Prognoseevaluation und werden nicht zur Gewichtsermittlung eingesetzt.

## 4. Daten, Prognosen und Schätzmethoden

### 4.1 Datenaufbereitung und zeitliche Trennung

Die Datenbasis besteht aus adjustierten ETF-Kursen von Yahoo Finance, die auf Monatsendwerte aggregiert werden. Vor der Optimierung sind fehlende Werte, Sortierung, Vollständigkeit und einheitliche Zeitindizes zu prüfen.

Der Backtest umfasst **Januar 2023 bis September 2026**, also **45 realisierte Monatsrenditen**. Für jede Allokationsentscheidung werden ausschließlich bis zum Entscheidungsmonat verfügbare Renditebeobachtungen verwendet. Die Rendite des Folgemonats dient erst anschließend zur Bewertung des Portfolios. Dadurch wird ein direkter Zugriff auf zukünftige Renditen in der Entscheidungsschleife vermieden.

### 4.2 Auswahl der Renditeprognose für die Optimierung

Die erwarteten Renditen sind ein zentraler Input der Modelle **MV, ELL, BS und STOCH**. Deshalb wurde vor der Portfoliokonstruktion untersucht, ob statistische und maschinelle Lernverfahren gegenüber einer einfachen historischen Renditeschätzung einen hinreichend belastbaren Prognosevorteil liefern.

#### 4.2.1 Verglichene Prognoseverfahren

| Verfahren | Grundidee | Potenzieller Vorteil | Wesentliche Schwäche |
|:---|:---|:---|:---|
| **HIST** – Historischer Mittelwert | Durchschnitt bisher beobachteter Monatsrenditen je ETF | Einfach, transparent, geringe Modellkomplexität | Langsame Anpassung an Strukturbrüche; Schätzfehler |
| **OLS** – Lineare Regression | Lineare Beziehung zwischen Merkmalen und Folgemonatsrendite | Interpretierbar und effizient | Begrenzte Abbildung nichtlinearer Zusammenhänge |
| **RF** – Random Forest | Mittelwert vieler Entscheidungsbäume | Nichtlineare Zusammenhänge und Interaktionen | Instabile Signale bei schwachem Signal-Rausch-Verhältnis |
| **XGB** – XGBoost | Sequentiell optimierte Entscheidungsbäume | Flexible Modellierung komplexer Muster | Überanpassung und zeitlich instabile Muster möglich |

Die Prognoseaufgabe lautet für ETF $i$ und Entscheidungsmonat $t$:

$$
\hat\mu_{i,t+1}=f(X_{i,t}),\qquad
R_{i,t+1}=\frac{P_{i,t+1}}{P_{i,t}}-1.
$$

Dabei bezeichnet $X_{i,t}$ die zum Entscheidungszeitpunkt verfügbaren Merkmale. Das tatsächlich realisierte $R_{i,t+1}$ ist bei der Prognose **noch nicht bekannt**.

**Verwendete Merkmale:** Momentum über **1, 3, 6 und 12 Monate**, realisierte Volatilität über **3, 6 und 12 Monate** sowie maximaler Drawdown über **12 Monate**. Die Modelle werden anhand zeitlich geordneter Beobachtungen bewertet.

#### 4.2.2 Bewertungskennzahlen

Die Prognosequalität wird nicht anhand einer einzigen Kennzahl beurteilt:

| Kennzahl | Aussage | Interpretation |
|:---|:---|:---|
| **RMSE** | Wurzel des mittleren quadrierten Prognosefehlers | Niedriger ist besser; große Fehler werden stärker bestraft |
| **MAE** | Mittlerer absoluter Prognosefehler | Niedriger ist besser |
| **Pearson-IC** | Zusammenhang zwischen prognostizierten und realisierten Renditen im ETF-Querschnitt | Höher ist besser; misst lineare Querschnittsinformation |
| **Rank-IC (Spearman)** | Zusammenhang zwischen prognostizierter und realisierter ETF-Rangfolge | Höher ist besser; misst Rangordnungsqualität |

Formal gilt für $n$ Prognosebeobachtungen:

$$
\mathrm{RMSE}=\sqrt{\frac1n\sum_{j=1}^n(y_j-\hat y_j)^2},
\qquad
\mathrm{MAE}=\frac1n\sum_{j=1}^n|y_j-\hat y_j|.
$$

Für die Portfolioallokation ist neben dem absoluten Prognosefehler auch relevant, ob das Modell **Renditeunterschiede zwischen ETFs** sinnvoll erkennt. Allerdings beweist selbst ein positiver IC noch keinen ökonomisch verwertbaren Vorteil nach Transaktionskosten.

#### 4.2.3 Zeitliche Validierung und Test: tatsächliche Ergebnisse aus Modul 02

Die Prognosemodelle werden anhand von **2.256 vollständig geprüften ETF-Monatsbeobachtungen** mit acht Merkmalen und dem Zielwert der **Folgemonatsrendite** verglichen. Der Label-Audit bestätigt die Übereinstimmung der Zielwerte mit den Monatspreisen (maximaler numerischer Fehler: $9{,}97\cdot10^{-17}$). Es gibt keine fehlenden Werte. Die zeitliche Trennung erfolgt ohne zufällige Splits; Folgemonatsrenditen stehen erst nach Ende des betreffenden Monats als Trainingslabels zur Verfügung.

- **Validierung:** Januar 2018 bis Dezember 2022 (**60 Monate**). Die Modell- und Hyperparameterwahl erfolgt **ausschließlich nach dem niedrigsten Validierungs-RMSE**.
- **Test:** Januar 2023 bis August 2026 (**44 Monate**). Die zuvor fixierten Modelle werden nur evaluiert; der Test wird **nicht** zur nachträglichen Modellauswahl verwendet.
- **Hyperparameter des validierungsseitig ausgewählten RF:** `n_estimators=120`, `max_depth=3`, `min_samples_leaf=15`. Für XGB wurden `n_estimators=120`, `max_depth=2`, `learning_rate=0.03`, `subsample=1.0`, `colsample_bytree=1.0` fixiert.

**Validierungsergebnisse (2018–2022)**

| Modell | RMSE ↓ | MAE ↓ | Pearson-IC ↑ | Spearman-IC ↑ |
|:---|---:|---:|---:|---:|
| **RF** | **0,054270** | 0,038017 | 0,054265 | 0,019431 |
| **HIST** | 0,054645 | 0,038163 | 0,058424 | 0,065967 |
| **OLS** | 0,054754 | **0,037941** | **0,073964** | **0,091259** |
| **XGB** | 0,055918 | 0,038637 | 0,016135 | 0,002985 |

**Ergebnis der vorab festgelegten Modellauswahl:** RF erzielt mit **0,054270** den niedrigsten Validierungs-RMSE und wird daher von Modul 02 ausgewählt. Sein Vorsprung gegenüber HIST beträgt **0,000375 RMSE-Punkte**. Andere Kennzahlen führen zu anderen Spitzenreitern: OLS hat den niedrigsten Validierungs-MAE und die höchsten Validierungs-ICs. Die Auswahlregel bleibt dennoch RMSE und wird nicht nachträglich geändert.

**Testergebnisse (2023–August 2026)**

| Modell | RMSE ↓ | MAE ↓ | Pearson-IC ↑ | Spearman-IC ↑ |
|:---|---:|---:|---:|---:|
| **HIST** | **0,040481** | **0,030624** | **0,105406** | **0,115385** |
| **RF** | 0,041247 | 0,031266 | −0,025131 | −0,041151 |
| **OLS** | 0,041341 | 0,031307 | 0,025049 | 0,020820 |
| **XGB** | 0,041453 | 0,031316 | −0,013962 | −0,026173 |

**Testbefund:** HIST hat in dieser Periode den geringsten RMSE und MAE sowie die höchsten Pearson- und Spearman-ICs. Der Test-RMSE von RF liegt um **0,000766** beziehungsweise rund **1,89 %** über dem HIST-RMSE. Zudem sind die durchschnittlichen monatlichen ICs von RF und XGB negativ, während HIST bei beiden IC-Maßen positiv abschneidet. Die IC-Berechnung basiert bei jedem Verfahren auf **60 gültigen Validierungs- und 44 gültigen Testmonaten**.

**Explorative Unsicherheitsdiagnose:** Der gepaarte Vergleich der monatlichen quadrierten Prognosefehler von RF gegenüber HIST ergibt über 44 Testmonate eine mittlere MSE-Differenz von **+0,000063** (RF minus HIST). Das ausgewiesene Block-Bootstrap-Intervall beträgt **[0,000009; 0,000086]**, bei Blocklänge sechs Monate und 2.000 Wiederholungen. Die positive Differenz spricht in dieser Stichprobe für geringere Prognosefehler von HIST. **Dieses Intervall ist explorativ**, da die Testperiode bereits betrachtet wurde; es ist kein Nachweis einer universellen oder unabhängig bestätigten Überlegenheit.

#### 4.2.4 Warum HIST als Renditeinput verwendet wird

**Die Entscheidung für HIST ist eine bewusst festgelegte Wahl des Portfolio-Untersuchungsdesigns, die durch die beobachteten Testergebnisse gestützt wird. Sie darf nicht als Ergebnis der ursprünglichen, ausschließlich validierungsbasierten Modellauswahl ausgegeben werden: Diese wählte RF.** Die beiden Entscheidungsebenen werden daher getrennt dokumentiert.

1. **Empirische Plausibilität im beobachteten Test:** HIST liefert über die 44 Testmonate den niedrigsten RMSE und MAE sowie die höchsten Querschnitts-ICs. Damit zeigt das aktuelle Experiment **keinen stabilen Testvorteil** der komplexeren ML-Verfahren gegenüber HIST. Besonders die negativen RF- und XGB-Test-ICs sprechen gegen die Annahme, ihre Prognoserangfolgen seien im beobachteten Zeitraum verlässlich besser.
2. **Kontrollierter Optimierungsvergleich:** MV, ELL, BS und STOCH verwenden denselben HIST-Erwartungsrenditevektor. So sind Unterschiede der realisierten Allokationen vor allem auf die jeweiligen Optimierungsziele, Risikomaße und Unsicherheitsannahmen zurückzuführen. EQ bleibt ohne Renditeprognose in seiner Zielfunktion.
3. **Weniger zusätzliche Schätzkomplexität:** HIST erfordert weder eine Regression noch die Hyperparameterwahl von Baumverfahren. Das vermeidet zusätzliche Modellfreiheitsgrade bei ohnehin verrauschten monatlichen ETF-Renditen. Geringere Komplexität garantiert allerdings keine bessere Prognose.
4. **Passende Unsicherheitsmodellierung:** Für den historischen Mittelwert lässt sich die Schätzunsicherheit mit einer **HAC-basierten Matrix** $\hat\Omega_t$ beschreiben. Diese fließt in die Unsicherheitsbehandlung von ELL und BS ein. Die im Prognosemodul separat diagnostizierte Fehlerkovarianz mit Ledoit-Wolf-Shrinkage (**0,1118**) ist **keine kalibrierte $\hat\Omega_t$-Matrix** und wird nicht mit ihr gleichgesetzt.
5. **Reproduzierbarkeit und ökonomische Interpretierbarkeit:** Der expandierende arithmetische Mittelwert ist für jeden ETF nachvollziehbar und ohne zusätzliche Trainingsarchitektur aktualisierbar. Die Untersuchung prüft primär, wie Optimierungsverfahren mit **unsicheren** Renditeschätzungen umgehen – nicht, welcher Prognosealgorithmus unter allen denkbaren Bedingungen optimal ist.

Für ETF $i$ ergibt sich zum Entscheidungszeitpunkt $t$:

$$
\hat\mu_{i,t}=\frac{1}{T_{i,t}}\sum_{s=1}^{T_{i,t}}R_{i,s}.
$$

Dabei ist $T_{i,t}$ die Anzahl der bis zum Entscheidungszeitpunkt verfügbaren Monatsrenditen. Das historische Fenster wächst im Zeitverlauf. Die zwölf ETF-Schätzwerte bilden den Vektor $\hat{\boldsymbol\mu}_t$; zukünftige Renditen werden nicht verwendet.

**Grenze der Schlussfolgerung:** Die besseren HIST-Testkennzahlen sind **ex post beobachtete** Ergebnisse. Sie machen HIST als Input ökonomisch plausibel, dürfen aber nicht als vorab unabhängige Bestätigung einer anhand derselben Testperiode getroffenen Modellentscheidung gelten. Insbesondere wurde **kein separater RF-basierter Walk-Forward-Portfoliovergleich** vorgelegt, der eine höhere Nettoperformance von HIST-basierten Portfolios gegenüber RF-basierten Portfolios belegen würde.

#### 4.2.5 Konsequenz für den Portfoliovergleich

Die empirische Untersuchung vergleicht **EQ, MV, ELL, BS und STOCH** bei identischem ETF-Universum, denselben Handelsrestriktionen und – sofern eine Renditeprognose erforderlich ist – dem gemeinsamen **HIST-Renditevektor**. Die 45 realisierten Backtest-Monate (**Januar 2023 bis September 2026**) sind von den **44 Testmonaten des Prognosemoduls** (**Januar 2023 bis August 2026**) zu unterscheiden: Die Prognosebewertung benötigt jeweils ein bekanntes Folgemonatslabel. Der Vergleich untersucht die Wirkung verschiedener Optimierungsziele auf Nettorendite, Volatilität, Drawdown, CVaR und Handelskosten; die Modelle verwenden teilweise unterschiedliche zusätzliche Risiko- und Unsicherheitsschätzer.

### 4.3 Kovarianz und HAC-Schätzunsicherheit

Die monatliche Renditekovarianzmatrix $\hat\Sigma_t$ wird in der Basiskonfiguration aus einem **rollierenden Fenster von 36 Monaten** geschätzt.

Für die Schätzunsicherheit des Mittelwerts wird eine **HAC-Schätzung** (*heteroskedasticity and autocorrelation consistent*) mit Bartlett-Gewichten verwendet:

$$
\hat S_t=\hat\Gamma_0+\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)
\left(\hat\Gamma_k+\hat\Gamma_k^{\top}\right),
\qquad \hat\Omega_t=\frac{\hat S_t}{T_t}.
$$

Die Basiskonfiguration verwendet **$L=3$ Verzögerungen (Lags)**. Der Unterschied ist zentral:

- $\hat\Sigma_t$ beschreibt die **Schwankungen der monatlichen Anlagerenditen**.
- $\hat\Omega_t$ beschreibt die **geschätzte Unsicherheit des Renditemittelwerts**.

Beide Matrizen erfüllen unterschiedliche Funktionen in den Optimierungsproblemen. Bei Bedarf werden numerische Korrekturen vorgenommen, damit die verwendeten Matrizen positiv semidefinit sind.

## 5. Mathematische Optimierungsmodelle

Es sei $\mathbf w\in\mathbb R^{12}$ der Vektor der Portfoliozielgewichte, $\hat\mu$ der geschätzte monatliche Renditevektor, $\hat\Sigma$ die monatliche Kovarianzmatrix und $\mathcal W$ die Menge zulässiger Portfolios (siehe Abschnitt 6).

### 5.1 EQ – Gleichgewichtung als Referenzstrategie

$$
\min_{\mathbf w\in\mathcal W}\left\|\mathbf w-\frac{1}{12}\mathbf 1\right\|_2^2.
$$

**Ökonomische Idee:** Kapital möglichst gleichmäßig auf alle zwölf ETFs verteilen. Anders als die übrigen Ansätze nutzt EQ weder geschätzte Renditen noch Kovarianzen in seiner Zielfunktion. Wegen der gemeinsamen Umschichtungsgrenze kann das realisierte Portfolio zeitweise von exakt $1/12$ je ETF abweichen.

### 5.2 MV – Klassische Mittelwert-Varianz-Optimierung (nach Harry Markowitz)

$$
\max_{\mathbf w\in\mathcal W}
\lbrace \hat\mu^\top\mathbf w-\lambda\,\mathbf w^\top\hat\Sigma\mathbf w \rbrace.
$$

**Ökonomische Idee:** Eine möglichst hohe erwartete Rendite erzielen und gleichzeitig die Varianz des Portfolios begrenzen. Der Parameter $\lambda\geq0$ steuert die Gewichtung des Varianzterms (**Basis: $\lambda=3$**). Das Verfahren kann auf kleine Unterschiede in geschätzten Renditen empfindlich reagieren.

### 5.3 ELL – Ellipsoidale robuste Optimierung

Die unbekannte tatsächliche Renditeerwartung wird innerhalb einer ellipsoidalen Unsicherheitsmenge modelliert:

$$
\mathcal U_{\mathrm{ELL}}=
\lbrace \hat\mu+\rho\hat\Omega^{1/2}\mathbf u:
\|\mathbf u\|_2\leq1 \rbrace.
$$

Daraus folgt die robuste Zielfunktion:

$$
\max_{\mathbf w\in\mathcal W}
\lbrace \hat\mu^\top\mathbf w
-\lambda\mathbf w^\top\hat\Sigma\mathbf w
-\rho\sqrt{\mathbf w^\top\hat\Omega\mathbf w} \rbrace.
$$

**Ökonomische Idee:** Portfolios werden zusätzlich bestraft, wenn ihre erwartete Rendite besonders stark von unsicheren Schätzungen abhängt. Der Radius $\rho$ steuert diese Absicherung (**Basis: $\rho=1$**). Bei **$\rho=0$** fällt die zusätzliche Strafe weg und ELL ist mathematisch identisch mit MV.

### 5.4 BS – Budgetierte robuste Optimierung nach Bertsimas–Sim

BS begrenzt ungünstige Abweichungen der erwarteten Einzelrenditen durch ein gemeinsames Unsicherheitsbudget. Für jedes Asset gilt:

$$
d_i=\kappa\sqrt{\hat\Omega_{ii}}.
$$

Hier skaliert $\kappa$ die individuelle Schätzunsicherheit. Das Budget $\Gamma\in[0,12]$ steuert, wie viele adverse Abweichungen aggregiert wirksam werden können. Die implementierte duale Formulierung für Long-only-Portfolios lautet:

$$
\begin{aligned}
\max_{\mathbf w\in\mathcal W,\,p\geq0,\,\mathbf q\geq0}\quad &
\hat\mu^\top\mathbf w-\lambda\mathbf w^\top\hat\Sigma\mathbf w
-\Gamma p-\sum_{i=1}^{12}q_i\\
\text{unter den Bedingungen}\quad &p+q_i\geq d_iw_i,
\quad i=1,\ldots,12.
\end{aligned}
$$

**Ökonomische Idee:** Nicht alle Renditeschätzungen müssen gleichzeitig ihren ungünstigsten Wert annehmen. Stattdessen wird die Gesamtheit möglicher negativer Abweichungen über $\Gamma$ budgetiert (**Basis: $\Gamma=3$, $\kappa=1$**). Bei **$\Gamma=0$** reduziert sich BS mathematisch auf MV. Die Implementierung verwendet in diesem Grenzfall unmittelbar dieselbe MV-Formulierung, um unnötige numerische Unterschiede zwischen Solver-Formulierungen zu vermeiden.

### 5.5 STOCH – Szenariobasierte Optimierung mit CVaR

STOCH verwendet **120 gemeinsame historische Monatsrenditeszenarien** und bestraft den *Conditional Value at Risk* (CVaR) der Portfolioverluste:

$$
\max_{\mathbf w\in\mathcal W,\,\eta,\,\mathbf u\geq0}
\lbrace \hat\mu^\top\mathbf w-\gamma_{\mathrm{CVaR}}
\left(\eta+\frac{1}{(1-\alpha)S}\sum_{s=1}^{S}u_s\right) \rbrace
$$

unter den Nebenbedingungen

$$
u_s\geq-\mathbf r_s^\top\mathbf w-\eta,
\qquad s=1,\ldots,S.
$$

Dabei ist $\eta$ eine Verlustschwelle und $u_s$ beschreibt die darüber hinausgehenden Verluste. In der Basis gelten **$\alpha=90\%$** und **$\gamma_{\mathrm{CVaR}}=1$**.

**Wichtig:** Dieses Modell optimiert die erwartete Rendite unter Berücksichtigung extremer **monatlicher Verluste**, **nicht** die Szenariovarianz. CVaR und maximaler Drawdown sind unterschiedliche Risikomaße.

## 6. Restriktionen und Transaktionskosten

### 6.1 Gemeinsame zulässige Portfolios

Alle fünf Verfahren unterliegen derselben zulässigen Menge:

$$
\mathcal W=\lbrace \mathbf w:\sum_{i=1}^{12}w_i=1,\quad
0\leq w_i\leq0{,}30,\quad
\|\mathbf w-\mathbf w_t^-\|_1\leq0{,}50 \rbrace.
$$

$\mathbf w_t^-$ bezeichnet die Gewichte unmittelbar **vor** einer Umschichtung. Die $L^1$-Grenze von 0,50 ist die **Summe der absoluten Gewichtsänderungen** und nicht gleichbedeutend mit einer einseitigen Umschichtungsquote von 50 %. Sie gilt nach der anfänglichen Investition.

| Annahme | Basiskonfiguration | Bedeutung |
|:---|:---|:---|
| Startkapital | **10.000 €** | Gleicher Ausgangswert für alle fünf Portfolios |
| Rebalancing | **Monatlich** | Zielgewichte auf Basis zuvor verfügbarer Informationen |
| Investitionsgrad | **100 %** | Voll investiert; keine Short-Positionen |
| Maximales ETF-Gewicht | **30 %** | Konzentrationsbegrenzung pro ETF |
| $L^1$-Turnover-Grenze | **0,50** | Begrenzung monatlicher Gewichtsänderungen |
| Handelsgebühr | **0,20 %** | Auf den absoluten gehandelten Eurobetrag |
| Kovarianzfenster | **36 Monate** | Rollierende Schätzung |
| STOCH-Szenarien | **120 Monate** | Historische gemeinsame Renditevektoren |
| Erstinvestition | **Aus Bargeld** | Auch anfängliche Käufe verursachen Gebühren |
| Optimierung | **CVXPY / CLARABEL** | Konvexe mathematische Optimierung |

### 6.2 Selbstfinanzierende Umschichtung

Transaktionskosten werden dem **vorhandenen Portfoliovermögen** belastet; es wird kein zusätzliches Kapital zugeführt. Für den Depotwert vor dem Handel $V^-$ und nach dem Handel $V^+$ gilt:

$$
V^+=V^- - c\sum_i\left|V^+w_i-h_i^-\right|,
\qquad h_i^+=V^+w_i.
$$

$c$ ist der proportionale Kostensatz und $h_i^-$ der Eurobestand der Position vor dem Handel. Die Gleichung wird numerisch mittels **Bisektion** gelöst. Dadurch reduzieren Handelskosten unmittelbar das tatsächlich investierbare Vermögen.

**Ausführungsvereinfachung:** Der Backtest arbeitet mit einer idealisierten Monatsendkonvention. Eine erst anhand des beobachteten Monatsschlusskurses berechnete Allokation könnte nicht tatsächlich noch zu genau diesem Schlusskurs umgesetzt werden. Es handelt sich daher um eine **modellbasierte Monatsrenditesimulation**, nicht um eine vollständig handelbare Intraday- oder Next-Open-Strategie. Geld-Brief-Spannen, Slippage, Steuern, Markteinfluss und Liquiditätsänderungen werden nicht separat modelliert.

## 7. Walk-Forward-Backtest und Ergebnisse

**Zeitraum:** Januar 2023 bis September 2026 · **45 Monatsrenditen** · **Startkapital: 10.000 € je Strategie**. Alle Modelle verwenden dieselben Marktbeobachtungen und vergleichbare Handelsrestriktionen. Die Basisparameter wurden nicht nachträglich aus den Sensitivitätssiegern abgeleitet.

### 7.1 Rendite- und Risikokennzahlen

Die breite Ergebnistabelle wird zur besseren Lesbarkeit in **zwei thematische Tabellen** aufgeteilt.

**Performance und Portfoliowert**

| Modell | Nettorendite p. a. | Volatilität p. a. | Sharpe ($r_f=0$) | Endvermögen |
|:---|---:|---:|---:|---:|
| **MV** | **15,97 %** | 9,73 % | **1,580** | **17.431,75 €** |
| **ELL** | 15,69 % | 9,80 % | 1,545 | 17.274,05 € |
| **BS** | 14,70 % | 9,19 % | 1,545 | 16.727,40 € |
| **EQ** | 11,25 % | 9,46 % | 1,178 | 14.913,49 € |
| **STOCH** | 9,52 % | **8,42 %** | 1,125 | 14.062,40 € |

**Verlustrisiken und Handelskosten**

| Modell | Maximaler Drawdown | Monatlicher CVaR | Gesamte Gebühren |
|:---|---:|---:|---:|
| **MV** | −7,18 % | −5,51 % | 109,07 € |
| **ELL** | −7,44 % | −5,52 % | 86,43 € |
| **BS** | **−5,34 %** | −5,59 % | 94,95 € |
| **EQ** | −7,64 % | −6,86 % | **44,51 €** |
| **STOCH** | −8,14 % | **−4,23 %** | 81,36 € |

*Hinweise:* Alle Renditen und Endvermögen sind **nach modellierten Transaktionskosten** angegeben. Bei Drawdown und dem hier negativ ausgewiesenen CVaR ist ein Wert näher an null günstiger. Für die Sharpe-Ratio wird $r_f=0$ als Vergleichskonvention verwendet; dies bedeutet nicht, dass die historischen risikofreien Zinsen tatsächlich null waren. Zahlen sind gerundet.

### 7.2 Rangmatrix der Basiskonfiguration

**Rang 1** steht jeweils für das beste Ergebnis der betreffenden Kennzahl. Die Ränge sind bewusst **nicht zu einer Gesamtnote addiert**, da die Kennzahlen unterschiedliche Ziele abbilden und teilweise dieselbe Information mehrfach messen.

| Modell | Rendite | Volatilität | Sharpe | Drawdown | CVaR | Endwert | Gebühren |
|:---|---:|---:|---:|---:|---:|---:|---:|
| **MV** | **1** | 4 | **1** | 2 | 2 | **1** | 5 |
| **ELL** | 2 | 5 | 3 | 3 | 3 | 2 | 3 |
| **BS** | 3 | 2 | 2 | **1** | 4 | 3 | 4 |
| **EQ** | 4 | 3 | 4 | 4 | 5 | 4 | **1** |
| **STOCH** | 5 | **1** | 5 | 5 | **1** | 5 | 2 |

### 7.3 Ökonomische Interpretation der fünf Verfahren

**MV – höchste realisierte Rendite und Sharpe-Ratio.** Die klassische Mittelwert-Varianz-Optimierung liegt im untersuchten Zeitraum bei beiden Kennzahlen vorn. Das zeigt, dass ein vergleichsweise einfacher Optimierungsansatz mit historischen Schätzgrößen unter Restriktionen erfolgreich sein kann. Daraus folgt jedoch weder eine allgemein hohe Prognosegüte des Renditemittelwerts noch eine Überlegenheit in zukünftigen Marktphasen.

**ELL – konkurrenzfähig, aber abhängig vom Robustheitsradius.** ELL liegt in der Basis nahe bei MV und verursacht geringere Handelskosten. Moderate Unsicherheitsstrafen können in einzelnen Sensitivitätsfällen zu höheren Renditen führen; eine zunehmende Absicherung reduziert aber irgendwann die realisierte Rendite. Robustheit besitzt somit einen ökonomischen Preis.

**BS – günstigster maximaler Drawdown in der Basis.** Mit **−5,34 %** weist BS den geringsten kumulierten Höchstverlust auf, obwohl die Rendite unter MV und ELL liegt. Die budgetierte Unsicherheit verändert die Allokation offenbar so, dass sie im untersuchten Verlauf günstiger auf kumulierte Verluste wirkt. Eine Minimierung des maximalen Drawdowns ist allerdings **nicht** die direkte BS-Zielfunktion; das Ergebnis ist stichprobenabhängig.

**STOCH – geringste Volatilität und günstigster monatlicher CVaR.** STOCH erreicht die niedrigste realisierte Volatilität (**8,42 %**) und den günstigsten monatlichen CVaR (**−4,23 %**). Dies geht in der Basis mit der geringsten Rendite und Sharpe-Ratio einher. Dass STOCH gleichzeitig **nicht** den besten maximalen Drawdown besitzt, unterstreicht den Unterschied zwischen monatlichem Tail-Risiko und kumuliertem Verlustpfad.

**EQ – transparent und kostengünstig.** Die Gleichgewichtung benötigt keine Rendite- oder Kovarianzschätzung in der Zielfunktion und weist mit **44,51 €** die niedrigsten modellierten Handelskosten auf. Im aktuellen Backtest ist sie jedoch weder Rendite- noch Sharpe-Siegerin. Die Ergebnisse widerlegen somit eine pauschale Annahme, einfache Gleichgewichtung sei bei Schätzunsicherheit stets überlegen.

## 8. Robustheits- und Sensitivitätsanalyse

Modul 05 untersucht **49 Konfigurationen einschließlich der Basis**. Es handelt sich um eine *One-Factor-at-a-Time*-Analyse (OFAT): Je Fall wird **ein Parameter** verändert, während das übrige Backtestdesign konstant bleibt.

### 8.1 Untersuchte Parameter

| Parameter | Sensitivitätswerte | Basiswert |
|:---|:---|:---|
| Handelsgebühr | 0 % · 0,10 % · 0,50 % · 1,00 % | 0,20 % |
| Max. ETF-Gewicht | 20 % · 40 % · 60 % · 100 % | 30 % |
| $L^1$-Turnover-Grenze | 0,25 · 0,75 · 1,00 | 0,50 |
| Kovarianzfenster | 24 · 60 Monate | 36 Monate |
| Risikoaversion $\lambda$ (MV/ELL/BS) | 0,5 · 1 · 2 · 5 · 8 · 12 | 3 |
| ELL-Radius $\rho$ | 0 · 0,25 · 0,5 · 1,5 · 2 · 3 | 1 |
| BS-Budget $\Gamma$ | 0 · 1 · 2 · 4 · 6 · 9 · 12 | 3 |
| BS-Skalierung $\kappa$ | 0,5 · 1,5 · 2 · 3 | 1 |
| STOCH-CVaR-Gewicht $\gamma$ | 0,25 · 0,5 · 1,5 · 2 · 3 | 1 |
| STOCH-Konfidenzniveau $\alpha$ | 95 % | 90 % |
| STOCH-Szenariofenster | 60 · 84 Monate | 120 Monate |
| HAC-Lags | 0 · 1 · 6 · 12 | 3 |

Für **jeden Fall** erstellt das Modul eine **absolute KPI-Matrix** und eine **separate Rangmatrix** mit Kennzahlensiegern. Wirtschaftlich kaum unterscheidbare Werte können innerhalb festgelegter Berichtstoleranzen als Gleichstand ausgewiesen werden. Ein Rang ist eine deskriptive Einordnung und **kein statistischer Signifikanztest**.

### 8.2 Die wichtigsten Befunde

1. **MV führt häufig bei Rendite und Sharpe.** Besonders bei Änderungen der Gebühren oder der Umschichtungsgrenze bleibt MV im untersuchten Raster konkurrenzfähig. Die 49 Fälle sind jedoch voneinander abhängig und keine 49 unabhängigen Markttests.
2. **Moderate ELL-Robustheit kann vorteilhaft sein.** Bei $\rho=0{,}25$ und $0{,}5$ gewinnt ELL im beobachteten Zeitraum Rendite und Sharpe; ab $\rho=1{,}5$ liegt MV vorn. Die Wirkung robuster Absicherung ist nicht monoton.
3. **Risikoaversion verändert die Rangfolge.** Bei $\lambda=0{,}5$, $1$ und $2$ führt ELL bei der Rendite, bei $\lambda=5$ MV und bei $\lambda=8$ und $12$ EQ. EQ und STOCH verwenden diesen spezifischen Parameter nicht in ihren Zielfunktionen.
4. **Konzentrationsgrenzen beeinflussen die Modellergebnisse.** Bei 20 % und 40 % maximalem ETF-Gewicht führt MV; bei 60 % und 100 % ELL. Eine größere zulässige Portfoliomenge garantiert keine höhere spätere Rendite.
5. **BS zeigt häufig den günstigsten maximalen Drawdown.** Das ist ein empirisches Ergebnis der beobachteten Vermögenspfade, keine garantierte Eigenschaft des BS-Modells.
6. **STOCH führt in allen 49 Fällen beim empirischen monatlichen CVaR.** Daraus folgt keine automatische Überlegenheit bei maximalem Drawdown, Sharpe oder Rendite.
7. **Gebühren gehören in den Performancevergleich.** MV bleibt in den ausgewiesenen Gebührenvarianten von 0 % bis 1 % Renditesieger; die Gebühren beeinflussen dennoch das tatsächlich verfügbare Endvermögen.

### 8.3 Ausgewählte Sensitivitätsergebnisse

**A. ELL-Robustheitsradius $\rho$**

| $\rho$ | ELL-Nettorendite p. a. | Renditesieger |
|---:|---:|:---|
| 0 | 15,97 % | ELL = MV |
| 0,25 | 16,15 % | **ELL** |
| 0,50 | **16,18 %** | **ELL** |
| 1,00 (Basis) | 15,69 % | MV |
| 1,50 | 14,95 % | MV |
| 2,00 | 13,92 % | MV |
| 3,00 | 12,27 % | MV |

Die Ergebnisse zeigen, dass ein moderater Robustheitsradius im untersuchten Zeitraum hilfreich sein kann. Bei stärkerer Absicherung sinkt die realisierte ELL-Rendite deutlich. **$\rho=0{,}5$ ist ein nachträglich beobachteter Sensitivitätssieger und keine unabhängig getestete Parametervorgabe.**

**B. Risikoaversion $\lambda$**

| $\lambda$ | Renditesieger | Dessen Nettorendite p. a. |
|---:|:---|---:|
| 0,5 | ELL | 14,89 % |
| 1 | ELL | 16,19 % |
| 2 | **ELL** | **16,44 %** |
| 3 (Basis) | MV | 15,97 % |
| 5 | MV | 14,03 % |
| 8 | EQ | 11,25 % |
| 12 | EQ | 11,25 % |

Bei hohen $\lambda$-Werten sinkt die Attraktivität der varianzbasierten Verfahren in der realisierten Renditewertung. Dass EQ gewinnt, bedeutet **nicht**, dass EQ selbst durch $\lambda$ verbessert wurde: EQ bleibt bei dieser Parametervariation unverändert.

**C. Maximales Gewicht je ETF**

| Gewichtsobergrenze | Renditesieger | Dessen Nettorendite p. a. |
|---:|:---|---:|
| 20 % | **MV** | **16,25 %** |
| 30 % (Basis) | MV | 15,97 % |
| 40 % | MV | 14,58 % |
| 60 % | ELL | 12,51 % |
| 100 % | ELL | 12,29 % |

Weniger restriktive Konzentrationsgrenzen können Portfolios stärker auf einzelne Renditeschätzungen ausrichten. Das kann das **optimierte Ziel** verbessern, aber die **realisierte zukünftige Performance** verschlechtern.

### 8.4 Mathematische Grenzfälle

Zwei zentrale Identitäten dienen der Prüfung der Modellkonsistenz:

$$
\boxed{\mathrm{ELL}(\rho=0)=\mathrm{MV}}
\qquad
\boxed{\mathrm{BS}(\Gamma=0)=\mathrm{MV}}.
$$

Bei identischen Daten, Nebenbedingungen und übrigen Parametern sind die jeweiligen Optimierungsprobleme mathematisch gleich. Modul 03 verwendet für diese Grenzfälle dieselbe MV-Formulierung; Modul 05 erstellt dazu numerische Diagnosen und berücksichtigt Gleichstände in den Rangtabellen.

**Wichtig:** Die Kennzeichnung „ELL = MV“ oder „BS = MV“ in der Siegerübersicht belegt für sich allein **keinen** bestandenen numerischen Audit. Dessen tatsächliches Ergebnis steht in `mathematical_equivalence_audit.csv`.

### 8.5 Aussagegrenzen der Sensitivitätsanalyse

- **Keine historische Parameterkalibrierung:** Eine Auswahl von $\rho=0{,}5$ oder $\lambda=2$ anhand der Ergebnisse von 2023–2026 würde die Testperiode nachträglich zur Auswahl verwenden.
- **Keine 49 unabhängigen Tests:** Alle Konfigurationen beruhen auf denselben Marktmonaten und teils sehr ähnlichen Portfolios.
- **Keine statistische Signifikanzbehauptung:** 45 Monatsbeobachtungen sind eine begrenzte Grundlage, insbesondere für Sharpe-, Drawdown- und Tail-Risk-Vergleiche.
- **Keine allgemeingültige Überlegenheit:** Ergebnisse hängen von ETF-Auswahl, Zeitraum, Restriktionen, Gebühren und Marktlage ab.
- **Keine unmittelbare Vergleichbarkeit der Risikoparameter:** $\lambda$, $\rho$, $\Gamma$, $\kappa$ und $\gamma_{\mathrm{CVaR}}$ gewichten unterschiedliche mathematische Größen.

## 9. Definition der Performancekennzahlen

$R_t$ bezeichnet die realisierte **monatliche Nettorendite**, $V_t$ den Depotwert und $T$ die Anzahl der Testmonate.

| Kennzahl | Definition | Bevorzugte Richtung |
|:---|:---|:---|
| **Annualisierte Nettorendite** | $(V_T/V_0)^{12/T}-1$ | Höher |
| **Annualisierte Volatilität** | $\mathrm{SD}(R_t)\sqrt{12}$ | Niedriger, bei sonst gleichen Bedingungen |
| **Sharpe-Ratio ($r_f=0$)** | $12\overline R/(\mathrm{SD}(R_t)\sqrt{12})$ | Höher |
| **Maximaler Drawdown** | $\min_t\left[V_t/\max_{s\leq t}V_s-1\right]$ | Näher an null |
| **Empirischer monatlicher CVaR** | Negativ ausgewiesenes mittleres Verlustergebnis im schlechtesten Anteil $(1-\alpha)$ der Monatsbeobachtungen, gemäß Implementierung | Näher an null |
| **Endvermögen (NAV)** | Depotwert nach modellierten Handelskosten | Höher |
| **Gesamte Handelsgebühren** | Summe aller modellierten Handelskosten in Euro | Niedriger, bei sonst gleichen Bedingungen |

**Arithmetisch versus geometrisch:** Die Sharpe-Ratio basiert auf dem arithmetischen Durchschnitt der Monatsrenditen. Die annualisierte Nettorendite basiert dagegen auf der geometrischen Vermögensentwicklung. Deshalb ergibt die Division der ausgewiesenen annualisierten Nettorendite durch die Volatilität im Allgemeinen **nicht** die angegebene Sharpe-Ratio.

**CVaR versus Drawdown:** Der CVaR bewertet extreme **einzelne Monatsverluste**. Der maximale Drawdown misst den größten **kumulierten Rückgang vom bisherigen Vermögenshöchststand**. Ein Portfolio kann beim einen Maß gut und beim anderen schlecht abschneiden.

## 10. Installation und Ausführung

### 10.1 Voraussetzungen

Empfohlen wird eine aktuelle Python-3-Umgebung. Die zentralen Bibliotheken lassen sich beispielsweise so installieren:

```bash
pip install numpy pandas scipy scikit-learn xgboost yfinance cvxpy
```

Die Optimierung nutzt **CVXPY** mit **CLARABEL**. Die verfügbaren Solver lassen sich überprüfen:

```python
import cvxpy as cp
print(cp.installed_solvers())
```

### 10.2 Eingabedaten

Die monatlichen adjustierten ETF-Kurse aus Modul 01 werden unter folgendem Pfad erwartet:

```text
data/prices_monthly.csv
```

Die Datei muss chronologisch sortierte Monatsendkurse aller zwölf ETFs enthalten und dem von `load_monthly_prices` in Modul 03 erwarteten Format entsprechen. Der genaue Einstiegspunkt von Modul 01 hängt von der tatsächlich hochgeladenen Dateiversion ab.

### 10.3 Ausführung

```bash
# Vollständiger Walk-Forward-Backtest mit fünf Strategien
python 04_walk_forward_backtest.py

# Kurzer Diagnoselauf: Basis, beide Grenzfälle, ein Gebührenfall
python 05_robustness_analysis.py --quick

# Vollständige Robustheitsanalyse mit 49 Konfigurationen
python 05_robustness_analysis.py
```

Optionale Argumente von Modul 04 und 05 umfassen `--prices`, `--output` und `--start`.

### 10.4 Ergebnisdateien

**Modul 04 – `output/module4/`**

| Datei | Inhalt |
|:---|:---|
| `backtest_metrics.csv` | Rendite- und Risikokennzahlen je Strategie |
| `monthly_net_returns.csv` | Realisierte monatliche Nettorenditen |
| `nav_eur.csv` | Monatliche Depotwertentwicklung |
| `monthly_weights.csv` | Portfolioallokationen im Zeitverlauf |
| `monthly_trades.csv` | Handelsvolumina und Gebühren |

**Modul 05 – `output/module5/`**

| Datei | Inhalt |
|:---|:---|
| `kpi_und_rangmatrix_je_parameter.md` | Gut lesbare KPI- und Rangtabellen für jeden Parameterfall |
| `kpi_je_parameter_und_modell.csv` | Absolute Kennzahlen sämtlicher Verfahren und Fälle |
| `rangmatrix_je_parameter.csv` | Kennzahlenbezogene Rangmatrix |
| `sieger_je_parameter.csv` | Übersicht der KPI-Sieger |
| `sensitivity_metrics.csv` | Ergebnisse aller Sensitivitätsläufe |
| `sensitivity_vs_baseline.csv` | Abweichungen von der Basiskonfiguration |
| `sensitivity_rankings.csv` | Rangfolgen der Parameterfälle |
| `baseline_diagnostics.csv` | Vergleichsdiagnostik gegenüber MV |
| `mathematical_equivalence_audit.csv` | Numerische Prüfung mathematischer Grenzfälle |

Die vollständigen 49 Einzeltabellen bleiben bewusst in der separaten Markdown-Ergebnisdatei. Das README enthält die **entscheidenden Ergebnisse und deren Interpretation**, ohne die GitHub-Startseite mit umfangreichen Terminalausgaben zu überladen.

## 11. Methodische Grenzen

1. **Begrenzter Beobachtungszeitraum:** 45 Monatsrenditen erlauben keine allgemeingültige Aussage über sämtliche Marktregime.
2. **Schätzunsicherheit:** Historische Mittelwerte, Kovarianzen und HAC-Matrizen bleiben fehleranfällig. Auch robuste Unsicherheitsmengen beruhen auf Modellannahmen.
3. **Modellauswahl und Testinterpretation:** Die Wahl des HIST-Inputs und die untersuchten Parameter sind Teil des Forschungsdesigns. Da Modellauswahl und Forschungsentscheidungen die Bewertung beeinflussen können, werden die Ergebnisse nicht als vollständig unabhängiger Nachweis einer allgemein überlegenen Prognosestrategie interpretiert.
4. **Ex-post Sensitivität statt Kalibrierung:** Die 49 Parameterfälle untersuchen Modellreaktionen, nicht die Güte einer vorab festgelegten historischen Parameterauswahl.
5. **Vereinfachte Handelsausführung:** Monatsendkonvention, keine separate Modellierung von Spread, Slippage, Steuern oder Markteinfluss.
6. **Rückwärtsgerichtete Szenarien:** Die 120 historischen STOCH-Szenarien können bislang unbekannte Krisenverläufe nicht zuverlässig repräsentieren; die empirische Tail-Risk-Schätzung basiert auf wenigen Extrembeobachtungen.
7. **Unterschiedliche Anlegerziele:** Höhere Rendite, geringere Volatilität, kleinerer Drawdown und günstigerer CVaR sind nicht gleichbedeutend. Es gibt keinen universellen Sieger für alle Präferenzen.
8. **Technische Reproduzierbarkeit:** Exakte Ergebnisse hängen von Datenstand, Softwareversionen, Solver und Codeversion ab. Für eine Veröffentlichung sollten Kursdatenstand und Python-Umgebung festgehalten werden.

## 12. Fazit

Die Ergebnisse des monatlichen ETF-Backtests von **2023 bis 2026** sprechen weder für eine generelle Überlegenheit besonders komplexer Optimierungsverfahren noch für einen grundsätzlichen Vorrang der einfachen Gleichgewichtung.

- **MV** erzielt in der Basiskonfiguration die höchste annualisierte Nettorendite (**15,97 %**) und Sharpe-Ratio (**1,580**).
- **ELL** bleibt in der Basis nahe bei MV und kann bei moderaten Unsicherheitsradien eine höhere Rendite erzielen; zu starke Absicherung verringert jedoch die Performance.
- **BS** erreicht den geringsten beobachteten maximalen Drawdown (**−5,34 %**), obwohl es nicht die höchste Rendite erzielt.
- **STOCH** weist die geringste Volatilität (**8,42 %**) und den günstigsten empirischen monatlichen CVaR (**−4,23 %**) auf, allerdings auch die niedrigste Rendite und Sharpe-Ratio der Basiskonfiguration.
- **EQ** ist transparent und verursacht die niedrigsten modellierten Handelsgebühren (**44,51 €**), führt aber weder bei Rendite noch bei Sharpe.

Die **49 Sensitivitätskonfigurationen** zeigen zudem, dass sich die Rangfolgen mit Risikoaversion, Unsicherheitsannahmen, Konzentrationsgrenzen und weiteren Modellparametern verändern. Robuste Optimierung kann einzelne Rendite- oder Risikoeigenschaften verbessern, jedoch **nicht bedingungslos und nicht monoton**.

**Antwort auf die Forschungsfrage:** Fortgeschrittene Portfoliooptimierung **kann** unter Schätzunsicherheit und realistischen Restriktionen eine einfache Diversifikation übertreffen. Welche Methode vorzuziehen ist, hängt jedoch vom jeweiligen Optimierungsziel, den Schätzannahmen, den Handelsbedingungen und dem Untersuchungszeitraum ab. Eine wissenschaftlich belastbare Gegenüberstellung muss deshalb **mehrere Kennzahlen**, **Transaktionskosten** und die **ex-post Natur der Sensitivitätsanalyse** transparent berücksichtigen.

---

*Das Projekt beinhaltet auf Basis der unteruschten Forschungsfrage entsprechende Ergebisse. Diese spiegeln keine Anlageberatung und keinen Nachweis real erzielbarer Live-Handelsergebnisse wider.*
