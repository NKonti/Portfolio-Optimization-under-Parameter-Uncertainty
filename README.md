Quantitative Portfolio Construction under Parameter Uncertainty
A reproducible research framework for comparing naive diversification, classical mean–variance optimization, ellipsoidal and budgeted robust optimization, and scenario-based CVaR optimization.
1. Overview and research question
Portfolio optimization is sensitive to estimation error. Expected returns, covariance matrices, and uncertainty sets are inferred from finite historical samples rather than observed as known future quantities. A more sophisticated mathematical model is therefore not necessarily a better investment strategy.
Research question: Do advanced portfolio construction methods deliver superior realized risk-adjusted performance to simple diversification when expected returns are uncertain and realistic trading constraints apply?

The current implementation combines monthly ETF data engineering, a comparison of return-forecasting methods, five constrained allocation rules, a self-financing walk-forward backtest, and a 49-configuration, one-factor-at-a-time sensitivity analysis. It evaluates performance after modeled transaction costs, not only in an idealized frictionless portfolio.

Scope: empirical and exploratory. The parameter sensitivity exercise is not nested historical hyperparameter calibration, and the highest-performing ex-post configuration must not be presented as a separately validated investment rule.
2. Asset universe
ETF	Exposure	ETF	Exposure
SPY	US large-cap equities	EFA	Developed-market equities outside the US
EEM	Emerging-market equities	TLT	Long-duration US Treasuries
IEF	Intermediate-duration US Treasuries	GLD	Gold
VNQ	US listed real estate	LQD	US investment-grade corporate bonds
HYG	US high-yield corporate bonds	XLF	US financial-sector equities
XLV	US healthcare-sector equities	XLE	US energy-sector equities


The universe contains 12 exchange-traded funds spanning equities, government bonds, credit, real estate, and gold. Sector ETFs overlap economically with the broad equity exposures; the 12 holdings should not be interpreted as 12 independent asset classes.
3. Research architecture
Module	File	Responsibility
01	01_data_engineering.py	Download/prepare adjusted prices, build monthly observations, validate data coverage and temporal integrity
02	02_return_forecasting.py	Compare historical-mean and statistical/ML return forecasts; document forecasting difficulty
03	03_portfolio_optimization.py	Estimate optimization inputs and solve EQ, MV, ELL, BS and STOCH under common constraints
04	04_walk_forward_backtest.py	Evaluate all five strategies on the same monthly timeline using separate self-financing accounts
05	05_robustness_analysis.py	Run 49 sensitivity configurations, KPI/rank matrices, baseline diagnostics and mathematical limiting-case checks


Modules 03–05 are implemented in the corresponding Python files in this project. The names of modules 01 and 02 reflect the intended numbered repository layout; ensure that your uploaded versions match these names and the documented data interface.
Reproducible workflow
Adjusted daily ETF prices
          |
          v
01 Data engineering -> monthly adjusted prices -> data/prices_monthly.csv
          |
          v
02 Forecast comparison (research/diagnostics)
          |
          v
03 Portfolio optimization: HIST mean, rolling covariance, HAC uncertainty
          |
          v
04 Monthly walk-forward backtest -> output/module4/
          |
          v
05 Robustness and sensitivity -> output/module5/
Important distinction: Although forecasting models are compared in the broader research workflow, the current portfolio optimization backtest uses an expanding historical arithmetic mean (HIST) as its expected-return input. It does not use Random Forest or XGBoost predictions to construct the five portfolios. This is a deliberate distinction from earlier versions of the project.
4. Data, forecasting and temporal design
4.1 Data preparation
The analysis uses adjusted ETF prices sourced from Yahoo Finance and aggregated to month-end observations. Adjusted prices account for historical price adjustments as provided by the source. The implementation should validate missing observations, chronological ordering, and consistent ETF coverage before portfolio optimization.

The observed backtest covers January 2023 through September 2026 (45 monthly returns). Each allocation uses information available through the preceding decision month; the return of the next month is used only to evaluate that allocation. This avoids direct future-return leakage within the backtest decision loop.
4.2 Forecasting research
The project's forecasting work examines next-month returns using momentum (1, 3, 6 and 12 months), realized volatility (3, 6 and 12 months), and 12-month drawdown features, with methods including OLS, Random Forest and XGBoost. These predictors can be informative diagnostics but are subject to substantial noise and unstable relationships.

Final allocation input: the historical mean, not a machine-learning forecast:
\[
\hat{\boldsymbol\mu}_t=\frac{1}{T_t}\sum_{s=1}^{T_t}\mathbf r_s.
\]
Here $\mathbf r_s\in\mathbb R^{12}$ denotes the vector of monthly ETF returns known by decision time $t$, and $T_t$ is the number of available historical months. This expanding-window estimate is easy to audit, but it remains subject to sampling uncertainty and regime changes.

The comparison of forecasting methods informed the research design. Accordingly, 2023–2026 should not be described as an entirely untouched, independent model-selection test.
4.3 Risk and estimation-uncertainty inputs
The monthly return covariance matrix $\hat\Sigma_t$ is estimated from a 36-month rolling window (baseline). The uncertainty of the historical mean is approximated using a Bartlett-kernel heteroskedasticity-and-autocorrelation-consistent (HAC) long-run covariance estimate:
\[
\hat S_t=\hat\Gamma_0+\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)
\left(\hat\Gamma_k+\hat\Gamma_k^\top\right),
\qquad
\hat\Omega_t=\frac{\hat S_t}{T_t},
\]
with baseline $L=3$ lags. $\hat\Omega_t$ describes estimated uncertainty in the sample mean, whereas $\hat\Sigma_t$ describes the variability of monthly asset returns. They are different objects and enter the optimization problems differently. Positive-semidefinite numerical corrections are applied where required.
5. Five portfolio construction approaches
Let $\mathbf w\in\mathbb R^{12}$ be target portfolio weights, $\hat\mu$ the estimated monthly mean return vector, $\hat\Sigma$ the estimated monthly covariance matrix, and $\mathcal W$ the common feasible set defined below.
5.1 Equal Weight (EQ): diversification benchmark
\[
\min_{\mathbf w\in\mathcal W}\left\|\mathbf w-\frac{1}{12}\mathbf 1\right\|_2^2.
\]
EQ aims for equal allocations while respecting the common constraints, including the turnover restriction. It does not use estimated returns or covariances in its allocation objective. Because the turnover restriction may bind, its realized weights need not be exactly $1/12$ every month.
5.2 Mean–Variance (MV): classical return–risk optimization
\[
\max_{\mathbf w\in\mathcal W}
\left\{\hat\mu^\top\mathbf w-\lambda\,\mathbf w^\top\hat\Sigma\mathbf w\right\}.
\]
$\lambda\geq0$ controls the penalty on variance. MV exploits estimated differences in expected returns but can react strongly to estimation noise. The baseline uses $\lambda=3$.
5.3 Ellipsoidal Robust Optimization (ELL)
The true expected-return vector is assumed to lie in an ellipsoidal uncertainty set centered on $\hat\mu$:
\[
\mathcal U_{\rm ELL}=\left\{\hat\mu+\rho\hat\Omega^{1/2}\mathbf u:\|\mathbf u\|_2\leq1\right\}.
\]
Its robust counterpart is
\[
\max_{\mathbf w\in\mathcal W}
\left\{\hat\mu^\top\mathbf w-\lambda\mathbf w^\top\hat\Sigma\mathbf w
-\rho\sqrt{\mathbf w^\top\hat\Omega\mathbf w}\right\}.
\]
The radius $\rho$ controls protection against expected-return estimation error; larger values imply a stronger worst-case penalty. At $\rho=0$, ELL reduces mathematically to MV with otherwise identical inputs and constraints.
5.4 Bertsimas–Sim Budgeted Robust Optimization (BS)
BS allows adverse expected-return deviations within a budgeted uncertainty set. Let

\(d_i=\kappa\sqrt{\hat\Omega_{ii}},\)

where $\kappa$ scales asset-specific uncertainty magnitudes and $\Gamma\in[0,12]$ controls the total uncertainty budget. The implemented long-only linear-program dual formulation is
\[
\begin{aligned}
\max_{\mathbf w\in\mathcal W,\,p\geq0,\,\mathbf q\geq0}\quad&
\hat\mu^\top\mathbf w-\lambda\mathbf w^\top\hat\Sigma\mathbf w
-\Gamma p-\sum_{i=1}^{12}q_i\\
\text{subject to}\quad&p+q_i\geq d_iw_i,\quad i=1,\ldots,12.
\end{aligned}
\]
Unlike ellipsoidal robustness, the uncertainty budget limits the aggregate adversarial deterioration. The baseline uses $\Gamma=3$ and $\kappa=1$. At $\Gamma=0$, BS reduces mathematically to MV. The implementation routes these exact limiting cases through the same MV objective to avoid avoidable solver-formulation discrepancies.
5.5 Stochastic CVaR Optimization (STOCH)
STOCH uses the most recent 120 joint historical monthly return scenarios $\mathbf r_s$ and penalizes the Conditional Value at Risk (CVaR) of portfolio losses:
\[
\max_{\mathbf w\in\mathcal W,\,\eta,\,\mathbf u\geq0}
\left\{\hat\mu^\top\mathbf w-\gamma_{\rm CVaR}\left(\eta+
\frac{1}{(1-\alpha)S}\sum_{s=1}^{S}u_s\right)\right\},
\]
subject to

\(u_s\geq-\mathbf r_s^\top\mathbf w-\eta,\qquad s=1,\ldots,S.\)

Here $\alpha=0.90$ and $\gamma_{\rm CVaR}=1$ in the baseline. $\eta$ is the loss threshold and $u_s$ represents loss exceedances. This formulation optimizes expected return against tail losses, not scenario variance. The realized CVaR reported below is an empirical monthly tail-loss statistic; it is not the same as maximum drawdown.
6. Shared constraints and realistic trading assumptions
The optimization problems use the same feasible allocation set:
\[
\mathcal W=\left\{\mathbf w:
\sum_{i=1}^{12}w_i=1,\quad
0\leq w_i\leq0.30,\quad
\|\mathbf w-\mathbf w_t^-\|_1\leq0.50
\right\}.
\]
The turnover constraint applies after the initial investment, where $\mathbf w_t^-$ denotes weights immediately before rebalancing. The $L^1$ limit is the sum of absolute weight changes, not a 50% one-way turnover definition. The baseline assumptions are:
Item	Baseline	Interpretation
Initial capital	€10,000	Identical starting NAV for all five portfolios
Rebalancing	Monthly	New targets based on previously available observations
Long-only / budget	$w_i\geq0$, $\sum_iw_i=1$	No short positions; fully invested after costs
Maximum ETF weight	30%	Concentration cap per instrument
$L^1$ turnover cap	0.50	Limits monthly changes in target weights
Trading cost	0.20%	Applied to absolute gross traded euro volume
Covariance history	36 months	Rolling monthly covariance
STOCH scenarios	120 months	Joint historical monthly scenarios
Initial allocation	From cash	Initial purchases also incur modeled fees
Solver	CLARABEL / CVXPY	Convex optimization


Self-financing trading. Fees are deducted from the portfolio itself rather than funded by an external cash injection. Given pre-trade holdings $h_i^-$ and total value $V^-$, the post-trade NAV $V^+$ solves
\[
V^+=V^- - c\sum_i\left|V^+w_i-h_i^-\right|,
\qquad h_i^+=V^+w_i,
\]
where $c$ is the proportional fee. This fixed-point equation is solved numerically by bisection. It avoids overstating performance by silently ignoring the effect of costs on investable capital.

Execution limitation: the strategy uses an idealized monthly-close information and execution convention. Allocations computed using a month-end close cannot literally be executed at that already-observed closing price. Thus the study is a model-based monthly return backtest, not a fully executable intraday or next-open trading simulation. Bid–ask spreads, slippage, taxes, market impact and changing market liquidity are not separately modeled.
7. Walk-forward backtest: January 2023–September 2026
All strategies start from €10,000, face identical market observations and common trading constraints, and are evaluated over 45 monthly return periods. The baseline parameters are not retroactively selected from the sensitivity winners.
7.1 Main performance results
Return rank	Model	Annualized net return	Annualized volatility	Sharpe ($r_f=0$)	Maximum drawdown	Monthly loss CVaR	Final NAV	Total fees
1	MV	15.97%	9.73%	1.580	−7.18%	−5.51%	€17,431.75	€109.07
2	ELL	15.69%	9.80%	1.545	−7.44%	−5.52%	€17,274.05	€86.43
3	BS	14.70%	9.19%	1.545	−5.34%	−5.59%	€16,727.40	€94.95
4	EQ	11.25%	9.46%	1.178	−7.64%	−6.86%	€14,913.49	€44.51
5	STOCH	9.52%	8.42%	1.125	−8.14%	−4.23%	€14,062.40	€81.36


Values are rounded for readability. Returns and NAV are net of modeled trading fees. A less negative drawdown/CVaR is preferable. Sharpe uses a zero risk-free-rate comparison convention, not a claim that historical risk-free rates were zero.
7.2 Baseline KPI ranking matrix
Rank 1 is best for each column. For volatility and fees, smaller values are better; for drawdown and signed CVaR, the value closer to zero is better.
Model	Net return	Volatility	Sharpe	Max. drawdown	CVaR	Final NAV	Fees
MV	1	4	1	2	2	1	5
ELL	2	5	3	3	3	2	3
BS	3	2	2	1	4	3	4
EQ	4	3	4	4	5	4	1
STOCH	5	1	5	5	1	5	2


The KPI columns should not be added into an arbitrary overall score. For example, final NAV and annualized return measure closely related outcomes, while Sharpe combines return and volatility. Equal weighting would implicitly double-count some characteristics and impose unmotivated investor preferences.
7.3 Interpretation by method
MV — strongest realized return and Sharpe. Classical mean–variance optimization outperformed the naive benchmark in this particular evaluation period. This demonstrates that a comparatively simple estimated-mean/covariance optimizer can perform well under realistic constraints. It does not establish that its estimated expected returns are generally accurate or that MV will dominate in future regimes.

ELL — competitive, parameter-sensitive robustness. ELL remained close to MV at baseline, with somewhat lower realized trading fees. A modest ellipsoidal penalty can improve outcomes in some sensitivity cases, but increasing the uncertainty radius eventually reduces realized return. Robustness is not free; protection against forecast errors may also suppress rewarded exposures.

BS — drawdown protection in the baseline. BS achieved the smallest maximum observed drawdown (−5.34%), despite lower annualized return than MV and ELL. Budgeted uncertainty can change allocations in a manner favorable to pathwise loss control, but minimizing drawdown is not BS's explicit objective and the result is sample-specific.

STOCH — lowest volatility and best empirical monthly CVaR. STOCH had the lowest realized volatility (8.42%) and the least adverse average monthly tail-loss metric (−4.23%), at the cost of the lowest annualized return and Sharpe. This illustrates the difference between period-by-period tail-loss control and cumulative path risk: STOCH did not have the smallest maximum drawdown.

EQ — simple and low-cost, but not the return winner. The benchmark incurred the lowest fees (€44.51) and avoids estimated-return inputs in its objective. In this study it did not beat the optimized methods on return or Sharpe, challenging any blanket claim that equal weighting always dominates under estimation uncertainty.
8. Robustness and sensitivity: 49 configurations
Module 05 performs a one-factor-at-a-time (OFAT) analysis: each case modifies one assumption relative to the baseline while retaining the common backtest design. The 49 cases include the baseline and variations in:
Parameter family	Examined settings
Trading fee	0%, 0.10%, 0.50%, 1.00% (baseline 0.20%)
Maximum ETF weight	20%, 40%, 60%, 100% (baseline 30%)
$L^1$ turnover limit	0.25, 0.75, 1.00 (baseline 0.50)
Covariance window	24, 60 months (baseline 36)
MV/ELL/BS risk aversion $\lambda$	0.5, 1, 2, 5, 8, 12 (baseline 3)
ELL uncertainty radius $\rho$	0, 0.25, 0.5, 1.5, 2, 3 (baseline 1)
BS uncertainty budget $\Gamma$	0, 1, 2, 4, 6, 9, 12 (baseline 3)
BS uncertainty scale $\kappa$	0.5, 1.5, 2, 3 (baseline 1)
STOCH CVaR penalty $\gamma$	0.25, 0.5, 1.5, 2, 3 (baseline 1)
STOCH confidence level $\alpha$	95% (baseline 90%)
STOCH scenario window	60, 84 months (baseline 120)
HAC lags	0, 1, 6, 12 (baseline 3)


Each configuration produces (A) an absolute KPI matrix and (B) a per-KPI rank matrix, with separate winners for annualized return, Sharpe, volatility, maximum drawdown and empirical monthly CVaR. Economically indistinguishable values may be labeled as ties under the reporting tolerance. A ranking is descriptive, not a hypothesis test.
8.1 Central findings
1. MV frequently leads on return and Sharpe. Across the explored parameter grid, it remains a strong baseline, particularly when trading fees or turnover caps change. This does not imply a probability of winning in new markets: the 49 configurations share the same return history and are highly dependent.
2. ELL can overtake MV with moderate uncertainty penalties. At $\rho=0.25$ and $0.5$, ELL is the realized return and Sharpe winner; for $\rho\geq1.5$ in the tested grid, MV leads on both. This illustrates a non-monotonic realized benefit of robustness.
3. Risk aversion matters materially. With $\lambda=0.5,1,2$, ELL leads realized return; with $\lambda=5$, MV leads; with $\lambda=8,12$, EQ has the highest return. Increasing risk aversion changes the optimized strategies, while EQ and the separately parameterized STOCH objective are not changed by $\lambda$.
4. Position limits affect model comparisons. MV leads return at maximum ETF weights of 20% and 40%; ELL leads at 60% and 100%. A larger feasible set can improve the optimization objective in sample while worsening the realized future performance.
5. BS often has the best maximum drawdown. This finding concerns realized portfolio paths, not a mathematical guarantee of the BS uncertainty set.
6. STOCH leads the empirical monthly CVaR comparison in every reported parameter configuration. This is consistent with its tail-loss objective but does not guarantee superior maximum drawdown, Sharpe or return.
7. Costs reduce realized returns without necessarily changing the winner. The MV strategy remains the return leader across the reported 0%–1% fee cases; transaction costs are therefore essential for reporting economically meaningful NAV outcomes.
8.2 Illustrative parameter results
ELL uncertainty radius:
$\rho$	ELL annualized net return	Return winner
0	15.97%	ELL = MV (mathematical limit)
0.25	16.15%	ELL
0.50	16.18%	ELL
1.00 (baseline)	15.69%	MV
1.50	14.95%	MV
2.00	13.92%	MV
3.00	12.27%	MV


Risk aversion:
$\lambda$	Return winner	Winner's annualized net return
0.5	ELL	14.89%
1	ELL	16.19%
2	ELL	16.44%
3 (baseline)	MV	15.97%
5	MV	14.03%
8	EQ	11.25%
12	EQ	11.25%


Maximum ETF position:
Position cap	Return winner	Winner's annualized net return
20%	MV	16.25%
30% (baseline)	MV	15.97%
40%	MV	14.58%
60%	ELL	12.51%
100%	ELL	12.29%


8.3 Mathematical limiting cases and numerical ties
The following are mathematical identities, not empirical performance claims:

\(\mathrm{ELL}(\rho=0)=\mathrm{MV},\qquad \mathrm{BS}(\Gamma=0)=\mathrm{MV}.\)

With identical inputs, objective and constraints, the limiting problems are the same. Module 03 explicitly uses the MV formulation for these cases; Module 05 includes numerical equivalence diagnostics and tie-aware ranking. The reporting of ELL = MV or BS = MV in the sensitivity winner table reflects this equivalence and numerical-tie handling. Do not infer that a numerical audit returned PASS solely from the winner labels; consult mathematical_equivalence_audit.csv for the actual test results.
8.4 What the sensitivity analysis does not establish
- No nested parameter tuning: selecting $\rho=0.5$ or $\lambda=2$ because it performed best in 2023–2026 would reuse the evaluation period for selection. These settings are illustrative ex-post outcomes, not independently validated recommendations.
- No independent 49-fold evidence: all cases reuse the same market months, often with nearly identical portfolios.
- No statistical significance claim: 45 monthly observations are a limited sample for differences in Sharpe, tail risk and drawdown; differences should not be interpreted as significant without a suitable uncertainty analysis.
- No universal dominance: results are conditional on the chosen universe, return history, constraints, fee model and market regime.
- No direct cross-model equivalence of risk penalties: $\lambda$, $\rho$, $\Gamma$, $\kappa$ and $\gamma_{\rm CVaR}$ operate on different quantities and do not represent the same level of investor risk aversion.
9. Performance measures
Let $R_t$ denote realized monthly net portfolio return, $V_t$ portfolio NAV, and $T$ the number of test months.
KPI	Definition	Preferred direction
Annualized net return	$(V_T/V_0)^{12/T}-1$	Higher
Annualized volatility	$\operatorname{sd}(R_t)\sqrt{12}$	Lower, all else equal
Sharpe ratio, $r_f=0$	$12\overline R/(\operatorname{sd}(R_t)\sqrt{12})$	Higher
Maximum drawdown	$\min_t[V_t/\max_{s\leq t}V_s-1]$	Closer to zero
Empirical monthly CVaR	Negative of the estimated mean loss in the worst $(1-\alpha)$ fraction of monthly outcomes, with the implemented empirical convention	Closer to zero
Final NAV	Portfolio wealth after trading costs	Higher
Total trading fees	Sum of modeled transaction costs in euros	Lower, all else equal


Arithmetic vs. geometric returns: Sharpe uses the arithmetic mean of monthly returns; annualized net return uses compounded NAV growth. Therefore dividing the displayed geometric annualized return by annualized volatility will generally not reproduce the reported Sharpe ratio.

Tail risk vs. drawdown: CVaR is based on individual monthly loss outcomes. Maximum drawdown is path-dependent and measures the worst cumulative peak-to-trough NAV decline. A strategy may excel at one and not the other.
10. Running the project
10.1 Requirements
Use a recent Python 3 environment with packages required by the relevant modules, notably:
pip install numpy pandas scipy scikit-learn xgboost yfinance cvxpy
The selected convex solver is CLARABEL, normally available through a compatible CVXPY installation. Check the installed solvers if necessary:
import cvxpy as cp
print(cp.installed_solvers())
10.2 Input file
Place the monthly adjusted price CSV from Module 01 at:
data/prices_monthly.csv
The file must contain chronologically ordered month-end prices for all 12 ETFs, in the format accepted by Module 03's load_monthly_prices function. The exact data-engineering entry point depends on the Module 01 version included in the repository.
10.3 Execute
# Five-strategy walk-forward backtest
python 04_walk_forward_backtest.py

# Short diagnostic run: baseline, both mathematical limits, one fee case
python 05_robustness_analysis.py --quick

# Full 49-configuration robustness analysis
python 05_robustness_analysis.py
Optional arguments for Modules 04 and 05 include --prices, --output and --start.
10.4 Generated outputs
Module 04 (output/module4/):
- backtest_metrics.csv — strategy-level net performance and risk metrics
- monthly_net_returns.csv — realized monthly net returns
- nav_eur.csv — portfolio wealth paths
- monthly_weights.csv — portfolio allocations over time
- monthly_trades.csv — trading/fee information
Module 05 (output/module5/):
- kpi_und_rangmatrix_je_parameter.md — readable, per-case KPI and ranking tables
- kpi_je_parameter_und_modell.csv — absolute KPI values for all models and cases
- rangmatrix_je_parameter.csv — rank matrix for each case
- sieger_je_parameter.csv — per-KPI winner overview
- sensitivity_metrics.csv — sensitivity case performance metrics
- sensitivity_vs_baseline.csv — differences from baseline
- sensitivity_rankings.csv — scenario rankings
- baseline_diagnostics.csv — pairwise diagnostic comparison with MV
- mathematical_equivalence_audit.csv — numerical limiting-case checks
The complete parameter-by-parameter Markdown report is preferable to copying 49 lengthy terminal tables into the README. The README highlights the most economically relevant results and links the detailed outputs by their repository-relative paths.
11. Limitations and reproducibility notes
1. Sample length and regime dependence. The main evaluation covers 45 monthly returns. The rankings may change in other regimes and are sensitive to the selected ETFs.
2. Expected-return uncertainty. HIST means and historical covariance/HAC estimates remain noisy. Robust uncertainty sets depend on assumptions and do not eliminate estimation error.
3. Research selection effects. The broader forecasting comparison influenced the final design; the period should not be overstated as completely independent of research choices.
4. Exploratory sensitivities. The parameter grid is not a nested training/validation/test exercise. It is a descriptive stress test of assumptions.
5. Simplified execution. Monthly-close information/execution timing is idealized. Real-world implementation requires realistic executable timestamps, spreads, slippage and potentially taxes.
6. Historical scenarios. STOCH's 120-month scenario set is backward-looking and may omit unprecedented market events. Empirical CVaR is itself estimated with limited tail observations.
7. No universal risk preference. Lower volatility, smaller drawdown, higher return and lower CVaR are different objectives; the project does not claim a single best strategy for all investors.
8. Data and software reproducibility. Exact outputs depend on the adjusted-price snapshot, package versions, solver behavior and the specific code revision. Preserve data-source dates and a frozen environment when publishing definitive numerical results.
12. Conclusions
The evidence from this particular 2023–2026 monthly ETF backtest does not support the blanket assertion that either the most sophisticated optimizer or the simplest diversification rule must dominate.
- MV achieved the strongest baseline annualized net return (15.97%) and Sharpe ratio (1.580).
- ELL was close at baseline and sometimes improved return and Sharpe under moderate uncertainty penalties; the benefit diminished as the robustness radius increased.
- BS delivered the smallest observed baseline maximum drawdown (−5.34%), showing a different empirical risk profile from the return leader.
- STOCH delivered the lowest volatility (8.42%) and the least adverse monthly empirical CVaR (−4.23%), but the lowest return and Sharpe in the baseline comparison.
- EQ remained a transparent, inexpensive benchmark, with the lowest modeled fees (€44.51), but did not win the baseline return or Sharpe comparison.
The 49 sensitivity configurations show that model rankings depend on risk aversion, robustness assumptions, concentration limits and other implementation choices. Robust optimization can improve particular risk or return outcomes, but the effect is neither unconditional nor monotonic.

Overall answer: sophisticated portfolio construction methods can outperform naive diversification in realized performance, but the benefits depend on the optimization objective, estimation assumptions, trading constraints and the chosen evaluation period. A defensible comparison therefore requires multiple complementary KPIs, explicit transaction-cost modeling and transparent disclosure of ex-post parameter sensitivity.
Research-oriented implementation. Not investment advice or a claim of live-trading profitability.
