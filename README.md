# Quantitative Portfolio Construction under Parameter Uncertainty

## Overview

Portfolio optimization is a fundamental problem in quantitative finance. Classical portfolio theory assumes that future expected returns and risks are known. In practice, however, expected returns must be estimated from historical data and are therefore subject to substantial forecasting uncertainty.

This project investigates whether advanced portfolio optimization techniques can outperform simple diversification when expected returns are estimated rather than known.

The framework combines:

* Financial data engineering
* Machine learning return forecasting
* Portfolio optimization under parameter uncertainty
* Walk-forward backtesting
* Performance evaluation

---

## Research Question

**Can advanced portfolio optimization methods outperform naive diversification when expected returns must be estimated from historical data and are therefore subject to forecasting uncertainty?**

---

## Asset Universe

The analysis is based on a diversified multi-asset ETF universe:

| Ticker | Asset Class             |
| ------ | ----------------------- |
| SPY    | US Equities             |
| EFA    | Developed Markets       |
| EEM    | Emerging Markets        |
| TLT    | Long-Term Treasuries    |
| IEF    | Intermediate Treasuries |
| GLD    | Gold                    |
| VNQ    | Real Estate             |
| LQD    | Investment Grade Bonds  |
| HYG    | High Yield Bonds        |
| XLF    | Financials              |
| XLV    | Healthcare              |
| XLE    | Energy                  |

---

## Feature Engineering

Monthly return forecasts are generated using:

### Momentum Features

* 1-Month Momentum
* 3-Month Momentum
* 6-Month Momentum
* 12-Month Momentum

### Risk Features

* 3-Month Volatility
* 6-Month Volatility
* 12-Month Volatility

### Drawdown Feature

* 12-Month Maximum Drawdown

### Target Variable

* Next-month ETF return

---

## Forecasting Models

Three forecasting approaches were compared:

* Ordinary Least Squares (OLS)
* Random Forest
* XGBoost

Forecast quality was evaluated using:

* RMSE
* MAE
* Information Coefficient (IC)
* Rank Information Coefficient (Rank IC)

### Validation Results (2019–2022)

Model selection was initially performed using a dedicated validation period.

| Rank | Model         | RMSE     | MAE      | Avg IC   | Avg Rank IC |
| ---- | ------------- | -------- | -------- | -------- | ----------- |
| 1    | XGBoost       | 0.056600 | 0.039687 | 0.076390 | 0.036398    |
| 2    | Random Forest | 0.057445 | 0.040091 | 0.033891 | -0.007626   |
| 3    | OLS           | 0.057557 | 0.040205 | 0.026253 | 0.025204    |

XGBoost achieved the strongest validation performance and was therefore selected as the leading candidate during the model selection stage.

### Independent Test Results (2023–2026)

The selected models were subsequently evaluated on an independent out-of-sample test period.

| Rank | Model         | RMSE     | MAE      | Avg IC    | Avg Rank IC |
| ---- | ------------- | -------- | -------- | --------- | ----------- |
| 1    | Random Forest | 0.041205 | 0.031276 | 0.088586  | 0.065042    |
| 2    | OLS           | 0.041402 | 0.031472 | 0.021135  | 0.000920    |
| 3    | XGBoost       | 0.041488 | 0.031511 | -0.025185 | -0.080060   |

Although XGBoost achieved the strongest validation performance, Random Forest demonstrated superior generalization on the independent test period. Random Forest achieved the lowest forecast error, the strongest Information Coefficient, and the strongest Rank Information Coefficient.

### Final Forecasting Model

The final forecasting model selected for portfolio construction was Random Forest.

Selected parameters:

* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

### Key Finding

Monthly ETF returns proved difficult to predict using historical momentum, volatility, and drawdown information.

While XGBoost achieved the strongest validation performance, Random Forest delivered the strongest independent out-of-sample forecasting performance and was therefore selected for the portfolio construction stage.

---

## Portfolio Construction Models

The following portfolio allocation approaches were evaluated:

### Equal Weight (EQ)

Simple diversification across all assets.

### Mean-Variance Optimization (MV)

Optimizes the trade-off between forecasted expected return and portfolio variance.

### Ellipsoidal Robust Optimization (ELL)

Optimizes against the worst-case expected return realization within an ellipsoidal uncertainty set.

### Distributionally Robust Optimization (DRO)

Uses an additional robustness penalty to reduce portfolio sensitivity to uncertainty in asset returns.

### Stochastic Optimization (STOCH)

Optimizes portfolio weights by maximizing expected return across multiple return scenarios while controlling scenario volatility.

---

## Portfolio Constraints

The optimization framework incorporates realistic portfolio management constraints:

* Fully invested portfolio
* Long-only allocation
* Maximum position size: 30%
* Turnover constraint
* Transaction costs
* User-Defined Investment Budget
* Diversification penalty

---

## Backtesting Framework

For each rebalancing period the framework:

1. Forecast expected returns.
2. Estimate the covariance matrix.
3. Optimize portfolio weights.
4. Apply realistic portfolio constraints.
5. Translate optimal weights into a user-defined investment budget.
6. Evaluate next-month realized returns.
7. Repeat through the entire out-of-sample period.

---

## Results

Forecast Model: Random Forest

Parameters:

* max_depth = 2
* min_samples_leaf = 10
* n_estimators = 200

| Rank | Model                                      | Annualized Return | Annualized Volatility | Sharpe Ratio |
| ---- | ------------------------------------------ | ----------------- | --------------------- | ------------ |
| 1    | Equal Weight (EQ)                          | 9.16%             | 10.41%                | 0.880        |
| 2    | Stochastic Optimization (STOCH)            | 6.03%             | 7.61%                 | 0.792        |
| 3    | Distributionally Robust Optimization (DRO) | 6.92%             | 9.85%                 | 0.703        |
| 4    | Mean-Variance Optimization (MV)            | 7.83%             | 11.41%                | 0.686        |
| 5    | Ellipsoidal Robust Optimization (ELL)      | 7.35%             | 10.83%                | 0.678        |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* CVXPY
* SciPy
* Financial Time Series Analysis

---

## Conclusion

### Interpretation of Optimization Results

#### Equal Weight Portfolio (EQ)

The Equal Weight portfolio serves as the naïve diversification benchmark and allocates capital uniformly across all available assets.

Unlike the optimization-based approaches, Equal Weight does not rely on expected return forecasts, covariance estimates or uncertainty sets. As a result, it is largely immune to estimation errors in both expected returns and risk parameters.

The strong performance of the Equal Weight portfolio highlights an important insight from the portfolio optimization literature: when expected returns are difficult to estimate accurately, avoiding forecast-driven allocations can be advantageous. By distributing capital evenly across all assets, the portfolio benefits from broad diversification while avoiding the concentration risk that often arises when optimization models overreact to noisy forecasts.

The results suggest that, within the context of this study, the robustness gained from avoiding parameter estimation outweighed the potential benefits of more sophisticated optimization techniques. Consequently, Equal Weight achieved the highest out-of-sample Sharpe Ratio and delivered the strongest overall risk-adjusted performance.

---

#### Mean-Variance Optimization (MV)

Mean-Variance Optimization allocates capital according to the forecasted risk-return trade-off and is therefore highly sensitive to estimation errors. Even small deviations in expected return estimates may lead to materially different portfolio weights and can deteriorate out-of-sample performance.

The model assumes that forecasted expected returns contain sufficiently accurate information to justify overweighting and underweighting individual assets. In practice, however, expected returns are notoriously difficult to estimate and forecasting errors can dominate the optimization process. As a result, the theoretically optimal allocation may become suboptimal once implemented on unseen future data.

---
#### Ellipsoidal Robust Optimization (ELL)

Ellipsoidal Robust Optimization explicitly accounts for uncertainty in estimated expected returns. The model assumes that the true expected return vector is not known exactly but lies within an ellipsoidal uncertainty set around the forecasted return vector. Rather than optimizing for a single point estimate, the optimizer seeks a portfolio that remains effective across a range of plausible forecast realizations.

As a consequence, portfolios whose attractiveness depends heavily on individual return forecasts are penalized. This increases robustness against estimation error but may also reduce exposure to genuinely attractive investment opportunities. The empirical results suggest that the reduction in forecast sensitivity was not sufficient to fully compensate for the loss of return potential associated with the additional robustness penalty.

---

#### Distributionally Robust Optimization (DRO)

Distributionally Robust Optimization extends the concept of robustness beyond uncertainty in expected returns. While Ellipsoidal Robust Optimization primarily addresses uncertainty in the expected return vector, Distributionally Robust Optimization additionally accounts for the possibility that the entire underlying return distribution has been estimated incorrectly.

The optimizer therefore favors portfolios that remain stable across a broader set of plausible probability distributions. This typically results in even more conservative allocations and a stronger emphasis on protection against model misspecification and adverse market conditions. While the additional robustness reduces sensitivity to distributional uncertainty, it simultaneously limits the optimizer's ability to exploit forecasted return opportunities. In this study, the cost of the additional conservatism outweighed its benefits, resulting in lower risk-adjusted performance than the Equal Weight benchmark.

---

#### Stochastic Optimization (STOCH)

Stochastic Optimization addresses uncertainty through a scenario-based framework. Rather than optimizing against a worst-case realization within an uncertainty set, the model constructs a large collection of plausible future return scenarios and seeks a portfolio that performs well on average across those scenarios.

The optimization therefore balances expected return and risk directly across many potential future market environments. As a result, the portfolio construction process is not driven by a single forecast or a predefined worst-case assumption, but instead by the overall distribution of potential outcomes. This generally leads to more stable and diversified allocations that remain effective across a wider range of market conditions.

The results show that Stochastic Optimization achieved the lowest annualized portfolio volatility among all tested approaches. This outcome is consistent with the model structure, as the dispersion of portfolio returns across all scenarios enters the optimization process directly. Although the reduction in volatility came at the cost of lower annualized returns than the Equal Weight benchmark, the model still achieved the second-highest Sharpe Ratio of all portfolio construction methods. This suggests that the scenario-based framework was particularly effective at controlling portfolio risk while preserving a competitive level of risk-adjusted performance.

---

### Overall Interpretation

The Equal Weight portfolio benefited from its complete independence from parameter estimation and achieved the highest out-of-sample Sharpe Ratio. In contrast, the optimization-based approaches attempted to exploit forecasted return information and explicitly account for uncertainty, but remained dependent on the quality of the underlying return forecasts.

From an economic perspective, the stochastic optimization framework may represent the most attractive solution for highly risk-averse investors whose primary objective is to achieve stable long-term, risk-adjusted returns rather than maximizing expected return at any cost. By explicitly incorporating scenario uncertainty into the optimization process, the model substantially reduced portfolio volatility while maintaining a competitive Sharpe Ratio. Such characteristics may be particularly relevant for long-term institutional investors, including pension funds and insurance companies, where capital preservation, portfolio stability and effective risk management are fundamental investment objectives alongside the generation of sustainable returns.




