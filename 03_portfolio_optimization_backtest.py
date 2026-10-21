"""Portfoliooptimierung unter Parameterschätzunsicherheit.

Dieses Modul bündelt fünf Verfahren auf derselben zulässigen Menge:
EQ, MV, ELL, BS und STOCH. Sämtliche Renditen und Risikomaße sind
MONATLICH und als Dezimalzahlen (z. B. 0.01 für 1 %) zu übergeben.

Zeitkonvention: ``history`` enthält ausschließlich Monatsrenditen, die am
Entscheidungszeitpunkt bereits vollständig bekannt sind. Die Schätzfunktionen
verwenden keine späteren Beobachtungen.

Die Optimierung liefert Zielgewichte, keine Handelsaufträge. Die
selbstfinanzierende Handelsausführung gehört in den späteren Backtest. Die Gewichtsdifferenz
(L1-Turnover) ist nicht identisch mit dem tatsächlichen Euro-Handelsvolumen.

Benötigt: numpy, cvxpy und CLARABEL (``pip install numpy cvxpy clarabel``).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None


METHODS = ("EQ", "MV", "ELL", "BS", "STOCH")


@dataclass(frozen=True)
class OptimizationConfig:
    """Einheitliche Handelsgrenzen und modellabhängige Risikoparameter.

    ``lambda_risk`` gewichtet die Varianz in MV/ELL/BS, ``rho`` die
    ellipsoidale Mittelwertunsicherheit, ``gamma_budget`` das BS-Budget,
    ``kappa`` die ETF-spezifischen Unsicherheitsradien und ``gamma_cvar``
    den CVaR in STOCH. Diese Parameter sind NICHT dimensionsgleich und
    müssen mittels historischer Kalibrierung festgelegt werden.
    """
    max_weight: float = 0.30
    turnover_limit: float = 0.50
    covariance_window: int = 36
    scenario_window: int = 120
    hac_lags: int = 3
    lambda_risk: float = 3.0
    rho: float = 1.0
    gamma_budget: float = 3.0
    kappa: float = 1.0
    gamma_cvar: float = 1.0
    alpha: float = 0.90
    eigen_floor: float = 0.0
    solver: str = "CLARABEL"


@dataclass(frozen=True)
class OptimizationResult:
    """Gewichte und Diagnose einer einzelnen Optimierung."""
    method: str
    weights: np.ndarray
    objective_value: float
    status: str
    turnover_l1: float | None


def _matrix(data, name: str) -> np.ndarray:
    """Validiert eine Matrix aus historischen Monatsrenditen (T x n)."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[1] < 2 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} muss eine endliche Matrix T x n mit n>=2 sein")
    return x


def _vector(data, n: int, name: str) -> np.ndarray:
    """Validiert einen endlichen Vektor mit genau n Elementen."""
    x = np.asarray(data, dtype=float)
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} muss ein endlicher Vektor der Länge {n} sein")
    return x


def _psd(a: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Projiziert symmetrisch auf den PSD-Kegel.

    Zerlegung A=Q diag(d) Q^T; negative Eigenwerte werden auf ``floor``
    gesetzt. Eine größere Korrektur sollte extern protokolliert werden;
    die Funktion ist eine numerische Maßnahme, keine statistische Heilung.
    """
    if floor < 0:
        raise ValueError("eigen_floor muss nichtnegativ sein")
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.all(np.isfinite(a)):
        raise ValueError("Erwartet wird eine endliche quadratische Matrix")
    eig, q = np.linalg.eigh((a + a.T) / 2)
    return (q * np.maximum(eig, floor)) @ q.T


def estimate_inputs(history: np.ndarray, cfg: OptimizationConfig):
    r"""Schätzt HIST-Mittelwert, rollierende Kovarianz und HAC-Unsicherheit.

    Für T beobachtete gemeinsame Monatsrenditen r_s gilt:

    .. math:: \hat\mu = T^{-1}\sum_{s=1}^T r_s.

    Die Renditekovarianz Sigma wird aus den letzten ``covariance_window``
    Monatsrenditen mit Stichproben-Nenner (m-1) geschätzt.

    Für die Mittelwertschätzunsicherheit verwenden wir eine Bartlett-HAC-
    Schätzung auf der GESAMTEN HIST-Stichprobe:

    .. math:: S = \Gamma_0 + \sum_{k=1}^{L}(1-k/(L+1))
              (\Gamma_k+\Gamma_k^T), \quad \Omega=S/T.

    Dabei ist Gamma_k = T^{-1} sum_{s=k+1}^T
    (r_s-mu)(r_{s-k}-mu)^T. HAC setzt für seine Interpretation u. a.
    hinreichende Stationarität und geeignete Abhängigkeitsstruktur voraus.
    Ein historisches Fenster kann Regimewechsel nicht eliminieren.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        mu (n,), Sigma (n,n), Omega (n,n), jeweils monatlich skaliert.
    """
    r = _matrix(history, "history")
    t, _ = r.shape
    if cfg.covariance_window < 2 or t < cfg.covariance_window:
        raise ValueError("Zu wenig Historie für das Kovarianzfenster")
    if cfg.hac_lags < 0 or cfg.hac_lags >= t:
        raise ValueError("Ungültige HAC-Lag-Länge")
    mu = r.mean(axis=0)
    sigma = _psd(np.cov(r[-cfg.covariance_window:], rowvar=False), cfg.eigen_floor)
    z = r - mu
    s = z.T @ z / t
    for k in range(1, cfg.hac_lags + 1):
        g = z[k:].T @ z[:-k] / t
        s += (1 - k / (cfg.hac_lags + 1)) * (g + g.T)
    omega = _psd(s / t, cfg.eigen_floor)
    return mu, sigma, omega


def optimize_portfolio(
    method: Literal["EQ", "MV", "ELL", "BS", "STOCH"],
    history: np.ndarray,
    pretrade_weights: np.ndarray | None = None,
    cfg: OptimizationConfig | None = None,
) -> OptimizationResult:
    r"""Löst eine der fünf konvexen Portfoliooptimierungen.

    Gemeinsame Menge W:
        sum_i w_i=1; 0<=w_i<=max_weight;
        ||w-w_pre||_1<=turnover_limit (entfällt bei Erstinvestition).

    EQ: min ||w-1/n||_2^2.
    MV: max mu^T w - lambda*w^T Sigma w.
    ELL: max mu^T w - lambda*w^T Sigma w
             - rho*sqrt(w^T Omega w).
        Dies entspricht min_{mu' in U} mu'^T w mit
        U={mu+rho*Omega^(1/2)*u : ||u||_2<=1}.
    BS: max mu^T w - lambda*w^T Sigma w - Gamma*p - sum(q_i),
        p+q_i >= kappa*sqrt(Omega_ii)*w_i; p,q_i>=0.
        Diese LP-Dualform ist für long-only w äquivalent zur budgetierten
        adversarialen Absenkung der Mittelwertparameter.
    STOCH: max mu^T w - gamma_cvar*(eta + sum(u_s)/((1-alpha)*S)),
        u_s >= -r_s^T w-eta; u_s>=0; Szenarien sind die letzten S
        gemeinsamen historischen Monatsrenditevektoren.

    Hinweis: Die voreingestellten Risiko- und Robustheitsparameter sind
    nur demonstrative Startwerte, KEINE validierte Kalibrierung. Die
    Turnover-Grenze wird ausschließlich in Gewichten angewendet.

    Parameters
    ----------
    method : str
        Einer der Werte EQ, MV, ELL, BS, STOCH.
    history : ndarray, shape (T,n)
        Ausschließlich vor dem Entscheidungstermin bekannte Monatsrenditen.
    pretrade_weights : ndarray, optional
        Vor dem Handel beobachtete, durch Kursdrift entstandene Gewichte.
        None kennzeichnet die erste Investition aus Bargeld.
    cfg : OptimizationConfig, optional
        Schätzfenster, gemeinsame Grenzen und Modellparameter.

    Raises
    ------
    ValueError
        Bei inkonsistenten Eingaben oder nicht zulässiger Lösung.
    RuntimeError
        Wenn Solver fehlt oder das Problem nicht optimal gelöst wird.
    """
    if cp is None:
        raise RuntimeError("cvxpy ist nicht installiert")
    cfg = cfg or OptimizationConfig()
    method = method.upper()
    if method not in METHODS:
        raise ValueError(f"Unbekannte Methode: {method}")
    r = _matrix(history, "history")
    t, n = r.shape
    if not (0 < cfg.max_weight <= 1) or n * cfg.max_weight < 1 - 1e-12:
        raise ValueError("Maximalgewicht macht das Portfolio unzulässig")
    if cfg.turnover_limit < 0 or cfg.lambda_risk < 0 or cfg.rho < 0:
        raise ValueError("Grenzen und Risikoparameter müssen nichtnegativ sein")
    if cfg.gamma_cvar < 0 or cfg.kappa < 0 or not (0 < cfg.alpha < 1):
        raise ValueError("Ungültige CVaR- oder BS-Parameter")
    if not (0 <= cfg.gamma_budget <= n):
        raise ValueError("BS-Budget muss zwischen 0 und ETF-Anzahl liegen")
    if cfg.scenario_window < 2:
        raise ValueError("scenario_window muss mindestens 2 sein")
    mu, sigma, omega = estimate_inputs(r, cfg)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1, w >= 0, w <= cfg.max_weight]
    pre = None
    if pretrade_weights is not None:
        pre = _vector(pretrade_weights, n, "pretrade_weights")
        if np.min(pre) < -1e-8 or abs(pre.sum() - 1) > 1e-7:
            raise ValueError("Vorhandene Gewichte müssen long-only und voll investiert sein")
        constraints.append(cp.norm1(w - pre) <= cfg.turnover_limit)

    # Mathematische Grenzfälle werden durch EXAKT dieselbe MV-Formulierung
    # gelöst. So vermeiden wir unnötige konische/duale Hilfsvariablen und
    # erhalten einen reproduzierbaren numerischen Äquivalenztest.
    effective_method = ("MV" if (method == "ELL" and cfg.rho == 0.0)
                        or (method == "BS" and cfg.gamma_budget == 0.0)
                        else method)
    if effective_method == "EQ":
        objective = cp.Minimize(cp.sum_squares(w - 1 / n))
    else:
        reward = mu @ w
        variance = cp.quad_form(w, cp.psd_wrap(sigma))
        if effective_method == "MV":
            objective = cp.Maximize(reward - cfg.lambda_risk * variance)
        elif effective_method == "ELL":
            eig, q = np.linalg.eigh(omega)
            root = np.diag(np.sqrt(np.maximum(eig, 0))) @ q.T
            objective = cp.Maximize(reward - cfg.lambda_risk * variance
                                    - cfg.rho * cp.norm(root @ w, 2))
        elif effective_method == "BS":
            p = cp.Variable(nonneg=True)
            q = cp.Variable(n, nonneg=True)
            d = cfg.kappa * np.sqrt(np.maximum(np.diag(omega), 0))
            constraints.append(p + q >= cp.multiply(d, w))
            objective = cp.Maximize(reward - cfg.lambda_risk * variance
                                    - cfg.gamma_budget * p - cp.sum(q))
        else:
            if t < cfg.scenario_window:
                raise ValueError("Zu wenig Beobachtungen für STOCH-Szenarien")
            scenarios = r[-cfg.scenario_window:]
            eta = cp.Variable()
            u = cp.Variable(cfg.scenario_window, nonneg=True)
            constraints.append(u >= -scenarios @ w - eta)
            cvar = eta + cp.sum(u) / ((1 - cfg.alpha) * cfg.scenario_window)
            objective = cp.Maximize(reward - cfg.gamma_cvar * cvar)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cfg.solver)
    except cp.error.SolverError as exc:
        raise RuntimeError(f"Solver {cfg.solver} konnte nicht lösen: {exc}") from exc
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
        raise RuntimeError(f"Optimierung nicht erfolgreich: {problem.status}")
    weights = np.asarray(w.value, dtype=float).reshape(-1)
    tol = 1e-5
    if (abs(weights.sum() - 1) > tol or np.min(weights) < -tol
            or np.max(weights) > cfg.max_weight + tol
            or (pre is not None and np.abs(weights - pre).sum() > cfg.turnover_limit + tol)):
        raise RuntimeError("Solver-Lösung verletzt die Portfolio-Restriktionen")
    # Winzige numerische Abweichungen werden nicht stillschweigend in neue
    # Portfolio-Gewichte umgewandelt; der Backtest validiert erneut.
    return OptimizationResult(method, weights, float(problem.value),
                              problem.status,
                              None if pre is None else float(np.abs(weights - pre).sum()))

# ============================================================================
# PRAKTISCHE ANWENDUNG – CSV aus Modul 1 -> fünf reale Zielallokationen
# ============================================================================

ETF_UNIVERSE = ("SPY", "EFA", "EEM", "TLT", "IEF", "GLD",
                "VNQ", "LQD", "HYG", "XLF", "XLV", "XLE")


def load_monthly_prices(csv_path):
    r"""Lädt und prüft den Monatspreisexport aus Modul 1.

    Unterstützt Wide-Format ``date,SPY,EFA,...`` oder Long-Format
    ``date,asset,price``. Für jeden Monat müssen alle zwölf ETFs einen
    positiven, endlichen Preis besitzen. Monatsabstände, Dubletten und
    ETF-Reihenfolge werden geprüft. Es findet KEINE Auffüllung fehlender
    Kurse und KEINE Rückrechnung aus zukünftigen Beobachtungen statt.

    Die Preise müssen *adjustierte* monatliche Preise sein, wie im
    bisherigen Modul-1-Export. Daraus berechnete Renditen sind einfache
    Monatsrenditen: r[i,t] = P[i,t] / P[i,t-1] - 1.

    Returns
    -------
    (dates, prices, returns)
        dates: pandas.DatetimeIndex der Preise; prices: DataFrame im
        festen ETF_UNIVERSE; returns: DataFrame der Monatsrenditen.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Für die CSV-Auswertung fehlt pandas: pip install pandas") from exc
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Monatspreisdatei nicht gefunden: {path}")
    raw = pd.read_csv(path)
    if "date" not in raw:
        raise ValueError("Die Preisdatei benötigt eine Spalte 'date'")
    raw["date"] = pd.to_datetime(raw["date"], errors="raise")
    if raw["date"].isna().any() or not raw["date"].dt.is_month_end.all():
        raise ValueError("Alle Datumswerte müssen gültige Monatsenden sein")
    if "asset" in raw:
        cols = [c for c in ("price", "adjusted_close", "adj_close", "close") if c in raw]
        if len(cols) != 1:
            raise ValueError("Long-Format benötigt genau eine eindeutige Preisspalte")
        if raw.duplicated(["date", "asset"]).any():
            raise ValueError("Doppelte Kombinationen aus Monat und ETF")
        if set(raw["asset"]) != set(ETF_UNIVERSE):
            raise ValueError("ETF-Universum stimmt nicht mit den zwölf ETFs überein")
        prices = raw.pivot(index="date", columns="asset", values=cols[0])
    else:
        if set(raw.columns) != {"date", *ETF_UNIVERSE}:
            raise ValueError("Wide-Format benötigt date und exakt die zwölf ETF-Spalten")
        if raw["date"].duplicated().any():
            raise ValueError("Doppelte Monatsdaten")
        prices = raw.set_index("date")
    prices = prices.sort_index().loc[:, list(ETF_UNIVERSE)]
    prices = prices.apply(pd.to_numeric, errors="raise")
    if prices.isna().any().any() or not np.isfinite(prices.to_numpy()).all():
        raise ValueError("Fehlende oder unendliche Monatspreise")
    if (prices.to_numpy() <= 0).any():
        raise ValueError("Monatspreise müssen positiv sein")
    expected = pd.date_range(prices.index[0], prices.index[-1], freq="ME")
    if not prices.index.equals(expected):
        raise ValueError("Monatspreise enthalten Lücken oder unregelmäßige Monatsenden")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    if len(returns) < 120:
        raise ValueError("Für STOCH werden mindestens 120 vollständige Monatsrenditen benötigt")
    return prices.index, prices, returns


def load_pretrade_weights(csv_path):
    """Lädt optionale tatsächliche Vorhandelsgewichte (asset,weight).

    Ohne diese Datei erfolgt eine **Erstinvestition** ohne Turnover-Grenze.
    Für spätere Rebalancings muss die Datei die vor Handel beobachteten
    Portfolioanteile enthalten; ein willkürliches Gleichgewichtungsportfolio
    darf nicht als Ersatz verwendet werden.
    """
    if csv_path is None:
        return None
    import pandas as pd
    df = pd.read_csv(csv_path)
    if set(df.columns) != {"asset", "weight"} or df["asset"].duplicated().any():
        raise ValueError("Gewichtsdatei benötigt eindeutige Spalten asset,weight")
    if set(df["asset"]) != set(ETF_UNIVERSE):
        raise ValueError("Vorhandelsgewichte: ETF-Universum stimmt nicht überein")
    weights = df.set_index("asset").loc[list(ETF_UNIVERSE), "weight"].to_numpy(dtype=float)
    _vector(weights, len(ETF_UNIVERSE), "pretrade_weights")
    if weights.min() < -1e-8 or abs(weights.sum() - 1) > 1e-7:
        raise ValueError("Vorhandelsgewichte müssen long-only sein und 100 % ergeben")
    return weights


def run_real_optimization(prices_csv, pretrade_csv=None, output_dir=None, cfg=None):
    r"""Berechnet aus Modul-1-Daten die fünf optimalen ETF-Gewichtsvektoren.

    Historie H_t: ausschließlich vollständig beobachtete Monatsrenditen
    bis zum letzten Datum der Preisdatei. Alle fünf Modelle verwenden
    dieselbe Historie und dieselben Restriktionen. HIST ist der arithmetische
    Mittelwert über die gesamte gemeinsame Renditehistorie, Sigma basiert
    auf den letzten 36 Monaten und Omega auf der Bartlett-HAC-Schätzung.

    Das Ergebnis ist eine *aktuelle Zielallokation*, kein historischer
    Walk-Forward-Backtest. Die aus Monatsenddaten berechnete Allokation
    kann nicht rückwirkend zum selben Monatsendkurs gehandelt werden.
    Euro-Zielwerte sind indikative ETF-Positionen bei Erstinvestition:
    10.000 / (1+0.002) = 9.980,04 EUR nach den initialen Gebühren.
    Bei vorhandenem Portfolio werden keine Eurobeträge behauptet, da
    aktuelle Bestände und das selbstfinanzierende Handelsmodell fehlen.

    Parameter sind demonstrative Startwerte, nicht historisch kalibriert.
    """
    from pathlib import Path
    import pandas as pd
    cfg = cfg or OptimizationConfig()
    _, prices, returns = load_monthly_prices(prices_csv)
    pre = load_pretrade_weights(pretrade_csv)
    history = returns.to_numpy(dtype=float)
    mu, sigma, omega = estimate_inputs(history, cfg)
    results = [optimize_portfolio(m, history, pre, cfg) for m in METHODS]
    weights = pd.DataFrame({r.method: r.weights for r in results}, index=ETF_UNIVERSE)
    weights.index.name = "asset"
    report = pd.DataFrame([{
        "method": r.method,
        "expected_monthly_return": float(mu @ r.weights),
        "expected_monthly_volatility": float(np.sqrt(max(r.weights @ sigma @ r.weights, 0))),
        "historical_scenario_cvar": _empirical_cvar(
            -history[-cfg.scenario_window:] @ r.weights, cfg.alpha),
        "turnover_l1": r.turnover_l1,
        "solver_status": r.status,
        "objective_value": r.objective_value,
    } for r in results])
    print("\n" + "=" * 74)
    print("MODUL 3 | FÜNF PORTFOLIOOPTIMIERUNGEN MIT ECHTEN MODUL-1-DATEN")
    print("=" * 74)
    print(f"Letzter verfügbarer Monatsschluss: {prices.index[-1].date()}")
    print(f"Historische Renditemonate: {len(returns)} | ETFs: {len(ETF_UNIVERSE)}")
    print(f"Kovarianz: {cfg.covariance_window} Monate | CVaR-Szenarien: {cfg.scenario_window}")
    print(f"Vorhandelsgewichte: {'übergeben' if pre is not None else 'keine (Erstinvestition)'}")
    print("WARNUNG: Modellparameter sind noch nicht historisch kalibriert.")
    print("WARNUNG: Zielgewichte sind kein ausgeführter Handel und kein Backtest.\n")
    print("OPTIMALE ETF-GEWICHTE IN PROZENT")
    print((weights * 100).round(2).to_string())
    print("\nMODELLKENNZAHLEN (MONATLICH, DEZIMALWERTE)")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    if pre is None:
        invested = 10_000 / 1.002
        print(f"\nINDIKATIVE ERSTINVESTITION: 10.000 EUR Budget, "
              f"{invested:,.2f} EUR ETF-Volumen, {10_000-invested:,.2f} EUR Gebühren")
        print("Keine realen Ausführungskurse oder Stückzahlen berechnet.")
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        weights.to_csv(folder / "portfolio_weights.csv", float_format="%.10f")
        report.to_csv(folder / "portfolio_metrics.csv", index=False, float_format="%.10f")
        print(f"\nCSV-Ergebnisse gespeichert in: {folder.resolve()}")
    return weights, report


def _empirical_cvar(losses, alpha):
    r"""Berechnet empirischen CVaR als Minimum der Rockafellar-Uryasev-Form.

    Für S gleichgewichtete Verluste L_s:
    CVaR_alpha = min_eta [eta + sum(max(L_s-eta,0)) / ((1-alpha)*S)].
    Ein Minimum liegt an einem beobachteten Verlustwert; diese exakte
    diskrete Auswertung benötigt keinen zusätzlichen Optimierungssolver.
    """
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("CVaR-Verluste müssen ein endlicher Vektor sein")
    return float(np.min(x[:, None] + np.maximum(x[None, :] - x[:, None], 0).sum(axis=1)
                        / ((1 - alpha) * len(x))))


def main(argv=None):
    """Startet unmittelbar die fünf ETF-Optimierungen aus der Preis-CSV."""
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Fünf ETF-Optimierungen in einer Python-Datei")
    parser.add_argument("--prices", type=Path, help="Modul-1-Datei prices_monthly.csv")
    parser.add_argument("--pretrade", type=Path, help="Optional: asset,weight vor dem Rebalancing")
    parser.add_argument("--output", type=Path, help="Ausgabeordner für Ergebnis-CSV")
    args = parser.parse_args(argv)
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir / "data" / "prices_monthly.csv",
                  script_dir.parent / "data" / "prices_monthly.csv",
                  Path.cwd() / "data" / "prices_monthly.csv"]
    prices_csv = args.prices or next((p for p in candidates if p.is_file()), None)
    if prices_csv is None:
        print("FEHLENDE EINGABEDATEI: data/prices_monthly.csv\n"
              "Bitte die Datei aus Modul 1 in den Unterordner data legen oder "
              "mit --prices DATEIPFAD angeben.\n"
              "Es werden absichtlich KEINE erfundenen ETF-Ergebnisse erzeugt.")
        return 2
    try:
        run_real_optimization(prices_csv, args.pretrade,
                              args.output or script_dir / "output" / "module3")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"FEHLER: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
