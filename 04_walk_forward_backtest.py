"""Modul 4 – Selbstfinanzierender monatlicher Walk-Forward-Backtest.

Vergleicht EQ, MV, ELL, BS und STOCH mit identischer Datenbasis, gleichem
Startkapital und denselben Restriktionen. Benötigt 03_portfolio_optimization.py
im selben Ordner sowie data/prices_monthly.csv aus Modul 1.

ZEITKONVENTION / GRENZE DER AUSSAGEKRAFT
---------------------------------------
Am Ende von Monat t werden ausschließlich Renditen bis einschließlich t
für die Schätzung benutzt. Das daraus berechnete Portfolio wird in einem
vereinfachten Monatsend-Modell unmittelbar vor der Rendite von t+1
positioniert. Tatsächlich kann man eine nach dem Schlusskurs berechnete
Allokation nicht zu genau diesem Schlusskurs ausführen. Der Backtest ist
somit ein modellbasierter Monatsrendite-Backtest und KEINE exakt ausführbare
Handelssimulation. Für exakte Ausführung wären weitere tägliche Daten nötig.

HANDELSRECHNUNG
---------------
Vor dem Handel existieren Bestände h_i^- mit Gesamtwert V^-.
Das Zielgewicht w_i gilt für die Bestände NACH Kosten:
    h_i^+ = V^+ w_i
    V^+ = V^- - c * sum_i |h_i^+ - h_i^-|.
Die eindeutige Lösung wird per Bisektion berechnet. Für die Erstinvestition
ist h^- = 0 und V^+ = V^-/(1+c). Bei monatlicher Kursentwicklung r_i
werden die Bestände mit (1+r_i) fortgeschrieben. Das Geld für Gebühren
wird nicht von außen zugeschossen. Die L1-Umschichtungsgrenze bezieht sich
auf die Gewichte VOR und NACH Rebalancing, nicht auf exakte Euro-Umsätze.

KALIBRIERUNG
------------
Die Risikoparameter sind die unveränderten Demonstrationswerte aus
OptimizationConfig. KEINE nachträgliche Optimierung auf 2023–2026.
Dies ist ein explorativer Vergleich, kein vollständig unberührter Test.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Die GitHub-Nummerierung bleibt erhalten. Ein normaler Python-Import kann
# nicht mit einer Ziffer beginnen; deshalb laden wir Modul 03 explizit.
MODUL_03 = Path(__file__).resolve().parent / "03_portfolio_optimization.py"
if not MODUL_03.is_file():
    raise FileNotFoundError(
        f"Optimierungsmodul nicht gefunden: {MODUL_03}. "
        "Lege 03_portfolio_optimization.py neben diese Datei."
    )
_spec = importlib.util.spec_from_file_location("portfolio_optimization", MODUL_03)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Modul 03 kann nicht geladen werden: {MODUL_03}")
_opt = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _opt  # Erforderlich für @dataclass und Modul-Referenzen.
_spec.loader.exec_module(_opt)

METHODS = _opt.METHODS
ETF_UNIVERSE = _opt.ETF_UNIVERSE
OptimizationConfig = _opt.OptimizationConfig
load_monthly_prices = _opt.load_monthly_prices
optimize_portfolio = _opt.optimize_portfolio
_empirical_cvar = _opt._empirical_cvar


@dataclass
class Account:
    """Ein Portfolio-Konto mit wertmäßigen ETF-Beständen nach Kosten."""
    holdings: np.ndarray
    nav: float
    total_fees: float = 0.0
    total_gross_traded: float = 0.0


def settle_trade(pre_holdings: np.ndarray, target_weights: np.ndarray,
                 fee_rate: float) -> tuple[np.ndarray, float, float]:
    r"""Löst die Selbstfinanzierungsbedingung für eine Handelsentscheidung.

    Für V-=sum(h-) und w>=0, sum(w)=1 löst die Bisektion
    f(x)=x+c*sum_i|x*w_i-h_i^-|-V-=0 auf [0,V-].
    Das Ergebnis x ist das nach Gebühren investierte Vermögen.

    Parameters
    ----------
    pre_holdings : ndarray
        Positive, aktuelle ETF-Bestandswerte vor dem Handel.
    target_weights : ndarray
        Optimale Zielgewichte mit Summe eins.
    fee_rate : float
        Proportionale Gebühr auf die Summe der absoluten Handelsbeträge.

    Returns
    -------
    (holdings_after, fees, gross_volume)
        ETF-Bestände nach Handel, Gebühren und Brutto-Handelsvolumen in EUR.
    """
    h = np.asarray(pre_holdings, dtype=float)
    w = np.asarray(target_weights, dtype=float)
    if h.ndim != 1 or w.shape != h.shape or not np.isfinite(h).all() or not np.isfinite(w).all():
        raise ValueError('Ungültige Handelsbestände oder Zielgewichte')
    if (h < -1e-8).any() or (w < -1e-7).any() or abs(w.sum()-1) > 1e-5:
        raise ValueError('Ungültige Long-only-Bestände oder Zielgewichte')
    if fee_rate < 0 or fee_rate >= 1:
        raise ValueError('Gebühr muss im Intervall [0,1) liegen')
    h = np.maximum(h, 0)
    w = np.maximum(w, 0) / np.maximum(w, 0).sum()
    v = float(h.sum())
    if v <= 0:
        raise ValueError('Vermögen muss positiv sein')
    lo, hi = 0.0, v
    for _ in range(90):
        mid = (lo + hi) / 2
        if mid + fee_rate * np.abs(mid * w - h).sum() > v:
            hi = mid
        else:
            lo = mid
    v_after = (lo + hi) / 2
    new_h = v_after * w
    gross = float(np.abs(new_h - h).sum())
    fees = fee_rate * gross
    if abs(new_h.sum() + fees - v) > max(1e-7, 1e-10*v):
        raise RuntimeError('Selbstfinanzierung verletzt')
    return new_h, fees, gross


def performance_statistics(monthly_returns: np.ndarray, nav: np.ndarray,
                           fee_total: float, gross_total: float,
                           initial_capital: float, alpha: float) -> dict:
    """Berechnet realisierte Netto-Performance aus einer NAV-Zeitreihe.

    Rendite p.a. = (NAV_final/NAV_initial)^(12/T)-1; Volatilität p.a.
    = std(R,ddof=1)*sqrt(12); Sharpe mit risikofreiem Zinssatz 0 % als
    EXPLIZITER Vergleichskonvention, nicht als historischer Marktzins.
    Maximaler Drawdown berücksichtigt den Anfangs-NAV von 10.000 EUR.
    CVaR ist als positiver/negativer durchschnittlicher VERLUST der
    schlechtesten (1-alpha) Monatsrenditen definiert.
    """
    r = np.asarray(monthly_returns, float)
    path = np.asarray(nav, float)
    t = len(r)
    if t < 2 or len(path) != t+1:
        raise ValueError('Zu wenig Renditemonate oder inkonsistente NAV-Länge')
    ann = float((path[-1] / initial_capital) ** (12/t) - 1)
    vol = float(np.std(r, ddof=1) * np.sqrt(12))
    sharpe = float(np.mean(r) * 12 / vol) if vol > 0 else np.nan
    dd = path / np.maximum.accumulate(path) - 1
    return {
        'months': t, 'final_nav_eur': float(path[-1]),
        'annualized_net_return': ann, 'annualized_volatility': vol,
        'sharpe_rf_zero': sharpe, 'max_drawdown': float(dd.min()),
        'realized_monthly_loss_cvar': _empirical_cvar(-r, alpha),
        'total_fees_eur': float(fee_total),
        'gross_trading_volume_eur': float(gross_total),
    }


def run_backtest(prices_csv: str | Path, output_dir: str | Path,
                 start: str = '2023-01-31', initial_capital: float = 10_000,
                 fee_rate: float = 0.002,
                 cfg: OptimizationConfig | None = None):
    r"""Führt fünf vollständig getrennte monatliche Portfolio-Konten.

    Für jeden OOS-Monat t wird ausschließlich die Historie vor t verwendet.
    Am Ende von t-1 wird das Portfolio optimiert und auf das kommende
    Renditeintervall ausgerichtet (vereinfachte Monatsend-Ausführung).
    Anfangskapital und Erstinvestitionskosten sind für alle Modelle gleich.
    In Folgemonaten wird der durch Renditen entstandene Bestand vor Handel
    für Turnover und die selbstfinanzierende Abrechnung verwendet.

    Output: NAV, Monatsrenditen, Allokationen, Handelsdetails, Kennzahlen.
    Keine Zukunftsdaten werden in Schätzungen verwendet. Alle Strategien
    haben exakt denselben Evaluierungszeitraum.
    """
    cfg = cfg or OptimizationConfig()
    if initial_capital <= 0 or not (0 <= fee_rate < 1):
        raise ValueError('Ungültiges Kapital oder Transaktionskosten')
    _, _, monthly = load_monthly_prices(prices_csv)
    start_date = pd.Timestamp(start)
    first = int(monthly.index.searchsorted(start_date))
    minimum = max(cfg.scenario_window, cfg.covariance_window)
    if first < minimum:
        raise ValueError(f'Vor dem Backtestbeginn werden mindestens {minimum} Renditemonate benötigt; vorhanden: {first}')
    if first >= len(monthly):
        raise ValueError('Backtestbeginn liegt nach der letzten verfügbaren Rendite')
    evaluation_dates = monthly.index[first:]
    accounts = {m: Account(np.zeros(len(ETF_UNIVERSE)), initial_capital) for m in METHODS}
    nav_rows = [{'date': monthly.index[first-1], **{m: initial_capital for m in METHODS}}]
    return_rows, weight_rows, trade_rows = [], [], []

    for idx in range(first, len(monthly)):
        period = monthly.index[idx]
        history = monthly.iloc[:idx].to_numpy(dtype=float)
        realized = monthly.iloc[idx].to_numpy(dtype=float)
        r_row = {'date': period}
        nav_row = {'date': period}
        for method in METHODS:
            account = accounts[method]
            before = float(account.holdings.sum()) if idx > first else initial_capital
            pre_weights = None if idx == first else account.holdings / before
            solution = optimize_portfolio(method, history, pre_weights, cfg)
            if idx == first:
                # Anfangsbestand: ausschließlich Cash. Einzahlungen über 10.000 EUR verboten.
                # settle_trade benötigt die Gesamtvermögenssumme: Cash als eigener
                # Vorhandelsbestand ist keine ETF-Position. Die analytische Lösung gilt.
                invested = initial_capital / (1 + fee_rate)
                holdings = invested * solution.weights
                gross = invested
                fees = initial_capital - invested
            else:
                holdings, fees, gross = settle_trade(account.holdings, solution.weights, fee_rate)
            post_nav = float(holdings.sum())
            new_holdings = holdings * (1 + realized)
            if (new_holdings < 0).any():
                raise ValueError(f'Monatsrendite unter -100 % am {period.date()}')
            end_nav = float(new_holdings.sum())
            account.holdings = new_holdings
            account.nav = end_nav
            account.total_fees += fees
            account.total_gross_traded += gross
            r_row[method] = end_nav / before - 1
            nav_row[method] = end_nav
            weight_rows.extend({'date': period, 'method': method, 'asset': asset,
                                'weight': float(weight)}
                               for asset, weight in zip(ETF_UNIVERSE, solution.weights))
            trade_rows.append({'date': period, 'method': method,
                               'pretrade_nav_eur': before,
                               'posttrade_nav_eur': post_nav,
                               'trading_fees_eur': fees,
                               'gross_trading_volume_eur': gross,
                               'turnover_l1': solution.turnover_l1,
                               'solver_status': solution.status})
        return_rows.append(r_row)
        nav_rows.append(nav_row)

    ret = pd.DataFrame(return_rows).set_index('date')
    nav = pd.DataFrame(nav_rows).set_index('date')
    weights = pd.DataFrame(weight_rows)
    trades = pd.DataFrame(trade_rows)
    stats = pd.DataFrame([{'method': m, **performance_statistics(
        ret[m].to_numpy(), nav[m].to_numpy(), accounts[m].total_fees,
        accounts[m].total_gross_traded, initial_capital, cfg.alpha)}
        for m in METHODS])

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    ret.to_csv(folder / 'monthly_net_returns.csv', float_format='%.10f')
    nav.to_csv(folder / 'nav_eur.csv', float_format='%.6f')
    weights.to_csv(folder / 'monthly_weights.csv', index=False, float_format='%.10f')
    trades.to_csv(folder / 'monthly_trades.csv', index=False, float_format='%.10f')
    stats.to_csv(folder / 'backtest_metrics.csv', index=False, float_format='%.10f')
    print('\n' + '='*80)
    print('MODUL 4 | WALK-FORWARD-BACKTEST: EQ, MV, ELL, BS, STOCH')
    print('='*80)
    print(f'Gemeinsamer Evaluierungszeitraum: {evaluation_dates[0].date()} bis {evaluation_dates[-1].date()}')
    print(f'Monate: {len(evaluation_dates)} | Anfangskapital: {initial_capital:,.2f} EUR')
    print(f'Handelsgebühr: {fee_rate:.2%} | L1-Limit: {cfg.turnover_limit:.2f}')
    print('ACHTUNG: vereinfachte Monatsend-Ausführungsannahme, kein real ausführbarer Handel.')
    print('ACHTUNG: Modellparameter noch nicht historisch kalibriert.')
    print('ACHTUNG: 2023+ nicht mehr vollständig unabhängige Testperiode (Modellwahl).')
    print('\nREALISIERTE NETTO-KENNZAHLEN')
    print(stats.to_string(index=False, float_format=lambda v: f'{v:.5f}'))
    print(f'\nErgebnisdateien: {folder.resolve()}')
    return stats, nav, ret, weights, trades


def main():
    """Startet direkt den Backtest und KEINE Tests."""
    parser = argparse.ArgumentParser(description='Monatlicher Walk-Forward-Backtest aller fünf Optimierungsverfahren')
    parser.add_argument('--prices', type=Path, default=None, help='Pfad zu prices_monthly.csv')
    parser.add_argument('--start', default='2023-01-31', help='Erster Evaluierungsmonat, YYYY-MM-DD')
    parser.add_argument('--output', type=Path, default=None, help='Ordner für CSV-Ergebnisse')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    candidates = [here/'data'/'prices_monthly.csv', here.parent/'data'/'prices_monthly.csv',
                  Path.cwd()/'data'/'prices_monthly.csv']
    prices = args.prices or next((p for p in candidates if p.is_file()), None)
    if prices is None:
        parser.error('data/prices_monthly.csv nicht gefunden; bitte --prices PFAD angeben')
    run_backtest(prices, args.output or here/'output'/'module4', start=args.start)


if __name__ == '__main__':
    main()
