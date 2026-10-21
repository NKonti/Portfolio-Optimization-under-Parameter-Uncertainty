"""Modul 1 – Datenaufbereitung und Feature Engineering für ETF-Portfolios.

Zweck
-----
Dieses Modul erstellt eine reproduzierbare monatliche Datenbasis für die
nachfolgenden Prognose-, Optimierungs- und Backtesting-Module. Es lädt
historische adjustierte ETF-Tageskurse, aggregiert sie auf abgeschlossene
Kalendermonate und berechnet historische Renditen und Prognosemerkmale.

Methodik
--------
- Rendite: R[i,t] = P[i,t] / P[i,t-1] - 1.
- Momentum: P[i,t] / P[i,t-k] - 1 für k = 1, 3, 6, 12 Monate.
- Volatilität: rollierende Stichprobenstandardabweichung der Monatsrenditen
  über 3, 6 bzw. 12 Monate (ddof=1).
- Drawdown: P[i,t] / max(P[i,t-11], ..., P[i,t]) - 1.
- Prognoseziel: target[i,t] = R[i,t+1].

Die Features einer Zeile t verwenden ausschließlich Beobachtungen bis t.
Das Target dieser Zeile ist erst nach Abschluss des Folgemonats bekannt.
Die letzte abgeschlossene Monatsbeobachtung wird daher separat ohne Target
für spätere Prognosen bereitgestellt. Der laufende Monat wird ausgeschlossen.

Datenuniversum
--------------
SPY, EFA, EEM, TLT, IEF, GLD, VNQ, LQD, HYG, XLF, XLV und XLE;
historische Daten ab 2010-01-01 von Yahoo Finance (yfinance).

Ausgabedateien (im Verzeichnis data/)
------------------------------------
- prices_daily_adjusted.csv: Cache der adjustierten Tageskurse.
- prices_monthly.csv: Adjustierte Monatsendkurse.
- returns_monthly.csv: Realisierte Monatsrenditen.
- feature_panel_all.csv: Vollständiges Panel einschließlich fehlender Werte.
- forecast_dataset_monthly.csv: Trainingspanel mit Features und bekanntem Target.
- latest_features_for_forecast.csv: Neueste Features ohne zukünftiges Target.
- data_quality_by_asset.csv: Abdeckung und fehlende Monatskurse je ETF.
- module1_report.json: Prüfresultate, Metadaten und Einschränkungen.

Qualitätssicherung
------------------
Prüft unter anderem eindeutige Datums-/ETF-Schlüssel, Vollständigkeit des
Trainingspanels, Vorhandensein aktueller Features und die exakte Ausrichtung
jedes Targets auf die realisierte Rendite des Folgemonats. Bei Fehlern wird
die Ausführung abgebrochen. Ein bestandener Check garantiert nicht die
wirtschaftliche Richtigkeit sämtlicher Quelldaten.

Grenzen
-------
Adjustierte Yahoo-Finance-Kurse sind nur eine Näherung für Total Returns.
Das rückwirkend gewählte ETF-Universum kann Selection/Survivorship Bias
verursachen. Monatsschlusssignale können nicht rückwirkend zum bereits
beobachteten Schlusskurs ausgeführt werden. Die Handelszeitpunktannahme
muss im späteren Backtest ausdrücklich modelliert werden.

Ausführung
----------
    pip install numpy pandas yfinance
    python 01_data_pipeline.py --refresh

Ohne --refresh werden vorhandene Tagesdaten aus dem Cache gelesen.
Dieses Modul trainiert keine Prognosemodelle und optimiert keine Portfolios.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "LQD", "HYG", "XLF", "XLV", "XLE"]
FEATURES = ["mom_1", "mom_3", "mom_6", "mom_12", "vol_3", "vol_6", "vol_12", "drawdown_12"]
START = "2010-01-01"


def last_completed_month_end(now_utc: pd.Timestamp) -> pd.Timestamp:
    """Gib das Monatsultimo des letzten vollständig abgeschlossenen Monats zurück.

    Args:
        now_utc: Aktueller Zeitstempel ohne Zeitzoneninformation in UTC.

    Returns:
        Monatsultimo als pandas Timestamp.
    """
    # Month-end timestamp of the previous calendar month, irrespective of current day.
    return (now_utc.to_period("M") - 1).to_timestamp("M")


def fetch_daily(tickers: list[str], start: str) -> pd.DataFrame:
    """Lade und validiere adjustierte ETF-Tageskurse von Yahoo Finance.

    Args:
        tickers: Erwartete ETF-Symbole und Spaltenreihenfolge.
        start: Startdatum inklusive, im Format YYYY-MM-DD.

    Returns:
        Nach Handelstag sortierte Tabelle numerischer adjustierter Kurse.

    Raises:
        RuntimeError: Wenn Daten oder erwartete Spalten fehlen.
    """
    import yfinance as yf
    raw = yf.download(tickers=tickers, start=start, auto_adjust=True,
                      progress=False, threads=False, group_by="column")
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data; check internet access.")
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Unexpected yfinance schema: expected multi-index columns.")
    close = raw["Close"].copy()
    missing = sorted(set(tickers) - set(close.columns))
    if missing:
        raise RuntimeError(f"Missing ETF price columns: {missing}")
    close = close.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close = close.sort_index()
    if close.index.has_duplicates:
        raise RuntimeError("Duplicate trading dates in downloaded data.")
    return close.apply(pd.to_numeric, errors="coerce")


def build(daily: pd.DataFrame, cutoff: pd.Timestamp):
    """Erzeuge Monatsrenditen, historische Features und zeitlich korrekte Targets.

    Args:
        daily: Tageskurse mit DatetimeIndex und einer Spalte pro ETF.
        cutoff: Letzter zulässiger abgeschlossener Monatsultimo.

    Returns:
        Tupel (prices, rets, full, training, latest): Monatskurse,
        Monatsrenditen, vollständiges Panel, nutzbare Trainingszeilen und
        aktuelle Feature-Zeilen ohne zukünftige Zielrendite.

    Notes:
        Das Trainingstarget einer Zeile t ist die Rendite t+1. Fehlende
        Kurswerte werden für die Renditeberechnung nicht fortgeschrieben.
    """
    # Resample only completed calendar months. A month with no quote is left NaN.
    daily = daily.loc[daily.index <= cutoff].copy()
    if daily.empty:
        raise RuntimeError("No prices prior to cutoff.")
    prices = daily.resample("ME").last().reindex(columns=TICKERS)
    prices = prices.loc[prices.index <= cutoff]
    if prices.empty or prices.isna().all(axis=1).any():
        raise RuntimeError("At least one month has no valid quotes for any ETF.")
    if (prices.dropna() <= 0).any().any():
        raise RuntimeError("Nonpositive ETF prices detected.")
    # Critical: pct_change must not forward-fill missing prices.
    rets = prices.pct_change(fill_method=None)
    # An asset's rolling features are invalid across missing observations.
    mom = {k: prices / prices.shift(k) - 1 for k in (1, 3, 6, 12)}
    vol = {k: rets.rolling(k, min_periods=k).std(ddof=1) for k in (3, 6, 12)}
    dd = prices / prices.rolling(12, min_periods=12).max() - 1
    target = rets.shift(-1)

    records = []
    for asset in TICKERS:
        frame = pd.DataFrame({
            "date": prices.index,
            "asset": asset,
            "price": prices[asset].to_numpy(),
            "return_t": rets[asset].to_numpy(),
            **{f"mom_{k}": mom[k][asset].to_numpy() for k in (1, 3, 6, 12)},
            **{f"vol_{k}": vol[k][asset].to_numpy() for k in (3, 6, 12)},
            "drawdown_12": dd[asset].to_numpy(),
            "target_return_next_month": target[asset].to_numpy(),
        })
        records.append(frame)
    full = pd.concat(records, ignore_index=True).sort_values(["date", "asset"]).reset_index(drop=True)
    usable = full.dropna(subset=FEATURES + ["price", "return_t"]).copy()
    training = usable.dropna(subset=["target_return_next_month"]).copy()
    latest = usable.loc[usable["date"] == prices.index.max()].copy()
    # At cutoff, next-month targets must be unknown.
    if not latest["target_return_next_month"].isna().all():
        raise AssertionError("Latest-month target must be unknown.")
    return prices, rets, full, training, latest


def validate(prices, rets, full, training, latest, cutoff):
    """Prüfe Struktur, Datenvollständigkeit und zeitliche Target-Ausrichtung.

    Args:
        prices: Adjustierte Monatsendkurse.
        rets: Monatsrenditen aus den Kursen.
        full: Panel aller Features und Targets.
        training: Panel mit vollständigen Trainingsmerkmalen und Targets.
        latest: Neueste Feature-Zeilen ohne bekanntes Target.
        cutoff: Letztes erlaubtes Monatsultimo.

    Returns:
        True, wenn sämtliche implementierten Prüfungen bestanden sind.

    Raises:
        AssertionError: Bei einem Verstoß gegen die Dateninvarianten.
    """
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise AssertionError("Monthly index is not unique/increasing.")
    if prices.index.max() > cutoff:
        raise AssertionError("Incomplete month included.")
    if full.duplicated(["date", "asset"]).any():
        raise AssertionError("Duplicate date/asset pairs.")
    if training[FEATURES + ["target_return_next_month"]].isna().any().any():
        raise AssertionError("Missing feature or target in training panel.")
    if training.empty or latest.empty:
        raise AssertionError("Empty training or latest feature panel.")
    if set(training["asset"].unique()) != set(TICKERS):
        raise AssertionError("One or more ETFs have no training rows.")
    if set(latest["asset"]) != set(TICKERS):
        raise AssertionError("Latest month does not contain all 12 ETFs with complete features.")
    # Independent alignment check: target at t equals asset's realized return at t+1.
    next_month = {d: prices.index[i + 1] for i, d in enumerate(prices.index[:-1])}
    for row in training.itertuples(index=False):
        next_dt = next_month.get(row.date)
        if next_dt is None:
            raise AssertionError("Training row with target beyond cutoff.")
        actual = rets.at[next_dt, row.asset]
        if not np.isclose(row.target_return_next_month, actual, atol=1e-12, rtol=1e-10):
            raise AssertionError(f"Target misalignment: {row.date} {row.asset}")
    # Verify that features are computed exclusively from prices at or before their date.
    if (training["date"] >= prices.index.max()).any():
        raise AssertionError("Training includes target from future beyond cutoff.")
    return True


def main():
    """Führe Download/Cache, Feature-Berechnung, Validierung und Export aus.

    Kommandozeilenargumente:
        --refresh: Erzwingt den erneuten Download der Tageskurse.
        --output-dir: Zielordner für CSV-Dateien und Prüfbericht (data/).

    Der Konsolenbericht enthält Datenabdeckung, Zeitraum und Prüfergebnis.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Redownload daily adjusted ETF prices")
    p.add_argument("--output-dir", default="data", help="Folder for input cache and outputs")
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "prices_daily_adjusted.csv"
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    cutoff = last_completed_month_end(now)
    print("=" * 65)
    print("MODULE 1 — DATA ENGINEERING & FEATURE VALIDATION")
    print("=" * 65)
    print("UTC run date:", now.strftime("%Y-%m-%d"))
    print("Last completed calendar month:", cutoff.date())
    print("ETF universe:", ", ".join(TICKERS))
    if args.refresh or not cache.exists():
        print("SOURCE: downloading adjusted daily prices from Yahoo Finance")
        daily = fetch_daily(TICKERS, START)
        daily.to_csv(cache, index_label="date", float_format="%.12g")
    else:
        print("SOURCE: loading cached adjusted daily prices (use --refresh to update)")
        daily = pd.read_csv(cache, index_col="date", parse_dates=["date"])
        daily = daily.reindex(columns=TICKERS)
    if daily.empty:
        raise RuntimeError("Daily data empty")
    prices, rets, full, training, latest = build(daily, cutoff)
    validate(prices, rets, full, training, latest, cutoff)

    prices.to_csv(out / "prices_monthly.csv", index_label="date")
    rets.to_csv(out / "returns_monthly.csv", index_label="date")
    full.to_csv(out / "feature_panel_all.csv", index=False)
    training.to_csv(out / "forecast_dataset_monthly.csv", index=False)
    latest.to_csv(out / "latest_features_for_forecast.csv", index=False)

    coverage = pd.DataFrame({
        "asset": TICKERS,
        "monthly_price_missing": [int(prices[a].isna().sum()) for a in TICKERS],
        "training_observations": [int((training.asset == a).sum()) for a in TICKERS],
        "first_training_date": [str(training.loc[training.asset == a, "date"].min().date()) for a in TICKERS],
        "last_training_date": [str(training.loc[training.asset == a, "date"].max().date()) for a in TICKERS],
    })
    coverage.to_csv(out / "data_quality_by_asset.csv", index=False)
    summary = {
        "run_utc": now.isoformat(), "last_completed_month": str(cutoff.date()),
        "first_monthly_date": str(prices.index.min().date()),
        "last_monthly_date": str(prices.index.max().date()),
        "monthly_observations": len(prices), "asset_count": len(TICKERS),
        "training_rows": len(training), "latest_feature_rows": len(latest),
        "training_date_min": str(training.date.min().date()),
        "training_date_max": str(training.date.max().date()),
        "checks_passed": True,
        "limitations": [
            "Adjusted prices from Yahoo Finance are a total-return proxy, not audited total-return indices.",
            "The current ETF universe is chosen retrospectively (survivorship/selection bias).",
            "Month-end signals assume trading after the closing price is observable; implementation at the same close is not claimed.",
            "Latest forecast features are exported separately without a future target.",
            "Historical daily prices are cached; --refresh is required for updated observations."
        ]
    }
    (out / "module1_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDATA COVERAGE")
    print(coverage.to_string(index=False))
    print("\nSUMMARY")
    for key, value in summary.items():
        if key != "limitations":
            print(f"{key}: {value}")
    print("\nDATA QUALITY CHECK: PASS")
    print("OUTPUT FOLDER:", out.resolve())
    print("Next step: send the complete terminal output and module1_report.json. Do not proceed to module 2 yet.")


if __name__ == "__main__":
    main()
