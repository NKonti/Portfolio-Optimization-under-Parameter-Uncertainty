"""Modul 2 — Zeitlich saubere Prognose und Modellauswahl (finales Evaluationsprotokoll, Version 5).

Zweck
-----
Vergleicht eine historische Mittelwert-Benchmark, gepoolte OLS, Random Forest
und XGBoost bei der Vorhersage monatlicher ETF-Renditen. Für jeden
Prognosemonat t werden ausschließlich Labels mit Feature-Datum < t genutzt:
R_(i,t+1) = f(X_(i,t)) + e_(i,t+1). Ein Trainingseintrag mit Feature-Datum s
ist am Ende des Monats t erst dann bekannt, wenn s+1 <= t.

Methodik
--------
Die Zeitachsen sind *Feature-Monate*. Validierung: 2018–2022; unabhängiger
Test: ab 2023. Die Startdaten sind explizite Forschungshypothesen und müssen
vor Einsicht in die Testergebnisse festgelegt bleiben. Pro Modell/Kandidat
werden in der Validierung expanding-window Out-of-Sample-Prognosen erstellt.
Die Auswahl erfolgt nach Validierungs-RMSE; Testdaten beeinflussen weder
Modellwahl noch Hyperparameter. In der Testphase wird am Monatsende auf allen
bis dahin bekannten Trainingslabels erneut geschätzt. Die Vorhersage für den
aktuellsten Monat wird ohne unbekanntes Label erstellt.

Die historische Benchmark nutzt für jeden ETF den Mittelwert aller zum
Prognosezeitpunkt bekannten Monatsrenditen. OLS, RF und XGB verwenden ein
gepooltes Panelmodell mit One-Hot-ETF-Kennungen, um unterschiedliche
langfristige Renditeniveaus berücksichtigen zu können. Standardisierung für
OLS wird nur innerhalb der jeweiligen Trainingsstichprobe geschätzt.

Metriken
--------
RMSE=sqrt(mean((y-yhat)^2)); MAE=mean(abs(y-yhat)). Pearson-IC und
Spearman-Rank-IC werden *monatlich im Querschnitt* berechnet und anschließend
über gültige Monate gemittelt. Konstante Prognosen erzeugen einen undefinierten
IC und werden als NaN dokumentiert, nicht künstlich zu null gesetzt.

Unsicherheit und Grenzen
------------------------
Validierungsresiduen werden pro Modell als OOS-Fehler gespeichert. Ihre
Kovarianz ist zunächst nur eine *historische Diagnose*, keine bewiesene
Konfidenzregion für die unbekannten bedingten Erwartungsrenditen. Eine robuste
Omega-Schätzung und ein Radius rho müssen in Modul 3 separat statistisch
begründet und ausschließlich mit historisch zulässigen Daten kalibriert
werden. Der stationäre Prozess wird nicht vorausgesetzt. Datenrevisionen,
ETF-Survivorship, strukturelle Brüche und kleine Querschnitte begrenzen die
Interpretation. Test-RMSE dient nur der unabhängigen Evaluation.

Eingaben
--------
CSV aus Modul 1: eine Paneldatei mit Spalten für Datum, ETF, acht Features
und Folgemonatsrendite; zusätzlich optional eine aktuelle Featuredatei.
Die Dateipfade sind an die im Modul-1-Konsolenbericht erkannten
Dateinamen angepasst; Spaltenzuordnung bleibt zentral konfigurierbar. Automatische
Dateierkennung wird bewusst nicht verwendet, um falsche CSVs zu vermeiden.

Ausgaben
--------
CSV mit Validierungs- und Testprognosen, Monatsmetriken,
Validierungsresiduen, Kandidatenranking, gewählten Modellen und aktuellen
Prognosen; JSON-Metadaten und Konsolen-Prüfbericht. Nach der
unabhängigen Evaluation wird ausschließlich das validierungsbasiert gewählte
Modell mit allen bis zum Live-Stichtag verfügbaren Labels final trainiert
und als joblib-Artefakt gespeichert. Dieses Artefakt ist niemals für den
historischen Backtest zulässig.

Reproduzierbarkeit: feste Seeds, deterministische Monatsaufteilung und
explizite Bibliotheksversionen. Rechenzeit: zahlreiche Neuschätzungen.
"""
from __future__ import annotations

import json
import hashlib
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.covariance import LedoitWolf

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit('XGBoost fehlt: pip install xgboost') from exc

# Pfade anhand des Modul-1-Exports; Spaltennamen bei Bedarf prüfen.
BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / 'data' / 'forecast_dataset_monthly.csv'
CURRENT_CSV = BASE_DIR / 'data' / 'latest_features_for_forecast.csv'
OUT_DIR = BASE_DIR / 'output' / 'module2'
DATE_COL = 'date'
ASSET_COL = 'asset'
TARGET_COL = 'target_return_next_month'
FEATURES = ['mom_1', 'mom_3', 'mom_6', 'mom_12',
            'vol_3', 'vol_6', 'vol_12', 'drawdown_12']
VALIDATION_START = '2018-01-31'
TEST_START = '2023-01-31'
MIN_TRAIN_MONTHS = 36
SEED = 42

# Kleine, vorab fixierte Suchräume; kein Test-Tuning.
CANDIDATES = {
    'HIST': [{}],
    'OLS': [{}],
    'RF': [{'n_estimators': 120, 'max_depth': d, 'min_samples_leaf': leaf}
           for d in [3, 6] for leaf in [5, 15]],
    'XGB': [{'n_estimators': 120, 'max_depth': d, 'learning_rate': lr,
             'subsample': 1.0, 'colsample_bytree': 1.0}
            for d in [2, 3] for lr in [0.03, 0.08]],
}


def load_panel(path: Path, *, with_target: bool) -> pd.DataFrame:
    """Lädt und validiert das monatliche ETF-Panel ohne stille Datenreparatur.

    Parameter: path (CSV-Datei), with_target (Label erforderlich?).
    Rückgabe: nach Datum/ETF sortiertes Panel mit Monatsend-Zeitstempeln.
    Fehler bei Duplikaten, fehlenden Werten oder unvollständigen Querschnitten.
    """
    if not path.exists():
        available = sorted(BASE_DIR.rglob('*.csv'))
        listing = '\n'.join(f'  - {p}' for p in available[:60]) or '  (keine CSV-Dateien gefunden)'
        raise FileNotFoundError(
            f'Erwartete Datei nicht gefunden: {path}\n'
            f'Aktuelles Arbeitsverzeichnis: {Path.cwd()}\n'
            f'Skriptverzeichnis: {BASE_DIR}\n'
            f'Gefundene CSV-Dateien im Projektordner:\n{listing}\n'
            'Bitte TRAIN_CSV oben auf den tatsächlichen Trainings-CSV-Export aus Modul 1 setzen. '
            'Die Datei wird absichtlich nicht automatisch ausgewählt.'
        )
    df = pd.read_csv(path)
    required = [DATE_COL, ASSET_COL, *FEATURES] + ([TARGET_COL] if with_target else [])
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f'{path}: fehlende Spalten {missing}; vorhanden: {list(df.columns)}')
    df = df[required].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='raise')
    if df[DATE_COL].isna().any():
        raise ValueError(f'{path}: leere Datumswerte')
    if not df[DATE_COL].dt.is_month_end.all():
        raise ValueError(f'{path}: Datum muss auf Monatsende liegen')
    if df.duplicated([DATE_COL, ASSET_COL]).any():
        raise ValueError(f'{path}: doppelte (Monat, ETF)-Kombinationen')
    for c in FEATURES + ([TARGET_COL] if with_target else []):
        df[c] = pd.to_numeric(df[c], errors='raise')
    if df[required].isna().any().any():
        raise ValueError(f'{path}: fehlende Werte in Pflichtspalten')
    if not np.isfinite(df[FEATURES + ([TARGET_COL] if with_target else [])].to_numpy()).all():
        raise ValueError(f'{path}: unendliche Werte')
    counts = df.groupby(DATE_COL)[ASSET_COL].nunique()
    if counts.nunique() != 1:
        raise ValueError(f'{path}: unterschiedlich viele ETFs je Monat')
    sets = df.groupby(DATE_COL)[ASSET_COL].apply(lambda x: frozenset(x))
    if sets.nunique() != 1:
        raise ValueError(f'{path}: ETF-Universum variiert zwischen Monaten')
    return df.sort_values([DATE_COL, ASSET_COL]).reset_index(drop=True)


def model_for(name: str, params: dict, assets: list[str]):
    """Erzeugt ein frisches Modell; alle Lernschritte werden pro Fold neu fitten.

    Die ETF-Dummies erlauben asset-spezifische Interzepte; unbekannte ETFs
    werden durch die Panelprüfungen bereits ausgeschlossen.
    """
    if name == 'HIST':
        return None
    transform = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler() if name == 'OLS' else 'passthrough')]), FEATURES),
        ('asset', OneHotEncoder(categories=[assets], handle_unknown='error', sparse_output=False), [ASSET_COL]),
    ])
    if name == 'OLS':
        estimator = LinearRegression()
    elif name == 'RF':
        estimator = RandomForestRegressor(**params, random_state=SEED, n_jobs=1)
    elif name == 'XGB':
        estimator = XGBRegressor(**params, objective='reg:squarederror', random_state=SEED,
                                 n_jobs=1, tree_method='hist', verbosity=0)
    else:
        raise ValueError(name)
    return Pipeline([('preprocess', transform), ('model', estimator)])


def forecast_one_month(history: pd.DataFrame, current: pd.DataFrame,
                       name: str, params: dict, assets: list[str]) -> np.ndarray:
    """Schätzt mit ausschließlich bekannten Labels und prognostiziert einen Monat.

    history: Zeilen mit Feature-Monat strikt vor dem aktuellen Feature-Monat.
    current: genau ein Monat, ein Eintrag je ETF. Rückgabe: Renditevektor
    in der Reihenfolge der current-Zeilen.
    """
    if history.empty or current.empty:
        raise ValueError('Leere Trainings- oder Prognosestichprobe')
    forecast_date = current[DATE_COL].min()
    if current[DATE_COL].nunique() != 1:
        raise AssertionError('Prognose darf nur einen Feature-Monat umfassen')
    if history[DATE_COL].max() >= forecast_date:
        raise AssertionError('LOOK-AHEAD: Trainingsmonat nicht vor Prognosemonat')
    # Zielrendite eines Feature-Monats s wird erst am Ende von s+1 bekannt.
    # Beim Forecast am Ende von t darf nur s+1 <= t verwendet werden.
    latest_label_available = history[DATE_COL] + pd.offsets.MonthEnd(1)
    if (latest_label_available > forecast_date).any():
        raise AssertionError('LOOK-AHEAD: noch nicht realisierte Zielrendite im Training')
    if history[DATE_COL].nunique() < MIN_TRAIN_MONTHS:
        raise ValueError('Zu wenige historische Monate')
    if name == 'HIST':
        means = history.groupby(ASSET_COL)[TARGET_COL].mean()
        return current[ASSET_COL].map(means).to_numpy(dtype=float)
    model = model_for(name, params, assets)
    model.fit(history[[*FEATURES, ASSET_COL]], history[TARGET_COL])
    return np.asarray(model.predict(current[[*FEATURES, ASSET_COL]]), dtype=float)


def expanding_predictions(panel: pd.DataFrame, months: pd.DatetimeIndex,
                          name: str, params: dict, assets: list[str]) -> pd.DataFrame:
    """Erstellt echte zeitlich sequenzielle OOS-Prognosen mit Neuschätzung.

    Jede Zeile t nutzt Trainingslabels nur aus Feature-Monaten s < t.
    In der Simulation ist das Label für s=t-1 am Ende von t bekannt.
    """
    outputs = []
    for t in months:
        history = panel.loc[panel[DATE_COL] < t]
        current = panel.loc[panel[DATE_COL] == t]
        pred = forecast_one_month(history, current, name, params, assets)
        if not np.isfinite(pred).all():
            raise AssertionError(f'Nichtendliche Prognose: {name}, {t}')
        outputs.append(pd.DataFrame({DATE_COL: current[DATE_COL].to_numpy(),
                                     ASSET_COL: current[ASSET_COL].to_numpy(),
                                     'actual': current[TARGET_COL].to_numpy(),
                                     'prediction': pred,
                                     'error': current[TARGET_COL].to_numpy() - pred,
                                     'model': name}))
    return pd.concat(outputs, ignore_index=True)


def metrics(pred: pd.DataFrame) -> dict:
    """Berechnet OOS-Fehler und mittlere monatliche Querschnitts-ICs.

    Pearson misst lineare, Spearman monotone Rangkorrelation. Beide können
    bei konstanten Querschnitten mathematisch undefiniert sein.
    """
    monthly = []
    for _, g in pred.groupby(DATE_COL):
        if g['actual'].nunique() < 2 or g['prediction'].nunique() < 2:
            monthly.append((np.nan, np.nan))
        else:
            monthly.append((g['actual'].corr(g['prediction'], method='pearson'),
                            g['actual'].corr(g['prediction'], method='spearman')))
    vals = np.asarray(monthly, dtype=float)
    return {'rmse': float(np.sqrt(mean_squared_error(pred['actual'], pred['prediction']))),
            'mae': float(mean_absolute_error(pred['actual'], pred['prediction'])),
            'pearson_ic': float(np.nanmean(vals[:, 0])) if np.isfinite(vals[:, 0]).any() else None,
            'spearman_ic': float(np.nanmean(vals[:, 1])) if np.isfinite(vals[:, 1]).any() else None,
            'valid_ic_months': int(np.isfinite(vals[:, 0]).sum()),
            'months': int(pred[DATE_COL].nunique())}



def audit_source_alignment(panel: pd.DataFrame) -> None:
    """Prüft Folgemonatslabels gegen unabhängig exportierte Monatspreise.

    Der Preisexport aus Modul 1 wird gelesen, sofern er vorhanden ist. Die
    Referenzrendite wird aus den Preisen neu berechnet und nach ETF und
    Feature-Monat verglichen. Ohne Preisexport ist die Prüfung unvollständig
    und das Programm bricht bewusst ab.

    Hinweis: Dies validiert die Label-Ausrichtung, nicht die Berechnung aller
    Features aus täglichen Rohdaten oder die Korrektheit des Datenanbieters.
    """
    prices_path = BASE_DIR / 'data' / 'prices_monthly.csv'
    if not prices_path.exists():
        raise FileNotFoundError(f'Für den unabhängigen Label-Audit fehlt {prices_path}')
    prices = pd.read_csv(prices_path)
    # Modul 1 exportiert Monatspreise im Wide-Format (date, SPY, EFA, ...).
    # Auch das Long-Format (date, asset, price) wird unterstützt.
    if 'date' not in prices.columns:
        raise ValueError(f'Datumsspalte im Preisexport fehlt: {list(prices.columns)}')
    if 'asset' not in prices.columns:
        expected_assets = sorted(panel[ASSET_COL].unique().tolist())
        observed_assets = sorted(c for c in prices.columns if c != 'date')
        if observed_assets != expected_assets:
            raise ValueError(
                'ETF-Spalten im Wide-Preisexport stimmen nicht mit dem Trainingspanel überein: '
                f'erwartet={expected_assets}, vorhanden={observed_assets}'
            )
        prices = prices.melt(id_vars='date', var_name='asset', value_name='price')
        print('LABEL-AUDIT: Wide-Format erkannt und intern in Long-Format umgewandelt')
    candidates = [c for c in ('price', 'adjusted_close', 'adj_close', 'close') if c in prices.columns]
    if len(candidates) != 1:
        raise ValueError(f'Preisspalte nicht eindeutig: {list(prices.columns)}; bitte manuell prüfen')
    pc = candidates[0]
    prices[pc] = pd.to_numeric(prices[pc], errors='raise')
    if prices[pc].isna().any() or not np.isfinite(prices[pc].to_numpy()).all() or (prices[pc] <= 0).any():
        raise ValueError('Preisexport enthält fehlende, nichtendliche oder nichtpositive Kurse')
    prices['date'] = pd.to_datetime(prices['date'])
    if prices.duplicated(['date', 'asset']).any():
        raise ValueError('Doppelte ETF-Monate im Preisexport')
    prices = prices.sort_values(['asset', 'date'])
    prices['recomputed_target'] = prices.groupby('asset')[pc].shift(-1) / prices[pc] - 1
    merged = panel[['date', 'asset', TARGET_COL]].merge(
        prices[['date', 'asset', 'recomputed_target']], on=['date', 'asset'],
        how='left', validate='one_to_one')
    if merged['recomputed_target'].isna().any():
        raise AssertionError('Nicht alle Folgemonatsrenditen aus Preisen rekonstruierbar')
    delta = (merged[TARGET_COL] - merged['recomputed_target']).abs()
    if not (delta <= 1e-8).all():
        raise AssertionError(f'Label-Ausrichtung inkonsistent: max. Abweichung {delta.max():.10g}')
    print(f'LABEL-AUDIT: {len(merged)} Renditen aus Monatspreisen verifiziert; max. Fehler {delta.max():.3g}')


def evaluate_paired_errors(testing: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    """Deskriptiver monatlicher Verlustvergleich gegenüber HIST.

    d_t ist die Differenz der mittleren quadrierten ETF-Prognosefehler
    zwischen dem gewählten Modell und HIST. Positives d_t bevorzugt HIST.
    Ein zweiseitiges 95%-Intervall wird per Moving-Block-Bootstrap (Blocklänge
    sechs Monate) berechnet. Es ist eine explorative Unsicherheitsanalyse,
    kein unabhängiger Bestätigungstest nach bereits erfolgter Testsichtung.
    """
    if selected_model == 'HIST':
        return pd.DataFrame()
    sub = testing[testing['model'].isin([selected_model, 'HIST'])].copy()
    monthly = sub.assign(squared_error=sub['error'] ** 2).groupby(
        ['date', 'model'])['squared_error'].mean().unstack()
    if monthly.isna().any().any() or len(monthly) < 12:
        raise AssertionError('Unvollständiger gepaarter Testvergleich')
    d = (monthly[selected_model] - monthly['HIST']).to_numpy()
    n, block, B = len(d), min(6, len(d)), 2000
    rng = np.random.default_rng(SEED)
    starts = np.arange(n - block + 1)
    samples = np.empty(B)
    for b in range(B):
        indices = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, start + block))
        samples[b] = d[np.array(indices[:n])].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    result = pd.DataFrame([{'selected_model': selected_model, 'benchmark': 'HIST',
        'months': n, 'mean_monthly_mse_difference': d.mean(),
        'block_bootstrap_ci_lower': low, 'block_bootstrap_ci_upper': high,
        'block_length_months': block, 'bootstrap_repetitions': B,
        'interpretation': 'Explorativ; Testperiode bereits betrachtet'}])
    print('GEPAARTER TESTVERGLEICH (explorativ):')
    print(result.to_string(index=False))
    return result


def file_sha256(path: Path) -> str:
    """SHA-256-Fingerabdruck zur Nachvollziehbarkeit des verwendeten CSV-Inputs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Führt Kandidatenvergleich, feste Auswahl, Test und Exporte aus."""
    import sklearn
    import xgboost
    print('=' * 74 + '\nMODULE 2 — FORECASTING & MODEL SELECTION\n' + '=' * 74)
    print(f'Skriptverzeichnis: {BASE_DIR}')
    print(f'Trainingsdatei: {TRAIN_CSV}')
    print(f'Aktuelle Features: {CURRENT_CSV}')
    panel = load_panel(TRAIN_CSV, with_target=True)
    print('\nTRAINING DATA VALIDATION')
    print('Shape:', panel.shape)
    print('Columns:', panel.columns.tolist())
    print('First rows:')
    print(panel.head().to_string(index=False))
    print('Missing values:')
    print(panel.isna().sum().to_string())
    if len(panel) != 2256:
        raise ValueError(f'Erwartet: 2256 Trainingszeilen aus Modul 1; gefunden: {len(panel)}')
    if panel[ASSET_COL].nunique() != 12:
        raise ValueError('Erwartet: 12 ETFs aus Modul 1')
    audit_source_alignment(panel)
    print('ZEITLICHER AUDIT: Folgemonatslabel erst ab Ende des Folgemonats verfügbar')
    assets = sorted(panel[ASSET_COL].unique().tolist())
    months = pd.DatetimeIndex(sorted(panel[DATE_COL].unique()))
    if len(months) != len(pd.date_range(months.min(), months.max(), freq='ME')):
        raise ValueError('Trainingspanel enthält fehlende Kalendermonate')
    val_months = months[(months >= pd.Timestamp(VALIDATION_START)) & (months < pd.Timestamp(TEST_START))]
    test_months = months[months >= pd.Timestamp(TEST_START)]
    if not len(val_months) or not len(test_months):
        raise ValueError('Validierungs- oder Testperiode leer; Startdaten anpassen, ohne Testergebnisse zu verwenden')
    if sum(months < val_months.min()) < MIN_TRAIN_MONTHS:
        raise ValueError('Unzureichende Vorlaufhistorie vor Validierungsbeginn')
    print(f'ETFs: {len(assets)} | Zeilen: {len(panel)} | Features: {len(FEATURES)}')
    print(f'Validierung: {val_months.min().date()} bis {val_months.max().date()} ({len(val_months)} Monate)')
    print(f'Test:         {test_months.min().date()} bis {test_months.max().date()} ({len(test_months)} Monate)')
    print('Modellwahl: ausschließlich Validierungs-RMSE; keine zufälligen Splits')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ranking, validation_frames, chosen = [], {}, {}
    for name, candidates in CANDIDATES.items():
        for k, params in enumerate(candidates):
            print(f'Validierung: {name} Kandidat {k + 1}/{len(candidates)}', flush=True)
            frame = expanding_predictions(panel, val_months, name, params, assets)
            score = metrics(frame)
            ranking.append({'model': name, 'candidate': k, 'params': json.dumps(params, sort_keys=True), **score})
            validation_frames[(name, k)] = frame
    rank_df = pd.DataFrame(ranking).sort_values(['model', 'rmse', 'candidate'])
    rank_df.to_csv(OUT_DIR / 'validation_candidate_ranking.csv', index=False)
    val_selected, test_selected, summary = [], [], []
    for name in CANDIDATES:
        winner = rank_df.loc[rank_df['model'] == name].iloc[0]
        k = int(winner['candidate'])
        params = CANDIDATES[name][k]
        chosen[name] = params
        val = validation_frames[(name, k)].copy()
        val['split'] = 'validation'
        val_selected.append(val)
        print(f'Test: {name} (fixierte Parameter {params})', flush=True)
        test = expanding_predictions(panel, test_months, name, params, assets)
        test['split'] = 'test'
        test_selected.append(test)
        summary.append({'model': name, 'params': json.dumps(params, sort_keys=True),
                        **{f'val_{key}': value for key, value in metrics(val).items()},
                        **{f'test_{key}': value for key, value in metrics(test).items()}})
    validation = pd.concat(val_selected, ignore_index=True)
    testing = pd.concat(test_selected, ignore_index=True)
    validation.to_csv(OUT_DIR / 'validation_oos_predictions.csv', index=False)
    testing.to_csv(OUT_DIR / 'test_oos_predictions.csv', index=False)
    validation[[DATE_COL, ASSET_COL, 'model', 'error']].to_csv(OUT_DIR / 'validation_oos_errors.csv', index=False)
    summary_df = pd.DataFrame(summary).sort_values('val_rmse')
    summary_df.to_csv(OUT_DIR / 'model_summary.csv', index=False)
    # Monatliche Prognosequalität transparent speichern (nicht zur Modellwahl).
    monthly_records = []
    for split_name, dataset in [('validation', validation), ('test', testing)]:
        for (model_name, month), group in dataset.groupby(['model', DATE_COL]):
            record = {'split': split_name, 'model': model_name, DATE_COL: month,
                      'rmse': float(np.sqrt(np.mean(group['error'].to_numpy() ** 2))),
                      'mae': float(np.mean(np.abs(group['error'].to_numpy()))),
                      'pearson_ic': np.nan, 'spearman_ic': np.nan}
            if group['actual'].nunique() > 1 and group['prediction'].nunique() > 1:
                record['pearson_ic'] = group['actual'].corr(group['prediction'], method='pearson')
                record['spearman_ic'] = group['actual'].corr(group['prediction'], method='spearman')
            monthly_records.append(record)
    pd.DataFrame(monthly_records).to_csv(OUT_DIR / 'monthly_forecast_diagnostics.csv', index=False)
    # Bestes Modell nach Validierungs-RMSE, nicht nach Testleistung.
    best_name = str(summary_df.iloc[0]['model'])
    print('\nERGEBNISSE (Test nur zur Evaluation)')
    print(summary_df[['model', 'val_rmse', 'test_rmse', 'val_mae', 'test_mae',
                      'val_pearson_ic', 'test_pearson_ic', 'val_spearman_ic',
                      'test_spearman_ic']].to_string(index=False))
    print(f'Gewähltes Prognosemodell (Validierung): {best_name}')
    print('Gültige IC-Monate (Validierung / Test):')
    print(summary_df[['model', 'val_valid_ic_months', 'test_valid_ic_months']].to_string(index=False))
    print('Hinweis: Testmetriken sind ausschließlich deskriptiv; keine nachträgliche Modellwahl.')
    paired = evaluate_paired_errors(testing, best_name)
    if not paired.empty:
        paired.to_csv(OUT_DIR / 'exploratory_paired_test_comparison.csv', index=False)
    # Einfache, nur diagnostische Fehlerkovarianz des gewählten Modells.
    residuals = validation.loc[validation['model'] == best_name].pivot(
        index=DATE_COL, columns=ASSET_COL, values='error').reindex(columns=assets)
    if residuals.isna().any().any():
        raise AssertionError('Unvollständige Validierungs-Fehlermatrix')
    empirical_cov = residuals.cov()
    empirical_cov.to_csv(OUT_DIR / 'validation_error_covariance_DIAGNOSTIC.csv')
    # Shrinkage stabilisiert die numerische Fehlerkovarianz; sie ist NICHT Omega.
    shrinkage = LedoitWolf().fit(residuals.to_numpy(dtype=float))
    pd.DataFrame(shrinkage.covariance_, index=assets, columns=assets).to_csv(
        OUT_DIR / 'validation_error_covariance_shrunk_DIAGNOSTIC.csv')
    eigenvalues = np.linalg.eigvalsh(shrinkage.covariance_)
    if eigenvalues.min() <= 0:
        raise AssertionError('Geschrumpfte Fehlerkovarianz nicht positiv definit')
    print(f'Fehlerkovarianz (Diagnose): Ledoit-Wolf-Shrinkage={shrinkage.shrinkage_:.4f}, '
          f'kleinster Eigenwert={eigenvalues.min():.8g}')
    print('ACHTUNG: Fehlerkovarianz ist keine kalibrierte Omega-Matrix.')
    # Finale produktive Schätzung: ERST NACH abgeschlossener, fixierter Evaluation.
    # Sie verwendet alle bis zum Prognosestichtag tatsächlich verfügbaren Labels.
    # Historische OOS-Prognosen bleiben unverändert und dürfen NICHT durch
    # Vorhersagen des finalen, auf späteren Daten trainierten Modells ersetzt werden.
    live_audit = {'status': 'not_created', 'selected_model': best_name}
    if CURRENT_CSV.exists():
        current = load_panel(CURRENT_CSV, with_target=False)
        latest = current.loc[current[DATE_COL] == current[DATE_COL].max()].copy()
        if sorted(latest[ASSET_COL].tolist()) != assets:
            raise ValueError('Aktuelle Feature-Datei enthält nicht exakt das Trainingsuniversum')
        forecast_date = latest[DATE_COL].min()
        if forecast_date <= panel[DATE_COL].max():
            raise ValueError('Aktuelle Features müssen nach dem letzten gelabelten Feature-Monat liegen')
        # Keine unbekannten Labels und keine zukünftigen Feature-Monate.
        eligible = panel.loc[
            (panel[DATE_COL] < forecast_date) &
            ((panel[DATE_COL] + pd.offsets.MonthEnd(1)) <= forecast_date)
        ].copy()
        if eligible.empty:
            raise ValueError('Keine zum Live-Stichtag verfügbaren Trainingslabels')
        if len(eligible) != len(panel):
            print('LIVE-AUDIT: Nicht verfügbare Trainingslabels wurden ausgeschlossen')
        # Alle Kandidaten zu Vergleichszwecken prognostizieren; allein die
        # validierungsbasierte Auswahl bestimmt den produktiven Forecast.
        for name, params in chosen.items():
            latest[f'forecast_{name}'] = forecast_one_month(eligible, latest, name, params, assets)
        latest[[DATE_COL, ASSET_COL, *[f'forecast_{n}' for n in chosen]]].to_csv(
            OUT_DIR / 'latest_forecasts.csv', index=False)
        # Das gewählte Modell exakt EINMAL auf der vollständigen zulässigen
        # Historie fitten und seine Live-Prognose mit der Pipeline vergleichen.
        selected_params = chosen[best_name]
        if best_name == 'HIST':
            final_artifact = {'model': 'HIST',
                              'means_by_asset': eligible.groupby(ASSET_COL)[TARGET_COL].mean().to_dict(),
                              'assets': assets}
            final_prediction = latest[ASSET_COL].map(final_artifact['means_by_asset']).to_numpy(dtype=float)
        else:
            final_artifact = model_for(best_name, selected_params, assets)
            final_artifact.fit(eligible[[*FEATURES, ASSET_COL]], eligible[TARGET_COL])
            final_prediction = np.asarray(
                final_artifact.predict(latest[[*FEATURES, ASSET_COL]]), dtype=float)
        if not np.allclose(final_prediction, latest[f'forecast_{best_name}'].to_numpy(),
                           rtol=1e-10, atol=1e-12):
            raise AssertionError('Finales Modell und unabhängiger Live-Forecast stimmen nicht überein')
        if not np.isfinite(final_prediction).all():
            raise AssertionError('Finale Live-Prognose ist nicht endlich')
        artifact_path = OUT_DIR / 'final_live_model_ONLY_NOT_FOR_BACKTEST.joblib'
        joblib.dump(final_artifact, artifact_path)
        pd.DataFrame({DATE_COL: latest[DATE_COL].to_numpy(),
                      ASSET_COL: latest[ASSET_COL].to_numpy(),
                      'model': best_name,
                      'forecast_return_next_month': final_prediction}).to_csv(
            OUT_DIR / 'final_live_forecast_ONLY_NOT_FOR_BACKTEST.csv', index=False)
        live_audit = {
            'status': 'trained', 'selected_model': best_name,
            'selected_params': selected_params,
            'feature_month': str(forecast_date.date()),
            'forecast_for_month': str((forecast_date + pd.offsets.MonthEnd(1)).date()),
            'first_training_feature_month': str(eligible[DATE_COL].min().date()),
            'last_training_feature_month': str(eligible[DATE_COL].max().date()),
            'training_rows': int(len(eligible)),
            'training_months': int(eligible[DATE_COL].nunique()),
            'model_artifact': artifact_path.name,
            'warning': 'Finales Live-Modell verwendet spätere Daten und ist für historische Backtests unzulässig.'
        }
        print(f'LIVE-FINAL: {best_name}, Feature-Stichtag {forecast_date.date()}, '
              f'Trainingsmonate {eligible[DATE_COL].nunique()}, Trainingszeilen {len(eligible)}')
        print(f'LIVE-FINAL: Prognose für {(forecast_date + pd.offsets.MonthEnd(1)).date()}')
        print('LIVE-FINAL: Modellartefakt gespeichert; strikt NICHT für Backtests verwenden')
    else:
        print(f'HINWEIS: aktuelle Featuredatei {CURRENT_CSV} fehlt; Live-Modell übersprungen.')
    metadata = {'train_csv': str(TRAIN_CSV), 'current_csv': str(CURRENT_CSV),
                'validation_start': VALIDATION_START, 'test_start': TEST_START,
                'assets': assets, 'features': FEATURES, 'seed': SEED,
                 'input_sha256': {'training': file_sha256(TRAIN_CSV),
                                  'current': file_sha256(CURRENT_CSV) if CURRENT_CSV.exists() else None,
                                  'monthly_prices': file_sha256(BASE_DIR / 'data' / 'prices_monthly.csv')},
                'selected_model': best_name, 'selected_params': chosen,
                'final_live_model': live_audit,
                'evaluation_protocol': {
                    'selection': 'validation RMSE only',
                    'test': 'independent chronological OOS evaluation; already inspected; no retuning',
                    'live': 'refit selected model using all labels available at feature month',
                    'backtest': 'use only historical expanding-window OOS predictions, never final live model'
                },
                'versions': {'numpy': np.__version__, 'pandas': pd.__version__,
                             'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__},
                'label_availability_rule': 'Label for feature month s available at end of s+1',
                'warning': 'Empirische und Ledoit-Wolf-Fehlerkovarianz sind Diagnosen, keine kalibrierte Omega-Matrix. Testperiode nicht zum Tuning verwenden.'}
    (OUT_DIR / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    assert validation.groupby('model').size().eq(len(val_months) * len(assets)).all()
    assert testing.groupby('model').size().eq(len(test_months) * len(assets)).all()
    assert validation[DATE_COL].max() < testing[DATE_COL].min()
    if not (validation['actual'] - validation['prediction'] - validation['error']).abs().lt(1e-12).all():
        raise AssertionError('Validierungsresiduen inkonsistent')
    if not (testing['actual'] - testing['prediction'] - testing['error']).abs().lt(1e-12).all():
        raise AssertionError('Testresiduen inkonsistent')
    for fname in ['validation_oos_predictions.csv', 'test_oos_predictions.csv',
                  'model_summary.csv', 'monthly_forecast_diagnostics.csv', 'metadata.json']:
        if not (OUT_DIR / fname).is_file() or (OUT_DIR / fname).stat().st_size == 0:
            raise AssertionError(f'Fehlender oder leerer Export: {fname}')
    if live_audit['status'] == 'trained':
        for fname in ['final_live_model_ONLY_NOT_FOR_BACKTEST.joblib',
                      'final_live_forecast_ONLY_NOT_FOR_BACKTEST.csv']:
            if not (OUT_DIR / fname).is_file() or (OUT_DIR / fname).stat().st_size == 0:
                raise AssertionError(f'Fehlender oder leerer finaler Live-Export: {fname}')
    print(f'Exports: {OUT_DIR.resolve()}\nchecks_passed: True')


if __name__ == '__main__':
    warnings.filterwarnings('error', category=RuntimeWarning)
    main()
