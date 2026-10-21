"""Modul 5 – Wissenschaftliche Diagnostik, Grenzfallprüfung und Sensitivitätsanalyse.

Dieses Modul liest die Ergebnisse aus Modul 4 und untersucht, ob die Rangfolge
von EQ, MV, ELL, BS und STOCH gegenüber Änderungen vorab definierter
Modellannahmen stabil bleibt. Es lädt die nummerierten Module 03 und 04,
ohne ihre GitHub-Dateinamen umzubenennen.

WICHTIGE WISSENSCHAFTLICHE ABGRENZUNG
-----------------------------------
Die hier ausgeführten Parameter- und Restriktionsvariationen sind
SENSITIVITÄTSANALYSEN, KEINE historisch verschachtelte Kalibrierung.
Es wird ausdrücklich KEIN bester Parametersatz anhand der Ergebnisse von
2023–2026 ausgewählt und als unabhängig getestet ausgegeben. Die in
Modul 03 hinterlegten Parameter bleiben Demonstrationswerte. Für eine
wirkliche OOS-Kalibrierung ist eine getrennte, zeitlich verschachtelte
Trainings-/Validierungslogik erforderlich.

Sämtliche Renditen sind nach den im vereinfachten Monatsend-Backtest
modellierten Handelskosten. Die Ergebnisse sind keine exakt handelbare
Simulation. Ein Sharpe-Ratio-Vergleich auf 45 Monaten hat erhebliche
Stichprobenunsicherheit; p-Werte werden bewusst nicht behauptet.

AUFRUF
------
    python 05_robustness_analysis.py
    python 05_robustness_analysis.py --prices data/prices_monthly.csv
    python 05_robustness_analysis.py --quick

Erforderlich: 03_portfolio_optimization.py und 04_walk_forward_backtest.py
im gleichen Verzeichnis sowie data/prices_monthly.csv.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def _load_numbered_module(filename: str, internal_name: str):
    """Lädt eine nummerierte Python-Datei mit korrekter sys.modules-Registrierung.

    Python erlaubt Dateinamen wie ``04_walk_forward_backtest.py``, aber nicht
    ``from 04_walk_forward_backtest import ...``. Deshalb wird das Modul
    explizit über den Dateipfad geladen. Die Registrierung vor exec_module
    stellt u. a. die korrekte Auflösung von Dataclasses sicher.
    """
    path = ROOT / filename
    if not path.is_file():
        raise FileNotFoundError(f"Benötigtes Modul fehlt: {path}")
    spec = importlib.util.spec_from_file_location(internal_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Kann Modul nicht laden: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[internal_name] = module
    spec.loader.exec_module(module)
    return module


# Modul 04 lädt Modul 03 bereits selbst. Dadurch wird keine unnummerierte
# Kopie von Modul 03 benötigt.
backtest = _load_numbered_module("04_walk_forward_backtest.py", "research_backtest04")
METHODS = backtest.METHODS
OptimizationConfig = backtest.OptimizationConfig


def _read_baseline(folder: Path):
    """Liest und validiert die von Modul 04 erzeugten Ergebnisdateien.

    Die Monatsrenditen müssen genau denselben Monatsindex und dieselben
    Strategien wie die NAV-Tabelle besitzen. NAV hat zusätzlich den
    Anfangswert. Unvollständige Dateien führen zu einem Fehler statt zu
    stillschweigend falschen Vergleichen.
    """
    paths = {name: folder / name for name in
             ("monthly_net_returns.csv", "nav_eur.csv", "backtest_metrics.csv")}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Fehlende Backtest-Ergebnisse: " + ", ".join(missing))
    returns = pd.read_csv(paths["monthly_net_returns.csv"], index_col=0, parse_dates=True)
    nav = pd.read_csv(paths["nav_eur.csv"], index_col=0, parse_dates=True)
    metrics = pd.read_csv(paths["backtest_metrics.csv"])
    for method in METHODS:
        if method not in returns or method not in nav:
            raise ValueError(f"Strategie {method} fehlt in Modul-4-Ergebnissen")
    if len(nav) != len(returns) + 1 or not nav.index[1:].equals(returns.index):
        raise ValueError("NAV und Monatsrenditen haben unterschiedliche Zeitachsen")
    if not np.isfinite(returns[list(METHODS)].to_numpy()).all():
        raise ValueError("Nichtendliche Monatsrenditen gefunden")
    if not set(METHODS).issubset(set(metrics["method"])):
        raise ValueError("Nicht alle Methoden sind in backtest_metrics.csv enthalten")
    return returns, nav, metrics


def _diagnostics(returns: pd.DataFrame, nav: pd.DataFrame,
                 metrics: pd.DataFrame) -> pd.DataFrame:
    r"""Erstellt paarweise Performance- und Risikodiagnostik gegen MV.

    Für Strategie m und MV gilt die monatliche Renditedifferenz
        d_t = r_{m,t} - r_{MV,t}.
    ``mean_active_monthly_return`` ist das Stichprobenmittel von d_t.
    ``tracking_error_annualized`` ist std(d_t, ddof=1)*sqrt(12).
    ``information_ratio`` ist mean(d_t)*12 / tracking_error_annualized.
    Die Kennzahlen sind rein deskriptiv; insbesondere wird keine
    statistische Signifikanz aus 45 Beobachtungen abgeleitet.

    Der empirische Verlust-CVaR wird als POSITIVER Verlustbetrag definiert:
        L_t = -r_t
        CVaR_alpha(L) = Mittel der schlechtesten 1-alpha Verlustanteile.
    Bei durchgehend positiven Renditen kann dieser CVaR dennoch negativ
    sein, weil auch die schlechtesten Monatsverluste Gewinne sein können.
    """
    records = []
    mv = returns["MV"].to_numpy(float)
    for method in METHODS:
        r = returns[method].to_numpy(float)
        d = r - mv
        te = float(np.std(d, ddof=1) * np.sqrt(12))
        active = float(np.mean(d))
        ann_return = float(metrics.set_index("method").loc[method, "annualized_net_return"])
        records.append({
            "method": method,
            "annualized_net_return": ann_return,
            "mean_active_monthly_return_vs_mv": active,
            "tracking_error_annualized_vs_mv": te,
            "information_ratio_vs_mv": (active * 12 / te if te > 1e-12 else np.nan),
            "positive_month_fraction": float(np.mean(r > 0)),
            "worst_month": float(np.min(r)),
            "best_month": float(np.max(r)),
            "correlation_with_mv": (float(np.corrcoef(r, mv)[0, 1])
                                    if np.std(r) > 0 and np.std(mv) > 0 else np.nan),
            "nav_reconciliation_max_abs_error": float(np.max(np.abs(
                nav[method].to_numpy()[1:] /
                nav[method].to_numpy()[:-1] - 1 - r))),
        })
    result = pd.DataFrame(records)
    if result["nav_reconciliation_max_abs_error"].max() > 1e-7:
        raise ValueError("Monatsrenditen und NAV-Rechnung stimmen nicht überein")
    return result


def _cases(quick: bool):
    r"""Erstellt ein transparentes One-Factor-at-a-Time-Sensitivitätsraster.

    Jede Variante ändert genau eine Größe gegenüber der Baseline.
    Dies ist ausdrücklich KEINE Parameteroptimierung: Die Ergebnisse
    dürfen nicht zur nachträglichen Auswahl des besten Modells für dieselbe
    Evaluierungsperiode verwendet werden. Die Kombinationen sind nicht
    unabhängig und dürfen nicht als unabhängige Experimente gelten.

    Grenzfälle rho=0 und Gamma=0 werden separat numerisch auditiert.
    """
    cases = [("baseline", {}, 0.002)]
    if quick:
        return cases + [
            ("ell_rho_0", {"rho": 0.0}, 0.002),
            ("bs_gamma_0", {"gamma_budget": 0.0}, 0.002),
            ("fee_50bp", {}, 0.005),
        ]

    cases += [
        ("fee_0bp", {}, 0.0),
        ("fee_10bp", {}, 0.001),
        ("fee_50bp", {}, 0.005),
        ("fee_100bp", {}, 0.01),
        ("max_weight_20pct", {"max_weight": 0.20}, 0.002),
        ("max_weight_40pct", {"max_weight": 0.40}, 0.002),
        ("max_weight_60pct", {"max_weight": 0.60}, 0.002),
        ("max_weight_100pct", {"max_weight": 1.0}, 0.002),
        ("turnover_25pct", {"turnover_limit": 0.25}, 0.002),
        ("turnover_75pct", {"turnover_limit": 0.75}, 0.002),
        ("turnover_100pct", {"turnover_limit": 1.0}, 0.002),
        ("covariance_24m", {"covariance_window": 24}, 0.002),
        ("covariance_60m", {"covariance_window": 60}, 0.002),
        ("lambda_05", {"lambda_risk": 0.5}, 0.002),
        ("lambda_1", {"lambda_risk": 1.0}, 0.002),
        ("lambda_2", {"lambda_risk": 2.0}, 0.002),
        ("lambda_5", {"lambda_risk": 5.0}, 0.002),
        ("lambda_8", {"lambda_risk": 8.0}, 0.002),
        ("lambda_12", {"lambda_risk": 12.0}, 0.002),
        ("ell_rho_0", {"rho": 0.0}, 0.002),
        ("ell_rho_025", {"rho": 0.25}, 0.002),
        ("ell_rho_05", {"rho": 0.5}, 0.002),
        ("ell_rho_15", {"rho": 1.5}, 0.002),
        ("ell_rho_2", {"rho": 2.0}, 0.002),
        ("ell_rho_3", {"rho": 3.0}, 0.002),
        ("bs_gamma_0", {"gamma_budget": 0.0}, 0.002),
        ("bs_gamma_1", {"gamma_budget": 1.0}, 0.002),
        ("bs_gamma_2", {"gamma_budget": 2.0}, 0.002),
        ("bs_gamma_4", {"gamma_budget": 4.0}, 0.002),
        ("bs_gamma_6", {"gamma_budget": 6.0}, 0.002),
        ("bs_gamma_9", {"gamma_budget": 9.0}, 0.002),
        ("bs_gamma_12", {"gamma_budget": 12.0}, 0.002),
        ("bs_kappa_05", {"kappa": 0.5}, 0.002),
        ("bs_kappa_15", {"kappa": 1.5}, 0.002),
        ("bs_kappa_2", {"kappa": 2.0}, 0.002),
        ("bs_kappa_3", {"kappa": 3.0}, 0.002),
        ("stoch_gamma_025", {"gamma_cvar": 0.25}, 0.002),
        ("stoch_gamma_05", {"gamma_cvar": 0.5}, 0.002),
        ("stoch_gamma_15", {"gamma_cvar": 1.5}, 0.002),
        ("stoch_gamma_2", {"gamma_cvar": 2.0}, 0.002),
        ("stoch_gamma_3", {"gamma_cvar": 3.0}, 0.002),
        ("stoch_alpha_95", {"alpha": 0.95}, 0.002),
        ("stoch_scenarios_60m", {"scenario_window": 60}, 0.002),
        ("stoch_scenarios_84m", {"scenario_window": 84}, 0.002),
        ("hac_lags_0", {"hac_lags": 0}, 0.002),
        ("hac_lags_1", {"hac_lags": 1}, 0.002),
        ("hac_lags_6", {"hac_lags": 6}, 0.002),
        ("hac_lags_12", {"hac_lags": 12}, 0.002),
    ]
    return cases


def _equivalence_audit(scenarios_folder: Path) -> pd.DataFrame:
    r"""Prüft die beiden exakten mathematischen Reduktionsfälle.

    Bei gleichen Daten, identischer Risikoaversion und identischen
    Handelsrestriktionen muss ELL(rho=0) = MV gelten. Ebenso muss
    BS(Gamma=0) = MV gelten. Wir vergleichen drei Ebenen:

      1. Zielgewichte für jeden ETF in jedem Evaluierungsmonat;
      2. realisierte monatliche Nettorenditen;
      3. gesamten Vermögensverlauf einschließlich Anfangskapital.

    Für Gewichte und Renditen gelten absolute numerische Toleranzen,
    für NAV eine Euro-Toleranz. Bei nicht eindeutigen optimalen Gewichten
    können unterschiedliche Solver-Lösungen dieselbe Zielfunktion
    erreichen; ein Fehlbefund ist dann zu diagnostizieren, nicht zu
    ignorieren. Die Funktion schreibt die maximalen Abweichungen aus.
    """
    comparisons = [
        ("ELL_rho_0_vs_MV", "ell_rho_0", "ELL"),
        ("BS_Gamma_0_vs_MV", "bs_gamma_0", "BS"),
    ]
    limits = {"weights": 1e-6, "returns": 1e-8, "nav_eur": 0.001}
    rows = []
    base = scenarios_folder / "baseline"
    for label, scenario, method in comparisons:
        folder = scenarios_folder / scenario
        for required in ("monthly_weights.csv", "monthly_net_returns.csv", "nav_eur.csv"):
            if not (base / required).is_file() or not (folder / required).is_file():
                raise FileNotFoundError(f"Für Grenzfallprüfung fehlt {required} in {base} oder {folder}")
        wb = pd.read_csv(base / "monthly_weights.csv")
        wx = pd.read_csv(folder / "monthly_weights.csv")
        b = wb.loc[wb.method == "MV", ["date", "asset", "weight"]].rename(columns={"weight": "baseline"})
        x = wx.loc[wx.method == method, ["date", "asset", "weight"]].rename(columns={"weight": "case"})
        joined = b.merge(x, on=["date", "asset"], how="outer", validate="one_to_one", indicator=True)
        if not (joined["_merge"] == "both").all() or joined.empty:
            raise ValueError(f"Grenzfall {label}: Gewichts-Zeitachsen stimmen nicht überein")
        weight_error = float((joined["baseline"] - joined["case"]).abs().max())
        rb = pd.read_csv(base / "monthly_net_returns.csv", index_col=0, parse_dates=True)
        rx = pd.read_csv(folder / "monthly_net_returns.csv", index_col=0, parse_dates=True)
        nb = pd.read_csv(base / "nav_eur.csv", index_col=0, parse_dates=True)
        nx = pd.read_csv(folder / "nav_eur.csv", index_col=0, parse_dates=True)
        if not rb.index.equals(rx.index) or not nb.index.equals(nx.index):
            raise ValueError(f"Grenzfall {label}: Rendite-/NAV-Zeitachsen weichen ab")
        return_error = float((rb["MV"] - rx[method]).abs().max())
        nav_error = float((nb["MV"] - nx[method]).abs().max())
        passed = (weight_error <= limits["weights"] and
                  return_error <= limits["returns"] and
                  nav_error <= limits["nav_eur"])
        rows.append({"comparison": label, "status": "PASS" if passed else "FAIL",
                     "max_abs_weight_difference": weight_error,
                     "max_abs_monthly_return_difference": return_error,
                     "max_abs_nav_difference_eur": nav_error,
                     "weight_tolerance": limits["weights"],
                     "return_tolerance": limits["returns"],
                     "nav_tolerance_eur": limits["nav_eur"]})
    return pd.DataFrame(rows)


def _parameter_group(scenario: str) -> tuple[str, str]:
    """Ordnet Szenarien ihrer veränderten Annahme zu (ohne Performancewertung)."""
    if scenario == "baseline":
        return "00 BASIS", "Standardparameter"
    groups = [
        ("fee_", "01 TRANSAKTIONSKOSTEN", "Handelsgebühr"),
        ("max_weight_", "02 KONZENTRATION", "Max. ETF-Gewicht"),
        ("turnover_", "03 UMSCHICHTUNG", "L1-Turnover-Limit"),
        ("covariance_", "04 KOVARIANZFENSTER", "Kovarianzfenster"),
        ("lambda_", "05 RISIKOAVERSION", "Lambda (MV/ELL/BS)"),
        ("ell_rho_", "06 ELLIPSOIDALE ROBUSTHEIT", "ELL-Radius rho"),
        ("bs_gamma_", "07 BS-BUDGET", "BS-Gamma"),
        ("bs_kappa_", "08 BS-UNSICHERHEIT", "BS-Kappa"),
        ("stoch_gamma_", "09 STOCH-RISIKOGEWICHT", "STOCH-Gamma"),
        ("stoch_alpha_", "10 CVAR-NIVEAU", "STOCH-Alpha"),
        ("stoch_scenarios_", "11 SZENARIENFENSTER", "STOCH-Szenariofenster"),
        ("hac_lags_", "12 HAC-SCHAETZUNG", "HAC-Lags"),
    ]
    for prefix, group, label in groups:
        if scenario.startswith(prefix):
            return group, label
    return "99 SONSTIGES", "Sonstiger Parameter"


def _scenario_value(row: pd.Series) -> str:
    """Erzeugt lesbare Parameterwerte aus den tatsächlich verwendeten Einstellungen."""
    name = str(row["scenario"])
    if name == "baseline":
        return "Basis"
    if name.startswith("fee_"):
        return f'{float(row["fee_rate"])*100:.2f} %'
    if name.startswith("max_weight_"):
        return f'{float(row["max_weight"])*100:.0f} %'
    if name.startswith("turnover_"):
        return f'{float(row["turnover_limit"]):.2f}'
    if name.startswith("covariance_"):
        return f'{int(row["covariance_window"])} Monate'
    if name.startswith("lambda_"):
        return f'{float(row["lambda_risk"]):g}'
    if name.startswith("ell_rho_"):
        return f'{float(row["rho"]):g}'
    if name.startswith("bs_gamma_"):
        return f'{float(row["gamma_budget"]):g}'
    if name.startswith("bs_kappa_"):
        return f'{float(row["kappa"]):g}'
    if name.startswith("stoch_gamma_"):
        return f'{float(row["gamma_cvar"]):g}'
    if name.startswith("stoch_alpha_"):
        return f'{float(row["alpha"])*100:.0f} %'
    if name.startswith("stoch_scenarios_"):
        return f'{int(row["scenario_window"])} Monate'
    if name.startswith("hac_lags_"):
        return f'{int(row["hac_lags"])}'
    return name


def _presentation_tables(sensitivity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Erstellt je Szenario reale KPI-Werte und unabhängige Kennzahl-Rangplätze.

    Höher ist besser: Nettorendite, Sharpe, Endvermögen.
    Niedriger ist besser: Volatilität, CVaR-Verlustbetrag, Handelskosten.
    Beim Maximum Drawdown ist der größere (weniger negative) Wert besser.
    Es wird KEIN willkürlicher aggregierter Gesamtrang berechnet.
    """
    data = sensitivity.copy()
    order = list(dict.fromkeys(data["scenario"].tolist()))
    data["scenario"] = pd.Categorical(data["scenario"], categories=order, ordered=True)
    # Keine Series.map(tuple) auf Categorical: Pandas kann daraus einen
    # MultiIndex erzeugen, der sich nicht als Series initialisieren laesst.
    groups = [_parameter_group(str(name)) for name in data["scenario"].astype(str).tolist()]
    data["parameter_group"] = [group for group, _ in groups]
    data["parameter_name"] = [name for _, name in groups]
    data["parameter_value"] = data.apply(_scenario_value, axis=1)
    columns = {
        "return": ("annualized_net_return", False),
        "volatility": ("annualized_volatility", True),
        "sharpe": ("sharpe_rf_zero", False),
        "drawdown": ("max_drawdown", False),
        "cvar": ("realized_monthly_loss_cvar", False),
        "nav": ("final_nav_eur", False),
        "fees": ("total_fees_eur", True),
    }
    # Rundungs-/Solver-Rauschen darf keine künstlichen Sieger erzeugen.
    # Toleranzen gelten in den NATIVEN Einheiten der jeweiligen KPI.
    # Rang = 1 + Anzahl eindeutig besserer Modelle; tolerierte Gleichstände
    # bekommen denselben Rang. Kein Performance-Tuning dieser Toleranzen.
    tie_tolerance = {
        "return": 1e-5,       # 0,001 Prozentpunkte annualisiert
        "volatility": 1e-5,
        "sharpe": 1e-4,
        "drawdown": 1e-5,
        "cvar": 1e-5,
        "nav": 0.25,         # 25 Cent bei EUR 10.000 Startkapital
        "fees": 0.10,        # 10 Cent Handelskosten
    }
    for key, (source, ascending) in columns.items():
        tol = tie_tolerance[key]
        def rank_with_ties(group):
            vals = group.to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"Nichtendlicher KPI: {source}")
            ranks = [1 + int(np.sum((vals < v - tol) if ascending
                                    else (vals > v + tol))) for v in vals]
            return pd.Series(ranks, index=group.index, dtype=int)
        data[f"rank_{key}"] = data.groupby("scenario", observed=True)[source].transform(rank_with_ties)
    data = data.sort_values(["parameter_group", "scenario", "rank_return", "method"], kind="stable")
    winners = []
    for scenario, frame in data.groupby("scenario", sort=False, observed=True):
        first = frame.iloc[0]
        row = {"scenario": str(scenario), "parameter_group": first["parameter_group"],
               "parameter_name": first["parameter_name"], "parameter_value": first["parameter_value"]}
        for key in columns:
            matched = frame.loc[frame[f"rank_{key}"] == 1, "method"]
            row[f"winner_{key}"] = " = ".join(sorted(matched.astype(str)))
        winners.append(row)
    return data, pd.DataFrame(winners)


def _de(value: float, decimals: int = 2) -> str:
    """Formatiert Dezimalzahlen für deutsche Konsolen- und Markdown-Tabellen."""
    return f"{float(value):,.{decimals}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def _kpi_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Stellt absolute Backtest-Kennzahlen pro Modell ohne Rang-Vermischung dar."""
    frame = frame.sort_values(["rank_return", "method"], kind="stable")
    return pd.DataFrame({
        "Modell": frame["method"].astype(str).to_numpy(),
        "Rendite p.a.": [_de(x*100)+" %" for x in frame["annualized_net_return"]],
        "Vola p.a.": [_de(x*100)+" %" for x in frame["annualized_volatility"]],
        "Sharpe": [_de(x, 3) for x in frame["sharpe_rf_zero"]],
        "Max. DD": [_de(x*100)+" %" for x in frame["max_drawdown"]],
        "CVaR mtl.": [_de(x*100)+" %" for x in frame["realized_monthly_loss_cvar"]],
        "Endwert EUR": [_de(x) for x in frame["final_nav_eur"]],
        "Kosten EUR": [_de(x) for x in frame["total_fees_eur"]],
    })


def _rank_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Zeigt die Rangfolge je KPI; 1 ist stets die vorteilhafte Richtung."""
    frame = frame.sort_values(["rank_return", "method"], kind="stable")
    return pd.DataFrame({
        "Modell": frame["method"].astype(str).to_numpy(),
        "Rendite": frame["rank_return"].to_numpy(),
        "Vola": frame["rank_volatility"].to_numpy(),
        "Sharpe": frame["rank_sharpe"].to_numpy(),
        "Max. DD": frame["rank_drawdown"].to_numpy(),
        "CVaR": frame["rank_cvar"].to_numpy(),
        "Endwert": frame["rank_nav"].to_numpy(),
        "Kosten": frame["rank_fees"].to_numpy(),
    })


def _markdown_table(frame: pd.DataFrame) -> str:
    """Erzeugt eine direkt in GitHub lesbare Markdown-Tabelle ohne Zusatzpakete."""
    cols = list(frame.columns)
    rows = ["| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, record in frame.iterrows():
        rows.append("| " + " | ".join(str(record[c]) for c in cols) + " |")
    return "\n".join(rows)


def _print_readable_results(detail: pd.DataFrame, summary: pd.DataFrame,
                            audit: pd.DataFrame, output: Path) -> None:
    """Druckt und exportiert je Parameterfall KPI-Matrix UND separate Rangmatrix.

    Die Reihenfolge der Modelle orientiert sich an der Rendite, die Rangmatrix
    bewertet aber jede Kennzahl unabhängig. Kein unzulässiger Gesamtscore.
    """
    print("\n" + "=" * 100)
    print("MODUL 5 | KPI-MATRIX UND RANGMATRIX JE PARAMETER")
    print("=" * 100)
    print("Rang 1 = bester KPI-Wert. Keine historische Parameterkalibrierung.")
    print("\nMATHEMATISCHE GRENZFALLPRUEFUNG")
    print(audit.to_string(index=False, float_format=lambda v: f"{v:.7g}"))
    if (audit["status"] == "FAIL").any():
        print("WARNUNG: Mindestens ein mathematischer Grenzfall ist FEHLGESCHLAGEN!")
    report = ["# Modul 5 – KPI- und Rangmatrix je Parameter", "",
              "*Ex-post-Sensitivität, keine unabhängig validierte Parameterkalibrierung.*", "",
              "**Rang 1:** höchste Rendite/Sharpe/Endwert, niedrigste Volatilität/Kosten, "
              "geringster Verlust bei Drawdown und CVaR. Kein Gesamtscore.", ""]
    for group_name, group in detail.groupby("parameter_group", sort=False, observed=True):
        print("\n" + "=" * 100 + "\n" + str(group_name) + "\n" + "=" * 100)
        report.extend([f"## {group_name}", ""])
        for scenario, frame in group.groupby("scenario", sort=False, observed=True):
            first = frame.iloc[0]
            result = summary.loc[summary["scenario"] == str(scenario)].iloc[0]
            heading = ("Parameter: Baseline – Standardkonfiguration" if str(scenario) == "baseline"
                       else f'Parameter: {first["parameter_name"]} = {first["parameter_value"]}')
            winners = (f'Rendite: {result["winner_return"]} | Sharpe: {result["winner_sharpe"]} | '
                       f'Vola: {result["winner_volatility"]} | Max. DD: {result["winner_drawdown"]} | '
                       f'CVaR: {result["winner_cvar"]}')
            kpis = _markdown_table(_kpi_matrix(frame))
            ranks = _markdown_table(_rank_matrix(frame))
            print(f"\n{heading}\nKPI-Sieger – {winners}\n")
            print("A) TATSAECHLICHE KPI-WERTE\n" + kpis)
            print("\nB) RANGMATRIX (1 = BESTER WERT)\n" + ranks)
            report.extend([f"### {heading}", "", f"**KPI-Sieger:** {winners}", "",
                           "**A) Tatsächliche KPI-Werte**", "", kpis, "",
                           "**B) Rangmatrix (1 = bester Wert)**", "", ranks, ""])
    print("\n" + "=" * 100 + "\nGESAMTUEBERSICHT – KPI-SIEGER JE PARAMETER\n" + "=" * 100)
    cols = ["parameter_name", "parameter_value", "winner_return", "winner_sharpe",
            "winner_volatility", "winner_drawdown", "winner_cvar"]
    print(summary[cols].to_string(index=False))
    output.joinpath("kpi_und_rangmatrix_je_parameter.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8")
    print(f"\nMarkdown-Bericht: {(output / 'kpi_und_rangmatrix_je_parameter.md').resolve()}")

def run_analysis(prices: Path, output: Path, start: str = "2023-01-31",
                 quick: bool = False):
    """Berechnet Baseline-Diagnostik und getrennte Sensitivitäts-Backtests.

    Für jeden Sensitivitätsfall wird Modul 04 mit einer frischen
    OptimizationConfig ausgeführt. Die Konten aller fünf Strategien werden
    jeweils mit demselben Kapital initialisiert; keine Gewichte werden
    zwischen den Szenarien übernommen. Die historischen Prognosefenster
    werden ausschließlich mit Daten vor dem jeweiligen Renditemonat
    geschätzt. Die Sensitivitätsfälle werden NICHT als neue
    Optimierungsparameter in den Haupt-Backtest zurückgeschrieben.

    Schreibt:
      baseline_diagnostics.csv, sensitivity_metrics.csv,
      sensitivity_vs_baseline.csv, sensitivity_rankings.csv.
    Außerdem enthält ``scenarios/`` sämtliche Modul-4-Detaildateien.
    """
    output.mkdir(parents=True, exist_ok=True)
    base_cfg = OptimizationConfig()
    all_metrics = []
    baseline_data = None
    for name, changes, fee_rate in _cases(quick):
        cfg = replace(base_cfg, **changes)
        scenario_folder = output / "scenarios" / name
        print(f"Berechne {name} ...", flush=True)
        with redirect_stdout(StringIO()):
            metrics, nav, ret, _, _ = backtest.run_backtest(
                prices, scenario_folder, start=start, fee_rate=fee_rate, cfg=cfg
            )
        if baseline_data is None:
            baseline_data = (ret, nav, metrics)
        part = metrics.copy()
        part.insert(0, "scenario", name)
        part["fee_rate"] = fee_rate
        part["max_weight"] = cfg.max_weight
        part["lambda_risk"] = cfg.lambda_risk
        part["rho"] = cfg.rho
        part["gamma_budget"] = cfg.gamma_budget
        part["kappa"] = cfg.kappa
        part["gamma_cvar"] = cfg.gamma_cvar
        part["alpha"] = cfg.alpha
        part["hac_lags"] = cfg.hac_lags
        part["turnover_limit"] = cfg.turnover_limit
        part["covariance_window"] = cfg.covariance_window
        part["scenario_window"] = cfg.scenario_window
        all_metrics.append(part)
    audit = _equivalence_audit(output / "scenarios")
    audit.to_csv(output / "mathematical_equivalence_audit.csv", index=False)
    ret, nav, metrics = baseline_data
    diagnostics = _diagnostics(ret, nav, metrics)
    sensitivity = pd.concat(all_metrics, ignore_index=True)
    baseline = sensitivity[sensitivity.scenario == "baseline"].set_index("method")
    delta_rows = []
    for _, row in sensitivity.iterrows():
        reference = baseline.loc[row["method"]]
        delta_rows.append({
            "scenario": row["scenario"], "method": row["method"],
            "delta_annualized_return_pp": 100 * (
                row["annualized_net_return"] - reference["annualized_net_return"]),
            "delta_volatility_pp": 100 * (
                row["annualized_volatility"] - reference["annualized_volatility"]),
            "delta_sharpe": row["sharpe_rf_zero"] - reference["sharpe_rf_zero"],
            "delta_max_drawdown_pp": 100 * (
                row["max_drawdown"] - reference["max_drawdown"]),
            "delta_final_nav_eur": row["final_nav_eur"] - reference["final_nav_eur"],
        })
    deltas = pd.DataFrame(delta_rows)
    rankings = sensitivity[["scenario", "method", "annualized_net_return",
                            "sharpe_rf_zero", "max_drawdown"]].copy()
    # Die Export-Ränge werden weiter unten aus derselben toleranzbasierten
    # Rangmatrix wie die Konsolenausgabe übernommen.
    diagnostics.to_csv(output / "baseline_diagnostics.csv", index=False)
    sensitivity.to_csv(output / "sensitivity_metrics.csv", index=False)
    deltas.to_csv(output / "sensitivity_vs_baseline.csv", index=False)
    detail, summary = _presentation_tables(sensitivity)
    rankings = rankings.merge(detail[["scenario", "method", "rank_return", "rank_sharpe"]].rename(
        columns={"rank_return": "return_rank", "rank_sharpe": "sharpe_rank"}),
        on=["scenario", "method"], validate="one_to_one")
    rankings.to_csv(output / "sensitivity_rankings.csv", index=False)
    detail.to_csv(output / "kpi_je_parameter_und_modell.csv", index=False)
    detail[["scenario", "method"] + [f"rank_{key}" for key in ("return", "volatility", "sharpe", "drawdown", "cvar", "nav", "fees")]].to_csv(
        output / "rangmatrix_je_parameter.csv", index=False)
    summary.to_csv(output / "sieger_je_parameter.csv", index=False)
    _print_readable_results(detail, summary, audit, output)
    print(f"\nAnalysierte Sensitivitaetsfaelle: {len(_cases(quick))}")
    return diagnostics, sensitivity, deltas, rankings


def main():
    """Startet die Analysen direkt, ohne Testmodus oder versteckte Parameterwahl."""
    parser = argparse.ArgumentParser(description="Modul 5: Diagnostik und Sensitivität")
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--start", default="2023-01-31")
    parser.add_argument("--quick", action="store_true",
                        help="Nur Baseline, zwei Grenzfälle und ein Kostenszenario")
    args = parser.parse_args()
    candidates = [ROOT / "data" / "prices_monthly.csv",
                  ROOT.parent / "data" / "prices_monthly.csv",
                  Path.cwd() / "data" / "prices_monthly.csv"]
    prices = args.prices or next((p for p in candidates if p.is_file()), None)
    if prices is None:
        parser.error("data/prices_monthly.csv fehlt; verwende --prices PFAD")
    run_analysis(prices, args.output or ROOT / "output" / "module5",
                 start=args.start, quick=args.quick)


if __name__ == "__main__":
    main()
