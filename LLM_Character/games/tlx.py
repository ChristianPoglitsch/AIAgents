# -*- coding: utf-8 -*-

"""
SimVille TLX Auswertung – Konsolen-Version
2×2 ANOVA: NPC × Schwierigkeit

- Datei: SimVille.xlsx (gleicher Ordner wie Skript)
- Keine Outputs auf Disk
- Alle Ergebnisse direkt in der Konsole
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# --------------------------------------------------
# KONFIGURATION
# --------------------------------------------------
DATA_FILENAME = "SimVille.xlsx"

TLX_COLS = {
    "Mental Demand": "How mentally demanding was the task?(1 = very low, 7 = very high)",
    "Physical Demand": "How physically demanding was the task?(1 = very low, 7 = very high)",
    "Temporal Demand": "How hurried or rushed was the pace of the task?(1 = very low, 7 = very high)",
    "Performance": "How successful were you in accomplishing what you were asked to do?(1 = very low, 7 = very high)",
    "Effort": "How hard did you have to work to achieve your level of performance?(1 = very low, 7 = very high)",
    "Frustration": "How insecure, discouraged, irritated, or stressed did you feel during the task?(1 = very low, 7 = very high)",
}


# --------------------------------------------------
# HILFSFUNKTIONEN
# --------------------------------------------------
def normalize_enter_scene(s: str) -> str:
    if pd.isna(s):
        return s
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def derive_factors(enter_scene: str) -> tuple[str, str]:
    s = normalize_enter_scene(enter_scene)
    npc = "1 NPC" if "Order 1 NPC" in s else "5 NPCs"
    difficulty = "Challenging" if "Challenging" in s else "Easy"
    return npc, difficulty


def partial_eta_squared(aov: pd.DataFrame, effect: str) -> float:
    ss_effect = float(aov.loc[effect, "sum_sq"])
    ss_error = float(aov.loc["Residual", "sum_sq"])
    return ss_effect / (ss_effect + ss_error)


def run_anova(df: pd.DataFrame, dv_col: str):
    tmp = df[[dv_col, "NPC", "Difficulty"]].dropna()

    model = ols(f'Q("{dv_col}") ~ C(NPC) * C(Difficulty)', data=tmp).fit()
    aov = sm.stats.anova_lm(model, typ=2)

    for eff in ["C(NPC)", "C(Difficulty)", "C(NPC):C(Difficulty)"]:
        aov.loc[eff, "partial_eta2"] = partial_eta_squared(aov, eff)

    cell_desc = (
        tmp.groupby(["NPC", "Difficulty"])[dv_col]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
        .sort_values(["NPC", "Difficulty"])
    )

    return aov, cell_desc


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / DATA_FILENAME

    if not data_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {data_path}")

    print("\n📄 Lade Datei:", data_path.name)

    df = pd.read_excel(data_path)

    # Faktoren erzeugen
    df["Enter scene"] = df["Enter scene"].apply(normalize_enter_scene)
    df[["NPC", "Difficulty"]] = df["Enter scene"].apply(
        lambda x: pd.Series(derive_factors(x))
    )

    # TLX Gesamtwert
    df["TLX_Total"] = df[list(TLX_COLS.values())].mean(axis=1)

    # --------------------------------------------------
    # Zellgrößen
    # --------------------------------------------------
    print("\n==============================")
    print("ZELLGRÖSSEN")
    print("==============================")
    print(
        df.groupby(["NPC", "Difficulty"])
        .size()
        .reset_index(name="N")
        .to_string(index=False)
    )

    # --------------------------------------------------
    # TLX Gesamt
    # --------------------------------------------------
    print("\n==============================")
    print("TLX GESAMTWERT")
    print("==============================")

    aov_total, cell_total = run_anova(df, "TLX_Total")

    print("\nDeskriptive Statistiken:")
    print(cell_total.to_string(index=False))

    print("\nANOVA (Typ II, inkl. partial η²):")
    print(aov_total.round(4).to_string())

    # --------------------------------------------------
    # Einzelne Dimensionen
    # --------------------------------------------------
    for dim_name, col in TLX_COLS.items():
        print("\n" + "=" * 30)
        print(f"{dim_name.upper()}")
        print("=" * 30)

        aov, cell = run_anova(df, col)

        print("\nDeskriptive Statistiken:")
        print(cell.to_string(index=False))

        print("\nANOVA (Typ II, inkl. partial η²):")
        print(aov.round(4).to_string())

    print("\n✅ Analyse abgeschlossen")


if __name__ == "__main__":
    main()
