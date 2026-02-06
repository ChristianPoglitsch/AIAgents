# -*- coding: utf-8 -*-

"""
SimVille TLX Auswertung – Konsolen-Version
KORREKT: 2×2 Repeated Measures ANOVA (NPC × Difficulty)

Annahme:
- Die ersten 4 Zeilen gehören zu User 1
- Die nächsten 4 Zeilen gehören zu User 2
- ...
- insgesamt 25 User → 100 Zeilen

Outputs nur Konsole.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM


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


def compute_partial_eta2(F: float, df_num: float, df_den: float) -> float:
    """
    partial eta squared from F and dfs:
    ηp² = (F * df_num) / (F * df_num + df_den)
    """
    return (F * df_num) / (F * df_num + df_den)


def run_rm_anova(df: pd.DataFrame, dv_col: str):
    """
    Correct RM-ANOVA: within-subjects 2x2 design.
    Needs columns: dv_col, NPC, Difficulty, Participant
    """
    tmp = df[["Participant", dv_col, "NPC", "Difficulty"]].dropna().copy()

    # RM-ANOVA
    res = AnovaRM(
        data=tmp,
        depvar=dv_col,
        subject="Participant",
        within=["NPC", "Difficulty"],
    ).fit()

    # Table format
    aov = res.anova_table.copy()
    aov = aov.rename(
        columns={
            "F Value": "F",
            "Num DF": "df_num",
            "Den DF": "df_den",
            "Pr > F": "p",
        }
    )

    aov["partial_eta2"] = aov.apply(
        lambda r: compute_partial_eta2(r["F"], r["df_num"], r["df_den"]),
        axis=1,
    )

    # Cell descriptives
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

    # --------------------------------------------------
    # Participant-ID erzeugen: jeweils 4 Zeilen = 1 Person
    # --------------------------------------------------
    if len(df) % 4 != 0:
        raise ValueError(
            f"Zeilenanzahl ist nicht durch 4 teilbar: {len(df)}.\n"
            "Bitte prüfen, ob wirklich 4 Zeilen pro Person vorhanden sind."
        )

    df["Participant"] = (np.arange(len(df)) // 4) + 1  # 1..25

    n_participants = df["Participant"].nunique()
    print(f"✅ Participant IDs generiert: {n_participants} Personen")

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
    print("TLX GESAMTWERT (RM-ANOVA)")
    print("==============================")

    aov_total, cell_total = run_rm_anova(df, "TLX_Total")

    print("\nDeskriptive Statistiken:")
    print(cell_total.to_string(index=False))

    print("\nRM-ANOVA (within-subject), inkl. partial η²:")
    print(aov_total.round(4).to_string())

    # --------------------------------------------------
    # Einzelne Dimensionen
    # --------------------------------------------------
    for dim_name, col in TLX_COLS.items():
        print("\n" + "=" * 30)
        print(f"{dim_name.upper()} (RM-ANOVA)")
        print("=" * 30)

        aov, cell = run_rm_anova(df, col)

        print("\nDeskriptive Statistiken:")
        print(cell.to_string(index=False))

        print("\nRM-ANOVA (within-subject), inkl. partial η²:")
        print(aov.round(4).to_string())

    print("\n✅ Analyse abgeschlossen")


if __name__ == "__main__":
    main()
