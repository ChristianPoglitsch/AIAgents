# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.anova import AnovaRM

import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

# --------------------------------------------------
# VIOLIN PLOTTING
# --------------------------------------------------
def plot_nasa_tlx_violins_like_paper(df: pd.DataFrame, tlx_cols: dict[str, str]):
    df = add_condition_labels(df)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.ravel()

    x_labels = list(COND_ORDER)  # left-to-right
    x_pos = {lab: i for i, lab in enumerate(x_labels)}

    for ax, (dim_name, col) in zip(axes, tlx_cols.items()):
        tmp = df[["Participant", "Condition", col, "NPC", "Difficulty"]].dropna().copy()

        # --- RM-ANOVA p-values
        p_npc = p_diff = p_int = None
        try:
            aov, _ = run_rm_anova(df, col)
            p_npc = float(aov.loc["NPC", "p"]) if "NPC" in aov.index else None
            p_diff = float(aov.loc["Difficulty", "p"]) if "Difficulty" in aov.index else None
            p_int = float(aov.loc["NPC:Difficulty", "p"]) if "NPC:Difficulty" in aov.index else None
        except Exception:
            pass

        ax.set_title(dim_name)

        # ---------------------------------
        # VIOLINS (VERTICAL)
        # ---------------------------------
        data_for_plot = []
        positions = []

        for i, cond in enumerate(x_labels):
            sub = tmp[tmp["Condition"] == cond][col]
            if len(sub) > 0:
                data_for_plot.append(sub)
                positions.append(i)

        parts = ax.violinplot(
            data_for_plot,
            positions=positions,
            vert=True,          # <-- vertical
            showmeans=True,
            showextrema=False,
            widths=0.8,
        )

        # Color each violin
        for i, pc in enumerate(parts["bodies"]):
            cond = x_labels[i]
            pc.set_facecolor(ROW_COLORS.get(cond, "#3182bd"))
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        # ---------------------------------
        # AXES STYLING
        # ---------------------------------
        ax.set_ylim(1, 7)
        ax.set_yticks(range(1, 8))
        ax.tick_params(axis="y", labelsize=14)

        ax.set_xticks(range(len(x_labels)))

        # Only bottom row gets x labels (vertical equivalent of "only left column gets labels")
        if ax in axes[3:]:  # bottom row in 2x3 layout: axes 3,4,5
            ax.set_xticklabels(x_labels, rotation=25, ha="right")
            ax.tick_params(axis="x", labelsize=14)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ---------------------------------
        # SIGNIFICANCE BRACKETS (TOP)
        # ---------------------------------
        # We'll draw "horizontal" brackets above violins:
        # NPC: between Multiparty + and Multiparty -
        # Difficulty: between Dyadic + and Dyadic -
        # Interaction: between Dyadic - and Multiparty +
        def draw_h_bracket(ax, x1, x2, y, text, cap=0.12, lw=1.3):
            x_low, x_high = (x1, x2) if x1 < x2 else (x2, x1)
            ax.plot([x_low, x_high], [y, y], lw=lw, color="black", zorder=10, clip_on=False)
            ax.plot([x_low, x_low], [y, y - cap], lw=lw, color="black", zorder=10, clip_on=False)
            ax.plot([x_high, x_high], [y, y - cap], lw=lw, color="black", zorder=10, clip_on=False)
            ax.text(
                (x_low + x_high) / 2,
                y + 0.05,
                text,
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
                zorder=11,
                clip_on=False,
            )

        bracket_y0 = 7.35     # start above top of scale
        bracket_step = 0.35   # stack upward
        k = 0

        if (p_npc is not None) and (p_npc < ALPHA):
            y = bracket_y0 + k * bracket_step
            k += 1
            x1 = x_pos["Multiparty – Positive"]
            x2 = x_pos["Multiparty – Negative"]
            label = f"{p_to_marker(p_npc)}\nSetting\np={format_p(p_npc)}"
            draw_h_bracket(ax, x1, x2, y=y, text=label)

        if (p_diff is not None) and (p_diff < ALPHA):
            y = bracket_y0 + k * bracket_step
            k += 1
            x1 = x_pos["Dyadic – Positive"]
            x2 = x_pos["Dyadic – Negative"]
            label = f"{p_to_marker(p_diff)}\nEmotional Tone\np={format_p(p_diff)}"
            draw_h_bracket(ax, x1, x2, y=y, text=label)

        if (p_int is not None) and (p_int < ALPHA):
            y = bracket_y0 + k * bracket_step
            k += 1
            x1 = x_pos["Dyadic – Negative"]
            x2 = x_pos["Multiparty – Positive"]
            label = f"{p_to_marker(p_int)}\nSetting × Emotional Tone\np={format_p(p_int)}"
            draw_h_bracket(ax, x1, x2, y=y, text=label)

        # Make sure the bracket text isn't cut off
        ax.set_ylim(1, 8.6)

    fig.suptitle(
        "NASA-TLX Distributions (Violin Plots)",
        y=0.98,
        fontsize=16,
    )

    fig.text(
        0.5,
        0.93,
        "Setting = Dyadic vs Multiparty     Emotional Tone = Positive vs Negative     Interaction = Setting × Emotional Tone",
        ha="center",
        fontsize=14,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

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

NPC_ORDER = ["1 NPC", "5 NPCs"]
DIFF_ORDER = ["Easy", "Challenging"]

COND_ORDER = [
    "Dyadic – Positive",
    "Dyadic – Negative",
    "Multiparty – Positive",
    "Multiparty – Negative",
]

ROW_COLORS = {
    "Dyadic – Positive": "#1f77b4",        # blue
    "Dyadic – Negative": "#ff7f0e",        # orange
    "Multiparty – Positive": "#2ca02c",    # green
    "Multiparty – Negative": "#d62728",    # red
}

ALPHA = 0.05


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
    return (F * df_num) / (F * df_num + df_den)


def run_rm_anova(df: pd.DataFrame, dv_col: str):
    tmp = df[["Participant", dv_col, "NPC", "Difficulty"]].dropna().copy()

    res = AnovaRM(
        data=tmp,
        depvar=dv_col,
        subject="Participant",
        within=["NPC", "Difficulty"],
    ).fit()

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

    cell_desc = (
        tmp.groupby(["NPC", "Difficulty"])[dv_col]
        .agg(N="count", Mean="mean", SD="std")
        .reset_index()
        .sort_values(["NPC", "Difficulty"])
    )

    return aov, cell_desc


def p_to_marker(p: float) -> str:
    if p < 0.05:
        return "(*)"
    return ""


def format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def add_condition_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["NPC"] = pd.Categorical(df["NPC"], categories=NPC_ORDER, ordered=True)
    df["Difficulty"] = pd.Categorical(df["Difficulty"], categories=DIFF_ORDER, ordered=True)

    def map_condition(row):
        if row["NPC"] == "1 NPC" and row["Difficulty"] == "Easy":
            return "Dyadic – Positive"
        if row["NPC"] == "1 NPC" and row["Difficulty"] == "Challenging":
            return "Dyadic – Negative"
        if row["NPC"] == "5 NPCs" and row["Difficulty"] == "Easy":
            return "Multiparty – Positive"
        if row["NPC"] == "5 NPCs" and row["Difficulty"] == "Challenging":
            return "Multiparty – Negative"
        return np.nan

    df["Condition"] = df.apply(map_condition, axis=1)
    df["Condition"] = pd.Categorical(df["Condition"], categories=COND_ORDER, ordered=True)
    return df


# --------------------------------------------------
# BRACKETS (READABLE)
# --------------------------------------------------
def draw_bracket(ax, y1, y2, x, text, cap=0.15, lw=1.3):
    """
    Compact vertical bracket with multiline label.
    """
    y_low, y_high = (y1, y2) if y1 < y2 else (y2, y1)

    ax.plot([x, x], [y_low, y_high], lw=lw, color="black", zorder=10)
    ax.plot([x - cap, x], [y_low, y_low], lw=lw, color="black", zorder=10)
    ax.plot([x - cap, x], [y_high, y_high], lw=lw, color="black", zorder=10)

    ax.text(
        x + 0.05,
        (y_low + y_high) / 2,
        text,
        va="center",
        ha="left",
        fontsize=10,
        weight="bold",
        zorder=11,
        clip_on=False,
    )


# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
def plot_nasa_tlx_bubbles_like_paper(df: pd.DataFrame, tlx_cols: dict[str, str]):
    df = add_condition_labels(df)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    axes = axes.ravel()

    x_vals = np.arange(1, 8)
    size_scale = 350

    # y layout: top-to-bottom like paper
    y_labels = list(COND_ORDER)[::-1]
    y_pos = {lab: i for i, lab in enumerate(y_labels)}

    for ax, (dim_name, col) in zip(axes, tlx_cols.items()):
        tmp = df[["Participant", "Condition", col, "NPC", "Difficulty"]].dropna().copy()

        # --- RM-ANOVA p-values
        p_npc = p_diff = p_int = None
        try:
            aov, _ = run_rm_anova(df, col)
            p_npc = float(aov.loc["NPC", "p"]) if "NPC" in aov.index else None
            p_diff = float(aov.loc["Difficulty", "p"]) if "Difficulty" in aov.index else None
            p_int = float(aov.loc["NPC:Difficulty", "p"]) if "NPC:Difficulty" in aov.index else None
        except Exception:
            pass

        ax.set_title(dim_name)

        # --- bubbles
        tmp["_rating"] = tmp[col].round().astype(int)
        tmp = tmp[tmp["_rating"].between(1, 7)]

        for cond in y_labels:
            sub = tmp[tmp["Condition"] == cond]
            counts = sub["_rating"].value_counts().reindex(x_vals, fill_value=0)

            xs, ys, ss, ts = [], [], [], []
            for x in x_vals:
                c = int(counts.loc[x])
                if c > 0:
                    xs.append(x)
                    ys.append(y_pos[cond])
                    ss.append(c * size_scale)
                    ts.append(c)

            if xs:
                ax.scatter(
                    xs, ys,
                    s=ss,
                    c=ROW_COLORS.get(cond, "#3182bd"),
                    alpha=0.80,
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=2,
                )
                for x, y, t in zip(xs, ys, ts):
                    ax.text(
                        x, y, str(t),
                        ha="center", va="center",
                        fontsize=8,
                        color="white",
                        weight="bold",
                        zorder=3,
                    )

        # --- axes styling + MUCH more room for bracket labels
        ax.set_xlim(0.5, 10.5)
        ax.set_xticks(x_vals)
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

        # >>> INCREASE AXIS FONT SIZES HERE <<<
        ax.tick_params(axis='x', labelsize=14)   # x-axis tick labels bigger
        ax.tick_params(axis='y', labelsize=14)   # y-axis tick labels bigger

        ax.grid(True, alpha=0.25, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # -------------------------------------------------
        # BRACKETS: spaced horizontally (readable)
        # Your requested placements:
        #   NPC: between Multiparty + and Multiparty -
        #   Difficulty: between Dyadic + and Dyadic -
        #   Interaction: between Dyadic - and Multiparty + (boundary pos/neg)
        # -------------------------------------------------
        bracket_x0 = 8.2       # first bracket x
        bracket_step = 0.75    # spacing between brackets (increase if needed)
        k = 0

        # NPC main effect -> bracket within Multiparty block
        if (p_npc is not None) and (p_npc < ALPHA):
            x = bracket_x0 + k * bracket_step
            k += 1
            y1 = y_pos["Multiparty – Positive"]
            y2 = y_pos["Multiparty – Negative"]
            label = f"{p_to_marker(p_npc)}\nSetting\np={format_p(p_npc)}"
            draw_bracket(ax, y1, y2, x=x, text=label)

        # Difficulty main effect -> bracket within Dyadic block
        if (p_diff is not None) and (p_diff < ALPHA):
            x = bracket_x0 + k * bracket_step
            k += 1
            y1 = y_pos["Dyadic – Positive"]
            y2 = y_pos["Dyadic – Negative"]
            label = f"{p_to_marker(p_diff)}\nEmotional Tone\np={format_p(p_diff)}"
            draw_bracket(ax, y1, y2, x=x, text=label)

        # Interaction -> bracket across the boundary Dyadic- vs Multiparty+ (pos/neg boundary)
        if (p_int is not None) and (p_int < ALPHA):
            x = bracket_x0 + k * bracket_step
            k += 1
            y1 = y_pos["Dyadic – Negative"]
            y2 = y_pos["Multiparty – Positive"]
            label = f"{p_to_marker(p_int)}\nSetting × Emotional Tone\np={format_p(p_int)}"
            draw_bracket(ax, y1, y2, x=x, text=label)

    fig.suptitle(
        "NASA-TLX Bubble-Frequencies",
        y=0.98,
        fontsize=15,
    )

    fig.text(
        0.5,
        0.92,
        "Setting = Dyadic vs Multiparty     Emotional Tone = Positive vs Negative     Interaction = Setting × Emotional Tone",
        ha="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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

    if len(df) % 4 != 0:
        raise ValueError(
            f"Zeilenanzahl ist nicht durch 4 teilbar: {len(df)}.\n"
            "Bitte prüfen, ob wirklich 4 Zeilen pro Person vorhanden sind."
        )

    df["Participant"] = (np.arange(len(df)) // 4) + 1
    print(f"✅ Participant IDs generiert: {df['Participant'].nunique()} Personen")

    df["Enter scene"] = df["Enter scene"].apply(normalize_enter_scene)
    df[["NPC", "Difficulty"]] = df["Enter scene"].apply(lambda x: pd.Series(derive_factors(x)))

    df["TLX_Total"] = df[list(TLX_COLS.values())].mean(axis=1)

    print("\n==============================")
    print("ZELLGRÖSSEN")
    print("==============================")
    print(
        df.groupby(["NPC", "Difficulty"])
        .size()
        .reset_index(name="N")
        .to_string(index=False)
    )

    print("\n==============================")
    print("TLX GESAMTWERT (RM-ANOVA)")
    print("==============================")
    aov_total, cell_total = run_rm_anova(df, "TLX_Total")
    print("\nDeskriptive Statistiken:")
    print(cell_total.to_string(index=False))
    print("\nRM-ANOVA (within-subject), inkl. partial η²:")
    print(aov_total.round(4).to_string())

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

    #plot_nasa_tlx_bubbles_like_paper(df, TLX_COLS)
    plot_nasa_tlx_violins_like_paper(df, TLX_COLS)


if __name__ == "__main__":
    main()
