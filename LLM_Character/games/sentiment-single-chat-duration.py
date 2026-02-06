# -*- coding: utf-8 -*-
"""
log_sentiment_rm_anova.py

Reads all .txt log files from a folder (each file = one player),
extracts [DialogueUser] and [MessageNpc] utterances, computes VADER sentiment
per utterance, aggregates mean compound sentiment per player x quest x difficulty,
and runs a 2x2 repeated-measures ANOVA (Quest x Difficulty).

UTF-8 SAFE VERSION:
- Includes encoding header
- Avoids emojis / unicode-only console characters
- Uses UTF-8 reading with errors="replace"

Outputs (console):
- # log files found + example names
- long utterance-level preview
- aggregated per player-condition preview
- per speaker (user / npc):
  - wide table for ALL players (may contain NaN)
  - missing cell report
  - RM-ANOVA table (complete cases only)

Dependencies:
  pip install nltk pandas statsmodels
Optional:
  pip install pingouin

If needed once:
  python -c "import nltk; nltk.download('vader_lexicon')"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

from statsmodels.stats.anova import AnovaRM


# ----------------------------
# CONFIG
# ----------------------------
LOGS_FOLDER = "logs/"  # folder containing .txt logs
START_QUEST = 0
START_DIFFICULTY = "Challenging"

# Restrict to the intended 2x2 design (prevents tutorial/other quest ids contaminating cells)
QUEST_ALLOWED = {"0", "1"}  # adjust if your study uses different ids
DIFF_ALLOWED = {"Easy", "Challenging"}


# ----------------------------
# VADER SETUP
# ----------------------------
def _ensure_vader_lexicon() -> None:
    """Ensure the VADER lexicon is available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


# ----------------------------
# PARSING HELPERS
# ----------------------------
def extract_user_text(raw: str) -> Optional[str]:
    """
    Handles user dialogue payloads like:
      - Plain text: Hello?
      - JSON: {"text":" hello"}
    Returns extracted text or None.
    """
    raw = raw.strip()
    if not raw:
        return None

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if isinstance(data, dict) and "text" in data:
            text_val = data.get("text")
            if text_val is None:
                return None
            text = str(text_val).strip()
            return text or None
        return None

    return raw


def normalize_difficulty(val: str) -> str:
    """Normalize difficulty labels so they match your buckets consistently."""
    v = (val or "").strip()
    if not v:
        return v
    v_low = v.lower()

    if v_low in ("easy", "e", "low"):
        return "Easy"
    if v_low in ("challenging", "hard", "h", "high"):
        return "Challenging"

    if v in ("Easy", "Challenging"):
        return v

    # Unknown label: keep original string
    return v


# ----------------------------
# DATA STRUCTURE
# ----------------------------
@dataclass
class UtteranceRow:
    player: str
    speaker: str  # "user" or "npc"
    quest: int
    difficulty: str
    compound: float


# ----------------------------
# LOG READER
# ----------------------------
class LogReaderRM:
    QUEST_PATTERN = re.compile(r"\[Scene\]\s+\[Quest\]\s+(\d+)")
    DIFFICULTY_PATTERN = re.compile(
        r"\[Scene\]\s+\[Complexity\]\s+Difficulty\s*=\s*([A-Za-z]+)"
    )

    # Exclude: [DialogueUser] [User] User talk start/stop
    USER_DIALOGUE_PATTERN = re.compile(r"\[DialogueUser\]\s+(?!\[\s*User\s*\])(.+)")

    # Exclude: [MessageNpc] Evaluation: X
    NPC_MESSAGE_PATTERN = re.compile(r"\[MessageNpc\]\s+(?!Evaluation\s*:)(.+)")

    def __init__(self, folder_path: str, start_quest: int, start_difficulty: str):
        _ensure_vader_lexicon()
        self.sia = SentimentIntensityAnalyzer()
        self.folder_path = Path(folder_path)

        self.start_quest = start_quest
        self.start_difficulty = normalize_difficulty(start_difficulty)

        self.rows: List[UtteranceRow] = []

    def read_all(self) -> List[Path]:
        files = sorted(self.folder_path.glob("*.txt"))
        for fp in files:
            self._read_file(fp)
        return files

    def _read_file(self, file_path: Path) -> None:
        player_id = file_path.stem

        current_quest_id = self.start_quest
        current_difficulty = self.start_difficulty

        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                m = self.QUEST_PATTERN.search(line)
                if m:
                    current_quest_id = int(m.group(1))
                    continue

                m = self.DIFFICULTY_PATTERN.search(line)
                if m:
                    current_difficulty = normalize_difficulty(m.group(1))
                    continue

                m = self.USER_DIALOGUE_PATTERN.search(line)
                if m:
                    payload = m.group(1).strip()
                    text = extract_user_text(payload)
                    if text:
                        scores = self.sia.polarity_scores(text)
                        self.rows.append(
                            UtteranceRow(
                                player=player_id,
                                speaker="user",
                                quest=current_quest_id,
                                difficulty=current_difficulty,
                                compound=float(scores["compound"]),
                            )
                        )
                    continue

                m = self.NPC_MESSAGE_PATTERN.search(line)
                if m:
                    text = m.group(1).strip()
                    if text:
                        scores = self.sia.polarity_scores(text)
                        self.rows.append(
                            UtteranceRow(
                                player=player_id,
                                speaker="npc",
                                quest=current_quest_id,
                                difficulty=current_difficulty,
                                compound=float(scores["compound"]),
                            )
                        )
                    continue

    def build_long_df(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=["player", "speaker", "quest", "difficulty", "compound"])

        return pd.DataFrame(
            {
                "player": [r.player for r in self.rows],
                "speaker": [r.speaker for r in self.rows],
                "quest": [r.quest for r in self.rows],
                "difficulty": [r.difficulty for r in self.rows],
                "compound": [r.compound for r in self.rows],
            }
        )


# ----------------------------
# ANALYSIS HELPERS
# ----------------------------
def aggregate_per_player_condition(df_long: pd.DataFrame) -> pd.DataFrame:
    """Aggregates utterance-level compound into one mean per player x speaker x quest x difficulty."""
    if df_long.empty:
        return pd.DataFrame(
            columns=["player", "speaker", "quest", "difficulty", "compound_mean", "n_utterances"]
        )

    df_agg = (
        df_long.groupby(["player", "speaker", "quest", "difficulty"], as_index=False)
        .agg(
            compound_mean=("compound", "mean"),
            n_utterances=("compound", "size"),
        )
    )
    return df_agg


def _expected_cells() -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [sorted(QUEST_ALLOWED), sorted(DIFF_ALLOWED)],
        names=["quest", "difficulty"],
    )


def rm_anova_2x2_keep_all(df_agg: pd.DataFrame, speaker: str) -> Dict[str, object]:
    """
    Keeps ALL players in aggregates; runs RM-ANOVA on complete cases only.
    Returns dict with:
      - anova table (complete cases)
      - wide table (all players; NaNs show missing cells)
      - missing report
    """
    d = df_agg[df_agg["speaker"] == speaker].copy()
    if d.empty:
        raise ValueError("No data for speaker=%s" % speaker)

    d["quest"] = d["quest"].astype(str)
    d["difficulty"] = d["difficulty"].astype(str)

    # Keep only intended 2x2 design labels
    d = d[d["quest"].isin(QUEST_ALLOWED) & d["difficulty"].isin(DIFF_ALLOWED)].copy()

    # Defensive collapse (if any duplicates exist)
    d = (
        d.groupby(["player", "quest", "difficulty"], as_index=False)
        .agg(
            compound_mean=("compound_mean", "mean"),
            n_utterances=("n_utterances", "sum"),
        )
    )

    wide_all = d.pivot_table(
        index="player",
        columns=["quest", "difficulty"],
        values="compound_mean",
        aggfunc="mean",
    )

    # Ensure all expected columns exist
    exp = _expected_cells()
    for col in exp:
        if col not in wide_all.columns:
            wide_all[col] = pd.NA
    wide_all = wide_all.reindex(columns=exp)

    # Missing report
    missing = []
    for player, row in wide_all.iterrows():
        miss_cells = []
        for (q, diff), val in row.items():
            if pd.isna(val):
                miss_cells.append("quest=%s, diff=%s" % (q, diff))
        if miss_cells:
            missing.append({"player": player, "missing_cells": "; ".join(miss_cells)})

    missing_report = (
        pd.DataFrame(missing).sort_values("player")
        if missing
        else pd.DataFrame(columns=["player", "missing_cells"])
    )

    # Complete cases only
    complete_players = wide_all.dropna().index.tolist()
    if len(complete_players) < 2:
        raise ValueError(
            "Not enough complete players for RM-ANOVA (speaker=%s). Complete=%d / %d"
            % (speaker, len(complete_players), wide_all.shape[0])
        )

    complete_long = d[d["player"].isin(complete_players)].copy()

    aovrm = AnovaRM(
        data=complete_long,
        depvar="compound_mean",
        subject="player",
        within=["quest", "difficulty"],
    ).fit()

    anova_table = aovrm.anova_table.reset_index().rename(columns={"index": "effect"})

    return {
        "anova": anova_table,
        "wide_all": wide_all,
        "missing_report": missing_report,
        "n_players_total": int(wide_all.shape[0]),
        "n_players_complete": int(len(complete_players)),
    }


def pretty_print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print("\n=== %s ===" % title)
    if df.empty:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows", max_rows,
        "display.max_columns", 50,
        "display.width", 140,
    ):
        print(df)


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    reader = LogReaderRM(
        folder_path=LOGS_FOLDER,
        start_quest=START_QUEST,
        start_difficulty=START_DIFFICULTY,
    )

    files = reader.read_all()
    print("\nTXT files found:", len(files))
    print("Example file names:", [p.name for p in files[:10]])

    df_long = reader.build_long_df()
    pretty_print_df("Long utterance-level data (head)", df_long.head(20), max_rows=20)

    df_agg = aggregate_per_player_condition(df_long)
    pretty_print_df("Aggregated per player x condition (head)", df_agg.head(40), max_rows=40)

    for spk in ("user", "npc"):
        try:
            res = rm_anova_2x2_keep_all(df_agg, speaker=spk)

            print("\n--- Speaker=%s ---" % spk)
            print("Players total:   %d" % res["n_players_total"])
            print("Players complete (2x2): %d" % res["n_players_complete"])

            pretty_print_df(
                "Wide 2x2 table (ALL players; NaN indicates missing cells) speaker=%s" % spk,
                res["wide_all"].reset_index(),
                max_rows=30,
            )

            if not res["missing_report"].empty:
                pretty_print_df(
                    "Missing-cell report (why some players excluded from RM-ANOVA) speaker=%s" % spk,
                    res["missing_report"],
                    max_rows=50,
                )

            pretty_print_df("RM-ANOVA (complete cases only) speaker=%s" % spk, res["anova"], max_rows=50)

        except Exception as e:
            print("\n[ERROR] speaker=%s: %s" % (spk, e))
