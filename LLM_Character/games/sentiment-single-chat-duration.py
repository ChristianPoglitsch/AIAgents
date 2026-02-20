# -*- coding: utf-8 -*-
"""
log_sentiment_rm_anova.py

(Updated)
- Keeps your VADER sentiment pipeline
- Adds "talking time" extraction:
  * user_time:    [STT] STT start  -> [STT] STT stop
  * npc_time:     [LlmProcessing] ... LLM start chat completion -> [STT] TTS stop

Outputs (console):
- sentiment: same as before (wide + missing report + RM-ANOVA)
- talking time: wide tables + missing reports (no RM-ANOVA by default, but easy to add)

Dependencies:
  pip install nltk pandas statsmodels
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
# TIME PARSING
# ----------------------------
# Example prefix: [10:24:25.680] ...
TS_PATTERN = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]\s+")


def parse_ts_to_seconds(line: str) -> Optional[float]:
    """
    Parse a log timestamp like: [HH:MM:SS.mmm]
    Return seconds-from-midnight as float.
    """
    m = TS_PATTERN.match(line)
    if not m:
        return None
    hh, mm, ss, ms = map(int, m.groups())
    return hh * 3600 + mm * 60 + ss + (ms / 1000.0)


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
# DATA STRUCTURES
# ----------------------------
@dataclass
class UtteranceRow:
    player: str
    speaker: str  # "user" or "npc"
    quest: int
    difficulty: str
    compound: float


@dataclass
class TalkTimeRow:
    player: str
    speaker: str  # "user" or "npc"
    quest: int
    difficulty: str
    start_s: float
    stop_s: float
    duration_s: float
    kind: str  # "user_stt" or "npc_llm_to_tts"


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

    # Timing markers requested
    USER_STT_START = re.compile(r"\[STT\]\s+STT start")
    USER_STT_STOP = re.compile(r"\[STT\]\s+STT stop")

    NPC_LLM_START_CHAT = re.compile(
        r"\[LlmProcessing\]\s+\[LlmProcessing\]\s+LLM start chat completion"
    )
    NPC_TTS_STOP = re.compile(r"\[STT\]\s+TTS stop")
    NPC_TTS_START = re.compile(r"\[STT\]\s+TTS start")

    # For skipping evaluation completions in timing
    NPC_EVAL_MSG = re.compile(r"\[MessageNpc\]\s+Evaluation\s*:")

    def __init__(self, folder_path: str, start_quest: int, start_difficulty: str):
        _ensure_vader_lexicon()
        self.sia = SentimentIntensityAnalyzer()
        self.folder_path = Path(folder_path)

        self.start_quest = start_quest
        self.start_difficulty = normalize_difficulty(start_difficulty)

        self.rows: List[UtteranceRow] = []
        self.talk_rows: List[TalkTimeRow] = []

    def read_all(self) -> List[Path]:
        files = sorted(self.folder_path.glob("*.txt"))
        for fp in files:
            self._read_file(fp)
        return files

    def _read_file(self, file_path: Path) -> None:
        player_id = file_path.stem

        current_quest_id = self.start_quest
        current_difficulty = self.start_difficulty

        # --- timing state ---
        user_stt_start_s: Optional[float] = None

        npc_llm_start_s: Optional[float] = None
        npc_started_from_eval: bool = False  # if the chat completion is "Evaluation", skip timing
        npc_seen_tts_start: bool = False     # optional guard (helpful if you want it)

        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                t_s = parse_ts_to_seconds(line)

                # Scene factors
                m = self.QUEST_PATTERN.search(line)
                if m:
                    current_quest_id = int(m.group(1))
                    continue

                m = self.DIFFICULTY_PATTERN.search(line)
                if m:
                    current_difficulty = normalize_difficulty(m.group(1))
                    continue

                # ----------------------------
                # TALK-TIME: USER (STT start -> STT stop)
                # ----------------------------
                if self.USER_STT_START.search(line):
                    if t_s is not None:
                        # restart if already started (keep latest)
                        user_stt_start_s = t_s
                    continue

                if self.USER_STT_STOP.search(line):
                    if t_s is not None and user_stt_start_s is not None:
                        dur = max(0.0, t_s - user_stt_start_s)
                        self.talk_rows.append(
                            TalkTimeRow(
                                player=player_id,
                                speaker="user",
                                quest=current_quest_id,
                                difficulty=current_difficulty,
                                start_s=user_stt_start_s,
                                stop_s=t_s,
                                duration_s=dur,
                                kind="user_stt",
                            )
                        )
                        user_stt_start_s = None
                    continue

                # ----------------------------
                # TALK-TIME: NPC (LLM start chat completion -> TTS stop)
                # ----------------------------
                if self.NPC_LLM_START_CHAT.search(line):
                    if t_s is not None:
                        npc_llm_start_s = t_s
                        npc_started_from_eval = False
                        npc_seen_tts_start = False
                    continue

                # If the current chat completion is an evaluation, mark it so we skip timing
                if npc_llm_start_s is not None and self.NPC_EVAL_MSG.search(line):
                    npc_started_from_eval = True
                    continue

                if self.NPC_TTS_START.search(line):
                    if npc_llm_start_s is not None:
                        npc_seen_tts_start = True
                    continue

                if self.NPC_TTS_STOP.search(line):
                    if (
                        t_s is not None
                        and npc_llm_start_s is not None
                        and not npc_started_from_eval
                    ):
                        dur = max(0.0, t_s - npc_llm_start_s)
                        self.talk_rows.append(
                            TalkTimeRow(
                                player=player_id,
                                speaker="npc",
                                quest=current_quest_id,
                                difficulty=current_difficulty,
                                start_s=npc_llm_start_s,
                                stop_s=t_s,
                                duration_s=dur,
                                kind="npc_llm_to_tts",
                            )
                        )
                    # close regardless (prevents hanging state)
                    npc_llm_start_s = None
                    npc_started_from_eval = False
                    npc_seen_tts_start = False
                    continue

                # ----------------------------
                # SENTIMENT: user utterances
                # ----------------------------
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

                # ----------------------------
                # SENTIMENT: npc utterances
                # ----------------------------
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

    def build_talk_long_df(self) -> pd.DataFrame:
        if not self.talk_rows:
            return pd.DataFrame(
                columns=[
                    "player", "speaker", "quest", "difficulty",
                    "start_s", "stop_s", "duration_s", "kind"
                ]
            )

        return pd.DataFrame(
            {
                "player": [r.player for r in self.talk_rows],
                "speaker": [r.speaker for r in self.talk_rows],
                "quest": [r.quest for r in self.talk_rows],
                "difficulty": [r.difficulty for r in self.talk_rows],
                "start_s": [r.start_s for r in self.talk_rows],
                "stop_s": [r.stop_s for r in self.talk_rows],
                "duration_s": [r.duration_s for r in self.talk_rows],
                "kind": [r.kind for r in self.talk_rows],
            }
        )


# ----------------------------
# ANALYSIS HELPERS (sentiment)
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


# ----------------------------
# ANALYSIS HELPERS (talk time)
# ----------------------------
def aggregate_talktime_per_player_condition(df_talk: pd.DataFrame) -> pd.DataFrame:
    """Aggregates segment-level durations into one total per player x speaker x quest x difficulty."""
    if df_talk.empty:
        return pd.DataFrame(
            columns=["player", "speaker", "quest", "difficulty", "talk_s_total", "n_segments"]
        )

    df_agg = (
        df_talk.groupby(["player", "speaker", "quest", "difficulty"], as_index=False)
        .agg(
            talk_s_total=("duration_s", "sum"),
            n_segments=("duration_s", "size"),
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


def wide_2x2_keep_all_from_value(
    df_agg: pd.DataFrame,
    speaker: str,
    value_col: str,
) -> Dict[str, object]:
    """
    Build a wide 2x2 table (all players) for any aggregated value column, with missing report.
    No ANOVA here (you can add if you want).
    """
    d = df_agg[df_agg["speaker"] == speaker].copy()
    if d.empty:
        raise ValueError("No data for speaker=%s" % speaker)

    d["quest"] = d["quest"].astype(str)
    d["difficulty"] = d["difficulty"].astype(str)

    d = d[d["quest"].isin(QUEST_ALLOWED) & d["difficulty"].isin(DIFF_ALLOWED)].copy()

    # Defensive collapse (if any duplicates exist)
    d = (
        d.groupby(["player", "quest", "difficulty"], as_index=False)
        .agg(**{value_col: (value_col, "sum")})
    )

    wide_all = d.pivot_table(
        index="player",
        columns=["quest", "difficulty"],
        values=value_col,
        aggfunc="mean",
    )

    exp = _expected_cells()
    for col in exp:
        if col not in wide_all.columns:
            wide_all[col] = pd.NA
    wide_all = wide_all.reindex(columns=exp)

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

    return {
        "wide_all": wide_all,
        "missing_report": missing_report,
        "n_players_total": int(wide_all.shape[0]),
        "n_players_complete": int(wide_all.dropna().shape[0]),
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

    # ---- sentiment pipeline (unchanged) ----
    df_long = reader.build_long_df()
    pretty_print_df("Long utterance-level SENTIMENT data (head)", df_long.head(20), max_rows=20)

    df_agg = aggregate_per_player_condition(df_long)
    pretty_print_df("Aggregated SENTIMENT per player x condition (head)", df_agg.head(40), max_rows=40)

    for spk in ("user", "npc"):
        try:
            res = rm_anova_2x2_keep_all(df_agg, speaker=spk)

            print("\n--- SENTIMENT Speaker=%s ---" % spk)
            print("Players total:   %d" % res["n_players_total"])
            print("Players complete (2x2): %d" % res["n_players_complete"])

            pretty_print_df(
                "Wide 2x2 SENTIMENT table (ALL players; NaN indicates missing cells) speaker=%s" % spk,
                res["wide_all"].reset_index(),
                max_rows=30,
            )

            if not res["missing_report"].empty:
                pretty_print_df(
                    "SENTIMENT missing-cell report speaker=%s" % spk,
                    res["missing_report"],
                    max_rows=50,
                )

            pretty_print_df("RM-ANOVA SENTIMENT (complete cases only) speaker=%s" % spk, res["anova"], max_rows=50)

        except Exception as e:
            print("\n[ERROR] SENTIMENT speaker=%s: %s" % (spk, e))

    # ---- talking-time pipeline (new) ----
    df_talk_long = reader.build_talk_long_df()
    pretty_print_df("Long TALK-TIME segments (head)", df_talk_long.head(30), max_rows=30)

    df_talk_agg = aggregate_talktime_per_player_condition(df_talk_long)
    pretty_print_df("Aggregated TALK-TIME per player x condition (head)", df_talk_agg.head(40), max_rows=40)

    for spk in ("user", "npc"):
        try:
            res_t = wide_2x2_keep_all_from_value(df_talk_agg, speaker=spk, value_col="talk_s_total")

            print("\n--- TALK-TIME Speaker=%s ---" % spk)
            print("Players total:   %d" % res_t["n_players_total"])
            print("Players complete (2x2): %d" % res_t["n_players_complete"])

            pretty_print_df(
                "Wide 2x2 TALK-TIME table (seconds; ALL players; NaN indicates missing cells) speaker=%s" % spk,
                res_t["wide_all"].reset_index(),
                max_rows=30,
            )

            if not res_t["missing_report"].empty:
                pretty_print_df(
                    "TALK-TIME missing-cell report speaker=%s" % spk,
                    res_t["missing_report"],
                    max_rows=50,
                )

        except Exception as e:
            print("\n[ERROR] TALK-TIME speaker=%s: %s" % (spk, e))
