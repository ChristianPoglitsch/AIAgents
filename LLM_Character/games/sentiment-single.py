"""
log_sentiment_rm_anova.py

Reads all .txt log files from a folder (each file = one player),
extracts [DialogueUser] and [MessageNpc] utterances, computes VADER sentiment
per utterance, then aggregates per player x quest x difficulty (mean compound),
and finally runs a 2x2 repeated-measures ANOVA (Quest x Difficulty).

Outputs:
- Mean sentiment per player-condition (user / npc)
- RM-ANOVA tables (user / npc)

Dependencies:
  pip install nltk pandas statsmodels
Optional (recommended):
  pip install pingouin

First run may need VADER lexicon:
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

# Statsmodels fallback
from statsmodels.stats.anova import AnovaRM


def _ensure_vader_lexicon() -> None:
    """Ensure the VADER lexicon is available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def extract_user_text(raw: str) -> Optional[str]:
    """
    Handles user dialogue payloads like:
      - Plain text: Hello?
      - JSON: {"text":" you you"}
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
    """
    Normalize difficulty labels so they match your buckets consistently.
    Adjust this mapping if your logs use different names.
    """
    v = (val or "").strip()
    if not v:
        return v
    v_low = v.lower()

    # Common variants
    if v_low in ("easy", "e", "low"):
        return "Easy"
    if v_low in ("challenging", "hard", "h", "high"):
        return "Challenging"

    # If logs already use the desired label, keep it (title-cased)
    if v in ("Easy", "Challenging"):
        return v

    # Otherwise keep original (but stable formatting)
    return v


@dataclass
class UtteranceRow:
    player: str
    speaker: str  # "user" or "npc"
    quest: int
    difficulty: str
    compound: float


class LogReaderRM:
    QUEST_PATTERN = re.compile(r"\[Scene\]\s+\[Quest\]\s+(\d+)")
    DIFFICULTY_PATTERN = re.compile(
        r"\[Scene\]\s+\[Complexity\]\s+Difficulty\s*=\s*(\w+)"
    )

    # Exclude: [DialogueUser] [User] User talk start/stop
    USER_DIALOGUE_PATTERN = re.compile(
        r"\[DialogueUser\]\s+(?!\[\s*User\s*\])(.+)"
    )
    # Exclude: [MessageNpc] Evaluation: X
    NPC_MESSAGE_PATTERN = re.compile(
        r"\[MessageNpc\]\s+(?!Evaluation\s*:)(.+)"
    )

    def __init__(
        self,
        folder_path: str,
        start_quest: int = 0,
        start_difficulty: str = "Challenging",
    ):
        _ensure_vader_lexicon()
        self.sia = SentimentIntensityAnalyzer()
        self.folder_path = Path(folder_path)

        self.start_quest = start_quest
        self.start_difficulty = normalize_difficulty(start_difficulty)

        # collected utterances (long format)
        self.rows: List[UtteranceRow] = []

    def read_all(self) -> None:
        for file_path in sorted(self.folder_path.glob("*.txt")):
            self._read_file(file_path)

    def _read_file(self, file_path: Path) -> None:
        player_id = file_path.stem  # each file = one player

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
            return pd.DataFrame(
                columns=["player", "speaker", "quest", "difficulty", "compound"]
            )

        return pd.DataFrame(
            {
                "player": [r.player for r in self.rows],
                "speaker": [r.speaker for r in self.rows],
                "quest": [r.quest for r in self.rows],
                "difficulty": [r.difficulty for r in self.rows],
                "compound": [r.compound for r in self.rows],
            }
        )

    @staticmethod
    def aggregate_per_player_condition(df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates utterance-level compounds into one mean per
        player x speaker x quest x difficulty.
        """
        if df_long.empty:
            return pd.DataFrame(
                columns=[
                    "player",
                    "speaker",
                    "quest",
                    "difficulty",
                    "compound_mean",
                    "n_utterances",
                ]
            )

        df_agg = (
            df_long.groupby(
                ["player", "speaker", "quest", "difficulty"], as_index=False
            )
            .agg(
                compound_mean=("compound", "mean"),
                n_utterances=("compound", "size"),
            )
        )
        return df_agg

    @staticmethod
    def rm_anova_2x2(df_agg: pd.DataFrame, speaker: str) -> Dict[str, pd.DataFrame]:
        """
        Runs 2x2 RM-ANOVA (within: quest, difficulty) on aggregated
        per-player means, for given speaker.
        """
        d = df_agg[df_agg["speaker"] == speaker].copy()
        if d.empty:
            raise ValueError(f"No data for speaker={speaker}")

        d["quest"] = d["quest"].astype(str)
        d["difficulty"] = d["difficulty"].astype(str)

        cell_counts = d.groupby("player").size()
        complete_players = set(cell_counts[cell_counts == 4].index)
        d_complete = d[d["player"].isin(complete_players)].copy()

        if d_complete.empty:
            raise ValueError(
                f"After filtering to complete cases, no data left for speaker={speaker}"
            )

        try:
            import pingouin as pg  # type: ignore

            anova = pg.rm_anova(
                data=d_complete,
                dv="compound_mean",
                within=["quest", "difficulty"],
                subject="player",
                detailed=True,
            )
            return {"anova": anova, "data_used": d_complete}

        except ModuleNotFoundError:
            aovrm = AnovaRM(
                data=d_complete,
                depvar="compound_mean",
                subject="player",
                within=["quest", "difficulty"],
            ).fit()
            table = (
                aovrm.anova_table.reset_index()
                .rename(columns={"index": "effect"})
            )
            return {"anova": table, "data_used": d_complete}


def pretty_print_df(title: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    print(f"\n=== {title} ===")
    if df.empty:
        print("(empty)")
        return
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        50,
        "display.width",
        140,
    ):
        print(df)


if __name__ == "__main__":
    reader = LogReaderRM(
        folder_path="logs/", start_quest=0, start_difficulty="Challenging"
    )
    reader.read_all()

    df_long = reader.build_long_df()
    pretty_print_df(
        "Long utterance-level data (head)", df_long.head(20), max_rows=20
    )

    df_agg = reader.aggregate_per_player_condition(df_long)
    pretty_print_df(
        "Aggregated per player x condition (head)",
        df_agg.head(40),
        max_rows=40,
    )

    for spk in ("user", "npc"):
        try:
            res = reader.rm_anova_2x2(df_agg, speaker=spk)
            pretty_print_df(
                f"RM-ANOVA (speaker={spk})", res["anova"], max_rows=50
            )
        except Exception as e:
            print(f"\n[ERROR] speaker={spk}: {e}")
