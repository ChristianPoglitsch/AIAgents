"""
log_sentiment_anova.py

Reads all .txt log files from a folder, routes [DialogueUser] and [MessageNpc] text into
one of four (Quest x Difficulty) buckets, runs VADER sentiment per utterance, and then
computes TWO-WAY ANOVA (Quest x Difficulty) on VADER compound scores for:
  - User utterances
  - NPC utterances

Exclusions:
- User markers like:   [DialogueUser] [User] User talk start/stop
- NPC eval lines like: [MessageNpc] Evaluation: 0/1
- User JSON lines like: [DialogueUser] {"text":" you you"}  -> extracts only "text"

Dependencies:
  pip install nltk pandas statsmodels

First run may need VADER lexicon:
  python -c "import nltk; nltk.download('vader_lexicon')"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def _ensure_vader_lexicon() -> None:
    """
    Ensure the VADER lexicon is available. Downloads it if missing.
    Note: requires internet access the first time.
    """
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


def extract_user_text(raw: str) -> Optional[str]:
    """
    Handles user dialogue payloads like:
      - Plain text: Hello?
      - JSON: {"text":" you you"}
    Returns the extracted text or None if it can't be extracted.
    """
    raw = raw.strip()
    if not raw:
        return None

    # JSON payload
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

    # Plain text payload
    return raw


@dataclass
class SentimentAccumulator:
    """
    Accumulates VADER sentiment scores across many texts.
    Four dimensions: neg, neu, pos, compound
    """
    count: int = 0
    sums: Dict[str, float] = field(
        default_factory=lambda: {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
    )

    def add(self, scores: Dict[str, float]) -> None:
        self.count += 1
        for k in self.sums:
            self.sums[k] += float(scores.get(k, 0.0))

    def mean(self) -> Dict[str, float]:
        if self.count == 0:
            return {k: 0.0 for k in self.sums}
        return {k: self.sums[k] / self.count for k in self.sums}


class BaseQuest:
    """
    Stores utterances + VADER sentiment for a specific (quest_id, difficulty) bucket.
    Also stores per-utterance compound lists for ANOVA.
    """

    def __init__(self, quest_id: int, difficulty: str, sia: SentimentIntensityAnalyzer):
        self.quest_id = quest_id
        self.difficulty = difficulty
        self.sia = sia

        self.user_dialogues: List[str] = []
        self.npc_messages: List[str] = []

        self.user_sent = SentimentAccumulator()
        self.npc_sent = SentimentAccumulator()

        # For ANOVA: raw compound per utterance
        self.user_compounds: List[float] = []
        self.npc_compounds: List[float] = []

    def add_user_dialogue(self, text: str) -> None:
        scores = self.sia.polarity_scores(text)
        self.user_dialogues.append(text)
        self.user_sent.add(scores)
        self.user_compounds.append(float(scores["compound"]))

    def add_npc_message(self, text: str) -> None:
        scores = self.sia.polarity_scores(text)
        self.npc_messages.append(text)
        self.npc_sent.add(scores)
        self.npc_compounds.append(float(scores["compound"]))

    def report_means(self) -> str:
        u = self.user_sent.mean()
        n = self.npc_sent.mean()
        return (
            f"Quest={self.quest_id}, Difficulty={self.difficulty}\n"
            f"  User: count={self.user_sent.count}, "
            f"neg={u['neg']:.3f}, neu={u['neu']:.3f}, pos={u['pos']:.3f}, compound={u['compound']:.3f}\n"
            f"  NPC : count={self.npc_sent.count}, "
            f"neg={n['neg']:.3f}, neu={n['neu']:.3f}, pos={n['pos']:.3f}, compound={n['compound']:.3f}"
        )


class Quest0Challenging(BaseQuest):
    def __init__(self, sia: SentimentIntensityAnalyzer):
        super().__init__(0, "Challenging", sia)


class Quest0Easy(BaseQuest):
    def __init__(self, sia: SentimentIntensityAnalyzer):
        super().__init__(0, "Easy", sia)


class Quest1Challenging(BaseQuest):
    def __init__(self, sia: SentimentIntensityAnalyzer):
        super().__init__(1, "Challenging", sia)


class Quest1Easy(BaseQuest):
    def __init__(self, sia: SentimentIntensityAnalyzer):
        super().__init__(1, "Easy", sia)


class LogReader:
    QUEST_PATTERN = re.compile(r"\[Scene\]\s+\[Quest\]\s+(\d+)")
    DIFFICULTY_PATTERN = re.compile(r"\[Scene\]\s+\[Complexity\]\s+Difficulty\s*=\s*(\w+)")

    # Exclude: [DialogueUser] [User] User talk start/stop
    USER_DIALOGUE_PATTERN = re.compile(r"\[DialogueUser\]\s+(?!\[\s*User\s*\])(.+)")

    # Exclude: [MessageNpc] Evaluation: X
    NPC_MESSAGE_PATTERN = re.compile(r"\[MessageNpc\]\s+(?!Evaluation\s*:)(.+)")

    def __init__(self, folder_path: str, start_quest: int, start_difficulty: str):
        _ensure_vader_lexicon()
        self.sia = SentimentIntensityAnalyzer()

        self.folder_path = Path(folder_path)

        self.current_quest_id = start_quest
        self.current_difficulty = start_difficulty

        self.quests: Dict[Tuple[int, str], BaseQuest] = {
            (0, "Challenging"): Quest0Challenging(self.sia),
            (0, "Easy"): Quest0Easy(self.sia),
            (1, "Challenging"): Quest1Challenging(self.sia),
            (1, "Easy"): Quest1Easy(self.sia),
        }
        self.current_quest = self.quests[(self.current_quest_id, self.current_difficulty)]

    def read_all(self) -> None:
        for file_path in sorted(self.folder_path.glob("*.txt")):
            self._read_file(file_path)

    def _read_file(self, file_path: Path) -> None:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if m := self.QUEST_PATTERN.search(line):
                    self.current_quest_id = int(m.group(1))
                    self._update_current_quest()
                    continue

                if m := self.DIFFICULTY_PATTERN.search(line):
                    self.current_difficulty = m.group(1)
                    self._update_current_quest()
                    continue

                if m := self.USER_DIALOGUE_PATTERN.search(line):
                    payload = m.group(1).strip()
                    text = extract_user_text(payload)
                    if text:
                        self.current_quest.add_user_dialogue(text)
                    continue

                if m := self.NPC_MESSAGE_PATTERN.search(line):
                    text = m.group(1).strip()
                    if text:
                        self.current_quest.add_npc_message(text)
                    continue

    def _update_current_quest(self) -> None:
        key = (self.current_quest_id, self.current_difficulty)
        if key not in self.quests:
            raise ValueError(f"Unknown quest/difficulty combination encountered: {key}")
        self.current_quest = self.quests[key]

    def report_means(self) -> str:
        order = [(0, "Challenging"), (0, "Easy"), (1, "Challenging"), (1, "Easy")]
        lines = ["=== VADER Sentiment Report (mean scores) ==="]
        for k in order:
            lines.append(self.quests[k].report_means())
        return "\n".join(lines)

    def build_anova_dataframe(self, speaker: str) -> pd.DataFrame:
        """
        speaker: "user" or "npc"
        Returns a long-format dataframe with columns:
          compound, quest, difficulty
        """
        if speaker not in ("user", "npc"):
            raise ValueError("speaker must be 'user' or 'npc'")

        rows = []
        for (quest_id, difficulty), bucket in self.quests.items():
            compounds = bucket.user_compounds if speaker == "user" else bucket.npc_compounds
            for c in compounds:
                rows.append(
                    {
                        "compound": float(c),
                        "quest": str(quest_id),       # categorical
                        "difficulty": difficulty,     # categorical
                    }
                )

        return pd.DataFrame(rows)

    @staticmethod
    def two_way_anova(df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-way ANOVA with interaction:
          compound ~ C(quest) * C(difficulty)

        Returns ANOVA table (Type II sums of squares).
        """
        if df.empty:
            raise ValueError("ANOVA dataframe is empty. No utterances collected?")

        model = ols("compound ~ C(quest) * C(difficulty)", data=df).fit()
        return sm.stats.anova_lm(model, typ=2)


def pretty_print_anova(title: str, anova_table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    # Make the output a bit nicer
    out = anova_table.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.6g}")
    print(out.to_string())


if __name__ == "__main__":
    # Adjust this folder to your logs folder
    reader = LogReader(
        folder_path="logs/",
        start_quest=0,
        start_difficulty="Challenging",
    )

    reader.read_all()

    # 1) Print mean summary
    print(reader.report_means())

    # 2) Two-way ANOVA for USER compound
    df_user = reader.build_anova_dataframe("user")
    anova_user = reader.two_way_anova(df_user)
    pretty_print_anova("Two-way ANOVA (USER compound): compound ~ quest * difficulty", anova_user)

    # 3) Two-way ANOVA for NPC compound
    df_npc = reader.build_anova_dataframe("npc")
    anova_npc = reader.two_way_anova(df_npc)
    pretty_print_anova("Two-way ANOVA (NPC compound): compound ~ quest * difficulty", anova_npc)
