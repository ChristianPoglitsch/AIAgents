"""
log_sentiment_anova.py

Reads all .txt log files from a folder, routes [DialogueUser] and [MessageNpc] text into
one of four (Quest x Difficulty) buckets, runs VADER sentiment per utterance, and then
computes TWO-WAY ANOVA (Quest x Difficulty) on VADER compound scores for:
  - User utterances
  - NPC utterances

Additionally:
- Extracts per-participant (per file) mean latency for STT/LLM/TTS per condition,
  and runs RM-ANOVA for latency:
    - 2x3: ScenarioType (Dyadic/Multiparty) x Component (STT/LLM/TTS), tone averaged
    - optional 2x2x3: ScenarioType x Tone x Component

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
import statistics
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

_TS_BRACKET = re.compile(r"^\[(?P<ts>[^\]]+)\]\s+")


def parse_timestamp_from_line(line: str) -> Optional[datetime]:
    """
    Tries to parse a timestamp at the beginning of the line.
    Supports:
      - [2025-01-31 12:34:56.789] ...
      - [2025-01-31T12:34:56.789Z] ...
      - [12:34:56.789] ...
      - 2025-01-31 12:34:56.789 ...
      - 12:34:56.789 ...
    If only time is present, uses today's date.
    Returns None if not parseable.
    """
    s = line.strip()
    if not s:
        return None

    # 1) [ ... ] prefix?
    m = _TS_BRACKET.match(s)
    if m:
        ts = m.group("ts").strip()
        dt = _parse_ts_string(ts)
        if dt:
            return dt

    # 2) unbracketed: try first token(s)
    first = s.split(" ", 2)
    candidates = []
    if len(first) >= 1:
        candidates.append(first[0])
    if len(first) >= 2:
        candidates.append(first[0] + " " + first[1])

    for cand in candidates:
        dt = _parse_ts_string(cand)
        if dt:
            return dt

    return None


def _parse_ts_string(ts: str) -> Optional[datetime]:
    ts = ts.strip()
    if not ts:
        return None

    # ISO 8601 (also with Z)
    try:
        iso = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        return dt
    except Exception:
        pass

    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%H:%M:%S.%f",
        "%H:%M:%S",
    ]
    for fmt in fmts:
        try:
            parsed = datetime.strptime(ts, fmt)
            if fmt.startswith("%H"):
                today = date.today()
                parsed = datetime.combine(today, parsed.time())
            return parsed
        except Exception:
            continue

    return None


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


@dataclass
class SentimentAccumulator:
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


@dataclass
class TimingAccumulator:
    durations_s: List[float] = field(default_factory=list)

    def add_duration(self, seconds: float) -> None:
        if seconds is None:
            return
        if seconds < 0:
            return
        self.durations_s.append(float(seconds))

    def summary(self) -> Dict[str, float]:
        if not self.durations_s:
            return {
                "count": 0,
                "mean_s": 0.0,
                "median_s": 0.0,
                "std_s": 0.0,
                "min_s": 0.0,
                "max_s": 0.0,
            }
        xs = self.durations_s
        mean = sum(xs) / len(xs)
        median = statistics.median(xs)
        std = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        return {
            "count": float(len(xs)),
            "mean_s": mean,
            "median_s": median,
            "std_s": std,
            "min_s": min(xs),
            "max_s": max(xs),
        }


@dataclass
class ParticipantLatency:
    participant: str
    # key: (quest_id, difficulty, component) -> mean latency seconds
    means: Dict[Tuple[int, str, str], float] = field(default_factory=dict)


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

        self.user_compounds: List[float] = []
        self.npc_compounds: List[float] = []

        self.user_talk_time = TimingAccumulator()
        self.stt_time = TimingAccumulator()
        self.llm_time = TimingAccumulator()
        self.tts_time = TimingAccumulator()

        self.has_eval_1: bool = False

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

    def merge_from(self, other: "BaseQuest") -> None:
        """Merge another bucket into this one (for global aggregation across files)."""
        self.user_dialogues.extend(other.user_dialogues)
        self.npc_messages.extend(other.npc_messages)

        self.user_compounds.extend(other.user_compounds)
        self.npc_compounds.extend(other.npc_compounds)

        self.user_sent.count += other.user_sent.count
        for k in self.user_sent.sums:
            self.user_sent.sums[k] += other.user_sent.sums.get(k, 0.0)

        self.npc_sent.count += other.npc_sent.count
        for k in self.npc_sent.sums:
            self.npc_sent.sums[k] += other.npc_sent.sums.get(k, 0.0)

        self.user_talk_time.durations_s.extend(other.user_talk_time.durations_s)
        self.stt_time.durations_s.extend(other.stt_time.durations_s)
        self.llm_time.durations_s.extend(other.llm_time.durations_s)
        self.tts_time.durations_s.extend(other.tts_time.durations_s)

        self.has_eval_1 = self.has_eval_1 or other.has_eval_1

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

    def report_timings(self) -> str:
        def fmt(name: str, acc: TimingAccumulator) -> str:
            s = acc.summary()
            return (
                f"    {name}: n={int(s['count'])}, mean={s['mean_s']:.3f}s, "
                f"median={s['median_s']:.3f}s, std={s['std_s']:.3f}s, "
                f"min={s['min_s']:.3f}s, max={s['max_s']:.3f}s"
            )

        lines = [
            f"Quest={self.quest_id}, Difficulty={self.difficulty}",
            fmt("UserTalk", self.user_talk_time),
            fmt("STT", self.stt_time),
            fmt("LLM", self.llm_time),
            fmt("TTS", self.tts_time),
            f"    Evaluation:1 found? {'YES' if self.has_eval_1 else 'NO'}",
        ]
        return "\n".join(lines)


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

    USER_TALK_START_PATTERN = re.compile(r"\[DialogueUser\]\s+\[User\]\s+User talk start")
    USER_TALK_STOP_PATTERN = re.compile(r"\[DialogueUser\]\s+\[User\]\s+User talk stop")

    STT_START_PATTERN = re.compile(r"\[STT\]\s+STT start")
    STT_STOP_PATTERN = re.compile(r"\[STT\]\s+STT stop")

    LLM_START_PATTERN = re.compile(r"\[LlmProcessing\]\s+\[LlmProcessing\]\s+LLM start chat completion")
    LLM_STOP_PATTERN = re.compile(r"\[LlmProcessing\]\s+\[LlmProcessing\]\s+LLM stop chat completion")

    # NOTE: In your logs TTS seems to be tagged with [STT] too
    TTS_START_PATTERN = re.compile(r"\[STT\]\s+TTS start")
    TTS_STOP_PATTERN = re.compile(r"\[STT\]\s+TTS stop")

    EVAL1_PATTERN = re.compile(r"\[MessageNpc\]\s+Evaluation\s*:\s*1\b")

    USER_DIALOGUE_PATTERN = re.compile(r"\[DialogueUser\]\s+(?!\[\s*User\s*\])(.+)")
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

        self.participant_latencies: List[ParticipantLatency] = []

        # Mapping (adjust if your experiment differs)
        self.quest_to_scenario_type = {0: "Dyadic", 1: "Multiparty"}
        self.difficulty_to_tone = {"Easy": "Positive", "Challenging": "Negative"}

    def read_all(self) -> None:
        for file_path in sorted(self.folder_path.glob("*.txt")):
            self._read_file(file_path)

    def _read_file(self, file_path: Path) -> None:
        participant_id = file_path.stem

        # Local buckets (per participant)
        quests_local: Dict[Tuple[int, str], BaseQuest] = {
            (0, "Challenging"): Quest0Challenging(self.sia),
            (0, "Easy"): Quest0Easy(self.sia),
            (1, "Challenging"): Quest1Challenging(self.sia),
            (1, "Easy"): Quest1Easy(self.sia),
        }

        current_quest_id = self.current_quest_id
        current_difficulty = self.current_difficulty
        current_bucket = quests_local[(current_quest_id, current_difficulty)]

        open_blocks: Dict[str, Tuple[datetime, Tuple[int, str]]] = {}

        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                if m := self.QUEST_PATTERN.search(line):
                    current_quest_id = int(m.group(1))
                    key = (current_quest_id, current_difficulty)
                    if key in quests_local:
                        current_bucket = quests_local[key]
                    continue

                if m := self.DIFFICULTY_PATTERN.search(line):
                    current_difficulty = m.group(1)
                    key = (current_quest_id, current_difficulty)
                    if key in quests_local:
                        current_bucket = quests_local[key]
                    continue

                if m := self.USER_DIALOGUE_PATTERN.search(line):
                    payload = m.group(1).strip()
                    text = extract_user_text(payload)
                    if text:
                        current_bucket.add_user_dialogue(text)
                    continue

                if m := self.NPC_MESSAGE_PATTERN.search(line):
                    text = m.group(1).strip()
                    if text:
                        current_bucket.add_npc_message(text)
                    continue

                ts = parse_timestamp_from_line(line)

                if self.EVAL1_PATTERN.search(line):
                    current_bucket.has_eval_1 = True

                def _start_block(key: str) -> None:
                    if ts is None:
                        return
                    open_blocks[key] = (ts, (current_quest_id, current_difficulty))

                def _stop_block(key: str) -> None:
                    if ts is None:
                        return
                    if key not in open_blocks:
                        return
                    start_ts, start_bucket_key = open_blocks.pop(key)
                    delta = (ts - start_ts).total_seconds()
                    bucket = quests_local.get(start_bucket_key)
                    if bucket is None or delta is None or delta < 0:
                        return
                    if key == "user_talk":
                        bucket.user_talk_time.add_duration(delta)
                    elif key == "stt":
                        bucket.stt_time.add_duration(delta)
                    elif key == "llm":
                        bucket.llm_time.add_duration(delta)
                    elif key == "tts":
                        bucket.tts_time.add_duration(delta)

                if self.USER_TALK_START_PATTERN.search(line):
                    _start_block("user_talk")
                    continue
                if self.USER_TALK_STOP_PATTERN.search(line):
                    _stop_block("user_talk")
                    continue

                if self.STT_START_PATTERN.search(line):
                    _start_block("stt")
                    continue
                if self.STT_STOP_PATTERN.search(line):
                    _stop_block("stt")
                    continue

                if self.LLM_START_PATTERN.search(line):
                    _start_block("llm")
                    continue
                if self.LLM_STOP_PATTERN.search(line):
                    _stop_block("llm")
                    continue

                if self.TTS_START_PATTERN.search(line):
                    _start_block("tts")
                    continue
                if self.TTS_STOP_PATTERN.search(line):
                    _stop_block("tts")
                    continue

        # Participant-level mean latencies
        pl = ParticipantLatency(participant=participant_id)
        for (qid, diff), bucket in quests_local.items():
            if bucket.stt_time.durations_s:
                pl.means[(qid, diff, "STT")] = bucket.stt_time.summary()["mean_s"]
            if bucket.llm_time.durations_s:
                pl.means[(qid, diff, "LLM")] = bucket.llm_time.summary()["mean_s"]
            if bucket.tts_time.durations_s:
                pl.means[(qid, diff, "TTS")] = bucket.tts_time.summary()["mean_s"]
        self.participant_latencies.append(pl)

        # Merge local buckets into global ones so your original sentiment ANOVAs still work
        for key, local_bucket in quests_local.items():
            self.quests[key].merge_from(local_bucket)

    def report_timings(self) -> str:
        order = [(0, "Challenging"), (0, "Easy"), (1, "Challenging"), (1, "Easy")]
        lines = ["=== Timing Report (durations in seconds) + Evaluation:1 flag ==="]
        for k in order:
            lines.append(self.quests[k].report_timings())
        return "\n".join(lines)

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
                        "quest": str(quest_id),
                        "difficulty": difficulty,
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def two_way_anova(df: pd.DataFrame) -> pd.DataFrame:
        """
        Two-way ANOVA with interaction:
          compound ~ C(quest) * C(difficulty)
        Returns ANOVA table (Type II SS).
        """
        if df.empty:
            raise ValueError("ANOVA dataframe is empty. No utterances collected?")
        model = ols("compound ~ C(quest) * C(difficulty)", data=df).fit()
        return sm.stats.anova_lm(model, typ=2)

    # ---------------------- LATENCY RM-ANOVA ----------------------

    def build_latency_dataframe(self) -> pd.DataFrame:
        """
        Long-format DF:
          participant, scenario_type, tone, component, latency_s
        """
        rows = []
        for pl in self.participant_latencies:
            for (qid, diff, comp), mean_s in pl.means.items():
                scenario_type = self.quest_to_scenario_type.get(qid, str(qid))
                tone = self.difficulty_to_tone.get(diff, diff)
                rows.append(
                    {
                        "participant": pl.participant,
                        "scenario_type": scenario_type,
                        "tone": tone,
                        "component": comp,
                        "latency_s": float(mean_s),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _drop_incomplete_subjects(df: pd.DataFrame, within_cols: List[str]) -> pd.DataFrame:
        """
        Drops participants that do not have a full factorial set of within-subject cells.
        This prevents AnovaRM from failing on missing cells.
        """
        if df.empty:
            return df

        # expected number of cells per participant
        expected = 1
        for col in within_cols:
            expected *= df[col].nunique(dropna=True)

        counts = (
            df.groupby("participant")[within_cols]
              .apply(lambda x: x.drop_duplicates().shape[0])
        )
        keep = counts[counts == expected].index
        return df[df["participant"].isin(keep)].copy()

    @staticmethod
    def rm_anova_latency_2x3(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        2x3 RM-ANOVA: scenario_type x component (tone averaged).
        """
        if df.empty:
            raise ValueError("Latency dataframe is empty. Did you parse timing markers?")

        # Average over tone
        df2 = (
            df.groupby(["participant", "scenario_type", "component"], as_index=False)["latency_s"]
              .mean()
        )

        df2 = LogReader._drop_incomplete_subjects(df2, ["scenario_type", "component"])
        if df2["participant"].nunique() < 2:
            raise ValueError("Not enough complete participants for 2x3 RM-ANOVA after filtering.")

        aov = AnovaRM(
            data=df2,
            depvar="latency_s",
            subject="participant",
            within=["scenario_type", "component"],
        ).fit()

        table = aov.anova_table.reset_index().rename(columns={"index": "Effect"})
        latex = table.to_latex(index=False, float_format="%.4f")
        return table, latex

    @staticmethod
    def rm_anova_latency_2x2x3(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        2x2x3 RM-ANOVA: scenario_type x tone x component.
        """
        if df.empty:
            raise ValueError("Latency dataframe is empty. Did you parse timing markers?")

        df2 = LogReader._drop_incomplete_subjects(df, ["scenario_type", "tone", "component"])
        if df2["participant"].nunique() < 2:
            raise ValueError("Not enough complete participants for 2x2x3 RM-ANOVA after filtering.")

        aov = AnovaRM(
            data=df2,
            depvar="latency_s",
            subject="participant",
            within=["scenario_type", "tone", "component"],
        ).fit()

        table = aov.anova_table.reset_index().rename(columns={"index": "Effect"})
        latex = table.to_latex(index=False, float_format="%.4f")
        return table, latex


def pretty_print_anova(title: str, anova_table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    out = anova_table.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda x: f"{x:.6g}")
    print(out.to_string())


if __name__ == "__main__":
    reader = LogReader(
        folder_path=r"C:\Development\EmpathicAgents\LLM_Character\games\logs/",
        start_quest=0,
        start_difficulty="Challenging",
    )

    reader.read_all()

    # ---- Latency RM-ANOVA ----
    df_lat = reader.build_latency_dataframe()
    print("\n=== Latency long DF preview ===")
    print(df_lat.head(12).to_string(index=False))

    # 2x3 RM-ANOVA
    lat_table_2x3, lat_latex_2x3 = reader.rm_anova_latency_2x3(df_lat)
    print("\n=== 2x3 RM-ANOVA (Latency): scenario_type x component (tone averaged) ===")
    print(lat_table_2x3.to_string(index=False))
    print("\n=== LaTeX (2x3 RM-ANOVA table) ===")
    print(lat_latex_2x3)

    # Optional 2x2x3 RM-ANOVA
    try:
        lat_table_2x2x3, lat_latex_2x2x3 = reader.rm_anova_latency_2x2x3(df_lat)
        print("\n=== 2x2x3 RM-ANOVA (Latency): scenario_type x tone x component ===")
        print(lat_table_2x2x3.to_string(index=False))
        print("\n=== LaTeX (2x2x3 RM-ANOVA table) ===")
        print(lat_latex_2x2x3)
    except Exception as e:
        print(f"\n[WARN] Could not run 2x2x3 RM-ANOVA: {e}")

    # ---- Sentiment (global across all files) ----
    print(reader.report_means())
    print(reader.report_timings())

    df_user = reader.build_anova_dataframe("user")
    anova_user = reader.two_way_anova(df_user)
    pretty_print_anova("Two-way ANOVA (USER compound): compound ~ quest * difficulty", anova_user)

    df_npc = reader.build_anova_dataframe("npc")
    anova_npc = reader.two_way_anova(df_npc)
    pretty_print_anova("Two-way ANOVA (NPC compound): compound ~ quest * difficulty", anova_npc)
