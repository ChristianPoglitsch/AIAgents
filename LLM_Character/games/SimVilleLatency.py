# -*- coding: utf-8 -*-
"""
SimVilleLatency.py

1) Finds log files (recursively) under a folder.
2) Extracts participant-level mean latencies for STT/LLM/TTS per condition.
3) Runs 2x3 repeated-measures ANOVA:
   scenario_type (Dyadic vs Multiparty) x component (STT, LLM, TTS)
   with difficulty averaged.

Dependencies:
  pip install pandas statsmodels
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import pandas as pd
from statsmodels.stats.anova import AnovaRM


# ----------------------------
# Timestamp parsing
# ----------------------------

TS_BRACKET = re.compile(r"^\[(?P<ts>[^\]]+)\]\s*")


def parse_ts_string(ts: str) -> Optional[datetime]:
    ts = ts.strip()
    if not ts:
        return None

    try:
        iso = ts.replace("Z", "+00:00")
        return datetime.fromisoformat(iso)
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
                parsed = datetime.combine(date.today(), parsed.time())
            return parsed
        except Exception:
            continue

    return None


def parse_timestamp_from_line(line: str) -> Optional[datetime]:
    s = line.strip()
    if not s:
        return None

    m = TS_BRACKET.match(s)
    if m:
        dt = parse_ts_string(m.group("ts"))
        if dt:
            return dt

    parts = s.split()
    if not parts:
        return None

    candidates = [parts[0]]
    if len(parts) >= 2:
        candidates.append(parts[0] + " " + parts[1])

    for cand in candidates:
        dt = parse_ts_string(cand)
        if dt:
            return dt

    return None


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class TimingAccumulator:
    durations_s: List[float] = field(default_factory=list)

    def add(self, seconds: float) -> None:
        if seconds is None or seconds < 0:
            return
        self.durations_s.append(float(seconds))

    def mean(self) -> Optional[float]:
        if not self.durations_s:
            return None
        return sum(self.durations_s) / len(self.durations_s)


@dataclass
class LatencyCell:
    participant: str
    scenario_type: str
    difficulty: str
    component: str
    latency_s: float


# ----------------------------
# Reader + RM-ANOVA
# ----------------------------

class LatencyRMReader:
    QUEST_PATTERN = re.compile(r"\[\s*scene\s*\]\s*\[\s*quest\s*\]\s*(\d+)", re.IGNORECASE)
    DIFFICULTY_PATTERN = re.compile(
        r"\[\s*scene\s*\]\s*\[\s*complexity\s*\]\s*difficulty\s*=\s*(\w+)",
        re.IGNORECASE,
    )

    STT_START_PATTERN = re.compile(r"\[\s*stt\s*\].*stt\s*start", re.IGNORECASE)
    STT_STOP_PATTERN  = re.compile(r"\[\s*stt\s*\].*stt\s*stop", re.IGNORECASE)

    LLM_START_PATTERN = re.compile(r"\[\s*llmprocessing\s*\].*llm\s*start", re.IGNORECASE)
    LLM_STOP_PATTERN  = re.compile(r"\[\s*llmprocessing\s*\].*llm\s*stop", re.IGNORECASE)

    TTS_START_PATTERN = re.compile(r".*tts\s*start", re.IGNORECASE)
    TTS_STOP_PATTERN  = re.compile(r".*tts\s*stop", re.IGNORECASE)

    def __init__(
        self,
        logs_root: str,
        extensions: Tuple[str, ...] = (".txt", ".log"),
        recursive: bool = True,
        start_quest: int = 0,
        start_difficulty: str = "Challenging",
        debug: bool = True,
    ):
        self.logs_root = Path(logs_root)
        self.extensions = tuple(e.lower() for e in extensions)
        self.recursive = recursive
        self.start_quest = start_quest
        self.start_difficulty = start_difficulty
        self.debug = debug

        self.quest_to_scenario_type = {0: "Dyadic", 1: "Multiparty"}
        self.cells: List[LatencyCell] = []

    def _iter_log_files(self) -> List[Path]:
        if not self.logs_root.exists():
            raise FileNotFoundError(f"Log folder does not exist: {self.logs_root}")

        pattern = "**/*" if self.recursive else "*"
        files = [p for p in self.logs_root.glob(pattern) if p.is_file() and p.suffix.lower() in self.extensions]
        return sorted(files)

    def discover(self) -> List[Path]:
        files = self._iter_log_files()
        if self.debug:
            print("LOG ROOT:", str(self.logs_root))
            print("EXISTS:", self.logs_root.exists())
            print("RECURSIVE:", self.recursive)
            print("EXTENSIONS:", self.extensions)
            print("FILES FOUND:", len(files))
            if len(files) == 0:
                # show what's actually in the folder (first level)
                sample = list(self.logs_root.glob("*"))
                print("TOP-LEVEL ENTRIES (first 50):")
                for p in sample[:50]:
                    print("  -", p.name)
        return files

    def read_all(self) -> None:
        files = self.discover()
        for fp in files:
            self.read_file(fp)

    def read_file(self, file_path: Path) -> None:
        participant = file_path.stem
        current_quest = self.start_quest
        current_difficulty = self.start_difficulty

        acc: Dict[Tuple[int, str, str], TimingAccumulator] = {}
        open_blocks: Dict[str, Tuple[datetime, int, str]] = {}

        def get_acc(q: int, d: str, comp: str) -> TimingAccumulator:
            key = (q, d, comp)
            if key not in acc:
                acc[key] = TimingAccumulator()
            return acc[key]

        def start_block(key: str, ts: Optional[datetime]) -> None:
            if ts is None:
                return
            open_blocks[key] = (ts, current_quest, current_difficulty)

        def stop_block(key: str, comp: str, ts: Optional[datetime]) -> None:
            if ts is None:
                return
            if key not in open_blocks:
                return
            start_ts, q0, d0 = open_blocks.pop(key)
            delta = (ts - start_ts).total_seconds()
            if delta is None or delta < 0:
                return
            get_acc(q0, d0, comp).add(delta)

        # quick per-file debug counters
        cnt = {"stt_start": 0, "stt_stop": 0, "llm_start": 0, "llm_stop": 0, "tts_start": 0, "tts_stop": 0, "ts": 0}

        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue

                m = self.QUEST_PATTERN.search(line)
                if m:
                    current_quest = int(m.group(1))
                    continue

                m = self.DIFFICULTY_PATTERN.search(line)
                if m:
                    current_difficulty = m.group(1)
                    continue

                ts = parse_timestamp_from_line(line)
                if ts is not None:
                    cnt["ts"] += 1

                if self.STT_START_PATTERN.search(line):
                    cnt["stt_start"] += 1
                    start_block("stt", ts)
                    continue
                if self.STT_STOP_PATTERN.search(line):
                    cnt["stt_stop"] += 1
                    stop_block("stt", "STT", ts)
                    continue

                if self.LLM_START_PATTERN.search(line):
                    cnt["llm_start"] += 1
                    start_block("llm", ts)
                    continue
                if self.LLM_STOP_PATTERN.search(line):
                    cnt["llm_stop"] += 1
                    stop_block("llm", "LLM", ts)
                    continue

                if self.TTS_START_PATTERN.search(line):
                    cnt["tts_start"] += 1
                    start_block("tts", ts)
                    continue
                if self.TTS_STOP_PATTERN.search(line):
                    cnt["tts_stop"] += 1
                    stop_block("tts", "TTS", ts)
                    continue

        if self.debug:
            print("\nFILE:", file_path.name)
            print("COUNTS:", cnt)

        for (q, d, comp), ta in acc.items():
            m = ta.mean()
            if m is None:
                continue
            scenario_type = self.quest_to_scenario_type.get(q, str(q))
            self.cells.append(LatencyCell(participant, scenario_type, d, comp, float(m)))

    def build_df(self) -> pd.DataFrame:
        if not self.cells:
            return pd.DataFrame(columns=["participant", "scenario_type", "difficulty", "component", "latency_s"])
        return pd.DataFrame([c.__dict__ for c in self.cells])

    @staticmethod
    def drop_incomplete_subjects(df: pd.DataFrame, within_cols: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        expected = 1
        for col in within_cols:
            expected *= df[col].nunique(dropna=True)
        counts = df.groupby("participant")[within_cols].apply(lambda x: x.drop_duplicates().shape[0])
        keep = counts[counts == expected].index
        return df[df["participant"].isin(keep)].copy()

    def run_rm_anova_2x3(self) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
        df = self.build_df()
        if df.empty:
            raise ValueError("No latency cells extracted. First fix file discovery and marker matching.")

        df2 = df.groupby(["participant", "scenario_type", "component"], as_index=False)["latency_s"].mean()
        df2 = self.drop_incomplete_subjects(df2, ["scenario_type", "component"])

        if df2["participant"].nunique() < 2:
            raise ValueError("Not enough complete participants after filtering for RM-ANOVA.")

        aov = AnovaRM(
            data=df2,
            depvar="latency_s",
            subject="participant",
            within=["scenario_type", "component"],
        ).fit()

        table = aov.anova_table.reset_index().rename(columns={"index": "Effect"})
        latex = table.to_latex(index=False, float_format="%.4f")
        return table, latex, df2


def main() -> None:
    # IMPORTANT: set this to the correct folder that actually contains the logs
    logs_root = r"D:\Development\EmpathicAgents\LLM_Character\games\logs"

    reader = LatencyRMReader(
        logs_root=logs_root,
        extensions=(".txt", ".log"),
        recursive=True,
        debug=True,
    )

    reader.read_all()

    df_raw = reader.build_df()
    print("\nRAW participant-level latency cells (first 30 rows):")
    print(df_raw.head(30).to_string(index=False))

    table, latex, df_used = reader.run_rm_anova_2x3()
    print("\nParticipants used for ANOVA:", df_used["participant"].nunique())
    print("\nRM-ANOVA (2x3): scenario_type x component (difficulty averaged)")
    print(table.to_string(index=False))

    print("\nLaTeX ANOVA table:")
    print(latex)


if __name__ == "__main__":
    main()