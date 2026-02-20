import pickle
import re

from botc_base import *
from botc import *

GOOD_RE = re.compile(r"\bgood\b", re.IGNORECASE)
EVIL_RE = re.compile(r"\bevil\b", re.IGNORECASE)

def _get_winner_from_state(game_state):
    """
    Returns: "Good", "Evil", or None (unknown)
    Uses the same mechanism as your previous working script.
    """
    try:
        return "Good" if game_state.good_win() else "Evil"
    except Exception:
        return None

# ============================================================
# Ground truth extraction
# ============================================================

def _get_true_player_info(game_state):
    truth = {}

    ap = getattr(game_state, "active_players", None)
    if isinstance(ap, dict) and ap:
        for name, pobj in ap.items():
            truth[str(name)] = {
                "alignment": getattr(pobj, "alignment", None),
                "role": getattr(pobj, "role", None),
            }

    if not truth:
        pl = getattr(game_state, "players", None)
        if isinstance(pl, dict) and pl:
            for name, pobj in pl.items():
                truth[str(name)] = {
                    "alignment": getattr(pobj, "alignment", None),
                    "role": getattr(pobj, "role", None),
                }

    # Normalize alignment + role
    for _, info in truth.items():
        a = info.get("alignment")
        if a is not None and not isinstance(a, str):
            a = str(a)
        if isinstance(a, str):
            aa = a.strip().lower()
            if "good" in aa or aa.startswith("g"):
                info["alignment"] = "Good"
            elif "evil" in aa or aa.startswith("e"):
                info["alignment"] = "Evil"

        r = info.get("role")
        if r is not None and not isinstance(r, str):
            r = str(r)
        if isinstance(r, str):
            info["role"] = r.strip()

    return truth


def _get_last_private_info_matrix(game_state):
    feats = getattr(getattr(game_state, "features", None), "features", None)
    if not isinstance(feats, dict):
        return {}

    out = {}
    for speaker, targets in feats.items():
        out[str(speaker)] = {}
        if not isinstance(targets, dict):
            continue
        for target, vec in targets.items():
            if isinstance(vec, (list, tuple)) and len(vec) >= 2:
                out[str(speaker)][str(target)] = "" if vec[1] is None else str(vec[1])
            else:
                out[str(speaker)][str(target)] = ""
    return out


def _build_role_regexes_from_truth(truth):
    roles = sorted(
        {
            info.get("role")
            for info in truth.values()
            if isinstance(info.get("role"), str) and info.get("role")
        }
    )
    role_res = []
    for r in roles:
        pat = re.escape(r).replace(r"\ ", r"\s+")
        role_res.append((r, re.compile(rf"\b{pat}\b", re.IGNORECASE)))
    return role_res


def _predict_alignment_from_text(text):
    has_good = bool(GOOD_RE.search(text))
    has_evil = bool(EVIL_RE.search(text))
    if has_evil and not has_good:
        return "Evil"
    if has_good and not has_evil:
        return "Good"
    return None


def _predict_role_from_text(text, role_res):
    for role_name, rx in role_res:
        if rx.search(text):
            return role_name
    return None


# ============================================================
# Winner extraction (NEW)
# ============================================================

def _normalize_team(x):
    if x is None:
        return None
    if isinstance(x, bool):
        # ambiguous unless you know which team it refers to, so don't guess
        return None
    if not isinstance(x, str):
        x = str(x)
    s = x.strip().lower()
    if "good" in s or s in {"g", "town", "townsfolk"}:
        return "Good"
    if "evil" in s or s in {"e", "demon", "minion"}:
        return "Evil"
    return None


def _get_winner_alignment(game_state):
    """
    Best-effort winner detection across common field names / shapes.
    Returns: "Good", "Evil", or None.
    """

    # 1) Try common direct attributes
    for attr in (
        "winner_alignment",
        "winning_alignment",
        "winner",
        "winning_team",
        "winning_side",
        "win_team",
        "result",
        "outcome",
        "game_result",
        "final_result",
    ):
        val = getattr(game_state, attr, None)
        norm = _normalize_team(val)
        if norm:
            return norm

    # 2) Try dict-like payloads
    for attr in ("result", "outcome", "game_result", "final_result", "summary"):
        val = getattr(game_state, attr, None)
        if isinstance(val, dict):
            for k in ("winner", "winning_team", "winning_side", "alignment", "team"):
                norm = _normalize_team(val.get(k))
                if norm:
                    return norm

    # 3) Try boolean flags (only if explicit)
    for good_attr in ("good_won", "good_win", "is_good_win"):
        v = getattr(game_state, good_attr, None)
        if v is True:
            return "Good"
    for evil_attr in ("evil_won", "evil_win", "is_evil_win"):
        v = getattr(game_state, evil_attr, None)
        if v is True:
            return "Evil"

    return None


# ============================================================
# Core evaluation (for ANY speaker alignment)
# ============================================================

def analyze_all_pairs_gamestate_correctness_for_alignment(game_state, speaker_alignment):
    """
    Computes belief accuracy ONLY for speakers of given alignment.

    Returns:
      obs
      align_correct
      role_correct
      align_obs_good
      align_correct_good
      align_obs_evil
      align_correct_evil
      speaker_count
      P
    """
    truth = _get_true_player_info(game_state)
    if not truth:
        return {"error": "No ground truth extracted"}

    players = sorted(truth.keys())
    P = len(players)

    speakers = [p for p in players if truth.get(p, {}).get("alignment") == speaker_alignment]
    if not speakers:
        return {"error": f"No {speaker_alignment} speakers found"}

    last_private = _get_last_private_info_matrix(game_state)
    role_res = _build_role_regexes_from_truth(truth)

    obs = 0
    align_correct = 0
    role_correct = 0

    align_obs_good = 0
    align_correct_good = 0
    align_obs_evil = 0
    align_correct_evil = 0

    for s in speakers:
        for t in players:
            if s == t:
                continue
            obs += 1

            txt = ""
            if s in last_private:
                txt = last_private[s].get(t, "") or ""

            true_align = truth[t].get("alignment")
            true_role = truth[t].get("role")

            pred_align = _predict_alignment_from_text(txt)
            pred_role = _predict_role_from_text(txt, role_res)

            # Alignment overall + per target team
            if true_align == "Good":
                align_obs_good += 1
                if pred_align == "Good":
                    align_correct += 1
                    align_correct_good += 1
            elif true_align == "Evil":
                align_obs_evil += 1
                if pred_align == "Evil":
                    align_correct += 1
                    align_correct_evil += 1

            # Role overall
            if pred_role is not None and isinstance(true_role, str) and true_role:
                if pred_role.lower() == true_role.lower():
                    role_correct += 1

    return {
        "P": P,
        "speaker_count": len(speakers),
        "obs": obs,
        "align_correct": align_correct,
        "role_correct": role_correct,
        "align_obs_good": align_obs_good,
        "align_correct_good": align_correct_good,
        "align_obs_evil": align_obs_evil,
        "align_correct_evil": align_correct_evil,
    }


def _acc(c, t):
    return (c / t) if t else 0.0


# ============================================================
# Main
# ============================================================
def main():
    with open("2026_r6.pkl", "rb") as f:
        mcts_all = pickle.load(f)

    total_stats = {
        "Good": {"obs": 0, "align": 0, "role": 0,
                 "good_obs": 0, "good_cor": 0,
                 "evil_obs": 0, "evil_cor": 0},
        "Evil": {"obs": 0, "align": 0, "role": 0,
                 "good_obs": 0, "good_cor": 0,
                 "evil_obs": 0, "evil_cor": 0},
    }

    # NEW: win counters (episode-level)
    win_counts = {"Good": 0, "Evil": 0, "Unknown": 0}

    for episode, mcts in enumerate(mcts_all, 1):
        root = mcts.get_root_node()
        terminals = mcts.get_all_terminal_nodes(root)
        node = terminals[0] if terminals else root

        print(f"\n===== Episode {episode} =====")

        # NEW: winner (same as your working script)
        winner = _get_winner_from_state(node.state)
        if winner is None:
            win_counts["Unknown"] += 1
            print("Winner: UNKNOWN")
        else:
            win_counts[winner] += 1
            print(f"Winner: {winner}")

        for alignment in ["Good", "Evil"]:
            rep = analyze_all_pairs_gamestate_correctness_for_alignment(node.state, alignment)
            if "error" in rep:
                print(f"{alignment}: {rep['error']}")
                continue

            total_stats[alignment]["obs"] += rep["obs"]
            total_stats[alignment]["align"] += rep["align_correct"]
            total_stats[alignment]["role"] += rep["role_correct"]
            total_stats[alignment]["good_obs"] += rep["align_obs_good"]
            total_stats[alignment]["good_cor"] += rep["align_correct_good"]
            total_stats[alignment]["evil_obs"] += rep["align_obs_evil"]
            total_stats[alignment]["evil_cor"] += rep["align_correct_evil"]

            print(
                f"{alignment} Speakers ({rep['speaker_count']} players) | "
                f"Align={rep['align_correct']}/{rep['obs']} ({_acc(rep['align_correct'], rep['obs']):.3f}) | "
                f"Role={rep['role_correct']}/{rep['obs']} ({_acc(rep['role_correct'], rep['obs']):.3f})"
            )

    print("\n================ FINAL SUMMARY ================")
    for alignment in ["Good", "Evil"]:
        s = total_stats[alignment]
        print(f"\n--- {alignment} Speakers ---")
        print(f"Alignment Accuracy: {s['align']}/{s['obs']} = {_acc(s['align'], s['obs']):.4f}")
        print(f"Role Accuracy: {s['role']}/{s['obs']} = {_acc(s['role'], s['obs']):.4f}")
        print(
            f"Against Good targets: {s['good_cor']}/{s['good_obs']} "
            f"= {_acc(s['good_cor'], s['good_obs']):.4f}"
        )
        print(
            f"Against Evil targets: {s['evil_cor']}/{s['evil_obs']} "
            f"= {_acc(s['evil_cor'], s['evil_obs']):.4f}"
        )

    # NEW: Win rate summary
    total_eps = len(mcts_all)
    known = win_counts["Good"] + win_counts["Evil"]
    print("\n--- Win Rates (by episode) ---")
    print(f"Episodes: {total_eps}")
    print(f"Good wins: {win_counts['Good']}")
    print(f"Evil wins: {win_counts['Evil']}")
    print(f"Unknown result episodes: {win_counts['Unknown']}")
    print(f"Good win rate (known only): {_acc(win_counts['Good'], known):.4f}")
    print(f"Evil win rate (known only): {_acc(win_counts['Evil'], known):.4f}")
    
if __name__ == "__main__":
    main()
    