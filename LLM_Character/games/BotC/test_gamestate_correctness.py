import unittest
import types

# Updated import: new generic function that supports both Good and Evil speakers
from botc_load_tree import analyze_all_pairs_gamestate_correctness_for_alignment


class DummyPlayer:
    def __init__(self, alignment, role):
        self.alignment = alignment
        self.role = role


def make_game_state(players_dict, feature_matrix):
    gs = types.SimpleNamespace()
    gs.active_players = players_dict
    gs.features = types.SimpleNamespace(features=feature_matrix)
    return gs


class TestGameStateCorrectness(unittest.TestCase):
    def test_analyze_all_pairs_counts_and_accuracy_good_speakers(self):
        # Ground truth: 3 players, Good speakers are A and C
        # obs = speakers * (P - 1) = 2 * 2 = 4
        players = {
            "A": DummyPlayer("Good", "Empath"),
            "B": DummyPlayer("Evil", "Poisoner"),
            "C": DummyPlayer("Good", "Ravenkeeper"),
        }

        feature_matrix = {
            "A": {
                "B": [0, "I think B is EVIL and likely the Poisoner."],       # align correct, role correct
                "C": [0, "C seems GOOD; probably the Ravenkeeper."],          # align correct, role correct
            },
            "B": {
                "A": [0, "A could be good and evil depending... also maybe Empath"],
                "C": [0, ""],
            },
            "C": {
                "A": [0, "A is good but I'm calling them Poisoner."],         # align correct, role WRONG (true Empath)
                "B": [0, "B is evil."],                                       # align correct, role missing
            },
        }

        rep = analyze_all_pairs_gamestate_correctness_for_alignment(
            make_game_state(players, feature_matrix),
            "Good",
        )

        self.assertEqual(rep["P"], 3)
        self.assertEqual(rep["speaker_count"], 2)
        self.assertEqual(rep["obs"], 4)

        # Alignment correctness (Good speakers):
        # A->B correct (1)
        # A->C correct (2)
        # C->A correct (3)
        # C->B correct (4)
        self.assertEqual(rep["align_correct"], 4)

        # Split by target alignment:
        # Targets that are Good: A and C
        # Observations to Good targets: A->C, C->A => 2, both predicted "Good" => 2 correct
        self.assertEqual(rep["align_obs_good"], 2)
        self.assertEqual(rep["align_correct_good"], 2)

        # Observations to Evil targets: A->B, C->B => 2, both predicted "Evil" => 2 correct
        self.assertEqual(rep["align_obs_evil"], 2)
        self.assertEqual(rep["align_correct_evil"], 2)

        # Role correctness (Good speakers):
        # A->B Poisoner correct (1)
        # A->C Ravenkeeper correct (2)
        # C->A predicts Poisoner but true Empath => incorrect
        # C->B no role => incorrect
        self.assertEqual(rep["role_correct"], 2)

    def test_analyze_all_pairs_counts_and_accuracy_evil_speakers(self):
        # Same setup, but now only Evil speakers count (B only)
        # obs = 1 * (3-1) = 2 (B->A, B->C)
        players = {
            "A": DummyPlayer("Good", "Empath"),
            "B": DummyPlayer("Evil", "Poisoner"),
            "C": DummyPlayer("Good", "Ravenkeeper"),
        }

        feature_matrix = {
            "A": {
                "B": [0, "evil Poisoner"],
                "C": [0, "good Ravenkeeper"],
            },
            "B": {
                "A": [0, "A is GOOD and the Empath."],        # align correct, role correct
                "C": [0, "C seems GOOD."],                    # align correct, role missing
            },
            "C": {
                "A": [0, "good Empath"],
                "B": [0, "evil Poisoner"],
            },
        }

        rep = analyze_all_pairs_gamestate_correctness_for_alignment(
            make_game_state(players, feature_matrix),
            "Evil",
        )

        self.assertEqual(rep["P"], 3)
        self.assertEqual(rep["speaker_count"], 1)
        self.assertEqual(rep["obs"], 2)

        # Alignment: B->A Good (correct), B->C Good (correct) => 2
        self.assertEqual(rep["align_correct"], 2)

        # Targets: both A and C are Good => align_obs_good=2, align_correct_good=2, evil target obs=0
        self.assertEqual(rep["align_obs_good"], 2)
        self.assertEqual(rep["align_correct_good"], 2)
        self.assertEqual(rep["align_obs_evil"], 0)
        self.assertEqual(rep["align_correct_evil"], 0)

        # Role: B->A Empath correct (1), B->C role missing (0) => 1
        self.assertEqual(rep["role_correct"], 1)

    def test_truth_extraction_fallback_to_players_dict(self):
        # If active_players missing, it should fall back to game_state.players
        # Here: only A is Good => obs = 1 * (2-1) = 1
        gs = types.SimpleNamespace()
        gs.active_players = None
        gs.players = {
            "A": DummyPlayer("Good", "Empath"),
            "B": DummyPlayer("Evil", "Poisoner"),
        }
        gs.features = types.SimpleNamespace(
            features={
                "A": {"B": [0, "evil Poisoner"]},
                "B": {"A": [0, "good Empath"]},  # will be counted if we test Evil; ignored here (Good)
            }
        )

        rep = analyze_all_pairs_gamestate_correctness_for_alignment(gs, "Good")
        self.assertEqual(rep["P"], 2)
        self.assertEqual(rep["speaker_count"], 1)
        self.assertEqual(rep["obs"], 1)
        self.assertEqual(rep["align_correct"], 1)
        self.assertEqual(rep["role_correct"], 1)

        # Split: only target is B (Evil)
        self.assertEqual(rep["align_obs_good"], 0)
        self.assertEqual(rep["align_correct_good"], 0)
        self.assertEqual(rep["align_obs_evil"], 1)
        self.assertEqual(rep["align_correct_evil"], 1)

    def test_alignment_normalization_accepts_g_e_prefix(self):
        # alignment values like "g" / "e" should normalize to Good/Evil
        # Only A is Good => obs = 1
        gs = types.SimpleNamespace()
        gs.active_players = {
            "A": DummyPlayer("g", "Empath"),
            "B": DummyPlayer("e", "Poisoner"),
        }
        gs.features = types.SimpleNamespace(
            features={
                "A": {"B": [0, "evil"]},  # correct alignment
                "B": {"A": [0, "good"]},  # used only when testing Evil speakers
            }
        )

        rep_good = analyze_all_pairs_gamestate_correctness_for_alignment(gs, "Good")
        self.assertEqual(rep_good["speaker_count"], 1)
        self.assertEqual(rep_good["obs"], 1)
        self.assertEqual(rep_good["align_correct"], 1)
        self.assertEqual(rep_good["align_obs_evil"], 1)
        self.assertEqual(rep_good["align_correct_evil"], 1)

        rep_evil = analyze_all_pairs_gamestate_correctness_for_alignment(gs, "Evil")
        self.assertEqual(rep_evil["speaker_count"], 1)
        self.assertEqual(rep_evil["obs"], 1)
        self.assertEqual(rep_evil["align_correct"], 1)
        self.assertEqual(rep_evil["align_obs_good"], 1)
        self.assertEqual(rep_evil["align_correct_good"], 1)


if __name__ == "__main__":
    unittest.main()