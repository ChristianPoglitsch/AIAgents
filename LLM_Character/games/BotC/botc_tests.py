import json
import time

from botc_base import ConversationManager
from botc import BloodOnTheClocktowerState
from botc import roles

# --- --- main --- ---

log = []
start_time = time.time()  # Start timing

folder_path = 'training_botc'
conversationManager = ConversationManager()

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

print('Start Game')
game_state = BloodOnTheClocktowerState(['A', 'B', 'C', 'D', 'E'], roles)
# Create a dummy ConversationManager and add a conversation.
conv_manager = ConversationManager()

game_state.update_game_state()
next_player = game_state.get_player()

# firsrt night phase
action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'D', 'Target': 'A', 'Effect': 'Poison'}
game_state.apply_action(conv_manager, action, None, False, False)

# first day

# nomination
game_state.update_game_state()
game_state.add_next_player("A")
next_player = game_state.get_player()
action = {"type": "Nominate", "Speaker": next_player, "Nominee": "B"}
game_state.apply_action(conv_manager, action, None, False, False)

# vote
next_player = game_state.get_player()
action = {"type": "No Action", "Speaker": next_player}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()
next_player = game_state.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()
next_player = game_state.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()
next_player = game_state.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()
next_player = game_state.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()

# night actions
next_player = game_state.get_player()

action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'D', 'Target': 'A', 'Effect': 'Poison'}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()

action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'E', 'Target': 'C'}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()

print('Game over: ' + str(game_state.is_terminal()))


# second day
team_red_win = True

if team_red_win:
    # nomination
    game_state.add_next_player("A")
    next_player = game_state.get_player()
    action = {"type": "Nominate", "Speaker": next_player, "Nominee": "D"}
    game_state.apply_action(conv_manager, action, None, False, False)

    # vote
    next_player = game_state.get_player()
    action = {"type": "Vote", "Speaker": next_player, "Target": "D"}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()
    next_player = game_state.get_player()
    action = {"type": "No Action", "Speaker": next_player}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()
    next_player = game_state.get_player()
    action = {"type": "No Action", "Speaker": next_player}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()

    # night actions
    next_player = game_state.get_player()

    action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': next_player, 'Target': 'A', 'Effect': 'Poison'}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()

    next_player = game_state.get_player()
    action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': next_player, 'Target': 'C'}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()
else:
    # nomination
    game_state.add_next_player("A")
    next_player = game_state.get_player()
    action = {"type": "Nominate", "Speaker": next_player, "Nominee": "E"}
    game_state.apply_action(conv_manager, action, None, False, False)

    # vote
    next_player = game_state.get_player()
    action = {"type": "Vote", "Speaker": next_player, "Target": "E"}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()
    next_player = game_state.get_player()
    action = {"type": "Vote", "Speaker": next_player, "Target": "E"}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()
    next_player = game_state.get_player()
    action = {"type": "No Action", "Speaker": next_player}
    game_state.apply_action(conv_manager, action, None, False, False)
    game_state.update_game_state()



print('Game over: ' + str(game_state.is_terminal()))

print('Gamer over')
