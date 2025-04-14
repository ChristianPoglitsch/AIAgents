import time
import copy
from botc_base import ConversationManager
from botc import *

# --- --- main --- ---

log = []
start_time = time.time()  # Start timing

folder_path = 'training_botc'
conversationManager = ConversationManager()

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

roles = {
    # Townsfolk Roles
    'Washerwoman': Washerwoman('Washerwoman', 'Town', 'Good', 'Learns that one of two players is a specific Townsfolk.', None),
    'Investigator': Investigator('Investigator', 'Town', 'Good', 'Learns that one of two players is a specific Minion.', None),
    'Ravenkeeper': Ravenkeeper('Ravenkeeper', 'Town', 'Good', 'If killed at night, select one player and learn the role of this player.', None),
    'Soldier': Soldier('Soldier', 'Town', 'Good', 'Cannot die at night.', None),
    'Empath': Empath('Empath', 'Town', 'Good', 'Learns how many of their two alive neighbors are evil.', None),

    # Minion Roles
    'Poisoner': Poisoner('Poisoner', 'Minion', 'Evil', 'Each night, poison one player.', 'Poison'),
    
    # Demon Roles
    'Imp': Imp('Imp', 'Demon', 'Evil', 'Each night, chooses a player to die. If you kill yourself this way, a Minion becomes the Imp.', None)
}

print('Start Game')
game_state = BloodOnTheClocktowerState(['A', 'B', 'C', 'D', 'E'], roles, False)
# Create a dummy ConversationManager and add a conversation.
conv_manager = ConversationManager()

game_state.update_game_state()
next_player = game_state.get_player()

# firsrt night phase
action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'D', 'Target': 'A', 'Effect': 'Poison'}
game_state.apply_action(conv_manager, action, None, False, False)
game_state_ravenkeeper = copy.deepcopy(game_state)
game_state_soldier = copy.deepcopy(game_state)
game_state_imp = copy.deepcopy(game_state)


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


action = {'type': 'Action', 'Description': 'Kill player tonight.', 'Speaker': 'E', 'Target': 'C'}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()



# second day

# Evil win
game_state_evil = copy.deepcopy(game_state)
game_state_evil.add_next_player("A")
next_player = game_state_evil.get_player()
action = {"type": "Nominate", "Speaker": next_player, "Nominee": "D"}
game_state_evil.apply_action(conv_manager, action, None, False, False)

# vote
next_player = game_state_evil.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "D"}
game_state_evil.apply_action(conv_manager, action, None, False, False)
game_state_evil.update_game_state()
next_player = game_state_evil.get_player()
action = {"type": "No Action", "Speaker": next_player}
game_state_evil.apply_action(conv_manager, action, None, False, False)
game_state_evil.update_game_state()
next_player = game_state_evil.get_player()
action = {"type": "No Action", "Speaker": next_player}
game_state_evil.apply_action(conv_manager, action, None, False, False)
game_state_evil.update_game_state()

# night actions
next_player = game_state_evil.get_player()

action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'D', 'Target': 'A', 'Effect': 'Poison'}
game_state_evil.apply_action(conv_manager, action, None, False, False)
game_state_evil.update_game_state()

next_player = game_state_evil.get_player()
action = {'type': 'Action', 'Description': 'Kill player tonight.', 'Speaker': 'E', 'Target': 'C'}
game_state_evil.apply_action(conv_manager, action, None, False, False)
game_state_evil.update_game_state()

print('Game over: ' + str(game_state_evil.terminal_message()))


# Good Win
game_state_good = copy.deepcopy(game_state)
game_state_good.add_next_player("A")
next_player = game_state_good.get_player()
action = {"type": "Nominate", "Speaker": next_player, "Nominee": "E"}
game_state_good.apply_action(conv_manager, action, None, False, False)

# vote
next_player = game_state_good.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "E"}
game_state_good.apply_action(conv_manager, action, None, False, False)
game_state_good.update_game_state()
next_player = game_state_good.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "E"}
game_state_good.apply_action(conv_manager, action, None, False, False)
game_state_good.update_game_state()
next_player = game_state_good.get_player()
action = {"type": "No Action", "Speaker": next_player}
game_state_good.apply_action(conv_manager, action, None, False, False)
game_state_good.update_game_state()

print('Game over: ' + str(game_state_good.terminal_message()))


# ravenkeeper
game_state_ravenkeeper.update_game_state()
action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'E', 'Target': 'B'}
game_state_ravenkeeper.apply_action(conv_manager, action, None, False, False)
game_state_ravenkeeper.update_game_state()
action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'B', 'Target': 'E'}
game_state_ravenkeeper.apply_action(conv_manager, action, None, False, False)


# soldier
game_state_soldier.update_game_state()
action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'E', 'Target': 'C'}
game_state_soldier.apply_action(conv_manager, action, None, False, False)
game_state_soldier.update_game_state()


# imp

# first day

# nomination
game_state_imp.update_game_state()
game_state_imp.add_next_player("A")
next_player = game_state_imp.get_player()
action = {"type": "Nominate", "Speaker": next_player, "Nominee": "B"}
game_state_imp.apply_action(conv_manager, action, None, False, False)

# vote
next_player = game_state_imp.get_player()
action = {"type": "No Action", "Speaker": next_player}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()
next_player = game_state_imp.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()
next_player = game_state_imp.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()
next_player = game_state_imp.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()
next_player = game_state_imp.get_player()
action = {"type": "Vote", "Speaker": next_player, "Target": "B"}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()

# night actions
next_player = game_state_imp.get_player()

action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'E', 'Target': 'E'}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()

action = {'type': 'Action', 'Description': 'Each night, poison one player.', 'Speaker': 'D', 'Target': 'C'}
game_state_imp.apply_action(conv_manager, action, None, False, False)
game_state_imp.update_game_state()


# slayer

roles = {
    # Townsfolk Roles
    'Washerwoman': Washerwoman('Washerwoman', 'Town', 'Good', 'Learns that one of two players is a specific Townsfolk.', None),
    'Investigator': Investigator('Investigator', 'Town', 'Good', 'Learns that one of two players is a specific Minion.', None),
    'Empath': Empath('Empath', 'Town', 'Good', 'Learns how many of their two alive neighbors are evil.', None),
    'Ravenkeeper': Ravenkeeper('Ravenkeeper', 'Town', 'Good', 'If killed at night, select one player and learn the role of this player.', None),
    'Slayer': Slayer('Slayer', 'Town', 'Good', 'Once per game, publicly chooses a player. If that player is the Demon, they die.', None),
    'Soldier': Soldier('Soldier', 'Town', 'Good', 'Cannot die at night.', None),

    # Minion Roles
    'Poisoner': Poisoner('Poisoner', 'Minion', 'Evil', 'Each night, poison one player.', 'Poison'),
    
    # Demon Roles
    'Imp': Imp('Imp', 'Demon', 'Evil', 'Each night, chooses a player to die. If you kill yourself this way, a Minion becomes the Imp.', None)
}

game_state = BloodOnTheClocktowerState(['A', 'B', 'C', 'D', 'E'], roles, False)
action = {'type': 'Action', 'Description': 'Slay.', 'Speaker': 'C', 'Target': 'E'}
game_state.apply_action(conv_manager, action, None, False, False)
game_state.update_game_state()
print('Game over: ' + str(game_state.terminal_message()))
