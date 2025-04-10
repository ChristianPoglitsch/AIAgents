import random
import time

from datasets import load_from_disk

from botc_base import BasicGameState
from botc_base import MCTS
from botc_base import ConversationManager
from botc_base import MCTS
from botc_base import init_model
from botc_base import simulation_policy


model = []

server_based = True
store_data = False
show_training_data = False

reward_terminal_good    = 0.33
reward_terminal_evil    = 0.33
reward_good_action      = 0.33
reward_node = 0.0

num_child_node = 1 # 3
num_games = 1 # 35
num_iterations = 150 # 50

print_input = False
print_output = True
max_token = 400

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
#model_id = "deepseek-ai/deepseek-llm-7b-chat"
#model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"
model_id = "trained/Mistral-7B-Instruct-v0.3_merged"
#model_id = "trained/deepseek-llm-7b-chat_merged"
#model_id = "trained\\Teuken-7B-instruct-research-v0.4_merged"

# --- --- game --- ---

class Role:
    def __init__(self, role, team, alignment, description, action):
        self.role = role
        self.team = team
        self.alignment = alignment
        self.description = description
        self.action = action
        self.is_poisoned = False
        self.alive = True
        self.nominated = False
        self.is_nominated = False
        self.Information = None
        self.neighbors = None
        self.player_info = None
        self.num_conversations = 0
 
    def set_info(self, alive_players, player_infos, day_count = 0):
        return None

    def get_action_space(self, phase):
        actions = []        
        return actions
    
    def has_action(self, phase):
        return False
    
    def apply_action(self, other_players, game_state = None):
        None
        
    def set_poison(self, poison: bool) -> None:
        self.is_poisoned = poison
 
    def get_poison(self) -> bool:
        return self.is_poisoned
    
    def get_reward(self, node) -> int:
        state = node.state
        reward = reward_node
        speaker = node.action.get("Speaker")

        if speaker is not None:
            num_conversations = sum(
                state.features.features[speaker][partner][0] 
                for partner in state.features.features[speaker]
            )
            if num_conversations > self.num_conversations:
                reward += num_conversations / len(node.state.active_players)
                self.num_conversations = num_conversations

        if self.alignment == 'Good' and state.execution:
            if state.active_players[state.execution].alignment == 'Evil':
                reward += reward_terminal_good

        return reward
  
    def get_alignment(self) -> str:
        return self.alignment  

    def get_role(self) -> int:
        return self.role
 
    def get_team(self) -> int:
        return self.team    

    def get_description(self) -> int:
        return self.description

    def set_neighbors(self, left_neighbor, right_neighbor):
        self.neighbors = [left_neighbor, right_neighbor]
        
    def get_neighbors(self):
        return self.neighbors
    
    def target_kill_night(self):
        return True

    # Getter and Setter for 'alive'
    def get_alive(self):
        return self.alive
    
    def set_alive(self, value):
        if isinstance(value, bool):
            self.alive = value
        else:
            raise ValueError("alive must be a boolean value")

    # Getter and Setter for 'nominated'
    def get_nominated(self):
        return self.nominated
    
    def set_nominated(self, value):
        if isinstance(value, bool):
            self.nominated = value
        else:
            raise ValueError("nominated must be a boolean value")

    # Getter and Setter for 'is_nominated'
    def get_is_nominated(self):
        return self.is_nominated
    
    def set_is_nominated(self, value):
        if isinstance(value, bool):
            self.is_nominated = value
        else:
            raise ValueError("is_nominated must be a boolean value")

    # Getter and Setter for 'Information'
    def get_information(self):
        return self.Information
    
    def set_information(self, value):
        self.Information = value
        
    def add_information(self, value):
        self.Information = self.Information + value
        

class Washerwoman(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)

    def set_info(self, alive_players, player_infos, day_count = 0):
        minion_players = [p for p, data in alive_players.items() if data.get_role() and data.get_team() == "Town" and p != self.role]
    
        if minion_players:
            # Select one Townsfolk player and get their role
            chosen_town = random.choice(minion_players)
            chosen_town_role = alive_players[chosen_town].get_role()
        
            # Get the list of players excluding the Washerwoman and the chosen Townsfolk
            available_players = [p for p in alive_players if p != self.role and p != chosen_town]
        
            # Select another random player from the available players
            random_player = random.choice(available_players)
        
            # Update the Washerwoman's information
            self.player_info = f"You learn that {chosen_town} or {random_player} is a {chosen_town_role}"
        elif self.is_poisoned:
            chosen_town = random.choice(alive_players)
            chosen_town_role = alive_players[chosen_town].get_role()
        
            # Get the list of players excluding the Washerwoman and the chosen Townsfolk
            available_players = [p for p in alive_players if p != self.role and p != alive_players]
        
            # Select another random player from the available players
            random_player = random.choice(available_players)
        
            # Update the Washerwoman's information
            self.player_info = f"You learn that {chosen_town} or {random_player} is a {chosen_town_role}"           
        else:
            player_info = None
            
        return self.player_info
  
class Investigator(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)

    def set_info(self, alive_players, player_infos, day_count = 0):
        # Get a list of all Townsfolk players
        minion_players = [p for p, data in alive_players.items() if data.get_role() and data.get_team() == "Minion" and p != self.role]
    
        if minion_players:
            # Select one Townsfolk player and get their role
            chosen_town = random.choice(minion_players)
            chosen_town_role = alive_players[chosen_town].get_role()
        
            # Get the list of players excluding the Washerwoman and the chosen Townsfolk
            available_players = [p for p in alive_players if p != self.role and p != chosen_town]
        
            # Select another random player from the available players
            random_player = random.choice(available_players)
        
            # Update the Washerwoman's information
            self.player_info = f"You learn that {chosen_town} or {random_player} is a {chosen_town_role}"
        elif self.is_poisoned:
            chosen_town = random.choice(alive_players)
            chosen_town_role = alive_players[chosen_town].get_role()
        
            # Get the list of players excluding the Washerwoman and the chosen Townsfolk
            available_players = [p for p in alive_players if p != self.role and p != alive_players]
        
            # Select another random player from the available players
            random_player = random.choice(available_players)
        
            # Update the Washerwoman's information
            self.player_info = f"You learn that {chosen_town} or {random_player} is a {chosen_town_role}"           
        else:
            self.player_info = None
            
        return self.player_info   


class Empath(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)

    def set_info(self, alive_players, player_infos, day_count = 0):

        alive_neighbors = [neighbor for neighbor in player_infos.get_neighbors() if alive_players[neighbor].get_alive()]
        evil_neighbors = [neighbor for neighbor in alive_neighbors if alive_players[neighbor].get_alignment() == "Evil"]
        self.player_info = f"You sense that {len(evil_neighbors)} of your alive neighbors are evil."
           
        if self.is_poisoned:
            evil_neighbors = random.randint(0, 2)
            self.player_info = f"You sense that {evil_neighbors} of your alive neighbors are evil."
        return self.player_info

class Slayer(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)
        
    def target_kill_night(self):
        return False
  
class Ravenkeeper(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)
        self.target_tonight = False
        
    def target_kill_night(self):
        if self.alive is True:
            self.target_tonight = True
        return True
        
    def has_action(self, phase):
        if phase == 'Night':
            return True
        return False

    def get_action_space(self, phase):
        action_space = super().get_action_space(phase)
        if phase == 'Night' and self.target_tonight is True:
            action_space.append(str(f'{{"type": "Action", "Description": "Select a player (target) to get the role of this player.", "Speaker": None, "Target": "None"}}'))
        return action_space
    
    def apply_action(self, action, other_players):
        if self.target_tonight is True:
            target = action.get("Target")
            self.player_info = "Role of player " + target + " is " + other_players[target].role
            self.target_tonight = False
            return self.player_info

    def get_reward(self, node) -> int:
        reward = 0
        if self.player_info is not None:
            reward = reward + reward_good_action
        return reward + super().get_reward(node)
    
class Soldier(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)
        
    def target_kill_night(self):
        return False

class FortuneTeller(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action) 

class Poisoner(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)
        self.poisoned_player = None
        self.poisoned_tonight = False

    def set_info(self, alive_players, player_infos, day_count = 0):
        demon = next((p for p, data in alive_players.items() if data.get_role() and data.get_role() == "Imp"), None)
            
        # Update the Minion's Information field with their demon
        if demon:
            player_info = f"The Demon is {demon}."
        else:
            player_info = "No Demon found."
            
        return player_info

    def has_action(self, phase):
        if phase == 'Night':
            return True
        return False

    def get_action_space(self, phase):
        action_space = super().get_action_space(phase)
        if phase == 'Night':
            action_space.append(str(f'{{"type": "Action", "Description": "{self.description}", "Speaker": null, "Target": "None", "Effect": "Poison"}}'))
        return action_space
    
    def apply_action(self, action, other_players):
        if self.poisoned_player is not None:
            other_players[self.poisoned_player].set_poison(False)
        target = action.get("Target")
        self.poisoned_player = target
        other_players[self.poisoned_player].set_poison(True)
        self.poisoned_tonight = True
        
    def get_reward(self, node) -> int:
        reward = 0
        if self.poisoned_tonight:
            self.poisoned_tonight = False
            reward = reward + reward_terminal_evil
        return reward + super().get_reward(node)
    
class Imp(Role):
    def __init__(self, name, team, alignment, description, action):
        super().__init__(name, team, alignment, description, action)
        self.killed_tonight = False

    def has_action(self, phase):
        if phase == 'Night':
            return True
        return False

    def set_info(self, alive_players, player_infos, day_count = 0):
        minions = [p for p, data in alive_players.items() if data.get_role() and data.get_team() == "Minion"]
            
        # Update the Imp's Information field with their minions
        if minions:
            player_info = f"Your minions are {', '.join(minions)}."
        else:
            player_info = "You have no minions."
            
        return player_info

    def get_action_space(self, phase):
        action_space = super().get_action_space(phase)
        if phase == 'Night':
            action_space.append(str(f'{{"type": "Action", "Description": "{self.description}", "Speaker": null, "Target": "None"}}'))
        return action_space
    
    def apply_action(self, action, other_players):
        target = action.get("Target")
        if other_players[target].target_kill_night():
            other_players[target].set_alive(False)
        self.killed_tonight = True
        
    def get_reward(self, node) -> int:
        reward = 0
        if self.killed_tonight:
            self.killed_tonight = False
            reward = reward + reward_terminal_evil
        return reward + super().get_reward(node)

roles = {
    # Townsfolk Roles
    'Washerwoman': Washerwoman('Washerwoman', 'Town', 'Good', 'Learns that one of two players is a specific Townsfolk.', None),
    #'Librarian': Role('Librarian', 'Town', 'Good', 'Learns that one of two players is a specific Outsider or that no Outsiders are in play.', None),
    'Investigator': Investigator('Investigator', 'Town', 'Good', 'Learns that one of two players is a specific Minion.', None),
    #'Chef': Role('Chef', 'Town','Good',  'Learns how many pairs of evil players are sitting next to each other.', None),
    'Empath': Empath('Empath', 'Town', 'Good', 'Learns how many of their two alive neighbors are evil.', None),
    ##'Fortune Teller': FortuneTeller('Fortune Teller', 'Town', 'Good', 'Each night, chooses two players. Learns if one is the Demon, but there is a Red Herring.', None),
    #'Undertaker': Role('Undertaker', 'Town', 'Good', 'Each night, learns which character died by execution that day.', None),
    #'Monk': Role('Monk', 'Town', 'Good', 'Each night, chooses a player (not themselves). That player is safe from the Demon that night.', None),
    'Ravenkeeper': Ravenkeeper('Ravenkeeper', 'Town', 'Good', 'If killed at night, learns one players character.', None),
    #'Virgin': Role('Virgin', 'Town', 'Good', 'If nominated for the first time, and the nominator is a Townsfolk, they are executed immediately.', None),
    'Slayer': Slayer('Slayer', 'Town', 'Good', 'Once per game, publicly chooses a player. If that player is the Demon, they die.', None),
    ##'Soldier': Soldier('Soldier', 'Town', 'Good', 'Cannot die at night.', None),
    #'Mayor': Role('Mayor', 'Town', 'Good', 'If only three players live & no execution occurs, your team wins. Might not die at night.', None),
    
    # Outsider Roles
    #'Butler': Role('Butler', 'Outsider', 'Good', 'Each night, chooses a player. Can only vote if that player votes first.', None),
    #'Drunk': Role('Drunk', 'Outsider', 'Good', 'You do not know you are the Drunk. You think you are a Townsfolk, but you are not.', None),
    #'Recluse': Role('Recluse', 'Outsider', 'Good', 'Might register as evil & as a Minion or Demon, even if dead.', None),
    #'Saint': Role('Saint', 'Outsider', 'Good', 'If executed, your team loses.', None),
    
    # Minion Roles
    'Poisoner': Poisoner('Poisoner', 'Minion', 'Evil', 'Each night, poison one player.', 'Poison'),
    #'Baron': Role('Baron', 'Minion', 'Evil', 'Two extra Outsiders are in play.', None),
    #'Scarlet Woman': Role('Scarlet Woman', 'Minion', 'Evil', 'If there are five or more players alive and the Demon dies, you become the Demon.', None),
    #'Spy': Role('Spy', 'Minion', 'Evil', 'Might register as good. Sees the Grimoire each night.', None),
    
    # Demon Roles
    'Imp': Imp('Imp', 'Demon', 'Evil', 'Each night, chooses a player to die. If you kill yourself this way, a Minion becomes the Imp.', None)
}

first_night_order = ["Poisoner", "Washerwoman", "Investigator", "Empath"]
night_order = ["Poisoner", "Imp", "Empath", "Ravenkeeper"]

def roles_to_string(roles):
    """
    Converts the roles dictionary into a formatted string.
    
    :param roles: Dictionary of roles where each value is a Role object.
    :return: A formatted string listing each role with its alignment and description.
    """
    role_strings = []
    for role_name, role_obj in roles.items():
        role_strings.append(f"{role_name} ({role_obj.ability}): {role_obj.team}")
    
    return "\n".join(role_strings)


# ------------------ GameState and ActionSpace ------------------

# Stub definitions for GameState methods:
class BloodOnTheClocktowerState(BasicGameState):
    def __init__(self, players, available_roles):
        super().__init__(players)

        self.active_players = {}
        role_list = list(available_roles.values())  
        self.nominated = ''
        self.num_votes = 0

        self.conv_count_day = 0
        self.max_conv_count_per_day = 5

        self.assign_roles(players, role_list)

        self.phase = 'Night' # Day, Nominate, Night
        self.nominations = None
        self.nominated = None
        self.nomination_count = 0
        self.nomination_count_max = 0
        self.num_nominations = 0
        self.day_count = 0
        self.execution = None
        
        self.active_player = None
        self.active_player_action = None
        self.game_over = False


    def assign_roles(self, players, role_list):
        """
        Assigns exactly three Townsfolk, one Minion, and one Imp from the role list.
        Ensures correct role distribution while maintaining randomness.
        """
        # Step 1: Categorize roles
        town_roles = [role for role in role_list if role.team == "Town"]
        minion_roles = [role for role in role_list if role.team == "Minion"]
        imp_roles = [role for role in role_list if role.team == "Demon"]

        # Step 2: Ensure enough roles exist
        if len(town_roles) < 3 or len(minion_roles) < 1 or len(imp_roles) < 1:
            raise ValueError("Not enough roles available to meet selection criteria.")

        randomize_roles = True

        if randomize_roles:
            # Step 3: Select roles
            selected_roles = (
                random.sample(town_roles, 3) +  # Pick 3 Townsfolk
                random.sample(minion_roles, 1) +  # Pick 1 Minion
                random.sample(imp_roles, 1)  # Pick 1 Imp
            )

            # Step 4: Shuffle the selected roles to randomize assignment
            random.shuffle(selected_roles)
        else:
            # Step 3b: Fix order
            selected_roles = [town_roles[0]]
            selected_roles.append(town_roles[3]) # 1
            selected_roles.append(town_roles[4]) # 2
            selected_roles.append(minion_roles[0])
            selected_roles.append(imp_roles[0])

        # Step 5: Assign roles to players
        self.active_players = {}  # Reset player assignments
        for idx, player in enumerate(players):
            if idx < len(selected_roles):  # Ensure no out-of-index error
                assigned_role = selected_roles[idx]
            else:
                assigned_role = random.choice(role_list)  # Fallback for extra players

            # Step 6: Assign the role and initialize the player
            left_neighbor = players[(idx - 1) % len(players)]
            right_neighbor = players[(idx + 1) % len(players)]

            assigned_role.set_neighbors(left_neighbor, right_neighbor)

            self.active_players[player] = assigned_role

    def update_game_state(self):
        
        if self.is_terminal():
            return

        # initial infos 
        if  self.day_count == 0:
            self.night_info(first_night_order)
            self.day_count = self.day_count + 1

        if self.phase == 'Day':
            self.conv_count_day = self.conv_count_day + 1

        if self.phase == 'Night' and self.get_next_players_count() == 0:
            self.phase = 'Day'
            self.day_count = self.day_count + 1
            self.execution = None
        elif self.phase == 'Day' and self.conv_count_day >= self.max_conv_count_per_day and self.phase != 'Nominate':
            self.conv_count_day = 0
            self.day_count = self.day_count + 1
            self.phase = 'Night'
            self.empty_next_players()
            self.night_info(night_order)
            
        if self.phase == 'Nominate':
            count = self.count_next_players()
            if count == 0:
                num_alive_players = len(self.active_players)
                if self.num_votes > int(num_alive_players / 2):
                    self.active_players[self.nominated].set_alive(False)
                    self.nomination_count_max = self.num_votes
                    self.execution = self.nominated
                self.phase = 'Night'
                self.empty_next_players()
                self.night_info(night_order)
            
     
    def get_alive_neighbors(self, players, idx, alive_players):
        """
        Returns the left and right alive neighbors of a player in a circular list.

        :param players: List of all players.
        :param idx: Index of the current player.
        :param alive_players: Dictionary containing player statuses.
        :return: A list of up to two alive neighbors in [left, right] order.
        """
        n = len(players)
        left_idx, right_idx = (idx - 1) % n, (idx + 1) % n

        # Find the first alive left neighbor
        left_neighbor = None
        for i in range(n):
            candidate = players[(left_idx - i) % n]
            if alive_players[candidate].get_alive():
                left_neighbor = candidate
                break

        # Find the first alive right neighbor
        right_neighbor = None
        for i in range(n):
            candidate = players[(right_idx + i) % n]
            if alive_players[candidate].get_alive():
                right_neighbor = candidate
                break

        return [left_neighbor, right_neighbor]
               
    def night_info(self, order):
        """
        Processes night information for players in the specified order.
    
        :param order: List of role names to determine processing order.
        """
        # Filter and sort players based on order
        sorted_players = sorted(
            [player for player in self.active_players if self.active_players[player].role in order and self.active_players[player].alive == True],
            key=lambda p: order.index(self.active_players[p].role)
        )

        # Process each player in order
        for player in sorted_players:
            player_info = self.active_players[player]
            information = player_info.set_info(self.active_players, player_info, self.day_count)
        
            if information is not None:
                player_info.set_information(information)
                
            if player_info.has_action(self.phase):
                self.add_next_player(player)

    def get_action_space_description(self, current_player):
        """
        Returns a string describing the available actions for the current player.
        In this Blood on the Clocktower adaptation:
          - All alive players can send a message.
          - During the Day phase, players may nominate or vote.
          - During the Night phase, players with a special ability (as defined by their role)
          - A NoAction option is always available.
      
        :param current_player: The name of the current player.
        :return: A string describing the available actions, each on a new line.
        """
        actions = []
        player_info = self.active_players.get(current_player)
    
        action = player_info.get_action_space(self.phase)
        if action is not None and len(action) > 0:
            actions.extend(action)
            
        # Day phase actions
        if self.phase == "Day" and self.conv_count_day < self.max_conv_count_per_day:
            # Always available action: Message
            actions.append('{"type": "Message", "Speaker": None, "Audience": None, "Message": None}')
            # Nominate action (if the player hasn't already nominated someone)
            actions.append('{"type": "Nominate", "Speaker": None, "Nominee": None}')
            # Vote action is available in the day phase
        elif self.phase == "Day":
            actions.append('{"type": "Nominate", "Speaker": None, "Nominee": None}')
            # Vote action is available in the day phase            
        elif self.phase == "Nominate":
            actions.append('{"type": "Vote", "Speaker": None}')

        # Always include a NoAction option.
        actions.append(str(self.no_action).replace("'", '"'))
        
        return "\n".join(actions)

    def is_terminal(self):
        """
        Returns True if the game is over:
        - All Imps are dead, or
        - Only two or fewer players are alive.
        """
        # Check if at least one living player has the Imp role.
        imp_alive = any(
            player_info.get_alive() and player_info.get_role() == 'Imp'
            for player_info in self.active_players.values()
        )
    
        # Count the number of alive players.
        alive_count = sum(
            1 for player_info in self.active_players.values() if player_info.get_alive()
        )
    
        return (not imp_alive) or (alive_count <= 2)

    def good_win(self):
        imp_alive = any(
            player_info.get_alive() and player_info.get_role() == 'Imp'
            for player_info in self.active_players.values()
        )        
        return (not imp_alive)

    def evil_win(self):
        imp_alive = any(
            player_info.get_alive() and player_info.get_role() == 'Imp'
            for player_info in self.active_players.values()
        )
    
        # Count the number of alive players.
        alive_count = sum(
            1 for player_info in self.active_players.values() if player_info.get_alive()
        )
    
        return (imp_alive) and (alive_count <= 2)

    def terminal_message(self):
        if self.good_win():
            return 'Good Win'
        if self.evil_win():
            return 'Evil Win'


    def get_game_state(self, player):
        """
        Returns a human-readable public state description tailored for Blood on the Clocktower.

        Public information includes:
          - The current game phase (Day or Night).
          - A list of players along with their status (alive or dead).

        Private information for the requesting player includes:
          - Their own role and any associated private ability details.
          - If the player is the Empath, they receive information about how many of their alive neighbors are evil.

        Additional game state features are appended as defined in game_state_features_to_string().
        """

        # Public game state information.
        phase_info = f"Current phase: {self.phase}"
        if self.phase == 'Nominate':
            phase_info = phase_info + f"\nNominated is {self.nominated}"
        players_info = "Players: " + ", ".join(
            [f"{p} ({'Alive' if self.active_players[p].get_alive() else 'Dead'})" for p in self.players]
        )

        # Private info for the current player.
        player_info = self.active_players[player]
        private_info = f"Your role: {player_info.get_role()} - {player_info.get_description()}"
        if player_info.alignment == 'Good':
            private_info = private_info + ' Try to nominate and vote for players who might be evil.\n'
        elif player_info.alignment == 'Evil':
            private_info = private_info + ' Try to nominate and vote for players who might be good.\n'
        # Append additional state features as needed.
        additional_info = self.game_state_features_to_string(player)

        roles_info = "These roles are in the game: " + ", ".join(sorted(roles)) + " .You can use the rules to bluff."

        return f"{phase_info}\n{players_info}\n{roles_info}\n{private_info}\n{player_info.get_information()}\n\n{additional_info}"

    def get_player_info(self):
        players_info = "Players: " + ", ".join(
            [f"{p} ({'Alive' if self.active_players[p].get_alive() else 'Dead'})" for p in self.players]
        )
        return players_info
    
    def get_demons_alive(self):
        demons_alive = any(
            player_info.get_alive() and player_info.get_role() == 'Imp'
            for player_info in self.active_players.values()
        )
        return demons_alive

    def generate_prompt(self, current_player, conversation_manager):
        """
        Creates a prompt string for the LLM based on the current game state,
        available actions, and conversation history.
        
        :param current_player: The name of the current player.
        :param action_space_description: A string describing the available action templates.
        :param conversation_manager: ConversationManager
        :return: A prompt string in plain text.
        """
        # Get the public state description for the current player.
        state_description = self.get_game_state(current_player)
        conversation_history = conversation_manager.get_conversation_history_for_player(current_player)
        #get_player_plan = conversation_manager.get_player_plan(current_player)

        prompt = (
            "You are a helpful board game AI assistant for Blood on the Clocktower.\n"
            f"Current Player: {current_player}\n\n"
            "The available actions are given below, but each action is incomplete and missing parameters marked as None.\n"
            "Available Actions Description:\n"
            f"{self.get_action_space_description(current_player)}\n\n"
            "Game State:\n"
            f"{state_description}\n\n"
            "Chronological conversation History:\n"
            f"{conversation_history}\n\n"
            #"Current plans\n"
            #f"{get_player_plan}\n\n"
        )        

        prompt = prompt + "First, consider a possible answer. Then, provide the corresponding action. Audience of the messages can only be other players. You can write to multiple other players. Do not use All.\n"
        prompt = prompt + "Please output one complete possible action from the Available Actions Description list in JSON format.\n"
        prompt = prompt + "Do NOT use any markdown formatting (e.g., ```json) in your response and use double quotes. Replace all None parts in the action.\n"        

        return prompt
    
    def apply_action(self, conversation_manager, action, model, print_output, server_based):
        """
        Updates the game state and conversation history based on the selected action.
        For both "Message" and "Guess" actions, calls complete_action_with_llm to generate a
        complete action and returns that result.    

        :param conv_manager: The conversation manager to track interactions.
        :param action: A dictionary representing the action (from LLM or another source).
        """
    
        action_type = action.get("type")
        speaker = action.get("Speaker")
        
        if speaker == None:
            return False

        self.active_player = self.active_players[speaker]
        self.active_player_action = action

        if action_type == "Message":
            # Extract speaker and audience from the action.
            speaker = action.get("Speaker")
            audience = action.get("Audience")        
            # Ensure audience is handled as a list.
            self.add_next_player(audience)
            self.plan_action(speaker, conversation_manager, model, print_output, server_based)
            
        elif action_type == "Action":
            target = action.get("Target")
            speaker = action.get("Speaker")
            if target is not None and speaker is not None:
                self.active_players[speaker].apply_action(action, self.active_players)
                
        elif action_type == 'Nominate':
            self.num_nominations = self.num_nominations + 1
            self.phase = 'Nominate'
            self.nominated = action.get("Nominee")
            self.empty_next_players()
            # Add all alive players to next_players list
            for player, player_info in self.active_players.items():
                if player_info.get_alive():
                    self.add_next_player(player)
            
        elif action_type == 'Vote':
            self.num_votes = self.num_votes + 1
            
        return True


def reward_function(node, new_node):
    return new_node.state.active_player.get_reward(new_node)


def add_convs(node, conv_manager):
    # Add current node's outcomes
    conv_manager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())

    # Base case: No children -> return
    if not node.children:
        return

    # Recurse for each child
    for child in node.children:
        add_convs(child, conv_manager)

# --- --- main --- ---

def play_game():
    start_time = time.time()  # Start timing

    folder_path = 'training_botc'
    conversationManager = ConversationManager()

    # Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
    player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    num_correct_games = 0
    model = init_model(model_id, server_based, max_token)

    good_wins = 0
    evil_wins = 0

    for i in range(num_games):
        print('Start Game')
        print(str(i) + ' / ' + str(num_games))
        game_state = BloodOnTheClocktowerState(['A', 'B', 'C', 'D', 'E'], roles)
        # Create a dummy ConversationManager and add a conversation.
        conv_manager = ConversationManager()

        # Create an MCTS instance
        mcts = MCTS(simulation_policy, reward_function, num_child_node, iterations=num_iterations)

        # Run MCTS to get the best action/state
        best_node = mcts.search(game_state, conv_manager, model, print_output, server_based)

        nodes = mcts.get_all_terminal_nodes(mcts.get_root_node())

        print(mcts.print_tree())

        print('Alive players: ' + str(best_node.state.get_player_info()))
        print('Good win: ' + str(best_node.state.good_win()))
        print('Evil win: ' + str(best_node.state.evil_win()))
    
        for player in best_node.state.active_players:
            print(best_node.state.active_players[player])

        if best_node.state.good_win():
            good_wins = good_wins + 1
        if best_node.state.evil_win():
            evil_wins = evil_wins + 1
        #if len(best_node.state.alive_players) > 2:
        #    num_correct_games = num_correct_games + 1
        #    log.extend(best_node.conversation_manager.get_prompt_outcomes())        
        #
        #    if log and store_data:
        #        conversationManager.append_prompt_outcomes(best_node.conversation_manager.get_prompt_outcomes())

        if store_data:
            #add_convs(mcts.get_root(), conversationManager)
            for node in nodes:
                conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())
            conversationManager.prompt_outcome_log = list(dict.fromkeys(conversationManager.prompt_outcome_log))

        num_correct_games = num_correct_games + 1
        
        if show_training_data:
            dataset = load_from_disk(folder_path)
            print("Dataset loaded from:", folder_path)
            for record in dataset:
                print("--- --- ---")
                print(record["input"])
                print("*** *** ***")
                print(record["output"])
                print("--- --- ---")

    print("Good wins: " + str(good_wins) + " / Evil wins: " + str(evil_wins))

    end_time = time.time()
    elapsed_time = end_time - start_time

    if store_data:
        conversationManager.export_prompt_outcome_log(folder_path, False)
    print(f"Execution time: {elapsed_time:.6f} seconds")
    

def main():
    play_game()

if __name__ == "__main__":
    main()


