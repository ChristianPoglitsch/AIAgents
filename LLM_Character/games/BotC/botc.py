import json
import random
import copy
import time

from datasets import load_from_disk

from botc_base import BasicGameState
from botc_base import MCTSNode
from botc_base import MCTS
from botc_base import ConversationManager
from botc_base import MCTS
from botc_base import init_model
from botc_base import simulation_policy


model = []

server_based = True
store_data = False
show_training_data = False

reward_terminal = 16
reward_small = 4
reward_node = 0.25

num_child_node = 1 # 3
num_games = 5 # 35
num_iterations = 16 # 50

print_output = True
max_token = 150

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
#model_id = "deepseek-ai/deepseek-llm-7b-chat"
#model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"
model_id = "trained/Mistral-7B-Instruct-v0.3_merged"
#model_id = "trained/deepseek-llm-7b-chat_merged"
#model_id = "trained\\Teuken-7B-instruct-research-v0.4_merged"

# --- --- game --- ---

class Role:
    def __init__(self, name, team, alignment, ability):
        self.name = name
        self.team = team
        self.ability = ability
        self.alignment = alignment
        
roles = {
    # Townsfolk Roles
    'Washerwoman': Role('Washerwoman', 'Town', 'Good', 'Learns that one of two players is a specific Townsfolk.'),
    'Librarian': Role('Librarian', 'Town', 'Good', 'Learns that one of two players is a specific Outsider or that no Outsiders are in play.'),
    'Investigator': Role('Investigator', 'Good', 'Town', 'Learns that one of two players is a specific Minion.'),
    #'Chef': Role('Chef', 'Town','Good',  'Learns how many pairs of evil players are sitting next to each other.'),
    #'Empath': Role('Empath', 'Town', 'Good', 'Learns how many of their two alive neighbors are evil.'),
    #'Fortune Teller': Role('Fortune Teller', 'Town', 'Good', 'Each night, chooses two players. Learns if one is the Demon, but there is a Red Herring.'),
    #'Undertaker': Role('Undertaker', 'Town', 'Good', 'Each night, learns which character died by execution that day.'),
    #'Monk': Role('Monk', 'Town', 'Good', 'Each night, chooses a player (not themselves). That player is safe from the Demon that night.'),
    #'Ravenkeeper': Role('Ravenkeeper', 'Town', 'Good', 'If killed at night, learns one players character.'),
    #'Virgin': Role('Virgin', 'Town', 'Good', 'If nominated for the first time, and the nominator is a Townsfolk, they are executed immediately.'),
    #'Slayer': Role('Slayer', 'Town', 'Good', 'Once per game, publicly chooses a player. If that player is the Demon, they die.'),
    #'Soldier': Role('Soldier', 'Town', 'Good', 'Cannot die at night.'),
    #'Mayor': Role('Mayor', 'Town', 'Good', 'If only three players live & no execution occurs, your team wins. Might not die at night.'),
    
    # Outsider Roles
    #'Butler': Role('Butler', 'Outsider', 'Good', 'Each night, chooses a player. Can only vote if that player votes first.'),
    #'Drunk': Role('Drunk', 'Outsider', 'Good', 'You do not know you are the Drunk. You think you are a Townsfolk, but you are not.'),
    #'Recluse': Role('Recluse', 'Outsider', 'Good', 'Might register as evil & as a Minion or Demon, even if dead.'),
    #'Saint': Role('Saint', 'Outsider', 'Good', 'If executed, your team loses.'),
    
    # Minion Roles
    'Poisoner': Role('Poisoner', 'Minion', 'Evil', 'Each night and in FirstNights, poison one player.'),
    #'Baron': Role('Baron', 'Minion', 'Evil', 'Two extra Outsiders are in play.'),
    #'Scarlet Woman': Role('Scarlet Woman', 'Minion', 'Evil', 'If there are five or more players alive and the Demon dies, you become the Demon.'),
    #'Spy': Role('Spy', 'Minion', 'Evil', 'Might register as good. Sees the Grimoire each night.'),
    
    # Demon Roles
    'Imp': Role('Imp', 'Demon', 'Evil', 'Each night, chooses a player to die. If you kill yourself this way, a Minion becomes the Imp.')
}

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

        self.alive_players = {}

        role_list = list(available_roles.values())  
        #random.shuffle(role_list)  # Shuffle to randomize role assignment

        n = len(players)
        for idx, player in enumerate(players):
            left_neighbor = players[(idx - 1) % n]
            right_neighbor = players[(idx + 1) % n]

            # Assign a role from the shuffled list, ensuring roles cycle if there are more players than roles
            assigned_role = role_list[idx % len(role_list)]  

            self.alive_players[player] = {
                'role': assigned_role,
                'alignment': assigned_role.alignment,  # Ensure alignment is stored
                'alive': True,
                'nominated': False,
                'is_nominated': False,
                'effects': None,
                'Information': None,
                'neighbors': [left_neighbor, right_neighbor]
            }
               
        # initial infos
        for player, player_info in self.alive_players.items():
            role = player_info.get('role')
            if role is not None and role.name == "Washerwoman":
                # Update the player's Information field with FirstNight details.
                player_info['Information'] = "FirstNight Info: You learn that one of two players is a specific Townsfolk."
            elif role.team == "Imp":
                # Find all minions in the game
                minions = [p for p, data in self.alive_players.items() if data['role'] and data['role'].alignment == "Minion"]
            
                # Update the Imp's Information field with their minions
                if minions:
                    player_info['Information'] = f"FirstNight Info: Your minions are {', '.join(minions)}."
                else:
                    player_info['Information'] = "FirstNight Info: You have no minions."
            elif role.team == "Minion":
                # Find the Imp (the Demon)
                demon = next((p for p, data in self.alive_players.items() if data['role'] and data['role'].name == "Imp"), None)
            
                # Update the Minion's Information field with their demon
                if demon:
                    player_info['Information'] = f"FirstNight Info: The Demon is {demon}."
                else:
                    player_info['Information'] = "FirstNight Info: No Demon found."

        self.phase = 'FirstNight' # Day, Nomination, Night, FirstNight
        self.nominations = []
        self.nominated = []
        self.nomination_count = -1
        self.roles = {player: None for player in players}
        self.private_knowledge = {player: None for player in players}
        self.day_count = 1
        self.effects = {}
        self.execution = None


    def get_action_space_description(self, current_player):
        """
        Returns a string describing the available actions for the current player.
        In this Blood on the Clocktower adaptation:
          - All alive players can send a message.
          - During the Day phase, players may nominate or vote.
          - During the Night phase, players with a special ability (as defined by their role)
            receive a NightAction option.
          - A NoAction option is always available.
      
        :param current_player: The name of the current player.
        :return: A string describing the available actions, each on a new line.
        """
        actions = []
        player_info = self.alive_players.get(current_player)
    
        if not player_info or not player_info['alive']:
            return "No actions available (player is dead or not found)."
   
    
        # Day phase actions
        if self.phase == "Day":
            # Always available action: Message
            actions.append('{"type": "Message", "Speaker": null, "Audience": null, "Message": null}')
            # Nominate action (if the player hasn't already nominated someone)
            if not player_info.get('nominated', False):
                actions.append('{"type": "Nominate", "Nominator": null, "Nominee": null}')
            # Vote action is available in the day phase
            actions.append('{"type": "Vote", "Voter": null, "VoteTarget": null}')
    
        # Night phase actions
        #elif self.phase == "Night":
            # If the player's role has a special ability, add a NightAction.
                
        #elif self.phase == "FirstNight":
            # FirstNight phase actions: Special handling for Washerwoman.

 
        role = player_info.get('role')
        if role is not None and role.ability:
            actions.append(
                f'{{"type": "NightAction", "Target": " ", "Description": "{role.ability}"}}'
            )
    
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
            player_info['alive'] and player_info['role'].name == 'Imp'
            for player_info in self.alive_players.values()
        )
    
        # Count the number of alive players.
        alive_count = sum(
            1 for player_info in self.alive_players.values() if player_info['alive']
        )
    
        return (not imp_alive) or (alive_count <= 2)

    def get_game_state(self, player):
        """
        Returns a human-readable public state description tailored for Blood on the Clocktower.
    
        Public information includes:
          - The current game phase (Day or Night).
          - A list of players along with their status (alive or dead).
      
        Private information for the requesting player includes:
          - Their own role and any associated private ability details.
    
        Additional game state features are appended as defined in game_state_features_to_string().
        """
        
        role = self.alive_players[player]
        # Public game state information.
        phase_info = f"Current phase: {self.phase}"
        players_info = "Players: " + ", ".join(
            [f"{p} ({'Alive' if self.alive_players[p]['alive'] else 'Dead'})" for p in self.players]
        )
    
        # Private info for the current player.
        player_role = self.alive_players[player].get('role')
        if player_role:
            private_info = f"Your role: {player_role.name} - {player_role.ability}"
        else:
            private_info = "Your role has not been assigned."
    
        # Append additional state features as needed.
        additional_info = self.game_state_features_to_string(player)
    
        return f"{phase_info}\n{players_info}\n{private_info}\n\n{additional_info}\n{self.alive_players[player]['Information']}"

    
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

        prompt = prompt + "First, consider a possible answer. Then, provide the corresponding action.\n"
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

        if action_type == "Message":
            # Extract speaker and audience from the action.
            speaker = action.get("Speaker")
            audience = action.get("Audience")        
            # Ensure audience is handled as a list.
            self.add_next_player(audience)
            self.plan_action(speaker, conversation_manager, model, print_output, server_based)

        elif action_type == "NightAction":
                # Extract role, description, and optionally target or speaker information.
                role = action.get("Role")
                description = action.get("Description")
                speaker = action.get("Speaker")
        
                # Process the night action based on the role's ability.
                # This is a stub method; implement role-specific logic as needed.
                self.perform_night_ability(speaker, action)
        
                # Log the night action in the conversation history.
                conversation_manager.add_message_to_conversation(
                    speaker, speaker, f"I performed my night action as {role}: {description}")


def reward_function(node, new_node):
    return 0

# --- --- main --- ---

log = []
start_time = time.time()  # Start timing

folder_path = 'training'
conversationManager = ConversationManager()

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

num_correct_games = 0
num_correct_high_reward_games = 0
model = init_model(model_id, server_based, max_token)

for i in range(num_games):
    print('Start Game')
    print(str(i) + ' / ' + str(num_games))
    game_state = BloodOnTheClocktowerState(['A', 'B', 'C', 'D', 'E'], roles)
    game_state.add_next_player('D')
    # Create a dummy ConversationManager and add a conversation.
    conv_manager = ConversationManager()

    # Create an MCTS instance
    mcts = MCTS(simulation_policy, reward_function, num_child_node, iterations=num_iterations)

    # Run MCTS to get the best action/state
    best_node = mcts.search(game_state, conv_manager, model, print_output, server_based)

    print(mcts.print_tree())

    print('Secret number: ' + str(best_node.state.secret_number))
    print('Guess: ' + str(best_node.state.guess))
    print('Liar: ' + str(game_state.liar))
    #conv_manager.get_all_conversations_for_player_print()
    #conv_manager.print_all_conversations()
    if game_state.guess is not None and game_state.secret_number is not None:
        print("Result: " + str(int(game_state.guess) == int(game_state.secret_number)))

    if str(best_node.state.guess) == str(best_node.state.secret_number) and best_node.value >= 100:
        num_correct_high_reward_games = num_correct_high_reward_games + 1

    if str(best_node.state.guess) == str(best_node.state.secret_number):
        num_correct_games = num_correct_games + 1
        log.extend(best_node.conversation_manager.get_prompt_outcomes())        

        if log and store_data:
            conversationManager.append_prompt_outcomes(best_node.conversation_manager.get_prompt_outcomes())
            #best_node.conversation_manager.set_prompt_outcomes(log)
            #best_node.conversation_manager.export_prompt_outcome_log(folder_path, True)

    if show_training_data:
        dataset = load_from_disk(folder_path)
        print("Dataset loaded from:", folder_path)
        for record in dataset:
            print("--- --- ---")
            print(record["input"])
            print("*** *** ***")
            print(record["output"])
            print("--- --- ---")

print("Successfull games: " + str(num_correct_games) + " / Successfull games (w high Reward): " + str(num_correct_high_reward_games) + " / Played games: " + str(num_games))

end_time = time.time()
elapsed_time = end_time - start_time

if store_data and num_correct_games > 0:
    conversationManager.export_prompt_outcome_log(folder_path, True)
print(f"Execution time: {elapsed_time:.6f} seconds")