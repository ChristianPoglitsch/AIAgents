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

server_based = False
store_data = False
show_training_data = False

reward_terminal = 16
reward_small = 4
reward_node = 0.25

num_child_node = 1 # 3
num_games = 1 # 35
num_iterations = 16 # 50

print_output = True
max_token = 450

model_id = "mistralai/Mistral-7B-Instruct-v0.3"
#model_id = "deepseek-ai/deepseek-llm-7b-chat"
#model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"
model_id = "trained/Mistral-7B-Instruct-v0.3_merged"
#model_id = "trained/deepseek-llm-7b-chat_merged"
#model_id = "trained\\Teuken-7B-instruct-research-v0.4_merged"

# --- --- game --- ---

class SimpleNumberGuessGameState(BasicGameState):
    def __init__(self, players):
        super().__init__(players)
        # Secret number is either 0 or 100.
        self.secret_number = random.randint(0, 100)
        # Randomly select one respondent (from B, C, D) to be the liar.
        self.liar = random.choice(['B', 'C', 'D'])
        # The guess is initialized to None.
        self.guess = None

    def get_action_space_description(self, current_player):
        """
        Returns the action space description tailored to the current player.
        Only Player A has the 'Guess' action.

        :param current_player: The name of the current player.
        :return: A string describing the available actions, each on a new line.
        """
        actions = []

        if current_player == "A":
            actions.append('{"type": "Guess", "Speaker": null, "Number": null}')
    
        actions.append('{"type": "Message", "Speaker": null, "Audience": null, "Message": null}')
        actions.append(str(self.no_action).replace("'", '"'))

        return "\n".join(actions)

    def is_terminal(self):
        """Game ends when a guess is made."""
        #return self.guess
        if self.guess is not None:
            return True
        return False

    def get_game_state(self, player):
        """
        Returns a human-readable public state description.
        For Player A, the secret number remains hidden.
        For other players, the secret number is revealed.
        If the player is the liar, that fact is indicated.
        """
        if player == "A":
            secret_info = "Secret number is unknown to you."
        else:
            secret_info = f"Secret number is {self.secret_number}."
            if player == self.liar:
                secret_info += " (You are the liar!)"
        player_info = "Players: " + ", ".join(self.players)
        return f"{secret_info}\n{player_info}" + "\n\n" + self.game_state_features_to_string(player)
    
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
            "You are a helpful board game AI assistant for the number guessing minigame. "
            "Player A's goal is to determine the secret number [0 to 100] by asking other player for their respondents or by making a guess if player A has enough information about the game state. "
            "Other players return the secret number. Some players are liars. They return a number not equal to the secret number.\n"
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

        elif action_type == "Guess":
            guessed_number = action.get("Number")
            speaker = action.get("Speaker")

            # Update game state with the guessed number.
            self.guess = guessed_number
        
            # Log the guess in the conversation manager.
            conversation_manager.add_message_to_conversation(speaker, speaker, f"My guess is {guessed_number}.")
            
        return True, 0


# ------------------ MCTS with LLM Integration ------------------

def reward_function(node, new_node):
    reward = 0
    speaker = new_node.action.get("Speaker")
    num_convs = 1
    num_player_talked_to = 3
    conv_old = node.state.features.count_num_player_conversations_greater(speaker, num_convs)
    conv_new = new_node.state.features.count_num_player_conversations_greater(speaker, num_convs) 
    
    if new_node.is_terminal and str(new_node.state.guess) == str(new_node.state.secret_number) and conv_new >= num_player_talked_to:
        return reward_terminal * 8;
    elif new_node.is_terminal and str(new_node.state.guess) == str(new_node.state.secret_number):
        return reward_terminal

    if new_node.is_terminal and str(new_node.state.guess) != str(new_node.state.secret_number):
        return -reward_terminal;

    num_convs = 1
    conv_old = node.state.features.count_num_player_conversations(speaker, num_convs)
    conv_new = new_node.state.features.count_num_player_conversations(speaker, num_convs)
    
    if conv_new > conv_old:
        reward = reward + reward_small

    conv_old = node.state.features.updated_private_info(speaker)
    conv_new = new_node.state.features.updated_private_info(speaker)

    if conv_new > conv_old:
        reward = reward + reward_small

    if reward <= 0:
        reward = reward -reward_node  # Reward
    return reward

# ------------------ Main ------------------

log = []
start_time = time.time()  # Start timing

folder_path = 'training'
conversationManager = ConversationManager()

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

num_correct_games = 0
num_correct_high_reward_games = 0
model = init_model(model_id, server_based, max_token)
model = [model]
for i in range(num_games):
    print('Start Game')
    print(str(i) + ' / ' + str(num_games))
    game_state = SimpleNumberGuessGameState(['A', 'B', 'C', 'D'])
    game_state.add_next_player('A')
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
