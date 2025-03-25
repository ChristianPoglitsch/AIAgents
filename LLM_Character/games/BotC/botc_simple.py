from hmac import new
import json
import random
import torch
import copy
import math
import time

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import os
import shutil
from datasets import Dataset
from datasets import load_from_disk, concatenate_datasets

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages

# ------------------ LLM Integration Stub ------------------

model = []

server_based = False
use_trained = False
store_data = False
show_training_data = False

reward_terminal = 16
reward_small = 4
reward_node = 0.25

num_child_node = 1 # 3
num_games = 5 # 30
num_iterations = 40 # 50

print_output = True


def init_model() -> LLM_API:
    if server_based:
        return init_model_server()
    else:
        return init_model_local()
        #if use_trained:
        #    return init_model_local_trained()
        #else:
        #    return init_model_local()

def init_model_server() -> LLM_API:
    model = OpenAIComms()
    model_id = "gpt-4o"
    model.max_tokens = 200
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

def init_model_local() -> LLM_API:
    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #model_id = "deepseek-ai/deepseek-llm-7b-chat"
    #model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"
    if use_trained:
        model_id = "trained/Mistral-7B-Instruct-v0.3_merged"
    model.max_tokens = 200
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

def init_model_local_trained() -> LLM_API:
    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    #model_id = "deepseek-ai/deepseek-llm-7b-chat"
    #model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"
    model.max_tokens = 200
    model.init(model_id, "trained\\Mistral-7b-v3-finetune")
    wrapped_model = LLM_API(model)
    return wrapped_model

#def get_model() -> LLM_API:
#    return model


# ------------------ Private player info ------------------

class PlayerFeatures:
    def __init__(self, players):
        """
        Initializes the private feature matrix for all players.
        Each player gets a dictionary mapping every other player to a feature vector:
        [# conversations with that player, private info string].
        
        :param players: List of player names.
        """
        self.features = {
            player: {other: [0, "None"] for other in players if other != player}
            for player in players
        }

    def generate_private_info_update_prompt(self, player, conversation_history):
        """
        Generates an LLM prompt to update a player's private features based on recent conversation history.
    
        :param player: The player whose private features should be updated.
        :param conversation_history: A string containing the last x conversation messages involving that player.
        :return: A prompt string.
        """
        current_features = self.features.get(player, {})
        prompt = (
            "You are an assistant tasked with updating a player's private features based on recent conversation history. "
            "The player's private feature state contains, for each other player, the number of conversations they've had "
            "and a string with privately generated information about that player.\n\n"
            f"Current Player: {player}\n\n"
            "Recent Conversation History:\n"
            f"{conversation_history}\n\n"
            "Current Feature State for other player:\n"
        )
        for other, stats in current_features.items():
            prompt += f"{other}: Conversations = {stats[0]}, Private Info = {stats[1]}\n"
        prompt += (
            "\nBased on the conversation history and the messages, please update the Private Info (only text, as summary of the conversation history and the current Feature State) for each other player. Do not add the Current Player."
            "First, think about the update the number of conversations, second about an update for Private info about other players. Third, reply the Feature State struct in JSON format."            
            "Output the updated state in JSON format with keys for each player and values being an object "
            "with 'Conversations' and 'Private Info' fields."
            "Do NOT use any markdown formatting (e.g., ```json) in your response and use double quotes. Only return the JSON."
            
        )
        return prompt

    def update_features_from_json(self, player, json_data):
        """
        Updates the feature vectors for the given player using data provided in json_data.
        
        :param player: The player whose feature space will be updated.
        :param json_data: A dictionary (parsed from JSON) with keys representing other players and values as dictionaries with:
                          { "conversations": int, "private_info": str or None }
        Example json_data:
        {
          "B": {
            "conversations": 1,
            "private_info": "Secret number is 77."
          },
          "C": {
            "conversations": 0,
            "private_info": null
          },
          "D": {
            "conversations": 0,
            "private_info": null
          }
        }
        """
        if player not in self.features:
            raise ValueError(f"Player {player} not found in features.")
            
        for other_player, data in json_data.items():
            # Optionally, add a new entry if other_player is not yet present.
            conversations = data.get("Conversations", 0)
            private_info = data.get("Private Info") if data.get("Private Info") is not None else "None"
            self.features[player][other_player] = [conversations, private_info]

    def count_num_player_conversations(self, player, num_convs):
        """
        Returns the number of players with whom the given player has had more than `num_convs` conversations.

        :param player: The player whose conversation history will be checked.
        :param num_convs: The threshold number of conversations.
        :return: An integer count of players.
        """
        if player not in self.features:
            raise ValueError(f"Player {player} not found in features.")

        return sum(1 for stats in self.features[player].values() if stats[0] == num_convs)
    
    def updated_private_info(self, player):
        """
        Returns the number of players with whom the given player has had more than `num_convs` conversations.

        :param player: The player whose conversation history will be checked.
        :param num_convs: The threshold number of conversations.
        :return: An integer count of players.
        """
        if player not in self.features:
            raise ValueError(f"Player {player} not found in features.")

        return sum(1 for stats in self.features[player].values() if stats[1] != 'None')

# ------------------ Game Environment / State ------------------

class BasicGameState:
    def __init__(self, players):
        # Common elements: list of players.
        self.players = players
        self.next_players = []
        self.no_action = {'type': 'No Action', 'Speaker': None}
        
        # Initialize features for each player.
        # Feature vector: [#asked, response for number]
        self.features = PlayerFeatures(players)

    def get_no_action(self):
        return self.no_action

    def randomly_select_player(self):
        """Randomly select one player from the list."""
        return random.choice(self.players)

    def add_next_player(self, player):
        """Add a player or a list of players to the queue."""
        if isinstance(player, list):  # Check if input is a list
            for p in player:
                self.next_players.append(p)  # Add only valid players
        else:
            self.next_players.append(player)  # Add a single valid player

    def get_player(self):
        """Get the first player in queue or randomly select if empty."""
        if self.next_players:
            return self.next_players.pop(0)  # FIFO: Get first in queue
        return random.choice(self.players)  # Select randomly if queue is empty

    def get_game_state(self, player):
        """
        This method should be implemented by subclasses to return a 
        human-readable public state description.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_game_state_prompt(self, current_player, conversation_history):
        return self.features.generate_private_info_update_prompt(current_player, conversation_history)

    def generate_prompt(self, current_player, conversation_history):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def is_terminal(self):
        """Game ends when a guess is made."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_action_space_description(self, current_player):
        raise NotImplementedError("Subclasses must implement this method.")

    def create_action(self, player, conversation_manager):
        """
        Calls complete_action_with_llm for the given respondent and logs any completed action(s) of type "Message"
        into the conversation manager.
    
        :param respondent: The respondent for which the action completion is requested.
        :param conversation_manager: The conversation manager to log messages.
        """
        # Call the LLM to complete the action for this respondent.
        return complete_action_with_llm(player, self.generate_prompt(player, conversation_manager))

    def apply_action(self, conv_manager, action):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def game_state_features_to_string(self, player):
        """
        Returns a human-readable string representation of the entire feature space for the specified player.
        The feature space is a dictionary mapping every other player in the game to a feature vector,
        where the feature vector is of the form [# conversations, private info string].

        :param player: The player for whom to return the feature space.
        :return: A formatted string representing the feature space for the specified player.
        """
        if player not in self.features.features:
            return f"Player {player} not found in feature space."
    
        result_lines = [f"Feature space for player {player}:"]
        for other_player, feature_vector in self.features.features[player].items():
            line = (f"{other_player}: number of conversations = {feature_vector[0]}, "
                    f"private info = {feature_vector[1]}")
            result_lines.append(line)
        return "\n".join(result_lines)


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
        :return: A list of dictionaries describing the available actions.
        """
        actions = [
            {'type': 'Message', 'Speaker': None, 'Audience': None, 'Message:': None},
            self.no_action
        ]
        if current_player == "A":
            # Insert the Guess action at the beginning if the current player is A.
            actions.insert(0, {'type': 'Guess', 'Speaker': None, 'Number': None})

        actions = str(actions).replace("'", '"')
        return actions

    def is_terminal(self):
        """Game ends when a guess is made."""
        return self.guess

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

        prompt = prompt + "Please output one complete possible action from the Available Actions Description list in JSON format.\n"
        prompt = prompt + "Do NOT use any markdown formatting (e.g., ```json) in your response and use double quotes. Replace all None parts in the action.\n"
        prompt = prompt + "First, think about a possible answer, second reply the action.\n"

        return prompt


    def apply_action(self, conv_manager, action):
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
            #union_set = set(speaker) | set(audience)  # Union of both sets

            # Update game state for player
            prompt_state, action_state = complete_action_with_llm(speaker, self.generate_game_state_prompt(speaker, conv_manager.get_conversation_history_for_player(speaker)))
            if action_state is not str and action_state.get("action") is None and action_state.get("error") is None:
                self.features.update_features_from_json(speaker, action_state)
                conv_manager.store_prompt_outcome(prompt_state, json.dumps(action_state))

        elif action_type == "Guess":
            guessed_number = action.get("Number")
            speaker = action.get("Speaker")

            # Update game state with the guessed number.
            self.guess = guessed_number
        
            # Log the guess in the conversation manager.
            conv_manager.add_message_to_conversation(speaker, speaker, f"My guess is {guessed_number}.")


# ------------------ Graph Construction ------------------

def game_state_to_graph_data(game_state):
    """
    Converts the number guessing game state into a PyTorch Geometric Data object.
    Nodes: one for each player.
    Node features: a 3-dimensional vector for each player.
    Edges: For simplicity, create a complete graph among players.
    """
    node_features = [game_state.features[p] for p in game_state.players]
    x = torch.tensor(node_features, dtype=torch.float)  # shape: (num_players, 3)
    
    num_nodes = len(game_state.players)
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_list.append((i, j))
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

# ------------------ Conversation Classes ------------------

class Conversation:
    def __init__(self, participants):
        """
        Initializes a conversation with a list of players.
        """
        self.participants = participants  # List of players involved
        self.history = []  # List to store conversation messages

    def add_message(self, sender, message):
        """Adds a message to the conversation history."""
        if sender in self.participants:
            self.history.append({'sender': sender, 'message': message})
        else:
            raise ValueError(f"{sender} is not a participant in this conversation.")

    def get_history_text(self):
        """Concatenate all messages into one string."""
        return " ".join([f"{entry['sender']}: {entry['message']}" for entry in self.history])

    def get_participants(self):
        return self.participants

class ConversationManager:
    def __init__(self):
        self.conversations = {}  # key: tuple(participants), value: Conversation instance
        self.prompt_outcome_log = []  # List to store tuples of (prompt, outcome) as strings
        self.player_plans = {}  # Dictionary to store plans for players
        
    # Setter function to store a tuple of (player, plan)
    def store_player_plan(self, player, plan_str):
        """
        Stores a player's game plan as both a string and a parsed JSON object.

        :param player: The name of the player.
        :param plan_str: The plan as a string (expected to be in JSON format).
        """
        try:
            plan_json = json.loads(plan_str)  # Convert string to JSON
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for plan.")

        self.player_plans[player] = (plan_str, plan_json)

    # Getter function to retrieve a player's stored plan
    def get_player_plan(self, player):
        """
        Retrieves the stored plan for a player.

        :param player: The name of the player.
        :return: A tuple of (plan_str, plan_json) if found, otherwise None.
        """
        return self.player_plans.get(player, None)

    def start_conversation(self, participants):
        participants_tuple = tuple(sorted(participants))
        if participants_tuple not in self.conversations:
            conv = Conversation(participants)
            self.conversations[participants_tuple] = conv

    def add_message_to_conversation(self, participants, sender, message):
        participants_tuple = tuple(sorted(participants))
        if participants_tuple not in self.conversations:
            self.start_conversation(participants)
        self.conversations[participants_tuple].add_message(sender, message)

    def add_action_to_conversation(self, action):
        if not isinstance(action, list):
            action = [action]
        for act in action:
            speaker = act.get("Speaker")
            audience = act.get("Audience")
            if not isinstance(audience, list):
                audience = [audience]
            participants = [speaker] + audience
            if act.get("type") in ["Message"]:
                self.add_message_to_conversation(participants, speaker, act)

    def get_conversation_for_player(self, player_name, num_convs = 5) -> Conversation:
        result = []
        for participants_tuple, conv in self.conversations.items():
            if player_name in participants_tuple:
                result.append(conv.history[-num_convs:])
        return result  # Return only the last `num_convs` conversations

    def get_conversation_history_for_player(self, current_player, num_convs=5) -> str:
        """
        Retrieves and returns a formatted conversation history for the given player.

        :param current_player: The name of the player for whom to retrieve the conversation history.
        :param num_convs: The number of recent conversations to retrieve (default: 5).
        :return: A formatted string containing the conversation history, or a default message if none exist.
        """
        convs = self.get_conversation_for_player(current_player, num_convs)

        if not convs or not any(convs):  # Check if convs is empty or contains empty lists
            return "No conversation history."

        formatted_conversations = []
    
        for conversation in convs:
            for entry in conversation:
                sender = entry.get("sender", "Unknown")
                message = entry.get("message", {})
                formatted_message = str(message)  # Convert message dictionary to string if necessary
                formatted_conversations.append(f"{sender}: {formatted_message}")

        return "\n".join(formatted_conversations)

    def print_all_conversations(self):
        """
        Prints all conversations stored in the ConversationManager.
        For each conversation, it prints the list of participants and then each message.
        """
        for participants, conversation in self.conversations.items():
            print(f"Conversation among: {', '.join(participants)}")
            for entry in conversation.history:
                print(f"{entry['sender']}: {entry['message']}")
            print("-" * 40)

    def extract_all_unique_participants(self):
        """
        Extracts a list of unique individual participant names from the ConversationManager.
        :return: A list of unique participant names.
        """
        unique_participants = []
        for conv in self.conversations.values():
            for participant in conv.get_participants():
                if participant not in unique_participants:
                    unique_participants.append(participant)
        return unique_participants

    def get_all_conversations_for_player_print(self):
        """
        Prints all conversations stored in the ConversationManager that involve each unique player.
        """
        unique_participants = self.extract_all_unique_participants()
        for player_name in unique_participants:
            convs = self.get_conversation_for_player(player_name)
            if convs:
                print(f"Conversation where {player_name} is involved:")
                for conv in convs:
                    for entry in conv.history:
                        print(f"{entry['sender']}: {entry['message']}")
                print("-" * 40)
            else:
                print(f"No conversations found for player: {player_name}")

    def store_prompt_outcome(self, prompt, outcome):
        """
        Stores a tuple of (prompt, outcome) in the prompt outcome log.
        
        :param prompt: The prompt string sent to the LLM.
        :param outcome: The outcome string received from the LLM.
        """
        self.prompt_outcome_log.append((prompt, outcome))

    def get_prompt_outcomes(self):
        """
        Returns the list of all stored (prompt, outcome) tuples.
        
        :return: List of tuples (prompt, outcome).
        """
        return self.prompt_outcome_log

    def append_prompt_outcomes(self, other_instance):
        """
        Adds all stored (prompt, outcome) tuples from the current instance to another instance.
        
        :param other_instance: Another instance of PromptOutcomeStore where outcomes will be added.
        """
        self.prompt_outcome_log.extend(other_instance)

    def set_prompt_outcomes(self, log):
        """
        Stores the list of all stored (prompt, outcome) tuples.
        
        :param log: List of tuples (prompt, outcome).
        """
        self.prompt_outcome_log = log

    def export_prompt_outcome_log(self, file_path, append=False):
        """
        Exports the prompt_outcome_log as a Hugging Face Dataset formatted for Mistral-7B-Instruct training.
    
        Each record follows the ChatML-style format:
            <s>[INST] input [/INST] output </s>

        If `input_text` is available, it's included as additional context.

        The dataset is then saved to disk at the provided file path.
    
        If the file_path exists, it is removed beforehand.

        Assumption:
            self.prompt_outcome_log is a list of tuples (input, output).
    
        :param file_path: The directory path where the dataset will be saved.
        :param append: If True, load the existing dataset from file_path and append new data.        
        """
        inputs, outputs = [], []

        for i, o in self.prompt_outcome_log:
            inputs.append(i.strip())
            outputs.append(o.strip())

        # Convert to Hugging Face Dataset format
        dataset = Dataset.from_dict({"input": inputs, "output": outputs})

        if append and os.path.exists(file_path):
            existing_dataset = load_from_disk(file_path)
            combined_dataset = concatenate_datasets([existing_dataset, dataset])
        else:
            combined_dataset = dataset

        # Save dataset
        # Remove existing dataset directory
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
        combined_dataset.save_to_disk(file_path)

# ------------------ GNN Model ------------------

class ActionPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        GNN that outputs log probabilities over actions.
        :param input_dim: Dimensionality of node features (here, 3).
        :param hidden_dim: Hidden layer size.
        :param output_dim: Number of actions (5).
        """
        super(ActionPredictionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Aggregate node features (average pooling)
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return F.log_softmax(x, dim=0)

# ------------------ LLM Action Completion Function ------------------

def complete_action_with_llm(current_player, prompt):
    """
    Given the global actions (action templates) for the number guessing game,
    ask the LLM to complete an action (fill in missing parameters) and return a JSON object.
    The JSON should have two keys: "conversation" (a list of conversation messages)
    and "action" (the selected complete action as a string).
    
    :param current_player: Current player name (should be 'A').
    :param prompt: generated prompt
    :return: The LLM's JSON response as a dictionary.
    """ 
    
    if print_output:
        print(prompt)

    # Prepare messages for the LLM.
    if server_based:
        role = "developer"
    else:
       role = "user"
    messages = AIMessages()
    message = AIMessage(message=prompt, role=role, class_type="MessageAI", sender=role)
    messages.add_message(message)
    
    # Initialize the LLM.
    #model = get_model()
    llm_response = model.query_text(messages)
    
    try:
        result = json.loads(llm_response)
    except Exception as e:
        if print_output:
            print("Error parsing LLM response: " + ' \nMessage:\n' + llm_response, e)
        result = {"error": "No Action", 'Speaker': current_player}

    if print_output:
        print("LLM Responds: " + current_player +"\n", result)

    print('--- --- --- ---')
    return prompt, result


def complete_game_state_with_llm(game_state, current_player, conversation_manager):
    """
    Given the global actions (action templates) for the number guessing game,
    ask the LLM to complete an action (fill in missing parameters) and return a JSON object.
    The JSON should have two keys: "conversation" (a list of conversation messages)
    and "action" (the selected complete action as a string).
    
    :param game_state: Current game state (NumberGuessGameState).
    :param current_player: Current player name (should be 'A').
    :param conversation_manager: ConversationManager instance with conversation history.
    :param global_actions: List of action templates.
    :return: The LLM's JSON response as a dictionary.
    """
    conversation_history = conversation_manager.get_conversation_history_for_player(current_player)
    
    # Create a prompt.
    prompt = game_state.generate_game_state_prompt(current_player, conversation_history)    
   
    # Prepare messages for the LLM.
    if server_based:
        role = "developer"
    else:
        role = "user"
    messages = AIMessages()
    message = AIMessage(message=prompt, role=role, class_type="MessageAI", sender=role)
    messages.add_message(message)
    
    # Initialize the LLM.
    #model = get_model()
    llm_response = model.query_text(messages)
    
    try:
        result = json.loads(llm_response)
    except Exception as e:
        print("Error parsing LLM response: " + ' \nMessage:\n' + llm_response, e)
        result = {"action": "No Action", 'Speaker': current_player}

    print("LLM Responds: " + current_player +"\n", result)

    # Take most plausible results
    if isinstance(result, list):
        result = random.choice(result)
        print("Choosen respond: " + current_player +"\n", result)

    return prompt, llm_response, result


# ------------------ MCTS with LLM Integration ------------------

def simulation_policy(node):
    """
    Uses ActionProcessor to simulate actions and generate the next game state and conversation state.
    
    :param game_state: The current game state.
    :param conversation_manager: The current conversation manager.
    :return: A new (game_state, conversation_manager) pair.
    """
    # Create deep copies to avoid modifying the original objects
    game_state = node.state
    conversation_manager = node.conversation_manager
    terminal_state = False
    result_action = []
    prompt = ''

    player = game_state.get_player()
    for i in range(num_child_node):
        model.set_temperature(max(0.2, 1.2 - i * 0.2))
        prompt, result = game_state.create_action(player, conversation_manager)
        if result not in result_action:
            result_action.append(result)

    # Process each action in result_action
    child_nodes = []
    for action in result_action:
        game_state_copy = copy.deepcopy(game_state)
        conversation_manager_copy = copy.deepcopy(conversation_manager)
        terminal_state = False  # Reset for each iteration

        # If action is of type "Message"
        if action.get("type") == "Message":
            speaker = action.get("Speaker")
            audience = action.get("Audience")
            if not isinstance(audience, list):
                audience = [audience]  # Ensure audience is a list
            participants = [speaker] + audience

            conversation_manager_copy.add_message_to_conversation(participants, speaker, action)

        conversation_manager_copy.store_prompt_outcome(prompt, json.dumps(action))
        game_state_copy.apply_action(conversation_manager_copy, action)

        if game_state_copy.is_terminal() is not None:
            terminal_state = True

        child_node = MCTSNode(game_state_copy, action, conversation_manager_copy, terminal_state, parent=node)
        child_nodes.append(child_node)  # Collect nodes in a list

    return child_nodes  # Always return a list of nodes


def reward_function(node, new_node):
    if new_node.is_terminal and str(new_node.state.guess) == str(new_node.state.secret_number):
        return reward_terminal;

    if new_node.is_terminal and str(new_node.state.guess) != str(new_node.state.secret_number):
        return -reward_terminal;

    reward = 0
    speaker = new_node.action.get("Speaker")
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


class MCTSNode:
    def __init__(self, state=None, action=None, conversation_manager=None, terminal_state=False, parent=None):
        self.state = state
        self.conversation_manager = conversation_manager
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.is_terminal = terminal_state
        self.action = action  # Action that led to this node

    def best_child(self, exploration_weight=1.0):
        if not self.children:
            return None  # Return None if there are no children to select from

        if self.parent is None:
            # If there is no parent, use only the exploitation term (greedy selection)
            return max(self.children, key=lambda child: child.value / (child.visits + 1e-6))

        # Compute UCT (Upper Confidence Bound for Trees) formula when the parent exists
        return max(
            self.children,
            key=lambda child: (
                child.value / (child.visits + 1e-6)  # Exploitation term
                + exploration_weight * ((2 * (self.parent.visits + 1) / (child.visits + 1e-6)) ** 0.5)  # Exploration term
            )
        )

    def best_leaf(self, exploration_weight=1.0):
        if not self.children:
            return self  # If this is a leaf node, return itself

        # If there are children, select the best one using UCT
        best_node = self.best_child(exploration_weight)
    
        return best_node.best_leaf(exploration_weight)  # Recursively go down until a leaf node is found

    def expand(self):
        # For the purpose of the example, let's expand the node by adding some dummy children
        # Assume that state changes with the action taken at this node.
        if not self.is_terminal:
            self.children = [
                MCTSNode(state=f"{self.state} -> Action A", action="Action A", parent=self),
                MCTSNode(state=f"{self.state} -> Action B", action="Action B", parent=self)
            ]
            #print(f"Expanding node at state: {self.state}")
        else:
            print(f"Node at state {self.state} is terminal, cannot expand.")

    def simulate(self):
        # A dummy simulation function
        # Here, we just return a constant value to simulate an action outcome
        return random.random()  # For simplicity, always return random value as the result of a simulation

    def __str__(self):
        # String representation for easy printing of the node's details
        return f"State: {self.state}, Value: {self.value}, Visits: {self.visits}, Terminal: {self.is_terminal}"


class MCTS:
    def __init__(self, simulation_policy, reward_function, iterations=100):
        self.simulation_policy = simulation_policy
        self.reward_function = reward_function
        self.iterations = iterations
        self.root = None

    def search(self, initial_state, conversation_manager):
        self.root = self.get_node(initial_state, conversation_manager, initial_state.get_no_action())

        for _ in range(self.iterations):
            node = self.select(self.root)
            new_nodes = self.expand(node)

            if not new_nodes or not isinstance(new_nodes, list):  # Check if the list is empty or None
                continue

            for new_node in new_nodes:
                if new_node is None or new_node.action.get("type") == new_node.state.get_no_action().get("type"):
                    continue
                reward = self.reward_function(node, new_node)
                self.backpropagate(new_node, reward)

            print('*** *** *** *** *** *** ***')
        return self.root.best_leaf(exploration_weight=0.0)

    #def select(self, node):        
    #    while node.is_fully_expanded():
    #        node = node.best_child()
    #    return self.expand(node)

    def select(self, node):
        """Balanced exploration (new nodes) and exploitation (best child)"""

        if not node.is_terminal:
            if not node.children:
                return node  # No children, return itself
            
            # Otherwise, use UCT to select the best-explored child
            while(node.children):
                node = node.best_child()

        return node  # Return terminal node

    def expand(self, node):
        if node is None:
            return None
        if node.is_terminal:  # Stop expansion if game is over
            return node.parent if node.parent else node  # Move up if terminal  

        child_nodes = self.simulation_policy(node)
        
        node.children.extend(child_nodes)
        return child_nodes

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def get_node(self, state, conversation_manager, action):
        return MCTSNode(state, action, conversation_manager)

    def print_tree(self, node=None, depth=0):
        """
        Recursively prints the tree structure.

        :param node: The current node being printed (defaults to root if None).
        :param depth: The current depth in the tree (for indentation).
        """
        if node is None:
            if not self.root:
                print("Tree is empty.")
                return
            node = self.root  # Get the root node

        indent = "  " * depth
        if isinstance(node.state, str) or node.state is None:
            print(f"{indent}-")
        else:
            terminal_status = " (Terminal)" if node.is_terminal else ""
            print(f"{indent}- Guessed Number: {node.state.guess}, Correct Number: {node.state.secret_number} Visits: {node.visits}, Value: {node.value:.2f}{terminal_status}")

        for child in node.children:
            self.print_tree(child, depth + 1)

# ------------------ Main ------------------

# Initialize the root node
# Create the root node for the conversation
root = MCTSNode()
mcts = MCTS(simulation_policy, reward_function, iterations=num_iterations)
# Simulate expanding the tree
print("Starting tree expansion...\n")

# Expand the root node
root.expand()

# Display the root and its children after expansion
print(f"Root node: {root}")
for i, child in enumerate(root.children):
    print(f"Child {i+1}: {child}")

# Now, simulate the traversal and expansion process
for step in range(50):
    #print(f"\nStep {step + 1}:")
    
    # Select the best child based on the UCT formula
    best_leaf = root.best_leaf(exploration_weight=1.0)
    
    if best_leaf:
        #print(f"Selected leaf node: {best_leaf}")
        
        # Simulate an action on the best leaf node
        simulation_result = best_leaf.simulate()
        #print(f"Simulation result: {simulation_result}")
        
        # Update the best leaf node's value based on the simulation result
        best_leaf.value += simulation_result
        best_leaf.visits += 1  # Increment visits after simulating

        #print(f"Updated leaf after simulation: {best_leaf}")

        # Expand the selected leaf (if needed)
        best_leaf.expand()
    else:
        print("No leaf to explore.")
mcts.print_tree(root)



log = []
start_time = time.time()  # Start timing

folder_path = 'training'
conversationManager = ConversationManager()

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

gnn_model = ActionPredictionGNN(input_dim=128, hidden_dim=16, output_dim=2)

num_correct_games = 0
model = init_model()

for i in range(num_games):
    print('Start Game')
    print(str(i) + ' / ' + str(num_games))
    game_state = SimpleNumberGuessGameState(['A', 'B', 'C', 'D'])
    game_state.add_next_player('A')
    # Create a dummy ConversationManager and add a conversation.
    conv_manager = ConversationManager()

    # Create an MCTS instance
    mcts = MCTS(simulation_policy, reward_function, iterations=num_iterations)

    # Run MCTS to get the best action/state
    best_node = mcts.search(game_state, conv_manager)

    print(mcts.print_tree())

    print('Secret number: ' + str(best_node.state.secret_number))
    print('Guess: ' + str(best_node.state.guess))
    print('Liar: ' + str(game_state.liar))
    #conv_manager.get_all_conversations_for_player_print()
    #conv_manager.print_all_conversations()
    if game_state.guess is not None and game_state.secret_number is not None:
        print("Result: " + str(int(game_state.guess) == int(game_state.secret_number)))

    if str(best_node.state.guess) == str(best_node.state.secret_number):
        log.extend(best_node.conversation_manager.get_prompt_outcomes())
        num_correct_games = num_correct_games + 1

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

print("Successfull games: " + str(num_correct_games) + " / Played games: " + str(num_games))

end_time = time.time()
elapsed_time = end_time - start_time

if store_data and num_correct_games > 0:
    conversationManager.export_prompt_outcome_log(folder_path, True)
print(f"Execution time: {elapsed_time:.6f} seconds")
