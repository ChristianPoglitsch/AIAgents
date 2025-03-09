import json
import random
import torch

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

import os
from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages

# ------------------ LLM Integration Stub ------------------

model = []

server_based = False
use_trained = True

def init_model() -> LLM_API:
    if server_based:
        return init_model_server()
    else:
        if use_trained:
            return init_model_local_trained()
        else:
            return init_model_local()

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

    def increment_conversation(self, player, other_player):
        """
        Increments the conversation count for the specified player's observation about another player.
        
        :param player: The name of the player whose features are being updated.
        :param other_player: The name of the other player with whom a conversation occurred.
        """
        if player in self.features and other_player in self.features[player]:
            self.features[player][other_player][0] += 1
        else:
            raise ValueError(f"Either {player} or {other_player} is not a valid player.")

    def update_private_info(self, player, other_player, new_info):
        """
        Updates the private info string for a player's observation about another player.
        
        :param player: The name of the player whose features are being updated.
        :param other_player: The other player for whom to update the information.
        :param new_info: A string containing the new private information.
        """
        if player in self.features and other_player in self.features[player]:
            self.features[player][other_player][1] = new_info
        else:
            raise ValueError(f"Either {player} or {other_player} is not a valid player.")

    def __str__(self):
        """
        Returns a string representation of the entire private feature matrix.
        """
        result_lines = []
        for player, info in self.features.items():
            result_lines.append(f"Private features for {player}:")
            for other, stats in info.items():
                result_lines.append(f"  {other}: Conversations: {stats[0]}, Private Info: {stats[1]}")
        return "\n".join(result_lines)

    def generate_private_info_update_prompt(self, player, conversation_history):
        """
        Generates an LLM prompt to update a player's private features based on recent conversation history.
    
        :param private_features_obj: Instance of PlayerPrivateInformation.
        :param player: The player whose private features should be updated.
        :param conversation_history: A string containing the last x conversation messages involving that player.
        :return: A prompt string.
        """
        current_features = self.features.get(player, {})
        prompt = (
            "You are an assistant tasked with updating a player's private features based on recent conversation history. "
            "The player's private feature state contains, for each other player, the number of conversations they've had "
            "and a string with privately generated information about that player.\n\n"
            "Recent Conversation History:\n"
            f"{conversation_history}\n\n"
            "Current Private Feature State:\n"
        )
        for other, stats in current_features.items():
            prompt += f"{other}: Conversations = {stats[0]}, Private Info = {stats[1]}\n"
        prompt += (
            "\nBased on the conversation history, please update the private feature state for each other player "
            "and output the updated state in JSON format with keys for each player and values being an object "
            "with 'conversations' and 'private_info' fields."
            "Do NOT use any markdown formatting (e.g., ```json) in your response and use double quotes."
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
            # Only update features if the other player exists in this player's feature space.
            if other_player in self.features[player]:
                conversations = data.get("conversations", 0)
                # If private_info is None, store it as "None" (or choose a default value).
                private_info = data.get("private_info") if data.get("private_info") is not None else "None"
                self.features[player][other_player] = [conversations, private_info]
            else:
                # Optionally, add a new entry if other_player is not yet present.
                conversations = data.get("conversations", 0)
                private_info = data.get("private_info") if data.get("private_info") is not None else "None"
                self.features[player][other_player] = [conversations, private_info]


# ------------------ Game Environment / State ------------------

class BasicGameState:
    def __init__(self, players):
        # Common elements: list of players.
        self.players = players

    def randomly_select_player(self):
        """Randomly select one player from the list."""
        return random.choice(self.players)

    def get_game_state(self, player):
        """
        This method should be implemented by subclasses to return a 
        human-readable public state description.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_game_state_prompt(self, current_player, conversation_history):
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_prompt(self, current_player, conversation_history):
        raise NotImplementedError("Subclasses must implement this method.")


class SimpleNumberGuessGameState(BasicGameState):
    def __init__(self, players):
        super().__init__(players)
        # Secret number is either 0 or 100.
        self.secret_number = random.choice([0, 100])
        # Randomly select one respondent (from B, C, D) to be the liar.
        self.liar = random.choice(['B', 'C', 'D'])
        # Initialize features for each player.
        # Feature vector: [#asked, response for number]
        self.features = PlayerFeatures(players)

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
            {'type': 'No Action', 'Speaker': None}
        ]
        if current_player == "A":
            # Insert the Guess action at the beginning if the current player is A.
            actions.insert(0, {'type': 'Guess', 'Speaker': None, 'Number': None})
        return actions

    def update_features_from_json(self, player, json_data):
        self.features.update_features_from_json(player, json_data)

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
        return f"{secret_info}\n\n{player_info}" + "\n" + self.game_state_features_to_string(player)

    def generate_game_state_prompt(self, current_player, conversation_history):
        return self.features.generate_private_info_update_prompt(current_player, conversation_history)

    def generate_prompt(self, current_player, conversation_history):
        """
        Creates a prompt string for the LLM based on the current game state,
        available actions, and conversation history.
        
        :param current_player: The name of the current player.
        :param action_space_description: A string describing the available action templates.
        :param conversation_history: A string representing the conversation history.
        :return: A prompt string in plain text.
        """
        # Get the public state description for the current player.
        state_description = self.get_game_state(current_player)
        
        prompt = (
            "You are a helpful board game AI assistant for the number guessing minigame. "
            "Player A's goal is to determine the secret number [0 to 100] by asking other player for their respondents or by making a guess if player A has enough information about the game state. "
            "Other players return the secret number. Some players are liars. They return a number not equal to the secret number. "
            f"Current Player: {current_player}\n\n"
            "The available actions are given below, but each action is incomplete and missing parameters marked as None.\n"
            "Available Actions Description:\n"
            f"{self.get_action_space_description(current_player)}\n\n"
            "Game State:\n"
            f"{state_description}\n\n"
            "Chronological conversation History:\n"
            f"{conversation_history}\n\n"
            "Please output one complete possible action from the Available Actions Description list in JSON format. "
            "Do NOT use any markdown formatting (e.g., ```json) in your response and use double quotes. Replace all None parts in the action.\n\n"
        )
        return prompt


    def apply_action(self, action_processor, conversation_manager, action):
        """
        Updates the game state and conversation history based on the selected action.
        For both "Message" and "Guess" actions, calls complete_action_with_llm to generate a
        complete action and returns that result.
    
        :param game_state: The current game state (NumberGuessGameState).
        :param conversation_manager: The conversation manager to track interactions.
        :param action: A dictionary representing the action (from LLM or another source).
        :return: The completed action as generated by complete_action_with_llm.
        """
    
        action_type = action.get("type")
        result_action = None

        if action_type == "Message":
            # Extract speaker and audience from the action.
            speaker = action.get("Speaker")
            audience = action.get("Audience")        
            # Ensure audience is handled as a list.
            if not isinstance(audience, list):
                audience = [audience]
        
            # For each respondent in the audience:
            for respondent in audience:
                result_action = action_processor.create_action(self, respondent, conversation_manager)

            # Update game state for player
            prompt_state, chat_completed_state, action_state = complete_action_with_llm(speaker, self.generate_game_state_prompt(speaker, conversation_manager.get_conversation_history_for_player(speaker)))
            
            if action_state is not str and action_state.get("action") is None:
                game_state.update_features_from_json(speaker, action_state)
                conversation_manager.store_prompt_outcome(prompt_state, chat_completed_state)

        elif action_type == "Guess":
            guessed_number = action.get("Number")
            speaker = action.get("Speaker")

            # Update game state with the guessed number.
            self.guess = guessed_number
        
            # Log the guess in the conversation manager.
            conversation_manager.add_message_to_conversation(speaker, speaker, f"My guess is {guessed_number}.")
            result_action = None

        elif action_type == "No Action":
            #print("No action selected")
            result_action = None

        else:
            #raise ValueError(f"Invalid action type: {action_type}")
            print(f"Invalid action type: {action_type}")
            result_action = None

        return result_action


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

    def get_conversation_for_player(self, player_name) -> Conversation:
        result = []
        for participants_tuple, conv in self.conversations.items():
            if player_name in participants_tuple:
                result.append(conv)
        return result

    def get_conversation_history_for_player(self, current_player) -> str:
        """
        Retrieves and returns a formatted conversation history for the given player.
    
        :param conversation_manager: The ConversationManager instance that manages all conversations.
        :param current_player: The name of the player for whom to retrieve the conversation history.
        :return: A string containing the conversation history, or a default message if none exist.
        """
        convs = self.get_conversation_for_player(current_player)
        if convs:
            conversation_history = "\n".join(conv.get_history_text() for conv in convs)
        else:
            conversation_history = "No conversation history."
        return conversation_history

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

    def set_prompt_outcomes(self, log):
        """
        Stores the list of all stored (prompt, outcome) tuples.
        
        :param log: List of tuples (prompt, outcome).
        """
        self.prompt_outcome_log = log

    def export_prompt_outcome_log(self, file_path):
        """
        Exports the prompt_outcome_log as a Hugging Face Dataset formatted for Mistral-7B-Instruct training.
    
        Each record follows the ChatML-style format:
            <s>[INST] instruction + input [/INST] output </s>

        If `input_text` is available, it's included as additional context.

        The dataset is then saved to disk at the provided file path.
    
        If the file_path exists, it is removed beforehand.

        Assumption:
            self.prompt_outcome_log is a list of tuples (instruction, input_text, output).
    
        :param file_path: The directory path where the dataset will be saved.
        """
        # Remove existing dataset directory
        if os.path.exists(file_path):
            os.system(f"rm -rf {file_path}")  # Alternative: use `shutil.rmtree(file_path)`

        inputs, outputs = [], []

        for input, output in self.prompt_outcome_log:
            inputs.append(input.strip())
            outputs.append(output.strip())

        # Convert to Hugging Face Dataset format
        dataset = Dataset.from_dict({"input": inputs, "output": outputs})

        # Save dataset
        dataset.save_to_disk(file_path)



# ------------------ Actions classes ------------------

class ActionProcessor:
    def __init__(self, game_state, conversation_manager):
        """
        Initializes the action processor with the current game state and conversation manager.
        
        :param game_state: The current game state (e.g., an instance of NumberGuessGameState).
        :param conversation_manager: The conversation manager handling conversation histories.
        """
        self.game_state = game_state
        self.conversation_manager = conversation_manager
        self.action_queue = []  # List to hold actions (each action is a dictionary)
        self.action_budget = 30

    def add_action(self, action):
        """
        Adds an action or a list of actions to the action queue.
    
        :param action: A dictionary representing an action, or a list containing one or more such dictionaries.
        """
        if isinstance(action, list):
            # If action is a list, extend the queue with all its items.
            self.action_queue.extend(action)
        else:
            # Otherwise, append the single action.
            self.action_queue.append(action)
         
        if not isinstance(action, list):
            action = [action]            
        for act in action:
            self.conversation_manager.add_action_to_conversation(act)

    def process_actions(self):
        """
        Processes all actions in the queue by applying each action using the apply_action function.
        Any actions returned by apply_action (if the result is a list, all its items) are added to the queue.
        The process continues until the queue is empty.
        """

        terminal_state = False

        while not terminal_state and self.action_budget > 0:

            if not self.action_queue:
                action = self.create_action(game_state, game_state.randomly_select_player(), self.conversation_manager)
                # If a result is returned, add it/them to the queue.
                if action is not None:
                    if isinstance(action, list):
                        self.action_queue.extend(action)  # Add all items if result is a list.
                    else:
                        self.action_queue.append(action)  # Add the single action.

            # Pop the first action from the queue.
            action = self.action_queue.pop(0)
            #action_type = action.get("type")

            # Apply the action. This function updates the game state and conversation manager.
            action = self.game_state.apply_action(self, self.conversation_manager, action)
            self.action_budget = self.action_budget - 1

            # If a result is returned, add it/them to the queue.
            if action is not None:
                if isinstance(action, list):
                    self.action_queue.extend(action)  # Add all items if result is a list.
                else:
                    self.action_queue.append(action)  # Add the single action.
    
            if game_state.is_terminal() is not None:
                terminal_state = True

        return

    def create_action(self, game_state, player, conversation_manager):
        """
        Calls complete_action_with_llm for the given respondent and logs any completed action(s) of type "Message"
        into the conversation manager.
    
        :param game_state: The current game state.
        :param respondent: The respondent for which the action completion is requested.
        :param conversation_manager: The conversation manager to log messages.
        """
        # Call the LLM to complete the action for this respondent.
        prompt, chat_completed, result_action = complete_action_with_llm(player, game_state.generate_prompt(player, conversation_manager.get_conversation_history_for_player(player)))

        # If the result is a list, iterate over its items.
        if isinstance(result_action, list):
            for item in result_action:
                if item.get("type") in ["Message"]:
                    speaker = item.get("Speaker")
                    audience = item.get("Audience")
                    # Ensure audience is a list
                    if not isinstance(audience, list):
                        audience = [audience]
                    participants = [speaker] + audience
                    conversation_manager.add_message_to_conversation(participants, speaker, item)

        else:
            # If result_action is not a list and is of type "Message"
            if result_action.get("type") in ["Message"]:
                speaker = result_action.get("Speaker")
                audience = result_action.get("Audience")
                if not isinstance(audience, list):
                    audience = [audience]
                participants = [speaker] + audience
                conversation_manager.add_message_to_conversation(participants, speaker, result_action)
                
        conversation_manager.store_prompt_outcome(prompt, chat_completed)
            
        return result_action

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
    
    #print("LLM Prompt:\n", prompt)
    
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

# ------------------ ISMCTS with LLM Integration ------------------

def ismcts_with_llm(game_state, current_player, num_simulations, conversation_manager, gnn_model):
    """
    A simplified ISMCTS-like function that uses an LLM to decide on an action for the number guessing game.
    This function is adapted for the minigame, where Player A's possible actions are:
      "Ask B", "Ask C", "Ask D", "Guess 0", and "Guess 1".
    The LLM is provided with the game state and conversation history and must output a complete action (as a string)
    along with its reasoning.
    
    :param game_state: The current NumberGuessGameState instance.
    :param current_player: The current player's name (should be 'A').
    :param num_simulations: Number of ISMCTS simulations (not used extensively in this simple stub).
    :param conversation_manager: ConversationManager instance.
    :param gnn_model: The GNN model (not directly used in this LLM stub version).
    :return: The recommended action from the LLM as a string, along with its reasoning.
    """
    
    actionProcessor = ActionProcessor(game_state, conversation_manager)
    actionProcessor.process_actions()
    
    return 0


# ------------------ Main ------------------

model = init_model()

log = []

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

gnn_model = ActionPredictionGNN(input_dim=128, hidden_dim=16, output_dim=2)


for x in range(5):

    game_state = SimpleNumberGuessGameState(['A', 'B', 'C', 'D'])

    # Create a dummy ConversationManager and add a conversation.
    conv_manager = ConversationManager()

    # Use the ISMCTS with LLM integration to get an action decision.
    _ = ismcts_with_llm(game_state, current_player='A', num_simulations=50, 
                               conversation_manager=conv_manager,
                               gnn_model=gnn_model)

    print('Secret number: ' + str(game_state.secret_number))
    print('Guess: ' + str(game_state.guess))
    print('Liar: ' + str(game_state.liar))
    #conv_manager.get_all_conversations_for_player_print()
    conv_manager.print_all_conversations()
    print("Result: " + str(game_state.guess == game_state.secret_number))

    if game_state.guess == game_state.secret_number:
        log.extend(conv_manager.get_prompt_outcomes())


if log:
    conv_manager.set_prompt_outcomes(log)
    file_name = 'training.csv'
    conv_manager.export_prompt_outcome_log(file_name)

    dataset = load_from_disk(file_name)
    print("Dataset loaded from:", file_name)
    for record in dataset:
        print(record["input"])
   
