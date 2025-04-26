import json
import random
import copy
import os
import math
import shutil
import re
from datasets import Dataset
from datasets import load_from_disk, concatenate_datasets

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages

num_conv_history_reasoning = 2
print_input = False

# ------------------ LLM Integration Stub ------------------

def init_model(model_id : str, server_based: str, max_token : int) -> LLM_API:
    if server_based:
        return init_model_server(max_token)
    else:
        return init_model_local(model_id, max_token)

def init_model_server(max_token : int) -> LLM_API:
    model = OpenAIComms()
    model_id = "gpt-4o"
    model.max_tokens = max_token
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

def init_model_local(model_id : str, max_token : int) -> LLM_API:
    model = LocalComms()
    model.max_tokens = max_token
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

#def init_model_local_trained() -> LLM_API:
#    model = LocalComms()
#    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
#    #model_id = "deepseek-ai/deepseek-llm-7b-chat"
#    #model_id = "openGPT-X/Teuken-7B-instruct-research-v0.4"
#    model.max_tokens = max_token
#    model.init(model_id, "trained\\Mistral-7b-v3-finetune")
#    wrapped_model = LLM_API(model)
#    return wrapped_model


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

    def generate_private_info_update_prompt(self, player, conversation_history, game_state):
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
        #for other, stats in current_features.items():
        #    prompt += f"{other}: Conversations = {stats[0]}, Private Info = {stats[1]}\n"
        

        data = {}
        for other, stats in current_features.items():
            data[other] = {"Conversations": stats[0], "Private Info": stats[1]}
        json_string = json.dumps(data, indent=4)  # Convert dictionary to JSON string with indentation
        prompt += json_string
        prompt += (
            "\n\nBased on the conversation history and the messages, please update the Private Info (only text, as summary of the conversation history and the current Feature State) for each other player. Do not add the Current Player.\n"
            "First, think about the update the number of conversations, second think about an update for private info about other players.\n"            
            "Return the updated Feature State in JSON format with keys for each player and values being an object "
            "with 'number of conversations' and 'private info' fields. Do NOT use any markdown formatting (e.g., ```json) in your response and use double quotes."
            
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
            "private_info": None
          },
          "D": {
            "conversations": 0,
            "private_info": None
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
    
    def count_num_player_conversations_greater(self, player, num_convs):
        """
        Returns the number of players with whom the given player has had more than `num_convs` conversations.

        :param player: The player whose conversation history will be checked.
        :param num_convs: The threshold number of conversations.
        :return: An integer count of players.
        """
        if player not in self.features:
            raise ValueError(f"Player {player} not found in features.")

        return sum(1 for stats in self.features[player].values() if stats[0] >= num_convs)

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

    def get_next_players_count(self):
        """
        Returns the number of players in the next_players list.
        """
        return len(self.next_players)

    def update_game_state(self):
        None

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
            
    def empty_next_players(self):
        """Empties the next_players queue."""
        self.next_players.clear()  # Clears the list of players
        
    def count_next_players(self):
        """Returns the count of players in the next_players queue."""
        return len(self.next_players)

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
        return self.features.generate_private_info_update_prompt(current_player, conversation_history, self.get_game_state(current_player))

    def generate_prompt(self, current_player, conversation_history):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def is_terminal(self):
        """Game ends when a guess is made."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def terminal_message(self):
        return ''

    def get_action_space_description(self, current_player):
        raise NotImplementedError("Subclasses must implement this method.")

    def create_action(self, player, conversation_manager, model, print_output, server_based, previous_results = None):
        """
        Calls complete_action_with_llm for the given respondent and logs any completed action(s) of type "Message"
        into the conversation manager.
    
        :param respondent: The respondent for which the action completion is requested.
        :param conversation_manager: The conversation manager to log messages.
        """
        # Call the LLM to complete the action for this respondent.
        prompt_template = self.generate_prompt(player, conversation_manager)
        prompt = self.generate_prompt(player, conversation_manager)
        if previous_results is not None:
            prompt = prompt + "\n Do not use this actions as result: " + previous_results
        _, result, errors = complete_action_with_llm(player, prompt, model, print_output, server_based)
        return prompt_template, result, errors
    
    def plan_action(self, speaker, conv_manager, model, print_output, server_based):
        """
        Calls complete_action_with_llm for the given respondent and logs any completed action(s) of type "Message"
        into the conversation manager.
    
        :param respondent: The respondent for which the action completion is requested.
        :param conversation_manager: The conversation manager to log messages.
        """
        # Call the LLM to complete the action for this respondent.
        prompt_state, action_state, error = complete_action_with_llm(speaker, self.generate_game_state_prompt(speaker, conv_manager.get_conversation_history_for_player(speaker, num_conv_history_reasoning)), model, print_output, server_based)
        if action_state is not str and action_state.get("action") is None and action_state.get("error") is None:
            self.features.update_features_from_json(speaker, action_state)
            conv_manager.store_prompt_outcome(prompt_state, json.dumps(action_state))
        return error

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

    def get_conversation_for_player(self, player_name, num_convs = 1) -> Conversation:
        result = []
        for participants_tuple, conv in self.conversations.items():
            if player_name in participants_tuple:
                result.append(conv.history[-num_convs:])
        return result  # Return only the last `num_convs` conversations

    def get_conversation_history_for_player(self, current_player, num_convs = 1) -> str:
        """
        Retrieves and returns a formatted conversation history for the given player.

        :param current_player: The name of the player for whom to retrieve the conversation history.
        :param num_convs: The number of recent conversations to retrieve (default: 3).
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

        return "\n".join(formatted_conversations[-num_convs:])

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
        prompt_outcome_log = list(dict.fromkeys(self.prompt_outcome_log))

        for i, o in prompt_outcome_log:
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

# ------------------ LLM Action Completion Function ------------------

def extract_json_start(text):
    match = re.match(r'^\s*(\{.*?\})', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return json.dumps(data)
        except json.JSONDecodeError:
            return text
    return None

def complete_action_with_llm(current_player, prompt, model, print_output, server_based):
    """
    Given the global actions (action templates) for the number guessing game,
    ask the LLM to complete an action (fill in missing parameters) and return a JSON object.
    The JSON should have two keys: "conversation" (a list of conversation messages)
    and "action" (the selected complete action as a string).
    
    :param current_player: Current player name (should be 'A').
    :param prompt: generated prompt
    :return: The LLM's JSON response as a dictionary.
    """ 
    
    if print_input:
        print(prompt)

    errors = 0
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
    llm_response = llm_response.strip()
    llm_response = extract_json_start(llm_response)

    try:
        result = json.loads(llm_response)
    except Exception as e:
        errors = errors + 1
        if print_output:
            print('- ___ - ___ -')
            print("Error parsing LLM response: " + ' \nMessage:\n' + str(llm_response), e)
            print('- ___ - ___ -')
        result = {"error": "No Action", 'Speaker': current_player}

    if print_output:
        print("LLM Responds: " + current_player +"\n", result)
        print('--- --- --- ---')
    return prompt, result, errors

# ------------------ MCTS with LLM Integration ------------------

def simulation_policy(node, models, print_output, server_based, num_child_node):
    """
    Uses ActionProcessor to simulate actions and generate the next game state and conversation state.
    
    :param game_state: The current game state.
    :param conversation_manager: The current conversation manager.
    :return: A new (game_state, conversation_manager) pair.
    """
    game_state = node.state
    game_state.update_game_state()
    conversation_manager = node.conversation_manager
    player = game_state.get_player()
    terminal_state = False
    result_action = []
    prompt = ''
    previous_results = None
    llm_erros = 0
    
    if len(models) > 1:
        if game_state.active_players[player].alignment == 'Good':
            model = models[0]
        else:
            model = models[1]
    else:
        model = models[0]
    
    num_max_nodes = num_child_node # int(max(1, (random.random() * num_child_node + 1)))
    for i in range(num_max_nodes):
        if not player in game_state.players:
            continue
        model.set_temperature(0.8) # (max(0.8, 1.2 - i * 0.1))
        prompt, result, erros = game_state.create_action(player, conversation_manager, model, print_output, server_based, previous_results)
        llm_erros = llm_erros + erros
        if result not in result_action:
            result_action.append(result)
            if previous_results is None:
                previous_results = str(result) + "\n"
            else: 
                previous_results = previous_results + str(result) + "\n"

    # Process each action in result_action
    child_nodes = []
    for action in result_action:
        # Create deep copies to avoid modifying the original objects
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
        success, errors = game_state_copy.apply_action(conversation_manager_copy, action, model, print_output, server_based)
        llm_erros = llm_erros + errors
        if not success:
            continue
        
        if game_state_copy.is_terminal() is not None and game_state_copy.is_terminal() is True:
            terminal_state = True

        child_node = MCTSNode(game_state_copy, action, conversation_manager_copy, terminal_state, parent=node)
        child_nodes.append(child_node)  # Collect nodes in a list

    return child_nodes, llm_erros  # Always return a list of nodes

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

    def best_child(self, exploration_constant=1.41):
        if not self.children:
            return None  # No children to choose from

        if self.parent is None:
            # No parent → pure exploitation (greedy)
            return max(self.children, key=lambda child: child.value / (child.visits + 1e-6))

        return max(
            self.children,
            key=lambda child: (
                (child.value / (child.visits + 1e-6)) +
                exploration_constant * ((math.log(self.parent.visits + 1) / (child.visits + 1e-6)) ** 0.5)
            )
        )

    def best_leaf(self, exploration_weight=1.41):
        if not self.children:
            return self  # If this is a leaf node, return itself

        # If there are children, select the best one using UCT
        best_node = self.best_child(exploration_weight)
    
        return best_node.best_leaf(exploration_weight)  # Recursively go down until a leaf node is found

    def __str__(self):
        # String representation for easy printing of the node's details
        return f"State: {self.state}, Value: {self.value}, Visits: {self.visits}, Terminal: {self.is_terminal}"


class MCTS:
    def __init__(self, simulation_policy, reward_function, num_child_node, iterations=100, exploration_weight=1.41):
        self.simulation_policy = simulation_policy
        self.reward_function = reward_function
        self.iterations = iterations
        self.root = None
        self.num_child_node = num_child_node
        self.exploration_weight = exploration_weight
        self.errors = 0

    def search(self, initial_state, conversation_manager, model, print_output, server_based):
        self.root = self.get_node(initial_state, conversation_manager, initial_state.get_no_action())

        for index in range(self.iterations):
            node = self.select(self.root)
            new_nodes = self.expand(node, model, print_output, server_based)

            if not new_nodes or not isinstance(new_nodes, list):  # Check if the list is empty or None
                continue

            for new_node in new_nodes:
                if new_node is None or new_node.action.get("type") == new_node.state.get_no_action().get("type"):
                    continue
                reward = self.reward_function(node, new_node)
                self.backpropagate(new_node, reward)

            print('*** *** *** *** *** *** *** ' + str(index) + ' / ' + str(self.iterations))
        return self.root.best_leaf()

    def select(self, node):
        """Balanced exploration (new nodes) and exploitation (best child)"""

        if not node.is_terminal:
            if not node.children:
                return node  # No children, return itself
            
            # Otherwise, use UCT to select the best-explored child
            node = node.best_leaf(exploration_weight=self.exploration_weight)

        return node  # Return terminal node

    def get_root_node(self):
        return self.root

    def get_all_terminal_nodes(self, node):
        """Return a list of all terminal nodes reachable from the given node."""
        terminal_nodes = []
        
        if node.is_terminal:
            terminal_nodes.append(node)
        else:
            for child in node.children:
                terminal_nodes.extend(self.get_all_terminal_nodes(child))

        return terminal_nodes

    def expand(self, node, model, print_output, server_based):
        if node is None:
            return None
        if node.is_terminal:  # Stop expansion if game is over
            return node.parent if node.parent else node  # Move up if terminal  

        child_nodes, errors = self.simulation_policy(node, model, print_output, server_based, self.num_child_node)
        self.errors = self.errors + errors
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
            terminal_status = " (Terminal) " + node.state.terminal_message() if node.is_terminal else ""
            print(f"{indent}- Visits: {node.visits}, Value: {node.value:.2f}{terminal_status}")

        for child in node.children:
            self.print_tree(child, depth + 1)
            
    def get_root(self):
        return self.root
