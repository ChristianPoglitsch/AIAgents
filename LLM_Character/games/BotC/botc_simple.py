import json
import random
import torch

import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_local import LocalComms
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages

# ------------------ Game Environment for Number Guessing ------------------

class NumberGuessGameState:
    def __init__(self):
        # Players: A is the seeker; B, C, and D are respondents.
        self.players = ['A', 'B', 'C', 'D']
        # Secret number is either 0 or 1.
        self.secret_number = random.choice([0, 1])
        # Randomly select one respondent to be the liar.
        self.liar = random.choice(['B', 'C', 'D'])
        # Initialize features for each player.
        # Feature vector: [is_A, asked, response]
        # For Player A: is_A=1, asked=0, response=-1 (not applicable)
        # For others: is_A=0, asked=0, response=-1 (no answer yet)
        self.features = {
            'A': [1, 0, -1],
            'B': [0, 0, -1],
            'C': [0, 0, -1],
            'D': [0, 0, -1]
        }

    def is_terminal(self):
        """Game ends when a guess is made."""
        return self.guess is not None

    def get_public_state(self, player):
        """Return a human-readable public state description with role-based visibility."""
    
        if player == "A":
            secret_info = "Secret number is unknown to you."
        else:
            secret_info = f"Secret number is {self.secret_number}."
            if player == self.liar:
                secret_info += " (You are the liar!)"

        # List of all players
        player_info = "Players: " + ", ".join(self.players)

        message = f"{secret_info}\n\n{player_info}"
    
        return message

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

    def start_conversation(self, participants):
        participants_tuple = tuple(sorted(participants))
        if participants_tuple not in self.conversations:
            conv = Conversation(participants)
            self.conversations[participants_tuple] = conv

    def add_message_to_conversation(self, participants, sender, message):
        #print('sender: ' + str(sender) + ' participants: ' + str(sorted(participants)) + ' message: ' + str(message))
        participants_tuple = tuple(sorted(participants))
        if participants_tuple not in self.conversations:
            self.start_conversation(participants)
        self.conversations[participants_tuple].add_message(sender, message)

    def add_action_to_conversation(self, action):
        if not isinstance(action, list):
                action = [action]        
        for act in action:
            speaker = act.get("Speaker")
            # Get the "Audience" from the action.
            audience = act.get("Audience")
            # If audience is not a list, wrap it in a list.
            if not isinstance(audience, list):
                audience = [audience]
            # Merge speaker and audience into one participants list.
            participants = [speaker] + audience
    
            if act.get("type") in ["Message"]:
                self.add_message_to_conversation(participants, speaker, act)

    def get_conversation_for_player(self, player_name):
        result = []
        for participants_tuple, conv in self.conversations.items():
            if player_name in participants_tuple:
                result.append(conv)
        return result

    def print_all_conversations(conversation_manager):
        """
        Prints all conversations stored in the ConversationManager.
        For each conversation, it prints the list of participants and then each message.
        """
        for participants, conversation in conversation_manager.conversations.items():
            print(f"Conversation among: {', '.join(participants)}")
            for entry in conversation.history:
                print(f"{entry['sender']}: {entry['message']}")
            print("-" * 40)

# ------------------ Actions classes ------------------

def apply_action(game_state, conversation_manager, action):
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
            participants = [speaker, respondent]
        
            # Call the LLM to complete the action for this respondent.
            result_action = complete_action_with_llm(game_state, respondent, conversation_manager)

            # Check if temp_result is a list; if so, iterate over its items.
            if isinstance(result_action, list):
                for item in result_action:
                    if item.get("type") in ["Message"]:
                        speaker = item.get("Speaker")
                        audience = item.get("Audience")
                        participants = [speaker, audience]
                        conversation_manager.add_message_to_conversation(participants, item.get("Speaker"), item)
            else:
                if result_action.get("type") in ["Message"]:
                    speaker = result_action.get("Speaker")
                    audience = result_action.get("Audience")
                    participants = [speaker, audience]


    elif action_type == "Guess":
        guessed_number = action.get("Number")
        speaker = action.get("Speaker")
        if guessed_number in [0, 1]:
            # Update game state with the guessed number.
            game_state.guess = guessed_number
        
            # Log the guess in the conversation manager.
            conversation_manager.add_message_to_conversation(speaker, speaker, f"My guess is {guessed_number}.")
        else:
            raise ValueError("Invalid guess. Must be 0 or 1.")

    elif action_type == "No Action":
        #print("No action selected")
        result_action = None

    else:
        #raise ValueError(f"Invalid action type: {action_type}")
        print(f"Invalid action type: {action_type}")
        result_action = None

    return result_action


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
        self.action_budget = 20

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
        while self.action_queue and self.action_budget > 0:
            # Pop the first action from the queue.
            action = self.action_queue.pop(0)
            # Apply the action. This function updates the game state and conversation manager.
            action = apply_action(self.game_state, self.conversation_manager, action)
            self.action_budget = self.action_budget - 1        

            # If a result is returned, add it/them to the queue.
            if action is not None:
                if isinstance(action, list):
                    self.action_queue.extend(action)  # Add all items if result is a list.
                else:
                    self.action_queue.append(action)  # Add the single action.
    
        return

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

# ------------------ Global Actions ------------------

action_space_description = [
    {'type': 'Guess', 'Speaker': None, 'Number': None},
    {'type': 'Message', 'Speaker': None, 'Audience': None, 'Message:': None},
    {'type': 'No Action', 'Speaker': None}
]    

# ------------------ LLM Integration Stub ------------------

def init_model() -> LLM_API:
    model = OpenAIComms()
    model_id = "gpt-4o"
    model.max_tokens = 200
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model

def init_model_local() -> LLM_API:
    model = LocalComms()
    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    model.max_tokens = 200
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model


# ------------------ LLM Action Completion Function ------------------

def complete_action_with_llm(game_state, current_player, conversation_manager):
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
    # For our minigame, we create a simple description.
    state_description = f"Game state: {game_state.get_public_state(current_player)}"
    conversation_history = ""
    convs = conversation_manager.get_conversation_for_player(current_player)
    if convs:
        conversation_history = "\n".join(conv.get_history_text() for conv in convs)
    else:
        conversation_history = "No conversation history."
    
    # Assume Player A has no role in the sense of game roles; use "Seeker"
    #current_role = " "
    
    # Create a prompt.
    prompt = (
        "You are a helpful board game AI assistant for the number guessing minigame. "
        "Player A's goal is to determine the correct number (0 or 1) by asking one of the respondents or by making a guess. Other players have to reply with the correct number, except the liar. This player returns the wrong number."
        "The available global actions are given below, but each action is incomplete and missing parameters. "
        "Based on the game state, conversation history, and your reasoning, please select one action from the list and complete it by filling in all missing parts.\n"
        f"Current Player: {current_player}\n"
        #f"Role: {current_role}\n\n"
        "Available Actions Description:\n"
        f"{action_space_description}\n\n"
        "Game State:\n" + state_description + "\n\n"
        "Conversation History:\n" + conversation_history + "\n\n"
        "Please output one to four complete possible actions from the Available Actions Description list in JSON format. Do NOT use ```json or ``` in your response. Replace None parts of the action."
    )
    
    #print("LLM Prompt:\n", prompt)
    
    # Prepare messages for the LLM.
    messages = AIMessages()
    message = AIMessage(message=prompt, role="developer", class_type="LLMActionCompletion", sender="developer")
    messages.add_message(message)
    
    # Initialize the LLM.
    model = init_model()
    llm_response = model.query_text(messages)
    
    try:
        result = json.loads(llm_response)        
    except Exception as e:
        #print("Error parsing LLM response:", e)
        result = {"action": "No Action", 'Speaker': current_player}

    #print("LLM Responds:\n", result)

    return result

# ------------------ ISMCTS with LLM Integration ------------------

def ismcts_with_llm(game_state, current_player, num_simulations, conversation_manager, global_actions, gnn_model):
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
    :param global_actions: List of global action templates.
    :param gnn_model: The GNN model (not directly used in this LLM stub version).
    :return: The recommended action from the LLM as a string, along with its reasoning.
    """
    # Init prompt
    action = complete_action_with_llm(game_state, current_player, conversation_manager)
    actionProcessor = ActionProcessor(game_state, conversation_manager)
    actionProcessor.add_action(action)
    actionProcessor.process_actions()
    
    return action

# ------------------ Example Usage ------------------

# Create a dummy number guessing game state.
class SimpleNumberGuessGameState(NumberGuessGameState):
    pass  # Inherit from NumberGuessGameState without changes for this example.

game_state = SimpleNumberGuessGameState()

# Create a dummy ConversationManager and add a conversation.
conv_manager = ConversationManager()

# Define a dummy player_to_idx mapping for graph construction (for players A, B, C, D).
player_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

gnn_model = ActionPredictionGNN(input_dim=128, hidden_dim=16, output_dim=2)

# Use the ISMCTS with LLM integration to get an action decision.
decision = ismcts_with_llm(game_state, current_player='A', num_simulations=100, 
                           conversation_manager=conv_manager, global_actions=action_space_description,
                           gnn_model=gnn_model)

print('Secret number: ' + str(game_state.secret_number))
print('Liar: ' + str(game_state.liar))
conv_manager.print_all_conversations()
