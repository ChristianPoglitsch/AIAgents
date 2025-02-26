import math
import random
import openai
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn

from LLM_Character.llm_comms.llm_api import LLM_API
from LLM_Character.llm_comms.llm_openai import OpenAIComms
from LLM_Character.messages_dataclass import AIMessage, AIMessages

class Role:
    def __init__(self, name, team, ability):
        self.name = name
        self.team = team
        self.ability = ability

DEVELOPER = 'developer'
ASSISTENT = 'assistant'
USER = 'user'

roles = {
    # Townsfolk Roles
    'Washerwoman': Role('Washerwoman', 'Town', 'Learns that one of two players is a specific Townsfolk.'),
    'Librarian': Role('Librarian', 'Town', 'Learns that one of two players is a specific Outsider or that no Outsiders are in play.'),
    'Investigator': Role('Investigator', 'Town', 'Learns that one of two players is a specific Minion.'),
    'Chef': Role('Chef', 'Town', 'Learns how many pairs of evil players are sitting next to each other.'),
    'Empath': Role('Empath', 'Town', 'Learns how many of their two alive neighbors are evil.'),
    'Fortune Teller': Role('Fortune Teller', 'Town', 'Each night, chooses two players. Learns if one is the Demon, but there is a Red Herring.'),
    'Undertaker': Role('Undertaker', 'Town', 'Each night, learns which character died by execution that day.'),
    'Monk': Role('Monk', 'Town', 'Each night, chooses a player (not themselves). That player is safe from the Demon that night.'),
    'Ravenkeeper': Role('Ravenkeeper', 'Town', 'If killed at night, learns one players character.'),
    'Virgin': Role('Virgin', 'Town', 'If nominated for the first time, and the nominator is a Townsfolk, they are executed immediately.'),
    'Slayer': Role('Slayer', 'Town', 'Once per game, publicly chooses a player. If that player is the Demon, they die.'),
    'Soldier': Role('Soldier', 'Town', 'Cannot die at night.'),
    'Mayor': Role('Mayor', 'Town', 'If only three players live & no execution occurs, your team wins. Might not die at night.'),
    
    # Outsider Roles
    'Butler': Role('Butler', 'Outsider', 'Each night, chooses a player. Can only vote if that player votes first.'),
    'Drunk': Role('Drunk', 'Outsider', 'You do not know you are the Drunk. You think you are a Townsfolk, but you are not.'),
    'Recluse': Role('Recluse', 'Outsider', 'Might register as evil & as a Minion or Demon, even if dead.'),
    'Saint': Role('Saint', 'Outsider', 'If executed, your team loses.'),
    
    # Minion Roles
    'Poisoner': Role('Poisoner', 'Minion', 'Each night, poisons one player.'),
    'Baron': Role('Baron', 'Minion', 'Two extra Outsiders are in play.'),
    'Scarlet Woman': Role('Scarlet Woman', 'Minion', 'If there are five or more players alive and the Demon dies, you become the Demon.'),
    'Spy': Role('Spy', 'Minion', 'Might register as good. Sees the Grimoire each night.'),
    
    # Demon Roles
    'Imp': Role('Imp', 'Demon', 'Each night, chooses a player to die. If you kill yourself this way, a Minion becomes the Imp.')
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

class Conversation:
    def __init__(self, participants):
        """
        Initializes a conversation with a list of participating players.
        :param participants: List of player names involved in the conversation.
        """
        self.participants = participants  # List of players in the conversation
        self.history = []  # List to store the conversation history

    def add_message(self, sender, message):
        """
        Adds a message to the conversation history.
        :param sender: The player sending the message.
        :param message: The content of the message.
        """
        if sender in self.participants:
            self.history.append({'sender': sender, 'message': message})
        else:
            raise ValueError(f"{sender} is not a participant in this conversation.")

    def print_conversation(self):
        """
        Prints the entire conversation history.
        """
        print("Conversation History:")
        for entry in self.history:
            print(f"{entry['sender']}: {entry['message']}")

    def get_history_text(self):
        """
        Returns the conversation history as a formatted string.
        """
        history_text = ""
        for entry in self.history:
            history_text += f"{entry['sender']}: {entry['message']}\n"
        return history_text

    def get_participants(self):
        """
        Returns the list of participants in the conversation.
        """
        return self.participants
    

class ConversationManager:
    def __init__(self):
        """
        Initializes the ConversationManager to keep track of multiple conversations.
        """
        self.conversations = {}  # Dictionary to hold conversations with player tuples as keys

    def start_conversation(self, participants):
        """
        Starts a new conversation with a list of participants.
        :param participants: List of player names involved in the conversation.
        """
        participants_tuple = tuple(sorted(participants))  # Sort participants to handle unordered tuples
        if participants_tuple not in self.conversations:
            conversation = Conversation(participants)
            self.conversations[participants_tuple] = conversation
            print(f"Started a new conversation with participants: {participants}")
        else:
            print(f"Conversation with participants {participants} already exists.")

    def add_message_to_conversation(self, participants, sender, message):
        """
        Adds a message to an existing conversation.
        :param participants: List of player names involved in the conversation.
        :param sender: The player sending the message.
        :param message: The content of the message.
        """
        participants_tuple = tuple(sorted(participants))  # Sort participants for consistency
        if participants_tuple not in self.conversations:
            self.start_conversation(participants)
        conversation = self.conversations[participants_tuple]
        conversation.add_message(sender, message)

    def print_conversation(self, participants):
        """
        Prints the conversation history of the specified participants.
        :param participants: List of player names involved in the conversation.
        """
        participants_tuple = tuple(sorted(participants))  # Sort participants for consistency
        if participants_tuple in self.conversations:
            conversation = self.conversations[participants_tuple]
            conversation.print_conversation()
        else:
            print(f"No conversation found with participants: {participants}")

    def get_conversation_for_player(self, player_name):
        """
        Retrieve all conversations that involve the specified player.
        :param player_name: The name of the player for whom we want to get the conversation history.
        :return: A list of Conversation objects.
        """
        conversations_for_player = []
        for participants_tuple, conversation in self.conversations.items():
            if player_name in participants_tuple:
                conversations_for_player.append(conversation)
        return conversations_for_player

    def get_conversations(self):
        """
        Returns all current conversations.
        """
        return self.conversations

# ------------------ GameState and ActionSpace ------------------

# Stub definitions for GameState methods:
# (In a complete implementation, GameState would include these methods.)
class GameState:
    def __init__(self, players, phase='Day'):
        self.phase = phase
        self.alive_players = {player: {'role': None, 'alive': True, 'nominated': False, 'effects': None} for player in players}
        self.nominations = []
        self.roles = {player: None for player in players}
        self.private_knowledge = {player: None for player in players}
        self.day_count = 1
        self.effects = {}
        self.execution = None
        # For GNN conversion, we assume some player features and an edge_index:
        self.player_features = torch.eye(len(players))  # Dummy features: Identity matrix.
        self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # Dummy connectivity.

    def get_public_state(self):
        return {
            'phase': self.phase,
            'alive_players': self.alive_players,
            'nominations': self.nominations,
            'day_count': self.day_count,
        }

    def get_legal_actions(self):
        # For demonstration, assume each alive player can nominate any other alive player.
        legal_actions = []
        if self.phase == 'Day':
            for nominator in self.alive_players:
                if self.alive_players[nominator]['alive']:
                    for nominee in self.alive_players:
                        if nominator != nominee and self.alive_players[nominee]['alive']:
                            legal_actions.append({'type': 'nominate', 'nominator': nominator, 'nominee': nominee})
        # For Night, assume Imp can kill.
        elif self.phase == 'Night':
            for player in self.alive_players:
                if self.roles.get(player) == 'Imp' and self.alive_players[player]['alive']:
                    for target in self.alive_players:
                        if player != target and self.alive_players[target]['alive']:
                            legal_actions.append({'type': 'kill', 'killer': player, 'target': target})
        return legal_actions

    def take_action(self, action):
        # This stub returns a new state with the action applied.
        new_state = deepcopy(self)
        if action['type'] == 'nominate':
            new_state.nominations.append((action['nominator'], action['nominee']))
        elif action['type'] == 'kill':
            new_state.alive_players[action['target']]['alive'] = False
        # Switch phase if needed.
        if len(new_state.nominations) >= len(new_state.alive_players) - 1:
            new_state.phase = 'Night' if self.phase == 'Day' else 'Day'
            new_state.nominations.clear()
        return new_state

    def is_terminal(self):
        # For demonstration, assume game ends if less than 2 players are alive.
        alive_count = sum(1 for p in self.alive_players if self.alive_players[p]['alive'])
        return alive_count < 2

    def get_reward(self):
        # Return a dummy reward (e.g., 1 or -1) for terminal states.
        return random.choice([1, -1])

class ActionSpace:
    def __init__(self):
        self.actions = [
            {'type': 'nominate', 'nominator': None, 'nominee': None},
            {'type': 'vote', 'voter': None, 'vote': None},
            {'type': 'kill', 'killer': None, 'target': None},
            {'type': 'protect', 'protector': None, 'target': None},
            {'type': 'bluff', 'bluffer': None, 'claim': None},
            {'type': 'reveal', 'revealer': None},
            {'type': 'disrupt', 'disruptor': None},
            {'type': 'check', 'checker': None, 'target': None},
            {'type': 'announce', 'announcer': None, 'message': None},
            {'type': 'vote_grimoire', 'voter': None, 'vote': None}
        ]
    
    def get_action_template(self, action_type):
        for action in self.actions:
            if action['type'] == action_type:
                return action.copy()
        return None

    def list_action_types(self):
        return [action['type'] for action in self.actions]
    
    def get_formatted_action_space(self):
        formatted = "Action Space:\n"
        for action in self.actions:
            formatted += f"- {action}\n"
        return formatted

def get_game_state_description(game_state):
    public_state = game_state.get_public_state()
    description = f"Phase: {public_state['phase']}\n"
    description += f"Day Count: {public_state['day_count']}\n"
    description += "Alive Players:\n"
    for player, info in public_state['alive_players'].items():
        status = "Alive" if info['alive'] else "Dead"
        description += f"  - {player}: Status={status}\n"
    description += "Nominations:\n"
    for nominator, nominee in public_state['nominations']:
        description += f"  - {nominator} nominated {nominee}\n"
    return description

def get_conversation_history_for_player(conversation_manager, player_name):
    """
    Returns the conversation history (as a formatted string) for the specified player.
    """
    conversations = conversation_manager.get_conversation_for_player(player_name)
    history_text = ""
    for conv in conversations:
        history_text += conv.get_history_text() + "\n"
    return history_text


# ------------------ large Language Model ------------------

def complete_action_struct_with_llm(game_state, action_space, current_player, conversation_manager, roles, global_action):
    """
    Given the global_actions list, ask the LLM to select one action template from the list 
    and complete it by filling in all missing parameters. The LLM is provided with the current 
    game state, conversation history, available action descriptions, and the roles in the game.
    
    The LLM should output one complete action (in JSON format) chosen from the provided global actions.
    
    :param game_state: Current game state object.
    :param action_space: Object providing a formatted description of available actions.
    :param current_player: Name of the current player.
    :param conversation_manager: ConversationManager instance with conversation history.
    :param roles: Dictionary mapping role names to Role objects.
    :param global_actions: List of action templates (dictionaries with some parameters as None).
    :return: The LLM's response as a completed action structure in JSON.
    """
    # Generate a human-readable description of the current game state.
    state_description = get_game_state_description(game_state)
    # Get a formatted description of the available action space.
    action_space_description = action_space.get_formatted_action_space()
    # Retrieve the conversation history related to the current player.
    conversation_history = get_conversation_history_for_player(conversation_manager, current_player)
    
    # Get the current player's role, or "Unknown" if not set.
    current_role = game_state.roles[current_player] if game_state.roles[current_player] else "Unknown"
    
    # Compose the prompt for the LLM.
    prompt = (
        "You are a helpful board game AI assistant for Blood on the Clocktower. "
        "Below is a list of available global action templates. Each template is an incomplete action structure, "
        "with some parameters missing (marked as None). Based on the game state, conversation history, and role information, "
        "please select one action from the list and complete it by filling in all missing parameters. "
        "Return your answer as a single complete action in JSON format.\n\n"
        f"Selected Action:\n{global_action}\n\n"
        f"Current Player: {current_player}\n"
        f"Role: {current_role}\n\n"
        "Available Actions Description:\n"
        f"{action_space_description}\n\n"
        "Roles in the Game:\n"
        f"{roles_to_string(roles)}\n\n"
        "Game State:\n"
        f"{state_description}\n\n"
        "Conversation History:\n"
        f"{conversation_history}\n\n"
        "Please output one complete action in JSON format, selected from the global actions list."
    )
    
    # Debug print of the prompt (optional)
    print(prompt)
    
    # Prepare the messages for the LLM (assuming AIMessages, AIMessage, DEVELOPER, etc. are defined in your system).
    messages = AIMessages()
    message = AIMessage(message=prompt, role=DEVELOPER, class_type="ActionCompletion", sender=DEVELOPER)
    messages.add_message(message)
    
    # Initialize the LLM model (this function should be defined in your integration).
    model = init_model()
    
    # Query the LLM with the messages.
    response = model.query_text(messages)
    return response


# ------------------ Reinforcement Learning ------------------

class ActionPredictionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        A simple GNN that outputs log probabilities over action types.
        :param input_dim: Dimensionality of input node features.
        :param hidden_dim: Size of hidden layers.
        :param output_dim: Number of possible action types.
        """
        super(ActionPredictionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        """
        Forward pass through the GNN.
        :param data: A PyTorch Geometric Data object.
        :return: Log probabilities (via log_softmax) over action types.
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Aggregate node features (here, we simply average them) to get a graph-level representation.
        x = torch.mean(x, dim=0)
        x = self.fc(x)
        return F.log_softmax(x, dim=0)

# ------------------ Stub for Converting GameState to Graph Data ------------------

def game_state_to_graph_data(game_state):
    """
    Converts the (determinized) GameState into a PyTorch Geometric Data object.
    For this example, we assume game_state has attributes:
      - game_state.player_features: tensor of shape (num_players, feature_dim)
      - game_state.edge_index: tensor of shape [2, num_edges]
    In practice, you should implement this to encode your game state's structure.
    
    :param game_state: A determinized GameState instance.
    :return: A PyG Data object.
    """
    return Data(x=game_state.player_features, edge_index=game_state.edge_index)

# ------------------ ISMCTS Node Definition ------------------

class ISMCTSNode:
    def __init__(self, public_state, parent=None):
        self.public_state = public_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0
        self.action_taken = None

    def is_fully_expanded(self, legal_actions):
        return len(self.children) == len(legal_actions)

    def best_child(self, exploration_weight=1.0):
        return max(
            self.children,
            key=lambda child: child.total_reward / child.visits +
                              exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
        )

# ------------------ Determinization ------------------

def determinize_state(game_state, current_player):
    """
    Creates a determinized version of the game state by randomly assigning unknown private information.
    :param game_state: The full game state.
    :param current_player: The name of the current player.
    :return: A determinized GameState instance.
    """
    determinized_state = deepcopy(game_state)
    possible_roles = ['Imp', 'Fortune Teller', 'Villager']
    for player in determinized_state.alive_players:
        if determinized_state.roles[player] is None:
            determinized_state.roles[player] = random.choice(possible_roles)
            determinized_state.alive_players[player]['role'] = determinized_state.roles[player]
    return determinized_state

# ------------------ Global Action List ------------------

global_actions = ['nominate', 'vote', 'kill', 'protect', 'bluff',
                  'reveal', 'disrupt', 'check', 'announce', 'vote_grimoire']

# ------------------ ISMCTS with GNN Integration ------------------

def ismcts_with_gnn(game_state, current_player, num_simulations, conversation_manager, player_to_idx, embedding_dim, gnn_model):
    """
    Runs ISMCTS integrating the ActionPredictionGNN for predictive action selection.
    :param game_state: The current full GameState.
    :param current_player: The current player's name.
    :param num_simulations: Number of ISMCTS simulations.
    :param conversation_manager: ConversationManager instance.
    :param player_to_idx: Mapping from player names to indices.
    :param embedding_dim: Dimension for conversation embeddings.
    :param gnn_model: A pre-trained ActionPredictionGNN model.
    :return: The action recommended by ISMCTS.
    """
    public_state = game_state.get_public_state()
    root = ISMCTSNode(public_state)
    
    for _ in range(num_simulations):
        determinized_state = determinize_state(game_state, current_player)
        node = root
        legal_actions = determinized_state.get_legal_actions()
        
        # --- SELECTION ---
        while not determinized_state.is_terminal() and node.is_fully_expanded(legal_actions):
            node = node.best_child()
            determinized_state = determinized_state.take_action(node.action_taken)
            legal_actions = determinized_state.get_legal_actions()
        
        # --- EXPANSION ---
        if not determinized_state.is_terminal():
            legal_actions = determinized_state.get_legal_actions()
            untried_actions = [action for action in legal_actions if action not in [child.action_taken for child in node.children]]
            if untried_actions:
                action = random.choice(untried_actions)
                new_state = determinized_state.take_action(action)
                child_node = ISMCTSNode(new_state.get_public_state(), parent=node)
                child_node.action_taken = action
                node.children.append(child_node)
                node = child_node
                determinized_state = new_state
        
        # --- SIMULATION (Rollout) using the GNN for predictive modeling ---
        while not determinized_state.is_terminal():
            legal_actions = determinized_state.get_legal_actions()
            if not legal_actions:
                break
            
            # Convert the determinized state to graph data.
            graph_data = build_graph(determinized_state, conversation_manager, player_to_idx, embedding_dim)
            # Use the GNN to predict action log probabilities.
            log_probs = gnn_model(graph_data)  # Output shape: (output_dim,)
            probs = torch.exp(log_probs)
            
            # Filter the predictions to legal actions.
            legal_indices = []
            for action in legal_actions:
                try:
                    idx = global_actions.index(action['type'])
                    legal_indices.append(idx)
                except ValueError:
                    pass
            
            if not legal_indices:
                chosen_action = random.choice(legal_actions)
            else:
                legal_probs = probs[legal_indices]
                legal_probs = legal_probs / legal_probs.sum()  # Normalize.
                sampled_idx = torch.multinomial(legal_probs, 1).item()
                chosen_global_idx = legal_indices[sampled_idx]
                # Choose the first legal action that matches the predicted action type.
                chosen_action = next(a for a in legal_actions if a['type'] == global_actions[chosen_global_idx])
            
            determinized_state = determinized_state.take_action(chosen_action)
        
        # --- BACKPROPAGATION ---
        reward = determinized_state.get_reward()
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    best_child = root.best_child(exploration_weight=0)
    return best_child.action_taken


# Example function: converts game state into player node features.
# For this example, we assume player features are already computed.
def gamestate_to_gnn_features(game_state):
    """
    Converts a GameState instance into a tensor of features for each player node.
    In this simplified example, we assume each player node feature is provided.
    
    :param game_state: An instance of GameState.
    :return: A tensor of shape (num_players, feature_dim)
    """
    # Assume game_state has an attribute 'player_features' already computed.
    return game_state.player_features


# Example function: converts conversation data into conversation node features
# and edges connecting conversation nodes to player nodes.
def conversations_to_features_and_edges(conversation_manager, player_to_idx, embedding_dim=128):
    """
    Converts all conversations from a ConversationManager into:
      - conversation_features: A tensor of shape (num_conversation_nodes, embedding_dim)
      - conv_edge_index: A tensor of shape [2, num_edges] connecting conversation nodes to player nodes.
    :param conversation_manager: ConversationManager instance.
    :param player_to_idx: Dictionary mapping player names to node indices.
    :param embedding_dim: Dimensionality of conversation embedding.
    :return: (conversation_features, conv_edge_index)
    """
    conversation_features_list = []
    conv_edges = []
    # Conversation nodes start after the player nodes.
    conversation_start_idx = len(player_to_idx)
    
    for conv_idx, (participants_tuple, conv_obj) in enumerate(conversation_manager.get_conversations().items()):
        conv_text = conv_obj.get_history_text()
        # Dummy encoding: in practice, use a text encoder (e.g., SentenceTransformer).
        conv_embedding = torch.full((embedding_dim,), 0.5)  
        conversation_features_list.append(conv_embedding)
        conv_node_idx = conversation_start_idx + conv_idx
        for player in conv_obj.get_participants():
            if player in player_to_idx:
                player_idx = player_to_idx[player]
                conv_edges.append((conv_node_idx, player_idx))
                conv_edges.append((player_idx, conv_node_idx))
    
    if conversation_features_list:
        conversation_features = torch.stack(conversation_features_list, dim=0)
    else:
        conversation_features = torch.empty((0, embedding_dim))
    
    if conv_edges:
        conv_edge_index = torch.tensor(conv_edges, dtype=torch.long).t().contiguous()
    else:
        conv_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return conversation_features, conv_edge_index

# Function to build a complete graph that includes player nodes and conversation nodes.
def build_graph(game_state, conversation_manager, player_to_idx, embedding_dim=128):
    """
    Builds a PyTorch Geometric Data object by combining player features and conversation features.
    :param game_state: A determinized GameState instance.
    :param conversation_manager: ConversationManager containing conversation data.
    :param player_to_idx: Dictionary mapping player names to indices (player nodes come first).
    :param embedding_dim: Embedding dimension for conversation nodes.
    :return: A PyG Data object representing the full graph.
    """
    # Retrieve player features (assumed to be stored in game_state.player_features).
    player_node_features = game_state.player_features  # shape: (num_players, player_feature_dim)
    
    # Convert conversation data to features and edges.
    conversation_features, conv_edge_index = conversations_to_features_and_edges(conversation_manager, player_to_idx, embedding_dim)
    
    # Ensure player features and conversation features have the same dimension.
    if conversation_features.numel() > 0 and player_node_features.size(1) != conversation_features.size(1):
        projection = nn.Linear(player_node_features.size(1), conversation_features.size(1))
        player_node_features = projection(player_node_features)
    
    # Combine player and conversation features.
    if conversation_features.numel() > 0:
        x = torch.cat([player_node_features, conversation_features], dim=0)
    else:
        x = player_node_features

    # Create dummy player-player edges (for example, a fully connected graph among players).
    num_players = player_node_features.size(0)
    player_edge_list = [(i, j) for i in range(len(player_to_idx)) for j in range(len(player_to_idx)) if i != j]
    if player_edge_list:
        player_edge_index = torch.tensor(player_edge_list, dtype=torch.long).t().contiguous()
    else:
        player_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # Combine player-player edges with conversation edges.
    # Note: conv_edge_index already uses conversation node indices offset by len(player_to_idx).
    if conv_edge_index.numel() > 0:
        edge_index = torch.cat([player_edge_index, conv_edge_index], dim=1)
    else:
        edge_index = player_edge_index

    return Data(x=x, edge_index=edge_index)



def init_model() -> LLM_API:
    model = OpenAIComms()
    model_id = "gpt-4o"
    model.max_tokens = 128
    model.init(model_id)
    wrapped_model = LLM_API(model)
    return wrapped_model


conversation_manager = ConversationManager()
conversation_manager.start_conversation(['Alice', 'Bob'])
conversation_manager.add_message_to_conversation(['Alice', 'Bob'], 'Alice', "I think we should target Charlie.")
conversation_manager.add_message_to_conversation(['Alice', 'Bob'], 'Bob', "Maybe, but I'm not sure if he's really suspicious.")

# Define players and create a GameState instance
players = ['Alice', 'Bob', 'Charlie']

# Define a player_to_idx mapping.
player_to_idx = {'Alice': 0, 'Bob': 1, 'Charlie': 2}

game_state = GameState(players, phase='Day')
# Set roles (using simple strings for this example)
game_state.alive_players['Alice']['role'] = 'Imp'
game_state.alive_players['Bob']['role'] = 'Fortune Teller'
game_state.alive_players['Charlie']['role'] = 'Villager'
game_state.roles['Alice'] = 'Imp'
game_state.roles['Bob'] = 'Fortune Teller'
game_state.roles['Charlie'] = 'Villager'
game_state.nominations.append(('Alice', 'Charlie'))


conversations = conversation_manager.get_conversations()
conversation_manager.get_conversation_for_player('Alice')
print('---')

# Loop through all conversations and print the details
for participants, conversation in conversations.items():
    print(f"Conversation with participants: {participants}")
    conversation.print_conversation()  # Prints the conversation history for these participants
    print()  # Add a blank line for readability between conversations
    

# Initialize the action space
action_space = ActionSpace()

# Define the current player for whom to choose an action
current_player = 'Alice'

# Ask ChatGPT for a next possible action for the current player
if True:
    next_action_suggestion = complete_action_struct_with_llm(game_state, action_space, current_player, conversation_manager, roles, global_actions[4])
    print("ChatGPT suggests the following possible action(s):")
    print(next_action_suggestion)
 

# Build the complete graph that combines player and conversation nodes.
graph_data = build_graph(game_state, conversation_manager, player_to_idx, embedding_dim=128)

print("Final graph node feature tensor shape:", graph_data.x.shape)
print("Final graph edge_index shape:", graph_data.edge_index.shape)


# Create a dummy GNN model.
# Suppose we have already built our graph data from game state and conversations.
graph_data = build_graph(game_state, conversation_manager, player_to_idx, embedding_dim=128)
print("Graph node features shape:", graph_data.x.shape)  # Expected shape: (num_nodes, 128)

# Correctly set the GNN input dimension to 128, which is the feature dimension of graph_data.x.
gnn_model = ActionPredictionGNN(input_dim=128, hidden_dim=16, output_dim=len(global_actions))

# Now run the model.
output = gnn_model(graph_data)
print("GNN output:", output)


optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()

# Run ISMCTS with GNN integration.
#best_action = ismcts_with_gnn(game_state, current_player='Alice', num_simulations=1, 
#                              conversation_manager=conversation_manager, player_to_idx=player_to_idx, 
#                              embedding_dim=128, gnn_model=gnn_model)
#print("Recommended action for Alice:", best_action)
