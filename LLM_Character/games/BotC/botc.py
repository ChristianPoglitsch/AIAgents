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

class GameState:
    def __init__(self, players, phase='Day'):
        # Public information
        self.phase = phase
        self.alive_players = {player: {'role': None, 'alive': True, 'nominated': False, 'effects': None} for player in players}
        self.nominations = []  # List of tuples (nominator, nominee)
        
        # Private information (not visible to all)
        self.roles = {player: None for player in players}  # E.g., {'Alice': 'Imp', 'Bob': 'Fortune Teller'}
        self.private_knowledge = {player: None for player in players}  # E.g., {'Bob': ('Charlie', False)}
        
        # Extra data for tracking game progress
        self.day_count = 1
        self.effects = {}  # Ongoing effects like poisoning, etc.
        self.execution = None

    def print_game_state(self):
        print("Game State:")
        print(f"Phase: {self.phase}")
        print(f"Day Count: {self.day_count}")
        print("Players:")
        for player, info in self.alive_players.items():
            role = info['role'] if info['role'] else "Unknown"
            status = "Alive" if info['alive'] else "Dead"
            print(f"  {player}: Role={role}, Status={status}, Nominated={info['nominated']}")
        print("Nominations:")
        for nominator, nominee in self.nominations:
            print(f"  {nominator} nominated {nominee}")
        print(f"Execution: {self.execution if self.execution else 'None'}")

    def get_public_state(self):
        return {
            'phase': self.phase,
            'alive_players': self.alive_players,
            'nominations': self.nominations,
            'day_count': self.day_count,
        }

    def get_private_state(self, player):
        return {
            'role': self.roles[player],
            'private_knowledge': self.private_knowledge[player],
        }

    def is_terminal(self):
        alive_roles = {self.roles[p] for p in self.alive_players if self.alive_players[p]['alive']}
        return 'Imp' not in alive_roles or len(alive_roles) <= 1

    def take_action(self, action):
        new_state = GameState(list(self.alive_players.keys()), phase=self.phase)
        new_state.roles = self.roles.copy()
        new_state.private_knowledge = self.private_knowledge.copy()
        new_state.nominations = self.nominations.copy()
        new_state.effects = self.effects.copy()
        new_state.day_count = self.day_count
        new_state.alive_players = self.alive_players.copy()
        new_state.execution = self.execution

        if action['type'] == 'nominate':
            new_state.nominations.append((action['nominator'], action['nominee']))
        elif action['type'] == 'execute':
            new_state.alive_players[action['target']]['alive'] = False
        # Additional action types can be handled here.

        if len(new_state.nominations) >= len(new_state.alive_players) - 1:
            new_state.phase = 'Night' if self.phase == 'Day' else 'Day'
            new_state.nominations.clear()

        return new_state

    def get_reward(self):
        return random.choice([1, -1])

    def get_legal_actions(self):
        legal_actions = []
        if self.phase == 'Day':
            for nominator in self.alive_players:
                if self.alive_players[nominator]['alive']:
                    for nominee in self.alive_players:
                        if nominator != nominee and self.alive_players[nominee]['alive']:
                            legal_actions.append({'type': 'nominate', 'nominator': nominator, 'nominee': nominee})
        elif self.phase == 'Night':
            for player in self.alive_players:
                if self.roles[player] == 'Imp' and self.alive_players[player]['alive']:
                    for target in self.alive_players:
                        if player != target and self.alive_players[target]['alive']:
                            legal_actions.append({'type': 'kill', 'killer': player, 'target': target})
        return legal_actions

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

def ask_chatgpt_for_next_action(game_state, action_space, current_player, conversation_manager, roles):
    """
    Ask ChatGPT what action the current player should take based on the current game state,
    available action space, and the conversation history related to that player.
    """
    state_description = get_game_state_description(game_state)
    action_space_description = action_space.get_formatted_action_space()
    conversation_history = get_conversation_history_for_player(conversation_manager, current_player)
    
    # Retrieve the current player's role (if available)
    current_role = game_state.roles[current_player] if game_state.roles[current_player] else "Unknown"
    
    prompt = (
        "You are a helpful board game AI assistant for Blood on the Clocktower. "
        "Based on the following game state, conversation history, and available actions, "
        "please choose a possible next action for the current player. "
        "Select one of the provided action types and provide any necessary details tailored for the player's role.\n\n"
        f"Current Player: {current_player}\n"
        f"Role: {current_role}\n\n"
        f"{action_space_description}\n"
        f"Roles in the game:\n\n"
        f"{roles_to_string(roles)}\n\n"
        "Game State:\n"
        f"{state_description}\n"
        "Conversation History:\n"
        f"{conversation_history}\n\n"
        "Only reply with one possible action. Fill out the missing parts of the action labeled as None."
    )
    
    print(prompt)
    
    model = init_model()
    
    messages = AIMessages()
    message = AIMessage(message=prompt, role=DEVELOPER, class_type="Introduction", sender=DEVELOPER)
    messages.add_message(message)
    response = model.query_text(messages)
    return response


# ------------------ Reinforcement Learning ------------------

class ISMCTSNode:
    def __init__(self, public_state, parent=None):
        """
        Each node represents an information set (i.e., the public state) 
        as seen from a player's perspective.
        :param public_state: The public part of the game state.
        :param parent: The parent node in the search tree.
        """
        self.public_state = public_state  # Public view of the game state.
        self.parent = parent              # Parent node.
        self.children = []                # List of child nodes.
        self.visits = 0                   # Number of times this node was visited.
        self.total_reward = 0             # Cumulative reward from simulations.
        self.action_taken = None          # The action that led to this node.

    def is_fully_expanded(self, legal_actions):
        """
        Checks if the node is fully expanded. In our case, if the number of children
        equals the number of legal actions in the current public state.
        :param legal_actions: List of legal actions available.
        :return: True if fully expanded, False otherwise.
        """
        return len(self.children) == len(legal_actions)

    def best_child(self, exploration_weight=1.0):
        """
        Selects the child node with the highest UCB1 score.
        :param exploration_weight: Weight for exploration term in UCB1.
        :return: Child node with highest UCB1 score.
        """
        return max(
            self.children,
            key=lambda child: child.total_reward / child.visits +
                              exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
        )

def determinize_state(game_state, current_player):
    """
    Creates a 'determinization' of the game state. That is, given the public state 
    and unknown private information, randomly assign plausible values to the unknown parts.
    :param game_state: The full game state.
    :param current_player: The player for whom the determinization is done.
    :return: A determinized version of the game state.
    """
    # Create a deep copy of the game state so we don't modify the original.
    determinized_state = deepcopy(game_state)
    
    # For demonstration, we assume the unknown roles are randomly assigned.
    possible_roles = ['Imp', 'Fortune Teller', 'Villager']
    for player in determinized_state.alive_players:
        if determinized_state.roles[player] is None:
            determinized_state.roles[player] = random.choice(possible_roles)
            # Also update the public state for this simulation.
            determinized_state.alive_players[player]['role'] = determinized_state.roles[player]
    
    # In a full implementation, you might also sample private knowledge here.
    return determinized_state

def ismcts(game_state, current_player, num_simulations):
    """
    Runs the ISMCTS algorithm:
      - The current player only knows the public state.
      - For each simulation, a determinization is performed to fill in unknown information.
    :param game_state: The current full game state.
    :param current_player: The player for whom we are computing the action.
    :param num_simulations: Number of simulations to run.
    :return: The action recommended by ISMCTS.
    """
    # Obtain the public state (information set) from the game state.
    public_state = game_state.get_public_state()
    root = ISMCTSNode(public_state)
    
    for _ in range(num_simulations):
        # For each simulation, generate a determinized state.
        determinized_state = determinize_state(game_state, current_player)
        
        # Start the simulation at the root node.
        node = root
        
        # Get legal actions based on the determinized state.
        legal_actions = determinized_state.get_legal_actions()
        
        # --- SELECTION ---
        # Traverse the tree until reaching a node that is either not fully expanded or terminal.
        while not determinized_state.is_terminal() and node.is_fully_expanded(legal_actions):
            node = node.best_child()
            # Apply the action that led to this node to update the determinized state.
            determinized_state = determinized_state.take_action(node.action_taken)
            legal_actions = determinized_state.get_legal_actions()
        
        # --- EXPANSION ---
        if not determinized_state.is_terminal():
            legal_actions = determinized_state.get_legal_actions()
            # Expand by adding child nodes for untried actions.
            untried_actions = [action for action in legal_actions 
                               if action not in [child.action_taken for child in node.children]]
            if untried_actions:
                # Randomly choose one untried action.
                action = random.choice(untried_actions)
                new_state = determinized_state.take_action(action)
                child_node = ISMCTSNode(new_state.get_public_state(), parent=node)
                child_node.action_taken = action
                node.children.append(child_node)
                # Move to the new child node.
                node = child_node
                determinized_state = new_state
        
        # --- SIMULATION (Rollout) ---
        # From the current node, simulate a random playout until a terminal state is reached.
        while not determinized_state.is_terminal():
            legal_actions = determinized_state.get_legal_actions()
            if not legal_actions:
                break  # No legal actions available.
            action = random.choice(legal_actions)
            determinized_state = determinized_state.take_action(action)
        
        # --- BACKPROPAGATION ---
        # Get the reward from the terminal state.
        reward = determinized_state.get_reward()
        # Backpropagate the reward up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # Choose the best action from the root node (without exploration, i.e., greedy).
    best_child = root.best_child(exploration_weight=0)
    return best_child.action_taken  # Return the action that led to the best child.

# ------------------------------
# Example usage:
#
# Assume you have implemented a GameState class with the following methods:
# - get_public_state(): returns the public view of the game state.
# - get_legal_actions(): returns a list of legal actions.
# - take_action(action): returns a new game state after applying the action.
# - is_terminal(): returns True if the game state is terminal.
# - get_reward(): returns a numerical reward for the terminal state.
#
# Example:
# players = ['Alice', 'Bob', 'Charlie']
# game_state = GameState(players, phase='Day')
# game_state.alive_players['Alice']['role'] = 'Imp'
# game_state.alive_players['Bob']['role'] = 'Fortune Teller'
# game_state.alive_players['Charlie']['role'] = 'Villager'
# game_state.roles['Alice'] = 'Imp'
# game_state.roles['Bob'] = 'Fortune Teller'
# game_state.roles['Charlie'] = 'Villager'
# game_state.nominations.append(('Alice', 'Charlie'))
#
# Run ISMCTS for current player 'Alice' with 1000 simulations:
# best_action = ismcts(game_state, current_player='Alice', num_simulations=1000)
# print("Recommended action for Alice:", best_action)


class BotCGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BotCGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        # First GCN layer with ReLU activation.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Second GCN layer; output can be used for state evaluation or classification.
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def gamestate_to_gnn_features(game_state):
    """
    Converts a GameState instance into a tensor of features for each player node.
    
    Each player's feature vector consists of:
      - Role indicator: 1 if the player's role is known, 0 if unknown.
      - Alive status: 1 if alive, 0 otherwise.
      - Nominated status: 1 if nominated, 0 otherwise.
      - Day count (as a scalar, can be normalized later).
      
    :param game_state: An instance of GameState.
    :return: A torch.Tensor of shape (num_players, feature_dim)
    """
    features = []
    day_feature = game_state.day_count  # Using raw day count as a feature
    for player, info in game_state.alive_players.items():
        # Role indicator: 1 if role is known, 0 otherwise.
        role_indicator = 1 if game_state.roles.get(player) is not None else 0
        
        # Alive status: 1 if the player is alive, else 0.
        alive_indicator = 1 if info['alive'] else 0
        
        # Nominated status: 1 if nominated, else 0.
        nominated_indicator = 1 if info['nominated'] else 0
        
        # Combine features into a single vector.
        feature_vector = [role_indicator, alive_indicator, nominated_indicator, day_feature]
        features.append(feature_vector)
    
    return torch.tensor(features, dtype=torch.float)


# Example function to convert conversation history into an embedding.
# In a real application, you would use a text encoder (e.g., SentenceTransformer)
# Here, we simulate the embedding with a dummy vector (e.g., fixed random vector).
def encode_conversation_text(text, embedding_dim=128):
    # For demonstration, we simply return a tensor filled with a constant.
    # Replace this with your text encoding mechanism.
    return torch.full((embedding_dim,), 0.5)

def conversations_to_features_and_edges(conversation_manager, player_to_idx, embedding_dim=128):
    """
    Converts all conversations from a ConversationManager into:
      - conversation_features: A tensor of shape (num_conversation_nodes, embedding_dim)
      - edge_index: A tensor of shape (2, num_edges) connecting conversation nodes to player nodes.
    
    Args:
        conversation_manager: An instance of ConversationManager.
        player_to_idx: A dictionary mapping player names to node indices in the overall graph.
                       (For example, if players are the first nodes in your graph.)
        embedding_dim: The dimensionality of the conversation embedding.
        
    Returns:
        conversation_features: Tensor of conversation node features.
        edge_index: Tensor of shape [2, num_edges] indicating edges from conversation nodes to players.
                    In our case, we assume an undirected edge, so we add both directions.
    """
    conversation_features_list = []  # List to store conversation embeddings.
    edges = []  # List to store edges between conversation nodes and player nodes.
    
    # We'll assume that conversation nodes will be added after all player nodes.
    # So if we have N players, then the first conversation node will have index N, etc.
    conversation_start_idx = len(player_to_idx)
    
    # Retrieve all conversations from the manager.
    conversations = conversation_manager.get_conversations()
    
    # Iterate over each conversation in the dictionary.
    for conv_idx, (participants_tuple, conv_obj) in enumerate(conversations.items()):
        # Get the aggregated conversation text.
        conv_text = conv_obj.get_history_text()
        # Convert the text to an embedding (using our dummy encoder).
        conv_embedding = encode_conversation_text(conv_text, embedding_dim=embedding_dim)
        conversation_features_list.append(conv_embedding)
        
        # The index of this conversation node in the final graph:
        conv_node_idx = conversation_start_idx + conv_idx
        
        # Create edges between this conversation node and all players in the conversation.
        for player in conv_obj.get_participants():
            # Get the player's index from the provided mapping.
            if player in player_to_idx:
                player_idx = player_to_idx[player]
                # Add edge from conversation node to player.
                edges.append((conv_node_idx, player_idx))
                # And add reverse edge (undirected graph).
                edges.append((player_idx, conv_node_idx))
    
    # Stack all conversation features into a single tensor.
    if conversation_features_list:
        conversation_features = torch.stack(conversation_features_list, dim=0)
    else:
        conversation_features = torch.empty((0, embedding_dim))
    
    # Convert edge list to a tensor of shape [2, num_edges].
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return conversation_features, edge_index



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
    next_action_suggestion = ask_chatgpt_for_next_action(game_state, action_space, current_player, conversation_manager, roles)
    print("ChatGPT suggests the following possible action(s):")
    print(next_action_suggestion)
    



# Convert the game state to GNN node features.
player_node_features = gamestate_to_gnn_features(game_state)
print("Player node features tensor:")
print(player_node_features)


# Create a mapping for player nodes.
# For example, assume we have 3 players with indices: Alice->0, Bob->1, Charlie->2.
player_to_idx = {'Alice': 0, 'Bob': 1, 'Charlie': 2}

# Convert the conversation data to conversation features and edges.
conversation_features, conv_edge_index = conversations_to_features_and_edges(conversation_manager, player_to_idx, embedding_dim=128)

print("Conversation features tensor shape:", conversation_features.shape)
print("Edge index tensor shape:", conv_edge_index.shape)
print("Edge index tensor:", conv_edge_index)


print("Player node features shape:", player_node_features.shape)
print("Conversation features shape:", conversation_features.shape)
embedding_dim = conversation_features.shape[1]  # Get dimension of conv features
feature_dim = player_node_features.shape[1]  # Get dimension of player features
# If dimensions are different, apply a linear layer to project features
if feature_dim != embedding_dim:
    projection = nn.Linear(feature_dim, embedding_dim)  # Create transformation layer
    player_node_features = projection(player_node_features)  # Project player features
# Kombiniere alle Knotenfeatures in einer Feature-Matrix.
# Zum Beispiel: Zuerst die Spieler, dann die Statements.
x = torch.cat([player_node_features, conversation_features], dim=0)

# Optional: Falls du Kantenattribute (z.B. Gewichtungen) berechnen moechtest, kannst du diese hier hinzufuegen.
edge_attr = None  # Hier koenntest du beispielsweise Kosinus-aehnlichkeiten der verbundenen Knoten berechnen.

# Erstelle das Data-Objekt fuer den Graphen.
graph_data = Data(x=x, edge_index=conv_edge_index, edge_attr=edge_attr)


# Beispiel: Wir nehmen an, dass die Knotenfeatures 384-dim sind.
# Wir koennten das Modell so definieren, dass es beispielsweise 128 Hidden Units verwendet
# und 2 Klassen (z. B. "good" vs. "evil" oder weitere Spiel-spezifische Zustaende) ausgibt.
model = BotCGNN(input_dim=x.size(1), hidden_dim=128, output_dim=2)

# Fuehre einen Forward-Pass durch:
output = model(graph_data.x, graph_data.edge_index)
print("GNN Output:", output)
