import math
import random
import openai
import random

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

def ask_chatgpt_for_next_action(game_state, action_space, current_player, conversation_manager):
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
        f"{action_space_description}\n\n"
        "Game State:\n"
        f"{state_description}\n\n"
        "Conversation History:\n"
        f"{conversation_history}\n\n"
        "Possible action(s) for the current player:"
    )
    
    model = init_model()
    
    messages = AIMessages()
    message = AIMessage(message=prompt, role=DEVELOPER, class_type="Introduction", sender=DEVELOPER)
    messages.add_message(message)
    response = model.query_text(messages)
    return response

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        # Compare the number of children with the number of legal actions from this state.
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.0):
        """Return the child with the highest UCB1 score."""
        return max(
            self.children, 
            key=lambda child: child.total_reward / child.visits + 
                              exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6))
        )

def mcts(initial_state, num_simulations):
    root = MCTSNode(initial_state)
    
    for _ in range(num_simulations):
        node = root
        
        # Selection: Traverse the tree until a node is reached that is not fully expanded or is terminal.
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # Expansion: If the node is not terminal, expand by adding one or more child nodes.
        if not node.state.is_terminal():
            legal_actions = node.state.get_legal_actions()
            # You can choose to expand just one child or all children.
            for action in legal_actions:
                new_state = node.state.take_action(action)
                child_node = MCTSNode(new_state, parent=node)
                node.children.append(child_node)
            # After expansion, choose one of the newly added children at random for simulation.
            node = random.choice(node.children)
        
        # Simulation (Rollout): Simulate a random playout from the node until a terminal state is reached.
        current_state = node.state
        while not current_state.is_terminal():
            legal_actions = current_state.get_legal_actions()
            if not legal_actions:
                break  # No legal action available.
            action = random.choice(legal_actions)
            current_state = current_state.take_action(action)
        
        # Backpropagation: Propagate the reward back through the tree.
        reward = current_state.get_reward()
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    # Return the state of the best child from the root (without exploration)
    return root.best_child(exploration_weight=0).state



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
    next_action_suggestion = ask_chatgpt_for_next_action(game_state, action_space, current_player, conversation_manager)
    print("ChatGPT suggests the following possible action(s):")
    print(next_action_suggestion)
    
