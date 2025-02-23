import math
import random

class Role:
    def __init__(self, name, team, ability):
        self.name = name
        self.team = team
        self.ability = ability

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

# Example action space using dictionaries
action_space = [
    {'type': 'nominate', 'nominator': None, 'nominee': None},  # A player nominates another player
    {'type': 'vote', 'voter': None, 'vote': None},  # A player votes during the voting phase
    {'type': 'kill', 'killer': None, 'target': None},  # The Imp kills a target at night
    {'type': 'protect', 'protector': None, 'target': None},  # Example: Monk protects a target
    {'type': 'bluff', 'bluffer': None, 'claim': None},  # A player bluffs about their role
    {'type': 'reveal', 'revealer': None},  # Example: Investigator reveals information
    {'type': 'disrupt', 'disruptor': None},  # Example: Poisoner poisons a player
    {'type': 'check', 'checker': None, 'target': None},  # Example: Empath checks the status of neighbors
    {'type': 'announce', 'announcer': None, 'message': None},  # A player makes a public announcement
    {'type': 'vote_grimoire', 'voter': None, 'vote': None}  # Example: Grimoire voting action
]

# Example action space using classes
class Action:
    def __init__(self, action_type, **kwargs):
        self.type = action_type
        self.params = kwargs

# Defining the 10 actions
action_space = [
    Action('nominate', nominator=None, nominee=None),
    Action('vote', voter=None, vote=None),
    Action('kill', killer=None, target=None),
    Action('protect', protector=None, target=None),
    Action('bluff', bluffer=None, claim=None),
    Action('reveal', revealer=None),
    Action('disrupt', disruptor=None),
    Action('check', checker=None, target=None),
    Action('announce', announcer=None, message=None),
    Action('vote_grimoire', voter=None, vote=None)
]

# Example of accessing an action and updating it
example_nominate_action = action_space[0]
example_nominate_action.params['nominator'] = 'Alice'
example_nominate_action.params['nominee'] = 'Bob'

class Player:
    def __init__(self, name, role=None, alive=True):
        """
        Initialize a player in the game.
        
        :param name: The name of the player.
        :param role: The role of the player (e.g., 'Imp', 'Villager', etc.).
        :param alive: Whether the player is alive or dead.
        """
        self.name = name
        self.role = role  # E.g., 'Imp', 'Fortune Teller', 'Villager'
        self.alive = alive
        self.nominated = False
        self.private_knowledge = []  # E.g., [(player, status)] for Empath
        self.effects = []  # E.g., 'Poisoned', 'Drunk'
    
    def add_private_knowledge(self, knowledge):
        """
        Add private knowledge to the player's information.
        
        :param knowledge: Information to add (e.g., a tuple like (player, True/False)).
        """
        self.private_knowledge.append(knowledge)
    
    def apply_effect(self, effect):
        """
        Apply a status effect to the player (e.g., 'Poisoned', 'Drunk').
        
        :param effect: The effect to apply.
        """
        if effect not in self.effects:
            self.effects.append(effect)
    
    def remove_effect(self, effect):
        """
        Remove a status effect from the player.
        
        :param effect: The effect to remove.
        """
        if effect in self.effects:
            self.effects.remove(effect)
    
    def reset_for_new_day(self):
        """
        Reset daily attributes for the player, such as nomination status.
        """
        self.nominated = False

    def __repr__(self):
        """
        Representation of the player for debugging purposes.
        """
        return (f"Player(name={self.name}, role={self.role}, alive={self.alive}, "
                f"nominated={self.nominated}, private_knowledge={self.private_knowledge}, effects={self.effects})")


class GameState:
    def __init__(self, players, phase='Day'):
        # Public information
        self.phase = phase
        self.alive_players = {player: {'role': None, 'alive': True, 'nominated': False, 'effects' : None} for player in players}
        self.nominations = []  # List of tuples (nominator, nominee)
        
        # Private information (not visible to all)
        self.roles = {player: None for player in players}  # Roles, e.g., {'Alice': 'Imp', 'Bob': 'Fortune Teller'}
        self.private_knowledge = {player: None for player in players}  # E.g., {'Bob': ('Charlie', False)}
        
        # Extra data for tracking game progress
        self.day_count = 1
        self.effects = {}  # Any ongoing effects, like poisoning or drunk states

    def print_game_state(self):
        print("Game State:")
        print(f"Phase: {self.phase}")
        print(f"Day Count: {self.day_count}")
        print("Players:")
        for player, info in self.alive_players.items():
            role = info['role'].name if info['role'] else "Unknown"
            status = "Alive" if info['alive'] else "Dead"
            print(f"  {player}: Role={role}, Status={status}, Nominated={info['nominated']}")
        print("Nominations:")
        for nominator, nominee in self.nominations:
            print(f"  {nominator} nominated {nominee}")
        print(f"Execution: {self.execution if self.execution else 'None'}")

    def get_public_state(self):
        """Returns a version of the state visible to all players."""
        return {
            'phase': self.phase,
            'alive_players': self.alive_players,
            'nominations': self.nominations,
            'day_count': self.day_count,
        }

    def get_private_state(self, player):
        """Returns the private state for a specific player."""
        return {
            'role': self.roles[player],
            'private_knowledge': self.private_knowledge[player],
            # Add any other player-specific information
        }

    def is_terminal(self):
        """Check if the game has reached a terminal state."""
        # Terminal state could be when the Imp is dead, or only evil players are alive
        alive_roles = {self.roles[p] for p in self.alive_players if self.alive_players[p]}
        return 'Imp' not in alive_roles or len(alive_roles) <= 1

    def take_action(self, action):
        """Apply an action to the game state and return a new state."""
        new_state = GameState(self.phase, self.alive_players.keys())
        new_state.roles = self.roles.copy()
        new_state.private_knowledge = self.private_knowledge.copy()
        new_state.nominations = self.nominations.copy()
        new_state.effects = self.effects.copy()
        new_state.day_count = self.day_count

        # Apply action logic
        if action['type'] == 'nominate':
            new_state.nominations.append((action['nominator'], action['nominee']))
        elif action['type'] == 'execute':
            new_state.alive_players[action['target']] = False
        # Handle other actions like poisoning, bluffing, etc.
        
        # Transition phase if needed
        if len(new_state.nominations) >= len(self.alive_players) - 1:
            new_state.phase = 'Night' if self.phase == 'Day' else 'Day'
            new_state.nominations.clear()

        return new_state

    def get_reward(self):
        # Return the reward for the current state
        # Example: +1 if the Imp is not suspected, -1 if suspected
        return random.choice([1, -1])

    def get_legal_actions(self):
        """Return a list of legal actions from the current state."""
        legal_actions = []
        if self.phase == 'Day':
            for nominator in self.alive_players:
                if self.alive_players[nominator]:
                    for nominee in self.alive_players:
                        if nominator != nominee and self.alive_players[nominee]:
                            legal_actions.append({'type': 'nominate', 'nominator': nominator, 'nominee': nominee})
        elif self.phase == 'Night':
            # Define night actions, such as the Imp selecting a target
            for player in self.alive_players:
                if self.roles[player] == 'Imp' and self.alive_players[player]:
                    for target in self.alive_players:
                        if player != target and self.alive_players[target]:
                            legal_actions.append({'type': 'kill', 'killer': player, 'target': target})
        return legal_actions

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_actions())

    def best_child(self, exploration_weight=1.0):
        """Return the child with the highest UCB1 score."""
        return max(self.children, key=lambda child: child.total_reward / child.visits + 
                   exploration_weight * math.sqrt(math.log(self.visits) / (child.visits + 1e-6)))

def mcts(initial_state, num_simulations):
    root = MCTSNode(initial_state)
    
    for _ in range(num_simulations):
        node = root
        
        # Selection
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child()
        
        # Expansion
        if not node.state.is_terminal():
            legal_actions = node.state.get_legal_actions()
            for action in legal_actions:
                new_state = node.state.take_action(action)
                node.children.append(MCTSNode(new_state, node))
        
        # Simulation (Rollout)
        current_node = node
        while not current_node.state.is_terminal():
            legal_actions = current_node.state.get_legal_actions()
            action = random.choice(legal_actions)
            current_node = MCTSNode(current_node.state.take_action(action))
        
        # Backpropagation
        reward = current_node.state.get_reward()
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent

    return root.best_child(exploration_weight=0).state


class BotCEnvironment:
    def __init__(self, players, roles):
        self.state = GameState(players)
        self.assign_roles(roles)
    
    def print_game_state(self):
        self.state.print_game_state()

    def assign_roles(self, roles):
        role_list = list(roles.values())
        random.shuffle(role_list)
        for i, player in enumerate(self.state.alive_players):
            self.state.alive_players[player]['role'] = role_list[i]
    
    def next_phase(self):
        if self.state.phase == 'Night':
            self.state.phase = 'Day'
            self.state.day_count += 1
        elif self.state.phase == 'Day':
            self.state.phase = 'Night'
    
    def nomination_phase(self):
        self.state.phase = 'Nominations'
        
    def voting_phase(self):
        self.state.phase = 'Voting'

    def execute_vote(self):
        if self.state.execution:

            self.state.alive_players[self.state.execution]['alive'] = False
            print(f"{self.state.execution} was executed.")
            #self.state.execution = None
    
    def night_action(self, player, target):
        role = self.state.alive_players[player]['role']
        if role.name == 'Imp' and self.state.phase == 'Night':
            self.state.alive_players[target]['alive'] = False
            print(f"{player} (Imp) killed {target}.")
    
    def nominate(self, nominator, target):
        if self.state.phase == 'Nominations' and self.state.alive_players[nominator]['alive']:
            self.state.alive_players[target]['nominated'] = True
            self.state.nominations.append((nominator, target))
            print(f"{nominator} nominated {target}.")
    
    def vote(self, voter, target):
        if self.state.phase == 'Voting' and self.state.alive_players[voter]['alive']:
            if target in self.state.alive_players and self.state.alive_players[target]['nominated']:
                self.state.execution = target
                print(f"{voter} voted to execute {target}.")


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
        if not participants_tuple in self.conversations:
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
            Retrieve and print all conversations that involve the specified player.
            :param player_name: The name of the player for whom we want to get the conversation history.
            """
            conversations_for_player = []
        
            # Iterate through all conversations
            for participants_tuple, conversation in self.conversations.items():
                # Check if the player is part of the conversation
                if player_name in participants_tuple:
                    conversations_for_player.append(conversation)

            # Print all conversations involving the player
            if conversations_for_player:
                print(f"Conversation history for {player_name}:")
                for conversation in conversations_for_player:
                    conversation.print_conversation()
            else:
                print(f"No conversations found for player: {player_name}")

    def get_conversations(self):
        """
        Returns all current conversations.
        """
        return self.conversations



players = ['Alice', 'Bob', 'Charlie', 'Dana', 'Eve']
gameState = GameState(players)
env = BotCEnvironment(players, roles)

env.next_phase()  # Transition to Day
env.nomination_phase()  # Transition to nomination
env.nominate('Alice', 'Bob')  # Alice nominates Bob
env.voting_phase()  # Transition to Voting
env.vote('Charlie', 'Bob')  # Charlie votes to execute Bob
env.vote('Eve', 'Charlie')  # Charlie votes to execute Bob
env.execute_vote()  # Execute the vote
env.print_game_state()


# Example usage
conversation_manager = ConversationManager()

# Add messages to the conversations
conversation_manager.add_message_to_conversation(['Alice', 'Bob'], 'Alice', "Hey Bob, what's up?")
conversation_manager.add_message_to_conversation(['Alice', 'Bob'], 'Bob', "Not much, just thinking about the vote.")
conversation_manager.add_message_to_conversation(['Bob', 'Charlie'], 'Bob', "Charlie, do you trust Alice?")

# Print a specific conversation
#conversation_manager.print_conversation(['Alice', 'Bob'])
#conversation_manager.print_conversation(['Bob', 'Charlie'])

# List all current conversations (the keys are the player tuples)
# Example usage: After adding some conversations and messages
conversations = conversation_manager.get_conversations()
conversation_manager.get_conversation_for_player('Alice')
print('---')

# Loop through all conversations and print the details
for participants, conversation in conversations.items():
    print(f"Conversation with participants: {participants}")
    conversation.print_conversation()  # Prints the conversation history for these participants
    print()  # Add a blank line for readability between conversations
