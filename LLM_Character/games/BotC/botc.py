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

class GameState:
    def __init__(self, players):
        self.players = {player: {'role': None, 'alive': True, 'nominated': False, 'effects' : None} for player in players}
        self.phase = 'Night'
        self.day_count = 0
        self.nominations = []
        self.execution = None

    def print_game_state(self):
        print("Game State:")
        print(f"Phase: {self.phase}")
        print(f"Day Count: {self.day_count}")
        print("Players:")
        for player, info in self.players.items():
            role = info['role'].name if info['role'] else "Unknown"
            status = "Alive" if info['alive'] else "Dead"
            print(f"  {player}: Role={role}, Status={status}, Nominated={info['nominated']}")
        print("Nominations:")
        for nominator, nominee in self.nominations:
            print(f"  {nominator} nominated {nominee}")
        print(f"Execution: {self.execution if self.execution else 'None'}")

class BotCEnvironment:
    def __init__(self, players, roles):
        self.state = GameState(players)
        self.assign_roles(roles)
    
    def print_game_state(self):
        self.state.print_game_state()

    def assign_roles(self, roles):
        role_list = list(roles.values())
        random.shuffle(role_list)
        for i, player in enumerate(self.state.players):
            self.state.players[player]['role'] = role_list[i]
    
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
            self.state.players[self.state.execution]['alive'] = False
            print(f"{self.state.execution} was executed.")
            self.state.execution = None
    
    def night_action(self, player, target):
        role = self.state.players[player]['role']
        if role.name == 'Imp' and self.state.phase == 'Night':
            self.state.players[target]['alive'] = False
            print(f"{player} (Imp) killed {target}.")
    
    def nominate(self, nominator, target):
        if self.state.phase == 'Nominations' and self.state.players[nominator]['alive']:
            self.state.players[target]['nominated'] = True
            self.state.nominations.append((nominator, target))
            print(f"{nominator} nominated {target}.")
    
    def vote(self, voter, target):
        if self.state.phase == 'Voting' and self.state.players[voter]['alive']:
            if target in self.state.players and self.state.players[target]['nominated']:
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

# Loop through all conversations and print the details
for participants, conversation in conversations.items():
    print(f"Conversation with participants: {participants}")
    conversation.print_conversation()  # Prints the conversation history for these participants
    print()  # Add a blank line for readability between conversations

