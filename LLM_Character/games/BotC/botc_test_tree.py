import random

from botc_base import MCTS
from botc_base import ConversationManager
from botc_base import MCTS
from botc_base import MCTSNode
from botc_base import simulation_policy
from botc_base import BasicGameState

num_games = 1
num_iterations = 3
num_child_node = 2

def reward_function(node, new_node):
    return 1.0

class Simple(BasicGameState):
    def __init__(self, players):
        super().__init__(players)
  
def simulation_policy(node, model, print_output, server_based, num_child_node):
    child_nodes = []
    for i in range(num_child_node):
        action = {"Action": f"Action {i}"}
        child_nodes.append(MCTSNode(state=Simple(''), action=action, parent=node))
    return child_nodes

conversationManager = ConversationManager()

for i in range(num_games):
    print('Start Game')

    # Create a dummy ConversationManager and add a conversation.
    conv_manager = ConversationManager()
    simple = Simple('')
    
    # Create an MCTS instance
    mcts = MCTS(simulation_policy, reward_function, num_child_node, iterations=num_iterations, exploration_weight=1.0)

    # Run MCTS to get the best action/state
    best_node = mcts.search(simple, conv_manager, None, None, None)

    print(mcts.print_tree())

