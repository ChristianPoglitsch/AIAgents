import random

from botc_base import MCTS
from botc_base import ConversationManager
from botc_base import MCTS
from botc_base import MCTSNode
from botc_base import simulation_policy
from botc_base import BasicGameState

num_games = 1
num_iterations = 200
num_child_node = 3

def reward_function(node, new_node):
    multiply = new_node.action.get('Action')
    get_reward = float(multiply)
    return get_reward

class Simple(BasicGameState):
    def __init__(self, players):
        super().__init__(players)
        
    def is_terminal(self):
        value = random.random()
        if value < 0.05:
            return True
        return False
  
def simulation_policy(node, model, print_output, server_based, num_child_node):
    child_nodes = []
    num_max_nodes = int(max(1, (random.random() * num_child_node + 1)))
    for i in range(num_max_nodes):
        action = {"Action": f"{i}"}
        child_nodes.append(MCTSNode(state=Simple(''), action=action, parent=node, terminal_state=node.state.is_terminal()))
    return child_nodes

conversationManager = ConversationManager()

for i in range(num_games):
    print('Start Game')

    # Create a dummy ConversationManager and add a conversation.
    conv_manager = ConversationManager()
    simple = Simple('')
    
    # Create an MCTS instance
    mcts = MCTS(simulation_policy, reward_function, num_child_node, iterations=num_iterations, exploration_weight=1.41)

    # Run MCTS to get the best action/state
    best_node = mcts.search(simple, conv_manager, None, None, None)

    print(mcts.print_tree())

