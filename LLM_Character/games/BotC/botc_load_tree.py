import pickle
from botc_base import *
from botc import *

mcts_all = None
mcts_all2 = None
# Load from file
with open('mcts_tree_gpt4o.pkl', 'rb') as f:
    mcts_all = pickle.load(f)

with open('mcts_tree_reward.pkl', 'rb') as f:
    mcts_all2 = pickle.load(f)
mcts_all = mcts_all + mcts_all2
with open('mcts_tree_reward2.pkl', 'rb') as f:
    mcts_all2 = pickle.load(f)
mcts_all = mcts_all + mcts_all2
with open('mcts_tree_reward3.pkl', 'rb') as f:
    mcts_all2 = pickle.load(f)
mcts_all = mcts_all + mcts_all2
with open('mcts_tree_reward4.pkl', 'rb') as f:
    mcts_all2 = pickle.load(f)
mcts_all = mcts_all + mcts_all2

##with open('mcts_tree.pkl', 'wb') as f:
##    pickle.dump(mcts_all, f)

good_wins = 0
evil_wins = 0
num_nodes = 0
conversationManager = ConversationManager()

for mcts in mcts_all:
    mcts.print_tree()

    #nodes = mcts.get_all_terminal_nodes(mcts.get_root_node())
    nodes = mcts.get_all_nodes(mcts.get_root_node())

    for node in nodes:
        if node.state.good_win(): # node.value > 0.0
            good_wins = good_wins + 1
            conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())
        if node.state.evil_win():
            evil_wins = evil_wins + 1
            #conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())
        num_nodes = num_nodes + 1

print("Good wins: " + str(good_wins) + " / Evil wins: " + str(evil_wins) + " / Num nodes: " + str(num_nodes))
folder_path = 'training_botc'
conversationManager.export_prompt_outcome_log(folder_path, False)
