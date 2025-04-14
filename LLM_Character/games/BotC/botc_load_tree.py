import pickle
from botc_base import *
from botc import *

mcts_all = None

# Load from file
with open('mcts_tree.pkl', 'rb') as f:
    mcts_all = pickle.load(f)
    
good_wins = 0
evil_wins = 0
conversationManager = ConversationManager()

for mcts in mcts_all:
    mcts.print_tree()

    nodes = mcts.get_all_terminal_nodes(mcts.get_root_node())    

    for node in nodes:
        if node.state.good_win():
            good_wins = good_wins + 1
            #conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())
        if node.state.evil_win():
            evil_wins = evil_wins + 1
        conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())

print("Good wins: " + str(good_wins) + " / Evil wins: " + str(evil_wins))
folder_path = 'training_botc'
conversationManager.export_prompt_outcome_log(folder_path, False)
