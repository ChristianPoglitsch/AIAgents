import pickle
from botc_base import *
from botc import *

mcts_all = None
mcts_all2 = None
# Load from file
with open('mcts_tree_mistral_training_trained-basic.pkl', 'rb') as f:
    mcts_all = pickle.load(f)

#with open('mcts_tree_reward1.pkl', 'rb') as f:
#    mcts_all2 = pickle.load(f)
#mcts_all = mcts_all + mcts_all2
#with open('mcts_tree_reward2.pkl', 'rb') as f:
#    mcts_all2 = pickle.load(f)
#mcts_all = mcts_all + mcts_all2
#with open('mcts_tree_reward3.pkl', 'rb') as f:
#    mcts_all2 = pickle.load(f)
#mcts_all = mcts_all + mcts_all2
#with open('mcts_tree_reward4.pkl', 'rb') as f:
#    mcts_all2 = pickle.load(f)
#mcts_all = mcts_all + mcts_all2

##with open('mcts_tree.pkl', 'wb') as f:
##    pickle.dump(mcts_all, f)

good_wins = 0
evil_wins = 0
num_nodes = 0
conversationManager = ConversationManager()
errors = 0
elapsed_time = 0
num_trees_terminal_state = 0

index = 0
for mcts in mcts_all:
    mcts.print_tree()

    #nodes = mcts.get_all_terminal_nodes(mcts.get_root_node())
    nodes = mcts.get_all_nodes(mcts.get_root_node())
    errors = errors + mcts.errors
    elapsed_time += mcts.end_time - mcts.start_time
    has_terminal_state = False
    
    for node in nodes:
        if node.state.good_win():
            good_wins = good_wins + 1
            if not has_terminal_state:
                num_trees_terminal_state = num_trees_terminal_state + 1
                has_terminal_state = True
            conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())
        if index <= 100 and node.state.evil_win():
            evil_wins = evil_wins + 1
            #if not has_terminal_state:
            #    num_trees_terminal_state = num_trees_terminal_state + 1
            #    has_terminal_state = True
            conversationManager.append_prompt_outcomes(node.conversation_manager.get_prompt_outcomes())
        index = index + 1


print("Good wins: " + str(good_wins) + " / Evil wins: " + str(evil_wins) + " / Errors: " + str(errors) + " / Tree terminal state: " + str(num_trees_terminal_state) + " / #Games: " + str(len(mcts_all)))
print(f"Execution time: {elapsed_time:.6f} seconds")
folder_path = 'training_botc'
conversationManager.export_prompt_outcome_log(folder_path, False)
