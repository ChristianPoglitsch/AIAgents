import pickle
from botc_base import *
from botc import *

mcts_all = None

# Load from file
with open('mcts_tree.pkl', 'rb') as f:
    mcts_all = pickle.load(f)

mcts = mcts_all[0]
mcts.print_tree()
