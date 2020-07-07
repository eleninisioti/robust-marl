""" This script can be used to plot the adversarial policy

"""
# ----- generic imports
import pickle
import matplotlib.pyplot as plt
import seaborn
import sys

# ----- project-specific imports -----
sys.path.insert(0,"../source")
from q_agent import *

# ----- set up -----
# parse input
project = sys.argv[1]
adversary = sys.argv[2]
plot_interm = sys.argv[3] # indicates whethet to plot intermediate policies

# set up for plots
seaborn.set()
symbols = {"Qlearning": "Q", "minimaxQ": "M", "RomQ": "R"}

# load project data
config = pickle.load(open("../projects/" + project + "/config.pkl",  "rb"))
capacity = config.capacity
trials = config.trials

for trial in range(trials):

  if plot_interm:

    epochs = config.interm_epochs[trial]
    adversary_dirs = []
    for epoch in epochs:
      adversary_dirs.append("../projects/" + project + "/trial_" + str(
        trial) + "/epoch_" + str(epoch) )
  else:
    adversary_dirs = ["../projects/" + project + "/policies/adversary_" + \
                   adversary + "/trial_" + str(trial) ]

  for dir in adversary_dirs:

    plots_dir = dir
    adversary_file = dir + "/adv_policy.pkl"
    adv_policy = pickle.load(open(adversary_file, "rb"))
    adv_policy = adv_policy["sigma"]
    adv_nodes = adv_policy[0]
    adv_actions = adv_policy[1]

    # plot state visits
    grid = np.ones(shape=(capacity+2, capacity+2))
    img = plt.imshow(grid.T, origin="lower", cmap="gray", vmin=0, vmax=1.5)
    plt.xlabel("$s_1$", color="green")
    plt.ylabel("$s_2$", color="blue")
    plt.title(r'$\sigma_{}^*(s)$'.format(symbols[adversary]))
    plt.axvline(x=3.5, color="red", ymax=0.8)
    plt.axhline(y=3.5, color="red", xmax=0.8)

    # plot adversarial actions
    for s1 in range(capacity+1):
      for s2 in range(capacity+1):

        current_state = [s1,s2]

        current_entry = [slice(None)] * len(current_state)
        for idx, el in enumerate(current_state):
          current_entry[idx] = el

        current_node = int(adv_nodes[tuple(current_entry)][0])
        current_actions = adv_actions[tuple(current_entry)]
        serve_action = current_actions[0]
        send_action = current_actions[1]

        if current_node == 0:

          plt.arrow(float(s1), float(s2), - serve_action/2, send_action/2,
                    color="green",
                    head_width=0.05, head_length=0.1, length_includes_head=True)
        else:

          plt.arrow(float(s1), float(s2),  send_action/2, - serve_action/2,
                    color="blue",
                    head_width=0.05, head_length=0.1, length_includes_head=True)

    plt.savefig(plots_dir + "/sigma_" + str(trial) + ".png")
    plt.clf()