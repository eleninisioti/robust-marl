""" This script can be used to plot the adversarial policy

"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import numpy as np
import seaborn
import sys
sys.path.insert(0,"../source")
from q_agent import *
import pandas as pd

# ----- set up -----
seaborn.set()
project = sys.argv[1]
adversaries = [sys.argv[2]]

# load project data
#config = pickle.load(open("../projects/" + project + "/config.pkl",  "rb"))
capacity = 3
trials = 4
symbols = {"Qlearning": "Q", "minimaxQ": "M", "RomQ": "R"}

for trial in range(trials):

  for adv_idx, adversary in enumerate(adversaries):

    adversary_file = "../projects/" + project + "/policies/adversary_" + \
                     adversary + "/trial_" + str(trial) + "_adv_policy.pkl"
    print(adversary_file)

    plots_dir =  "../projects/" + project + "/policies/adversary_" + \
                     adversary

    adv_policy = pickle.load(open(adversary_file, "rb"))
    adv_policy = adv_policy["sigma"]
    adv_nodes = adv_policy[0]
    adv_actions = adv_policy[1]

    #print(adv_policy[0])


    # plot state space
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

        print(adv_nodes[tuple(current_entry)])

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