""" This script can be used to plot the convergence of Q-values.
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

# get data
config = pickle.load(open("../projects/" + project + "/config.pkl",  "rb"))
trials = config.trials

for trial in range(trials):

  # plot V and Q values for final policy
  data = pickle.load(open("../projects/" + project + "/trial_" + str(
    trial) + "/train_data.pkl", "rb"))
  agents = data["agents"]

  for idx, agent in enumerate(agents):
    updates = agent.updates
    plt.plot(list(range(len(updates))), updates)
    plt.xlabel("Update")
    plt.ylabel("Difference")
    plt.savefig("../projects/" + project + "/trial_" + str(trial)
                + "/plots/updates_" + str(idx)+ ".eps")
    plt.clf()


