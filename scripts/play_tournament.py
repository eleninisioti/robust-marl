""" This script can be used to plot the performance of intermediate training
steps.


"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import numpy as np
import seaborn
import sys
from pathlib import Path
import pandas as pd
import tikzplotlib

import os
sys.path.insert(0,"../source")
from q_agent import *

adv_determ = int(sys.argv[1])
def_determ = int(sys.argv[2])
attack_type = "worst"

# ----- set up -----
seaborn.set()
top_dir = "../projects/debug"
directories = ["Qlearning_A", "minimaxQ_A", "RomQ_A"]
methods = [ "Qlearning", "minimaxQ", "RomQ"]
attack_type = "worst"
adversaries = ["Qlearning", "minimaxQ", "RomQ"]

# ----- each method competes against each method -----
for method_idx, method in enumerate(methods):


  project_dir = top_dir + "/" + directories[method_idx]

  # ---- load config data -----
  config = pickle.load(open(project_dir + "/config.pkl", "rb"))
  delta_values = config.delta_values

  for adversary in adversaries:

    results = {"delta": [], "reward": []}

    for delta in delta_values:

      for trial in range(config.trials):
        epoch = config.epochs[-1]

        # load performance data
        eval_dir = project_dir + "/data/eval/trial_" + str(trial) + "/epoch_"\
                   + str(epoch) + "/adv_" + adversary + "_attack_" + attack_type

        data_test = pickle.load(open(eval_dir + "/data_" + str(delta)
                                     + ".pkl",  "rb"))
        rewards = data_test["performance"]["rewards"]
        results["reward"].append(np.mean(rewards))
        results["delta"].append(delta)

    # save data for plotting
    results_dir = project_dir + "/data/eval/tournament/adv_" + adversary +\
                  "_attack_" + attack_type

    if not os.path.exists(results_dir):
      os.makedirs(results_dir)

    dataframe = pd.DataFrame(data=results)
    pickle.dump({"robust_trials": dataframe, "robust_episodes": dataframe},
            open(results_dir + "/robust_results.pkl", "wb"))


