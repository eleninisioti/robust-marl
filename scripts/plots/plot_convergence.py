""" This script can be used to compare the convergence rate of different
methods.

Plots for the duration of episodes and mean rewards are produced.

Output: 1) evolution of episode duration without attacks during evaluation (one
plot for all methods)
2) evolution of mean sample reward without attacks during evaluation (one plot
for all methods)
3) evolution of mean sample reward for different probabilities of attack
during evalution (one plot for each method)
"""

# ----- generic imports -----
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import sys
import pandas as pd
import os
import matplotlib

# ----- project-specific imports ------
sys.path.insert(0,"../source")
from q_agent import *

# parse input
top_dir = sys.argv[1]
attack_type = "worst"

# ----- set up -----
project = "../projects/" + top_dir
methods = ["Qlearning", "minimaxQ", "RomQ"]
symbols = {"Qlearning": "{Q-learning}", "minimaxQ": "{minimax-Q}", "RomQ":
  "{RoM-Q}"}
plot_dir = "../projects/" + top_dir + "/plots"
if not os.path.exists(plot_dir):
   os.makedirs(plot_dir)

seaborn.set()
directories = [project + "/" + method for method in methods]
params = {'legend.fontsize': 'large',
         'figure.figsize': (6, 4),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
matplotlib.rcParams.update(params)
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
matplotlib.rcParams['mathtext.default']='regular'
matplotlib.rcParams["font.weight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"

# ----- plot convergence of duration without attacks for all methods -----
delta = 0
for idx, directory in enumerate(directories):

  config = pickle.load(open(directory + "/config.pkl", "rb"))
  trials = list(range(20))
  rewards_dict = {"sample": [], "reward": []}
  duration_dict = {"sample": [], "duration": []}
  method = methods[idx]

  for trial in trials:

    interm_epochs = config.epochs

    trial_dir = directory + "/data/eval/trial_" + str(trial)
    epoch_samples = 0
    for epoch in interm_epochs:

      epoch_dir = trial_dir + "/epoch_" + str(epoch) + "/adv_" \
                    + methods[idx] + "_attack_" + attack_type

      # load test data
      current_file = epoch_dir + "/data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      duration = data["performance"]["durations"]
      duration_dict["duration"].append(np.mean(duration))
      duration_dict["sample"].append(epoch_samples)
      epoch_samples += int(config.train_samples/len(interm_epochs))


  dataframe = pd.DataFrame(data=duration_dict)
  seaborn.lineplot(x="sample", y="duration", data=dataframe, ci=100,
                   err_style="band", label=symbols[method])

plt.plot([0, interm_epochs[-1]], [50, 50], color='black', linestyle ="--",
         linewidth=2)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel("Sample")
plt.ylabel("Average time to reset, $T_{reset}$")
plt.legend()
plt.savefig(plot_dir + "/comp_dur.png", bbox_inches='tight')
plt.clf()

# ----- plot convergence of rewards without attacks for all methods -----
delta = 0
for idx, directory in enumerate(directories):

  config = pickle.load(open(directory + "/config.pkl", "rb"))
  trials = list(range(20))

  rewards_dict = {"sample": [], "reward": []}
  duration_dict = {"sample": [], "duration": []}
  method = methods[idx]

  for trial in trials:

    interm_epochs = config.epochs
    trial_dir = directory + "/data/eval/trial_" + str(trial)
    epoch_samples = 0
    for epoch in interm_epochs:
      
      # load test data
      epoch_dir = trial_dir + "/epoch_" + str(epoch) + "/adv_" \
                    + methods[idx] + "_attack_" + attack_type

      current_file = epoch_dir + "/data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      rewards = data["performance"]["rewards"]
      rewards_dict["reward"].append(np.mean([np.sum(el) for el in rewards]))
      rewards_dict["sample"].append(epoch_samples)
      epoch_samples += int(config.train_samples/len(interm_epochs))

  dataframe = pd.DataFrame(data=rewards_dict)
  seaborn.lineplot(x="sample", y="reward", data=dataframe, ci=100,
                   err_style="band", label=symbols[method])

plt.plot([0, interm_epochs[-1]], [14,14], color='black', linestyle ="--",
         linewidth=2)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel("Sample")
plt.ylabel("Average sample reward, $r$")
plt.legend(loc="lower right")
plt.savefig(plot_dir + "/comp_rewards.png", bbox_inches='tight')
plt.clf()

print(plot_dir)


