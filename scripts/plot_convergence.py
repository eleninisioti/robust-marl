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
attack_type = sys.argv[2]

# ----- set up -----
project = "../projects/" + top_dir
methods = ["Qlearning", "minimaxQ", "RomQ"]
#methods = ["Qlearning"]
seaborn.set()
directories = [project + "/" + method for method in methods]

seaborn.set()
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
  trials = list(range(config.trials))
  rewards_dict = {"sample": [], "reward": []}
  duration_dict = {"sample": [], "duration": []}
  method = methods[idx]

  for trial in trials:

    interm_epochs = config.interm_epochs[trial]

    trial_dir = directory + "/trial_" + str(trial)
    epoch_samples = 0
    for epoch in interm_epochs:

      epoch_dir = trial_dir + "/epoch_" + str(epoch) + "/adversary_" \
                    + methods[idx] + "_attack_" + attack_type

      # load test data
      current_file = epoch_dir + "/data/test_data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      duration = data["performance"]["durations"]
      duration_dict["duration"].append(np.mean(duration))
      duration_dict["sample"].append(epoch_samples)
      epoch_samples += int(config.train_samples/len(interm_epochs))


  dataframe = pd.DataFrame(data=duration_dict)
  seaborn.lineplot(x="sample", y="duration", data=dataframe, ci=100,
                   err_style="band", label=method)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel("Sample")
plt.ylabel("Average time to reset, $T_{reset}$")
plt.legend()
plt.title("$\delta=0$")
plt.savefig("../plots/comp_dur.png")
plt.clf()

# ----- plot convergence of rewards without attacks for all methods -----
delta = 0
for idx, directory in enumerate(directories):

  config = pickle.load(open(directory + "/config.pkl", "rb"))
  trials = list(range(config.trials))

  rewards_dict = {"sample": [], "reward": []}
  duration_dict = {"sample": [], "duration": []}
  method = methods[idx]

  for trial in trials:

    interm_epochs = config.interm_epochs[trial]
    trial_dir = directory + "/trial_" + str(trial)
    epoch_samples = 0
    for epoch in interm_epochs:

      epoch_dir = trial_dir + "/epoch_" + str(epoch) + "/adversary_" \
                    + methods[idx] + "_attack_" + attack_type

      # load test data
      current_file = epoch_dir + "/data/test_data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      rewards = data["performance"]["rewards"]
      rewards_dict["reward"].append(np.mean(rewards))
      rewards_dict["sample"].append(epoch_samples)
      epoch_samples += int(config.train_samples/len(interm_epochs))


  dataframe = pd.DataFrame(data=rewards_dict)
  seaborn.lineplot(x="sample", y="reward", data=dataframe, ci=100,
                   err_style="band", label=method)

plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.xlabel("Sample")
plt.ylabel("Average sample reward, $r$")
plt.title("$\delta=0$")
plt.legend(loc="lower right")
plt.savefig("../plots/comp_rewards.png")
plt.clf()


# ----- plot convergence of rewards for each method for different
# probabilities of attack  -----
for idx, directory in enumerate(directories):

  config = pickle.load(open(directory + "/config.pkl", "rb"))
  trials = list(range(config.trials))

  rewards_dict = {"sample": [], "reward": []}
  duration_dict = {"sample": [], "duration": []}
  method = methods[idx]
  delta_values = config.delta_values

  for delta in delta_values[::2]:

    for trial in trials:
      interm_epochs = config.interm_epochs[trial]
      trial_dir = directory + "/trial_" + str(trial)

      epoch_samples = 0
      for epoch in interm_epochs:

        epoch_dir = trial_dir + "/epoch_" + str(epoch) + "/adversary_" \
                      + methods[idx] + "_attack_" + attack_type


        # load test data
        current_file = epoch_dir + "/data/test_data_" + str(delta) + ".pkl"
        data = pickle.load(open(current_file, "rb"))
        rewards = data["performance"]["rewards"]
        sum_negative = np.sum([el for el in rewards if el<0])
        sum_positive = np.sum([el for el in rewards if el>0])

        rewards_dict["reward"].append(np.mean(rewards))
        rewards_dict["sample"].append(epoch_samples)

        epoch_samples += int(config.train_samples/len(interm_epochs))


    dataframe = pd.DataFrame(data=rewards_dict)
    seaborn.lineplot(x="sample", y="reward", data=dataframe, ci=100,
                     err_style="band", label="$\delta=$" + str(delta))

  plt.xlabel("Sample")
  plt.ylabel("Average episode reward, $r$")
  plt.title("$\delta=0$")
  plt.legend(loc="lower center", prop={'size': 8}, ncol=3)
  plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
  plt.savefig(directory + "/plots/rewards_attacks.png")
  plt.clf()