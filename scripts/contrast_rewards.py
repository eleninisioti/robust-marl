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

import os
sys.path.insert(0,"../source")
from q_agent import *


# ----- set up -----
seaborn.set()
directory = "../projects/" + sys.argv[1]
step = 200
trial = 0
conv_episode = 7000 # from data

# ---- load config data -----
config = pickle.load(open(directory + "/config.pkl", "rb"))
episodes = config.episodes -1
delta_values = config.delta_values
delta_values = delta_values[::2]

results_train = []
results_test = []
samples = []
episodes_duration = []

for delta_idx, delta in enumerate(delta_values):

  train_dict = {"sample": [], "reward": []}
  test_dict = {"sample": [], "reward": []}

  for trial in range(config.trials):
    trial_dir = directory + "/trial_" + str(trial)

    results_train.append([])
    results_test.append([])
    interm_episodes = list(range(episodes))[0::(int(episodes/20)+1)]
    samples.append([])
    episodes_duration.append([])

    #episode_samples = 0
    for episode in interm_episodes:
      episode = int(episode)

      current_dir = trial_dir + "/episode_" + str(episode)

      # load train data
      train_file = current_dir + "/train_performance.pkl"
      train_data = pickle.load(open(train_file, "rb"))
      agents_file = current_dir + "/agents.pkl"
      agents_data = pickle.load(open(agents_file, "rb"))

      episode_samples = agents_data["episode_samples"]
      train_rewards = train_data["performance"]["episode_rewards"]
      results_train[delta_idx].append(train_rewards[-1])

      train_dict["reward"].append(np.mean(train_rewards[-1]))
      train_dict["sample"].append(episode)

      # load test data
      current_file = current_dir + "/test_data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      test_rewards = data["performance_test"]["episode_rewards"]
      results_test[delta_idx].append(np.mean(test_rewards))

      samples[delta_idx].append(episode_samples)

      test_dict["reward"].append(np.mean(test_rewards))
      test_dict["sample"].append(episode)

  train_dataframe = pd.DataFrame(data=train_dict)
  test_dataframe = pd.DataFrame(data=test_dict)
  seaborn.lineplot(x="sample", y="reward", data=train_dataframe,
                   label="train")
  seaborn.lineplot(x="sample", y="reward", data=test_dataframe, label="test")

  plt.ylabel("Rewards, $r$")
  plt.xlabel("Sample")
  plt.legend()
  plt.savefig(directory + "/plots/train_test_" + str(delta) + ".eps")
  plt.clf()