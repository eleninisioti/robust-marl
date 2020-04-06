""" This script can be used to compare the performance of different learning
algorithms


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

projects = ["Qlearning2", "minimaxQ", "RomQ"]
methods = ["Qlearning", "minimaxQ", "RomQ"]
seaborn.set()
directories = ["../projects/samples_final/" + method for method in projects]
trials = 7
max_delta = 1
delta_values = [0]
delta_values.extend(np.arange(start=0.1, stop=max_delta + 0.01, step=0.1))

# for trial in range(trials):
#
#   for idx, directory in enumerate(directories):
#
#     method_samples= []
#     method_rewards = []
#     method_duration = []
#
#     config = pickle.load(open(directory + "/config.pkl", "rb"))
#
#     interm_episodes = config.interm_episodes[trial]
#     final_episode = config.episodes
#     trial_dir = directory + "/trial_" + str(trial)
#
#
#     method = methods[idx]
#
#     # ----- load performance data -----
#     results = []
#     samples = []
#     episodes_duration = []
#     delta = delta_values[0]
#     results.append([])
#     samples.append([])
#     episodes_duration.append([])
#
#     for episode in interm_episodes:
#       episode = int(episode)
#
#       episode_dir = trial_dir + "/episode_" + str(episode) + "/adversary_" \
#                     + methods[idx] + "_attack_worst"
#
#       # load train data
#       train_file = trial_dir + "/episode_" + str(episode) + "/agents.pkl"
#       train_data = pickle.load(open(train_file, "rb"))
#       episode_samples = train_data["episode_samples"]
#
#       # episode_samples += episode
#       # load test data
#       current_file = episode_dir + "/data/test_data_" + str(delta) + ".pkl"
#       data = pickle.load(open(current_file, "rb"))
#       rewards = data["performance_test"]["episode_rewards"]
#       duration = data["performance_test"]["episodes_duration"]
#
#       mean_reward = np.mean(rewards)
#       method_rewards.append(mean_reward)
#       method_duration.append(np.mean(duration))
#       method_samples.append(episode_samples)
#
#     plt.plot(method_samples, method_rewards, label=methods[idx])
#
#   plt.ylabel("Average episode reward, $r$")
#   plt.xlabel("Sample")
#   plt.legend(loc="lower center", prop={'size': 10}, ncol=3)
#   plt.savefig("../plots/trial_" + str(trial) +"rewards_conv.png")
#   plt.clf()
#
#
# for trial in range(trials):
#
#   for idx, directory in enumerate(directories):
#
#     method_samples= []
#     method_rewards = []
#     method_duration = []
#
#     config = pickle.load(open(directory + "/config.pkl", "rb"))
#     interm_episodes = config.interm_episodes[trial]
#
#     final_episode = config.episodes
#     trial_dir = directory + "/trial_" + str(trial)
#
#     delta_values = config.delta_values
#     delta_values = delta_values[::2]
#
#     method = methods[idx]
#
#     # ----- load performance data -----
#     results = []
#     samples = []
#     episodes_duration = []
#     delta = delta_values[0]
#     results.append([])
#     samples.append([])
#     episodes_duration.append([])
#
#     for episode in interm_episodes:
#       episode = int(episode)
#
#       episode_dir = trial_dir + "/episode_" + str(episode) + "/adversary_" \
#                     + methods[idx] + "_attack_worst"
#
#       # load train data
#       train_file = trial_dir + "/episode_" + str(episode) + "/agents.pkl"
#       train_data = pickle.load(open(train_file, "rb"))
#       episode_samples = train_data["episode_samples"]
#
#       # episode_samples += episode
#       # load test data
#       current_file = episode_dir + "/data/test_data_" + str(delta) + ".pkl"
#       data = pickle.load(open(current_file, "rb"))
#       rewards = data["performance_test"]["episode_rewards"]
#       duration = data["performance_test"]["episodes_duration"]
#
#       mean_reward = np.mean(rewards)
#       method_rewards.append(mean_reward)
#       method_duration.append(np.mean(duration))
#       method_samples.append(episode_samples)
#
#     plt.plot(method_samples, method_duration, label=methods[idx])
#
#   plt.ylabel("Average episode reward, $r$")
#   plt.xlabel("Sample")
#   plt.legend(loc="lower center", prop={'size': 10}, ncol=3)
#   plt.savefig("../plots/trial_" + str(trial) +"episode_conv.png")
#   plt.clf()

# ----- plots averaged over trials -----
bins = list(range(0,200000, 5000))
# print(bins)
# for idx, directory in enumerate(directories):
#
#   rewards_dict = {"sample":[], "reward": []}
#   duration_dict = {"sample": [], "duration": []}
#   method = methods[idx]
#   for trial in range(trials):
#     print("trial", trial)
#
#
#     if method == "RomQ":
#       temp_episodes = [82000, 84000, 84000, 84000, 84000, 84000, 84000,
#                    84000]
#       interm_episodes = []
#       for trial_idx, final_episode in enumerate(temp_episodes):
#
#         trial_episodes = np.arange(0, final_episode+1, 2000)
#         interm_episodes.append(trial_episodes)
#     elif method == "minimaxQ":
#       temp_episodes = [10000, 10000, 10000, 10000, 10000, 10000, 10000]
#       interm_episodes = []
#       for trial_idx, final_episode in enumerate(temp_episodes):
#         trial_episodes = np.arange(0, final_episode + 1, 2000)
#         interm_episodes.append(trial_episodes)
#     else:
#       config = pickle.load(open(directory + "/config.pkl", "rb"))
#       interm_episodes = config.interm_episodes
#     #final_episode = interm_episodes[-1]
#     trial_dir = directory + "/trial_" + str(trial)
#
#     delta_values = config.delta_values
#     delta_values = delta_values[::2]
#
#     method = methods[idx]
#
#     # ----- load performance data -----
#     results = []
#     samples = []
#     episodes_duration = []
#     delta = delta_values[0]
#     results.append([])
#     samples.append([])
#     episodes_duration.append([])
#     previous_len = 0
#
#     for episode in interm_episodes[trial]:
#       episode = int(episode)
#
#       episode_dir = trial_dir + "/episode_" + str(episode) + "/adversary_" \
#                     + methods[idx] + "_attack_worst"
#
#       # load train data
#       train_file = trial_dir + "/episode_" + str(episode) + "/agents.pkl"
#       train_data = pickle.load(open(train_file, "rb"))
#       episode_samples = train_data["episode_samples"]
#
#       # episode_samples += episode
#       # load test data
#       current_file = episode_dir + "/data/test_data_" + str(delta) + ".pkl"
#       data = pickle.load(open(current_file, "rb"))
#       rewards = data["performance_test"]["episode_rewards"]
#       duration = data["performance_test"]["episodes_duration"]
#
#       mean_reward = np.mean(rewards)
#
#       if episode_samples < 200000:
#         rewards_dict["reward"].extend([np.mean(rewards)])
#         for bin in bins:
#           if np.mean(episode_samples) > bin:
#             mean_episode_samples = bin
#         rewards_dict["sample"].extend([mean_episode_samples])
#
#     # df_temp = pd.DataFrame(data=rewards_dict)
#     # seaborn.lineplot(x="sample", y="reward", data=df_temp, label=method)
#     # plt.xlabel("Sample")
#     # plt.ylabel("Average episode reward, $r$")
#     # plt.legend()
#     # plt.savefig(      "../plots/comp_rewards_trial_" + str(trial) + "_ethod_" + method + ".png")
#     # plt.clf()
#
#   df = pd.DataFrame(data=rewards_dict)
#   seaborn.lineplot(x="sample", y="reward", data=df, label=method, ci=95,
#                    err_style="band")
#
#   # print(bins)
#   # #pd.np.digitize(df.sample, bins=bins)
#   # df['sample'] = pd.qcut(df['sample'], 1000, duplicates='drop')
#   #df = df.loc[(df['sample'].isin(range(1, 300000, 100)))]
#
#
# plt.xlabel("Sample")
# plt.ylabel("Average episode reward, $r$")
# plt.legend()
# plt.savefig("../plots/comp_rewards.png")
# plt.clf()

for idx, directory in enumerate(directories):

  rewards_dict = {"sample": [], "reward": []}
  duration_dict = {"sample": [], "duration": []}
  method = methods[idx]

  for trial in range(trials):
    print("trial", trial)

    #config = pickle.load(open(directory + "/config.pkl", "rb"))
    if method == "RomQ":
      temp_episodes = [82000, 84000, 84000, 84000, 84000, 84000, 84000,
                       84000]
      interm_episodes = []
      for trial_idx, final_episode in enumerate(temp_episodes):
        trial_episodes = np.arange(0, final_episode + 1, 2000)
        interm_episodes.append(trial_episodes)
    elif method == "minimaxQ":
      temp_episodes = [10000, 10000, 10000, 10000, 10000, 10000, 10000]
      interm_episodes = []
      for trial_idx, final_episode in enumerate(temp_episodes):
        trial_episodes = np.arange(0, final_episode + 1, 2000)
        interm_episodes.append(trial_episodes)
    else:
      config = pickle.load(open(directory + "/config.pkl", "rb"))
      interm_episodes = config.interm_episodes


    #final_episode = interm_episodes[-1]
    trial_dir = directory + "/trial_" + str(trial)

    delta_values = config.delta_values
    delta_values = delta_values[::2]

    method = methods[idx]

    # ----- load performance data -----
    results = []
    samples = []
    episodes_duration = []
    delta = delta_values[0]
    results.append([])
    samples.append([])
    episodes_duration.append([])

    for episode in interm_episodes[trial]:
      episode = int(episode)

      episode_dir = trial_dir + "/episode_" + str(episode) + "/adversary_" \
                    + methods[idx] + "_attack_worst"

      # load train data
      train_file = trial_dir + "/episode_" + str(episode) + "/agents.pkl"
      train_data = pickle.load(open(train_file, "rb"))
      episode_samples = train_data["episode_samples"]

      # episode_samples += episode
      # load test data
      current_file = episode_dir + "/data/test_data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      rewards = data["performance_test"]["episode_rewards"]
      duration = data["performance_test"]["episodes_duration"]

      if episode_samples < 200000:

        duration_dict["duration"].extend([np.mean(duration)])
        for bin in bins:
          if np.mean(episode_samples) > bin:
            mean_episode_samples = bin
        duration_dict["sample"].extend([mean_episode_samples])

  dataframe = pd.DataFrame(data=duration_dict)
  seaborn.lineplot(x="sample", y="duration", data=dataframe, ci=95,
                   err_style="band", label=method)


plt.xlabel("Sample")
plt.ylabel("Average episode reward, $r$")
plt.legend()
plt.savefig("../plots/comp_dur.png")
plt.clf()