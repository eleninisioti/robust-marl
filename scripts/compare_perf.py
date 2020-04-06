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

methods = ["Qlearning", "minimaxQ2", "RomQ2"]
adversaries = ["Qlearning", "minimaxQ", "RomQ"]
seaborn.set()
directories = ["../projects/final/" + method for method in methods]
trials = 3


rewards_list = [[]]*len(methods)
duration_list = [[]]*len(methods)


# plot rewards for each method and compare across methods and trials
for trial in range(trials):


  for method_idx, directory in enumerate(directories):
    rewards_dict = {"sample": [], "reward": []}
    duration_dict = {"sample": [], "duration": []}
    
    config = pickle.load(open(directory + "/config.pkl", "rb"))
    interm_episodes = list(range(config.episodes))[
                      0::(int(config.episodes / 10))]
    final_episode = interm_episodes[-1]
    trial_dir = directory + "/trial_" + str(trial)

    delta_values = config.delta_values
    delta_values = delta_values[::2]

    method = methods[method_idx]

    # ----- load performance data -----
    results = []
    samples = []
    episodes_duration = []
    for delta_idx, delta in enumerate(delta_values):
      results.append([])
      samples.append([])
      episodes_duration.append([])
  
      for episode in interm_episodes:
        episode = int(episode)


        episode_dir = trial_dir + "/episode_" + str(episode) + "/adversary_" \
                        + adversaries[method_idx] + "_attack_worst"

        # load train data
        train_file =  trial_dir + "/episode_" + str(episode) + "/agents.pkl"
        train_data = pickle.load(open(train_file, "rb"))
        episode_samples = train_data["episode_samples"]

        #episode_samples += episode
        # load test data
        current_file = episode_dir + "/data/test_data_" + str(delta) + ".pkl"
        data = pickle.load(open(current_file, "rb"))
        rewards = data["performance_test"]["episode_rewards"]
        duration = data["performance_test"]["episodes_duration"]

        mean_reward = np.mean(rewards)
        results[delta_idx].append(mean_reward)
        samples[delta_idx].append(episode_samples)

        episodes_duration[delta_idx].append(np.mean(duration))

        if delta_idx == 0:

          rewards_dict["sample"].append(episode)
          rewards_dict["reward"].append(mean_reward)
          duration_dict["sample"].append(episode_samples)
          duration_dict["duration"].append(episode_samples)

    temp_list = [rewards_dict]
    for idx, el in enumerate(rewards_list[method_idx]):
      el2 = rewards_list[method_idx][idx]
      temp_list.append([el2])
    rewards_list[method_idx] = temp_list

    temp_list = [duration_dict]
    for idx, el in enumerate(duration_list[method_idx]):
      el2 = duration_list[method_idx][idx]

      temp_list.append(el2)
    duration_list[method_idx] = temp_list

# plot each trial seperately:
for trial in range(trials):  
  for method_idx, method in enumerate(adversaries): 
    
    temp_dict = rewards_list[method_idx][trial]

    samples = temp_dict["sample"]
    rewards = temp_dict["reward"]
    
    plt.plot(samples,  rewards, label=method)

  plt.ylabel("Average episode reward, $r$")
  plt.xlabel("Episode")
  plt.legend(loc="lower center", prop={'size': 10}, ncol=3)
  plt.savefig("../plots/trial_" + str(trial) +"rewards.png")
  plt.clf()



for method_idx, method in enumerate(methods):

  method_data = rewards_list[method_idx]
  method_dict = {}
  for dict in method_data:
    method_dict.update(dict)

  dataframe= pd.DataFrame(data=method_dict)
  seaborn.lineplot(x="sample", y="reward", data=dataframe, ci=90,
                   err_style="band", label=method)
plt.xlabel("Sample")
plt.ylabel("Average episode reward, $r$")
plt.legend()
plt.savefig("../plots/compare_rewards.png")
plt.clf()

for method_idx, method in enumerate(methods):
  dataframe = pd.DataFrame(data=duration_list[method_idx])
  seaborn.lineplot(x="sample", y="duration", data=dataframe, ci=90,
                   err_style="band", label=method)
plt.xlabel("Sample")
plt.ylabel("Average episode duration, $r$")
plt.legend()
plt.savefig("../plots/compare_duration.png")
plt.clf()
  


for method_idx, directory in enumerate(directories):
  method = methods[method_idx]
  

  # ---- load config data -----
  config = pickle.load(open(directory + "/config.pkl", "rb"))
  trial_dir = directory + "/trial_" + str(trial)
  episodes = config.episodes - 1
  delta_values = config.delta_values
  delta_values = delta_values[::2]

  # ----- load performance data -----
  data_for_conf = {"delta": [], "reward": []}
  for delta in delta_values:
    episode = episodes
    current_dir = trial_dir + "/episode_" + str(episode)
    current_file = current_dir + "/test_data_" + str(delta) + ".pkl"
    data = pickle.load(open(current_file, "rb"))
    rewards = data["performance_test"]["episode_rewards"]
    for test_episode_reward in rewards:
      data_for_conf["reward"].extend([test_episode_reward])
      data_for_conf["delta"].extend([delta])

  dataframe = pd.DataFrame(data=data_for_conf)

  seaborn.lineplot(x="delta", y="reward", data=dataframe, label=method, ci=90,
                 err_style="band")

# delta
plt.ylabel("Test performance, $r$")
plt.xlabel("Episode")
plt.legend(loc='upper center', prop={'size': 10}, ncol=3)
plt.savefig("../plots/compare_delta.png")
plt.clf()


