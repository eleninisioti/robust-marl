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


# ----- set up -----
seaborn.set()
project_dir = "../projects/" + sys.argv[1]
adversary = sys.argv[2]
attack_type = sys.argv[3]
method = sys.argv[4]
N = 2
capacity = 3
symbols = {"Qlearning": "Q", "minimaxQ": "M", "RomQ": "R"}

plots_dir = project_dir + "/plots/adversary_" + adversary + "_attack_" + \
            attack_type
if not os.path.exists(plots_dir):
  os.makedirs(plots_dir)

# ---- load config data -----
#config = pickle.load(open(project_dir + "/config.pkl", "rb"))
#final_episode = config.episodes
#trials = list(range(config.trials))
#final_episode = 80000
if method== "RomQ":
  temp_episodes = [82000, 84000, 84000, 84000, 84000, 84000, 84000,
                   84000]
elif method == "minimaxQ":
  temp_episodes = [10000, 10000, 10000, 10000, 10000, 10000, 10000]
else:
  config = pickle.load(open(project_dir + "/config.pkl", "rb"))
  final_episode = config.interm_episodes
  print(final_episode)
  temp_episodes2 = []
  for trial in range(7):
    temp_episodes2.append(10000-1)
  temp_episodes = []
  for trial in range(7):
    temp_episodes.append(max(final_episode[trial]))
trials = list(range(7))
max_delta = 1
delta_values = [0]
delta_values.extend(np.arange(start=0.1, stop=max_delta + 0.01, step=0.1))

print(temp_episodes)

# plot performance versus delta
trial_dict = {"delta": [], "reward": []}
episodes_dict = {"delta": [], "reward": []}

for trial in trials:
  if method == "Qlearning":
    final_episode = temp_episodes2[trial]
  elif method=="RomQ":
    final_episode = temp_episodes[trial]
  else:
    final_episode = temp_episodes[trial] - 1
    #interm_episodes = config.interm_episodes[trial]

  # directory for loading data
  trial_dir = project_dir +  "/trial_" + str(trial)

  episode_dir = trial_dir + "/episode_" + str(final_episode) + "/adversary_" \
                + adversary + "_attack_" + attack_type

  for delta in delta_values:

    data_test = pickle.load(open(episode_dir + "/data/test_data_" + str(delta) +
                                 ".pkl",  "rb"))
    performance = data_test["performance_test"]
    rewards = data_test["performance_test"]["episode_rewards"]
    trial_dict["reward"].append(np.mean(rewards))
    trial_dict["delta"].append(delta)

    for reward in rewards:
      episodes_dict["reward"].append(reward)
      episodes_dict["delta"].append(delta)


dataframe_trials = pd.DataFrame(data=trial_dict)
seaborn.lineplot(x="delta", y="reward", data=dataframe_trials, ci=90,
                 err_style="band")
plt.xlabel("Probability of attack, $\delta$")
plt.ylabel("Test performance, $r$")
plt.legend()
plt.savefig(plots_dir + "/robust_trials.png")
plt.clf()

dataframe_episodes = pd.DataFrame(data=episodes_dict)
seaborn.lineplot(x="delta", y="reward", data=dataframe_episodes, ci=90,
                 err_style="band")
plt.xlabel("Probability of attack, $\delta$")
plt.ylabel("Test performance, $r$")
plt.legend()
plt.savefig(plots_dir + "/robust_also_episodes.png")
plt.clf()

pickle.dump({"robust_trials": dataframe_trials,
             "robust_episodes": dataframe_episodes},
            open(plots_dir + "/robust_results.pkl", "wb"))


# ----- plot visits heatmaps -----
for trial in trials:
  if method == "Qlearning":
    final_episode = temp_episodes[trial]
  else:
    final_episode = temp_episodes[trial]

  # ----- plot heatmap -----
  trial_dir = project_dir + "/trial_" + str(trial)
  episode_dir = trial_dir + "/episode_" + str(final_episode) + "/adversary_" \
                + adversary +"_attack_" + attack_type

  # directory for saving plots
  trial_dir = project_dir + "/trial_" + str(trial)
  trial_plots_dir = trial_dir + "/plots/" + "/adversary_" + adversary + "_attack_" + \
            attack_type
  if not os.path.exists(trial_plots_dir):
    os.makedirs(trial_plots_dir)

  # load Qtable
  train_data = pickle.load(open(trial_dir + "/episode_" +
                                str(final_episode) + "/train_data.pkl", "rb"))
  agents = train_data["agents"]
  Qtables = [agent.Qtable for agent in agents]
  for delta in delta_values:
    if method == "Qlearning":
      final_episode = temp_episodes2[trial]
    elif method =="RomQ":
      final_episode = temp_episodes[trial]

    else:
      final_episode = temp_episodes[trial] -1
    episode_dir = trial_dir + "/episode_" + str(final_episode) + \
                   "/adversary_" \
                  + adversary + "_attack_" + attack_type
    data_test = pickle.load(open(episode_dir +  "/data/test_data_" + str(
      delta) + ".pkl",  "rb"))
    performance = data_test["performance_test"]


    actions = performance["actions"]
    states = performance["states"]
    curr_states = performance["current_states"]

    iters = len(performance["episode_rewards"])
    if N ==2:
      visits = np.zeros(shape=(capacity+2, capacity+2))

      for iter in range(iters):
        current_states = states[iter]

        visits[current_states[0], current_states[1] ] +=1

        curr_state = curr_states[iter]
        if curr_state == [0,0]:
          visits[0,0] += 1

    # make a color map of fixed colors
    cmap = 'inferno'
    img = plt.imshow(visits.T, cmap=plt.get_cmap(cmap),
                     origin="lower",norm=matplotlib.colors.LogNorm())


    plt.xlabel("$s_1$", color="green")
    plt.ylabel("$s_2$", color="blue")
    #plt.legend("Green arrow: Node 1 \\ Blue arrow: Node 2")
    plt.title(r'$\pi_{}^*(s)$'.format(symbols[adversary]))
    #plt.title("$\mathcal{N}(s)$")
    plt.axvline(x=3.5,color="red",ymax=0.8)
    plt.axhline(y=3.5,color="red",xmax=0.8)

    # plot policy
    actions = performance["actions"]
    states = performance["states"]
    current_states = performance["current_states"]
    nteststeps = len(actions)

    for s1 in range(capacity+1):
      for s2 in range(capacity+1):

        offset = 0
        current_entry = [slice(None)] * 2
        current_entry[0] = s1
        current_entry[1] = s2

        time_steps = [time_step for time_step, state in enumerate(
          current_states)
                      if state==current_entry]
        #print(time_steps, current_entry)
        if len(time_steps) < 1:
         continue
        time_step = time_steps[0]

        current_actions = actions[time_step]

        # transform actions from absolute indices
        current_actions = [1 if action==2 else action for action in
                           current_actions ]

        # find actions of first agent
        Qtable = Qtables[0]

        Qtable_current = Qtable[tuple(current_entry)]
        actions1 = np.argmax(Qtable_current)
        actions1 = list(np.unravel_index(actions1,
                              Qtable_current.shape))

        if len(Qtables) > 1:
          Qtable = Qtables[1]
          Qtable_current = Qtable[tuple(current_entry)]
          actions2 = np.argmax(Qtable_current)
          actions2 = list(np.unravel_index(actions2,
                                           Qtable_current.shape))
        else:
          actions2 = actions1
        current_actions = [actions1[0], actions1[1], actions2[2], actions2[3]]

        #print(temp_actions, current_actions)
        serve_action_1 = current_actions[0]
        send_action_1 = current_actions[1]

        # find actions of second agent

        serve_action_2 = current_actions[2]
        send_action_2 = current_actions[3]

        # draw arrow (when an agent serves, they diminish their own state by 1,
        # when they send they increase the other agent's state by 1, )
        plt.arrow(s1, s2, - serve_action_1/2 + offset, send_action_1/2 - offset,
                  color="green",
                  head_width=0.05, head_length=0.1, length_includes_head=True)


        # draw arrow
        plt.arrow(s1, s2, send_action_2/2 - offset, - serve_action_2/2 + offset,
                  color="blue",
                  head_width=0.05, head_length=0.1, length_includes_head=True)


    # make a color bar
    plt.colorbar(img, cmap=plt.get_cmap(cmap))
    plt.title("$\delta=$" + str(round(delta,2)))
    print(trial_plots_dir)
    plt.savefig(trial_plots_dir + "/heatmap" + symbols[adversary] + str(
      delta) + ".png")
    plt.clf()

