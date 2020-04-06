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
config = pickle.load(open(project_dir + "/config.pkl", "rb"))
adversary = sys.argv[2]
attack_type = sys.argv[3]
symbols = {"Qlearning": "Q", "minimaxQ": "M", "RomQ": "R"}
trials = config.trials
final_episode = config.episodes
interm_episodes = list(range(final_episode))[
                      0::(int(final_episode / 10))]
interm_episodes.append(final_episode-1)
trial = 0

# ----- plot visits heatmaps -----
for episode in interm_episodes:

  # ----- plot heatmap -----
  trial_dir = project_dir + "/trial_" + str(trial)
  episode_dir = trial_dir + "/episode_" + str(final_episode) + "/adversary_" \
                + adversary +"_attack_" + attack_type

  # directory for saving plots
  trial_dir = project_dir + "/trial_" + str(trial)
  episode_plots_dir = episode_dir + "/plots/" + "/adversary_" + adversary + \
                "_attack_" +  attack_type

  if not os.path.exists(episode_plots_dir):
    os.makedirs(episode_plots_dir)

  # load Qtable
  train_data = pickle.load(open(trial_dir + "/train_data.pkl", "rb"))
  agents = train_data["agents"]
  Qtables = [agent.Qtable for agent in agents]


  data_test = pickle.load(open(episode_dir +  "/data/test_data_" + str(
    delta) + ".pkl",  "rb"))
  performance = data_test["performance_test"]


  actions = performance["actions"]
  states = performance["states"]
  curr_states = performance["current_states"]

  iters = len(performance["episode_rewards"])
  if config.N ==2:
    visits = np.zeros(shape=(config.capacity+2, config.capacity+2))

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

  for s1 in range(config.capacity+1):
    for s2 in range(config.capacity+1):

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
  plt.savefig(episode_plots_dir + "/heatmap" + symbols[adversary] + ".png")
  plt.clf()

# # ----- load performance data -----

for trial in trials:
  results = []
  samples = []
  episodes_duration = []

  trial_dir = project_dir + "/trial_" + str(trial)
  trial_plots_dir = trial_dir + "/plots" + "/adversary_" + adversary + "_attack_" + \
                     attack
  for delta_idx, delta in enumerate(delta_values):
    results.append([])
    interm_episodes = list(range(episodes))[0::(int(episodes / 10))]
    interm_episodes.append(episodes - 1)
    #interm_episodes = interm_episodes[1:]
    samples.append([])
    episodes_duration.append([])

    #episode_samples = 0
    for episode in interm_episodes:
      episode = int(episode)

      current_dir = trial_dir + "/episode_" + str(episode)

      # load train data
      train_file = current_dir + "/agents.pkl"
      train_data = pickle.load(open(train_file, "rb"))
      episode_samples = train_data["episode_samples"]

      # load test data
      current_file = current_dir + "/adversary_" + adversary + "_attack_" + \
                     attack + \
                     "/data/test_data_" + str(delta) + ".pkl"
      data = pickle.load(open(current_file, "rb"))
      rewards = data["performance_test"]["episode_rewards"]
      duration = data["performance_test"]["episodes_duration"]


      mean_reward = np.mean(rewards)
      results[delta_idx].append(mean_reward)
      samples[delta_idx].append(episode_samples)
      episodes_duration[delta_idx].append(np.mean(duration))

  # ----- plots about convergence -----
  # using rewards
  plt.plot(samples[0], results[0], label="Qlearning")
  plt.ylabel("Test performance, $r$")
  plt.xlabel("Samples")
  plt.legend()
  plt.savefig(trial_plots_dir + "/rewards_conv.png")
  plt.clf()

  # using duration of episode
  #total_episodes = [item*400 for item in range(len(episodes_duration[0]))]
  plt.plot(samples[0],  episodes_duration[0], label="Qlearning")
  plt.ylabel("Duration")
  plt.xlabel("Sample$")
  plt.legend()
  plt.savefig(trial_plots_dir + "/episodes_conv.png")
  plt.clf()

  # plot performance versus samples for all delta values
  for delta_idx, delta_result in enumerate(results):
    plt.plot(samples[delta_idx], delta_result,
             label="$\delta=$" + str(round(delta_values[delta_idx], 2)))
  plt.ylabel("Test performance, $r$")
  plt.xlabel("Sample")
  plt.legend()
  plt.savefig(trial_plots_dir + "/rewards_delta_conv.png")
  plt.clf()

  pickle.dump({"samples": samples, "results": results,
               "episodes_duration":  episodes_duration},
              open(trial_plots_dir + "/intermediate_results.pkl", "wb"))












