""" This script can be used to plot the heatmap of state visits


Two identical plots are produced, one refers to the learning and the other to
 the execution time. Each plot consists of 3 sub-plots
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
episode_train = config.episodes -1
data_train = {"time_steps": [], "rewards": []}
data_train = {}
all_episodes = np.arange(0, config.episodes-1, 200)


for trial in range(trials):

  # plot V and Q values for final policy
  data = pickle.load(open("../projects/" + project + "/trial_" + str(
    trial) + "/train_data.pkl", "rb"))


  if config.method in ["Qlearning", "SARSA", "DoubleQ"]:

    agents = [data["agents"][0], data["agents"][0]]

  else:
    agents = data["agents"]
    Vtables = [agent.V for agent in agents]

  Qtables = [agent.Qtable for agent in agents]
  Vvalues = np.zeros(shape=(config.capacity + 2, config.capacity + 2))
  #plt.figure(figsize=(10, 10))

  for idx, agent in enumerate(agents):
    Qtable = Qtables[idx]
    if config.method  not in ["Qlearning", "SARSA", "DoubleQ"]:
      Vtable = Vtables[idx]
    img = plt.imshow(np.zeros(shape=(config.capacity + 2, config.capacity + 2)),
                     origin="lower", cmap=plt.cm.Blues)

    for s1 in range(config.capacity + 2):
      for s2 in range(config.capacity + 2):
        offset = 0
        current_entry = [slice(None)] * 2
        current_entry[0] = s1
        current_entry[1] = s2

        if config.method in ["Qlearning", "SARSA", "DoubleQ"]:
          Qtable_current = Qtable[tuple(current_entry)]
          Vvalues[s1, s2] = np.max(Qtable_current)
        else:
          Vvalues[s1, s2] = Vtable[tuple(current_entry)]
        plt.text(s1, s2, str(round(Vvalues[s1, s2], 2)), va='center',
                 ha='center', color="red", fontsize=12, fontweight='bold')

    #img = plt.imshow(Vvalues,origin="lower")

    plt.xlabel("$s_1$", color="green")
    plt.ylabel("$s_2$", color="blue")
    #plt.colorbar(img, cmap=plt.cm.Blues)
    plt.title("$V^*(s)$")
    plt.savefig("../projects/" + project + "/trial_" + str(trial)
                + "/plots/Value_final_ " + str(idx) + ".eps")
    plt.clf()

    plt.figure(figsize=(20, 20))

    img = plt.imshow(np.zeros(shape=(config.capacity + 2, config.capacity + 2)),
                     origin="lower", cmap=plt.cm.Blues)

    for s1 in range(config.capacity + 2):
      for s2 in range(config.capacity + 2):
        offset = 0
        current_entry = [slice(None)] * 2
        current_entry[0] = s1
        current_entry[1] = s2
        Qtable_current = Qtable[tuple(current_entry)]

        Qvalues = np.ndarray.tolist(np.ndarray.flatten(Qtable_current))

        qstring = [str(idx+1) + ": " + str(value) for idx, value in enumerate(
          Qvalues)]
        single_string = "\n".join(qstring)

        plt.text(s1, s2, single_string, va='center', ha='left', fontsize=10)

    description = "$a1_s, a1_t, a2_s, a2_t$ \n 0, 0,0,0, \n 0,0,0,1 \n 0,0,1," \
                  "0 \n 0,0,1,1 \n 0,1,0,0 \n 0,1,0,1 \n 0,1,1,0 \n 0,1,1," \
                  "1 \n 1,0,0,0"
    plt.text(5, 1, description, va='center', ha='center', fontsize=10)
    plt.xlabel("$s_1$", color="green")
    plt.ylabel("$s_2$", color="blue")
    plt.title("$Q^*(s,a)$")
    plt.savefig("../projects/" + project + "/trial_" + str(trial) +
                "/plots/Qtable_final_ " + str(idx) + ".png")
    plt.clf()



  # ----- plot evolution of rewards with episodes -----
  # trial_performance = []
  #
  #
  # for episode in range(episode_train):
  #
  #   # load data
  #   data_train = pickle.load(open("../projects/" + project + "/trial_" + str(
  #     trial) + "/episode_" + str(episode) + "/train_performance.pkl", "rb"))
  #
  #   trial_performance.append(data_train["performance"])
  #
  # data_train["time_steps"].extend(list(range(len(trial_performance))))
  # data_train["rewards"].extend(trial_performance)


# dataframe = pd.DataFrame(data=data_train)
#
# seaborn.lineplot(x="time_steps", y="rewards", data=dataframe, ci="sd")
# plt.ylabel("System reward, $\\bar{r}$")
# plt.xlabel("Episode, $e$")
# plt.legend()
# plt.savefig("../projects/" + project + "/performance_episodes.eps")
# plt.clf()
#
# # save data for tuning
# pickle.dump(dataframe, open("../projects/" + project + "/system_reward.pkl",
#                           "wb"))



