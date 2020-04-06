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
delta_values = config.delta_values
data_train = {"time_steps": [], "rewards": []}
data_train = {}


for trial in range(trials):

  # ----- plot heatmap -----

  for delta in delta_values:

    data_test = pickle.load(open("../projects/" + project + "/trial_" + str(
      trial) + "/episode_" + str(episode_train) +  "/test_data_" + str(delta) + ".pkl",  "rb"))
    performance = data_test["performance_test"]


    # heatmap of state visits
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
    plt.title("$\mathcal{N}(s)$")
    plt.axvline(x=3.5,color="red",ymax=0.8)
    plt.axhline(y=3.5,color="red",xmax=0.8)

    # plot policy
    actions = performance["actions"]
    states = performance["states"]
    current_states = performance["current_states"]
    nteststeps = len(actions)

    for s1 in range(config.capacity+1):
      for s2 in range(config.capacity+1):
        print(s1,s2)

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
        Qtable = data_test["Qtables"][0]

        Qtable_current = Qtable[tuple(current_entry)]
        actions1 = np.argmax(Qtable_current)
        actions1 = list(np.unravel_index(actions1,
                              Qtable_current.shape))
        print(Qtable_current[tuple(actions1)])

        if len(data_test["Qtables"]) > 1:
          Qtable = data_test["Qtables"][1]
          Qtable_current = Qtable[tuple(current_entry)]
          actions2 = np.argmax(Qtable_current)
          actions2 = list(np.unravel_index(actions2,
                                           Qtable_current.shape))
        else:
          actions2 = actions1
        #current_actions = [actions1[0], actions1[1], actions2[2], actions2[3]]

        #print(temp_actions, current_actions)
        serve_action_1 = current_actions[0]
        send_action_1 = current_actions[1]

        # find actions of second agent

        serve_action_2 = current_actions[2]
        send_action_2 = current_actions[3]

        # actions of vertical-axis agent
        # serve_action_1 = actions[iter][0]
        # send_action_1 = actions[iter][1]

        # draw arrow (when an agent serves, they diminish their own state by 1,
        # when they send they increase the other agent's state by 1, )
        plt.arrow(s1, s2, - serve_action_1 + offset, send_action_1 - offset,
                  color="green",
                  head_width=0.05, head_length=0.1, length_includes_head=True)

        # actions of horizontal-axis agent
        # serve_action_2 = actions[iter][2]
        # send_action_2 = actions[iter][3]

        # draw arrow
        plt.arrow(s1, s2, send_action_2 - offset, - serve_action_2 + offset,
                  color="blue",
                  head_width=0.05, head_length=0.1, length_includes_head=True)

        # if tuple([s1,s2]) not in temp_visits:
        #   temp_visits[tuple([s1, s2])] = [serve_action_1, send_action_1,
        #                                serve_action_2,
        #                      send_action_2]
        # else:
        #   print(temp_visits[tuple([s1, s2])],[serve_action_1, send_action_1,
        #                                 serve_action_2,
        #                      send_action_2]  )
        #   if  [serve_action_1, send_action_1, serve_action_2,
        #                      send_action_2] != temp_visits[tuple([s1,s2])]:
        #     print("not equal")


    # make a color bar
    plt.colorbar(img, cmap=plt.get_cmap(cmap))
    plt.title("$\delta=$" + str(round(delta,2)))
    plt.savefig("../projects/" + project + "/trial_" + str(trial) +
                "/plots/test_grid_" + str(delta) + ".eps")
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



