""" This script contains plotting functions for debugging purposes
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
import os

# ----- set up -----
seaborn.set()
project = sys.argv[1]
adversary = sys.argv[2]
attack_type = "worst"

# load project's configuration data
config = pickle.load(open("../projects/" + project + "/config.pkl",  "rb"))
trials = config.trials

# process log data
# logs = config.logs
# debug_dir = "../projects/" + project + "/plots/debug"
# if not os.path.exists(debug_dir):
#   os.makedirs(debug_dir)
# defenders = logs[0]["defenders"]
# plt.plot(list(range(len(defenders[-100:-1]))), defenders[-100:-1])
# plt.xlabel("sample")
# plt.ylabel("Defender idx")
# plt.savefig(debug_dir + "/defenders.png")
# plt.clf()

# check statistical behavior of nodes of nodes
keep_stats = {}
for trial in range(trials):

  # load data at the end of training
  final_epoch = config.interm_epochs[trial][-1]
  data = pickle.load(open("../projects/" + project + "/trial_" + str(
    trial) + "/epoch_" + str(final_epoch) + "/adversary_" + adversary +
                          "_attack_worst/data/test_data_0.pkl", "rb"))

  # get all network nodes
  nodes = data["nodes"]


  for node_idx, node in enumerate(nodes):

    statistics = node.statistics
    nsteps = len(statistics["generations"])
    gen_rate = np.sum(statistics["generations"])/nsteps
    exec_rate = statistics["executions"] / nsteps
    arr_rate = statistics["arrivals"] / nsteps
    dep_rate = statistics["departures"] / nsteps

    if node_idx in keep_stats.keys():
      keep_stats[node_idx]["generations"] += gen_rate
      keep_stats[node_idx]["executions"] += exec_rate
      keep_stats[node_idx]["arrivals"] += arr_rate
      keep_stats[node_idx]["departures"] += dep_rate
    else:
      keep_stats[node_idx] = {"generations": gen_rate, "executions": exec_rate,
                              "arrivals": arr_rate,
                          "departures": dep_rate}

      # a node's input needs to be at least larger than its output
      condition = gen_rate + arr_rate - exec_rate - dep_rate
      print("Positive for correct statistics:", condition)


zerosum = 0
for idx, node in enumerate(nodes):
  print("Statistics for node ", idx)
  print("Average generation rate", keep_stats[idx]["generations"]/trials)
  print("Average execution rate", keep_stats[idx]["executions"]/trials)
  print("Average arrivals rate", keep_stats[idx]["arrivals"] / trials)
  print("Average departures rate", keep_stats[idx]["departures"]/trials)

  increase_rate = (keep_stats[idx]["arrivals"] +  keep_stats[idx][
    "generations"] - keep_stats[idx]["executions"] -  keep_stats[idx][
    "departures"])/trials
  print("Increase rate", increase_rate )
  zerosum += keep_stats[idx]["departures"] - keep_stats[idx]["arrivals"]

# one node's input is the other node's output
print("Zero for correct statistics:", zerosum)


for trial in range(trials):

  final_epoch = config.interm_epochs[trial][-1]

  # plot V and Q values for final policy
  data = pickle.load(open("../projects/" + project + "/trial_" + str(
    trial) + "/epoch_" + str(final_epoch) + "/train_data.pkl", "rb"))

  agents = data["agents"]

  Qtables = [agent.Qtable for agent in agents]
  plt.figure(figsize=(20, 20))

  img = plt.imshow(np.zeros(shape=(config.capacity + 2, config.capacity + 2)),
                   origin="lower", cmap=plt.cm.Blues)

  for s1 in range(config.capacity + 2):
    for s2 in range(config.capacity + 2):
      offset = 0
      current_entry = [slice(None)] * 2
      current_entry[0] = s1
      current_entry[1] = s2
      Qtable_current = Qtables[0][tuple(current_entry)]
      Qvalues = np.ndarray.tolist(np.ndarray.flatten(Qtable_current))

      qstring = [str(idx+1) + ": " + str(value) for idx, value in enumerate(
        Qvalues)]
      single_string = "\n".join(qstring)

      plt.text(s1, s2, single_string, va='center', ha='left', fontsize=10)

  description = "$a1_s, a1_t, a2_s, a2_t$ \n 0, 0,0,0, \n 0,0,0,1 \n 0,0,1," \
                "0 \n 0,0,1,1 \n 0,1,0,0 \n 0,1,0,1 \n 0,1,1,0 \n 0,1,1," \
                "1 \n 1,0,0,0"
  plt.text(5, 1, description, va='center', ha='center', fontsize=5)
  plt.xlabel("$s_1$", color="green")
  plt.ylabel("$s_2$", color="blue")
  plt.title("$Q^*(s,a)$")
  plots_dir = "../projects/" + project + "/trial_" + str(trial) + "/plots"
  if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
  plt.savefig(plots_dir + "/Qtable_final.png")
  plt.clf()


# ----- plot the implemeted actions when an attack always occurs during
# evaluation ------
# this can be used to visualise both the learned and adversarial policy
delta_values = [0]
for trial in range(trials):

  epochs = [config.interm_epochs[trial][-1]]
  for epoch in epochs:

    # load training data
    data_train = pickle.load(open("../projects/" + project + "/trial_" +
                                  str(trial) + "/epoch_" + str(epoch) +
                                  "/train_data.pkl", "rb"))
    Qtables = []
    agents = data_train["agents"]
    for agent in agents:
      Qtables.extend([agent.Qtable])

    datasets = {}

    # load data about performance during evaluation
    for delta in delta_values:

      datasets[delta] = pickle.load(open("../projects/" + project +
                                         "/trial_" + str(trial) + "/epoch_" +
                                         str(epoch) + "/adversary_" +
                                         adversary + "_attack_" +
                                         attack_type + "/data/test_data_" +
                                         str(delta) + ".pkl",  "rb"))
    for key, value in datasets.items():
      delta = key
      data_test = value
      performance = data_test["performance"]

      # heatmap of state visits
      actions = performance["actions"]
      next_states = performance["states"]
      current_states = performance["current_states"]

      iters = len(actions)
      visits = np.zeros(shape=(config.capacity+2, config.capacity+2))

      for iter in range(iters):
        next_state = next_states[iter]

        # I am keeping track of the text states, because I want to show
        # over-flows
        visits[next_state[0], next_state[1]] += 1

        # but I need to make sure that resets are also shown
        current_state = current_states[iter]
        if current_state == [0,0]:
          visits[0,0] += 1

      cmap = 'inferno'
      img = plt.imshow(visits.T, cmap=plt.get_cmap(cmap),
                       origin="lower", norm=matplotlib.colors.LogNorm())

      # plot optimal policy
      actions = performance["actions"]
      states = performance["states"]
      current_states = performance["current_states"]
      nteststeps = len(actions)

      for s1 in range(config.capacity+1):
        for s2 in range(config.capacity+1):

          current_entry = [slice(None)] * 2
          current_entry[0] = s1
          current_entry[1] = s2

          time_steps = [time_step for time_step, state in enumerate(
            current_states) if state == current_entry]
          # print(time_steps, current_entry)
          if len(time_steps) < 1:
            continue
          time_step = time_steps[0]

          current_actions = actions[time_step]

          # find actions of first node
          serve_action_1 = current_actions[0]
          send_action_1 = current_actions[1]
          if send_action_1 == 2:
            send_action_1 = 1

          # find actions of second node
          serve_action_2 = current_actions[2]
          send_action_2 = current_actions[3]
          if send_action_2 == 1:
            send_action_2 = 1

          # draw arrow (when a node executes, it reduces its own state by 1,
          # when they send they increase the other agent's state by 1, )
          plt.arrow(s1, s2, - serve_action_1/2, send_action_1/2,
                    color="green",
                    head_width=0.05, head_length=0.1, length_includes_head=True)

          # draw arrow
          plt.arrow(s1, s2, send_action_2/2, - serve_action_2/2,
                    color="blue",
                    head_width=0.05, head_length=0.1, length_includes_head=True)

      # make a color bar
      plt.colorbar(img, cmap=plt.get_cmap(cmap))

      # save plot
      plt.xlabel("$s_1$", color="green")
      plt.ylabel("$s_2$", color="blue")
      plt.title("$\delta=$" + str(round(delta,2)))
      plt.axvline(x=3.5,color="red", ymax=0.8)
      plt.axhline(y=3.5,color="red", xmax=0.8)

      plot_dir = "../projects/" + project + "/trial_" + str(trial) +\
                  "/epoch_" + str(epoch) + "/adversary_" + adversary + \
                  "_attack_" + attack_type + "/plots/debug"
      if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
      plot_file = plot_dir + "/test_grid_" + str(delta) + ".png"
      plt.savefig(plot_file)
      plt.clf()

# ----- plot convergence of value function -----
# plots evolution of value function for a selected set of states
plot_states = [[0,0], [3,3], [0,3], [3,0], [2,2]]
values_dict = {"epoch": [], "value": []}
for plot_state in plot_states:

  for trial in range(trials):

    epochs = config.interm_epochs[trial]
    values = []
    for epoch in epochs:
      print("epoch", epoch)

      # load Qtables
      data_train = pickle.load(open("../projects/" + project + "/trial_" +
                                    str(trial) + "/epoch_" + str(epoch) +
                                    "/train_data.pkl", "rb"))
      Qtables = []
      agents = data_train["agents"]
      for agent in agents:
        Qtables.extend([agent.Qtable])

      Qtable = Qtables[0] # only plot for one agent
      current_entry = [slice(None)] * 2
      current_entry[0] = plot_state[0]
      current_entry[1] = plot_state[1]

      current_qtable = Qtable[tuple(current_entry)]
      value = np.max(current_qtable)
      values_dict["epoch"].append(epoch)
      values_dict["value"].append(value)

      print("value", value)

  dataframe = pd.DataFrame(data=values_dict)
  seaborn.lineplot(x="epoch", y="value", data=dataframe, ci=100,
                   err_style="band", label="$s_1=$" + str(plot_state[0]) + 
                   "$s_2=" + str(plot_state[1]))

plt.xlabel("Epoch")
plt.ylabel("$V(s)$")
plt.title("$")
if not os.path.exists("../" + project + "/plots/debug"):
  os.makedirs("../" + project + "/plots/debug")
plt.savefig("../"  + project + "/plots/debug/values_evolv.png")
plt.clf()





