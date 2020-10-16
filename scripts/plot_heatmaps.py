""" This script can be used to plot the heatmap of state visits.

Heatmaps indicate number of state visits using colour and also contain the
greedy action for each state. Actions are represented as colour-coded arrows,
 one for each node, with rightward (s1) and upward (s2) arrows indicating the
off-loading of packets and leftward (s1) and downward (s2) arrows indicating
the execution of tasks.

Output: depending on the input flags, it produces train_grid and test_grid
files under the project's directory.
"""
# ----- generic imports -----
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn
import os
import sys

# ----- project-specific imports -----
sys.path.insert(0,"../source")
from q_agent import *

# ----- set up -----
# parse input
project = sys.argv[1] # name of project directory
only_final = int(sys.argv[2]) # if True, only plots for final epoch
evaluate = int(sys.argv[3]) # if True, only plots for evaluation
adversary = sys.argv[4] # name of adversarial policy
attack_type = sys.argv[5] # type of attack

# set up for plots
seaborn.set()
params = {'legend.fontsize': 'large',
         'figure.figsize': (6, 6),
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

# load project data
config = pickle.load(open("../projects/" + project + "/config.pkl",  "rb"))
trials = list(range(config.trials))
#trials = list(range(3))
delta_values = config.delta_values


# create a different plot for each trial and epoch
for trial in trials:

  if only_final:
    epochs = [config.epochs[-1]]
  else:
    epochs = config.epochs

  for epoch in epochs:

    # load training data
    train_dir = "../projects/" + project + "/data/train/trial_" +  str(trial) +\
                "/epoch_" + str(epoch)
    data_train = pickle.load(open(train_dir + "/data.pkl", "rb"))

    qtables = []
    agents = data_train["agents"]
    for agent in agents:
      qtables.extend([agent.Qtable])

    datasets = {}
    if evaluate:

      # load data about performance during evaluation
      for delta in delta_values:

        eval_dir = "../projects/" + project + "/data/eval/trial_" + str(
          trial) + "/epoch_" + str(epoch) + "/adv_" + adversary + "_attack_" +\
                   attack_type
        datasets[delta] = pickle.load(open(eval_dir + "/data_" +
                                           str(delta) + ".pkl",  "rb"))

      plot_train = False # do not plot state visits during training
    else:
      plot_train = True

      # load data about performance during training
      datasets[0] = data_train
      epochs = [config.epochs[-1]]

    for key, value in datasets.items():
      delta = key
      data_test = value
      performance = data_test["performance"]

      # heatmap of state visits
      actions = performance["actions"]
      states = performance["states"]

      iters = len(actions)
      visits = np.zeros(shape=(config.capacity+2, config.capacity+2))

      for iter in range(iters):
        next_state = states[iter]

        # keeping track of the next states in order to show over-flows
        visits[next_state[0], next_state[1]] += 1

      # show over-flows
      overflows = performance["overflows"]
      for el in overflows:
        visits[el[0], el[1]] += 1

      cmap = 'Blues'
      img = plt.imshow(visits.T, cmap=plt.get_cmap(cmap),
                       origin="lower", norm=matplotlib.colors.LogNorm())

      # ----- plot optimal policy -----
      for s1 in range(config.capacity+1):
        for s2 in range(config.capacity+1):

          current_entry = [slice(None)] * 2
          current_entry[0] = s1
          current_entry[1] = s2

          # find actions of first agent
          Qtable = qtables[0]
          Qtable_current = Qtable[tuple(current_entry)]
          actions1 = np.argmax(Qtable_current)
          actions1 = list(np.unravel_index(actions1, Qtable_current.shape))

          actions_temp = np.amax(Qtable_current)
          print(np.where(Qtable_current == actions_temp))
          # only minimax-Q has two Q-tables
          if len(qtables) > 1:
            Qtable = qtables[1]
            Qtable_current = Qtable[tuple(current_entry)]
            actions2 = np.argmax(Qtable_current)
            actions2 = list(np.unravel_index(actions2,
                                             Qtable_current.shape))
          else:
            actions2 = actions1
          current_actions = [actions1[0], actions1[1], actions2[2], actions2[3]]

          # actions of first node
          serve_action_1 = current_actions[0]
          send_action_1 = current_actions[1]

          # actions of second node
          serve_action_2 = current_actions[2]
          send_action_2 = current_actions[3]

          # draw arrow (when a node executes, it reduces its own state by 1,
          # when it off-loads, it increase the other node's state by 1 )
          plt.arrow(s1, s2, - serve_action_1/2, send_action_1/2,
                    color="green", linewidth=2,
                    head_width=0.05, head_length=0.1, length_includes_head=True)

          # draw arrow
          plt.arrow(s1, s2, send_action_2/2, - serve_action_2/2,
                    color="orange", linewidth=2,
                    head_width=0.05, head_length=0.1, length_includes_head=True)

      # make a color bar
      plt.colorbar(img, cmap=plt.get_cmap(cmap))

      # design plot
      plt.xlabel(r'$\boldsymbol{s_1}$', color="green")
      plt.ylabel(r'$\boldsymbol{s_2}$', color="orange")
      plt.title("$\delta=$" + str(round(delta,2)))

      # highlight over-flow area
      plt.axvline(x=3.5,color="red", ymax=0.8)
      plt.axhline(y=3.5,color="red", xmax=0.8)

      if plot_train:
        plot_dir = "../projects/" + project + "/plots/train/trial_" + str(
          trial) + "/epoch_" + str(epoch)

      else:
        plot_dir = "../projects/" + project + "/plots/eval/trial_" + str(
          trial) + "/epoch_" + str(epoch) + "/adv_" + adversary + "_attack_" +\
                   attack_type

      if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

      if plot_train:
        plot_file = plot_dir + "/grid.png"
      else:
        plot_file = plot_dir + "/grid_" + str(delta) + ".png"

      print(plot_file)

      plt.savefig(plot_file, bbox_inches='tight')
      plt.clf()

