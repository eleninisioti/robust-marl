""" This script can be used to compare the performance of different learning
algorithms


"""
import pickle
import matplotlib.pyplot as plt
import seaborn
import sys
import numpy as np
import os
import matplotlib

# ----- set up -----
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

top_dir = "../projects/" + sys.argv[1]
single_plot = int(sys.argv[2])
attack_type = "worst"
directories = ["/Qlearning", "/minimaxQ", "/RomQ"]
methods = ["Qlearning", "minimaxQ", "RomQ"]
adversaries = ["Qlearning","minimaxQ", "RomQ"]
symbols = {"Qlearning": "{Q-learning}", "minimaxQ": "{minimax-Q}", "RomQ":
  "{RoM-Q}"}
if single_plot:
  fig = plt.figure(figsize=(20, 20))

plot_dir = top_dir + "/plots"
if not os.path.exists(plot_dir):
   os.makedirs(plot_dir)


# ----- 1 figure per method -----
for method_idx, method in enumerate(methods):

  ax_index = method_idx + 1

  if single_plot:

    if method_idx == 0:
      ax = fig.add_subplot(3, 3, ax_index)
      ax.set_xlabel("Probability of attack, $\delta$")
      ax.set_ylabel("Test performance, $r$")

    else:
      ax = fig.add_subplot(3, 3, ax_index,  sharey=ax)
      
  else:
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()


  for adv_idx, adversary in enumerate(adversaries):

    directory = directories[method_idx]
    project_dir = top_dir + directory

    results_dir = project_dir + "/data/eval/tournament/adv_" + adversary +\
                  "_attack_" + attack_type

    results_file = results_dir + "/robust_results.pkl"

    # ----- load performance data -----
    robust_data = pickle.load(open(results_file, "rb"))
    robust = robust_data["robust_trials"]
    xbb = np.ndarray.tolist(robust["delta"].values)
    rewardbb = np.ndarray.tolist(robust["reward"].values)
    seaborn.lineplot(x="delta", y="reward", data=robust,
                     label=r'$\sigma_{'r'}^*('r's)$'.format(
      symbols[adversary]), ax=ax, ci=95,
                   err_style="band")

    # plt.plot(x="delta", y="reward", data=robust,
    #                  label=r'$\sigma_{}^*(s)$'.format(symbols[adversary]),
    #                  ax=ax)

  ax.set_title((r'$\pi_{}^*(s)$'.format(symbols[method])))

  ax.legend()

  if not single_plot:
    plt.savefig(plot_dir + "/" + method + "_defender.png", bbox_inches='tight')
    plt.clf()


# ---- 1 figure per adversary -----
for adv_idx, adversary in enumerate(adversaries):
  ax_index = adv_idx + 4

  if single_plot:

    ax = fig.add_subplot(3, 3, ax_index, sharey = ax)

  else:
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes()


  for method_idx, method in enumerate(methods):
    directory = directories[method_idx]

    project_dir = top_dir + directory

    results_dir = project_dir + "/data/eval/tournament/adv_" + adversary +\
                  "_attack_" + attack_type

    results_file = results_dir + "/robust_results.pkl"

    # ----- load performance data -----
    robust_data = pickle.load(open(results_file, "rb"))
    robust = robust_data["robust_trials"]
    seaborn.lineplot(x="delta", y="reward",  data=robust,  label=r'$\pi_{}^*('
                                                      r's)$'.format(
      symbols[method]), ax=ax, ci=95,
                   err_style="band")

    # plt.plot(x="delta", y="reward", data=robust,
    #                label=r'$\pi_{}^*(s)$'.format(symbols[method]))
    x = robust["delta"].values
    reward = robust["reward"].values
    #plt.plot(x,reward,label=r'$\sigma_{}^*(s)$'.format(symbols[adversary]))


  ax.set_title((r'$\sigma_{}^*(s)$'.format(symbols[adversary])))
  ax.set_xlabel("Probability of attack, $\delta$")
  ax.set_ylabel("Test performance, $r$")
  ax.legend()

  if not single_plot:
    plt.savefig(plot_dir + "/" + adversary + "_adversary.png", bbox_inches='tight')
    plt.clf()


# ----- each method against itself -----
if single_plot:
  ax = fig.add_subplot(3, 3, 8)
else:
  fig = plt.figure(figsize=(6, 6))
  ax = plt.axes()

for method_idx, method in enumerate(methods):

  for adv_idx, adversary in enumerate(adversaries):

    if method in adversary:
      directory = directories[method_idx]

      project_dir = top_dir + directory

      results_dir = project_dir + "/data/eval/tournament/adv_" + adversary + \
                    "_attack_" + attack_type

      results_file = results_dir + "/robust_results.pkl"

      robust_data = pickle.load(open(results_file, "rb"))
      robust = robust_data["robust_trials"]
      seaborn.lineplot(x="delta", y="reward",  data=robust,   label=r'$\sigma_{}^*('
                                     r's)$'.format(
        symbols[adversary]), ax=ax, ci=95,
                       err_style="band")

      # plt.plot(x="delta", y="reward", data=robust,
      #                  label=r'$\pi_{}^* = \sigma_{}^*(s)$'.format(
      #                    symbols[adversary], symbols[adversary]))
      x = robust["delta"].values
      reward = robust["reward"].values
      #plt.plot(x, reward, label=r'$\sigma_{}^*(s)$'.format(symbols[adversary]))
    # plt.savefig("temp_" + str(counter) + ".png")
    # plt.clf()
    # counter += 1


ax.set_xlabel("Probability of attack, $\delta$")
ax.set_ylabel("Test performance, $r$")
ax.legend()
plt.savefig(plot_dir + "/robust_combined.png", bbox_inches='tight')
plt.clf()




