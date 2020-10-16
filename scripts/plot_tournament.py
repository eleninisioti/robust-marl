""" This script can be used to compare the performance of different learning
algorithms


"""
import pickle
import matplotlib.pyplot as plt
import seaborn
import sys
import numpy as np

counter = 0
adv_determ = int(sys.argv[1])
def_determ = int(sys.argv[2])
attack_type = "worst"

# ----- set up -----
seaborn.set()
top_dir = "../projects/debug"
directories = ["/Qlearning_A", "/minimaxQ_A", "/RomQ_A"]
methods = ["Qlearning", "minimaxQ", "RomQ"]
adversaries = ["Qlearning","minimaxQ", "RomQ"]
symbols = {"Qlearning": "Q", "minimaxQ": "M", "RomQ": "R", "minimaxQ":
  "M", "RomQ": "R"}
fig = plt.figure(figsize=(20, 20))

# ----- 1 figure per method -----
for method_idx, method in enumerate(methods):

  ax_index = method_idx + 1

  if method_idx == 0:
    ax = fig.add_subplot(3, 3, ax_index)
    ax.set_xlabel("Probability of attack, $\delta$")
    ax.set_ylabel("Test performance, $r$")

  else:
    ax = fig.add_subplot(3, 3, ax_index,  sharey=ax)

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
      symbols[adversary]), ax=ax, ci=90,
                   err_style="band")

    # plt.plot(x="delta", y="reward", data=robust,
    #                  label=r'$\sigma_{}^*(s)$'.format(symbols[adversary]),
    #                  ax=ax)

  ax.set_title((r'$\pi_{}^*(s)$'.format(symbols[method])))

  ax.legend()
  # plt.savefig("temp_" + str(counter) + ".png")
  # counter +=1
  # plt.clf()

#plt.savefig("../plots/robust_method" + method + ".png")

# ---- 1 figure per adversary -----
for adv_idx, adversary in enumerate(adversaries):
  ax_index = adv_idx + 4

  ax = fig.add_subplot(3, 3, ax_index, sharey = ax)

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
      symbols[method]), ax=ax, ci=90,
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
  # plt.savefig("temp_" + str(counter) + ".png")
  # plt.clf()
  # counter += 1

# ----- each method against itself -----
ax = fig.add_subplot(3, 3, 8)

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
        symbols[adversary]), ax=ax, ci=90,
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
plt.savefig("../plots/robust_combined.png")
plt.clf()




