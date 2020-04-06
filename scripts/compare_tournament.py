""" This script can be used to compare the performance of different learning
algorithms


"""
import pickle
import matplotlib.pyplot as plt
import seaborn


# ----- set up -----
seaborn.set()
top_dir = "../projects/samples_final"
directories = ["Qlearning2", "minimaxQ","RomQ"]
methods = ["Qlearning", "minimaxQ", "RomQ"] # used for directories
symbols = {"Qlearning": "Q", "minimaxQ": "M", "RomQ": "R"}
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

  for adv_idx, adversary in enumerate(methods):

    directory = directories[method_idx]

    results_file = top_dir + "/" + directory + "/plots/adversary_" + adversary\
                   + "_attack_worst/robust_results.pkl"

    # ----- load performance data -----
    robust_data = pickle.load(open(results_file, "rb"))
    robust = robust_data["robust_trials"]

    seaborn.lineplot(x="delta", y="reward", data=robust,
                   label=r'$\sigma_{}^*(s)$'.format(symbols[adversary]), ax=ax, ci=90,
                   err_style="band")

  ax.set_title((r'$\pi_{}^*(s)$'.format(symbols[method])))

  ax.legend()
#plt.savefig("../plots/robust_method" + method + ".png")

# ---- 1 figure per adversary -----
for adv_idx, adversary in enumerate(methods):
  ax_index = adv_idx + 4

  ax = fig.add_subplot(3, 3, ax_index, sharey = ax)

  for method_idx, method in enumerate(methods):
    directory = directories[method_idx]

    results_file = top_dir + "/" + directory + "/plots/adversary_" + adversary\
                   + \
                "_attack_worst/robust_results.pkl"

    # ----- load performance data -----
    robust_data = pickle.load(open(results_file, "rb"))
    robust = robust_data["robust_trials"]



    seaborn.lineplot(x="delta", y="reward", data=robust,
                   label=r'$\pi_{}^*(s)$'.format(symbols[method]), ax=ax, ci=90,
                   err_style="band")

  ax.set_title((r'$\sigma_{}^*(s)$'.format(symbols[adversary])))
  ax.set_xlabel("Probability of attack, $\delta$")
  ax.set_ylabel("Test performance, $r$")
  ax.legend()
  #plt.savefig("../plots/robust_adversary_" + adversary + ".png")
  #plt.clf()



# ----- each method against itself -----
ax = fig.add_subplot(3, 3, 8)

for method_idx, method in enumerate(methods):

  for adv_idx, adversary in enumerate(methods):

    if adversary == method:
      directory = directories[method_idx]
      results_file = top_dir + "/" + directory + "/plots/adversary_" + \
                     adversary + \
                     "_attack_worst/robust_results.pkl"

      robust_data = pickle.load(open(results_file, "rb"))
      robust = robust_data["robust_trials"]


      seaborn.lineplot(x="delta", y="reward", data=robust,
                       label=r'$\pi_{}^* = \sigma_{}^*(s)$'.format(
                         symbols[adversary], symbols[adversary]), ax=ax, ci=90,
                   err_style="band")

#ax.set_title((r'$\pi_{}^*(s)$'.format(symbols[method])))
ax.set_xlabel("Probability of attack, $\delta$")
ax.set_ylabel("Test performance, $r$")
ax.legend()

plt.savefig("../plots/robust_combined.png")
plt.clf()




