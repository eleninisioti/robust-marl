""" This script can be used to visualise robustness.

The input to the string is a directory that contains different projects,
each one corresponding to a different probability of attack and
having a train_data.pkl file, from where data will be loaded.
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import numpy as np
import seaborn
import sys
from pathlib import Path



# ----- set up -----
seaborn.set()
directory = "../projects/" + sys.argv[1]


# ---- load data -----
# find all projects in directory
performance = []
deltas = []
for path in Path(directory).rglob('test_data.pkl'):
    path = path.absolute()
    data = pickle.load(open(path,"rb"))
    config = data[1]

    # get performance
    performance_test= data[0]
    deltas.append(data[2])
    rewards = performance_test["rewards"]
    iters = len(rewards)
    total_rewards = 0
    for iter in range(iters):
        system_reward = 0
        for node in range(config.N):
            system_reward += rewards[iter][node]
        total_rewards += system_reward

    performance.append(total_rewards)

# reorder data
order = np.argsort(deltas)
deltas = np.sort(deltas)
performance_re = [0]*len(performance)
for idx, el in enumerate(performance):
    performance_re[idx] = performance[order[idx]]

# ----- plotting -----
# temporary values
performance_Qlearning = [20000,18000, 16000, 14000, 12000, 10000, 8000,
                         6000, 4000, 2000]
performance_minimax = [20000,18000, 18000, 18000, 18000, 10000, 8000,
                         6000, 4000, 2000]
performance_romq = [20000]*len(performance_Qlearning)
plt.plot(deltas, performance_Qlearning, label="Qlearning")
plt.plot(deltas, performance_minimax, label="MinimaxQ")
plt.plot(deltas, performance_romq, label="RomQ")

plt.ylabel("Cum reward, $r$")
plt.xlabel("Probability of attack, $\delta$")
plt.title("Robustness analysis")
plt.legend(loc="lower left")
plt.savefig(directory + "/robustness.eps")
plt.clf()

