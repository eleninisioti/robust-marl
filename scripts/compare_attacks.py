""" This script can be used to plot and compare turnouts under different
learning configurations. """


import matplotlib.pyplot as plt
import pickle

data_dir = "../projects/qlearning_small_adv"

# load data
alpha_001 = pickle.load(open(data_dir + "/experiment_data001.pkl", "rb"))
alpha_01 = pickle.load(open(data_dir + "/experiment_data01.pkl", "rb"))
alpha_04 = pickle.load(open(data_dir + "/experiment_data04.pkl", "rb"))
alpha_04b = pickle.load(open(data_dir + "/experiment_data04b.pkl", "rb"))

# reduce granularity
alpha_001 = alpha_001[0][0::10]
alpha_01 = alpha_01[0][0::10]
alpha_04 = alpha_04[0][0::10]
alpha_04b = alpha_04b[0][0::10]

# plot data
plt.plot(list(range(len(alpha_001))), alpha_001,
         label=r'$\alpha = 0.01$')
plt.plot(list(range(len(alpha_01))), alpha_01,
         label=r'$\alpha = 0.1$')
plt.plot(list(range(len(alpha_04))), alpha_04,
         label=r'$\alpha = 0.4$, success')
plt.plot(list(range(len(alpha_04b))), alpha_04b,
         label=r'$\alpha = 0.4$, fail')
plt.xlabel("Time, $t(*10)$")
plt.ylabel("Turnout, W")
plt.legend(loc="lower right")
plt.savefig(data_dir + "/plots/compare.eps")
plt.clf()





