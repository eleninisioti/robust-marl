""" This script can be used to plot the operation of the agents during an
experiment.


Two identical plots are produced, one refers to the learning and the other to
 the execution time. Each plot consists of 3 sub-plots
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


# ----- set up -----


# ----- load data -----
project = "state_small"
data = pickle.load(open("../projects/" + project + "/experiment_data.pkl","rb"))
data_learn = data[0]
data_exec = data[1]

# ----- plot learning ----
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
turnout = data_learn[0]
axes[0].plot(list(range(len(turnout))), turnout, color="black")
axes[0].set_ylabel("Turnout, $T$")

attack = data_learn[1]
axes[1].plot(list(range(len(attack))), attack, color="black")
axes[1].set_ylabel("Attack size, $K$")

welfare = data_learn[2]
axes[2].plot(list(range(len(turnout))), welfare, color="black")
axes[2].set_ylabel("Social Welfare, $W$")
axes[2].set_yscale("symlog")
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


plt.xlabel("Time step, $t$")
plt.savefig("../projects/" + project + "/plots/operation_learning.eps")
plt.clf()


# ----- plot execution ----
fig, (axes) = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
turnout = data_exec[0]
axes[0].plot(list(range(len(turnout))), turnout, color="black")
axes[0].set_ylabel("Turnout, $T$")

attack = data_exec[1]
axes[1].plot(list(range(len(attack))), attack, color="black")
axes[1].set_ylabel("Attack size, $K$")

welfare = data_exec[2]
axes[2].plot(list(range(len(turnout))), welfare, color="black")
axes[2].set_ylabel("Social Welfare, $W$")
axes[2].set_yscale("symlog")
axes[2].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


plt.xlabel("Time step, $t$")
plt.savefig("../projects/" + project + "/plots/operation_execution.eps")
plt.clf()