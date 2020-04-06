""" This script can be used to visualise tuning.

The input to the string is a directory that contains different projects,
each one having an experiment_data.pkl file, from where data will be loaded.
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import numpy as np
import seaborn
import sys
from pathlib import Path
import os
sys.path.insert(0,"../source")
from q_agent import *


# ----- set up -----
seaborn.set()
directory = "../projects/" + sys.argv[1]
step = 200

# ---- load data -----
tuning_data = []
_, projects, _ = next(os.walk(directory))
for project in projects:

  # load data
  config = pickle.load(open(directory + "/" + project + "/config.pkl", "rb"))
  alpha = config.learning_rate
  e = config.epsilon
  performance = pickle.load(open(directory + "/" + project +
                                 "/system_reward.pkl", "rb"))

  # plot
  seaborn.lineplot(x="time_steps", y="rewards", data=performance, ci="sd",
                   label="$\\alpha=$" +str(alpha) +", $\\epsilon=$" + str(e))

# save plot
plt.ylabel("System reward, $\\bar{r}$")
plt.xlabel("Episode, $e$")
plt.legend()
plt.savefig(directory  + "/tuning.eps")
plt.clf()


