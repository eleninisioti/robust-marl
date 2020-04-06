""" This script can be used to plot the performance of intermediate training
steps.


"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib
import numpy as np
import seaborn
import sys
from pathlib import Path
import pandas as pd
import tikzplotlib

import os
sys.path.insert(0,"../source")
from q_agent import *


# ----- set up -----
seaborn.set()
project_dir = "../projects/" + sys.argv[1]
config = pickle.load(open(project_dir + "/config.pkl", "rb"))
trials = config.trials
final_episode = config.episodes-1

# ----- plot visits heatmaps -----
for trial in range(trials):
  trial_dir = project_dir + "/trial_" + str(trial)
  trial_plots_dir = trial_dir + "/plots"

  # load Qtable
  train_data = pickle.load(open(trial_dir + "/train_data.pkl", "rb"))
  agents = train_data["agents"]


  selections_first = []
  selections_second = []
  for agent in agents:
    selections_first.extend(agent.selections_first)
    selections_second.extend(agent.selections_second)
    print(agent.ties)

  plt.plot(list(range(len(selections_first))), selections_first,
           label="Node 1")
  plt.plot(list(range(len(selections_second))), selections_second,
           label="Node 2")

  plt.xlabel("Sample")
  plt.ylabel("Defender ID")
  plt.legend()
  plt.savefig(trial_plots_dir + "/defenders_evol.png")
  plt.clf()
