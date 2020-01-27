""" Main interface script for executing an experiment.

"""
import argparse
import matplotlib.pyplot as plt
import os
from bar import *
from agent import *
from erev_agent import *

def main(args):

  # ----- set up -----
  if not os.path.exists("../projects/" + args.project + "/plots"):
    os.makedirs("../projects/" + args.project + "/plots")

  # initialize bar
  bar = Bar(args.capacity, args.technique)

  # initialize agents
  agents = []
  for ag_idx in range(args.nagents):
    if args.technique == "Erev":
      agent = ErevAgent()

    elif args.technique == "Arthur":
      agent = Agent(id=ag_idx, bar=bar, nagents=args.nagents,
                        horizon=args.horizon)

    agents.append(agent)

  # ----- main learning phase ------
  turnouts = []
  optimism_stats = [] # what percentage believes that the bar will not be busy

  for iter in range(args.iterations):

    # all agents decide whether to go
    turnout = 0
    actions = []
    for agent in agents:
      action = agent.decide()
      turnout += action
      actions.append(action)


    # update agents
    optimists = 0
    for idx, agent in enumerate(agents):
      action = actions[idx]
      optimists += agent.update(bar.reward(action, turnout))

    # keep info for plotting
    turnouts.append(turnout)
    optimism_stats.append(optimists/args.nagents)

  # plot turnout with time
  plt.plot(list(range(args.iterations)), turnouts)
  plt.xlabel("Time, $T$")
  plt.ylabel("Turnout, $W$")
  plt.savefig("../projects/" + args.project + "/plots/turnout.eps")
  plt.clf()

  # plot optimism with time
  plt.plot(list(range(args.iterations)), optimism_stats)
  plt.xlabel("Time, $T$")
  plt.ylabel("Optimism, $O$")
  plt.savefig("../projects/" + args.project + "/plots/optimism.eps")
  plt.clf()



if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--nagents',
                        help='Number of agents',
                        type=int,
                        default=100)

  parser.add_argument('--horizon',
                      help='Number of time steps in the past agents look at.',
                      type=int,
                      default=5)

  parser.add_argument('--capacity',
                      help='Capacity of the bar',
                      type=int,
                      default=60)

  parser.add_argument('--iterations',
                      help='Number of learning iterations',
                      type=int,
                      default=100)

  parser.add_argument('--project',
                      help='Name of project',
                      type=str,
                      default="temp")

  parser.add_argument('--technique',
                      help='Technique employed to solve the problem. Choose '
                           'between Erev and Arthur',
                      type=str,
                      default="Erev")

  args = parser.parse_args()
  main(args)

