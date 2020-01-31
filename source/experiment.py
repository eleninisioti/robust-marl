""" Main interface script for executing an experiment.

"""
import argparse
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pickle

from bar import *
from agent import *
from erev_agent import *
from qlearning_agent import *

def attack(agents, capacity):
  """ Simulates an attack by adversaries.

  Adversaries choose the agents to attack and perform actions on their behalf.
  """
  # ----- choose victims -----
  # choose the K agents with the current highest probability of staying at home
  attacked = [0]*len(agents)
  for idx, agent in enumerate(agents):
    if agent.pure_eq[0][-1] > 0.95:
      attacked[idx] = 1
    if sum(attacked) >= args.nadvs:
      break

  # if not enough victims, choose agents that are not confident that they
  # will go to the bar
  if sum(attacked) < args.nadvs:
    for idx, agent in enumerate(agents):
      if agent.pure_eq[1][-1] < 0.5:
        attacked[idx] = 1
      if sum(attacked) >= args.nadvs:
        break

  # if still not enough victims, randomly choose some agents
  if sum(attacked) < args.nadvs:
    rand_victims = random.choices(list(range(len(agents))), k=args.nadvs-sum(
      attacked))
    for idx in rand_victims:
      attacked[idx] = 1

  for idx, agent in enumerate(agents):
    if attacked[idx] == 1:
      agent.attack(force_go=True)

  return attacked

def main(args):

  # ----- set up -----
  if not os.path.exists("../projects/" + args.project + "/plots/agents"):
    os.makedirs("../projects/" + args.project + "/plots/agents")

  # initialize bar
  bar = Bar(capacity=args.capacity, technique=args.technique,
            min_payoff=args.minpayoff)

  # initialize agents
  agents = []
  for ag_idx in range(args.nagents):
    if args.technique == "Erev":
      #rate = random.uniform(0,1)
      rate=1
      agent = ErevAgent(min_payoff=args.minpayoff, rate=rate)

    elif args.technique == "Arthur":
      agent = Agent(id=ag_idx, bar=bar, nagents=args.nagents,
                        horizon=args.horizon)

    elif args.technique == "Qlearning":
      agent = QlearningAgent(min_payoff=args.minpayoff)

    agents.append(agent)

  # ----- main learning phase ------
  turnouts = []
  optimism_stats = [] # what percentage believes that the bar will not be busy
  attacked = [0]*len(agents)

  for iter in range(args.iterations):

    # all agents decide whether to go
    turnout = 0
    actions = []

    if iter == args.safe_attack:
      attacked = attack(agents, args.capacity)

    # if iter == (args.safe_attack +1):
    #   attacked = [0] * len(agents)
    for idx, agent in enumerate(agents):
      action = agent.decide(attacked[idx])
      turnout += action
      actions.append(action)

    # update agents
    optimists = 0
    for idx, agent in enumerate(agents):
      action = actions[idx]
      optimists += agent.update(bar.reward(action, turnout), attacked[idx])

    # keep info for plotting
    turnouts.append(turnout)
    optimism_stats.append(optimists/args.nagents)

  # find meand and variance of error
  optimal = args.capacity
  error = [np.abs(optimal - t) for t in turnouts[int(0.5*len(
    turnouts)):]]
  print(turnouts[int(0.5*len(
    turnouts)):])
  print(optimal, error)
  print("Mean error: ", np.mean(error), " Mean variance: ", np.var(error))


  # plot optimism with time
  plt.plot(list(range(args.iterations)), optimism_stats)
  plt.xlabel("Time, $T$")
  plt.ylabel("Optimism, $O$")
  plt.savefig("../projects/" + args.project + "/plots/optimism.eps")
  plt.clf()

  # plot pure equilibria with time
  for idx, agent in enumerate(agents):
    stay_eq = agent.pure_eq[0]
    go_eq = agent.pure_eq[1]
    plt.plot(list(range(args.iterations)), stay_eq, label="Stay")
    plt.plot(list(range(args.iterations)), go_eq, label="Go")
    plt.xlabel("Time, $T$")
    plt.ylabel("Probability of pure equilbirum")
    plt.legend(loc="lower right")
    plt.title("Agent " + str(idx))
    if attacked[idx] == 1:
      plt.savefig("../projects/" + args.project +
                  "/plots/agents/attacked_agent_" + str(
      idx) + ".eps")
    else:
      plt.savefig("../projects/" + args.project +
                    "/plots/agents/agent_" + str(
          idx) + ".eps")

    plt.clf()

  # plot histogram of pure equilibria at the end of learning
  stay_agents = 0
  go_agents = 0
  for idx, agent in enumerate(agents):
    stay_agents += agent.pure_eq[0][-1]
    go_agents += agent.pure_eq[1][-1]
  stay_agents = stay_agents/len(agents)
  go_agents = go_agents/len(agents)
  plt.bar([0,1], [stay_agents, go_agents], tick_label=["Stay", "Go"])
  plt.ylabel("Percentage of agents with pure Equilbrium")
  plt.savefig("../projects/" + args.project + "/plots/eq_bar.eps")
  plt.clf()

  exec_turnouts = []
  attacked = [0] * len(agents)
  if not args.online:

    # Q-learning agents do not explore during execution
    if args.technique == "Qlearning":
      for agent in agents:
        agent.epsilon = 0
        agent.temperature = 0

    # execute learned policy
    for exec_iter in range(args.exec_iterations):

      # perform an attack
      if exec_iter == args.attack:
        attacked = attack(agents, args.capacity)

      # execute policy
      exec_turnout = 0

      for idx, agent in enumerate(agents):
        action = agent.decide(attacked[idx])
        exec_turnout += action

      exec_turnouts.append(exec_turnout)

    plt.plot(list(range(args.exec_iterations)), exec_turnouts,
             label="Execution")

  # plot turnout with time
  plt.plot(list(range(args.iterations)), turnouts, label="Learning")
  plt.xlabel("Time, $T$")
  plt.ylabel("Turnout, $W$")
  plt.legend(loc="lower right")
  plt.savefig("../projects/" + args.project + "/plots/turnout_total.eps")
  plt.clf()


  # save data for reproducibility
  pickle.dump([turnouts, exec_turnouts], file=open("../projects/" +
                                    args.project + "/experiment_data.pkl","wb"))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--online',
                      help='Learn online. Othewise, learns offline and '
                           'executes policy.',
                      default=False,
                      action="store_true")

  parser.add_argument('--nagents',
                        help='Number of agents',
                        type=int,
                        default=100)

  parser.add_argument('--minpayoff',
                        help='Minimum value of payoffs',
                        type=int,
                        default=0)


  parser.add_argument('--nadvs',
                      help='Number of adversaries',
                      type=int,
                      default=1)

  parser.add_argument('--attack',
                      help='Time step of attack',
                      type=int,
                      default=9999999999)

  parser.add_argument('--safe_attack',
                      help='Time step of attack during learnign',
                      type=int,
                      default=9999999999)

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
                      default=500)

  parser.add_argument('--exec_iterations',
                      help='Number of execution iterations',
                      type=int,
                      default=500)

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

