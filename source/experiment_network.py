""" Main interface script for executing an experiment.

"""
import argparse
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import pickle
import itertools

from bar import *
from agent import *
from erev_agent import *
from qlearning_agent import *
from node import *

def attack(agents, attack_type="random"):
  """ Simulates an attack by adversaries.

  Adversaries choose the agents to attack and perform actions on their behalf.
  """
  # ----- choose victims -----
  # choose the K agents with the current highest probability of staying at home
  attackers = [0]*len(agents)

  # random attack
  if attack_type == "random":
    attackers = random.sample(agents, args.K)

  # worst-case attack (picks partition that mimizes Q-value function

  elif attack_type == "worst":
    actions = []
    for agent in agents:
      actions.extend(agent.current_actions)

  return attackers

def main(args):

  # ----- set up -----
  if not os.path.exists("../projects/" + args.project + "/plots"):
    os.makedirs("../projects/" + args.project + "/plots")

  # create nodes/agents
  nodes = []

  # create network topology
  if args.topology == "ring":
    counter = 1
    for idx in range(1,args.N+1):
      left = idx - 1
      if left < 1:
        left = args.N
      right = idx + 1
      if right > args.N:
        right = 1
      neighbors = [0, left, right]

      #define capacity and costs for this node
      capacity = args.capacity # all nodes equal
      costs = {left: args.cost, right: args.cost}
      quick_node = False
      nodes.append(Node(capacity=capacity, neighbors=neighbors, idx=counter, \
                                                           costs=costs,
                        quick_node=quick_node))
      counter +=1

  elif args.topology == "star":
    # TODO: needs debugging
    counter = 0
    neighbors = list(range(len(nodes) - 1))
    capacity = args.capacity
    costs = {}
    for node in neighbors:
      costs[node] = args.cost

    nodes.append(Node(capacity, neighbors, idx=counter, costs=costs))
    for idx in range(1, args.N):
      counter += 1
      neighbors = [0]
      capacity = args.capacity
      costs = {0: args.cost}
      nodes.append(Node(capacity, neighbors, idx=counter, costs=costs))

  elif args.topology == "pair":
    neighbors = [0,2]
    costs =  {0:0, 2:1}
    node = Node(capacity=args.capacity, neighbors=neighbors, idx=1, \
                      costs=costs, quick_node=False)
    nodes.append(node)
    neighbors = [0, 1]
    costs = {0:0, 1:1}
    node = Node(capacity=args.capacity, neighbors=neighbors, idx=2, \
                costs=costs, quick_node=False)
    nodes.append(node)

  if args.decentralized:
    agents = []
    for ag_idx in range(args.N):
      agent = QlearningAgent(nodes=[nodes[ag_idx]], K=args.K,
                             prob_attack=args.PK,robust=not(args.classical))
      agents.append(agent)
  else:
    agents = [QlearningAgent(nodes=nodes, K=args.K, prob_attack=args.PK,
                             robust=not(args.classical)) ]

  performance = {"rewards":[], "actions": [], "states": []}

  # ----- main learning phase ------
  for episode in range(args.episodes):

    # reset nodes
    for agent in agents:
      nodes = agent.nodes
      for node in nodes:
        node.reset()

    for iter in range(args.horizon):

      # decide whether an attack takes place during training
      x = random.uniform(0, 1)
      if x < args.learn_attack_prob:
        attack = args.K
      else:
        attack = 0

      # find actions performed by agents
      actions = []
      for idx, agent in enumerate(agents):
        actions.extend(agent.decide(attack, explore=args.explore))

      if attack > 0:
        actions = attack(agents)


      for agent in agents:
        nodes = agent.nodes
        recipients = []

        # first find all transmissions
        for node in nodes:
          node_idx = node.idx
          num_neighbors = len(node.neighbors)
          recipient = actions[node_idx-1]
          recipients.append(recipient)

        # find rewards and new states
        new_states = []
        rewards = []
        arrivals = []
        departures = []
        for node in nodes:
          node_idx = node.idx
          costs = node.costs
          recipient = recipients[node_idx-1]



          # find how many packets the node has sent to others
          if recipient:
            departures.append(1)
          else:
            departures.append(0)


          # find how many agents sent a packet to this node
          arrivals.append(sum([1 for idx,recipient in enumerate(recipients) if
                          ((recipient==node_idx) and (nodes[idx].load>0))]))

        for node in nodes:
          arr = arrivals[node.idx-1]
          dep = departures[node.idx-1]

          underflow, overflow = node.transition(arr, dep)

          if overflow:
            reward = - args.chigh
          elif underflow:
            reward = - args.clow
          else:
            reward = args.utility

          # add cost of transmission
          if recipient:
            transmission_cost = costs[recipient]
            reward -= transmission_cost
          rewards.append(reward)
          new_states.append(node.load)

      # update agents based on transitions and rewards
      for idx, agent in enumerate(agents):
        agent.update(new_state=new_states[(idx*len(agent.nodes)):(idx*len(
          agent.nodes) + len(agent.nodes))],
                     reward=rewards[(idx*len(agent.nodes)):(idx*len(
          agent.nodes) + len(agent.nodes))],
                     current_t=iter)
      performance["rewards"].append(rewards)
      performance["states"].append(new_states)
      performance["actions"].append(actions)

      pickle.dump([performance, args], file=open("../projects/" +
                                      args.project + "/experiment_data.pkl","wb"))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--N',
                        help='Number of nodes/agents',
                        type=int,
                        default=100)

  parser.add_argument('--clow',
                        help='Punishment for underflow',
                        type=int,
                        default=7)

  parser.add_argument('--chigh',
                        help='Punishment for overflow',
                        type=int,
                        default=10)

  parser.add_argument('--utility',
                      help='Utility for executing a packet',
                      type=int,
                      default=5)

  parser.add_argument('--cost',
                      help='Cost of transmitting on an edge',
                      type=int,
                      default=0)


  parser.add_argument('--K',
                      help='Number of adversaries',
                      type=int,
                      default=1)

  parser.add_argument('--PK',
                      help='Probability of attack for Q-learning',
                      type=float,
                      default=0.01)


  parser.add_argument('--learn_attack_prob',
                      help='Probability of attack during learning.',
                      type=float,
                      default=0.1)

  parser.add_argument('--exec_attack_prob',
                      help='Probability of attack during execution.',
                      type=float,
                      default=0.1)

  parser.add_argument('--capacity',
                      help='Capacity of nodes',
                      type=int,
                      default=5)

  parser.add_argument('--horizon',
                      help='Number of iterations in episode',
                      type=int,
                      default=8)

  parser.add_argument('--episodes',
                      help='Number of learning episode',
                      type=int,
                      default=1000)

  parser.add_argument('--exec_iterations',
                      help='Number of execution iterations',
                      type=int,
                      default=500)

  parser.add_argument('--project',
                      help='Name of project',
                      type=str,
                      default="temp")

  parser.add_argument('--explore',
                      help='Exploration technique to use.',
                      type=str,
                      default="egreedy")

  parser.add_argument('--decentralized',
                      help='.Learning is decentralized',
                      default=False,
                      action="store_true")

  parser.add_argument('--classical',
                      help='.Indicates whether classical Q-learning will be '
                           'used. Otherwise, robust',
                      default=False,
                      action="store_true")

  parser.add_argument('--topology',
                      help='The network topology. Choose between Ring and '
                           'star.',
                      type=str,
                      default="ring")



  args = parser.parse_args()
  main(args)

