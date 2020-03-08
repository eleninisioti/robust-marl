""" Main interface script for executing an experiment.

"""
import argparse
import os
import random
import pickle

from agent import *
from q_agent import *
from romq_agent import *
from node import *

def perform_attack(agents, attack_size, attack_type="worst"):
  """ Simulates an attack by adversaries.

  Adversaries choose the agents to attack and perform actions on their behalf.
  """
  # ----- choose victims -----
  # choose the K agents with the current highest probability of staying at home
  attackers = [0]*len(agents)

  # random attack
  if attack_type == "random":
    attackers = random.sample(nodes, args.K)
    attack_actions = {}
    for item in attackers:
      serve = random.randint(0,1)
      send = random.choice(agents[item].neighbors)
      attack_actions[item] = [serve, send]

  # worst-case attack (picks partition that mimizes Q-value function

  elif attack_type == "worst":
    actions = []
    for agent in agents:
      actions.append(agent.find_adversarial_actions(attack_size))
    attack_actions = {}
    for action in actions:
      for key,value in action.items():
        attack_actions[key] = value

  return attack_actions

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
    costs =  {0:0, 2:3}
    node = Node(capacity=args.capacity, neighbors=neighbors, idx=1, \
                      costs=costs, serve_cost=8, quick_node=False)
    nodes.append(node)
    neighbors = [0, 1]
    costs = {0:0, 1:3}
    node = Node(capacity=args.capacity, neighbors=neighbors, idx=2, \
                costs=costs, serve_cost=1, quick_node=False)
    nodes.append(node)


  # ----- initialize agents -----
  if args.decentralized:
    agents = []
    for ag_idx in range(args.N):
      agent = QAgent(nodes=[nodes[ag_idx]], prob_attack=args.PK,robust=not(args.classical))
      agents.append(agent)
  else:
    agents = [QAgent(nodes=nodes, K=args.K, prob_attack=args.PK,
                             robust=not(args.classical)) ]

  # ----- main learning phase ------
  performance_train = {"rewards":[], "actions": [], "states": [], "overflows":
    [], "underflows": []}
  for episode in range(args.episodes):
    print("episode", episode)
    stop_episode = False

    # reset nodes
    for agent in agents:
      nodes = agent.nodes
      for node in nodes:
        node.reset()

    for iter in range(args.horizon):

      if stop_episode:
        break

      # decide whether an attack takes place during training
      x = random.uniform(0, 1)
      if x < args.learn_attack_prob:
        attack = args.K
      else:
        attack = 0

      attack_actions = {}
      if attack > 0:
        attack_actions = attack(agents)

      # find actions performed by agents
      actions = []
      for idx, agent in enumerate(agents):
        actions.extend(agent.execute_policy(attack_actions))

      # ----- interaction with the environment -----
      for agent in agents:
        nodes = agent.nodes
        recipients = []

        # first find all transmissions
        served = []
        for node in nodes:
          node_idx = node.idx
          served.append(actions[(node_idx-1)*2])
          recipient = actions[node_idx*2-1]
          # map recipient index to actual node
          recipient = node.neighbors[recipient]
          recipients.append(recipient)

        # find rewards and new states
        new_states = []
        rewards = []
        arrivals = []
        departures = []
        overflows = []
        underflows = []
        for node in nodes:
          node_idx = node.idx
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
          costs = node.costs

          recipient = recipients[node.idx - 1]

          underflow, overflow = node.transition(arr, dep, served[node.idx-1])

          if overflow:
            reward = - args.chigh
            stop_episode = True
          elif underflow:
            reward = - args.clow
          else:
            reward = args.utility


          # add cost of transmission
          if recipient:
            transmission_cost = costs[recipient]
            reward -= transmission_cost
          if served[node.idx-1]:
            reward -= node.serve_cost

          # add cost of execution
          rewards.append(reward)
          new_states.append(node.load)
          overflows.append(overflow)
          underflows.append(underflow)
      # update agents based on transitions and rewards
      for idx, agent in enumerate(agents):
        agent.update(next_state=new_states[(idx*len(agent.nodes)):(idx*len(
          agent.nodes) + len(agent.nodes))],
                     reward=rewards[(idx*len(agent.nodes)):(idx*len(
          agent.nodes) + len(agent.nodes))])
      performance_train["rewards"].append(rewards)
      performance_train["states"].append(new_states)
      performance_train["actions"].append(actions)
      performance_train["overflows"].append(overflows)
      performance_train["underflows"].append(underflows)

  # ---- main testing phase ----
  performance_test = {"rewards":[], "actions": [], "states": [],
                      "overflows": [], "underflows": []}

  for episode in range(args.test_episodes):
    stop_episode = False

    # reset nodes
    for agent in agents:
      nodes = agent.nodes
      for node in nodes:
        node.reset()

    for iter in range(args.horizon):
      if stop_episode:
        break

      # decide whether an attack takes place during testing
      x = random.uniform(0, 1)
      if x < args.exec_attack_prob:
        attack_size = args.K
      else:
        attack_size = 0

      attack_actions = {}
      if attack_size > 0:
        attack_actions = perform_attack(agents=agents,attack_size = attack_size)

      # find actions performed by agents
      actions = []
      for idx, agent in enumerate(agents):
        actions.extend(agent.execute_policy(attack_actions, exploration=False))

      # ----- interaction with the environment -----
      for agent in agents:
        nodes = agent.nodes
        recipients = []

        # first find all transmissions
        served = []
        for node in nodes:
          node_idx = node.idx
          served.append(actions[(node_idx - 1) * 2])
          recipient = actions[node_idx * 2 - 1]
          # map recipient index to actual node
          recipient = node.neighbors[recipient]
          recipients.append(recipient)

        # find rewards and new states
        new_states = []
        rewards = []
        arrivals = []
        departures = []
        overflows = []
        underflows = []
        for node in nodes:
          node_idx = node.idx
          recipient = recipients[node_idx - 1]

          # find how many packets the node has sent to others
          if recipient:
            departures.append(1)
          else:
            departures.append(0)

          # find how many agents sent a packet to this node
          arrivals.append(
            sum([1 for idx, recipient in enumerate(recipients) if
                 ((recipient == node_idx) and (nodes[idx].load > 0))]))

        for node in nodes:
          arr = arrivals[node.idx - 1]
          dep = departures[node.idx - 1]
          costs = node.costs

          recipient = recipients[node.idx - 1]
          underflow, overflow = node.transition(arr, dep,
                                                served[node.idx - 1])

          if overflow:
            reward = - args.chigh
            stop_episode = True
          elif underflow:
            reward = - args.clow
          else:
            reward = args.utility

          # add cost of transmission
          if recipient:
            transmission_cost = costs[recipient]
            reward -= transmission_cost
          if served[node.idx - 1]:
            reward -= node.serve_cost

          # add cost of execution
          rewards.append(reward)
          new_states.append(node.load)
          overflows.append(overflow)
          underflows.append(underflow)

      # update agents based on transitions and rewards
      for idx, agent in enumerate(agents):
        agent.update(
          next_state=new_states[(idx * len(agent.nodes)):(idx * len(
            agent.nodes) + len(agent.nodes))], reward=None, learn=False)

      performance_test["rewards"].append(rewards)
      performance_test["states"].append(new_states)
      performance_test["actions"].append(actions)
      performance_test["overflows"].append(overflows)
      performance_test["underflows"].append(underflows)

  pickle.dump([performance_train, performance_test, args], file=open(
    "../projects/" +  args.project + "/experiment_data.pkl","wb"))


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
                        default=20)

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
                      default=2)

  parser.add_argument('--PK',
                      help='Probability of attack for Q-learning',
                      type=float,
                      default=0.01)


  parser.add_argument('--learn_attack_prob',
                      help='Probability of attack during learning.',
                      type=float,
                      default=0)

  parser.add_argument('--exec_attack_prob',
                      help='Probability of attack during execution.',
                      type=float,
                      default=0)

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

  parser.add_argument('--test_episodes',
                      help='Number of testing episode',
                      type=int,
                      default=10000)

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

