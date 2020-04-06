""" Contains the implementation of a MinimaxQ-learning agent."""

import numpy as np
import random
import math
import itertools

from agent import *
from tools import *

class MinimaxQAgent(Agent):

  illcond = 0 # static counter for measuring illconditioned matrices


  def __init__(self, nodes, epsilon=0.01, alpha=0.1, decay=1,
               gamma=0.7, temperature=0.01, adv_idxs=[]):

    super().__init__(nodes, epsilon, alpha, gamma, temperature)
    self.decay = decay
    self.advs_idxs = adv_idxs # absolute indexes of adversaries

    idxs = [node.idx for node in nodes]
    self.defenders =[idx for idx in idxs if idx not in adv_idxs]
    self.control_nodes = [node for node in self.nodes if node.idx in
                                                        self.defenders]
    self.policy_action_space = []
    for idx in self.defenders:
      self.policy_action_space.append(2)
      self.policy_action_space.append(len(nodes[idx-1].neighbors))

    self.action_space = []
    for idx in idxs:
      self.action_space.append(2)
      self.action_space.append(len(nodes[idx - 1].neighbors))

    self.policy_space = self.state_space + self.policy_action_space
    self.policies = np.ones(tuple(self.policy_space))/np.sum(
      self.policy_action_space)
    self.V = np.zeros(tuple(self.state_space))

    current_entry = [slice(None)] * len(self.state_space)


    for s1 in range(self.nodes[0].capacity+2):
      for s2 in range(self.nodes[1].capacity+2):
        if s1==4 or s2==4:
          current_entry[0]=s1
          current_entry[1]=s2
          self.V[tuple(current_entry)] = -100



  def update(self, reward, next_state, opponent_action, learn=True):
    """ Updates an agent after interaction with the environment.
    """
    # map opponent action to non-absolute idxs
    trans_action = []
    for idx, action in enumerate(opponent_action):
      if idx%2 == 0:
        trans_action.append(action) # serve actions
      else:
        # get opponent idx
        opp_idx = self.advs_idxs[int(idx/2)]

        # get the node
        opp_node = [node for node in self.nodes if node.idx == opp_idx]
        opp_node = opp_node[0]

        # get its neighbors
        opp_neighbs = opp_node.neighbors

        position = [pos for pos, neighb in enumerate(opp_neighbs) if
                    neighb==action]

        position = position[0]

        trans_action.append(position)

    if 1 in self.advs_idxs: #only works for 2 agents

      self.current_action =  trans_action + self.current_action

    else:
      self.current_action =  self.current_action + trans_action

    if learn:
      self.update_Qvalue(reward=reward, next_state=next_state)
      self.update_policy()
      self.alpha *= self.decay

    self.current_state = next_state


  # def update_policy_deterministic(self):
  #   """
  #   This is a temporary function, for testing whether LP is wrong.
  #   """
  #   # isolate Qtable for current state
  #   current_entry = [slice(None)] * len(self.state_space)
  #   for idx, el in enumerate(self.current_state):
  #     current_entry[idx] = el
  #   Qcurrent = self.Qtable[tuple(current_entry)]
  #
  #   if 2 in self.advs_idxs:
  #     defender_dims = [0,1]
  #   else:
  #     defender_dims = [2,3]
  #
  #   max_actions_flat = np.argmax(Qcurrent)
  #
  #   # current_action = list(np.unravel_index(max_actions_flat,
  #   #                                            Qcurrent.shape))
  #
  #
  #   Qdefend = np.amax(Qcurrent, axis=tuple(defender_dims))
  #
  #   # find greedy action for attackers
  #   Qattacked = np.min(Qdefend)
  #
  #
  #   best_actions = np.where(Qcurrent == Qattacked)
  #   current_action = []
  #   for dim in best_actions:
  #     #current_action.append(np.random.choice(dim)) # break ties arbitrarily
  #     current_action.append(dim[0]) # take the first occurence
  #
  #   defend_action = []
  #   for node_idx, action in enumerate(current_action):
  #     if node_idx in defender_dims:
  #       defend_action.append(action)
  #
  #   # compute deterministic policies for each action column
  #   current_pi = []
  #   for node in self.control_nodes:
  #     node_action_space = [2]
  #     node_action_space.extend([len(node.neighbors)])
  #     current_pi.append(np.zeros(tuple(node_action_space)))
  #
  #   current_pi = current_pi[0] # only when one control node
  #   updated = False
  #   for idx, action in enumerate(defend_action[::2]):
  #     current_pi[action, defend_action[idx + 1]] = 1
  #     updated=True
  #
  #   if not updated:
  #     print("oops")
  #
  #   current_pi = current_pi/np.sum(current_pi)
  #
  #   current_entry = [slice(None)] * len(self.state_space)
  #   for idx, el in enumerate(self.current_state):
  #     current_entry[idx] = el
  #
  #   self.policies[tuple(current_entry)] = current_pi



  def update_policy(self, retry=False):
    """ Update the policy and corresponding value function of agent.

    The update requires the solution of a linear program.
    """

    # get Qtable for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    Qtable = self.Qtable[tuple(current_entry)]


    # make sure that the first dimension are the actions of attackers and the
    # second the actions of defenders. At initialisation, node 1 is the first
    # dimension and node 2 the second


    num_a = (len(self.nodes) - len(self.advs_idxs)) * 4
    num_o = len(self.advs_idxs) * 4
    Qtable_res = np.reshape(Qtable, (num_o, num_a))

    if 2 in self.advs_idxs:
      Qtable_minimax = Qtable_res.T
    else:
      Qtable_minimax = Qtable_res

    res = solve_LP(num_a, num_o, Qtable_minimax)
    current_pi = self.policies[tuple(current_entry)]

    if res.success:
      self.policies[tuple(current_entry)] = np.reshape(res.x[1:],
                                                     current_pi.shape)
      self.V[tuple(current_entry)] = res.x[0]

    elif not retry:
      return self.update_policy(retry=True)

    else:
      print("Alert : %s" % res.message)



  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    # entry = [slice(None)] * len(self.state_space)
    # entry[:len(next_state)] = next_state
    #
    # Qnext = self.Qtable[tuple(entry)]
    # return np.max(Qnext)
    return self.V[tuple(next_state)]


  # def greedy_action(self):
  #   """ Finds the greedy action based on the deterministic policy.
  #   """
  #
  #   # isolate Qtable for current state
  #   current_entry = [slice(None)] * len(self.state_space)
  #   for idx, el in enumerate(self.current_state):
  #     current_entry[idx] = el
  #   Qcurrent = self.Qtable[tuple(current_entry)]
  #
  #   # find greedy action
  #   max_actions_flat = np.argmax(Qcurrent)
  #   defender_dims = []
  #
  #   if 2 in self.advs_idxs:
  #     defender_dims = [0,1]
  #   else:
  #     defender_dims = [2,3]
  #   # for node_pos, node in enumerate(self.nodes):
  #   #   if node in self.control_nodes:
  #   #
  #   #     defender_dims.append(node_pos*2)
  #   #     defender_dims.append(node_pos*2 + 1)
  #
  #   current_action = list(np.unravel_index(max_actions_flat,
  #                                              Qcurrent.shape))
  #   defend_action = []
  #   attack_action = []
  #   for node_idx, action in enumerate(current_action):
  #     if node_idx in defender_dims:
  #       defend_action.append(action)
  #     else:
  #       attack_action.append(action)
  #
  #
  #   self.current_action = defend_action

    # bypass
    # self.current_action = list(np.unravel_index(max_actions_flat,
    #                                           Qcurrent.shape))

  def greedy_action(self, just_test):
    """ Performs greedy probabilistic action
    """
    # get state-specific policy
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    policy = self.policies[tuple(current_entry)]

    rand = np.random.rand()

    flat_pi = np.ndarray.flatten(policy)
    cumSumProb = np.cumsum(flat_pi)

    action = 0
    while rand > cumSumProb[action]:
      action += 1


    # only used to debug and just for testing
    #just_test = False
    if just_test:
      current_entry = [slice(None)] * len(self.state_space)
      for idx, el in enumerate(self.current_state):
        current_entry[idx] = el
      Qcurrent = self.Qtable[tuple(current_entry)]

      # find greedy action
      max_actions_flat = np.argmax(Qcurrent)
      current_action = list(np.unravel_index(max_actions_flat,
                                                  Qcurrent.shape))

      if 2 in self.advs_idxs:
        defender_dims = [0,1]
      else:
        defender_dims = [2,3]

      defend_action = []
      for node_idx, action in enumerate(current_action):
        if node_idx in defender_dims:
          defend_action.append(action)
      self.current_action = defend_action
    else:

      self.current_action = list(np.unravel_index(action, policy.shape))

    return self.current_action
