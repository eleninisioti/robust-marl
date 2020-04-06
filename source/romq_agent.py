""" Contains the implementation of a Q-learning agent."""

import numpy as np
import random
import math
import itertools

from agent import *
from tools import *

class RomQAgent(Agent):



  def __init__(self, nodes, epsilon=0.01, alpha=0.1, gamma=0.9,
               temperature=0.01, decay=1, explore_attack=False):

    super().__init__(nodes, epsilon, alpha, gamma,
                   temperature)

    self.decay = decay
    self.control_nodes = nodes
    self.defenders = [node.idx for node in nodes]
    self.explore_attack = explore_attack

    self.selection_first = 0
    self.selections_first = []
    self.selection_second = 0
    self.selections_second = []
    self.ties = 0

    # initialize policies
    self.policies = []
    for node in self.nodes:
      policy_action_space = [2, len(node.neighbors)]
      policy_space = self.state_space + policy_action_space
      self.policies.append(np.ones(tuple(policy_space)) / np.sum(
      policy_action_space))

    # initialize value function
    self.V = np.random.uniform(low=0, high=0.0001, size=tuple(self.state_space))
    current_entry = [slice(None)] * len(self.state_space)

    for s1 in range(self.nodes[0].capacity + 2):
      for s2 in range(self.nodes[1].capacity + 2):
        if s1 == 4 or s2 == 4:
          current_entry[0] = s1
          current_entry[1] = s2
          self.V[tuple(current_entry)] = -100


  def update(self, reward, next_state, learn=True):
    """ Updates an agent after interaction with the environment.
    """
    if learn:
      self.update_policy(next_state)
      self.update_Qvalue(reward=reward, next_state=next_state)
      self.alpha *= self.decay

    self.current_state = next_state



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

  def update_policy(self, next_state, retry=False):
    """ Update the policy and corresponding value function of agent.

    The update requires the solution of a linear program.
    """

    minV = np.max(self.Qtable)
    min_policy = []
    
    # get Qvalues for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(next_state):
      current_entry[idx] = el
    Qtable = self.Qtable[tuple(current_entry)]

    if self.explore_attack:
      x = random.uniform(0, 1)
      if (x < self.explore_attack):
        random_adv = random.randint(0,len(self.nodes)-1)
        candidate_advs = [self.nodes[random_adv]]
      else:
        candidate_advs = self.nodes
    else:
      candidate_advs = self.nodes

    count = 0

    for node_idx, node in enumerate(candidate_advs):

      # assume that the current node is the adversary
      adv_idxs = [node.idx]

      # remaining nodes are defenders
      def_idx = [node.idx for node in self.nodes if (node.idx) not in adv_idxs]
      def_idx = def_idx[0] # only works for two nodes


      # opponents' actions need to be in the first dimension



      current_pi = self.policies[def_idx-1][tuple(current_entry)]
      num_a = (len(self.nodes) - len(adv_idxs))*4
      num_o = len(adv_idxs)*4
      Qtable_res = np.reshape(Qtable, (num_o, num_a))
      if 2 in adv_idxs:
        Qtable_minimax = Qtable_res.T
      else:
        Qtable_minimax = Qtable_res

      res = solve_LP(num_a, num_o, Qtable_minimax)
      #res = lp_solve(Qtable_minimax, num_a, num_o)
      success = res.success

      if success:
        lp_policy = np.reshape(res.x[1:], current_pi.shape)

        V = res.x[0]

        if V <= minV:
          minV = V
          min_policy = lp_policy
          min_def = def_idx
          count+=1

      elif not retry:

        return self.update_policy(retry=True)
      else:
        print("Alert : %s" % res.message)

    self.V[tuple(current_entry)] = minV
    self.policies[min_def-1][tuple(current_entry)] = min_policy

    if min_def ==1:
      self.selection_first +=1
      self.selections_first.append(self.selection_first)
      self.selections_second.append(self.selection_second)

    elif min_def ==2:
      self.selection_second +=1
      self.selections_second.append(self.selection_second)
      self.selections_first.append(self.selection_first)

    if count ==2:
      self.ties+=1


    #print(min_def, self.selection_first)

  def greedy_action(self, just_test=False):
    """ Performs  the greedy action according to a probabilistic policy
    """
    # get current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    
    self.current_action = []
    for idx, node in enumerate(self.nodes):
      
      # get node's policy
      policy = self.policies[idx]
      
      # get node's policy for current state
      current_policy = policy[tuple(current_entry)]
      
      # randomly sample policy
      rand = np.random.rand()
  
      flat_pi = np.ndarray.flatten(current_policy)
      cumSumProb = np.cumsum(flat_pi)
  
      action = 0
      while rand > cumSumProb[action]:
        action+=1

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

        self.current_action = current_action

      else:

        self.current_action = list(np.unravel_index(action, policy.shape))

    return self.current_action








