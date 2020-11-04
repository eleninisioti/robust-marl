""" Contains the implementation of a RomQ-learning agent."""

# ----- generic imports -----
import numpy as np
import random
import math
import itertools

# ----- project-specific imports -----
from agent import Agent
from tools import solve_LP

class RomQAgent(Agent):
  """ An agent that uses RomQ-learning.

  Nodes are divided into defenders and opponents, and a policy is learnt over
  the joint action space of defenders. By creating a different MinimaxQAgent
  for each node, where all others are opponents, we can learn a different
  policy for each node.

  Attributes:
    control_nodes (list of Node): the nodes whose actions are controlled
    determ_execution (bool): indicates whether execution of policies during
       deployment should be deterministic
    V (array of float): an array of dimension state_space
    explore_attack (float): the probability with which random adversaries,
      instead of worst-case, are chosen
    attack_size (int): number of adversaries considered to compute target
      policy
  """

  def __init__(self, nodes, epsilon, alpha, gamma, attack_size):
    """ Initialize RoM-Q agent.

    Args:

      attack_size (int): number of adversaries considered to compute target
      policy
    """

    super().__init__(nodes=nodes, epsilon=epsilon, alpha=alpha, gamma=gamma)

    self.control_nodes = nodes

    # initialize policies
    self.policies = []
    for node in self.nodes:
      policy_action_space = [2, len(node.neighbors)]
      policy_space = self.state_space + tuple(policy_action_space)
      self.policies.append(np.ones(tuple(policy_space)) / np.sum(
      policy_action_space))

    # initialize value function
    self.V = np.random.uniform(low=0, high=0.0001, size=tuple(self.state_space))

    self.attack_size = attack_size

    # initialize data for logging
    self.log["defenders"] = []

  def update(self, reward, next_state, def_action=[], opponent_action=[],
             learn=True):
    """ Updates an agent after interaction with the environment.

    Args:
      reward (list of float): contains individual node rewards
      next_state (list of int): contains individual node loads
      learn (bool): indicates whether the Q-table will be updated
    """
    if learn:
      self.update_qvalue(reward=reward, next_state=next_state,
                         def_action=def_action)
      self.update_policy()

    self.current_state = next_state


  def compute_target(self, next_state):
    """ Computes the value of the target policy in the temporal difference
    learning update.

    Args:
      next_state (list of int): contains individual node loads
    """
    return self.V[tuple(next_state)]

  def update_policy(self, retry=False):
    """ Update the policy and corresponding value function of agent.

    The update requires the solution of a linear program

    Args:
      next_state (list of int): contains individual node loads
      retry (bool): indicates if we'll try to solve the LP for a second time.
    """

    # get q-values for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    qtable_state = self.Qtable[tuple(current_entry)]

    minV = np.max(self.Qtable)
    min_policy = np.array([])
    min_def = 1

    candidate_advs = self.nodes
    # ----- search for worst-case adversarial selection of nodes-----
    for node_idx, adv_node in enumerate(candidate_advs):

      # assume that the current node is the adversary
      adv_idxs = [adv_node.idx]

      # remaining nodes are defenders
      def_idx = [node.idx for node in self.nodes if (node.idx) not in adv_idxs]
      def_idx = def_idx[0] # only works for two nodes

      num_o = len(adv_idxs) * 4
      num_a = (len(self.nodes) - len(adv_idxs)) * 4

      # ----- swap axes in Q-table so that adversaries are first -----
      map = {}
      count = 0
      opp_nodes = [adv_node]
      for opp in opp_nodes:
        pos = self.nodes.index(opp)
        map[count] = pos
        map[count + 1] = pos + 1
        count += 1

      for key, value in map.items():
        qtable = np.swapaxes(qtable_state, key, value)

      qtable = np.reshape(qtable, (num_o, num_a))

      # keep only eligible actions
      opp_state = []
      for opp in adv_idxs:
        opp_state.append(self.current_state[opp - 1])

      def_state = self.current_state[def_idx-1]
      non_admissible = {0: [3, 2, 1], 1: [3]}
      if opp_state[0] in non_admissible.keys():
        inval_actions = non_admissible[opp_state[0]]
        num_o = len(adv_idxs)* (4-len(inval_actions))
        for inval in inval_actions:
          qtable = np.delete(qtable, inval, 0)

      if def_state in non_admissible.keys():
        inval_actions = non_admissible[def_state]
        num_a = 1*(4-len(inval_actions))
        for inval in inval_actions:
          qtable = np.delete(qtable, inval, 1)

      # solve linear program
      res = solve_LP(num_a, num_o, qtable)

      if res.success:
        current_pi = self.policies[def_idx - 1][tuple(current_entry)]

        if len(res.x[1:]) != num_a:
          # if some of the actions were invalid we need to map the result
          # appropriately
          lp_policy = np.zeros((num_a,))
          count = 0
          for i in range(num_a):
            if i not in inval_actions:
              lp_policy[i] = res.x[1:][count]
              count += 1
        else:
          lp_policy = res.x[1:]
        lp_policy = np.reshape(lp_policy, current_pi.shape)
        V = res.x[0]

        # keep adversarial selection with minimum value
        if V <= minV:
          minV = V
          min_policy = lp_policy
          min_def = def_idx

      elif not retry:
        return self.update_policy(retry=True)

      else:
        print("Alert : %s" % res.message)

        # if optimisation fails twice, keep the previous policy
        min_policy = self.policies[min_def-1][tuple(current_entry)]

    self.V[tuple(current_entry)] = minV

    self.policies[min_def-1][tuple(current_entry)] = min_policy

  def onpolicy_action(self):
    """ Performs the on-policy action.

    During training, actions are selected based on the probabilistic policy.
    During deployment, actions are again selected probabilistically or the
    greedy action is selected (depending on self.determ_execution).
    """
    # get q-values for current state
    current_entry = [slice(None)] * len(self.state_space)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    qcurrent = self.Qtable[tuple(current_entry)]

    # ----- execute deterministic policy during deployment -----
    max_actions_flat = np.argmax(qcurrent)
    current_action = list(np.unravel_index(max_actions_flat,
                                           qcurrent.shape))
    self.current_action = current_action

    return self.current_action








