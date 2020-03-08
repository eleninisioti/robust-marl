""" Contains the implementation of a generic temporal difference learning
agennt."""

import numpy as np
import random
import itertools

class Agent():

  def __init__(self, nodes, K, robust, epsilon=0.01, alpha=0.1,
               gamma=0.9, temperature=0.01, delta=0, explore="egreedy"):

    # ----- initialize Qtable -----
    # define state space
    space_shape = []
    for node in nodes:
      state_range = node.capacity
      space_shape.append(state_range+1)

    # define action space
    action_space = []
    for node in nodes:
      action_space.append(2)
      action_space.append(len(node.neighbors))

    space_shape.extend(action_space)
    self.space_shape = space_shape
    self.Qtable = np.random.random_sample(tuple(space_shape))

    # initialize states and actions
    self.nodes = nodes
    self.current_state = [0] * len(nodes)
    self.current_action = [0,0] * len(nodes)

    # learning parameters
    self.epsilon = epsilon # exploration rate
    self.alpha = alpha # learning rate
    self.gamma = gamma # discount factor
    self.temperature = temperature # for Boltzmann
    self.delta = delta # probability of attack
    self.explore = explore

  def find_adversarial_actions(self, K):
    """
    Finds adversarsial actions for current state and Qtable
    """
    current_entry = [slice(None)] * len(self.space_shape)
    for idx, el in enumerate(self.current_state):
      current_entry[idx] = el
    Qcurrent = self.Qtable[tuple(current_entry)]
    max_actions_flat = np.argmax(Qcurrent)
    max_actions = np.unravel_index(max_actions_flat, Qcurrent.shape)

    # find all possible subsets of K attackers in N nodes
    indexes = list(range(len(self.nodes)))
    attackers_partitions = list(itertools.combinations(indexes, K))

    Qadv = np.max(Qcurrent)
    actions = max_actions
    worst_partition = attackers_partitions[0]
    for partition in attackers_partitions:

      # find subset of defenders
      defenders = [node for node in indexes if node not in partition]

      # find actions of defenders (each node has one action for serving and
      # one for sending)
      # defend_actions = []
      # for idx, el in enumerate(max_actions):
      #   if (idx%2 == 0) and (idx not in partition):
      #     defend_actions[idx] = max_actions[idx]
      #     defend_actions[idx+1] = max_actions[idx + 1]

      # find subset of Qtable when defenders are maximizing
      indcs = [slice(None)] * len(max_actions)
      for defend in defenders:
        defend_ind = defend*2
        indcs[defend_ind] = max_actions[defend_ind]
        indcs[defend_ind+1] = max_actions[defend_ind+1]
      attacker_Qvalues = Qcurrent[tuple(indcs)]

      # minimize remaining actions over attackers
      min_actions = np.argmin(attacker_Qvalues)
      min_actions = np.unravel_index(min_actions, attacker_Qvalues.shape)

      # consolidate in single list (TODO: can be simplified)
      partition_actions = list(max_actions)
      counter = 0
      for idx, action in enumerate(partition_actions):
        if idx in partition:
          partition_actions[idx] = min_actions[counter]
          counter += 1

      # Qvalue of current partition
      Qvalue = Qcurrent[tuple(partition_actions)]
      if Qvalue < Qadv:
        Qadv = Qvalue
        actions = partition_actions
        worst_partition = partition

    # isolated IDs and actions of chosen partition
    adv_actions = {}
    for idx in worst_partition:
      trans_idx = idx*2
      absolute_idx = self.nodes[idx].idx
      adv_actions[absolute_idx] = [actions[trans_idx],actions[trans_idx + 1]]
    return adv_actions

  def execute_policy(self):
    """ Choose the action to perform based on the policy.
    """
    pass


  def update(self):
    """ Updates an agent after interaction with the environment.
    """
    pass

  def compute_target(self):
    """ Computes the value of the target policy in the temporal difference
    learning update.
     """
    pass

  def update_Qvalue(self, reward, next_state):
    """ Updates the Qvalue function of the agent using temporal difference
    learning  """
    current_entry = tuple(self.current_state + self.current_action)
    Qcurrent = self.Qtable[current_entry]
    entry = [slice(None)] * len(self.space_shape)
    entry[:len(next_state)] = next_state

    Qnext = self.Qtable[tuple(entry)]
    target = self.compute_target(Qnext)

    self.Qtable[current_entry] = \
      Qcurrent + self.alpha * (sum(reward) + self.gamma * target -
                               Qcurrent)


  def perform_action(self, attack_actions, exploration):
    """ Perform an action, depending on the learned policy, the
    exploration scheme and attacks present in the environment.
    ."""
    # ----- e-greedy -----
    if self.explore == "egreedy":
      x = random.uniform(0, 1)

      if ((x < self.epsilon) and exploration): # random move
        self.current_action = []
        for idx, node in enumerate(self.nodes):
          self.current_action.append(random.randint(2))
          self.current_action.append(random.choice(node.neighbors))
      else: # greedy move

        # isolate Qtable for current state
        current_entry = [slice(None)] * len(self.space_shape)
        for idx, el in enumerate(self.current_state):
          current_entry[idx] = el
        Qcurrent = self.Qtable[tuple(current_entry)]

        # find greedy action
        max_actions_flat = np.argmax(Qcurrent)
        self.current_action = list(np.unravel_index(max_actions_flat,
                                                Qcurrent.shape))


    # in case of an attack, implement attackers' actions

    attacked_nodes = []
    for key, value in attack_actions.items():
      nodes_keys = [node.idx for node in self.nodes]
      if key in nodes_keys:
        attacked_nodes.append(key)
    attacked_nodes.sort()
    for node in attacked_nodes:
      value = attack_actions[node]
      self.current_action[(node-1)*2] = value[0]
      self.current_action[(node-1)*2+1] = value[1]

    return self.current_action



