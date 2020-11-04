""" Main interface script for executing an experiment.
"""
# ----- generic imports ----
import os
import pickle

# ----- project-specific imports -----
from agent import *
from q_agent import QAgent
from romq_agent import RomQAgent
from minimaxq_agent import MinimaxQAgent
from tools import find_adversarial_policy, create_pair, env_interact
from parser import parse_flags


def create_agents(algorithm, nodes):
  """ Create agents that employ a desired reinforcement leanring algorithm.

   Args:
     algorithm (str): name of RL algorithm
     nodes (list of Node): the network nodes

   Returns:
     list of agents
  """
  if algorithm == "minimaxQ":

    opponent_idxs_1 = [2]
    opponent_idxs_2 = [1]

    agents = [MinimaxQAgent(nodes=nodes, opp_idxs=opponent_idxs_1,
                            alpha=args.learning_rate, epsilon=args.epsilon,
                            gamma=args.discount_factor),
              MinimaxQAgent(nodes=nodes, opp_idxs=opponent_idxs_2,
                            alpha=args.learning_rate, epsilon=args.epsilon,
                            gamma=args.discount_factor)]

  elif algorithm == "Qlearning":

    agents = [QAgent(nodes=nodes, alpha=args.learning_rate,
                     epsilon=args.epsilon, gamma=args.discount_factor)]

  elif algorithm == "RomQ":

    agents = [RomQAgent(nodes=nodes, alpha=args.learning_rate,
                        epsilon=args.epsilon,
                        attack_size=args.K,
                        gamma=args.discount_factor)]
  else:
    print("Error: algorithm ", algorithm, " is not implemented.")
    quit()

  return agents

def execute_trial(trial, args, samples, adv_policy = [], evaluation=False,
                  delta=0):

  results_dic = {"rewards": [], "actions": [], "states": [], "overflows": [],
                 "durations": []}
  sample = 0
  episode = 0
  stop_episode = False
  duration = 0
  
  control_nodes = []
  for agent in args.agents:
    control_nodes.extend(agent.control_nodes)

  while sample <= samples:

    # start new episode if args.horizon steps have taken place since the
    # previous one or one of the nodes has over-flown 
    new_episode = np.floor(sample / args.horizon)
    if (new_episode > episode) or stop_episode:
      episode = episode + 1
      change_episode = True

      # update duration
      results_dic["durations"].append(duration)
      duration = 0
    else:
      change_episode = False

    system_reset = stop_episode or change_episode
    if system_reset:
      # reset the state of all agents
      for agent in args.agents:
        agent.current_state = [0] * len(agent.state_space)

      # reset all nodes of the network 
      for node in control_nodes:
        node.reset()

    states = []
    for agent in args.agents:
      for node in agent.control_nodes:
        states.append(node.load)
    results_dic["states"].append(states)

    # interact with the environment
    actions, rewards, states, stop_episode = \
      env_interact(agents=args.agents, payoffs=args.payoffs,
                   evaluation=evaluation, adv_policy=adv_policy,
                   attack_size=args.K, prob_attack=delta,
                   attack_type="worst", current_state=states)

    results_dic["rewards"].append(rewards)
    results_dic["actions"].append(actions)

    if stop_episode:
      results_dic["overflows"].append(states)

    # ----- update agents based on new experience -----
    for idx, agent in enumerate(args.agents):

      # find opponent actions
      opponent_action = []
      opp_idxs = [node.idx - 1 for node in agent.nodes if node not in
                  agent.control_nodes]
      for opp in opp_idxs:
        exec_action = actions[opp * 2]
        off_action = actions[opp * 2 + 1]
        opponent_action.extend([exec_action, off_action])

      def_action = []
      def_idxs = [node.idx - 1 for node in agent.nodes if node in
                  agent.nodes]
      for defe in def_idxs:
        exec_action = actions[defe * 2]
        off_action = actions[defe * 2 + 1]
        def_action.extend([exec_action, off_action])

      agent_states = states
      agent_rewards = rewards
      agent.update(next_state=agent_states, reward=agent_rewards,
                   def_action=def_action, opponent_action=opponent_action,
                   learn=not evaluation)

    # save intermediate trained models for evaluation
    if sample in args.epochs and not evaluation:

      # save performance and agents for further analysis
      pickle.dump({"performance": results_dic,
                   "agents": args.agents},
                  file=open("../projects/" + args.project +
                            "/data/train/trial_" +
                            str(trial) + "/epoch_" + str(sample) +
                            "/data" + ".pkl", "wb"))

      # clear data for memory management
      results_dic = {"rewards": [], "actions": [], "states": [], "overflows":
        [], "durations": []}

    sample += 1
    duration += 1
    
  return results_dic


def main(args):

  # ----- set up -----
  # process input flags
  max_delta = args.eval_attack_prob
  delta_values = [0]
  delta_values.extend(np.arange(start=0.1, stop=max_delta + 0.01, step=0.1))
  args.delta_values = delta_values
  args.payoffs = {"overflow": args.chigh, "underflow": args.clow,
             "alive": args.utility}
  args.epochs = list(range(0, args.train_samples + 1,
                           int(args.train_samples/args.epochs)))


  # create project sub-directories
  policies_dir = "../projects/" + args.project + "/policies" + "/adversary_" \
                 + args.algorithm
  if not os.path.exists(policies_dir):
    os.makedirs(policies_dir)

  plots_dir_train = "../projects/" + args.project + "/plots/train"
  plots_dir_eval = "../projects/" + args.project + "/plots/eval"
  data_dir_train = "../projects/" + args.project + "/data/train"
  data_dir_eval = "../projects/" + args.project + "/data/eval"

  new_dirs = []
  for trial in range(20, args.trials):
    
    new_dirs.append(policies_dir + "/trial_" + str(trial))

    for epoch in args.epochs:
      new_dir = "/trial_" + str(trial) + "/epoch_" + str(epoch)
      new_dirs.append(plots_dir_train + new_dir)
      new_dirs.append(data_dir_train + new_dir)
      
      new_dir = "/trial_" +  str(trial) + "/epoch_" + str(epoch) + "/adv_" + \
                 args.adversary + "_attack_" + args.attack_type 
      new_dirs.append(plots_dir_eval + new_dir)
      new_dirs.append(data_dir_eval + new_dir)

  for dir in new_dirs:
    if not os.path.exists(dir):
      os.makedirs(dir)

  pickle.dump(args, open("../projects/" + args.project + "/config.pkl",  "wb"))
  # ----- simulations take place -----
  for trial in range(20, args.trials):

    # set seed for trial
    random.seed(trial)
    np.random.seed(trial)

    # ----- create network of nodes -----
    nodes = []
    if args.topology == "pair":
      nodes = create_pair(network_type=args.network_type,
                          capacity=args.capacity)

    # ----- create agents -----
    if args.train:
      args.agents = create_agents(args.algorithm, nodes)
      execute_trial(trial, args, samples=args.train_samples)

      # ----- save adversarial policy -----
      adv_policy = find_adversarial_policy(args.agents, attack_size=args.K)
      policy_file = "../projects/" + args.project + \
                    "/policies/adversary_" + args.adversary + \
                    "/trial_" + str(trial) + "/adv_policy.pkl"

      pickle.dump({"sigma": adv_policy},
                  open(policy_file, "wb"))

    # ---- main evaluation phase ----
    if args.evaluate:

      policy_file = policy_file = "../projects/" + args.project + \
                   "/policies/adversary_" + args.adversary +\
                 "/trial_" + str(trial) + "/adv_policy.pkl"

      adv_policy = pickle.load(open(policy_file, "rb"))
      adv_policy = adv_policy["sigma"]

      # load agents
      if args.evaluate_interm:
        epochs_for_eval = args.epochs

      else:
        epochs_for_eval = [args.epochs[-1]]

      for epoch in epochs_for_eval:
        dir = "../projects/" + args.project + "/data/train" + "/trial_" +  \
              str(trial) + "/epoch_" + str(epoch)
        eval_dir = "../projects/" + args.project + "/data/eval" + "/trial_" \
                   +  str(trial) + "/epoch_" + str(epoch) + "/adv_" + \
                 args.adversary + "_attack_" + args.attack_type 
        
        # load trained agents
        train_data = pickle.load(open(dir + "/data.pkl", "rb"))
        agents = []
        nodes = []
        for agent in train_data["agents"]:
          agents.append(agent)
          nodes.extend(agent.control_nodes)
        args.agents = agents

        # evaluate trained agents
        for current_delta in delta_values:  
          test_data = execute_trial(trial, args, evaluation=True, 
                               delta=current_delta, adv_policy=adv_policy,
                                    samples=args.eval_samples)

          pickle.dump({"nodes": nodes, "performance": test_data},
                      file=open(eval_dir + "/data_" + str(current_delta)
                                + ".pkl", "wb"))
  logs = []
  for agent in args.agents:
    args.logs.append(agent.log)
  args["logs"] = logs


if __name__ == '__main__':
  args = parse_flags()
  main(args)

