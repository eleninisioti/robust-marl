""" Main interface script for executing an experiment.
"""
# ----- generic imports ----
import os
import random
import pickle
import numpy

# ----- project-specific imports -----
from agent import *
from q_agent import *
from romq_agent import *
from node import *
from minimaxq_agent import *
from tools import *
from parser import *

def main(args):

  # ----- set up -----
  # create project sub-directories
  policies_dir = "../projects/" + args.project + "/policies/adversary_" \
                 + args.adversary
  if not os.path.exists(policies_dir):
    os.makedirs(policies_dir)

  plots_dir = "../projects/" + args.project + "/plots/adversary_" + \
              args.adversary + "_attack_type_" + args.attack_type
  if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

  for trial in range(args.trials):

    trial_plots_dir = "../projects/" + args.project + "/trial_" +  str(trial) +\
                      "/adversary_" + args.adversary + "_attack_" + \
                      args.attack_type + "/plots"
    if not os.path.exists(trial_plots_dir):
      os.makedirs(trial_plots_dir)

    trial_data_dir = "../projects/" + args.project + "/trial_" + str(trial) + \
                      "/adversary_" + args.adversary + "_attack_" + \
                      args.attack_type + "/data"
    if not os.path.exists(trial_data_dir):
      os.makedirs(trial_data_dir)

  # process flags
  max_delta = args.eval_attack_prob
  delta_values = [0]
  delta_values.extend(np.arange(start=0.1, stop=max_delta + 0.01, step=0.1))
  args.delta_values = delta_values
  payoffs = {"overflow": args.chigh, "underflow": args.clow,
             "alive": args.utility}

  # change project directory for evaluation
  args.orig_project = args.project
  interm_epochs = {}
  if args.evaluate:
    args.project = args.orig_project + "/eval" + "/adv_" + str(
    args.determ_adv) + "def_" + str(args.determ_execution)

  for trial in range(args.trials):
    interm_epochs[trial] = []

    # set seed
    random.seed(trial)
    np.random.seed(trial)

    # ----- create network of nodes -----
    nodes = []
    if args.topology == "pair":
      nodes = create_pair(network_type=args.network_type,
                          capacity=args.capacity)

    # ----- initialize agents -----
    if args.method == "minimaxQ":

      opponent_idxs_1 = [2]
      opponent_idxs_2 = [1]

      agents = [MinimaxQAgent(nodes=nodes, opp_idxs=opponent_idxs_1,
                              alpha=args.learning_rate, epsilon=args.epsilon,
                              determ_execution=args.determ_execution,
                              gamma=args.discount_factor),
                MinimaxQAgent(nodes=nodes, opp_idxs=opponent_idxs_2,
                              alpha=args.learning_rate, epsilon=args.epsilon,
                              determ_execution=args.determ_execution,
                              gamma=args.discount_factor)]

    elif args.method == "Qlearning":

      agents = [QAgent(nodes=nodes, alpha=args.learning_rate,
                       epsilon=args.epsilon, gamma=args.discount_factor)]

    elif args.method == "RomQ":

      agents = [RomQAgent(nodes=nodes, alpha=args.learning_rate,
                          epsilon=args.epsilon,
                          explore_attack=args.explore_attack,
                          attack_size=args.K,
                          determ_execution=args.determ_execution,
                          gamma=args.discount_factor)]

    if not args.evaluate:

      # ----- main learning phase ------
      performance_train = {"rewards": [], "actions": [], "states": []}

      sample = 0
      episode = 0
      stop_episode = False
      epoch = 0
      control_nodes = []

      for agent in agents:
        control_nodes.extend(agent.control_nodes)

      while sample < (args.train_samples+1):
        
        new_episode = np.floor(sample/args.horizon)

        # start new episode if args.horizon steps have taken place since the
        # previous one or one of the nodes has over-flown
        if (new_episode > episode) or stop_episode:
          episode = episode + 1
          change_episode = True
        else:
          change_episode = False

        system_reset = stop_episode or change_episode

        if system_reset:
          
          # reset all nodes of the MAS
          for agent in agents:
            agent.current_state = [0]*len(agent.state_space)
          for node in control_nodes:
              node.reset()

        # interact with the environment
        actions, rewards, new_states, stop_episode =\
          env_interact(agents=agents, payoffs = payoffs, evaluation=False,
                       attack_size=0, prob_attack=0, attack_type="worst")

        # ----- update agents based on new experience -----
        for idx, agent in enumerate(agents):

          # find opponent actions
          opponent_action = []
          opp_idxs = [node.idx-1 for node in agent.nodes if node not in
           agent.control_nodes]
          for opp in opp_idxs:
            exec_action = actions[opp * 2]
            off_action = actions[opp * 2 + 1]
            opponent_action.extend([exec_action, off_action])

          agent_new_states = new_states
          agent_rewards = rewards
          agent.update(next_state=agent_new_states, reward=agent_rewards,
                       opponent_action=opponent_action)

        performance_train["rewards"].append(rewards)
        performance_train["actions"].append(actions)
        performance_train["states"].append(new_states)

        # save intermediate trained models for evaluation
        if sample%((args.train_samples)/5) == 0:

          # create directory for intermediate episode
          if not os.path.exists("../projects/" + args.project + "/trial_" +
                                str(trial) + "/epoch_" + str(epoch)):
            os.makedirs("../projects/" + args.project + "/trial_" + str(trial) +
                        "/epoch_" + str(epoch))

          # save performance and agents for further analysis
          pickle.dump({"performance": performance_train,
                       "agents": agents},
                      file=open("../projects/" + args.project + "/trial_" +
                                str(trial) + "/epoch_" + str(epoch) +
                           "/train_data" + ".pkl", "wb"))

          interm_epochs[trial].append(epoch)

          # clear data for memory management
          performance_train = {"rewards": [], "actions": [], "states": []}
          epoch += 1

        sample += 1

      # ----- save adversarial policy -----
      adv_policy = find_adversarial_policy(agents, attack_size=args.K)
      if args.determ_adv:
        policies_dir = "../projects/" + args.project + \
                       "/policies/adversary_" + args.adversary + "/determ"
      else:
        policies_dir = "../projects/" + args.project + \
                       "/policies/adversary_" + args.adversary + "/prob"
      if not os.path.exists(policies_dir):
        os.makedirs(policies_dir)
      pickle.dump({"sigma": adv_policy},
                open( policies_dir + "/trial_" + str(trial)+
                      "_adv_policy.pkl", "wb"))

    # ---- main evaluation phase ----
    if args.evaluate:

      if not os.path.exists("../projects/" + args.project):
        os.makedirs("../projects/" + args.project)

      # load adversarial policy
      if args.determ_adv:
        policy_file = "../projects/" + args.orig_project + \
                      "/policies/adversary_" + args.adversary + \
        "/determ/trial_" + str(trial) + "_adv_policy.pkl"
      else:
        policy_file = "../projects/" + args.orig_project + \
                      "/policies/adversary_" + args.adversary + \
                      "/prob/trial_" + str(trial) + "_adv_policy.pkl"

      adv_policy = pickle.load(open(policy_file, "rb"))

      adv_policy = adv_policy["sigma"]

      # load training configuration data
      config_data = pickle.load( open("../projects/" + args.orig_project +
                                     "/config.pkl", "rb"))

      interm_epochs = config_data.interm_epochs
      
      # process flags
      payoffs = {"overflow": args.chigh, "underflow": args.clow,
                 "alive": args.utility}

      # load agents
      if args.evaluate_interm:
        epochs_for_eval = interm_epochs[trial]

      else:
        epochs_for_eval = [interm_epochs[trial][-1]]
        
      # load agents for evaluation  
      episode_dirs = ["../projects/" + args.orig_project + "/trial_" +
                      str(trial) + "/epoch_" + str(epoch) for epoch
                      in epochs_for_eval]

      agents_for_eval = []
      adv_policies_for_eval = []
      for dir in episode_dirs:
        train_data = pickle.load(open(dir + "/train_data.pkl", "rb"))

        agents = []
        for agent in train_data["agents"]:
          agent.determ_execution = args.determ_execution
          agents.append(agent)
        agents_for_eval.append(agents)

        if args.adversarial_interm:
          new_adv_policy = find_adversarial_policy(agents, attack_size=args.K)
          adv_policies_for_eval.append(new_adv_policy)
          pickle.dump({"sigma": new_adv_policy},
                      open(dir + "/adv_policy.pkl", "wb"))
        else:
          adv_policies_for_eval.append(adv_policy)

      for agents_idx, agents in enumerate(agents_for_eval):
        train_epoch = epochs_for_eval[agents_idx]

        print(delta_values)

        for current_delta in delta_values:
          print(current_delta)

          performance_test = {"actions": [], "states": [], "current_states": [],
                              "rewards": [], "durations": []}

          sample = 0
          episode = 0
          stop_episode = False
          duration = 0

          control_nodes = []
          for agent in agents:
            control_nodes.extend(agent.control_nodes)

          while sample < args.eval_samples:
            sample += 1
            duration += 1
            new_episode = np.floor(sample / args.horizon)

            # start new episode if args.horizon steps have taken place since the
            # previous one or one of the nodes has over-flown
            if (new_episode > episode) or stop_episode:
              episode = episode + 1
              change_episode = True

              # update duration
              performance_test["durations"].append(duration)
              duration = 0
            else:
              change_episode = False

            system_reset = stop_episode or change_episode

            if system_reset:
              # reset all nodes of the MAS to zero load
              for node in control_nodes:
                  node.reset()
              for agent in agents:
                agent.current_state = [0] * len(agent.state_space)

            # keep track of current state
            current_state = []
            for node in control_nodes:
              current_state.append(node.load)

            # interact with the environment
            actions, rewards, new_states, stop_episode =\
              env_interact(agents=agents, payoffs=payoffs, evaluation=True,
                           attack_size=args.K, prob_attack=current_delta,
                           current_state=current_state,
                           adversarial_policy=adv_policies_for_eval[agents_idx],
                           attack_type=args.attack_type)

            # update agents based on new experience
            for idx, agent in enumerate(agents):

              # find opponent actions
              opponent_action = []
              opp_idxs = [node.idx - 1 for node in agent.nodes if node not in
                          agent.control_nodes]
              for opp in opp_idxs:
                exec_action = actions[opp * 2]
                off_action = actions[opp * 2 + 1]
                opponent_action.extend([exec_action, off_action])

              agent_new_states = new_states
              agent_rewards = rewards
              agent.update(next_state=agent_new_states, reward=agent_rewards,
                           opponent_action=opponent_action, learn=False)

            performance_test["states"].append(new_states)
            performance_test["actions"].append(actions)
            performance_test["current_states"].append(current_state)
            performance_test["rewards"].append(np.sum(rewards))


          # ----- save evaluation data -----
          test_dir = "../projects/" + args.project + "/trial_" +  str(trial) +\
                     "/epoch_" + str(train_epoch) + "/adversary_" +\
                     args.adversary + "_attack_" + args.attack_type + "/data"
          if not os.path.exists(test_dir):
            os.makedirs(test_dir)

          nodes = []
          for agent in agents:
            nodes.extend(agent.control_nodes)
          pickle.dump({"nodes": nodes, "performance": performance_test},
                      file=open(
            test_dir+"/test_data_" + str(current_delta) + ".pkl", "wb"))

  # ----- keep track of intermediate episodes -----
  if not args.evaluate:
    new_config = args
    # add information about intermediate epochs
    for key, val in interm_epochs.items():
      val = list(set(val))
      interm_epochs[key] = val
    new_config.interm_epochs = interm_epochs
  else:
    new_config = pickle.load(open("../projects/" + args.orig_project +
                              "/config.pkl", "rb"))
    new_config.delta_values = delta_values
    #print(new_config)

  # add logged data to config pickle
  logs = []
  for agent in agents:
    logs.append(agent.log)
  new_config.logs = logs
  #print(new_config)
  pickle.dump(new_config, open("../projects/" + args.project + "/config.pkl",
                            "wb"))


if __name__ == '__main__':

  args = parse_flags()
  main(args)

