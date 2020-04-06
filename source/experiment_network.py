""" Main interface script for executing an experiment.

"""
import argparse
import os
import random
import pickle
import numpy

from agent import *
from q_agent import *
from doubleq_agent import *
from romq_agent import *
from node import *
from minimaxq_agent import *
from sarsa_agent import *
from tools import *

def main(args):

  # ----- set up -----
  
  # create project directories
  # interm_episodes = list(range(args.episodes*args.horizon))[0::(int(
  #   args.episodes*args.horizon / 20) )]
  # interm_episodes.append(args.episodes-1)
  #interm_episodes = interm_episodes[1:]

  for trial in range(args.trials):
    trial_plots_dir = "../projects/" + args.project + "/trial_" +  str(trial) +\
                      "/adversary_" + args.adversary + "/plots"

    if not os.path.exists(trial_plots_dir):

      os.makedirs(trial_plots_dir)

    # for episode in interm_episodes:
    #   episode_dir = "../projects/" + args.project + "/trial_" +  str(trial) +\
    #                 "/episode_" + str(episode) + "/adversary_" + \
    #                 args.adversary + "_attack_" + args.attack_type

      # if not os.path.exists(episode_dir + "/plots"):
      #   os.makedirs(episode_dir + "/plots")
      #
      # if not os.path.exists(episode_dir + "/data"):
      #   os.makedirs(episode_dir + "/data")

    policies_dir = "../projects/" + args.project + "/policies/adversary_"\
                         + args.adversary

    if not os.path.exists(policies_dir):

      os.makedirs(policies_dir)

    max_delta = args.exec_attack_prob
    delta_values = [0]
    delta_values.extend(np.arange(start=0.1, stop=max_delta + 0.01, step=0.1))

    args.delta_values = delta_values


  # ----- create network of nodes -----
  nodes = []
  if args.topology == "ring":
    nodes = create_ring_topology(N=args.N, capacity=args.capacity,
                                cost=args.cost)

  elif args.topology == "star":
    nodes = create_star_topology(N=args.N, capacity=args.capacity,
                                cost=args.cost)

  elif args.topology == "pair":
    nodes = create_pair(type=args.network_type, capacity=args.capacity)

  if not args.execute_only:

    interm_episodes = {}
  for trial in range(args.trials):
    if not args.execute_only:
      interm_episodes[trial] = []
    # set seed
    random.seed(trial)
    np.random.seed(trial)

    # ----- initialize agents -----
    if args.method == "minimaxQ":

      opponent_idxs_1 = [2]
      opponent_idxs_2 = [1]

      agents = [MinimaxQAgent(nodes=nodes, adv_idxs=opponent_idxs_1,
                              alpha=args.learning_rate, epsilon=args.epsilon),
                MinimaxQAgent(nodes=nodes, adv_idxs=opponent_idxs_2,
                              alpha=args.learning_rate, epsilon=args.epsilon)]

    elif args.method == "Qlearning":

      agents = [QAgent(nodes=nodes, alpha=args.learning_rate,
                       epsilon=args.epsilon,
                       adjust_parameters=args.adjust_parameters )]

    elif args.method == "SARSA":

      agents = [SarsaAgent(nodes=nodes, alpha=args.learning_rate,
                       epsilon=args.epsilon,
                           adjust_parameters=args.adjust_parameters)]

    elif args.method == "DoubleQ":

      agents = [DoubleAgent(nodes=nodes, alpha=args.learning_rate,
                       epsilon=args.epsilon,
                           adjust_parameters=args.adjust_parameters)]

    elif args.method == "RomQ":

      agents = [RomQAgent(nodes=nodes, alpha=args.learning_rate,
                          epsilon=args.epsilon,
                          explore_attack=args.explore_attack)]

    if args.execute_only:

      # test intermediate trained agents
      if args.test_intermediate:

        # find all intermediate files


        agents_for_test = []
        #config_data = pickle.load( open("../projects/" + args.project +
        #                                "/config.pkl", "rb"))

        #interm_episodes = config_data.interm_episodes
        #episodes_for_test = interm_episodes[trial]
        #episodes_for_test = episodes_for_test[1:]
        if args.method == "RomQ":
          temp_episodes = [82000, 84000, 88000, 84000, 94000, 84000, 92000,
                           84000]
        elif args.method == "minimaxQ":
          temp_episodes = [10000, 10000, 10000, 10000,10000, 10000,10000]
        else:
          temp_episodes = [args.episodes]*args.trials

        interm_episodes = []
        for trial_idx, final_episode in enumerate(temp_episodes):

          trial_episodes = np.arange(0, final_episode+1, 2000)
          interm_episodes.append(trial_episodes)
        interm_episodes = interm_episodes[trial]

        episodes_for_test = interm_episodes
        episode_dirs = ["../projects/" + args.project + "/trial_" +
                        str(trial) + "/episode_" + str(episode) for episode
                        in episodes_for_test]

        for dir in episode_dirs:

          train_data = pickle.load(open(dir + "/agents.pkl", "rb"))

          agents_for_test.append(train_data["agents"])


      else:
        if args.method == "RomQ":
          temp_episodes = [82000, 84000, 88000, 84000, 94000, 84000, 92000,
                           84000]
        elif args.method == "minimaxQ":
          temp_episodes = [10000, 10000, 10000, 10000,10000, 10000,10000]
        else:
          temp_episodes = [args.episodes]*args.trials
        # load saved Qtables
        train_data = pickle.load(open("../projects/" + args.project +
                                      "/trial_" + str(trial) + "/episode_" +
                                      str(temp_episodes[trial]) +
                                      "/train_data.pkl", "rb"))

        agents = train_data["agents"]
        agents_for_test = [train_data["agents"]]
        episodes_for_test = [temp_episodes[-1]]
        test_trials = list(range(args.trials))

    else:
      episodes_for_test = [args.episodes - 1]

      # ----- main learning phase ------

      performance_train = {"rewards": [], "actions": [], "states": []}

      sample= 0
      episode = 0
      stop_episode = False

      while sample < args.episodes*args.horizon:
        sample +=1
        new_episode = np.floor(sample/args.horizon)

        if (new_episode > episode) or stop_episode:
          episode = episode + 1
          change_episode = True
          print("episode is: ", str(episode), " samples are:", sample)
        else:
          change_episode = False


        step_rewards = []
        system_reset = (stop_episode) or (change_episode)

        if system_reset:
          print("resetting nodes due to:", stop_episode)
          # reset nodes
          for agent in agents:
            nodes = agent.nodes
            for node in nodes:
              node.reset()
            agent.current_state= [0]*len(agent.state_space)

        # interact with the environment
        actions, rewards, new_states, stop_episode =\
          env_interact(agents=agents, chigh=args.chigh, clow=args.clow,
                       utility=args.utility, exploration=True, K=0,
                       delta=0, attack_type="")


        # update agents based on new experience
        for idx, agent in enumerate(agents):

          if args.method == "minimaxQ":
            agent_rewards = rewards
            agent_new_states = new_states

            if 2 in agent.advs_idxs:
              opponent_action= actions[2:]
            else:
              opponent_action = actions[0:2]

            agent.update(next_state=agent_new_states, reward=agent_rewards,
                         opponent_action=opponent_action)
          else:
            agent_new_states = new_states[(idx * len(agent.nodes)):
                                          (idx * len(agent.nodes) +
                                           len(agent.nodes))]

            agent_rewards = rewards[(idx * len(agent.nodes)):(idx * len(
              agent.nodes) + len(agent.nodes))]

            agent.update(next_state=agent_new_states,
                         reward=agent_rewards)

        performance_train["rewards"].append(rewards)
        performance_train["actions"].append(actions)
        performance_train["states"].append(new_states)
        step_rewards.append(np.sum(rewards))



        # save intermediate trained models for testing
        #print(episode%(args.episodes/10), episode)
        if (episode%((args.episodes)/5) == 0):

          if not os.path.exists("../projects/" + args.project + "/trial_" +
                                str(trial) + "/episode_" + str(episode)):

            os.makedirs("../projects/" + args.project + "/trial_" + str(trial) +
                        "/episode_" + str(episode))



          # save agents for reloading Qtables, policies
          pickle.dump({"agents": agents, "episode": episode,
                       "episode_samples": sample},
                      file=open("../projects/" + args.project + "/trial_" +
                                str(trial) + "/episode_" + str(episode) +
                                "/agents.pkl", "wb"))

          pickle.dump({"performance_train": performance_train , "agents": agents},
                      file=open("../projects/" + args.project +
                      "/trial_" + str(trial) + "/episode_" + str(episode) +
                           "/train_data" + ".pkl", "wb"))
          interm_episodes[trial].append(episode)



          # each file should only contains the episodes after the end of the
          # previous file
          performance_train = {"rewards": [], "actions": [], "states": []}


    # ---- main testing phase ----

    # find optimal adversarial policy
    if not args.execute_only:
      agents_for_test = [agents]
    test_trials = list(range(args.trials))

    if trial in test_trials:

      if args.execute_only:

        if args.adversary not in ["randoma"]:

          adv_policy = pickle.load(open("../projects/" + args.project + "/policies/adversary_"\
                           + args.adversary + "/trial_" + str(trial)+
                           "_adv_policy.pkl", "rb"))

          adv_policy = adv_policy["sigma"]
        else:
          adv_policy = 0
      else:

        adv_policy = find_adversarial_policy(agents, attack_size=args.K)

        pickle.dump({"sigma": adv_policy},
                    open("../projects/" + args.project + "/policies/adversary_"\
                         + args.adversary + "/trial_" + str(trial)+
                         "_adv_policy.pkl", "wb"))

      for agents_idx, agents in enumerate(agents_for_test):
        episode_train = episodes_for_test[agents_idx]

        for current_delta in delta_values:

          performance_test = {"episode_rewards": [], "actions": [],
                              "states": [], "current_states": [],
                              "sample_rewards": [],
                           "episodes_duration": []}

          for episode in range(args.test_episodes):
            stop_episode = False
            step_rewards = []
            duration = 0

            # reset nodes
            for agent in agents:
              nodes = agent.nodes
              for node in nodes:
                node.reset()

            for iter in range(args.horizon):

              if stop_episode:
                break

              duration += 1

              current_state = []
              for node in nodes:
                current_state.append(node.load)

              actions, rewards, new_states, stop_episode =\
                env_interact(agents=agents, chigh=args.chigh, clow=args.clow,
                             utility=args.utility, exploration=False, K=args.K,
                             delta=current_delta, current_state=current_state,
                             adversarial_policy=adv_policy,
                             attack_type=args.attack_type)

              # update agents based on transitions and rewards
              for idx, agent in enumerate(agents):

                if args.method == "minimaxQ":
                  opponent_action = [actions[(adv_idx-1)*2] for adv_idx in
                                     agent.advs_idxs]

                  opponent_action.extend([actions[(adv_idx-1)*2+1] for adv_idx in
                                          agent.advs_idxs])

                  agent.update(next_state=new_states, reward=None,
                               opponent_action=opponent_action,
                               learn=False)
                else:

                  agent.update(next_state=new_states[(idx * len(agent.nodes)):(idx * len(
                    agent.nodes) + len(agent.nodes))], reward=None, learn=False)

              step_rewards.append(np.sum(rewards))

              performance_test["states"].append(new_states)
              performance_test["actions"].append(actions)
              performance_test["current_states"].append(current_state)


            performance_test["episode_rewards"].append(np.sum(step_rewards))
            performance_test["episodes_duration"].append(duration)

          test_dir = "../projects/" + args.project + "/trial_" +  str(trial) +\
                     "/episode_" + str(episode_train) + "/adversary_" +\
                     args.adversary + "_attack_" + args.attack_type + "/data"
          if not os.path.exists(test_dir):
            os.makedirs(test_dir)

          # save test data
          pickle.dump({"performance_test": performance_test},  file=open(
            test_dir+"/test_data_" +  str(current_delta) + ".pkl", "wb"))



  if not args.execute_only:
    for key, val in interm_episodes.items():
      val = list(set(val))
      interm_episodes[key] = val
    print(interm_episodes)
    args.interm_episodes = interm_episodes
    pickle.dump(args, open("../projects/" + args.project + "/config.pkl", "wb"))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--N',
                        help='Number of nodes/agents',
                        type=int,
                        default=100)

  parser.add_argument('--clow',
                        help='Punishment for underflow',
                        type=int,
                        default=0)

  parser.add_argument('--execute_only',
                      help='Execute policy without learning',
                      default=False,
                      action="store_true")

  parser.add_argument('--explore_attack',
                      help='Explore sub-optimal attacks (only for Rom-Q)',
                      default=0,
                      type=float)

  parser.add_argument('--adjust_parameters',
                      help='Adjust learning hyperparameters at each iteration.',
                      default=False,
                      action="store_true")

  parser.add_argument('--test_intermediate',
                      help='Indicates whether testing should evaluate all '
                           'intermediate trained agents.  ',
                      default=False,
                      action="store_true")

  parser.add_argument('--chigh',
                        help='Punishment for overflow',
                        type=int,
                        default=100)

  parser.add_argument('--utility',
                      help='Utility for executing a pack2et',
                      type=int,
                      default=8)

  parser.add_argument('--cost',
                      help='Cost of transmitting on an edge',
                      type=int,
                      default=0)

  parser.add_argument('--agents_file',
                      help='Name of file containing trained agents.',
                      type=str,
                      default="")

  parser.add_argument('--attack_type',
                      help='Choose betweetn randoma, randomb, randomc and '
                           'worst.',
                      type=str,
                      default="worst")

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

  parser.add_argument('--learning_rate',
                      help='Learning rate for temporal difference learning.',
                      type=float,
                      default=0.01)

  parser.add_argument('--epsilon',
                      help='Exploration rate for temporal difference learning.',
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

  parser.add_argument('--trials',
                      help='Number of Monte Carlo trials.',
                      type=int,
                      default=10)

  parser.add_argument('--method',
                      help='Indicates the learning method used. Choose '
                           'between Qlearning, SARSA, minimaxQ and RomQ.',
                      type=str,
                      default="Qlearning")

  parser.add_argument('--adversary',
                      help='Choose adversarial policy. Choices are Qlearning '
                           ' minimaxQ and RomQ.',
                      type=str,
                      default="Qlearning")

  parser.add_argument('--romq',
                      help='Indicates if ROM-Q is used.',
                      type=int,
                      default=0)

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

  parser.add_argument('--network_type',
                      help='Type of network. Choose between A, B and C.',
                      type=str,
                      default="A")

  parser.add_argument('--seed',
                      help='Seed used for generating random numbers.',
                      type=int,
                      default=0)

  args = parser.parse_args()
  main(args)

