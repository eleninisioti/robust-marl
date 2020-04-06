""" This script can be used to plot the operation of the agents during an
experiment.


Two identical plots are produced, one refers to the learning and the other to
 the execution time. Each plot consists of 3 sub-plots
"""
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import numpy as np
import seaborn
import sys
# ----- set up -----
#project = "benchmarks/QlearningA"
seaborn.set()
project = sys.argv[1]
data = pickle.load(open("../projects/" + project + "/experiment_data.pkl","rb"))
performance_train = data[0]
performance = data[1]
tuning = data[2]


# ------ plotting -----

# plot individual performances
rewards = performance["rewards"]
iters = len(rewards)

for node in range(tuning.N):
  node_rewards = []
  for iter in range(iters):
    current_reward = rewards[iter][node]
    node_rewards.append(current_reward)


  plt.plot(list(range(iters))[::10], node_rewards[::10], label="Node " + str(
    node))

plt.ylabel("Return, $R_t$")
plt.xlabel("Time step, $t$")
plt.legend()
plt.savefig("../projects/" + project + "/plots/rewards.eps")
plt.clf()


# plot social welfare
welfare = []
for reward in rewards:
  welfare.append(sum(reward))

plt.plot(list(range(iters))[::10], welfare[::10])
plt.ylabel("Welfare, $W_t$")
plt.xlabel("Time step, $t$")
plt.legend()
plt.savefig("../projects/" + project + "/plots/welfare.eps")
plt.clf()


# plot actions
actions = performance["actions"]
iters = len(actions)

for node in range(tuning.N):
  serve_node_actions = []
  send_node_actions = []
  for iter in range(iters):
    serve_action = actions[iter][node*2]
    send_action = actions[iter][node*2 + 1]
    serve_node_actions.append(serve_action)
    send_node_actions.append(send_action)


  plt.plot(list(range(iters))[::10], serve_node_actions[::10], label="Node " +
                                                                   str(node))

  plt.ylabel("Action, $A_t$")
  plt.xlabel("Time step, $t$")
  plt.legend()
  plt.savefig("../projects/" + project + "/plots/serve_actions_" + str(
    node) + ".eps")
  plt.clf()

  plt.plot(list(range(iters))[::10], send_node_actions[::10], label="Node " +
                                                                   str(
    node))

  plt.ylabel("Action, $A_t$")
  plt.xlabel("Time step, $t$")
  plt.legend()
  plt.savefig("../projects/" + project + "/plots/send_actions_" + str(
    node) + ".eps")
  plt.clf()



# plot states
states = performance["states"]
iters = len(states)

for node in range(tuning.N):
  node_states = []
  for iter in range(iters):
    current_states = states[iter][node]
    node_states.append(current_states)


  plt.plot(list(range(iters))[::10], node_states[::10], label="Node " + str(
    node))


plt.ylabel("Load, $L_t$")
plt.xlabel("Time step, $t$")
plt.legend()
plt.savefig("../projects/" + project + "/plots/states.eps")
plt.clf()

# heatmap of state visits
overflows = performance["overflows"]
underflows = performance["underflows"]

if tuning.N ==2:
  visits = np.zeros(shape=(tuning.capacity+1,tuning.capacity+1))
  for iter in range(iters):
    current_states = states[iter]
    current_overflows = overflows[iter]
    current_underflows = underflows[iter]
    visits[current_states[0] , current_states[1] ] +=1


# make a color map of fixed colors
cmap = 'inferno'
img = plt.imshow(visits, cmap=plt.get_cmap(cmap),
                 origin="lower")

plt.xlabel("$s_2$",color="blue")
plt.ylabel("$s_1$",color="green")
#plt.legend("Green arrow: Node 1 \\ Blue arrow: Node 2")
plt.title("$\mathcal{N}(s)$")
plt.axvline(x=3.5,color="red",ymax=0.8)
plt.axhline(y=3.5,color="red",xmax=0.8)

# plot policy
actions = performance["actions"]
states = performance["states"]
nteststeps = len(actions)

for s1 in range(tuning.capacity):
  for s2 in range(tuning.capacity):

    offset = 0
    Qtable = data[3]
    current_entry = [slice(None)] * 2
    current_entry[0] = s1
    current_entry[1] = s2
    Qtable = Qtable[tuple(current_entry)]
    actions = np.argmax(Qtable)
    actions = list(np.unravel_index(actions,
                          Qtable.shape))
    serve_action_1 = actions[0]
    send_action_1 = actions[1]
    serve_action_2 = actions[2]
    send_action_2 = actions[3]

    # actions of vertical-axis agent
    # serve_action_1 = actions[iter][0]
    # send_action_1 = actions[iter][1]

    # draw arrow (when an agent serves, they diminish their own state by 1,
    # when they send they increase the other agent's state by 1, )
    plt.arrow(s2, s1, send_action_1 - offset, - serve_action_1 + offset,
              color="green",
              head_width=0.05, head_length=0.1, length_includes_head=True)

    # actions of horizontal-axis agent
    # serve_action_2 = actions[iter][2]
    # send_action_2 = actions[iter][3]

    # draw arrow
    plt.arrow(s2, s1, - serve_action_2 + offset, send_action_2 - offset,
              color="blue",
              head_width=0.05, head_length=0.1, length_includes_head=True)

    # if tuple([s1,s2]) not in temp_visits:
    #   temp_visits[tuple([s1, s2])] = [serve_action_1, send_action_1,
    #                                serve_action_2,
    #                      send_action_2]
    # else:
    #   print(temp_visits[tuple([s1, s2])],[serve_action_1, send_action_1,
    #                                 serve_action_2,
    #                      send_action_2]  )
    #   if  [serve_action_1, send_action_1, serve_action_2,
    #                      send_action_2] != temp_visits[tuple([s1,s2])]:
    #     print("not equal")


# make a color bar
plt.colorbar(img, cmap=plt.get_cmap(cmap))
plt.savefig("../projects/" + project + "/plots/grid.eps")
plt.clf()

# ----- plot performance -----
seaborn.set_palette(seaborn.color_palette("colorblind"))
# plot evolution of rewards with time steps
rewards = performance_train["rewards"]
iters = len(rewards)

total_rewards = []
for iter in range(iters):
  system_reward = 0
  for node in range(tuning.N):
    system_reward += rewards[iter][node]
  total_rewards.append(system_reward)

plt.plot(list(range(iters)), total_rewards, label="Qlearning")
plt.ylabel("Reward, $r$")
plt.xlabel("Training step, $t$")
plt.legend()
plt.savefig("../projects/" + project + "/plots/performance_steps.eps")
plt.clf()

# plot evolution of rewards with episodes
rewards = performance_train["mean_episode_rewards"]
iters = len(rewards)
total_rewards = []
for iter in range(iters):
  system_reward = 0
  for node in range(tuning.N):
    system_reward += rewards[iter]
  total_rewards.append(system_reward)

plt.plot(list(range(iters)), total_rewards, label="Qlearning")
plt.ylabel("Average reward, $\\bar{r}$")
plt.xlabel("Episode, $e$")
plt.legend()
plt.savefig("../projects/" + project + "/plots/performance_episodes.eps")
plt.clf()


