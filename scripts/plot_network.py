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

# ----- set up -----
project = "pair_2"
data = pickle.load(open("../projects/" + project + "/experiment_data.pkl","rb"))
performance = data[0]
tuning = data[1]


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
  node_actions = []
  for iter in range(iters):
    current_actions = actions[iter][node]
    node_actions.append(current_actions)


  plt.plot(list(range(iters))[::10], node_actions[::10], label="Node " + str(
    node))

plt.ylabel("Action, $A_t$")
plt.xlabel("Time step, $t$")
plt.legend()
plt.savefig("../projects/" + project + "/plots/actions.eps")
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
if tuning.N ==2:
  visits = np.zeros(shape=(tuning.capacity,tuning.capacity))
  for iter in range(iters-200, iters):
    current_states = states[iter]
    visits[current_states[0], current_states[1]] +=1


# make a color map of fixed colors
cmap = 'YlOrRd'
bounds=[0,100,200,500,1000,5000,10000]


# tell imshow about color map so that only set colors are used
img = plt.imshow(visits, cmap=plt.get_cmap(cmap),
                 origin="lower")

# make a color bar
plt.colorbar(img,cmap=plt.get_cmap(cmap), boundaries=bounds)

plt.savefig("../projects/" + project + "/plots/grid.eps")


