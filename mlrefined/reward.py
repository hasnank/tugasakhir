import time
import sys
sys.path.append('../../')
from mlrefined_libraries import gridworld_library as lib
#%matplotlib tk
import matplotlib.pyplot as plt
plt.style.use('ggplot')

small_maze=lib.gridworld_enviro.environment(world_size='small', world_type = 'maze')

# compare random versus exploration-exploitation actions on training
training_rewards = []
methods = ['random','optimal']
for i in range(len(methods)):
    method = methods[i]
    
    # create instance of learner
    small_maze_qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze)
    
    # run q-learning
    start = time.clock()
    small_maze_qlearner.train(verbose = False, action_method = method,training_episodes = 100)
    end = time.clock()
    value = (end - start)
    print ('method ' + str(method) + ' completed training in ' + str(value) + ' seconds')
    
    # record rewards and history
    training_rewards.append(small_maze_qlearner.training_reward)

fig = plt.figure(figsize = (12,5))
ax = fig.add_subplot(1,1,1)

for i in range(len(methods)):
    ax.plot(training_rewards[i])
ax.set_xlabel('episode')
ax.set_ylabel('total reward')
ax.legend(['method = ' + str(methods[0]),'method = ' + str(methods[1])],loc='center left', bbox_to_anchor=(1, .5))
plt.show()    





