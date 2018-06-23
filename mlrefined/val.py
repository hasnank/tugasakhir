import sys
sys.path.append('../../')
from mlrefined_libraries import gridworld_library as lib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

small_maze = lib.gridworld_enviro.environment(world_size = 'small', world_type = 'maze')

# show the grid
#small_maze.color_gridworld()

# create an instance of the q-learner
qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze)

# run q-learning
qlearner.train(validate=True,verbose = False, action_method = 'random', training_episodes = 100)
series = qlearner.validation_reward
fig = plt.figure(figsize = (12,4))

# plot each reward history
plt.plot(series,color = 'b',linewidth = 2)

# clean up panel
ymin = min(series)
ymax = max(series)
ygap = abs((ymax - ymin)/float(10))
plt.ylim([ymin - ygap,ygap])
plt.xlabel('episode')
plt.ylabel('average reward')
plt.title('validation history')
plt.show()

