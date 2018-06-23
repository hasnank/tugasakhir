import sys
sys.path.append('../../')
from mlrefined_libraries import gridworld_library as lib

small_maze = lib.gridworld_enviro.environment(world_size = 'filename', world_type = 'maze', height=25, width=25)

# show the grid
#small_maze.color_gridworld()

# show preset rewards and gamma value
print ('the standard square reward is preset to ' + str(small_maze.standard_reward))
print ('the hazard reward is preset to ' + str(small_maze.hazard_reward))
print ('the goal reward is preset to ' + str(small_maze.goal_reward))
# create an instance of the q-learner
qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze)

# run q-learning
qlearner.train(verbose = False, action_method = 'random',training_episodes = 100)
# create instance of animator
animator = lib.gridworld_animators.animator()

### animate training runs of one algorithm ###
animator.animate_training_runs(gridworld = small_maze, learner = qlearner,episodes = [70,80])
