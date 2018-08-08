from tkinter import *
from tkinter.filedialog import askopenfilename
import sys
import os
from mlrefined_libraries import gridworld_library as lib
import matplotlib.pyplot as plt

sys.path.append('../../')

dirr = os.getcwd() + '/environments/'

# files = [dirr+'25x25_54percent', dirr+'50x50_20percent', dirr+'50x50_51percent']
# files = [dirr+'5x5_64percent', dirr+'5x5_80percent', dirr+'10x10_44percent', dirr+'10x10_57percent', dirr+'25x25_20percent'dirr+'25x25_54percent', dirr+'50x50_20percent', dirr+'50x50_51percent']
files = [dirr+'10x10_44percent', dirr+'10x10_57percent', dirr+'25x25_20percent', dirr+'25x25_54percent', dirr+'50x50_20percent', dirr+'50x50_51percent']
# files = [dirr+'5x5_64percent', dirr+'5x5_80percent']

for file in files:
    for isEight in range(2):
        # file = open(val, 'r')    
        name = os.path.basename(file)
        if file != '':
            hazard = open("mlrefined_libraries/gridworld_library/gridworld_levels/" + name + "_maze_hazards.csv", 'w')
            # goal = open("mlrefined_libraries/gridworld_library/gridworld_levels/" + name + "_maze_goal.csv", 'w')
            # start = open("mlrefined_libraries/gridworld_library/gridworld_levels/" + name + "_maze_start_schedule.csv", 'w')
            k=0
            for line in reversed(list(open(file))):
                 #print(line.rstrip())
                i=0
                for letter in line: 
                    if letter == '9':
                        newLine = str(k) + "," + str(i) + "\n"
                        hazard.write(newLine)
                    elif letter == '1':
                        newLine = str(k) + "," + str(i) + "\n"
                        # start.write(newLine)
                        start = [k,i]
                    elif letter == '5':
                        newLine = str(k) + "," + str(i) + "\n"
                        # goal.write(newLine)
                        goal = [k,i]
                    i=i+1
                width = i - 1
                k=k+1
            height = k
            hazard.close()
            # goal.close()
            # start.close()


        

        # run q-learning
        for it in range(1,6):
            small_maze = lib.gridworld_enviro.environment(world_size = name, world_type = 'maze', height=height, width=width, goal=goal, start=start, training_episodes = 1000, isEight = isEight, seed = it)
            # create an instance of the q-learner
            for eps in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
                qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze, start = start, name = name, exploit_param = eps)

                qlearner.train(verbose = False, action_method = 'exploit', validate = True)
