import pandas as pd
import os
import numpy as np

location = os.path.dirname(os.path.realpath(__file__))

# goal_csvname = location + '/mlrefined/' + 'filename_maze_goal.csv'
# start_csvname = location + '/mlrefined/' + 'filename_maze_start_schedule.csv'
hazard_csvname = location + '/mlrefined/mlrefined_libraries/gridworld_library/gridworld_levels/' + 'filename_maze_hazards.csv'

# goal = pd.read_csv(goal_csvname,header = None)
# agent = pd.read_csv(start_csvname,header = None)
hazard = pd.read_csv(hazard_csvname,header = None)

temp = []
grid = []
grid = np.zeros((30,30))
for i in range(len(hazard)):
    block = list(hazard.iloc[i])
    temp.append(block)

print (temp)