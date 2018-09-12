from tkinter import *
from tkinter.filedialog import askopenfilename
import sys
import os
from mlrefined_libraries import gridworld_library as lib
import matplotlib.pyplot as plt

class menu():
    def __init__(self):
        self.file = ''
        self.name = ''
        self.height = 0
        self.width = 0
        self.goal = []
        self.start = []
        self.var = IntVar()
        self.isEight = 0

    def openFile(self, event):
        print(self.isEight)
        dirr = os.getcwd() + '/environments'
        self.file = askopenfilename(parent=root, initialdir = dirr, title='Choose a file')
        self.name = os.path.basename(self.file)
        if self.file != '':
            hazard = open("mlrefined_libraries/gridworld_library/gridworld_levels/" + self.name + "_maze_hazards.csv", 'w')
            # goal = open("mlrefined_libraries/gridworld_library/gridworld_levels/" + self.name + "_maze_goal.csv", 'w')
            # start = open("mlrefined_libraries/gridworld_library/gridworld_levels/" + self.name + "_maze_start_schedule.csv", 'w')
            k=0
            for line in reversed(list(open(self.file))):
                 #print(line.rstrip())
                i=0
                for letter in line: 
                    if letter == '9':
                        newLine = str(k) + "," + str(i) + "\n"
                        hazard.write(newLine)
                    elif letter == '1':
                        newLine = str(k) + "," + str(i) + "\n"
                        # start.write(newLine)
                        self.start = [k,i]
                    elif letter == '5':
                        newLine = str(k) + "," + str(i) + "\n"
                        # goal.write(newLine)
                        self.goal = [k,i]
                    i=i+1
                self.width = i - 1
                k=k+1
            self.height = k
            hazard.close()
            # goal.close()
            # start.close()

    def changeEight(self):
        self.isEight = self.var.get()
        print(self.isEight)


    def run(self, event):
        small_maze = lib.gridworld_enviro.environment(world_size = self.name, world_type = 'maze', height=self.height, width=self.width, goal=self.goal, start=self.start, training_episodes = 1000, isEight = self.isEight)

        # show the grid
        #small_maze.color_gridworld()

        # show preset rewards and gamma value
        print ('the standard square reward is preset to ' + str(small_maze.standard_reward))
        print ('the hazard reward is preset to ' + str(small_maze.hazard_reward))
        print ('the goal reward is preset to ' + str(small_maze.goal_reward))
        
        # run q-learning
        # for i in range(5):
        #     # create an instance of the q-learner
        #     for val in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
        #         qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze, start = self.start, name = self.name, iter = i, exploit_param = val)

        #         qlearner.train(verbose = False, action_method = 'exploit', validate = True)
    
        qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze, start = self.start, name = self.name)

        qlearner.train(verbose = False, action_method = 'exploit', validate = True)
        
        # # create instance of animator
        # animator = lib.gridworld_animators.animator()

        # ### animate training runs of one algorithm ###
        # # animator.animate_training_runs(gridworld = small_maze, learner = qlearner,episodes = [0,999])
        # animator.animate_validation_runs(gridworld = small_maze, learner = qlearner, starting_locations = [self.start])

    # def val(self, event):
    #     plt.style.use('ggplot')

    #     small_maze = lib.gridworld_enviro.environment(world_size = self.name, world_type = 'maze', height=self.height, width=self.height, goal=self.goal)

    #     # create an instance of the q-learner
    #     qlearner = lib.gridworld_qlearn.learner(gridworld = small_maze, start = self.start)
        
    #     # run q-learning
    #     qlearner.train(validate=True, verbose = False, action_method = 'exploit', training_episodes = 1000)
    #     training_reward = qlearner.training_reward

    #     fig = plt.figure(figsize = (12,5))
    #     ax = fig.add_subplot(1,1,1)

    #     ax.plot(training_reward)
    #     ax.set_xlabel('episode')
    #     ax.set_ylabel('total reward')
    #     plt.show()    

    def animateModel(self, event):
        dirr = os.getcwd() + '/result/' + self.name
        if self.isEight:
            dirr += '/8direction'
        else:
            dirr += '/4direction'

        small_maze = lib.gridworld_enviro.environment(world_size = self.name, world_type = 'maze', height=self.height, width=self.width, goal=self.goal, start=self.start, training_episodes = 1000, isEight = self.isEight)

        file = askopenfilename(parent=root, initialdir = dirr, title='Choose a model file', filetypes = [("model files","*.model")])
        
        # create instance of animator
        animator = lib.gridworld_animators.animator()

        ### animate training runs of one algorithm ###
        # animator.animate_training_runs(gridworld = small_maze, learner = qlearner,episodes = [0,999])
        animator.animate_validation_runs(gridworld = small_maze, q = file, starting_locations = [self.start])


sys.path.append('../../')

root = Tk()

menu = menu()

chooseFileButton = Button(root, text="Choose File...")
chooseFileButton.bind("<Button-1>", menu.openFile)
chooseFileButton.pack()

eightModeCheck = Checkbutton(root, text = "8-mode", variable = menu.var, command = menu.changeEight)
eightModeCheck.pack()

runButton = Button(root, text="Run")
runButton.bind("<Button-1>", menu.run)
runButton.pack()

chooseModelButton = Button(root, text="Animate Model")
chooseModelButton.bind("<Button-1>", menu.animateModel)
chooseModelButton.pack()

# valButton = Button(root, text="Validate")
# valButton.bind("<Button-1>", menu.val)
# valButton.pack()

root.mainloop()