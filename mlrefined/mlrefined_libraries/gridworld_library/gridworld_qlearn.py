from __future__ import division
from statistics import mean
import numpy as np
import pandas as pd
import time
import copy
import os, errno
import pickle

class learner():
    def __init__(self,**args):
        # get some crucial parameters from the input gridworld
        self.grid = args['gridworld']
        self.name = args['name']
        
        # initialize q-learning params
        self.gamma = 1
        self.max_steps = 5*self.grid.width*self.grid.height
        self.exploit_param = 0.5
        self.action_method = 'exploit'
        self.training_episodes = self.grid.training_episodes
        self.validation_episodes = 1
        self.training_start_schedule = []
        self.validation_start_schedule = []

        if 'exploit_param' in args:
            self.exploit_param = round(1 - args['exploit_param'],1)
            self.action_method = 'exploit'
           
        # create start schedule for training / validation
        self.training_start_schedule = self.grid.training_start_schedule[:self.training_episodes]
        self.validation_start_schedule = self.grid.validation_start_schedule[:self.validation_episodes]

        self.start_point = []
        
        
        if self.grid.isEight:
            direction = 8
        else:
            direction = 4
        
        directory = "result/" + self.name

        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        directory += "/" + str(direction) + "direction"

        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        if 'iter' in args:
            self.iter = args['iter']
            directory += "/" + str(self.training_episodes) + "episode_seedtime_" + str(args['exploit_param']) + "epsilon"
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            directory += "/" + str(self.iter+1)
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        else:
            directory += "/" + str(self.training_episodes) + "episode_seedtime_" + str(self.exploit_param) + "epsilon"
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
        

        self.out = open(directory + "/report.txt", 'w+')
        self.out_stat = open(directory + "/stat.txt", 'w+')
        self.out_csv = open(directory + "/report.csv", 'w+')
        self.out_model = open(directory + "/Q.model", 'wb')

        if 'start' in args:
            self.start_point = args['start']

        self.step_train = []
        self.step_val = []
        self.num_goal_train = 0
        self.num_goal_val = 0

        
    ### Q-learning function - version 1 - take random actions ###
    def train(self,**args):
        # switches
        if "gamma" in args:
            self.gamma = args['gamma']
        if 'max_steps' in args:
            self.max_steps = args['max_steps']
        if 'action_method' in args:
            self.action_method = args['action_method']
        if 'exploit_param' in args:
            self.exploit = args['exploit_param']
            self.action_method = 'exploit'
        if 'training_episodes' in args:
            self.training_episodes = args['training_episodes']
            # return error if number of training episodes is too big
        if self.training_episodes > self.grid.training_episodes:
            print ('requesting too many training episodes, the maximum num = ' + str(self.grid.training_episodes))
            return        
        if 'validation_episodes' in args:
            self.validation_episodes = args['validation_episodes']
            # return error if number of training episodes is too big
        if self.validation_episodes > self.grid.validation_episodes:
            print ('requesting too many validation episodes, the maximum num = ' + str(self.grid.validation_episodes))
            return 
        
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        gamma = self.gamma
        
        # containers for storing various output
        self.training_episodes_history = {}
        self.training_reward = []
        self.validation_reward = []
        self.time_per_episode = []
        self.Q_history = []
        Q = np.zeros((self.grid.width*self.grid.height,len(self.grid.action_choices)))
        returns = np.zeros((self.grid.width*self.grid.height,len(self.grid.action_choices),self.training_episodes))
        counter = np.zeros((self.grid.width*self.grid.height,len(self.grid.action_choices)))
        self.Q_history.append(copy.deepcopy(Q))  # save current Q (for visualization of progress)

        self.out_csv.write('episode,train goal?,train time (ms),train reward,train step,val goal?,val reward,val step\n')
        
        ### start main Q-learning loop ###
        for n in range(self.training_episodes): 
            start = time.clock()
            
            # pick this episode's starting position
            grid.agent = self.training_start_schedule[n]
            # print(grid.agent)

            # update Q matrix while loc != goal
            episode_history = []      # container for storing this episode's journey
            action_history = []
            episode_history_done = []
            total_episode_reward = 0
            reward_before = []
            
            self.out.write('EPISODE ' + str(n+1) + '\n')
            self.out_csv.write(str(n+1) + ',')
            self.out.write('TRAIN: ')
            
            step = 0

            for step in range(self.max_steps):   
                # update episode history container
                episode_history.append(grid.agent)
                
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    self.out.write('GOAL')
                    self.out_csv.write('yes')
                    self.num_goal_train += 1
                    break
                
                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = self.action_method,Q = Q,exploit_param = self.exploit_param)
                action_history.append(a_k)
                
                # move based on this action
                s_k = grid.get_movin(action = a_k)
               
                # get reward     
                r_k = grid.get_reward(state_index = s_k)          
                
                   
                # update current location of agent 
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
                # update training reward
                reward_before.append(total_episode_reward)
                total_episode_reward+=r_k
                
                step += 1

            for step2 in range(len(action_history)):
                s_k_1 = grid.state_tuple_to_index(episode_history[step2])
                    
                a_k = action_history[step2]
                                 
                
                if (str(s_k_1) + str(a_k)) not in episode_history_done:
                    G = total_episode_reward - reward_before[step2]
                                        
                    returns[s_k_1,a_k,n] = G
                    counter[s_k_1,a_k] += 1

                    Q[s_k_1,a_k] = (sum(returns[s_k_1,a_k,:]))/counter[s_k_1,a_k]
                    
                    episode_history_done.append(str(s_k_1) + str(a_k))
                   
                

            # print out update if verbose set to True
            if 'verbose' in args:
                if args['verbose'] == True:
                    if np.mod(n+1,100) == 0:
                        print ('training episode ' + str(n+1) +  ' of ' + str(self.training_episodes) + ' complete')
            
            ### store this episode's computation time and training reward history
            stop = time.clock()
            self.time_per_episode.append(stop - start)
            self.training_episodes_history[str(n)] = episode_history
            self.training_reward.append(total_episode_reward)
            self.Q_history.append(copy.deepcopy(Q))  # save current Q (for visualization of progress)

         
            self.out.write('\n')
            self.out.write('time: ' + str((stop-start)*1000) + ' ms\n')
            self.out.write('reward: ' + str(total_episode_reward) + '\n')
            self.out.write('step: ' + str(step) + '\n')
            self.step_train.append(step)

            self.out_csv.write(',')
            self.out_csv.write(str((stop-start)*1000) + ',')
            self.out_csv.write(str(total_episode_reward) + ',')
            self.out_csv.write(str(step) + ',')
            

            ### store this episode's validation reward history
            if 'validate' in args:
                if args['validate'] == True:
                    reward = self.validate(Q)
                    self.validation_reward.append(reward)
            

        self.Q = Q  # make a global version

        self.out_stat.write('STATISTICS\n\n')
        self.out_stat.write('TRAIN\n')
        self.out_stat.write('minimum step: ' + str(min(self.step_train)) + '\n')
        self.out_stat.write('maximum step: ' + str(max(self.step_train)) + '\n')
        self.out_stat.write('average step: ' + str(mean(self.step_train)) + '\n')
        self.out_stat.write('goal percentage: ' + str(self.num_goal_train/self.training_episodes*100) + '%\n')

        self.out_stat.write('\n')

        self.out_stat.write('VALIDATE\n')
        self.out_stat.write('minimum step: ' + str(min(self.step_val)) + '\n')
        self.out_stat.write('maximum step: ' + str(max(self.step_val)) + '\n')
        self.out_stat.write('average step: ' + str(mean(self.step_val)) + '\n')
        self.out_stat.write('goal percentage: ' + str(self.num_goal_val/self.training_episodes*100) + '%\n')

        pickle.dump(Q, self.out_model)

        self.out.close()
        self.out_csv.close()
        self.out_stat.close()
        self.out_model.close()
            
        print ('q-learning algorithm complete')
       
    ### run validation episodes ###
    def validate(self,Q):
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        
        # run validation episodes
        total_reward = []
        self.out.write('VALIDATE: ')
        
        # run over validation episodes
        for i in range(self.validation_episodes):  
            
            # get this episode's starting position
            grid.agent = self.start_point

            # reward container for this episode
            episode_reward = 0
            step_count = 0

            # run over steps in single episode
            for j in range(grid.max_steps):
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    self.out.write('GOAL')
                    self.out_csv.write('yes')
                    self.num_goal_val += 1
                    break
                
                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = 'optimal',Q = Q)
                
                # move based on this action - if move takes you out of gridworld don't move and instead move randomly 
                s_k = grid.get_movin(action = a_k, illegal_move_response = 'random')
  
                # compute reward and save
                r_k = grid.get_reward(state_index = s_k)          
                episode_reward += r_k
    
                # update agent location
                grid.agent = grid.state_index_to_tuple(state_index = s_k)

                step_count += 1
                
            # after each episode append to total reward
            total_reward.append(episode_reward)
            self.step_val.append(step_count)

            self.out.write('\n')
            self.out.write('reward: ' + str(episode_reward) + '\n')
            self.out.write('step: ' + str(step_count) + '\n')
            self.out.write('\n')

            self.out_csv.write(',')
            self.out_csv.write(str(episode_reward) + ',')
            self.out_csv.write(str(step_count) + '\n')           
        
        # return total reward
        return np.median(total_reward)