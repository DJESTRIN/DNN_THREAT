# -*- coding: utf-8 -*-
"""DQN_AGENT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ySugWVDvS_XOtTMuK-7f0V_Wjr3WjjmV

Code adapted from: https://github.com/mahakal001/reinforcement-learning
"""

!pip install optuna

# Commented out IPython magic to ensure Python compatibility.
# %pip install -U gym>=0.21.0
# %pip install -U gym[atari,accept-rom-license]

""" Import dependencies """
import pdb
import torch
from torch import randint
import gym
from torch import nn
import copy
from collections import deque
import random
import gym
from tqdm import tqdm
import optuna
import numpy as np
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import os
from google.colab import drive
from torch.optim.lr_scheduler import ExponentialLR
from gym.wrappers import AtariPreprocessing
warnings.filterwarnings("ignore", category=UserWarning)

""" Save DQN details """
drive.mount('/content/drive', force_remount=True)
data_path ="//content//drive//My Drive//LISTON_LAB//SYNAPTIC_LAYER_PROJECT//"
os.chdir(data_path)

rew_arr = []
episode_count = 100
env = gym.make("SpaceInvaders-v0")
for i in range(episode_count):
    obs, done, rew = env.reset(), False, 0
    episode_rewards =[]
    while (done != True) :
        A =  randint(0,env.action_space.n,(1,))
        obs, reward, done, info = env.step(A.item())
  
        if done==True:
          reward = -(max(episode_rewards)*100)/len(episode_rewards)
        
        episode_rewards.append(reward)
        rew += reward

    rew_arr.append(rew)
    
print("average reward per episode :",sum(rew_arr)/ len(rew_arr))

class DQN_Agent:
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size, LR_Gamma):
        torch.manual_seed(seed)
        self.architecture = layer_sizes
        self.q_net = self.build_nn(layer_sizes)
        self.target_net = copy.deepcopy(self.q_net)
        self.q_net.cuda()
        self.target_net.cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=LR_Gamma)
        
        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float().cuda()
        self.experience_replay = deque(maxlen = exp_replay_size)  
        return
        
    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes)-1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index+1])
            act = nn.Tanh() if index < len(layer_sizes)-2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)
    
    def get_action(self, state, action_space_len, epsilon):
        with torch.no_grad():
            state = state.flatten()
            Qp = self.q_net(torch.from_numpy(state).float().cuda())
        Q, action = torch.max(Qp, axis=0)
        action = action if torch.rand(1,).item() > epsilon else torch.randint(0,action_space_len,(1,))
        return action
    
    def get_q_next(self, state):
        with torch.no_grad():
            state = state.reshape(self.batch_size,self.architecture[0])
            qp = self.target_net(state)
        q,_ = torch.max(qp, axis=1)    
        return q
    
    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return
    
    def sample_from_experience(self, sample_size):
        if(len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)   
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()   
        return s, a, rn, sn
    
    def train(self, batch_size):
        self.batch_size = batch_size
        s, a, rn, sn = self.sample_from_experience( sample_size = batch_size)
        if(self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0
        
        # predict expected return of current state using main network
        qp = self.q_net(s.cuda())
        pred_return, _ = torch.max(qp, axis=1)
        
        # get target return using target network
        q_next = self.get_q_next(sn.cuda())
        target_return = rn.cuda() + self.gamma * q_next
        
        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self.scheduler.step()

        self.network_sync_counter += 1       
        return loss.item()

def Train_Agent(Environment, Agent,Experience_Replay_Size, Episodes, epsilon, Seed, Batch_Size, epsilon_decay):   
    # initiliaze experiance replay      
    index = 0
    for i in range(Experience_Replay_Size):
        obs = Environment.reset()
        done = False
        while(done != True):
            obs = obs.flatten()
            A = Agent.get_action(obs, Environment.action_space.n, epsilon=1)
            obs_next, reward, done, _ = Environment.step(A.item())
            Agent.collect_experience([obs, A.item(), reward, obs_next])
            obs = obs_next
            index += 1
            if( index > Experience_Replay_Size ):
                break
                
    # Main training loop
    losses_list, reward_list, episode_len_list, epsilon_list, action_list  = [], [], [], [], []
    index = 128

    for i in tqdm(range(Episodes)):
        obs, done, losses, ep_len, rew = Environment.reset(), False, 0, 0, 0
        while(done != True):
            ep_len += 1
            obs = obs.flatten() 
            action = Agent.get_action(obs, Environment.action_space.n, epsilon)
            obs_next, reward, done, _ = Environment.step(action.item())
            Agent.collect_experience([obs, action.item(), reward, obs_next])
          
            obs = obs_next
            rew  += reward
            index += 1
            
            if(index > 128):
                index = 0
                for j in range(4):
                    loss = Agent.train(batch_size=Batch_Size)
                    losses += loss   

        if epsilon > 0.05 :
            epsilon -= epsilon_decay

        losses_list.append(losses/ep_len), reward_list.append(rew), episode_len_list.append(ep_len), epsilon_list.append(epsilon), action_list.append(action.item())

    return losses_list, reward_list, episode_len_list, epsilon_list, action_list

""" Get random seed from time """
def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day + dt_time.second

""" Plot Figures from training """
def get_figures(Losses, Rewards, Durations, Epsilons, Actions):
    plt.subplot(511)
    plt.plot(Losses)
    plt.ylabel('Losses')
    plt.subplot(512)
    plt.plot(Rewards)
    plt.ylabel('Rewards')
    plt.subplot(513)
    plt.plot(Durations)
    plt.ylabel('Durations')
    plt.subplot(514)
    plt.plot(Epsilons)
    plt.ylabel('Epsilons')
    plt.subplot(515)
    plt.plot(Actions)
    plt.ylabel('Actions')
    plt.savefig((str(to_integer(datetime.now())) + '.png'))
    plt.close()

""" Generate initial map of architecture """
def NeuralNetworkLayout(input_dim,output_dim,Number_of_hidden_Layers, Layer_range_max, Layer_range_min):
    random.sample(range(Layer_range_min, Layer_range_max), Number_of_hidden_Layers)
    architecture = [input_dim] + random.sample(range(Layer_range_min, Layer_range_max), Number_of_hidden_Layers) + [output_dim]
    return architecture

"""Hyperparameters"""

def objective(trial):
    # Hyperparameters under investigation 
    Learing_Rate = trial.suggest_float('Learing_Rate', 1e-3, 1e+3, log=True) #Initial learning rate
    Learing_Rate_Decay = trial.suggest_float('Learing_Rate_Decay', 1e-5, 1e-2, log=True) #Learning rate scheduler
    Epsilon_Decay = trial.suggest_float('Epsilon_Decay', 1e-3, 1e-1,log=True) #Epsilon decay
    Batch_Size = trial.suggest_int('Batch_Size', 8, 64) #Batch Size
    Critic_Frequency= trial.suggest_int('Critic_Frequency', 4, 8) #Epochs to update target network
    Number_of_hidden_Layers = trial.suggest_int('Number_of_hidden_Layers', 1, 10) #Number of hidden layers
    Layer_range_max = trial.suggest_int('Layer_range_max', 50, 500) #Max neurons per layer
    Layer_range_min = trial.suggest_int('Layer_range_min', 10, 49) #Min neurons per layer

    # Fixed Hyperparameters
    Seed = to_integer(datetime.now()) #randomly generated seed based on datetime
    Experience_Replay_Size = 256
    Epsilon = 1 #Initial epsilon for exploration/exploitation
    Episodes = 10 #Fixed number of episodes
    Environment = gym.make("SpaceInvaders-v0") #Current environment
    Environment = AtariPreprocessing(Environment, frame_skip=1)
    input_dim = Environment.observation_space.shape
    input_dim = input_dim[0]*input_dim[1]
    output_dim = Environment.action_space.n

    # Neural Network Layout
    Neural_Network_Architechture = NeuralNetworkLayout(input_dim,output_dim,Number_of_hidden_Layers, Layer_range_max, Layer_range_min)

    #Create DQN Agent
    Agent = DQN_Agent(seed = Seed, layer_sizes = Neural_Network_Architechture, lr = Learing_Rate, sync_freq = Critic_Frequency, 
                      exp_replay_size = Experience_Replay_Size, LR_Gamma = Learing_Rate_Decay)
    
    #Train DQN Agent 
    Losses, Rewards, Durations, Epsilons, Actions = Train_Agent(Environment, Agent, Experience_Replay_Size, Episodes, Epsilon, Seed, Batch_Size, Epsilon_Decay)
    Reward_average = np.mean(Rewards)
    get_figures(Losses, Rewards, Durations, Epsilons, Actions)

    trial.report(Reward_average, Episodes)
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()

    # output:  Average reward
    return Reward_average

study = optuna.create_study(study_name='BASIC_DQN_HYPERPARAMETERS', direction='maximize')
study.optimize(objective, n_trials=500)
optuna.visualization.plot_optimization_history(study)

Environment = gym.make('CartPole-v0')
input_dim = Environment.observation_space.shape[0]
output_dim = Environment.action_space.n
Neural_Network_Architechture = [input_dim, 175, 224, output_dim]

Agent = DQN_Agent(seed = Seed, layer_sizes = Neural_Network_Architechture, lr = 1.6337841207870763e-07, sync_freq = 4, exp_replay_size = 256)
Losses, Rewards, Durations, Epsilons = Train_Agent(Environment, Agent, 256, 10000, 1, to_integer(datetime.now()), 46)

Reward_average = np.mean(Rewards)
plt.plot(Rewards)

"""Test Agent"""

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "record_dir", force='True')

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
exp_replay_size = 256
agent = DQN_Agent(seed=1423, layer_sizes=[input_dim, 64, output_dim], lr=1e-3, sync_freq=5,
                  exp_replay_size=exp_replay_size)
agent.load_pretrained_model("cartpole-dqn.pth")

reward_arr = []
for i in tqdm(range(100)):
    obs, done, rew = env.reset(), False, 0
    while not done:
        A = agent.get_action(obs, env.action_space.n, epsilon=0)
        obs, reward, done, info = env.step(A.item())
        rew += reward
        # sleep(0.01)
        # env.render()

    reward_arr.append(rew)
print("average reward per episode :", sum(reward_arr) / len(reward_arr))