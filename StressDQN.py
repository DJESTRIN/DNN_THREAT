""" A DQN for analysing synaptic weights durign stress  """
# Import dependencies
import gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import tqdm 
import ipdb 
import numpy as np
import sys
sys.path.append("/data/dje4001/DNN_THREAT/")
from record_network import Record,tallformat

#Import torch and torch tools 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/data/dje4001/StressDQN')
import torch.optim.lr_scheduler as lr_scheduler

# Set current game
env_name="SpaceInvaders-v4"
env = gym.make(env_name)

# set up matplotlib REPLACE WITH TENSORBOARD
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

# Set device name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Experience Replay class
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Custom DQN agent 
class DQN(nn.Module):

    def __init__(self, n_channels, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(n_channels, 100,8,stride=2)
        self.layer2 = nn.Conv2d(100, 64,8,stride=2)
        self.layer3 = nn.Conv2d(64, 32,8,stride=2)
        self.layer4 = nn.Linear(9408, 512)
        self.layer5 = nn.Linear(512, 128)
        self.layer6 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x=torch.reshape(x,(x.shape[0],x.shape[3],x.shape[1],x.shape[2]))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x=torch.reshape(x,(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        ipdb.set_trace()
        return self.layer6(x)
    

#Set hyperparameters:

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 0.1

#Custom function for changing the learning rate. 
def custom_lr_scheduler(epoch):
    # LR to be 0.1 * (1/1+0.01*epoch)
    base_lr = 0.1
    factor = 0.01
    return base_lr/(1+factor*epoch)

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = state.shape[0] * state.shape[1] 

# Create neural networks
policy_net = DQN(3, n_actions).to(device)
target_net = DQN(3, n_actions).to(device)
rec=Record("/data/dje4001/DNN_THREAT/model_data/",env_name)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
scheduler = lr_scheduler.LambdaLR(optimizer, custom_lr_scheduler)
memory = ReplayMemory(10000)
steps_done = 0

# Selects action while maintaining exploration:exploitation balance via epsilon
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #reshaped_state=torch.reshape(state,(n_observations,1))
            ipdb.set_trace()
            return torch.argmax(policy_net(state).max(1)[1],axis=2)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
episode_durations = []

# REPLACE WITH TENSOR BOARD
def plot_durations(reward_rn, show_result=False):
    plt.figure(1)
    reward_rn=reward_rn.cpu()
    reward_oh=reward_rn.clone().detach()
    durations_t = torch.tensor(reward_oh, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #reshaped_state_batch=torch.reshape(state_batch,(BATCH_SIZE,n_observations))
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        #try:
         #   reshaped_nfns=torch.reshape(non_final_next_states,(BATCH_SIZE,n_observations))
        #except:
         #   reshaped_nfns=torch.reshape(non_final_next_states,(non_final_next_states.shape[0],n_observations))
         
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 10
else:
    num_episodes = 50

for i_episode in tqdm.tqdm(range(num_episodes)):
    scheduler.step()
    # Initialize the environment and get it's state
    state, info = env.reset()
    ipdb.set_trace()
    #state=np.mean(state,axis=2)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            #observation=np.mean(observation,axis=2)
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        #Record network weights and biases 
        rec.record(target_net,t,i_episode, 'target')
        rec.record(policy_net,t,i_episode, 'policy')
        
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            #plot_durations(reward)
            # if i_episode == 0 or i_episode == num_episodes - 1:
            #     result.release()
            break
    
    writer.add_scalar("Reward/Episode", reward, i_episode)

print('Complete')
plot_durations(reward,show_result=True)
plt.ioff()
plt.show()

