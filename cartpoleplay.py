
import gym
import math
import random

import joblib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
#from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
#%matplotlib inline

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display


class DQN(nn.Module):
    def __init__(self, img_height, img_width, n_state_space):
        super(DQN, self).__init__()

        #self.fc1 = nn.Linear(in_features=img_width*img_height*3, out_features=128)
        self.fc1 = nn.Linear(in_features=n_state_space, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)
        #self.out = nn.Linear(in_features=8, out_features=2)

    def forward(self, t):
        #t = t.flatten(start_dim=1)
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t).to(device).unsqueeze(0)
        t = F.relu(self.fc1(t))
        t = self.fc2(t)
        #t = self.out(t)
        return t


Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self. decay = decay

    def get_exploration_rate(self, current_step):
        #print("exp r: ", self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay))
        return self.end + (self.start - self.end) * math.exp(-1 * current_step * self.decay)


class Agent:
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        # rate = self.strategy.get_exploration_rate(self.current_step)
        # self.current_step += 1
        #
        # if rate > random.random():
        #     action = random.randrange(self.num_actions)
        #
        #     #print("explor ",action)
        #     return torch.tensor([action]).to(self.device), rate
        # else:
        #     with torch.no_grad():
        #         #print("exploit action", policy_net(state).argmax(dim=1))
        #         #print("values: ", policy_net(state))
        with torch.no_grad():
            return policy_net(state).argmax(dim=1)


class CartPoleEnvManager:
    def __init__(self, device):
        self.device = device
        self.env = gym.make('CartPole-v1', render_mode='human').unwrapped
        #self.env.render_mode = 'human'
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.n_state_space = self.env.observation_space.shape[0]
        print("obeservation space", self.n_state_space)

    def reset(self):
        state = self.env.reset()[0]
        self.current_screen = None
        return state

    def close(self):
        self.env.close()

    def render(self, mode='rgb_array'):
        self.env.render_mode = mode
        screen = self.env.render()
        #plt.imshow(screen)
        #plt.pause(0.002)
        if is_ipython: display.clear_output(wait=True)
        return screen

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        next_state, reward, self.done, _, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device), next_state

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = torch.from_numpy(screen)

        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40, 90)),
            T.ToTensor()
        ])

        return resize(screen).unsqueeze(0).to(self.device)

class QValues:
    print(torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    @staticmethod
    def get_current(policy_net, states, actions):
        #print("current :", policy_net(states).gather(dim=1, index=actions.unsqueeze(-1)))
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values


def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1, t2, t3, t4)


def plot(values, moving_avg_period, best, eps, loss):
    plt.figure(4)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = get_moving_average(moving_avg_period, values)
    if moving_avg[-1] > best:
        best = moving_avg[-1]
    plt.plot(get_moving_average(moving_avg_period, values))
    #plt.pause(0.0001)

    # plt.figure(1)
    # plt.clf()
    # plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('epsilon')
    plt.plot(eps)
    # plt.pause(0.0001)
    #
    # plt.figure(1)
    # plt.clf()
    # plt.title('Training...')
    # plt.xlabel('Episode')
    # plt.ylabel('loss')
    plt.plot(loss)
    #
    plt.pause(0.0001)



    #print("Episode", len(values), "\n", moving_avg_period, "episode moving avg:", moving_avg[-1])
    if is_ipython: display.clear_output(wait=True)
    return best

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period -1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()


# params
batch_size = 256
gamma = 0.98
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 100
memory_size = 10000
lr = 0.00025
num_episodes = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)
n_state_space = em.n_state_space
# policy_net = DQN(em.get_screen_height(), em.get_screen_width(), n_state_space).to(device)
policy_net = joblib.load('policy_net.pkl')
#policy_net.eval()
#target_net = DQN(em.get_screen_height(), em.get_screen_width(), n_state_space).to(device)

#target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()
#optimizer = optim.Adam(policy_net.parameters(), lr=lr)

episodes_durations = []
best = 0


def calculate_reward(state):
    # Weight parameters for balancing the car and pole
    x = state[0]
    x_velocity = state[1]
    y = state[2]
    y_velocity = state[3]
    car_weight = 1.0
    pole_weight = 1.0

    # Penalize if the car is too far from the center
    car_penalty = abs(x) * car_weight

    # Penalize if the pole angle is too large
    pole_penalty = abs(y) * pole_weight

    # Combine penalties
    total_penalty = car_penalty + pole_penalty

    # Apply exponential decay to the penalties to encourage stability
    total_penalty = total_penalty * (0.9 ** abs(x_velocity + y_velocity))
    return -total_penalty


for episode in range(num_episodes):
    state = em.reset()
    #state = em.get_state()
    l = 0
    for timestep in range(1000):
        policy_net.eval()
        action = agent.select_action(state, policy_net)
        reward, next_state = em.take_action(action)
        #reward += calculate_reward(next_state)
        #print("reward: ", reward)
        #next_state = em.get_state()
        #plt.imshow(next_state.squeeze(0).permute(1, 2, 0), interpolation='none')
        #plt.pause(0.002)

        if is_ipython: display.clear_output(wait=True)
        #t_state = torch.from_numpy(state).to(device).unsqueeze(0)
        #t_next_state = torch.from_numpy(next_state).to(device).unsqueeze(0)
        #memory.push(Experience(t_state, action, t_next_state, reward))
        state = next_state
        em.env.render_mode = 'human'
        em.env.render()

        # if memory.can_provide_sample(batch_size):
            # policy_net.train()
            # experiences = memory.sample(batch_size)
            # states, actions, rewards, next_states = extract_tensors(experiences)
            #
            # current_q_values = QValues.get_current(policy_net, states, actions)
            # next_q_values = QValues.get_next(target_net, next_states)
            # target_q_values = (next_q_values * gamma) + rewards
            # current_q_values = current_q_values.squeeze(1)
            # target_q_values = target_q_values.unsqueeze(0)[0]
            #current_q_values = current_q_values.detach()
            #print("target: ", target_q_values, "shape: ", target_q_values.shape)
            #print("q: ", current_q_values, "shape: ", current_q_values.shape)
            #print("??? = ", target_q_values.shape == current_q_values.shape)
            # loss = F.mse_loss(target_q_values, current_q_values)
            # l = loss.item()
            #print("loss:", l)
            # optimizer.zero_grad()
            # loss.backward()
            #nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
            # optimizer.step()
            # policy_net.eval()
        if em.done:
            episodes_durations.append(timestep)
            print('reward: ', timestep)
            #best = plot(episodes_durations, 100, best, eps=0, loss=l)
            #print("best ", best)
            break
    # if episode % target_update == 0:
    #     target_net.load_state_dict(policy_net.state_dict())
    #     target_net.eval()
#joblib.dump(policy_net, 'policy_net.pkl')
em.close()
