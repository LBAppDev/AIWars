import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
import joblib
import math


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions, Q_value):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        if Q_value:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
            self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        #print(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=50000, eps_end=0.01,
                 eps_dec=5e-5):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.action_space = [i for i in range(self.n_actions)]
        self.mem_cntr = 0

        #self.Q_eval = DeepQNetwork(self.lr, input_dims, 512, 256, n_actions, True)
        self.Q_eval = joblib.load('Q_eval8.pkl')
        self.Q_target = DeepQNetwork(self.lr, input_dims, 512, 256, n_actions, False)
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        #self.Q_target.eval()

        self.state_memory = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.max_mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.max_mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self, step, eps_step):
        if self.mem_cntr < self.batch_size:
            return
        freq = 300
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval.to(self.Q_eval.device))
        #print('loss: ', loss)
        loss.backward()
        self.Q_eval.optimizer.step()
        #self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        self.epsilon = self.eps_end + (1 - self.eps_end) * math.exp(-1 * eps_step * self.eps_dec)
        if time_step % freq == 0:
            update_target_network(self.Q_target, self.Q_eval)




def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


def update_target_network(target, value):
    tau = 0.8
    for target_param, local_param in zip(target.parameters(), value.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


a = 0

if __name__ == '__main__':
    env = gym.make('LunarLander-v2', render_mode='human')
    agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], lr=0.003, eps_dec=0.0001)
    scores, eps_history = [], []
    n_games = 400
    freq = 128
    #target_update = 640
    eps_step = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()[0]
        time_step = 0
        while not done:
            action = agent.choose_action(observation)

            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_,done)
            time_step += 1
            eps_step += 1
            env.render()
            agent.learn(time_step, eps_step)

            #if time_step % target_update == 0:


            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)



        avg_score = np.mean(scores[-100:])
        if i > 490:
            a = avg_score
        # if i % 100 == 0:
        #     print("episode i: ", i, "avg score: ", avg_score)
        print('episode ', i, 'score %.2f'%score, 'average score %.2f'%avg_score, 'epsilon %.2f'%agent.epsilon)
    x = [i + 1 for i in range(n_games)]
    joblib.dump(agent.Q_eval, 'Q_eval9.pkl')
    file_name = 'lunarLander_target.png'
    plot_learning_curve(x, scores, eps_history, file_name)
    print("score:", a)

