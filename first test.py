import numpy as np
import gym
import random
import time
from IPython.display import clear_output
from gym.envs.registration import register

# Define a new gym environment with deterministic transitions
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True},
)

# Create the environment
env = gym.make('FrozenLakeNotSlippery-v0')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

max_num_episodes = 48000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_of_all_episodes = []

# Q-learning
for episode in range(max_num_episodes):
    state = env.reset()
    state = state[0]
    done = False
    rewards_current_episode = 0
    for step in range(max_steps_per_episode):

        # exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, truncated, info = env.step(action)
        # update Q-table for Q(s,a)
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        state = new_state
        rewards_current_episode += reward

        if done or truncated:
            break
    # exploration rate decay
    exploration_rate = (min_exploration_rate + (max_exploration_rate - min_exploration_rate) *
                        np.exp(-exploration_decay_rate * episode))
    rewards_of_all_episodes.append(rewards_current_episode)

e = 1
for r in rewards_of_all_episodes:
    if r == 1:
        print(e, " ", r)
    e += 1
print(q_table)

rewards_per_thousand_episodes = np.split(np.array(rewards_of_all_episodes), max_num_episodes / 1000)
count = 1000
print("********* avrage reward per 1000 episodes ********")
for r in rewards_per_thousand_episodes:
    print(count, ":", str(sum(r/1000)))
    count += 1000

np.save('qtable.npy', q_table)


