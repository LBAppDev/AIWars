import numpy as np
import gym
import random
import time
from gym.envs.registration import register

# Define a new gym environment with deterministic transitions
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': True},
)

# Create the environment
env = gym.make('FrozenLakeNotSlippery-v0', render_mode="human")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n
print(action_space_size, state_space_size)

max_num_episodes = 48000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
q_table = np.load('qtable.npy')
print(q_table)
for episode in range(3):
    state = env.reset()
    state = state[0]
    done = False
    print("*******episode ", episode + 1, "********\n\n")
    time.sleep(1)
    for step in range(max_steps_per_episode):

        env.render()
        time.sleep(0.3)

        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, infor = env.step(action)
        if done:

            env.render()
            if reward == 1:
                print("you have reached the goal")
                time.sleep(3)
            else:
                print("you fell through a hole")
                time.sleep(3)

            break
        state = new_state
env.close()
