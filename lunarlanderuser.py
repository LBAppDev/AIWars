import gym
import keyboard

# Set up the LunarLander environment
env = gym.make('LunarLander-v2', render_mode='human')

# Define key mappings
key_mappings = {'w': 2, 'a': 1, 'd': 3,
                'none': 0}  # 'w' for throttle up, 'a' for rotate left, 'd' for rotate right, 'none' for no action

# Reset the environment to get initial observation
observation = env.reset()

# Control loop
while True:
    # Render the environment (optional, depending on whether you want to visualize)
    env.render()

    # Take input from the keyboard
    if keyboard.is_pressed('w'):
        action = key_mappings['w']
    elif keyboard.is_pressed('a'):
        action = key_mappings['a']
    elif keyboard.is_pressed('d'):
        action = key_mappings['d']
    else:
        action = key_mappings['none']

    # Take the action in the environment
    observation, reward, done, _, info = env.step(action)

    # If the episode is done, reset the environment
    if done:
        print("Episode finished. Resetting environment...")
        observation = env.reset()
