import gym
import keyboard

# Set up the CartPole environment
env = gym.make('CartPole-v1', render_mode='human')

# Define key mappings
key_mappings = {'a': 0, 'd': 1}  # 'a' for left, 'd' for right

# Reset the environment to get initial observation
observation = env.reset()

# Control loop
count = 0
while True:
    # Render the environment (optional, depending on whether you want to visualize)
    env.render()

    # Take input from the keyboard
    if keyboard.is_pressed('a'):
        action = key_mappings['a']
    elif keyboard.is_pressed('d'):
        action = key_mappings['d']
    else:
        action = None

    # Take the action in the environment
    if action is not None:
        observation, reward, done, _, info = env.step(action)
        count += 1

        # If the episode is done, reset the environment
        if done:
            print("Episode finished. Resetting environment...")
            observation = env.reset()
            print("your score is: ", count)
            count = 0
