import gym
import numpy as np
import random
import time
from IPython.display import clear_output

# Initialize environment
env = gym.make('Taxi-v3', render_mode="ansi")  # Text-based rendering

# Creating Q-Table
actions = env.action_space.n
state_space = env.observation_space.n
q_table = np.zeros((state_space, actions))

# Parameters for Q-Learning
num_episodes = 10000
max_steps_per_episode = 100
learning_rate = 0.1
discount_rate = 0.99
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001  # Adjusted decay rate

rewards_all_episodes = []

# Q-Learning Algorithm
for episode in range(num_episodes):
    state = env.reset()[0]  # Extract state from the tuple
    state = int(state)  # Ensure state is an integer
    done = False
    total_reward = 0

    for step in range(max_steps_per_episode):
        # Exploration vs Exploitation trade-off
        if random.uniform(0, 1) > exploration_rate:
            action = np.argmax(q_table[state, :])  # Exploitation
        else:
            action = env.action_space.sample()  # Exploration

        new_state, reward, done, truncated, _ = env.step(action)
        new_state = int(new_state)  # Ensure new_state is an integer

        # Update Q-Table using the Bellman Equation
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                 learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        total_reward += reward

        if done or truncated:
            break

    # Decay the exploration rate using exponential decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(total_reward)

print("***** Training Finished *****")

# Calculate and print average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 1000)
count = 1000

print("Average per thousand episodes:")
for r in rewards_per_thousand_episodes:
    print(count, ":", str(sum(r) / 1000))
    count += 1000

# Visualizing the agent's performance
for episode in range(3):
    state = env.reset()[0]  # Extract state
    state = int(state)
    done = False
    print("Episode:", episode + 1)
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        print(env.render())  # Fixed rendering issue
        time.sleep(0.3)

        action = np.argmax(q_table[state, :])  # Choose best action
        new_state, reward, done, truncated, _ = env.step(action)
        new_state = int(new_state)

        if done or truncated:
            clear_output(wait=True)
            print(env.render())  # Fixed rendering issue
            if reward == 20:  # Success case
                print("**** Reached Goal ****")
            else:
                print("**** Failed ****")
            time.sleep(2)
            break

        state = new_state

env.close()
