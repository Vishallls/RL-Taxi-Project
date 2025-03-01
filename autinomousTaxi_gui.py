import gymnasium as gym
import numpy as np
import random
import json
import matplotlib.pyplot as plt

# Initialize environment with graphical rendering
env = gym.make('Taxi-v3', render_mode="rgb_array")

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
exploration_decay_rate = 0.001  

rewards_all_episodes = []

def train_agent():
    global exploration_rate  
    for episode in range(num_episodes):
        state, _ = env.reset()  
        state = int(state)  

        done = False
        total_reward = 0

        for step in range(max_steps_per_episode):
            if random.uniform(0, 1) > exploration_rate:
                action = np.argmax(q_table[state, :])  
            else:
                action = env.action_space.sample()  

            new_state, reward, done, truncated, _ = env.step(action)
            new_state = int(new_state)  

            # Update Q-Table using Bellman Equation
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + \
                                     learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            state = new_state
            total_reward += reward

            if done or truncated:
                break

        # Decay exploration rate
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
        
        rewards_all_episodes.append(total_reward)

        # Show training progress every 1000 episodes
        if episode % 1000 == 0:
            print(f"\nEpisode {episode}: Total Reward = {total_reward}")

    # Save Q-table
    with open("q_table.json", "w") as f:
        json.dump(q_table.tolist(), f)

print("Training Started")
train_agent()
print("Training Finished")

# Save final Q-table
np.save("q_table.npy", q_table)


# Function to Display the Environment for a Fixed Duration
def display_env():
    img = env.render()  # Get RGB array of the environment
    plt.imshow(img)
    plt.axis('off')
    plt.draw()   # Draw the figure
    plt.pause(2)  # Display the image for 10 seconds
    plt.close()   # Close the figure automatically


# Test the trained agent
def test_agent(num_episodes=5):
    for episode in range(num_episodes):
        state, _ = env.reset()  
        state = int(state)

        done = False
        step_count = 0
        total_reward = 0

        print(f"\nEpisode {episode + 1}")
        
        while not done:
            action = np.argmax(q_table[state, :])  # Take best action
            new_state, reward, done, truncated, _ = env.step(action)
            state = new_state
            total_reward += reward
            step_count += 1

            # Display graphical state
            display_env()  

            if done or truncated:
                break

        print(f"Episode {episode + 1} finished in {step_count} steps with total reward: {total_reward}")

print("\nRunning Trained Agent")
test_agent()
