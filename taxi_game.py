import gymnasium as gym
import numpy as np
import json

# Initialize environment with ANSI render mode for text-based display
env = gym.make('Taxi-v3', render_mode="ansi")

# Load Q-Table
try:
    with open("q_table.json", "r") as f:
        q_table = np.array(json.load(f))
except FileNotFoundError:
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Mapping for passenger and drop-off locations
locations = {
    0: "R (Red)",
    1: "G (Green)",
    2: "Y (Yellow)",
    3: "B (Blue)",
    4: "In Taxi"
}

# Function to decode the state
def decode_state(state):
    taxi_row = (state // 25) % 5
    taxi_col = (state // 5) % 5
    pass_loc = (state // 125) % 5
    dest = state % 4
    return taxi_row, taxi_col, pass_loc, dest

# Function to display environment as a matrix with passenger and drop-off info
def display_env(state):
    taxi_row, taxi_col, pass_loc, dest = decode_state(state)
    env_desc = env.render().split('\n')
    print("\n".join(env_desc))
    print("\n--- Information ---")
    print(f"Taxi Position: Row {taxi_row}, Column {taxi_col}")
    print(f"Passenger Location: {locations[pass_loc]}")
    print(f"Drop-off Location: {locations[dest]}")

# Game loop for multiple rounds
def play_game(rounds=2):
    best_score = float('inf')
    best_reward = float('-inf')
    
    for round_num in range(1, rounds + 1):
        state, _ = env.reset()
        state = int(state)
        done = False
        step_count = 0
        total_reward = 0
        has_passenger = False

        print(f"\nRound {round_num} starts!")
        display_env(state)  # Display initial state

        while not done:
            # Get user input for action
            print("\nControls: w = Up, s = Down, a = Left, d = Right, p = Pick Up, o = Drop Off")
            action_input = input("Enter action: ").lower()
            
            if action_input == 'w':
                action = 1  # Move Up
            elif action_input == 's':
                action = 0  # Move Down
            elif action_input == 'a':
                action = 3  # Move Left
            elif action_input == 'd':
                action = 2  # Move Right
            elif action_input == 'p':
                action = 4  # Pick Up Passenger
            elif action_input == 'o':
                action = 5  # Drop Off Passenger
            else:
                print("Invalid input! Try again.")
                continue

            # Perform action
            new_state, reward, done, truncated, _ = env.step(action)
            state = int(new_state)
            total_reward += reward
            step_count += 1

            # Display environment after each step
            display_env(state)

            # Update has_passenger status
            if action == 4:
                taxi_row, taxi_col, pass_loc, dest = decode_state(state)
                if pass_loc == 4:  # In Taxi
                    print("Passenger picked up!")
                    has_passenger = True
                else:
                    print("No passenger to pick up here!")
            
            if action == 5:
                taxi_row, taxi_col, pass_loc, dest = decode_state(state)
                if pass_loc == 4 and (taxi_row, taxi_col) == [(0, 0), (0, 4), (4, 0), (4, 3)][dest]:
                    print("Passenger dropped off successfully!")
                    print(f"Round {round_num} Over! Steps: {step_count}, Total Reward: {total_reward}")
                    has_passenger = False
                    done = True
                else:
                    print("Incorrect drop-off location!")

            if done or truncated:
                if step_count < best_score or (step_count == best_score and total_reward > best_reward):
                    best_score = step_count
                    best_reward = total_reward
                break
        
        # Display score for the current round
        print(f"\n--- Round {round_num} Score ---")
        print(f"Steps Taken: {step_count}")
        print(f"Total Reward: {total_reward}")

    print(f"\nGame Over! Best Score: {best_score} steps, Best Reward: {best_reward}")

print("Starting Taxi Game...")
play_game()
