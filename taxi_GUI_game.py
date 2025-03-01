import gymnasium as gym
import tkinter as tk

# Create Taxi environment
env = gym.make('Taxi-v3', render_mode='human')
env.reset()

# Action mapping for arrow keys and interactions
action_map = {
    'Up': 1,      # Move north
    'Down': 0,    # Move south
    'Right': 2,   # Move east
    'Left': 3,    # Move west,
    'p': 4,       # Pick up
    'o': 5        # Drop off
}

# Variables to track rounds, steps, and rewards
round_num = 1
steps_taken = 0
total_reward = 0
max_rounds = 2

# Variables to track the best round
best_round = 0
best_steps = float('inf')  # Fewer steps is better
best_reward = float('-inf')

# Function to perform action and update display
def perform_action(action):
    global steps_taken, total_reward, round_num, best_round, best_steps, best_reward
    state, reward, done, _, _ = env.step(action)
    steps_taken += 1
    total_reward += reward
    env.render()  # Re-render in the same window
    
    # Check if drop-off is successful
    if done:
        # Display summary of current round
        print(f"\nRound {round_num} Completed!")
        print(f"Steps Taken: {steps_taken}")
        print(f"Total Reward: {total_reward}")
        
        # Check if this round is the best
        if total_reward > best_reward:
            best_round = round_num
            best_steps = steps_taken
            best_reward = total_reward
        
        # Increment round number
        round_num += 1
        
        # Check if maximum rounds reached
        if round_num > max_rounds:
            print("\nSimulation Completed!")
            print(f"\nBest Round: {best_round}")
            print(f"Steps Taken: {best_steps}")
            print(f"Total Reward: {best_reward}")
            root.destroy()  # Close Tkinter window
            env.close()    # Close Taxi-v3 environment
            return
        
        # Reset for next round
        steps_taken = 0
        total_reward = 0
        env.reset()
        env.render()

# Tkinter GUI setup for key capture only
root = tk.Tk()
root.title("Taxi-v3 Simulation")
root.geometry("1x1")  # Minimal window for key capture

# Key event handling for real-time movement
def on_key(event):
    key = event.keysym
    if key in action_map:
        perform_action(action_map[key])

root.bind_all('<Key>', on_key)

# Initial render of the environment
env.render()

# Main loop for Tkinter to capture keys continuously
root.mainloop()
