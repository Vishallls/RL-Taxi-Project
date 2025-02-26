import numpy as np
import tkinter as tk
import math
import random

# Grid size
GRID_SIZE = 10
OBSTACLES = set()
START_POS = (0, 0)
GOAL_POS = (GRID_SIZE - 1, GRID_SIZE - 1)
AGENT_POS = START_POS

# UI Emojis
AGENT_EMOJI = "🚗"
OBSTACLE_EMOJI = "🚧"
GOAL_EMOJI = "🏁"
EMPTY_CELL = "⬜"

# Q-learning parameters
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))  
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  
ALPHA, GAMMA, EPSILON = 0.3, 0.9, 0.2  

# Step and Score
step_count = 0
total_score = 0

# Tkinter GUI setup
root = tk.Tk()
root.title("RL Obstacle Avoidance Robot 🚗")

# Function to draw grid with emojis
def draw_grid():
    for widget in grid_frame.winfo_children():
        widget.destroy()  

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            cell_text = EMPTY_CELL
            if (x, y) == AGENT_POS:
                cell_text = AGENT_EMOJI
            elif (x, y) == GOAL_POS:
                cell_text = GOAL_EMOJI
            elif (x, y) in OBSTACLES:
                cell_text = OBSTACLE_EMOJI

            label = tk.Label(grid_frame, text=cell_text, font=("Arial", 16), width=3, height=1, bg="#D0E0F0")
            label.grid(row=x, column=y, padx=2, pady=2)
            label.bind("<Button-1>", lambda event, row=x, col=y: toggle_obstacle(row, col))  

    step_label.config(text=f"Steps: {step_count}")
    score_label.config(text=f"Score: {total_score}")

# Function to calculate Euclidean distance
def calculate_distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

# Function to calculate reward
def calculate_reward(prev_pos, new_pos):
    global step_count

    prev_distance = calculate_distance(prev_pos, GOAL_POS)
    new_distance = calculate_distance(new_pos, GOAL_POS)

    reward = -1  # Small penalty for each step
    if new_pos == GOAL_POS:
        reward += 5  # Bonus for reaching the goal
    elif new_distance < prev_distance:
        reward += 2  # Reward for moving towards the goal
    else:
        reward -= 2  # Penalty for moving away

    if new_pos in OBSTACLES:
        reward -= 10  # Heavy penalty for hitting an obstacle

    return reward

# Function to get valid moves
def get_valid_moves(pos):
    x, y = pos
    valid_moves = []
    
    for i, (dx, dy) in enumerate(ACTIONS):
        new_x, new_y = x + dx, y + dy
        if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and (new_x, new_y) not in OBSTACLES:
            valid_moves.append((i, (new_x, new_y)))  

    return valid_moves

# Function to train the Q-table
def train_agent(episodes=500):
    global Q_table

    for _ in range(episodes):
        state = START_POS
        for _ in range(100):  
            valid_moves = get_valid_moves(state)
            if not valid_moves:
                break  

            if np.random.rand() < EPSILON:
                action, next_state = random.choice(valid_moves)  
            else:
                action, next_state = max(valid_moves, key=lambda move: Q_table[state[0], state[1], move[0]])  

            reward = calculate_reward(state, next_state)

            # Update Q-table using Bellman equation
            best_next_action = max(get_valid_moves(next_state), key=lambda move: Q_table[next_state[0], next_state[1], move[0]], default=(0, next_state))[0]
            Q_table[state[0], state[1], action] += ALPHA * (
                reward + GAMMA * Q_table[next_state[0], next_state[1], best_next_action] - Q_table[state[0], state[1], action]
            )

            state = next_state  

    print("Training Complete!")

# Function to move agent autonomously
def move_agent():
    global AGENT_POS, step_count, total_score

    prev_pos = AGENT_POS
    valid_moves = get_valid_moves(AGENT_POS)

    if not valid_moves:
        status_label.config(text="🚗 No valid moves! Reset obstacles.")
        return

    # Choose best move from trained Q-table
    action, new_pos = max(valid_moves, key=lambda move: Q_table[AGENT_POS[0], AGENT_POS[1], move[0]])  

    # Only increment step count if the agent actually moves
    if new_pos != AGENT_POS:  
        step_count += 1
        print(f"Step {step_count}: Moving to {new_pos}")  # Debugging

    reward = calculate_reward(prev_pos, new_pos)
    total_score += reward
    AGENT_POS = new_pos  

    draw_grid()

    if AGENT_POS == GOAL_POS:
        status_label.config(text="🚗 Reached the Goal! 🏁")
        print(f"Total Steps Taken: {step_count}")  

# Function to toggle obstacles by clicking
def toggle_obstacle(x, y):
    if (x, y) == START_POS or (x, y) == GOAL_POS:
        return  

    if (x, y) in OBSTACLES:
        OBSTACLES.remove((x, y))  
    else:
        OBSTACLES.add((x, y))  

    draw_grid()

# Function to set goal position
def set_goal():
    global GOAL_POS
    x, y = int(goal_x.get()), int(goal_y.get())
    if (x, y) not in OBSTACLES:
        GOAL_POS = (x, y)
        draw_grid()

# Layout
grid_frame = tk.Frame(root, bg="#D0E0F0")
grid_frame.pack()

control_frame = tk.Frame(root)
control_frame.pack()

tk.Label(control_frame, text="Goal (x, y):").grid(row=0, column=0)
goal_x, goal_y = tk.Entry(control_frame, width=5), tk.Entry(control_frame, width=5)
goal_x.grid(row=0, column=1)
goal_y.grid(row=0, column=2)
tk.Button(control_frame, text="Set Goal", command=set_goal).grid(row=0, column=3)

# Training button
tk.Button(root, text="Train Agent 🏆", command=lambda: train_agent(500), font=("Arial", 12)).pack()

# Move Step button (Autonomous movement)
tk.Button(root, text="Move Step 🚗", command=move_agent, font=("Arial", 12)).pack()

step_label = tk.Label(root, text="Steps: 0", font=("Arial", 14))
step_label.pack()
score_label = tk.Label(root, text="Score: 0", font=("Arial", 14))
score_label.pack()
status_label = tk.Label(root, text="", font=("Arial", 14))
status_label.pack()

# Initial draw
draw_grid()
root.mainloop()
