import pygame
import numpy as np
import random
import time

# Initialize Pygame
pygame.init()

# Screen Settings
step_count = 0  # Initialize step counter here
WIDTH, HEIGHT = 800, 450
GRID_SIZE = 15
CELL_SIZE = WIDTH // GRID_SIZE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RL Obstacle Avoidance - Enhanced")

# Load Images
bg = pygame.image.load("day_background.png")
car_img = pygame.image.load("car.png")
goal_img = pygame.image.load("trophy.png")
barrier_img = pygame.image.load("barrier.png")  # Load the new barrier image

# Resize images to fit grid cells
bg = pygame.transform.scale(bg, (WIDTH, HEIGHT))
car_img = pygame.transform.scale(car_img, (CELL_SIZE, CELL_SIZE))
goal_img = pygame.transform.scale(goal_img, (CELL_SIZE, CELL_SIZE))
barrier_img = pygame.transform.scale(barrier_img, (CELL_SIZE, CELL_SIZE))  # Resize barrier image

# RL Settings
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Left, Right, Up, Down
ALPHA, GAMMA, EPSILON = 0.5, 0.9, 0.8  # Increased epsilon to encourage exploration
TRAIN_EPISODES = 1500  # Increased for better exploration
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

# Game Elements
START_POS = (1, 1)
GOAL_POS = (8, 8)
AGENT_POS = START_POS
OBSTACLES = set()

# Pygame Font Setup
font = pygame.font.Font(None, 40)

# Function to train Q-table
def train_q_table():
    global Q_table
    for _ in range(TRAIN_EPISODES):
        x, y = START_POS
        for _ in range(100):  # Max steps per episode
            if (x, y) == GOAL_POS:
                break
            
            # Exploration vs Exploitation
            if np.random.rand() < EPSILON:
                action = np.random.choice(4)  # Explore randomly
            else:
                action = np.argmax(Q_table[x, y])  # Exploit the best-known action
            
            # Move based on chosen action
            new_x, new_y = x + ACTIONS[action][0], y + ACTIONS[action][1]
            
            # Check if new position is valid
            if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and (new_x, new_y) not in OBSTACLES:
                # Define reward
                reward = 100 if (new_x, new_y) == GOAL_POS else -1
                
                # Update Q-table based on new position
                best_next_action = np.argmax(Q_table[new_x, new_y])
                Q_table[x, y, action] = Q_table[x, y, action] + ALPHA * (
                    reward + GAMMA * Q_table[new_x, new_y, best_next_action] - Q_table[x, y, action]
                )
                x, y = new_x, new_y
    print("Training Complete!")

# Function to move agent
def move_agent():
    global AGENT_POS, step_count  # Include step_count as global
    x, y = AGENT_POS
    action = np.argmax(Q_table[x, y])  # Choose the best action based on the Q-table

    # Debugging: Print the action and the Q-table values
    print(f"Q-values at position {x, y}: {Q_table[x, y]}")
    print(f"Action taken: {action} (Direction: {ACTIONS[action]})")

    new_x, new_y = x + ACTIONS[action][0], y + ACTIONS[action][1]
    if 0 <= new_x < GRID_SIZE and 0 <= new_y < GRID_SIZE and (new_x, new_y) not in OBSTACLES:
        animate_movement(AGENT_POS, (new_x, new_y))
        AGENT_POS = (new_x, new_y)
        
        score = 100 if AGENT_POS == GOAL_POS else -1
        step_count += 1  # Increment step counter
        print(f"Step: {step_count}, Position: {AGENT_POS}, Score: {score}")
        
    if AGENT_POS == GOAL_POS:
        print(f"🚗 Reached the Goal in {step_count} steps! 🏆")
        time.sleep(1)
        reset_grid()
        step_count = 0  # Reset step counter

# Function to animate movement
def animate_movement(start, end):
    start_x, start_y = start[1] * CELL_SIZE, start[0] * CELL_SIZE
    end_x, end_y = end[1] * CELL_SIZE, end[0] * CELL_SIZE
    steps = 10

    for i in range(1, steps + 1):
        temp_x = start_x + (end_x - start_x) * (i / steps)
        temp_y = start_y + (end_y - start_y) * (i / steps)
        screen.blit(bg, (0, 0))
        draw_grid()
        screen.blit(car_img, (temp_x, temp_y))
        pygame.display.update()
        time.sleep(0.02)

# Function to reset grid
def reset_grid():
    global AGENT_POS
    AGENT_POS = START_POS
    print("🔄 Grid Reset! Ready for Retraining.")

# Function to set a new goal
def set_goal(pos):
    global GOAL_POS
    x, y = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
    if (x, y) not in OBSTACLES:
        GOAL_POS = (x, y)
        print(f"🏆 New Goal Set at: {GOAL_POS}")

# Function to toggle obstacles
def toggle_obstacle(pos):
    x, y = pos[1] // CELL_SIZE, pos[0] // CELL_SIZE
    if (x, y) != START_POS and (x, y) != GOAL_POS:
        if (x, y) in OBSTACLES:
            OBSTACLES.remove((x, y))
        else:
            OBSTACLES.add((x, y))
        print(f"Obstacles: {OBSTACLES}")

# Function to draw grid and UI
def draw_grid():
    screen.blit(bg, (0, 0))

    # Draw obstacles using the barrier image
    for obs in OBSTACLES:
        screen.blit(barrier_img, (obs[1] * CELL_SIZE, obs[0] * CELL_SIZE))

    screen.blit(goal_img, (GOAL_POS[1] * CELL_SIZE, GOAL_POS[0] * CELL_SIZE))
    screen.blit(car_img, (AGENT_POS[1] * CELL_SIZE, AGENT_POS[0] * CELL_SIZE))

    # Instructions
    text1 = font.render("'T': Train | 'M': Move | 'R': Reset", True, (255, 255, 255))
    text2 = font.render("L-Click: Obstacle | R-Click: Goal", True, (255, 255, 255))
    screen.blit(text1, (50, HEIGHT - 80))
    screen.blit(text2, (50, HEIGHT - 40))

# Main Loop
running = True
while running:
    screen.fill((255, 255, 255))
    draw_grid()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                train_q_table()
            if event.key == pygame.K_m:
                move_agent()
            if event.key == pygame.K_r:
                reset_grid()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                toggle_obstacle(pygame.mouse.get_pos())
            elif event.button == 3:
                set_goal(pygame.mouse.get_pos())

    pygame.display.update()

pygame.quit()
