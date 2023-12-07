import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import time
import sys
sys.path.append("../build/src/Maze_cuda_functions/release")
import cu_matrix_add
import random

# Create any maze layout you'd like, here's an example
maze_x = 50
maze_y = 50
start_coord = (0, 0)
end_coord = (maze_x-1, maze_y-1)

maze_layout = cu_matrix_add.random_array(maze_x, maze_y, random.randint(1, 1000))
maze_layout = cu_matrix_add.dfs(maze_layout, maze_x, maze_y, 0, 0, maze_x-5, maze_y-5, random.randint(1, 10000))
maze_layout = cu_matrix_add.randomizeZerosCuda(maze_layout, maze_x, maze_y, .20, random.randint(1, 10000))



print(maze_layout)

# Actions the agent can take: Up, Down, Left, Right. Each action is represented as a tuple of two values: (row_change, column_change)
actions = [(-1, 0), # Up: Moving one step up, reducing the row index by 1
          (1, 0),   # Down: Moving on step down, increasing the row index by 1
          (0, -1),  # Left: Moving one step to the left, reducing the column index by 1
          (0, 1)]   # Right: Moving one step to the right, increasing the column index by 1

goal_reward = 100
wall_penalty = -10
step_penalty = -1
num_episodes = 100
# Create an instance of the maze and set the starting and ending positions

# Visualize the maze

class Maze:
    def __init__(self, maze, start_position, goal_position):
        # Initialize Maze object with the provided maze, start_position, and goal position
        self.maze = maze
        self.maze_height = maze_layout.shape[0] # Get the height of the maze (number of rows)
        self.maze_width = maze_layout.shape[1]  # Get the width of the maze (number of columns)
        self.start_position = start_position    # Set the start position in the maze as a tuple (x, y)
        self.goal_position = goal_position      # Set the goal position in the maze as a tuple (x, y)

    def show_maze(self):
        # Visualize the maze using Matplotlib
        plt.figure(figsize=(5,5))

        # Display the maze as an image in grayscale ('gray' colormap)
        cmap = mcolors.ListedColormap(['white', 'black', 'red'])
        plt.imshow(self.maze, cmap=cmap)

        # Add start and goal positions as 'S' and 'G'
        plt.text(self.start_position[0], self.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[0], self.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)

        # Remove ticks and labels from the axes
        plt.xticks([]), plt.yticks([])

        # Show the plot
        plt.show()

maze = Maze(maze_layout, start_coord, end_coord)
maze.show_maze()
epsilon_rates = cu_matrix_add.epsilon_greedy_cuda(101, 1.3, 0.01)


class QLearningAgent: 
    def __init__(self, maze, learning_rate=0.7, discount_factor=0.9, exploration_start=1.3, exploration_end=0.01, num_episodes=100):
        # Initialize the Q-learning agent with a Q-table containing all zeros
        # where the rows represent states, columns represent actions, and the third dimension is for each action (Up, Down, Left, Right)
        self.q_table = np.zeros((maze.maze_height, maze.maze_width, 4)) # 4 actions: Up, Down, Left, Right
        self.learning_rate = learning_rate          # Learning rate controls how much the agent updates its Q-values after each action
        self.discount_factor = discount_factor      # Discount factor determines the importance of future rewards in the agent's decisions
        self.exploration_start = exploration_start  # Exploration rate determines the likelihood of the agent taking a random action
        self.exploration_end = exploration_end
        self.num_episodes = num_episodes

    def get_action(self, state, current_episode): # State is tuple representing where agent is in maze (x, y)
        exploration_rate = epsilon_rates[current_episode]

        # Select an action for the given state either randomly (exploration) or using the Q-table (exploitation)
        if np.random.rand() < exploration_rate:
            return np.random.randint(4) # Choose a random action (index 0 to 3, representing Up, Down, Left, Right)
        else:
            return np.argmax(self.q_table[state]) # Choose the action with the highest Q-value for the given state

    def update_q_table(self, state, action, next_state, reward):
        state_x, state_y = state
        next_state_x, next_state_y = next_state
        # Find the best next action by selecting the action that maximizes the Q-value for the next state
        best_next_action = np.argmax(self.q_table[next_state])
        # Get the current Q-value for the current state and action
        current_q_value = self.q_table[state][action]
        next_q_value = self.q_table[next_state][best_next_action]


        #self.q_table[state][action] = cu_matrix_add.update_q_table_gpu(current_q_value, next_q_value, state_x, state_y, best_next_action, next_state_x, next_state_y, reward, self.learning_rate, self.discount_factor)[0]
        self.q_table[state][action] = current_q_value + self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - current_q_value)
 


def finish_episode(agent, maze, current_episode, train=True):
    # Initialize the agent's current state to the maze's start position
    current_state = maze.start_position
    is_done = False
    episode_reward = 0
    episode_step = 0
    path = [current_state]

    # Continue until the episode is done
    while not is_done:
        # Get the agent's action for the current state using its Q-table
        action = agent.get_action(current_state, current_episode)
        # Compute the next state based on the chosen action
        next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])

        # Check if the next state is out of bounds or hitting a wall
        if next_state[0] < 0 or next_state[0] >= maze.maze_height or next_state[1] < 0 or next_state[1] >= maze.maze_width or maze.maze[next_state[1]][next_state[0]] == 1:
            reward = wall_penalty
            next_state = current_state
        # Check if the agent reached the goal:
        elif next_state == (maze.goal_position):
            path.append(current_state)
            reward = goal_reward
            is_done = True
        # The agent takes a step but hasn't reached the goal yet
        else:
            path.append(current_state)
            reward = step_penalty

        # Update the cumulative reward and step count for the episode
        episode_reward += reward
        episode_step += 1

        # Update the agent's Q-table if training is enabled
        if train == True:
            agent.update_q_table(current_state, action, next_state, reward)
            
        # Move to the next state for the next iteration
        current_state = next_state

    # Return the cumulative episode reward, total number of steps, and the agent's path during the simulation
    return episode_reward, episode_step, path

# This function evaluates an agent's performance in the maze. The function simulates the agent's movements in the maze,
# updating its state, accumulating the rewards, and determining the end of the episode when the agent reaches the goal position.
# The agent's learned path is then printed along with the total number of steps taken and the total reward obtained during the
# simulation. The function also visualizes the maze with the agent's path marked in blue for better visualization of the
# agent's trajectory.

def test_agent(agent, maze, num_episodes=1):
    # Simulate the agent's behavior in the maze for the specified number of episodes
    episode_reward, episode_step, path = finish_episode(agent, maze, num_episodes, train=False)

    # Print the learned path of the agent
    #print("Learned Path:")
    #for row, col in path:
    #    print(f"({row}, {col})-> ", end='')
    print("Goal!")

    print("Number of steps:", episode_step)
    print("Total reward:", episode_reward)

    # Clear the existing plot if any
    if plt.gcf().get_axes():
        plt.cla()

    # Visualize the maze using matplotlib
    # plt.figure(figsize=(5,5))
    cmap = mcolors.ListedColormap(['white', 'black', 'red'])
    plt.imshow(maze.maze, cmap=cmap)

    # Mark the start position (red 'S') and goal position (green 'G') in the maze
    plt.text(maze.start_position[0], maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
    plt.text(maze.goal_position[0], maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)

    # Mark the agent's path with blue '#' symbols
    for position in path:
        plt.text(position[0], position[1], "X", va='center',ha='center', color='blue', fontsize=15)

    # Remove axis ticks and grid lines for a cleaner visualization
    plt.xticks([]), plt.yticks([])
    plt.grid(color='black', linewidth=2)
    plt.show()

    return episode_step, episode_reward, path

def train_agent(agent, maze, num_episodes=100):
    # Lists to store the data for plotting
    episode_rewards = []
    episode_steps = []

    # Loop over the specified number of episodes
    for episode in range(num_episodes):
        episode_reward, episode_step, path = finish_episode(agent, maze, episode, train=True)

        # Store the episode's cumulative reward and the number of steps taken in their respective lists
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

    # Plotting the data after training is completed
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward per Episode')

    average_reward = sum(episode_rewards) / len(episode_rewards)
    print(f"The average reward is: {average_reward}")

    plt.subplot(1, 2, 2)
    plt.plot(episode_steps)
    plt.xlabel('Episode')
    plt.ylabel('Steps Taken')
    plt.ylim(0, 10000)
    plt.title('Steps per Episode')

    average_steps = sum(episode_steps) / len(episode_steps)
    print(f"The average steps is: {average_steps}")

    plt.tight_layout()
    plt.show()



class MazeAnimation:
    def __init__(self, maze, agent, num_episodes):
        self.maze = maze
        self.agent = agent
        self.num_episodes = num_episodes
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        # self.path = []
        self.animation = FuncAnimation(self.fig, self.update, frames=self.explore_maze, repeat=False, blit=False, interval=500)
        #plt.show()

    def update(self, frame):
        state, path, episode, episode_step = frame
        self.ax.clear()
        cmap = mcolors.ListedColormap(['white', 'black', 'red'])
        self.ax.imshow(self.maze.maze, cmap=cmap)
        self.ax.text(self.maze.start_position[0], self.maze.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
        self.ax.text(self.maze.goal_position[0], self.maze.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)
        for position in path:
            self.ax.text(position[0], position[1], "#", va='center', color='blue', fontsize=20)
        # self.ax.text(state[0], state[1], 'A', ha='center', va='center', 
        self.ax.set_title(f'Episode: {episode + 1}, Steps: {episode_step}')  # Set the title with the episode number
        self.ax.set_xticks([]), self.ax.set_yticks([])
        self.ax.grid(color='black', linewidth=2)

    def explore_maze(self):
        current_state = self.maze.start_position
        for episode in range(self.num_episodes):  # Change the number of episodes as needed
            episode_reward, episode_step, path = finish_episode(self.agent, self.maze, episode)
            # self.path = path
            # for step in range(episode_step):
            #     action = self.agent.get_action(current_state, episode)
            #     next_state = (current_state[0] + actions[action][0], current_state[1] + actions[action][1])
            #     reward = maze.maze[next_state[1]][next_state[0]]
            yield current_state, path, episode, episode_step
            #     current_state = next_state

# Create instances of the maze and the Q-learning agent

agent = QLearningAgent(maze)
test_agent(agent, maze)
start_time = time.time()
train_agent(agent, maze, num_episodes=num_episodes)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
episode_step, episode_reward, path = test_agent(agent, maze, num_episodes=num_episodes)
# Create an instance of the maze animation
maze_animation = MazeAnimation(maze, agent, num_episodes)









