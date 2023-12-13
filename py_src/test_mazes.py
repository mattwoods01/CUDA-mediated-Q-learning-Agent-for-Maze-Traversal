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
start_coord = (5, 5)
end_coord = (maze_x-5, maze_y-5)

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
        cmap = mcolors.ListedColormap(['white', 'black', 'green', 'blue', 'red'])
        plt.imshow(self.maze, cmap=cmap)

        # Add start and goal positions as 'S' and 'G'
        plt.text(self.start_position[0], self.start_position[1], 'S', ha='center', va='center', color='red', fontsize=20)
        plt.text(self.goal_position[0], self.goal_position[1], 'G', ha='center', va='center', color='green', fontsize=20)

        # Remove ticks and labels from the axes
        plt.xticks([]), plt.yticks([])

        # Show the plot
        plt.show()



#start_time = time.time()

#end_time = time.time()
#execution_time = end_time - start_time
#print(f"random maze Execution time: {execution_time} seconds")

#maze = Maze(maze_layout, start_coord, end_coord)
#maze.show_maze()


total_ls = []
for i in range(5):
    execution_time_ls = []
    for i in range(10):     
        start_time = time.time()
        for j in range(10000):
            maze_layout = cu_matrix_add.random_array(maze_x, maze_y, start_coord, end_coord, random.randint(1, 1000))
            maze_layout = cu_matrix_add.randomizeZerosCuda(maze_layout, maze_x, maze_y, .2, random.randint(1, 10000))
            maze_layout = cu_matrix_add.generate_feature(maze_layout, maze_x, maze_y, start_coord, end_coord, random.randint(1, 10000))
            maze_layout = cu_matrix_add.gurantee_path(maze_layout, maze_x, maze_y, start_coord, end_coord, random.randint(1, 10000))
            maze_layout = cu_matrix_add.dfs(maze_layout, maze_x, maze_y, start_coord, end_coord, random.randint(1, 10000))
            #epsilon_rates = cu_matrix_add.epsilon_greedy_cuda_ctrl(151, 1.5, 0.01)

        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_ls.append(execution_time)
        #print(f" Execution time: {execution_time} seconds")

    average = sum(execution_time_ls) / len(execution_time_ls)
    total_ls.append(average)
    print(average)

total_average = sum(total_ls) / len(total_ls)
print(total_average)


    
















