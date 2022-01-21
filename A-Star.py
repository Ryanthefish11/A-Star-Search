import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from collections import defaultdict

# added packages
import heapq
from matplotlib import colors
from matplotlib import pyplot


# 
# 
# ---
# ## [50 pts] Problem 1:  Route-finding
# 
# Consider the map of the area to the west of the Engineering Center given below, with a fairly coarse Cartesian grid superimposed.
# 
# <img src="http://www.cs.colorado.edu/~tonyewong/home/resources/engineering_center_grid_zoom.png" style="width: 800px;"/>
# 
# The green square at $(x,y)=(1,15)$ is the starting location, and you would like to walk from there to the yellow square at $(25,9)$ with the **shortest total path length**. The filled-in blue squares are obstacles, and you cannot walk through those locations.  You also cannot walk outside of this grid.
# 
# Legal moves in the North/South/East/West directions have a step cost of 1. Moves in the diagonal direction (for example, from $(1,15)$ to $(2,14)$) are allowed, but they have a step cost of $\sqrt{2}$. 
# 
# Of course, you can probably do this problem (and likely have to some degree, in your head) without a search algorithm. But that will hopefully provide a useful "sanity check" for your answer.
# 
# #### Part A
# Write a function `adjacent_states(state)`:
# * takes a single argument `state`, which is a tuple representing a valid state in this state space
# * returns in some form the states reachable from `state` and the step costs. How exactly you do this is up to you. One possible format for what this function returns is a dictionary with the keys being the tuple locations and the values of the keys being the step costs. E.g: adjacent_states((1,1)) =  $\{(2,1):1, (2,2):1.414\}$
# 
# Print to the screen the output for `adjacent_states((1,15))`.

# In[2]:


maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])





def adjacent_states(state):
    y = state[0]
    x = state[1] 
    xmax = 50
    ymax = 30
    options = {}

    if(maze[x+1][y] == 0):
        
        options[(y,x+1)] = 1
    
    if(maze[x][y+1] == 0):
        
        options[(y+1,x)] = 1

    if(maze[x+1][y+1] == 0):
        
        options[(y+1,x+1)] = 1.414
    
    if(maze[x-1][y] == 0):
        
        options[(y,x-1)] = 1
    
    if(maze[x][y-1] == 0):
        
        options[(y-1,x)] = 1
    
    if(maze[x-1][y-1] == 0):
        
        options[(y-1,x-1)] = 1.414
        
    if(maze[x-1][y+1] == 0):
        
        options[(y+1, x-1)] = 1.414
    
    if(maze[x+1][y-1] == 0):
        
        options[(y-1,x+1)] = 1.414
    
    return options





print(adjacent_states((1,15)))


# #### Part B
# Three candidate heuristic functions might be:
# 1. `heuristic_cols(state, goal)` = number of columns between the argument `state` and the `goal`
# 1. `heuristic_rows(state, goal)` = number of rows between the argument `state` and the `goal`
# 1. `heuristic_eucl(state, goal)` = Euclidean distance between the argument `state` and the `goal`
# 
# Write a function `heuristic_max(state, goal)` that returns the maximum of all three of these heuristic functions for a given `state` and `goal`.



def heuristic_cols(state, goal):
    return(abs(goal[0] - state[0]))
    
def heuristic_rows(state, goal):
    return(abs(goal[1] - state[1]))
    
def heuristic_eucl(state, goal):
    x1 = state[0]
    x2 = goal[0]
    y1 = state[1]
    y2 = goal[1]
    ans = np.sqrt((x2 - x1)**2 + (y2 - y1) ** 2)
    return ans
    
def heuristic_max(state, goal):
    return max(heuristic_cols(state,goal), heuristic_rows(state,goal), heuristic_eucl(state,goal))




heuristic_cols((1,1),(25,9))
heuristic_rows((1,1),(25,9))
heuristic_eucl((1,1),(25,9))
heuristic_max((1,15),(25,9))


# #### Part C
# Is the Manhattan distance an admissible heuristic function for this problem?  Explain why or why not.

# The Manhattan distance is not an admissible heuristic function for this problem. This is because admissable heuristics must not overestimate the cost of reaching a destination. In this case, the Manhattan distance would be more costly than the Euclidean distance.

# #### Part D
# Use A\* search and the `heuristic_max` heuristic to find the shortest path from the initial state at $(1,15)$ to the goal state at $(25,9)$.


from heapq import heappush, heappop

class PriorityQueue:
    
    def __init__(self, iterable=[]):
        self.heap = []
        for value in iterable:
            heappush(self.heap, (0, value))
    
    def add(self, value, priority=0):
        heappush(self.heap, (priority, value))
    
    def pop(self):
        priority, value = heappop(self.heap)
        return value
    
    def __len__(self):
        return len(self.heap)



def rev_path(curr, start, end):

    reverse_path = [end]
    while end != start:
        end = curr[end]
        reverse_path.append(end)
    return list(reversed(reverse_path))




def astar2(start, goal):
    current = start
    cost = {}
    curr = dict()
    visited = set()
    explored = {}
    dist = {start: 0}
    frontier = PriorityQueue()
    frontier.add(start)
    result = [start]

    while(frontier):
        
        node = frontier.pop()
        if node in visited:
            continue
        if(node == goal):
            return rev_path(curr, start,node), dist[node]
        #frontier = {}
        
        visited.add(node)
        
        key = list(adjacent_states(node).values())
        count = 0
        for neighbor in adjacent_states(node):
            frontier.add(neighbor, priority = dist[node] + 1 + heuristic_max(neighbor,goal))
            count+=1
            if neighbor not in dist or dist[node] + 1 < dist[neighbor]:
                
                dist[neighbor] = dist[node] +key[count-1]
                curr[neighbor] = node
                
            
        

       

final = astar2((1,15),(25,9)) 
#print(final)
print('Path: ', final[0], '\nPath Cost: ', final[1])



# Make a figure depicting the optimal route from the initial state to the goal



def plot(maze, path):
    nrow, ncol = maze.shape
    
    # create colormap
    cmap = colors.ListedColormap(['coral', 'slategray'])

    fig, ax = plt.subplots()
    ax.imshow(maze, cmap=cmap, origin='lower')
    
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k')
    ax.set_xticks(np.arange(-.5, ncol, 1))
    ax.set_yticks(np.arange(-.5, nrow, 1))
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    xval = []
    yval = []
    for i in range(len(path)):
        xval.append(path[i][0])
        yval.append(path[i][1])
    plt.plot(xval,yval,color = 'b')


# In[12]:


line = final[0]

plot(maze, line)
