from enum import Enum
from queue import PriorityQueue
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from bresenham import bresenham


def create_grid(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    based on given obstacle data, drone altitude and safety distance
    arguments.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))

    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min + 1)))
    east_size = int(np.ceil((east_max - east_min + 1)))

    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))

    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size-1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size-1)),
                int(np.clip(east - d_east - safety_distance - east_min, 0, east_size-1)),
                int(np.clip(east + d_east + safety_distance - east_min, 0, east_size-1)),
            ]
            grid[obstacle[0]:obstacle[1]+1, obstacle[2]:obstacle[3]+1] = 1

    return grid, int(north_min), int(east_min)


def create_grid_and_edges(data, drone_altitude, safety_distance):
    """
    Returns a grid representation of a 2D configuration space
    along with Voronoi graph edges given obstacle data and the
    drone's altitude.
    """

    # minimum and maximum north coordinates
    north_min = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max = np.ceil(np.max(data[:, 0] + data[:, 3]))
    # print(north_min, north_max)

    # minimum and maximum east coordinates
    east_min = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max = np.ceil(np.max(data[:, 1] + data[:, 4]))
    # print(east_min, east_max)
    # given the minimum and maximum coordinates we can
    # calculate the size of the grid.
    north_size = int(np.ceil((north_max - north_min)))
    east_size = int(np.ceil((east_max - east_min)))
    # print(north_size, east_size)
    # Initialize an empty grid
    grid = np.zeros((north_size, east_size))
    # Center offset for grid
    north_min_center = np.min(data[:, 0])
    east_min_center = np.min(data[:, 1])

    # Points list for Voronoi
    points = []
    # Populate the grid with obstacles
    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(north - d_north - safety_distance - north_min_center),
                int(north + d_north + safety_distance - north_min_center),
                int(east - d_east - safety_distance - east_min_center),
                int(east + d_east + safety_distance - east_min_center),
            ]
            grid[obstacle[0]:obstacle[1], obstacle[2]:obstacle[3]] = 1

            # add center of obstacles to points list
            points.append([north - north_min, east - east_min])

    # TODO: create a voronoi graph based on
    # location of obstacle centres
    graph = Voronoi(points)

    # TODO: check each edge from graph.ridge_vertices for collision
    edges = []
    for v in graph.ridge_vertices:
        p1 = graph.vertices[v[0]]
        p2 = graph.vertices[v[1]]

        cells = list(bresenham(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])))
        hit = False

        for c in cells:
            # First check if we're off the map
            if np.amin(c) < 0 or c[0] >= grid.shape[0] or c[1] >= grid.shape[1]:
                hit = True
                break
            # Next check if we're in collision
            if grid[c[0], c[1]] == 1:
                hit = True
                break

        # If the edge does not hit on obstacle
        # add it to the list
        if not hit:
            # array to tuple for future graph creation step)
            p1 = (p1[0], p1[1])
            p2 = (p2[0], p2[1])
            edges.append((p1, p2))

    return grid, edges

# Assume all actions cost the same.
class Action(Enum):
    """
    An action is represented by a 3 element tuple.

    The first 2 values are the delta of the action relative
    to the current grid position. The third and final value
    is the cost of performing the action.
    """


    WEST = (0, -1, 1)
    EAST = (0, 1, 1)
    NORTH = (-1, 0, 1)
    SOUTH = (1, 0, 1)

    # Diagonal movements  add standard cost of 2 . But later will do sqrt(2) for diagnol movement
    NORTH_WEST = (-1, -1, 2)
    NORTH_EAST = (-1, 1, 2)
    SOUTH_WEST = (1, -1, 2)
    SOUTH_EAST = (1, 1, 2)

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


def valid_actions(grid, current_node):
    """
    Returns a list of valid actions given a grid and current node.
    """
    valid_actions = list(Action)
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node

    #print(" n = ",n," m = ",m )
    #print(" x = ",x," y = ",y )


    # check if the node is off the grid or
    # it's an obstacle

    try:
        if x - 1 < 0 or grid[x - 1, y] == 1:
            #print("Removing North ")
            valid_actions.remove(Action.NORTH)
    except IndexError:
        valid_actions.remove(Action.NORTH)

    try:
        if x + 1 > n or grid[x + 1, y] == 1:
            # print("Removing South ")
            valid_actions.remove(Action.SOUTH)
    except IndexError:
        valid_actions.remove(Action.SOUTH)

    try:
        if y - 1 < 0 or grid[x, y - 1] == 1:
            #print("Removing West ")
            valid_actions.remove(Action.WEST)
    except IndexError:
        valid_actions.remove(Action.WEST)

    try:
        if y + 1 > m or grid[x, y + 1] == 1:
            # print("Removing East ")
            valid_actions.remove(Action.EAST)
    except IndexError:
        valid_actions.remove(Action.EAST)

    try:
        if x - 1 < 0 and y - 1 < 0 or grid[x - 1, y - 1] == 1:
            # print("Removing Nort west  Grid ")
            valid_actions.remove(Action.NORTH_WEST)
    except IndexError:
        valid_actions.remove(Action.NORTH_WEST)

    try:
        if x - 1 < 0 and y + 1 > m or grid[x - 1, y + 1] == 1:
            # print("Removing Nort east Grid  ")
            valid_actions.remove(Action.NORTH_EAST)
    except IndexError:
        valid_actions.remove(Action.NORTH_EAST)

    try:
        if x + 1 > n and y - 1 < 0 or grid[x + 1, y - 1] == 1:
            # print("Removing South west  Grid ")
            valid_actions.remove(Action.SOUTH_WEST)
    except IndexError:
        valid_actions.remove(Action.SOUTH_WEST)

    try:
        if x + 1 > n and y + 1 > m or grid[x + 1, y + 1] == 1:
            # print("Removing South  east Grid ")
            valid_actions.remove(Action.SOUTH_EAST)
    except IndexError:
        valid_actions.remove(Action.SOUTH_EAST)


    return valid_actions


def a_star(grid, h, start, goal):
    """
    Given a grid and heuristic function returns
    the lowest cost path from start to goal.
    """

    path = []
    path_cost = 0
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    #Diagnol cost const to avoid  costly math.sqrt each time on the loop
    diagonal_cost = np.sqrt(2)

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            # Get the new vertexes connected to the current vertex
            for a in valid_actions(grid, current_node):
                #print(" a.delta[0] = ",a.delta[0] ," a.delta[1] = ",a.delta[0], " cost = " , a.cost)
                next_node = (current_node[0] + a.delta[0], current_node[1] + a.delta[1])


                # Adding Diagonal cost  if the movement is diagonal
                if a.cost == 2:
                   new_cost = current_cost + diagonal_cost + h(next_node, goal)
                else:
                   new_cost = current_cost + a.cost + h(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node, a)

    if found:
        # retrace steps
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])

            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************') 
    return path[::-1], path_cost


def a_star_graph(graph, heuristic, start, goal):
    """Modified A* to work with NetworkX graphs."""

    path = []
    queue = PriorityQueue()
    queue.put((0, start))
    visited = set(start)

    branch = {}
    found = False

    while not queue.empty():
        item = queue.get()
        current_cost = item[0]
        current_node = item[1]

        if current_node == goal:
            print('Found a path.')
            found = True
            break
        else:
            for next_node in graph[current_node]:
                #print("Came in Graph !!!!")
                cost = graph.edges[current_node, next_node]['weight']
                new_cost = current_cost + cost + heuristic(next_node, goal)

                if next_node not in visited:
                    visited.add(next_node)
                    queue.put((new_cost, next_node))

                    branch[next_node] = (new_cost, current_node)

    path = []
    path_cost = 0
    if found:

        # retrace steps
        path = []
        n = goal
        path_cost = branch[n][0]
        path.append(goal)
        while branch[n][1] != start:
            path.append(branch[n][1])
            n = branch[n][1]
        path.append(branch[n][1])
    else:
        print('**********************')
        print('Failed to find a path!')
        print('**********************')

    return path[::-1], path_cost

def heuristic(position, goal_position):
    return np.linalg.norm(np.array(position) - np.array(goal_position))

