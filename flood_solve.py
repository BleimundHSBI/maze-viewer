"""Programm to calculate the shortest path to an given maze"""

import time

import matplotlib.pyplot as plt
import maze_gen
import maze_plotter
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

COLOR_MAP = {
    "#": np.array([255, 10, 0]),
    "E": np.array([255, 0, 180]),
    " ": np.array([0, 0, 0]),
    "*": np.array([15, 255, 0]),
    "w": np.array([50, 255, 0]),
}

WALL = "#"
ESCAPE = "E"
FREE = " "
MARKER = "*"
WATER = "w"


# settings:
visual = False

coord_escape = (1, 1)
node_maze = []
maze = []

time_on_tree = 0


def is_escape(row, column):
    """returns if the cell(row, column) is an Exit"""
    return maze[row, column] == ESCAPE


def process_direction(direct, prev):
    """checks if direction is escape. If free it is marked. Return True if escape is found"""
    row = direct[0]
    column = direct[1]
    prev_row = prev[0]
    prev_column = prev[1]

    global tree
    global time_on_tree

    # check if in bounds
    if row >= len(maze) or row <= 0 or column >= len(maze[0]) or column <= 0:
        return False

    # check for escape
    if is_escape(row, column):
        node_maze[row][column] = Node(
            "escape", parent=node_maze[prev_row][prev_column], step=(row, column)
        )
        global coord_escape
        coord_escape = (row, column)
        return True

    # mark the next "water" block
    if maze[direct] == FREE:
        time1 = time.time()
        node_maze[row][column] = Node(
            (row, column), parent=node_maze[prev_row][prev_column], step=(row, column)
        )
        time2 = time.time()
        time_on_tree += time2 - time1

        maze[direct] = MARKER

    return False


def single_step():
    """searches for all water cells and process their coresponding directions"""
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i, j] == WATER:
                directions = [
                    (i + 1, j),  # Down
                    (i, j + 1),  # Right
                    (i - 1, j),  # Up
                    (i, j - 1),  # Left
                ]
                for direct in directions:
                    if process_direction(direct, (i, j)) is True:
                        return True
    return False


def solve_maze(start_x, start_y):
    """solves the maze with an flood fill aproach"""
    global node_maze
    node_maze = [None] * len(maze)
    for i in range(len(node_maze)):
        node_maze[i] = [None] * len(maze[0])

    maze[start_x, start_y] = WATER
    node_maze[start_x][start_y] = Node("start", step=(start_x, start_y))

    escape_found = False
    while escape_found is False:
        escape_found = single_step()
        maze_plotter.print_maze_to_display(maze, time_to_sleep=0.05)
        # convert marked cells to water for next step
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if maze[i, j] == MARKER:
                    maze[i, j] = WATER


# generate or load maze
maze, solved = maze_gen.getMaze(20, 20, 50)
print(solved)
# needed for vscode wsl debugger
# maze = maze_plotter.convert_file_to_field(
# "/home/philippbleimund/git/code-experimentation/aud_seminar/Seminar7/field3.txt"
# )

# maze = maze_plotter.convert_file_to_field("field3.txt")

maze_plotter.init(COLOR_MAP, wall=WALL, escape=ESCAPE, free=FREE, marker=MARKER)

if visual is True:
    maze_plotter.init_interactive(maze=maze)

maze_plotter.printArr(maze)

time1 = time.time()
solve_maze(1, 1)
time2 = time.time()

solved_node = node_maze[coord_escape[0]][coord_escape[1]]
root_node = node_maze[1][1]
print(RenderTree(solved_node))

# DotExporter(root_node).to_picture("tmp_graph.png") # before uncommenting check README.md

# get solving path
path = []
working_node = solved_node
while working_node.parent is not None:
    path.append(working_node.step)
    working_node = working_node.parent

print(path)

print("time to solve: " + str(time2 - time1))
print("time spend on tree: " + str(time_on_tree))

if visual is False:
    plt.imshow(maze_plotter.parse_path_to_rgb(path, maze))
    plt.show()
