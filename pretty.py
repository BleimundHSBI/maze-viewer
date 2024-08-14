"""Programm to calculate the shortest path to an given maze"""

import time
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import matplotlib.pyplot as plt

import maze_gen
import maze_plotter

COLOR_MAP = {
    "#": np.array([0, 0, 0]),
    "E": np.array([255, 0, 180]),
    " ": np.array([255, 255, 255]),
    "*": np.array([15, 255, 0]),
    "w": np.array([50, 255, 0]),
}

WALL = "#"
ESCAPE = "E"
FREE = " "
MARKER = "*"
WATER = "w"


# settings:
visual = True

coord_escape = (1, 1)
node_maze = []
maze = []


def is_escape(row, column):
    """returns if the cell(row, column) is an Exit"""
    return maze[row, column] == ESCAPE


def process_direction(direct, prev, age):
    """checks if direction is escape. If free it is marked. Return True if escape is found"""
    row = direct[0]
    column = direct[1]
    prev_row = prev[0]
    prev_column = prev[1]

    global tree

    # check if in bounds
    if row >= len(maze) or row < 0 or column >= len(maze[0]) or column < 0:
        return False

    # check for escape
    if is_escape(row, column):
        node_maze[row][column] = Node("escape", parent=node_maze[prev_row][prev_column], step=(row, column), age=age)
        global coord_escape
        coord_escape = (row, column)
        return True

    # mark the next "water" block
    if maze[direct] == FREE:
        node_maze[row][column] = Node(
            (row, column), parent=node_maze[prev_row][prev_column], step=(row, column), age=age
        )

        maze[direct] = MARKER

    return False


def single_step(age):
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
                    if process_direction(direct, (i, j), age) is True:
                        return True
    return False


def solve_maze(start_x, start_y):
    """solves the maze with an flood fill aproach"""
    global node_maze
    node_maze = [None] * len(maze)
    for i in range(len(node_maze)):
        node_maze[i] = [None] * len(maze[0])

    maze[start_x, start_y] = WATER
    node_maze[start_x][start_y] = Node("start", step=(start_x, start_y), age=0)

    escape_found = False
    age = 1
    while escape_found is False:
        escape_found = single_step(age)
        maze_plotter.print_maze_with_age_to_display(maze, node_maze[start_x][start_y], node_maze)
        # convert marked cells to water for next step
        for i in range(len(maze)):
            for j in range(len(maze[0])):
                if maze[i, j] == MARKER:
                    maze[i, j] = WATER

        age += 1


# generate or load maze
maze, solved = maze_gen.getMaze(20, 20, 0)
# print(solved)
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

if visual is False:
    plt.imshow(maze_plotter.parse_path_to_rgb(path, maze))
    plt.show()
elif visual is True:
    if False:
        maze_plotter.print_maze_with_age_to_display(
            maze,
            node_maze[1][1],
            node_maze,
            # top_text="length: " + str(len(path)),
            time_to_sleep=20,
            path_to_save="fig_clean.svg",
        )
    else:
        background = maze_plotter.parse_age_to_rgb(node_maze[1][1], node_maze, maze)
        maze_plotter.print_path_to_display(
            maze,
            path,
            background=background,
            # top_text="length: " + str(len(path)),
            time_to_sleep=20,
            path_to_save="fig.svg",
        )
