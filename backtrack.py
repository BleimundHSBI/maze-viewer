"""Programm to calculate the shortest path to an given maze"""

import time

import matplotlib.pyplot as plt
import maze_gen
import numpy as np
import maze_plotter


COLOR_MAP = {
    "#": np.array([255, 10, 0]),
    "E": np.array([255, 0, 180]),
    " ": np.array([0, 0, 0]),
    "*": np.array([15, 255, 0]),
}

WALL = "#"
ESCAPE = "E"
FREE = " "
MARKER = "*"


# settings:
visual = False
visual_steps = False
save_steps = True

num_solves = 0
all_paths = []

maze = []


def is_free(row, column, path):
    """return if the cell(row, column) is free"""
    if row < 0 or column < 0 or row >= len(maze) or column >= len(maze[row]):
        return False
    wall = maze[row, column] == WALL
    already_taken = (row, column) in path
    return not (wall or already_taken)


def is_escape(row, column):
    """returns if the cell(row, column) is an Exit"""
    return maze[row, column] == ESCAPE


# for future change already_taken to singel list of tupel to save RAM
def solve_maze(row, column, path):
    """solves the maze with a backtrace algorithm. Very Slow since every possible path is taken"""
    directions = [
        (row + 1, column),  # Down
        (row, column + 1),  # Right
        (row - 1, column),  # Up
        (row, column - 1),  # Left
    ]

    global all_paths
    global num_solves
    path.append((row, column))
    if visual is True and visual_steps is True:
        maze_plotter.print_path_to_display(maze, path)

    if is_escape(row, column):
        print("solved")
        if visual is True:
            if (num_solves % 1) == 0:
                maze_plotter.print_path_to_display(maze, path, num_solves)
            num_solves += 1
        if save_steps is True:
            all_paths.append(list(path))

        path.pop()
        return

    for d in directions:
        if is_free(d[0], d[1], path):
            solve_maze(d[0], d[1], path)

    path.pop()


# generate or load maze
maze, solved = maze_gen.getMaze(10, 10, 0)
# maze = maze_plotter.convert_file_to_field("field.txt")

maze_plotter.init(COLOR_MAP, wall=WALL, escape=ESCAPE, free=FREE, marker=MARKER)

if visual is True:
    maze_plotter.init_interactive(maze=maze)

maze_plotter.printArr(maze)

solve_maze(1, 1, [])

if len(all_paths) > 0:
    best = all_paths[0]
    for a in all_paths:
        if len(a) < len(best):
            best = a

if visual is False:
    plt.imshow(maze_plotter.parse_path_to_rgb(best, maze))
    plt.show()
elif visual is True:
    print("hellop")
    maze_plotter.print_path_to_display(
        maze,
        best,
        top_text="length: " + str(len(best)) + " solutions: " + str(len(all_paths)),
        time_to_sleep=20,
    )
