"""Programm to calculate the shortest path to an given maze"""

import numpy as np

WALL = "X"
ESCAPE = "E"
FREE = " "
MARKER = "*"

maze = []


def convert_file_to_field(filename):
    """searches in the ./ directory for string:filename and parses it's content to an two dimensional numpy.array"""
    global maze
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            maze.append([])
            for char in line:
                if char != "\n":
                    maze[len(maze) - 1].append(char)
    maze = np.asarray(maze)


def printArr(arr):
    for i in range(0, len(arr)):
        for j in range(0, len(arr[0])):
            print(arr[i, j], end="")
        print()


def is_free(row, column):
    """return if the cell(row, column) is free"""
    return maze[row, column] == FREE


def is_escape(row, column):
    """returns if the cell(row, column) is an Exit"""
    return maze[row, column] == ESCAPE


def get_sides(row, column):
    sides = []
    sides.append(maze[row, column])
    sides.append(maze[row, column])
    sides.append(maze[row, column])
    sides.append(maze[row, column])
    return sides


def get_free_sides(row, column):
    """return the number of free sides on cell(row, column)"""
    num_sides = 0
    all_sides = get_sides(row, column)
    for s in all_sides:
        if s == FREE:
            num_sides = num_sides + 1
    return num_sides


def is_junction(row, column):
    """returns if the cell(row, column) is an junction"""
    return get_free_sides(row, column) > 2


def is_dead_end(row, column):
    """returns true if the cell(row, column) is an dead end"""
    return get_free_sides(row, column) == 1


paths = []
all_paths = []


def solve_maze(row, column):
    directions = []
    directions.append((row + 1, column))
    directions.append((row, column + 1))
    directions.append((row - 1, column))
    directions.append((row, column - 1))

    maze[row, column] = MARKER
    paths.append((row, column))

    if is_escape(row, column):
        all_paths.append(list(paths))
        paths.pop()
        return True

    if is_dead_end(row, column):
        return False

    for d in directions:
        if is_free(d[0], d[1]):
            solve_maze(d[0], d[1])

    paths.pop()


convert_file_to_field("field.txt")
printArr(maze)
solve_maze(1, 1)
printArr(maze)
