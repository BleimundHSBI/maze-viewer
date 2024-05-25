"""Programm to calculate the shortest path to an given maze"""

import numpy as np

WALL = "X"
ESCAPE = "E"

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


def is_free(row, column):
    """return if the cell(row, column) is free"""
    return maze[row, column] != WALL


def is_escape(row, column):
    """returns if the cell(row, column) is an Exit"""
    return maze[row, column] != ESCAPE


def get_free_sides(row, column):
    """return the number of free sides on cell(row, column)"""
    num_sides = 0
    sides = []
    sides.append(maze[row, column])
    sides.append(maze[row, column])
    sides.append(maze[row, column])
    sides.append(maze[row, column])
    for s in sides:
        if s == " ":
            num_sides = num_sides + 1
    return num_sides


def is_junction(row, column):
    """returns if the cell(row, column) is an junction"""
    return get_free_sides(row, column) > 2


def is_dead_end(row, column):
    """returns true if the cell(row, column) is an dead end"""
    return get_free_sides(row, column) == 1


def find_escape(row, column):
    """implementation for finding the Exit of the maze"""
    solved = False
    while solved:
        for x in range(0, len(maze)):
            for y in range(0, len(maze)):
                
    return False
