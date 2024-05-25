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
    return False


def is_escape(row, column):
    """returns if the cell(row, column) is an Exit"""
    return False


def find_escape(row, column):
    """implementation for finding the Exit of the maze"""
    return False
