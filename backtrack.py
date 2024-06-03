"""Programm to calculate the shortest path to an given maze"""

import time

import matplotlib.pyplot as plt
import maze_gen
import numpy as np

WALL = "x"
ESCAPE = "E"
FREE = " "
MARKER = "*"

COLOR_MAP = {
    "x": np.array([255, 10, 0]),
    "E": np.array([255, 0, 180]),
    " ": np.array([0, 0, 0]),
    "*": np.array([15, 255, 0]),
}

maze = []


def parse_field_to_rgb(field):
    """parses the input field to an RGB tupel map"""
    rgb_field = np.ndarray(shape=(len(field), len(field[0]), 3), dtype=int)
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_history_to_rgb(paths_taken, field):
    """parses the input field to an RGB tupel map"""
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
    for i in range(0, len(paths_taken)):
        for j in range(0, len(paths_taken[i])):
            if field[paths_taken[i][j]] == MARKER:
                rgb_field[paths_taken[i][j]] = rgb_field[paths_taken[i][j]] + 50

    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_paths_to_rgb(paths_taken, field):
    """parses the input field to an RGB tupel map"""
    global maze
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
    for i in range(0, len(paths_taken)):
        if field[paths_taken[i]] == MARKER:
            rgb_field[paths_taken[i]] = rgb_field[paths_taken[i]] + 50

    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


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


def is_free(row, column, already_taken):
    """return if the cell(row, column) is free"""
    if row < 0 or column < 0 or row >= len(maze) or column >= len(maze[row]):
        return False
    wall = maze[row, column] == WALL
    marker = already_taken[row, column] == MARKER
    return not (wall or marker)


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


# maze = maze_gen.getMaze(10, 10, 50)
convert_file_to_field("field3.txt")
print(maze)
printArr(maze)
paths = []
all_paths = []

plt.ion()
fig, ax = plt.subplots()
image = ax.imshow(parse_field_to_rgb(maze))
plt.show()


def print_to_display():
    data = parse_paths_to_rgb(paths, maze)
    image.set_data(data)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # time.sleep(0.5)

# for future change already_taken to singel list of tupel to save RAM
def solve_maze(row, column, already_taken):
    directions = []
    directions.append((row + 1, column))
    directions.append((row, column + 1))
    directions.append((row - 1, column))
    directions.append((row, column - 1))

    global paths
    global all_paths
    paths.append((row, column))
    print_to_display()

    if is_escape(row, column):
        print("solved")
        all_paths.append(list(paths))
        paths.pop()
        return True

    maze[row, column] = MARKER

    for d in directions:
        if is_free(d[0], d[1], already_taken):
            new_taken = np.copy(already_taken)
            solve_maze(d[0], d[1], new_taken)

    paths.pop()


# convert_file_to_field(
# "/home/philippbleimund/git/code-experimentation/aud_seminar/Seminar7/field2.txt"
# )
printArr(maze)
solve_maze(1, 1)

printArr(maze)
