import numpy as np
import matplotlib.pyplot as plt


interactive = False


def init(color_map, wall="x", escape="E", free=" ", marker="*"):
    """inits the module with default values, overwrites with given values"""
    global WALL
    WALL = wall
    global ESCAPE
    ESCAPE = escape
    global FREE
    FREE = free
    global MARKER
    MARKER = marker

    global COLOR_MAP
    COLOR_MAP = color_map


def init_interactive(top_text="", maze=[]):
    """inits the module for the interactive mode"""
    global interactive
    interactive = True
    plt.ion()
    global fig
    global ax
    global text
    global image
    fig, ax = plt.subplots()
    text = ax.text(0.01, 0.99, str(top_text), color="white")
    image = ax.imshow(parse_field_to_rgb(maze))
    plt.show()


def parse_field_to_rgb(field):
    """parses the input field to an RGB tupel map"""
    rgb_field = np.ndarray(shape=(len(field), len(field[0]), 3), dtype=int)
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_history_to_rgb(paths_taken, field):
    """parses the input field with its history to an RGB tupel map"""
    increment = 255 / len(paths_taken)
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=float)
    for path in paths_taken:
        for step in path:
            if field[step] == MARKER:
                rgb_field[step] = rgb_field[step] + increment

    rgb_field = rgb_field.astype(int)

    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_path_to_rgb(path, field):
    """parses the input field with the path taken to an RGB tupel map"""
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
    for step in path:
        if field[step] == MARKER or field[step] == FREE:
            rgb_field[step] = rgb_field[step] + 255

    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER and field[i, j] != FREE:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def convert_file_to_field(filename):
    """searches in the ./ directory for string:filename and parses it's content to an two dimensional numpy.array"""
    maze = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            maze.append([])
            for char in line:
                if char != "\n":
                    maze[len(maze) - 1].append(char)
    return np.asarray(maze)


def print_to_display(maze, path, text=""):
    """when interactive mode activated this can be used to update the current view"""
    if interactive is True:
        data = parse_path_to_rgb(path, maze)
        image.set_data(data)
        text.set_text(str(text))
        fig.canvas.draw()
        fig.canvas.flush_events()
        # time.sleep(0.5)


def printArr(arr):
    """print the given array to the console"""
    for i in range(0, len(arr)):
        for j in range(0, len(arr[0])):
            print(arr[i, j], end="")
        print()
