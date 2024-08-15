import numpy as np
import time
import matplotlib.pyplot as plt
from cmap import Colormap
from anytree import Walker, LevelOrderIter
import collections.abc


interactive = False

fig, ax, text, image = None, None, None, None
cmap_custom = None

cmap = Colormap([(0, "blue"), ("green"), ("yellow"), ("red")])


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

    global cmap_custom
    cmap_custom = Colormap(["yellow", "red"])


def init_interactive(top_text=" ", maze=[]):
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
    image = ax.imshow(parse_field_to_rgb(maze), cmap="autumn")
    plt.show()


def parse_field_to_rgb(field):
    """parses the input field to an RGB tupel map"""
    rgb_field = np.ndarray(shape=(len(field), len(field[0]), 3), dtype=int)
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_history_to_rgb(field, paths_taken=None, heat_map=None):
    """parses the input field with its history to an RGB tupel map"""
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER and field[i, j]:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]

    # convert to heat map
    if heat_map is None and paths_taken is not None:
        hottest = 0
        heat_map = np.zeros(shape=(len(field), len(field[0])), dtype=int)
        for path in paths_taken:
            for step in path:
                heat_map[step] += 1
                hottest = max(hottest, heat_map[step])
    else:
        hottest = np.max(heat_map)

    # color in rgb map
    for i in range(len(heat_map)):
        for j in range(len(heat_map[0])):
            if heat_map[i, j] != 0:
                color = cmap(heat_map[i, j] / hottest).rgba8
                rgb_arr = np.array([color[0], color[1], color[2]])
                rgb_field[i, j] = rgb_arr

    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] == MARKER:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_path_to_rgb(path, field, prev=None, gradient=False):
    """parses the input field with the path taken to an RGB tupel map"""
    if prev is None:
        rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
        # draw background and non defined colors
        for i in range(0, len(field)):
            for j in range(0, len(field[0])):
                if field[i, j] != MARKER and field[i, j]:
                    rgb_field[i, j] = COLOR_MAP[field[i, j]]
    else:
        rgb_field = prev

    # draw path
    for step in path:
        if gradient:
            rgb_field[step][2] = rgb_field[step][2] + 255
        else:
            rgb_field[step] = np.array([0, 0, 255])

    # redraw escape
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] == ESCAPE:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_age_to_rgb(start, nodes, field):
    """parses the input field with all current paths to an RGB tupel map"""
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
    # draw background and non defined colors
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]

    node_iter = LevelOrderIter(start)
    *_, last = LevelOrderIter(start)
    longest = last.age
    for node in node_iter:
        color = cmap(node.age / longest).rgba8
        rgb_arr = np.array([color[0], color[1], color[2]])
        rgb_field[node.step] = rgb_arr

    # redraw escape
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] == ESCAPE:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]
    return rgb_field


def parse_age_to_rgb_old(start, nodes, field):
    """parses the input field with all current paths to an RGB tupel map"""
    rgb_field = np.zeros(shape=(len(field), len(field[0]), 3), dtype=int)
    # draw background and non defined colors
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] != MARKER:
                rgb_field[i, j] = COLOR_MAP[field[i, j]]

    lengths = np.zeros(shape=(len(field), len(field[0])), dtype=int)
    longest = 0
    # draw age in paths
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes[0])):
            if nodes[i][j] is not None:
                # determine level
                w = Walker()
                up, common, down = w.walk(start, nodes[i][j])
                lengths[i, j] = len(down)
                if lengths[i, j] > longest:
                    longest = lengths[i, j]
    cmap = Colormap([(0, "yellow"), ("red")])
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes[0])):
            if lengths[i, j] != 0:
                color = cmap(lengths[i, j] / longest).rgba8
                arr = np.array([color[0], color[1], color[2]])
                rgb_field[i, j] = arr

    # redraw escape
    for i in range(0, len(field)):
        for j in range(0, len(field[0])):
            if field[i, j] == ESCAPE:
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


def print_path_to_display(maze, path, top_text="", time_to_sleep=0, background=None, gradient=False, path_to_save=None):
    """when interactive mode activated this can be used to update the current view"""
    if interactive is True:
        data = parse_path_to_rgb(path, maze, background, gradient)
        image.set_data(data)
        text.set_text(str(top_text))
        if path_to_save is not None:
            fig.savefig(path_to_save, format="svg", dpi=1200)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if time_to_sleep > 0:
            plt.pause(time_to_sleep)


def print_history_to_display(maze, paths=None, top_text="", time_to_sleep=0, path_to_save=None, heatMap=None):
    """when interactive mode activated this can be used to update the current view"""
    if interactive is True:
        data = parse_history_to_rgb(maze, paths_taken=paths, heat_map=heatMap)
        image.set_data(data)
        text.set_text(str(top_text))
        if path_to_save is not None:
            fig.savefig(path_to_save, format="svg", dpi=1200)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if time_to_sleep > 0:
            plt.pause(time_to_sleep)


def print_maze_to_display(maze, top_text="", time_to_sleep=0):
    """when interactive mode activated this can be used to update the current view"""
    if interactive is True:
        data = parse_field_to_rgb(maze)
        image.set_data(data)
        text.set_text(str(top_text))
        fig.canvas.draw()
        fig.canvas.flush_events()
        if time_to_sleep > 0:
            plt.pause(time_to_sleep)


def print_maze_with_age_to_display(maze, start, nodes, wall_size=1, top_text="", time_to_sleep=0, path_to_save=None):
    """when interactive mode activated this can be used to update the current view"""
    if interactive is True:
        time1 = time.time()
        data = parse_age_to_rgb_old(start, nodes, maze)
        time2 = time.time()
        print("time to render: " + str(time2 - time1))
        image.set_data(data)
        text.set_text(str(top_text))
        if path_to_save is not None:
            fig.savefig(path_to_save, format="svg", dpi=1200)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if time_to_sleep > 0:
            plt.pause(time_to_sleep)


def printArr(arr):
    """print the given array to the console"""
    for i in range(0, len(arr)):
        for j in range(0, len(arr[0])):
            print(arr[i, j], end="")
        print()
