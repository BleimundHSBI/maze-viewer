"""Programm to calculate the shortest path to an given maze"""

import time
import numpy as np
from anytree import Node, RenderTree
from anytree.exporter import DotExporter
import matplotlib.pyplot as plt
import matplotlib as mpl
from cmap import Colormap

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
visual = False
visual_steps = False
save_steps = True

# "backtrack" or "flood"
algorithm = "backtrack"

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


def is_free(row, column, path):
    """return if the cell(row, column) is free"""
    if row < 0 or column < 0 or row >= len(maze) or column >= len(maze[row]):
        return False
    wall = maze[row, column] == WALL
    already_taken = (row, column) in path
    return not (wall or already_taken)


prev_path = set()
best_path = set()


def solve_maze_backtrack(row, column, heat_map, num_solves, longest):
    """solves the maze with a backtrace algorithm. Very Slow since every possible path is taken"""
    directions = [
        (row + 1, column),  # Down
        (row, column + 1),  # Right
        (row - 1, column),  # Up
        (row, column - 1),  # Left
    ]

    current_pos = (row, column)
    global prev_path
    prev_path.add(current_pos)
    heat_map[current_pos] += 1

    if visual is True and visual_steps is True and num_solves > 470:
        maze_plotter.print_path_to_display(maze, prev_path)

    if is_escape(row, column):
        if len(prev_path) < longest:
            longest = len(prev_path)
            global best_path
            best_path = prev_path.copy()

        num_solves += 1
        print("solved " + str(num_solves))
        if visual is True:
            if (num_solves % 1) == 0:
                maze_plotter.print_path_to_display(maze, prev_path, num_solves)
        if (num_solves % 470) == 0:
            print("here")

        prev_path.remove(current_pos)
        return num_solves, longest

    for d in directions:
        if is_free(d[0], d[1], prev_path):
            num_solves, longest = solve_maze_backtrack(d[0], d[1], heat_map, num_solves, longest)

    prev_path.remove(current_pos)
    return num_solves, longest


def generate(num):
    for i in range(num):
        global maze
        maze, solved = maze_gen.getMaze(10, 10, 5)
        maze_plotter.init(COLOR_MAP, wall=WALL, escape=ESCAPE, free=FREE, marker=MARKER)

        heatMap = np.zeros(shape=(len(maze), len(maze[0])), dtype=int)
        num_solves, longest = solve_maze_backtrack(1, 1, heatMap, 0, 99999999)

        solve_maze(1, 1)

        path = []
        solved_node = node_maze[coord_escape[0]][coord_escape[1]]
        print(RenderTree(solved_node))

        global best_path
        if num_solves > 0:
            path = list(best_path)

        cmap_ = Colormap([(0, "blue"), ("green"), ("yellow"), ("red")])
        maze_plotter.cmap = cmap_
        cmap = cmap_.to_matplotlib()
        norm = mpl.colors.Normalize(vmin=1, vmax=np.max(heatMap))
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        background = maze_plotter.parse_history_to_rgb(maze, heat_map=heatMap)
        plt.imshow(background)
        plt.savefig("figs/backtrack" + str(i) + "_clean.svg", format="svg", dpi=2400)
        plt.imshow(maze_plotter.parse_path_to_rgb(path, maze, prev=background, gradient=False))
        plt.savefig("figs/backtrack" + str(i) + ".svg", format="svg", dpi=2400)
        plt.clf()

        path = []
        # get solving path
        working_node = solved_node
        while working_node.parent is not None:
            path.append(working_node.step)
            working_node = working_node.parent

        cmap_ = Colormap([("yellow"), ("red")])
        maze_plotter.cmap = cmap_
        cmap = cmap_.to_matplotlib()
        norm = mpl.colors.Normalize(vmin=1, vmax=np.max(heatMap))
        plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
        background = maze_plotter.parse_age_to_rgb(node_maze[1][1], node_maze, maze)
        plt.imshow(background)
        plt.savefig("figs/flood" + str(i) + "_clean.svg", format="svg", dpi=2400)
        plt.imshow(maze_plotter.parse_path_to_rgb(path, maze, prev=background, gradient=True))
        plt.savefig("figs/flood" + str(i) + ".svg", format="svg", dpi=2400)
        plt.clf()


def default():
    # generate or load maze
    global maze
    maze, solved = maze_gen.getMaze(20, 20, 20)
    # print(solved)
    # needed for vscode wsl debugger
    # maze = maze_plotter.convert_file_to_field(
    #    "/home/philippbleimund/git/code-experimentation/aud_seminar/Seminar7/25x25.txt"
    # )

    # maze = maze_plotter.convert_file_to_field("25x25.txt")

    maze_plotter.init(COLOR_MAP, wall=WALL, escape=ESCAPE, free=FREE, marker=MARKER)

    if visual is True:
        maze_plotter.init_interactive(maze=maze)

    maze_plotter.printArr(maze)

    time1 = time.time()
    if algorithm == "backtrack":
        heatMap = np.zeros(shape=(len(maze), len(maze[0])), dtype=int)
        num_solves, longest = solve_maze_backtrack(1, 1, heatMap, 0, 99999999)
    elif algorithm == "flood":
        solve_maze(1, 1)
    time2 = time.time()

    path = []
    if algorithm == "flood":
        solved_node = node_maze[coord_escape[0]][coord_escape[1]]
        print(RenderTree(solved_node))
        # DotExporter(root_node).to_picture("tmp_graph.png") # before uncommenting check README.md

        # get solving path
        working_node = solved_node
        while working_node.parent is not None:
            path.append(working_node.step)
            working_node = working_node.parent

        print(path)
    elif algorithm == "backtrack":
        if num_solves > 0:
            path = list(best_path)

    print("time to solve: " + str(time2 - time1))

    if visual is False:
        if algorithm == "flood":
            plt.imshow(maze_plotter.parse_path_to_rgb(path, maze))
            plt.colorbar()
            plt.show()
        elif algorithm == "backtrack":
            plt.imshow(maze_plotter.parse_history_to_rgb(maze, heat_map=heatMap))
            cmap_ = Colormap([(0, "blue"), ("green"), ("yellow"), ("red")])
            cmap = cmap_.to_matplotlib()
            norm = mpl.colors.Normalize(vmin=1, vmax=np.max(heatMap))
            plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
            plt.savefig("backtrack.svg", format="svg", dpi=2400)
            plt.show()
    elif visual is True:
        if algorithm == "flood":
            maze_plotter.print_maze_with_age_to_display(
                maze,
                node_maze[1][1],
                node_maze,
                # top_text="length: " + str(len(path)),
                time_to_sleep=20,
                path_to_save="fig_clean.svg",
            )
            background = maze_plotter.parse_age_to_rgb(node_maze[1][1], node_maze, maze)
            maze_plotter.print_path_to_display(
                maze,
                path,
                background=background,
                # top_text="length: " + str(len(path)),
                time_to_sleep=20,
                path_to_save="fig.svg",
            )
        elif algorithm == "backtrack":
            maze_plotter.print_history_to_display(maze, heatMap=heatMap, time_to_sleep=20, path_to_save="fig.svg")


if __name__ == "__main__":
    generate(2)
