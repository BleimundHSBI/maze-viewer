from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.HuntAndKill import HuntAndKill
from mazelib.transmute.Perturbation import Perturbation
from mazelib.solve.ShortestPath import ShortestPath
import numpy as np
import time
import random


def getMaze(x, y, walls_removed):
    m = Maze()
    m.generator = HuntAndKill(x, y)
    m.generate()
    m.transmuters = Perturbation(6, 10)

    maze_grid = m.grid

    while walls_removed > 0:
        r_x = random.randint(1, len(maze_grid) - 2)
        r_y = random.randint(1, len(maze_grid[0]) - 2)
        if maze_grid[r_x, r_y] == 1:
            maze_grid[r_x, r_y] = 0
            walls_removed = walls_removed - 1

    m.grid = maze_grid

    time1 = time.time()
    m.solver = ShortestPath()
    m.start = (0, 1)
    m.end = (len(maze_grid) - 2, len(maze_grid[0]) - 1)
    m.solve()
    time2 = time.time()
    print("time to solve with lib: " + str(time2 - time1))

    maze_str = m.tostring()
    maze_lines = maze_str.split("\n")
    maze_array = [list(line) for line in maze_lines]
    maze_array = np.asarray(maze_array)

    maze_array[len(maze_array) - 2, len(maze_array[0]) - 2] = "E"

    return maze_array, m.tostring(True, True)
