from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from queue import Queue

import shortuuid
from cmap import Colormap

import numpy as np
import matplotlib.pyplot as plt
import sys


class Algorithm(ABC):

    EMPTY = 0
    WALL = 1
    ESCAPE = 2

    def __init__(self, maze: np.ndarray, start_pos: tuple[int, int], token_mapping: dict):
        self.maze = self._normalize_maze(maze, token_mapping)
        self.start_pos = start_pos

    @abstractmethod
    def _normalize_maze(self, maze, token_mapping) -> np.ndarray:
        pass

    @abstractmethod
    def nextStep(self):
        pass

    @abstractmethod
    def getCurentState(self) -> np.ndarray:
        pass

    @abstractmethod
    def getHistoricalView(self) -> np.ndarray:
        pass


@dataclass
class RecursiveStep():
    pos: tuple[int, int]
    depends: list = field(default_factory=list)
    id: str = field(default_factory=lambda: str(shortuuid.uuid()))
    waiting_for_execution: bool = True


class Backtrace(Algorithm):

    COLORS = {
        Algorithm.EMPTY: np.array([255, 255, 255, 255], dtype=np.uint8),
        Algorithm.WALL: np.array([0, 0, 0, 255], dtype=np.uint8),
        Algorithm.ESCAPE: np.array([0, 255, 0, 255], dtype=np.uint8),
        "PATH": np.array([255, 0, 180, 255], dtype=np.uint8)
    }

    CMAP = Colormap([(0, "blue"), ("green"), ("yellow"), ("red")])

    def __init__(self, maze, start_pos, token_mapping):
        super().__init__(maze, start_pos, token_mapping)

        self.solved = False

        self.stack = []
        self.stack.append(RecursiveStep(pos=self.start_pos))

        self.prev_path = set()
        self.best_path = set()
        self.heat_map = np.zeros_like(self.maze, dtype=float)
        self.prev_max_steps = sys.maxsize

    def _normalize_maze(self, maze, token_mapping):
        normalized_maze = np.zeros_like(maze, dtype=int)
        for key, value in token_mapping.items():
            if key == "EMPTY":
                normalized_maze[maze == value] = self.EMPTY
            elif key == "WALL":
                normalized_maze[maze == value] = self.WALL
            elif key == "ESCAPE":
                normalized_maze[maze == value] = self.ESCAPE

        return normalized_maze

    def nextStep(self):
        if self.solved == True:
            return

        self.backtrack(self.stack[-1])

        removed = set()
        for step in reversed(self.stack):
            if step.waiting_for_execution == True:
                continue
            for id in step.depends:
                if id in removed:
                    step.depends.remove(id)
                    removed.add(id)
            if not step.depends:
                self.stack.remove(step)
                self.prev_path.remove(step.pos)
                removed.add(step.id)

        self.solved = not self.stack

    def backtrack(self, curr_step: RecursiveStep):
        current_pos = curr_step.pos
        curr_step.waiting_for_execution = False
        directions = [
            (1 + current_pos[0], 0 + current_pos[1]),  # Down
            (0 + current_pos[0], 1 + current_pos[1]),  # Right
            (-1 + current_pos[0], 0 + current_pos[1]),  # Up
            (0 + current_pos[0], -1 + current_pos[1]),  # Left
        ]

        self.prev_path.add(current_pos)
        self.heat_map[current_pos] += 1

        if self.maze[current_pos] == self.ESCAPE:
            if len(self.prev_path) < self.prev_max_steps:
                self.prev_max_steps = len(self.prev_path)
                self.best_path = self.prev_path.copy()

            return

        dependencies = []

        for d in directions:
            if d[0] >= 0 and d[0] < len(self.maze) and d[1] >= 0 and d[1] < len(self.maze[0]) and \
                    (self.maze[d] == self.EMPTY or self.maze[d] == self.ESCAPE) and d not in self.prev_path:
                next_step = RecursiveStep(pos=d)
                self.stack.append(next_step)
                dependencies.append(next_step.id)

        curr_step.depends = curr_step.depends + dependencies

    def getCurentState(self) -> np.ndarray:
        colored_maze = np.empty(shape=(len(self.maze), len(self.maze[0]), 4), dtype=np.uint8)

        # base maze
        colored_maze[self.maze == self.EMPTY] = self.COLORS[self.EMPTY]
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        # current path
        rows, cols = zip(*self.prev_path)
        colored_maze[rows, cols] = self.COLORS["PATH"]

        return colored_maze

    def getHistoricalView(self) -> np.ndarray:

        colored_maze = np.empty(shape=(len(self.maze), len(self.maze[0]), 4), dtype=np.uint8)
        colored_maze[self.maze == self.EMPTY] = self.COLORS[self.EMPTY]

        # draw heatmap
        colored_maze = self.CMAP(self.heat_map / self.heat_map.max())

        # base maze
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        # current best path
        if self.best_path:
            rows, cols = zip(*self.best_path)
            colored_maze[rows, cols] = self.COLORS["PATH"]

        return colored_maze


class Breadth(Algorithm):

    COLORS = {
        Algorithm.EMPTY: np.array([255, 255, 255, 255], dtype=np.uint8),
        Algorithm.WALL: np.array([0, 0, 0, 255], dtype=np.uint8),
        Algorithm.ESCAPE: np.array([0, 255, 0, 255], dtype=np.uint8),
        "PATH": np.array([255, 0, 180, 255], dtype=np.uint8)
    }

    CMAP = Colormap([(0, "yellow"), ("red")])

    def __init__(self, maze, start_pos, token_mapping):
        super().__init__(maze, start_pos, token_mapping)

        self.found = False
        self.solved = False

        self.queue = Queue()
        self.queue.put(start_pos)

        self.visited = set()
        self.best_path = []
        self.heat_map = np.zeros_like(self.maze, dtype=float)
        self.current_heat = 0

    def _normalize_maze(self, maze, token_mapping):
        normalized_maze = np.zeros_like(maze, dtype=int)
        for key, value in token_mapping.items():
            if key == "EMPTY":
                normalized_maze[maze == value] = self.EMPTY
            elif key == "WALL":
                normalized_maze[maze == value] = self.WALL
            elif key == "ESCAPE":
                normalized_maze[maze == value] = self.ESCAPE

        return normalized_maze

    def nextStep(self):
        if self.solved == True:
            return

        if self.found == False:
            self.breadth()
        elif self.found == True:
            self.getPath()

    def breadth(self):

        current_pos = self.queue.get()
        self.visited.add(current_pos)
        self.current_heat += 1
        self.heat_map[current_pos] = self.current_heat

        if self.maze[current_pos] == self.ESCAPE:
            self.best_path.append(current_pos)
            self.found = True
            return

        directions = [
            (1 + current_pos[0], 0 + current_pos[1]),  # Down
            (0 + current_pos[0], 1 + current_pos[1]),  # Right
            (-1 + current_pos[0], 0 + current_pos[1]),  # Up
            (0 + current_pos[0], -1 + current_pos[1]),  # Left
        ]

        for d in directions:
            if d[0] >= 0 and d[0] < len(self.maze) and d[1] >= 0 and d[1] < len(self.maze[0]) and \
                    (self.maze[d] == self.EMPTY or self.maze[d] == self.ESCAPE) and d not in self.visited:
                self.queue.put(d)

    def getPath(self):

        current_pos = self.best_path[-1]

        if current_pos == self.start_pos:
            self.solved = True
            return

        directions = [
            (1 + current_pos[0], 0 + current_pos[1]),  # Down
            (0 + current_pos[0], 1 + current_pos[1]),  # Right
            (-1 + current_pos[0], 0 + current_pos[1]),  # Up
            (0 + current_pos[0], -1 + current_pos[1]),  # Left
        ]

        valid_fields = []
        for d in directions:
            if d[0] >= 0 and d[0] < len(self.maze) and d[1] >= 0 and d[1] < len(self.maze[0]) and self.heat_map[d] > 0:
                valid_fields.append(d)

        valid_fields = np.array(valid_fields)
        next_field = np.argmin(self.heat_map[valid_fields[:, 0], valid_fields[:, 1]])
        self.best_path.append(tuple(valid_fields[next_field]))

    def getCurentState(self) -> np.ndarray:
        colored_maze = np.empty(shape=(len(self.maze), len(self.maze[0]), 4), dtype=np.uint8)

        # base maze
        colored_maze[self.maze == self.EMPTY] = self.COLORS[self.EMPTY]
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        return colored_maze

    def getHistoricalView(self) -> np.ndarray:

        colored_maze = np.empty(shape=(len(self.maze), len(self.maze[0]), 4), dtype=np.uint8)

        # draw heatmap
        colored_maze = self.CMAP(self.heat_map / self.heat_map.max())
        colored_maze[self.heat_map == 0] = self.COLORS[self.EMPTY]

        # base maze
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        # current best path
        if self.best_path:
            rows, cols = zip(*self.best_path)
            colored_maze[rows, cols] = self.COLORS["PATH"]

        return colored_maze


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


if __name__ == "__main__":
    maze = convert_file_to_field("field.txt")
    tokens = {
        "EMPTY": " ",
        "WALL": "#",
        "ESCAPE": "E"
    }
    breath = Breadth(maze, (1, 1), tokens)

    while not breath.solved:
        breath.nextStep()

    image = breath.getHistoricalView()
    fig, ax = plt.subplots()
    im = ax.imshow(image)
    plt.show()
