from abc import ABC, abstractmethod
import typing
from dataclasses import dataclass, field
from queue import Queue

import shortuuid
from cmap import Colormap
from mazelib import Maze
from mazelib.generate.HuntAndKill import HuntAndKill
from mazelib.transmute.Perturbation import Perturbation
from tktooltip import ToolTip

import numpy as np
import matplotlib.pyplot as plt
import sys

import tkinter as tk
from tkinter import ttk

from matplotlib.backend_bases import key_press_handler
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure


class Algorithm(ABC):

    EMPTY = 0
    WALL = 1
    ESCAPE = 2

    def __init__(self, maze: np.ndarray, start_pos: tuple[int, int], token_mapping: dict):
        self.maze = self._normalize_maze(maze, token_mapping)
        self.start_pos = start_pos
        self.solved = False

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

    def getView(self) -> np.ndarray:
        if self.solved:
            return self.getHistoricalView()
        else:
            return self.getCurentState()


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
        if len(self.prev_path) > 0:
            rows, cols = zip(*self.prev_path)
            colored_maze[rows, cols] = self.COLORS["PATH"]

        return colored_maze

    def getHistoricalView(self) -> np.ndarray:

        colored_maze = np.empty(shape=(len(self.maze), len(self.maze[0]), 4), dtype=np.uint8)
        colored_maze[self.maze == self.EMPTY] = self.COLORS[self.EMPTY]

        # draw heatmap
        colored_maze = self.CMAP(self.heat_map / self.heat_map.max(), bytes=True)

        # base maze
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        # current best path
        if len(self.best_path) > 0:
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

        # draw heatmap
        colored_maze = self.CMAP(self.heat_map / self.heat_map.max(), bytes=True)
        colored_maze[self.heat_map == 0] = self.COLORS[self.EMPTY]

        if len(self.best_path) > 0:
            rows, cols = zip(*self.best_path)
            colored_maze[rows, cols] = self.COLORS["PATH"]

        # base maze
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        return colored_maze

    def getHistoricalView(self) -> np.ndarray:

        colored_maze = np.empty(shape=(len(self.maze), len(self.maze[0]), 4), dtype=np.uint8)

        # draw heatmap
        colored_maze = self.CMAP(self.heat_map / self.heat_map.max(), bytes=True)
        colored_maze[self.heat_map == 0] = self.COLORS[self.EMPTY]

        # base maze
        colored_maze[self.maze == self.WALL] = self.COLORS[self.WALL]
        colored_maze[self.maze == self.ESCAPE] = self.COLORS[self.ESCAPE]

        # current best path
        if self.best_path:
            rows, cols = zip(*self.best_path)
            colored_maze[rows, cols] = self.COLORS["PATH"]

        return colored_maze


class MazeVisualizer(tk.Tk):

    SPEED_CYCLE = [0, 1, 2, 4, 8]

    SPEEDS = {
        1: "1️⃣",
        2: "2️⃣",
        4: "4️⃣",
        8: "8️⃣",
        0: "*️⃣"
    }

    MAZELIB_TOKENS = {
        "EMPTY": 0,
        "WALL": 1,
        "ESCAPE": 2
    }

    def __init__(self, starter_maze: typing.Optional[np.ndarray] = None, starter_tokens: typing.Optional[dict] = None):
        if starter_maze is not None and starter_tokens is None:
            raise ValueError("starter_tokens is required when starter_maze is provided")

        super().__init__()
        self._setup_tk()

        self.speed = 1
        self.manual = True

        self.solver = None
        self.solver_2 = None
        if starter_maze is not None:
            self.maze = starter_maze
        else:
            self.maze = self._generate_maze(int(self.maze_x_size.get()),
                                            int(self.maze_y_size.get()),
                                            int(self.maze_traps.get()))
        if starter_tokens:
            self.tokens = starter_tokens
        else:
            self.tokens = self.MAZELIB_TOKENS

    def _setup_tk(self):
        self.title('Maze Visualizer')
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = figure_canvas = FigureCanvasTkAgg(self.figure, self)

        NavigationToolbar2Tk(figure_canvas, self)

        self.axes = self.figure.add_subplot()
        self.axes_2 = None

        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        button_frame = tk.Frame(self)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        maze_option_frame = tk.Frame(self)
        maze_option_frame.pack(side=tk.TOP, fill=tk.X)

        # Add buttons
        self.start_button = tk.Button(button_frame, text="▶️", command=self._start_maze)
        self.speed_button = tk.Button(button_frame, text="1️⃣", command=self._change_speed)
        self.pause_button = tk.Button(button_frame, text="⏸", command=self._pause_maze)
        self.reset_button = tk.Button(button_frame, text="⏹", command=self._reset_maze)
        self.step_button = tk.Button(button_frame, text="⏩", command=self._step_maze)
        self.new_button = tk.Button(button_frame, text="🔄", command=self._new_maze)

        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.speed_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.pause_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.reset_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.step_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.new_button.pack(side=tk.LEFT, padx=10, pady=5)

        ToolTip(self.start_button, msg="start the visualization", follow=False, delay=1)
        ToolTip(self.speed_button, msg="change the speed of the visualization", follow=False, delay=1)
        ToolTip(self.pause_button, msg="pause the visualization", follow=False, delay=1)
        ToolTip(self.reset_button, msg="reset the progress of the algorithm", follow=False, delay=1)
        ToolTip(self.step_button, msg="take a single step in the algorithm", follow=False, delay=1)
        ToolTip(self.new_button, msg="reset and create a new maze", follow=False, delay=1)

        self.maze_text = tk.Label(maze_option_frame, text="Maze options")
        self.maze_x_size_text = tk.Label(maze_option_frame, text="x:")
        self.maze_x_size = tk.Spinbox(maze_option_frame, from_=3, to=20, width=4)
        self.maze_y_size_text = tk.Label(maze_option_frame, text="y:")
        self.maze_y_size = tk.Spinbox(maze_option_frame, from_=3, to=20, width=4)
        self.maze_traps_text = tk.Label(maze_option_frame, text="traps:")
        self.maze_traps = tk.Spinbox(maze_option_frame, from_=0, to=200, width=4)
        current_var = tk.StringVar()
        self.combobox = ttk.Combobox(maze_option_frame, textvariable=current_var)
        self.combobox["values"] = ("Backtrack", "Breadth first", "Both")
        self.combobox.bind("<<ComboboxSelected>>", self._change_maze_solver)

        self.maze_text.pack(side=tk.LEFT, padx=5, pady=5)
        self.maze_x_size_text.pack(side=tk.LEFT, padx=5, pady=5)
        self.maze_x_size.pack(side=tk.LEFT, padx=5, pady=5)
        self.maze_y_size_text.pack(side=tk.LEFT, padx=5, pady=5)
        self.maze_y_size.pack(side=tk.LEFT, padx=5, pady=5)
        self.maze_traps_text.pack(side=tk.LEFT, padx=5, pady=5)
        self.maze_traps.pack(side=tk.LEFT, padx=5, pady=5)
        self.combobox.pack(side=tk.LEFT, padx=5, pady=5)

    def _generate_maze(self, x, y, remove):
        generator = HuntAndKill(x, y)
        maze = generator.generate()
        transmuter = Perturbation(x, 10)
        transmuter.transmute(maze, None, None)

        maze[len(maze) - 2, len(maze[0]) - 2] = 2

        # remove walls
        inner_maze = maze[1:-1, 1:-1]
        walls = np.where(inner_maze == self.MAZELIB_TOKENS["WALL"])
        walls = np.array(list(zip(walls[0], walls[1])))
        walls = walls[:, :] + 1
        chosen_index = np.random.randint(0, len(walls), size=remove)
        removed_walls = walls[chosen_index]
        maze[removed_walls[:, 0], removed_walls[:, 1]] = self.MAZELIB_TOKENS["EMPTY"]

        return maze

    def _timer(self):
        if self.manual == False:
            self._step_maze()
            if self.speed == 0:
                self.after(1, self._timer)
                return
            time = int((1 / self.speed) * 1000)
            self.after(time, self._timer)

    def _change_maze_solver(self, event):
        if self.combobox.get() == "Backtrack":
            self.solver = Backtrace(self.maze, (1, 1), self.tokens)
            self.solver_2 = None
            self.figure.clf()
            self.axes = self.figure.add_subplot()
            self.axes.imshow(self.solver.getView())
            self.axes_2 = None

        elif self.combobox.get() == "Breadth first":
            self.solver = Breadth(self.maze, (1, 1), self.tokens)
            self.solver_2 = None
            self.figure.clf()
            self.axes = self.figure.add_subplot()
            self.axes.imshow(self.solver.getView())
            self.axes_2 = None
        elif self.combobox.get() == "Both":
            self.solver = Backtrace(self.maze, (1, 1), self.tokens)
            self.solver_2 = Breadth(self.maze, (1, 1), self.tokens)
            self.figure.clf()
            self.axes = self.figure.add_subplot(121)
            self.axes.imshow(self.solver.getView())
            self.axes_2 = self.figure.add_subplot(122)
            self.axes_2.imshow(self.solver_2.getView())

        self.canvas.draw()

        pass

    def _change_speed(self):
        next_speed_index = (self.SPEED_CYCLE.index(self.speed) + 1) % len(self.SPEED_CYCLE)
        self.speed = self.SPEED_CYCLE[next_speed_index]

        self.speed_button.configure(text=self.SPEEDS[self.speed])
        pass

    def _start_maze(self):
        self.manual = False
        self._timer()
        pass

    def _pause_maze(self):
        self.manual = True
        pass

    def _step_maze(self):
        if self.solver:
            self.solver.nextStep()
            self.axes.cla()
            self.axes.imshow(self.solver.getView())
            self.canvas.draw()
        if self.solver_2:
            self.solver_2.nextStep()
            self.axes_2.cla()
            self.axes_2.imshow(self.solver_2.getView())
            self.canvas.draw()

        pass

    def _reset_maze(self):
        self._change_maze_solver(None)
        self._step_maze()
        pass

    def _new_maze(self):
        self.maze = self._generate_maze(int(self.maze_x_size.get()), int(
            self.maze_y_size.get()), int(self.maze_traps.get()))
        self.tokens = self.MAZELIB_TOKENS
        self._change_maze_solver(None)
        self._step_maze()
        pass


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
        "WALL": "#",
        "EMPTY": " ",
        "ESCAPE": "E"
    }
    MazeVisualizer(starter_maze=maze, starter_tokens=tokens).mainloop()
