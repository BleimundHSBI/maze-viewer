from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import shortuuid

import numpy as np
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

    def __init__(self, maze, start_pos, token_mapping):
        super().__init__(maze, start_pos, token_mapping)

        self.solved = False

        self.stack = []
        self.stack.append(RecursiveStep(pos=self.start_pos))

        self.prev_path = set()
        self.best_path = set()
        self.heat_map = np.zeros_like(self.maze, dtype=int)
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
        return np.zeros(shape=(1, 1))

    def getHistoricalView(self) -> np.ndarray:
        return np.zeros(shape=(1, 1))


if __name__ == "__main__":
    maze = np.array([[" ", " ", " ", "E"]])
    tokens = {
        "EMPTY": " ",
        "WALL": "#",
        "ESCAPE": "E"
    }
    back = Backtrace(maze, (0, 2), tokens)

    while True:
        back.nextStep()
