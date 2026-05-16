from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class World(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        raise NotImplementedError
    
    @abstractmethod
    def command_velocity(self, vel:np.array) -> np.array:
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError
    