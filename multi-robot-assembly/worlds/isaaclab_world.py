from __future__ import annotations
from world import World
import numpy as np

class IsaacLabWorld(World):

    def __init__(self):
        pass

    def setup(self):
        raise NotImplementedError
    
    def command_velocity(self, vel:np.array) -> np.array:
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    