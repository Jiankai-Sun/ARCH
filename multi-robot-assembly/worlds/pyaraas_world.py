from __future__ import annotations
from world import World
import numpy as np
import pyaraas
from pyatk_world import WORKCELL_NAME
from pyaraas.tools import PathPlanner

class PyARAASWorld(World):

    def __init__(self, enable_hardware=False):
        self.enable_hardware = enable_hardware

        self.w = pyaraas.start(WORKCELL_NAME, enable_hardware=self.enable_hardware)  
        self.path_planner = PathPlanner(self.w, num_workers = 4, roadmap_path = "./roadmap.pkl")
        

    def setup(self):
        raise NotImplementedError
    
    def command_velocity(self, vel:np.array) -> np.array:
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    