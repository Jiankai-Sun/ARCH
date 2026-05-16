from __future__ import annotations
from world import World
import numpy as np
import pyatk
from pathlib import Path

WORKCELL_NAME="MAR_PEG"

class PyATKWorld(World):

    def __init__(self, friction=0.02, is_remote = False, render=True):
        self.friction = friction
        ip = "10.140.68.92" if is_remote else "127.0.0.1"
        # Init the pyatk
        if(render):
            pyatk.init(True, addr=ip)
        else:
            pyatk.init(False, addr=ip)
    
        # Set the dir
        current_dir = Path().resolve()
        pyatk_dir = str(current_dir.joinpath("apa_workcells/APA"))
        pyatk.set_project_dir(pyatk_dir)
        # Load workcell (UR10e + gripper)
        gravity = pyatk.Vector(0, 0, 0)
        self.w = pyatk.load_workcell(WORKCELL_NAME, gravity=gravity)
    

    def setup(self):
        raise NotImplementedError
    
    def command_velocity(self, vel:np.array) -> np.array:
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    