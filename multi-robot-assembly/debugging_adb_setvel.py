import pyatk
from pyatk import Transform, Vector
import numpy as np
import os
import time
if __name__ == '__main__':
    WORKCELL_NAME="MAR_PEG"

    pyatk.init(True, addr="10.140.68.92")
    your_path = "/home/aidanc/multi-robot-assembly"
    pyatk_dir = os.path.join(your_path, "apa_workcells/APA")
    pyatk.set_project_dir(pyatk_dir)
    w = pyatk.load_workcell(WORKCELL_NAME, gravity=pyatk.Vector(0, 0, 0))
    
    r0 = w.get_robot("UR10e-0")
    
    g0 = w.get_gripper("Rq85-0")
    g0.set_collision_model(False, False)

    gear = w.add_part("Gear_medium_centered", "Gear_medium_0", Transform([0,0,0,0,0,0]))
    
    g0_T_part = Transform(np.array([0, 0, -18, 0, -np.pi, 0]))
    
    initial_world_T_g0 = g0.get_tcp_transform()
    world_T_part = initial_world_T_g0.multiply(g0_T_part)
    
    gear.set_transform(world_T_part) # set peg close to gripper
    g0.attach(gear, True) # then attach
            
    r0_ee_T_world = r0.get_transform().invert()
    r0_ee_T_tcp = r0_ee_T_world.multiply(g0.get_transform())
    r0_pos_offset = r0_ee_T_tcp.get_values()[:3]
    r0_ee_T_tcp.identity()
    r0_ee_T_tcp.set_position(pyatk.Vector(r0_pos_offset[0], r0_pos_offset[1], r0_pos_offset[2]+100)) # +100 to include peg length
    gear.create_vel_controlled_tool(True, 100, r0_ee_T_tcp)
    
    for _ in range(10000):
        r0_vel_output = gear.set_vel(
            Vector(0, 0, 100), 
            Vector(0, 0, 0), 
            1/100.0)
        time.sleep(0.1)