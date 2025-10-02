import json
import multiprocessing
import os
import queue

import numpy as np
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint
import trio

import pyatk
from pyatk import ARAASError, PlanningError


# the planning function is defined outside the PathPlanner class to avoid
# pickling issues and using mutliprocessing.
# https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function


def pyatk_planner(
        request,
        response,
        project_dir,
        workcell_name,
        roadmap_path="",
        roadmap_expansion=False,
):
    # initialize a pyatk process with the current workcell
    pyatk.init(False)
    pyatk.set_project_dir(project_dir)
    atk_workcell = pyatk.load_workcell(workcell_name)

    while True:
        plan_config = request.get()
        if plan_config == None:
            break

        # update the state of the workcell
        (
            current_state,
            unique_name,
            goal,
            min_joint_values,
            max_joint_values,
            timeout,
            planner_type,
        ) = plan_config

        atk_workcell.apply_state(json.dumps(current_state))
        atk_robot = atk_workcell.get_robot(unique_name)

        if min_joint_values and max_joint_values:
            atk_robot.set_planning_joint_limits(min_joint_values, max_joint_values)

        # NB: in hardware mode, we synchronize the digital twin robot tool space to
        # the real robot and use IK to update the joint angles. with the kinematics being
        # slightly off, the joint angles will be slightly different. that said, the pyaraas
        # workcell.capture_state will capture the true joint angles of the robot hardware
        # so the initial config for the planner (coming from the pyatk workcell that applies
        # that state) will have the true initial config of the robots.

        init = atk_robot.get_joint_angles()

        path_segment = atk_robot.find_path(
            init, goal, timeout, planner_type, True, roadmap_path
        )

        if len(path_segment) > 0:
            if roadmap_expansion:
                # overwrite existing map
                atk_robot.store_roadmap(roadmap_path)

            quality_checks = ["All"]
            path_quality = atk_robot.perform_path_quality_checks(
                path_segment, quality_checks
            )
            response.put((path_quality, path_segment))
        else:
            response.put(({}, path_segment))


def pyatk_articulation_planner(
        request,
        response,
        project_dir,
        workcell_name,
        roadmap_path="",
        roadmap_expansion=False,
):
    # initialize a pyatk process with the current workcell
    pyatk.init(False)
    pyatk.set_project_dir(project_dir)
    atk_workcell = pyatk.load_workcell(workcell_name)

    while True:
        plan_config = request.get()
        if plan_config == None:
            break

        # Need to unpack and set up articulation anew.
        # QU: are parts added to pyatk during state unpack, or are they transient? Check, but I think transient.
        # pickling error - cannot pass an araas part. Unpack and repack transform.
        (
            current_state,
            planner_parts_list,
            articulation_descriptor,
            grasp_name,
            goal,
            min_joint_values,
            max_joint_values,
            timeout,
            planner_type,
        ) = plan_config

        atk_workcell.apply_state(json.dumps(current_state))

        atk_link_list = []

        for item in planner_parts_list:
            part_type, part_name, part_pose = item
            part_transform = pyatk.Transform(part_pose)
            atk_link_list.append(atk_workcell.add_part(part_type, part_name, part_transform))

        atk_articulation = atk_workcell.add_articulation("new_articulation", atk_link_list,
                                                         json.dumps(articulation_descriptor), grasp_name)

        if min_joint_values and max_joint_values:
            atk_articulation.set_planning_joint_limits(min_joint_values, max_joint_values)

        init = atk_articulation.get_joint_angles()

        path_segment = atk_articulation.find_path(
            init, goal, timeout, planner_type, True, roadmap_path
        )

        if len(path_segment) > 0:
            if roadmap_expansion:
                # overwrite existing map
                atk_articulation.store_roadmap(roadmap_path)

            quality_checks = ["All"]
            path_quality = atk_articulation.perform_path_quality_checks(
                path_segment, quality_checks
            )
            response.put((path_quality, path_segment))
        else:
            response.put(({}, path_segment))

            atk_articulation.clear()  # free up memory, handle pointers


def calculate_total_distance(joint_trajectories):
    """
    Calculate the total distance each joint has to travel over the course of the trajectory.

    Parameters:
        joint_trajectories (list of list of floats): List of joint angles representing the trajectory.

    Returns:
        list of floats: Total distance traveled by each joint.
    """
    joint_trajectories = np.array(joint_trajectories)
    num_joints = joint_trajectories.shape[1]

    total_distances = np.zeros(num_joints)

    for i in range(1, joint_trajectories.shape[0]):
        total_distances += np.abs(joint_trajectories[i] - joint_trajectories[i - 1])

    return total_distances.tolist()


class PathPlanner:
    def __init__(
            self,
            workcell,
            num_workers=8,
            roadmap_path="",
            expand_roadmap=False,
            articulation_planner=False,
    ):
        """Spawn a set of subprocesses to handle the path planning workload.

        Each subprocess uss pyatk's path planning API and use queues to communicate
        between this PathPlanner object and its subprocesses.

        For each possible joint goal configuration of the robot for a specified end
        effector goal pose, we want to simulatenously plan paths in joint space, and
        then choose the "best" one. for now, the default number of solvers is 8 (most ik
        solvers for 6 dof robots return 8 solutions).

        Optional inputs: using and/or extending a roadmap for use in roadmap-based planners
        To avoid thread conflicts, roadmap expansion is only available during strictly sequential (n=1)
        workcell exploration.
        """

        if expand_roadmap:
            if len(roadmap_path) < 1:
                parent_path = os.path.join(os.getcwd(), ".araas_temp")

                os.makedirs(parent_path, exist_ok=True)
                roadmap_path = os.path.join(parent_path, "temp_ompl_pd.graph")

            print(
                "Setting simultaneous solvers to 1 for roadmap exploration and expansion"
            )
            num_workers = 1

        num_available_cpu = min(os.cpu_count(), num_workers)

        self.workcell = workcell
        self.roadmap_path = roadmap_path
        self.roadmap_expansion = expand_roadmap

        # start the pyatk path planner workers
        project_dir = workcell.project_folder
        workcell_name = workcell.name

        # the multi process planning function we call is defined outside this
        # class to avoid pickling issues and using mutliprocessing.
        self.request = multiprocessing.Queue()
        self.response = multiprocessing.Queue()
        self.articulation_planner = articulation_planner

        if self.articulation_planner:
            print("Targeting articulation mp function")
            self.processes = [
                multiprocessing.Process(
                    target=pyatk_articulation_planner,
                    args=(
                        self.request,
                        self.response,
                        project_dir,
                        workcell_name,
                        roadmap_path,
                        expand_roadmap,
                    ),
                )
                for _ in range(num_available_cpu)
            ]

        else:
            self.processes = [
                multiprocessing.Process(
                    target=pyatk_planner,
                    args=(
                        self.request,
                        self.response,
                        project_dir,
                        workcell_name,
                        roadmap_path,
                        expand_roadmap,
                    ),
                )
                for _ in range(num_available_cpu)
            ]

        for p in self.processes:
            p.start()

        # maintain a pyatk workcell for generating the parameters for the planning jobs
        pyatk.init(False)
        pyatk.set_project_dir(project_dir)
        self.atk_workcell = pyatk.load_workcell(workcell_name)
        self.ready = True

    def __del__(self):
        if self.ready:
            self.shutdown()

    def shutdown(self):
        for p in self.processes:
            # None message signals the worker to shutdown
            self.request.put(None)
        for p in self.processes:
            p.join()
            p.close()

        self.request.close()
        self.response.close()
        self.ready = False

    async def plan_linkage_path(
            self,
            articulation_parts_list,
            articulation_descriptor,
            grasp_name,
            world_T_ee_goal,
            number_of_keypoints,
            min_joint_values=None,
            max_joint_values=None,
            timeout=1.0,
            planner_type="LazyPRMstar",
    ):
        if not self.ready:
            return

        current_state = self.workcell.capture_state()

        # No provision for a different initial starting state to workcell snapshot state
        # (could add later)

        # Use standing pyatk instance to get IK solutions:
        # Take out link list if we can ensure fwd_kinematics call works
        goal_joint_configs, atk_articulation, atk_link_list = await self.get_link_configs_from_ee_pose(
            current_state, world_T_ee_goal, articulation_parts_list, articulation_descriptor, grasp_name,
            min_joint_values, max_joint_values
        )

        if len(goal_joint_configs) < 1:
            return

        # The planner class instance sets a boolean indicating articulation or not (default not)
        # Articulation planner points to a different mp call (to accomodate different unpacking procedure)

        # Cannot pass araas parts, unpack and repack:
        planner_parts_list = []
        for item in articulation_parts_list:
            part, part_type, part_name = item
            part_pose = part.get_transform().extract_row_major()
            # print(part_pose)
            planner_parts_list.append([part_type, part_name, part_pose])

        print("sending request to path planner")

        for goal in goal_joint_configs:
            self.request.put(
                (
                    current_state,
                    planner_parts_list,
                    articulation_descriptor,
                    grasp_name,
                    goal,
                    min_joint_values,
                    max_joint_values,
                    timeout,
                    planner_type,
                )
            )

        # wait for solution
        results = []

        while True:
            # sleep to allow the other async tasks to update
            await trio.sleep(0.1)
            try:
                # non-blocking get
                result = self.response.get(False)
                results.append(result)
                if len(results) == len(goal_joint_configs):
                    break
            except queue.Empty:
                continue

        best_path = self.choose_best_path(results)
        # best_path = self.choose_shortest_joint_path(results)

        if best_path == None:
            return "failed to find a valid path to the goal pose"

        # Instead of running the trajectory: create a trajectory using toppra with some default
        # dynamic options (make sure max speed and velocity is high, and duration is fairly long)

        # Use number_of_keypoints to define trajectory slicing.
        # atk_workcell (persistent) already holds atk_articulation instance, we should be able to just pass through

        sliced_trajectory = self.create_articulation_trajectory(best_path, atk_articulation, number_of_keypoints)
        print(len(sliced_trajectory))

        # Then convert sliced trajectory into a reference-frame key list for the articulation, using the IK
        # solver
        sliced_articulation_path = self.get_tcp_from_articulation_trajectory(sliced_trajectory, atk_articulation,
                                                                             atk_link_list)

        print(len(sliced_articulation_path))

        # return this to the calling function.
        return sliced_articulation_path

    def create_articulation_trajectory(self, keys, atk_articulation, number_of_keypoints=100):

        # Create a spline between path planner waypoints
        joint_waypoints = np.array(keys)
        param_array = np.linspace(0, 1, np.shape(joint_waypoints)[0])
        spline_path = ta.SplineInterpolator(param_array, joint_waypoints)

        accel_constraints = atk_articulation.get_max_joint_accelerations()  # created at articulation initialization, set these high
        vel_constraints = atk_articulation.get_max_joint_velocities()  # ditto
        pc_vel = constraint.JointVelocityConstraint(vel_constraints)
        pc_acc = constraint.JointAccelerationConstraint(accel_constraints)

        # We solve the parametrization problem using the`ParametrizeConstAccel` parametrizer. This parametrizer is the
        # classical solution, guarantee constraint and boundary conditions satisfaction.

        # Try without explicit duration (can we still access traj.duration?)
        # ok, bit weird? try setting a duration according to keypoint size
        sample_rate_ms = 100
        duration = number_of_keypoints * (sample_rate_ms / 1000)
        print(duration)
        instance = algo.TOPPRAsd(
            [pc_vel, pc_acc], spline_path, parametrizer="ParametrizeConstAccel"
        )
        instance.set_desired_duration(duration)

        traj = instance.compute_trajectory()

        # we are looking for n samples over (duration) in ms
        t_vec = np.linspace(
            0, traj.duration, int(np.floor(1000 * traj.duration / sample_rate_ms))
        )

        q = traj.eval(t_vec)  # sampled joint positions
        print(len(q))

        # Shouldn't need dynamic components but could include for debugging purposes
        # qd = traj.evald(t_vec)  # sampled joint velocities
        # qdd = traj.evaldd(t_vec)  # sampled joint accelerations

        return q

    def get_tcp_from_articulation_trajectory(self, sliced_trajectory, atk_articulation, atk_link_list):

        sliced_articulation_path = []

        for joint_config in sliced_trajectory:
            # Need to set link transforms according to each slice, update part locations,
            atk_articulation.set_joint_angles(joint_config)
            link_poses = atk_articulation.get_link_transforms()

            # Check if get_fwd_kinematics returns the same info as setting the parts, if not, fix it.
            tcp_pose = atk_articulation.get_fwd_kinematics(joint_config)
            sliced_articulation_path.append(tcp_pose)  # atk_articulation.get_flange_transform())

        return sliced_articulation_path

    def define_offset_transform(self, robot, reference_actor):
        # check that the refernce actor is attached to the correct robot, then calculate the fixed flange-TCP offset

        w_T_flange = robot.get_flange_transform()
        w_T_tcp = reference_actor.get_tcp_transform()

        # check that the reference actor is attached to a robot:
        robot_parent = []
        try:
            robot_parent = reference_actor.get_attached_robot()
        except:
            raise PlanningError("Planning reference frame is not attached to a robot")

        if robot_parent is not robot:
            raise PlanningError("Planning reference frame is not attached to correct robot")

        # Calculate offset in flange frame:
        tcp_T_flange = pyatk.Transform((w_T_tcp.invert()).multiply(w_T_flange))

        return tcp_T_flange

    async def plan_joint_path(
            self,
            robot,
            world_T_ee_goal,
            min_joint_values=None,
            max_joint_values=None,
            initial_joint_state=None,
            timeout=1.0,
            planner_type="LazyPRMstar",
            reference_actor=None,
    ):
        if not self.ready:
            return

        # check for null motions:
        current_pose = robot.get_flange_transform()
        # capture the current workcell state

        if self.articulation_planner:
            return "error: articulation planner can only be called with plan_linkage_path"

        current_state = self.workcell.capture_state()

        tcp_T_flange = None
        if reference_actor is not None:
            tcp_T_flange = self.define_offset_transform(robot, reference_actor)

        if tcp_T_flange is not None:
            # convert goal to a flange transform:
            world_T_ee_goal = world_T_ee_goal.multiply(tcp_T_flange)

        if current_pose.is_equal(0.1, 0.01, world_T_ee_goal):
            print("Robot is already at goal position, returning with no path ...")
            return

        # if we want to specify a starting robot state that is different from the capture state:
        if initial_joint_state:
            for robot_config in current_state["configs"]:
                if robot_config["unique_name"] == robot.unique_name:
                    robot_config["Joint 0"] = initial_joint_state[0]
                    robot_config["Joint 1"] = initial_joint_state[1]
                    robot_config["Joint 2"] = initial_joint_state[2]
                    robot_config["Joint 3"] = initial_joint_state[3]
                    robot_config["Joint 4"] = initial_joint_state[4]
                    robot_config["Joint 5"] = initial_joint_state[5]

        goal_joint_configs = await self.get_joint_configs_from_ee_pose(
            current_state, world_T_ee_goal, robot, min_joint_values, max_joint_values
        )

        if len(goal_joint_configs) < 1:
            return

        for goal in goal_joint_configs:
            self.request.put(
                (
                    current_state,
                    robot.unique_name,
                    goal,
                    min_joint_values,
                    max_joint_values,
                    timeout,
                    planner_type,
                )
            )

        # wait for the solution
        results = []
        while True:
            # sleep to allow the other async tasks to update
            await trio.sleep(0.1)
            try:
                # non-blocking get
                result = self.response.get(False)
                results.append(result)
                if len(results) == len(goal_joint_configs):
                    break
            except queue.Empty:
                continue

        # choose the best one
        best_path = self.choose_best_path(results)
        # best_path = self.choose_shortest_joint_path(results)
        if best_path == None:
            return
        else:
            return best_path

    async def plan_and_execute_raw_joint_trajectory(
            self,
            robot,
            target_joints,
            speed_factor=1.0,
            min_joint_values=None,
            max_joint_values=None,
            timeout=1.0,
            planner_type="LazyPRMstar",
    ):

        print("plan_and_execute_raw_joint_trajectory")
        print(target_joints)
        if not self.ready:
            return "planner has shutdown"

        if self.articulation_planner:
            return "error: articulation planner can only be called with plan_linkage_path"

        # capture the current workcell state
        current_state = self.workcell.capture_state()

        goal_joint_configs = [target_joints]

        if len(goal_joint_configs) < 1:
            return "a valid goal config for the request was not found"

        if (len(self.roadmap_path) < 1) and timeout < 2.0:
            print(
                "Solver timeout may be too short to find paths without an existing roadmap. Extending to 2.0s..."
            )
            timeout = 2.0

        # post a path planning job for each goal config
        for goal in goal_joint_configs:
            self.request.put(
                (
                    current_state,
                    robot.unique_name,
                    goal,
                    min_joint_values,
                    max_joint_values,
                    timeout,
                    planner_type,
                )
            )
        # wait for the solution
        results = []
        print("Waiting for solution")
        while True:
            # sleep to allow the other async tasks to update
            await trio.sleep(0.1)
            try:
                # non-blocking get
                result = self.response.get(False)
                results.append(result)
                if len(results) == len(goal_joint_configs):
                    break
            except queue.Empty:
                continue

        # choose the best one
        best_path = self.choose_best_path(results)
        # best_path = self.choose_shortest_joint_path(results)
        if best_path == None:
            return "failed to find a valid path to the goal pose"

        # start the joint trajectory
        print("run trajectory", best_path, )
        await self.run_trajectory(
            robot, best_path, speed_factor=speed_factor
        )
        return "finished"

    async def plan_and_execute_joint_trajectory(
            self,
            robot,
            world_T_ee_goal,
            duration=None,
            min_joint_values=None,
            max_joint_values=None,
            timeout=1.0,
            accel_constraints=None,
            vel_constraints=None,
            planner_type="LazyPRMstar",
            reference_actor=None,
    ):
        if not self.ready:
            return "planner has shutdown"

        if self.articulation_planner:
            return "error: articulation planner can only be called with plan_linkage_path"

        # capture the current workcell state
        current_state = self.workcell.capture_state()

        tcp_T_flange = None
        if reference_actor is not None:
            tcp_T_flange = self.define_offset_transform(robot, reference_actor)

        if tcp_T_flange is not None:
            # convert goal to a flange transform:
            world_T_ee_goal = world_T_ee_goal.multiply(tcp_T_flange)

        # check for null motions:
        current_pose = robot.get_flange_transform()

        if current_pose.is_equal(0.1, 0.01, world_T_ee_goal):
            print("Robot is already at goal position, returning ...")
            return

        # get the set of goal configs
        goal_joint_configs = await self.get_joint_configs_from_ee_pose(
            current_state, world_T_ee_goal, robot, min_joint_values, max_joint_values
        )

        if len(goal_joint_configs) < 1:
            return "a valid goal config for the request was not found"

        if (len(self.roadmap_path) < 1) and timeout < 2.0:
            print(
                "Solver timeout may be too short to find paths without an existing roadmap. Extending to 2.0s..."
            )
            timeout = 2.0

        # post a path planning job for each goal config
        for goal in goal_joint_configs:
            self.request.put(
                (
                    current_state,
                    robot.unique_name,
                    goal,
                    min_joint_values,
                    max_joint_values,
                    timeout,
                    planner_type,
                )
            )
        # wait for the solution
        results = []
        while True:
            # sleep to allow the other async tasks to update
            await trio.sleep(0.1)
            try:
                # non-blocking get
                result = self.response.get(False)
                results.append(result)
                if len(results) == len(goal_joint_configs):
                    break
            except queue.Empty:
                continue

        # choose the best one
        best_path = self.choose_best_path(results)
        # best_path = self.choose_shortest_joint_path(results)
        if best_path == None:
            return "failed to find a valid path to the goal pose"

        # start the joint trajectory
        await self.run_trajectory(
            robot, best_path, duration, accel_constraints, vel_constraints
        )
        return "finished"

    async def plan_multipoint_joint_trajectory(
            self,
            robot,
            goal_sequence,
            min_joint_values=None,
            max_joint_values=None,
            timeout=1.0,
            accel_constraints=None,
            vel_constraints=None,
            planner_type="LazyPRMstar",
            reference_actor=None,
    ):
        if not self.ready:
            return "planner has shutdown"

        if self.articulation_planner:
            return "error: articulation planner can only be called with plan_linkage_path"

        # capture the current workcell state (we will modify this internally for each subsequent path plan)

        current_state = self.workcell.capture_state()

        tcp_T_flange = None
        if reference_actor is not None:
            tcp_T_flange = self.define_offset_transform(robot, reference_actor)

        # initialise a joint waypoint sequence:
        full_path = []

        if len(self.roadmap_path) < 1:
            print(
                "Multi-point path planning works best with a persistent roadmap. Generating a default roadmap path ... "
            )
            parent_path = os.path.join(os.getcwd(), ".araas_temp")

            os.makedirs(parent_path, exist_ok=True)
            roadmap_path = os.path.join(parent_path, "temp_ompl_pd.graph")

            self.roadmap_path = roadmap_path
            self.roadmap_expansion = True

        # for each goal tcp point, generate a set of sub-paths in joint space:
        count = 0
        path_len = len(goal_sequence)

        for world_T_goal in goal_sequence:
            print("At pose " + str(count) + " of " + str(path_len))
            count += 1

            # get the set of goal configs
            if tcp_T_flange is not None:
                # convert goal to a flange transform:
                goal_pose = world_T_goal.multiply(tcp_T_flange)
            else:
                goal_pose = world_T_goal

            try:
                goal_joint_configs = await self.get_joint_configs_from_flange_pose(
                    current_state, goal_pose, robot, min_joint_values, max_joint_values
                )
            except:
                print("Requested joint configuration results in collision, skipping goal")
                continue

            # check validity
            if len(goal_joint_configs) < 1:
                print("A valid goal config for the request was not found, moving to next point")
                continue

            # post a path planning job for each goal config
            for goal in goal_joint_configs:
                self.request.put(
                    (
                        current_state,
                        robot.unique_name,
                        goal,
                        min_joint_values,
                        max_joint_values,
                        timeout,
                        planner_type,
                    )
                )

            # wait for the solution
            results = []

            while True:
                # sleep to allow the other async tasks to update
                await trio.sleep(0.1)
                try:
                    # non-blocking get
                    result = self.response.get(False)
                    results.append(result)
                    if len(results) == len(goal_joint_configs):
                        break
                except queue.Empty:
                    continue

            # choose the best one
            best_path = self.choose_best_path(results)
            # best_path = self.choose_shortest_joint_path(results)
            if best_path is None:
                print("Failed to find a valid path to the goal pose, jumping to next goal")
                continue
            elif best_path[0] == best_path[len(best_path) - 1]:
                raise PlanningError("Warning: Start and end positions are the same, no trajectory possible")

            # get final joint pose and update robot pose in the workcell state
            end_state = best_path[len(best_path) - 1]

            # this method is not robust to changing joint configs, need to generalise
            for robot_config in current_state["configs"]:
                if robot_config["unique_name"] == robot.unique_name:
                    robot_config["Joint 0"] = end_state[0]
                    robot_config["Joint 1"] = end_state[1]
                    robot_config["Joint 2"] = end_state[2]
                    robot_config["Joint 3"] = end_state[3]
                    robot_config["Joint 4"] = end_state[4]
                    robot_config["Joint 5"] = end_state[5]

            full_path.append(best_path)

        return full_path

    async def plan_and_execute_multipoint_joint_trajectory(
            self,
            robot,
            goal_sequence,
            duration=None,
            min_joint_values=None,
            max_joint_values=None,
            timeout=1.0,
            accel_constraints=None,
            vel_constraints=None,
            planner_type="LazyPRMstar",
            reference_actor=None,
    ):
        if not self.ready:
            return "planner has shutdown"

        # capture the current workcell state (we will modify this internally for each subsequent path plan)

        current_state = self.workcell.capture_state()

        tcp_T_flange = None
        if reference_actor is not None:
            tcp_T_flange = self.define_offset_transform(robot, reference_actor)

        # initialise a joint waypoint sequence:
        full_path = []

        if len(self.roadmap_path) < 1:
            print(
                "Multi-point path planning works best with a persistent roadmap. Generating a default roadmap path ... "
            )
            parent_path = os.path.join(os.getcwd(), ".araas_temp")

            os.makedirs(parent_path, exist_ok=True)
            roadmap_path = os.path.join(parent_path, "temp_ompl_pd.graph")

            self.roadmap_path = roadmap_path
            self.roadmap_expansion = True

        # for each goal tcp point, generate a set of sub-paths in joint space:
        for world_T_goal in goal_sequence:
            print("Starting new subpath")  # informative only

            # get the set of goal configs
            if tcp_T_flange is not None:
                # convert goal to a flange transform:
                goal_pose = world_T_goal.multiply(tcp_T_flange)
            else:
                goal_pose = world_T_goal

            # get the set of goal configs
            goal_joint_configs = await self.get_joint_configs_from_ee_pose(
                current_state, goal_pose, robot, min_joint_values, max_joint_values
            )

            # check validity
            if len(goal_joint_configs) < 1:
                return "a valid goal config for the request was not found"

            # post a path planning job for each goal config
            for goal in goal_joint_configs:
                self.request.put(
                    (
                        current_state,
                        robot.unique_name,
                        goal,
                        min_joint_values,
                        max_joint_values,
                        timeout,
                        planner_type,
                    )
                )

            # wait for the solution
            results = []

            while True:
                # sleep to allow the other async tasks to update
                await trio.sleep(0.1)
                try:
                    # non-blocking get
                    result = self.response.get(False)
                    results.append(result)
                    if len(results) == len(goal_joint_configs):
                        break
                except queue.Empty:
                    continue

            # choose the best one
            best_path = self.choose_best_path(results)
            # best_path = self.choose_shortest_joint_path(results)
            if best_path == None:
                return "failed to find a valid path to the goal pose"

            # get final joint pose and update robot pose in the workcell state
            end_state = best_path[len(best_path) - 1]

            # this method is not robust to changing joint configs, need to generalise
            for robot_config in current_state["configs"]:
                if robot_config["unique_name"] == robot.unique_name:
                    robot_config["Joint 0"] = end_state[0]
                    robot_config["Joint 1"] = end_state[1]
                    robot_config["Joint 2"] = end_state[2]
                    robot_config["Joint 3"] = end_state[3]
                    robot_config["Joint 4"] = end_state[4]
                    robot_config["Joint 5"] = end_state[5]

            full_path.append(best_path)

        # start the joint trajectory

        # we might need to repackage or flatten the path for the trajectory generator.
        flat_path = [waypoint for subpath in full_path for waypoint in subpath]
        await self.run_trajectory(
            robot, flat_path, duration, accel_constraints, vel_constraints
        )
        return "finished"

    def choose_best_path(self, results):
        # choose the result with the shortest ee path
        best_path = None
        best_score = None
        for i, r in enumerate(results):
            if len(r[1]) > 1:
                if best_score == None:
                    best_score = r[0]["path_length_end_effector"]
                    best_path = r[1]
                else:
                    if r[0]["path_length_end_effector"] < best_score:
                        best_score = r[0]["path_length_end_effector"]
                        best_path = r[1]
        return best_path

    def choose_shortest_joint_path(self, results):
        # choose the result with the shortest joint motion
        best_path = None
        best_score = None
        for r in results:
            if len(r[1]) > 1:
                if best_score == None:
                    best_score = r[0]["path_length_in_joint_space"]
                    best_path = r[1]
                else:
                    if r[0]["path_length_in_joint_space"] < best_score:
                        best_score = r[0]["path_length_in_joint_space"]
                        best_path = r[1]

        return best_path

    async def run_trajectory(
            self, robot, keys, speed_factor=1, accel_constraints=None, vel_constraints=None
    ):

        atk_robot = self.atk_workcell.get_robot(robot.unique_name)

        # provide the option to set acceleration, velocity constraints
        # use max robot settings if not available

        if accel_constraints is None:
            accel_constraints = atk_robot.get_max_joint_accelerations()
        if vel_constraints is None:
            vel_constraints = atk_robot.get_max_joint_velocities()

        joint_waypoints = np.array(keys)

        param_array = np.linspace(0, 1, np.shape(joint_waypoints)[0])

        spline_path = ta.SplineInterpolator(param_array, joint_waypoints)

        pc_vel = constraint.JointVelocityConstraint(vel_constraints)
        pc_acc = constraint.JointAccelerationConstraint(accel_constraints)

        # We solve the parametrization problem using the`ParametrizeConstAccel` parametrizer. This parametrizer is the
        # classical solution, guarantee constraint and boundary conditions satisfaction.

        # we can also enforce a specific duration:
        # if duration is None:
        max_distance = max(calculate_total_distance(keys))
        print('max_distance: ', max_distance)
        duration = 10 * max_distance * speed_factor
        duration = max(duration, 3)

        instance = algo.TOPPRAsd(
            [pc_vel, pc_acc], spline_path, parametrizer="ParametrizeConstAccel"
        )
        instance.set_desired_duration(duration)

        traj = instance.compute_trajectory()
        
        if traj is None:
            print('traj is {}, skip'.format(traj))
            return
        # extract the trajectory items
        sample_rate_ms = 100

        # we are looking for n samples over (duration) in ms
        t_vec = np.linspace(
            0, traj.duration, int(np.floor(1000 * traj.duration / sample_rate_ms))
        )
        q = traj.eval(t_vec)  # sampled joint positions
        qd = traj.evald(t_vec)  # sampled joint velocities
        qdd = traj.evaldd(t_vec)  # sampled joint accelerations

        # run a joint action
        if (len(q) > 1):
            result = await robot.execute_joint_trajectory(q, qd, qdd, t_vec)

    async def get_tcp_position_from_joint_trajectory(
            self, current_state, robot, joint_q
    ):
        """calculate forward kinematics over trajectory, for visualisation/storing output/etc"""
        self.atk_workcell.apply_state(json.dumps(current_state))
        atk_robot = self.atk_workcell.get_robot(robot.unique_name)
        init_config = atk_robot.get_joint_angles()

        ee_xforms = []
        for q in joint_q:
            atk_robot.set_joint_angles(q)
            ee_xforms.append(atk_robot.get_flange_transform())

        # return to initial config
        atk_robot.set_joint_angles(init_config)

        return ee_xforms

    async def get_joint_configs_from_ee_pose(
            self, current_state, world_T_ee_goal, robot, min_joint_values, max_joint_values
    ):
        self.atk_workcell.apply_state(json.dumps(current_state))
        atk_robot = self.atk_workcell.get_robot(robot.unique_name)
        if min_joint_values and max_joint_values:
            atk_robot.set_planning_joint_limits(min_joint_values, max_joint_values)

        init_config = atk_robot.get_joint_angles()

        possible_joint_solns = atk_robot.find_ik_solns(world_T_ee_goal)
        if len(possible_joint_solns) < 1:
            print(
                "Error: requested goal pose is not reachable in this workcell configuration. Check inputs and replan."
            )

        feasible_solutions = []
        for joint_config in possible_joint_solns:
            valid, corrected_config = atk_robot.shorten_joint_angles_from_ref(
                joint_config, init_config
            )
            if valid and atk_robot.check_valid_joint_angles(corrected_config, True):
                feasible_solutions.append(corrected_config)

        if len(feasible_solutions) < 1:
            print(
                "Error: requested goal pose is not reachable without collision. Check inputs and replan."
            )

        return feasible_solutions

    async def get_link_configs_from_ee_pose(
            self, current_state, world_T_ee_goal, articulation_parts_list, articulation_descriptor, grasp_name,
            min_joint_values, max_joint_values
    ):
        self.atk_workcell.apply_state(json.dumps(current_state))

        atk_link_list = []
        for item in articulation_parts_list:
            part, part_type, part_name = item  # can also get the name directly from the PyPart attribute
            atk_link_list.append(self.atk_workcell.add_part(part_type, part_name, part.get_transform()))

        atk_articulation = self.atk_workcell.add_articulation("new_articulation", atk_link_list,
                                                              json.dumps(articulation_descriptor), grasp_name)

        if min_joint_values and max_joint_values:
            atk_articulation.set_planning_joint_limits(min_joint_values, max_joint_values)

        init_config = atk_articulation.get_joint_angles()

        possible_joint_solns = atk_articulation.find_ik_solns(world_T_ee_goal, 0.8, 0.02)

        if len(possible_joint_solns) < 1:
            print(
                "Error: requested goal pose is not reachable in this workcell configuration. Check inputs and replan."
            )
            return possible_joint_solns, atk_articulation

        feasible_solutions = []
        # articulation validity checking is failing. Does not seem to be a collision checking issue?

        for joint_config in possible_joint_solns:
            print("I assume we get here")
            valid, corrected_config = atk_articulation.shorten_joint_angles_from_ref(
                joint_config, init_config
            )
            print("corrected config is valid? ")
            print(valid)

            if valid and atk_articulation.check_valid_joint_angles(corrected_config, True):
                feasible_solutions.append(corrected_config)

        if len(feasible_solutions) < 1:
            print(
                "Error: requested goal pose is not reachable without collision. Check inputs and replan."
            )

        return feasible_solutions, atk_articulation, atk_link_list