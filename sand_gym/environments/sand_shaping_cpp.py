# General
import numpy as np

# robosuite
from robosuite.utils.observables import Observable, sensor
from robosuite.models.base import MujocoModel

# sand gym environments
from sand_gym.environments.single_arm_env import SingleArmEnv

# sand gym models
from sand_gym.models.tasks import SandShapingTask
from sand_gym.models.arenas import SandShapingArena

# sand gym utils
from sand_gym.utils.sand_simulator import SandSimulator
import sand_gym.utils.ik as ik
from sand_gym.utils.reward import Reward
from sand_gym.utils.pose_samplers import UniformRandomPoseSampler
from sand_gym.utils.mujoco_sim import MujocoSimUtils
import sand_gym.utils.bcpp as bcpp


class CoveragePathPlanner(SingleArmEnv):
    """
    This class corresponds to the sand shaping task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!
        env_configuration (str): Specifies how to position the robots within the environment (default is "default").
            For most single arm environments, this argument has no impact on the robot setup.
        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param
        gripper_types (str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param
        use_camera_obs (bool): if True, every observation includes rendered image(s)
        use_object_obs (bool): if True, include object (cube) information in
            the observation.
        placement_initializer (ObjectPositionSampler): if provided, will
            be used to place objects on every reset, else a UniformRandomSampler
            is used by default.
        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.
        has_offscreen_renderer (bool): True if using off-screen rendering
        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse
        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.
        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.
        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).
        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.
        horizon (int): Every episode lasts for exactly @horizon timesteps.
        ignore_done (bool): True if never terminating the environment (ignore @horizon).
        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables
        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.
            :Note: At least one camera must be specified if @use_camera_obs is True.
            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).
        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.
        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.
        camera_segmentations (None or str or list of str or list of list of str): Camera segmentation(s) to use
            for each camera. Valid options are:

                `None`: no segmentation sensor used
                `'instance'`: segmentation at the class-instance level
                `'class'`: segmentation at the class level
                `'element'`: segmentation at the per-geom level

            If not None, multiple types of segmentations can be specified. A [list of str / str or None] specifies
            [multiple / a single] segmentation(s) to use for all cameras. A list of list of str specifies per-camera
            segmentation setting(s) to use.
        early_termination (bool): If True, episode is allowed to finish early.
        initial_gripper_pos_randomization (bool): If True, Gaussian noise will be added to the initial position of the gripper.
    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        base_types="default",
        controller_configs=None,
        lite_physics=False,
        gripper_types="default",
        use_camera_obs=True,
        use_object_obs=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=False,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        camera_config=None,
        early_termination=True,
        initial_gripper_pose_above_sand_bed=[0.0,0.0,0.0,0.0,0.0,0.0],
        initial_gripper_pos_randomization=False,
        initial_gripper_randomization_range=0.0,
        initial_gripper_randomization_angle=0.0,
        initial_gripper_pos_in_goal=False,
        goal_randomization=False,
        mujoco_passive_viewer=False,
        arena_config=None,
        sand_sim_config=None,
        reward_config=None,
        termination_config=None,
        success_config=None,
        observations_config=None,
        dataset_config=None,
        params = None,
        verbose=0
    ):
        assert gripper_types == "SandShapingGripperQuadratic",\
            "Tried to specify gripper other than SandShapingGripperQuadratic in SandShaping environment!"

        assert robots == "UR5e", \
            "Robot must be UR5e!"

        # params
        self.params = params

        # configs
        self.camera_config = camera_config
        self.arena_config = arena_config
        self.sand_sim_config = sand_sim_config
        self.reward_config=reward_config
        self.termination_config=termination_config
        self.success_config = success_config
        self.observations_config=observations_config
        self.observation_keys = self.observations_config["observation_keys"]
        self.flat_observations = self.observations_config.get("flat_observations", True)
        if "info_observation_keys" in observations_config:
            self.info_observation_keys = self.observations_config["info_observation_keys"]
        else:
            self.info_observation_keys = []
        self.dataset_config = dataset_config
        self.grid_size = self.sand_sim_config["grid_size"]

        # settings for table top
        self.table_full_size=self.arena_config["table_full_size"]
        self.table_friction=self.arena_config["table_friction"]
        self.table_offset=self.arena_config["table_offset"]
        self.robot_offset=self.arena_config["robot_offset"]
        self.robot_ori_offset=self.arena_config.get("robot_ori_offset", [0.0,0.0,0.0])
        self.mujoco_arena_origin = np.asarray(self.arena_config.get("mujoco_arena_origin", [0.0,0.0,0.0]))
        self.sand_box_offset = self.arena_config.get("sand_box_offset", np.asarray(self.table_offset) + np.asarray(self.robot_offset))
        self.has_legs=self.arena_config["has_legs"]
        self.table_top = self.table_offset
        self.arena_name = self.arena_config.get("arena_name", "sand_shaping_arena")
        self.arena_xml = "models/assets/arenas/" + self.arena_name + ".xml"

        # settings for joint initialization noise (Gaussian)
        self.mu = 0
        self.sigma = 0.010

        # Init and reset all rewards
        self.reset_all_rewards()

        # Init contacts
        self.table_contact = False
        self.sand_box_walls_contact = False

        # Init success
        self.success = False

        # reward configuration
        self.reward_utils = None

        self.initial_sand_height = self.sand_sim_config["init_height_in_mm"]/1000
                                            
        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # randomization settings
        self.initial_gripper_pose_above_sand_bed = np.array(initial_gripper_pose_above_sand_bed)
        
        self.initial_gripper_pos_randomization = initial_gripper_pos_randomization
        self.initial_gripper_randomization_range = np.array(initial_gripper_randomization_range)
        self.initial_gripper_randomization_angle = np.array(initial_gripper_randomization_angle)
        self.initial_gripper_pos_in_goal = initial_gripper_pos_in_goal
        
        if self.initial_gripper_pos_randomization:
            self.pose_sampler = None

        self.goal_randomization = goal_randomization
        # sand simulation settings
        self.sand_box_dims = self.sand_sim_config["sand_box_dims"]

        # misc settings
        self.early_termination = early_termination
        self.current_iteration = 0
        self.current_episode = 0

        # Coverage Path Planner
        self.bcpp_plan_world_coords = []
        
        # verbosity level
        self.verbose = verbose

        # Class objects
        self.sand_simulator = None
        self.reward_utils = None

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            base_types=base_types,
            controller_configs=controller_configs,
            lite_physics=lite_physics,
            gripper_types=gripper_types,
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            mujoco_passive_viewer=mujoco_passive_viewer
        )

    def check_dict_for_nan(self, observations, raise_error=True):       
        nan_detected = False
        for key, value in observations.items():
            if np.isnan(value).any():
                nan_detected = True
                print()
                print(f"NaN detected in observation '{key}'!")
                print(f"Observation '{key}': {value}")
        if nan_detected and raise_error:
            raise ValueError
        
    def reward(self, action=None, achieved_goal=None, desired_goal=None):
        """
        Reward function for the task.

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """               
        # Check and update success
        self._check_success()
        # Compute reward
        if self.reward_utils is not None:
            self.reward_value, self.reward_components = self.reward_utils.compute_reward(achieved_goal=achieved_goal, desired_goal=desired_goal)
            if np.isnan(self.reward_value).any():
                print(f"NaN detected in reward")
                print(f"reward: {self.reward_value}")
                raise ValueError
            
            self.check_dict_for_nan(self.reward_components)
        else:
            self.reward_value = 0.0
            self.reward_components = {}

        return self.reward_value


    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        self.robots[0].robot_model.set_base_xpos(self.robot_offset)
        self.robots[0].robot_model.set_base_ori(self.robot_ori_offset)
        sand_box_offset_temp = self.sand_box_offset

        # Load model for table top workspace
        mujoco_arena = SandShapingArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            has_legs=self.has_legs,
            sand_sim_config=self.sand_sim_config,
            sand_box_offset=sand_box_offset_temp,
            xml=self.arena_xml,
            mujoco_arena_origin=self.mujoco_arena_origin,
            )
        
        # Arena always gets set to zero origin
        mujoco_arena.set_origin(self.mujoco_arena_origin)
        
        # task includes arena, robot, and objects of interest
        self.model = SandShapingTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots]
        )
        
    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()
        # additional object references from this env
        self.gripper_body_name = self.robots[0].gripper[self.robots[0].arms[0]].root_body # gripper0_gripper_base
        self.gripper_body_id = self.sim.model.body_name2id(self.gripper_body_name)

        self.gripper_geom_name = self.robots[0].gripper[self.robots[0].arms[0]].naming_prefix +"gripper_interaction"
        self.gripper_geom_id = self.sim.model.geom_name2id(self.gripper_geom_name)
        tool_half_extends = self.sim.model.geom_aabb[self.gripper_geom_id][3:]
        self.tool_dims = (2*tool_half_extends[0], 2*tool_half_extends[1], 2*tool_half_extends[2])
        tool_height = self.tool_dims[2]
        robot_init_z_pos = tool_height+self.initial_sand_height+self.sand_box_dims[3]
        self.robot_init_pos = tuple(self.initial_gripper_pose_above_sand_bed[:3] + np.array([0.0,  0.0, robot_init_z_pos]))
        self.robot_init_quat = tuple(np.array([0.707107, -0.707107, -0.0, 0.0])) # Upright gripper orientation (x,y,z,w)
        self.robot_init_pose = [self.robot_init_pos, self.robot_init_quat]

        # Create pose sampler
        if self.initial_gripper_pos_randomization and self.pose_sampler is None:
            x_half = (self.initial_gripper_randomization_range[0]/2) - (self.tool_dims[0]/2)
            y_half = (self.initial_gripper_randomization_range[1]/2) - (self.tool_dims[1]/2)
            x_range=[-x_half, x_half]
            y_range=[-y_half, y_half]
            z_range=[0.0, self.initial_gripper_randomization_range[2]]
            # Create a placement sampler without collision checking
            self.pose_sampler = UniformRandomPoseSampler(
                name="pose_sampler",
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                rotation=self.initial_gripper_randomization_angle.tolist(),
                reference_pos=self.robot_init_pos,
                reference_quat=self.robot_init_quat,
            )

        if self.sand_simulator is None:
            self.sand_box_walls_geom_name_list = ["tray_bottom_col", "tray_front_col", "tray_back_col", "tray_left_col", "tray_right_col"]
            self.sand_box_geom_name = "sand_box_vis"
            sand_box_offset_temp = self.sand_box_offset
            
            if self.has_renderer:
                render_callback = self.render
            else:
                render_callback = None
            self.correct_render_config = self.sand_sim_config.get("correct_render_config", None)
            if self.correct_render_config is not None:
                self.correct_render = self.correct_render_config["correct_render"]
            else:
                self.correct_render = True
            self.mujoco_sim_utils = MujocoSimUtils(self)
            self.sand_simulator = SandSimulator(self.sim,
                                        self.grid_size,
                                        self.sand_box_dims[2],
                                        sand_geom_name=self.sand_box_geom_name,
                                        sand_box_offset=sand_box_offset_temp,
                                        init_height_in_mm=self.sand_sim_config["init_height_in_mm"],
                                        angle_of_repose=self.sand_sim_config["angle_of_repose"],
                                        height_perturbation_range_in_mm=self.sand_sim_config["height_perturbation_range_in_mm"],
                                        tool_body_name=self.gripper_body_name,
                                        tool_geom_name=self.gripper_geom_name,
                                        dataset_config=self.sand_sim_config["dataset_config"],
                                        render_callback=render_callback,
                                        correct_render_config=self.correct_render_config,
                                        step_simulation_callback=self.mujoco_sim_utils.update_mujoco_heightfield,
                                        add_visual_element_callback=self.mujoco_sim_utils.add_visual_element,
                                        goal_randomization=self.goal_randomization,
                                        env=self,
                                        verbose=self.verbose,
                                        )
            self.sand_simulator.reset()
        if self.reward_config is not None and self.reward_utils is None:
            self.reward_utils = Reward(config=self.reward_config, sim=self)
        
    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix

        if len(self.observation_keys) == 0:
            merged_observation_keys = list(self.info_observation_keys)
        elif len(self.info_observation_keys) == 0:
            merged_observation_keys = list(self.observation_keys)
        else:
            merged_observation_keys = list(self.observation_keys) + [item for item in list(self.info_observation_keys) if item not in list(self.observation_keys)]

        # Remove unnecessary observables
        if "joint_pos" not in merged_observation_keys:
            del observables[pf + "joint_pos"]
        if "joint_pos_cos" not in merged_observation_keys:
            del observables[pf + "joint_pos_cos"]
        if "joint_pos_sin" not in merged_observation_keys:
            del observables[pf + "joint_pos_sin"]
        if "joint_vel" not in merged_observation_keys:
            del observables[pf + "joint_vel"]
        if "joint_acc" not in merged_observation_keys:
            del observables[pf + "joint_acc"]
        if "gripper_qvel" not in merged_observation_keys:
            del observables[pf + "gripper_qvel"]
        if "gripper_qpos" not in merged_observation_keys:
            del observables[pf + "gripper_qpos"]
        if "eef_pos" not in merged_observation_keys:
            del observables[pf + "eef_pos"]
        if "eef_quat" not in merged_observation_keys:
            del observables[pf + "eef_quat"]
        if "eef_quat_site" not in merged_observation_keys:
            del observables[pf + "eef_quat_site"]

        sensors = []
        
        if "heightmap_diff" in merged_observation_keys:
            @sensor(modality="heightmap")
            def heightmap_diff(obs_cache):
                difference_heightmap = self.sand_simulator.get_current_difference_heightfield(flat=self.flat_observations, abs=False, goal_mask=self.observation_keys["heightmap_diff"].get("goal_mask", False), convert_to_obs_shape=True)
                return difference_heightmap
            
        if "gripper_to_goal_area_distance_mask_based" in merged_observation_keys:
            @sensor(modality="distance")
            def gripper_to_goal_area_distance_mask_based(obs_cache):
                gripper_to_goal_area_distance_mask_based = self.sand_simulator.get_gripper_to_goal_area_distance_mask_based(
                    tool_2d_world_pos=self._eef_xpos[:2],
                    dist_3d=True,
                    shrink_goal_area=self.reward_config["reward_keys"]["gripper_to_goal_area_distance_mask_based"]["shrink_goal_area"],
                    corner_dist=self.reward_config["reward_keys"]["gripper_to_goal_area_distance_mask_based"]["corner_dist"])
                return gripper_to_goal_area_distance_mask_based

        if "heightmap_goal" in merged_observation_keys:
            @sensor(modality="heightmap")
            def heightmap_goal(obs_cache):
                goal_heightmap = self.sand_simulator.get_goal_heighfield(flat=self.flat_observations, convert_to_obs_shape=True)
                return goal_heightmap

        if "goal_mask" in merged_observation_keys:
            @sensor(modality="mask")
            def goal_mask(obs_cache):
                goal_mask = self.sand_simulator.get_goal_mask(flat=self.flat_observations, shrink_goal_area=self.observation_keys["goal_mask"]["shrink_goal_area"], convert_to_obs_shape=True)
                return goal_mask

        if "tool_mask" in merged_observation_keys:
            @sensor(modality="mask")
            def tool_mask(obs_cache):
                tool_mask = self.sand_simulator.get_tool_mask(flat=self.flat_observations, convert_to_obs_shape=True)
                return tool_mask
                     
        if "heightmap_current" in merged_observation_keys:
            @sensor(modality="heightmap")
            def heightmap_current(obs_cache):
                current_heightmap = self.sand_simulator.get_current_heightfield(flat=self.flat_observations, goal_mask=self.observations_config["goal_mask"], convert_to_obs_shape=True)
                return current_heightmap

        if "heightmap_diff" in merged_observation_keys:
            sensors.append(heightmap_diff)

        if "gripper_to_goal_area_distance_mask_based" in merged_observation_keys:
            sensors.append(gripper_to_goal_area_distance_mask_based)
        
        if "heightmap_goal" in merged_observation_keys:
            sensors.append(heightmap_goal)

        if "goal_mask" in merged_observation_keys:
            sensors.append(goal_mask)

        if "tool_mask" in merged_observation_keys:
            sensors.append(tool_mask)

        if "heightmap_current" in merged_observation_keys:
            sensors.append(heightmap_current)

        names = [s.__name__ for s in sensors]

        # Create observables
        for name, s in zip(names, sensors):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
            )

        return observables

    def reset_all_rewards(self):
        # Reset all rewards
        self.reward_value = 0.0
        if self.reward_config is not None:
            self.reward_components = dict.fromkeys(self.reward_config["reward_keys"], 0.0)
        else:
            self.reward_components = {}

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset sand bed
        self.sand_simulator.reset()
        
        self.success = False
        self.table_contact = False
        self.sand_box_walls_contact = False
        if self.reward_utils is not None:
            self.reset_all_rewards()
            self.reward_utils.reset_episode_rewards()

        self.current_iteration = 0

        if self.initial_gripper_pos_randomization:
            self.robot_init_pose = self.pose_sampler.sample()
        
        if self.initial_gripper_pos_in_goal:
            temp_init_pos = self.sand_simulator.get_random_2d_gripper_pos_within_goal()
            self.robot_init_pose[0] = [-temp_init_pos[0], -temp_init_pos[1], self.robot_init_pos[2]]

        # get initial joint positions for robot
        self.init_offset = np.copy(self.sand_box_offset)
        self.init_qpos = ik.get_initial_qpos(self.robots[0], self.init_offset, self.robot_init_pose[0], self.robot_init_pose[1])

        # override initial robot joint positions
        self.robots[0].set_robot_joint_positions(self.init_qpos)

        # update controller with new initial joints
        self.robots[0].composite_controller.part_controllers["right"].update_initial_joints(self.init_qpos)
        self.robots[0].composite_controller.part_controllers["right"].reset_goal()


        tool_width = self.tool_dims[0] # Assumes quadratic gripper footprint
        goal_heightfield = self.sand_simulator.get_goal_heighfield()
        goal_heightfield_in_m = goal_heightfield/1000
        goal_mask = self.sand_simulator.get_goal_mask()
        self.bcpp_plan_grid_coords = bcpp.plan_h_bcp(goal_heightfield_in_m, goal_mask, tool_width, self.sand_simulator.grid_cell_size_in_m)
        self.bcpp_plan_world_coords = []
        for waypoint_in_grid_coords in self.bcpp_plan_grid_coords:
            waypoint_in_world_coords = self.sand_simulator.convert_grid_to_world(waypoint_in_grid_coords[:2]).tolist()
            x = waypoint_in_world_coords[0]-self.sand_box_offset[0]
            y = waypoint_in_world_coords[1]-self.sand_box_offset[1]
            z = waypoint_in_grid_coords[2]-self.sand_box_offset[2]
            self.bcpp_plan_world_coords.append([x,y,z])

    def step(self, action=None):
        x,y,z = self.bcpp_plan_world_coords[self.current_iteration]
        ee_pos = np.array([x,y,z], dtype=np.float32)
        obs, reward, done, info = super().step(ee_pos)
        return obs, reward, done, info

    def _post_action(self, action):
        """
        In addition to super method, add additional info if requested

        Args:
            action (np.array): Action to execute within the environment

        Returns:
            3-tuple:
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) info about current env step
        """
        self.sand_simulator.step_sand_simulator()
        reward, done, info = super()._post_action(action)

        # check for early termination
        if self.early_termination:
            done = done or self._check_terminated()
        
        info["current_iteration"] = self.current_iteration
        info["current_episode"] = self.current_episode
        info["success"] = self.success # Required for CustomEvalCallback
        # done = True
        if done:
            info["execution_steps"] = self.current_iteration + 1
            if self.reward_utils is not None:
                info["episode_reward"] = self.reward_utils.get_episode_reward()
                info["episode_rewards"] = self.reward_utils.get_episode_rewards()
            info["height_diff"] = self.sand_simulator.get_episode_height_diff(goal_mask=False)
            info["goal_height_diff"] = self.sand_simulator.get_episode_height_diff(goal_mask=True)
            info["in_goal_cells_changed"] = self.sand_simulator.get_episode_manipulated_cells(criterion="unequal_init")
            info["out_goal_cells_changed"] = self.sand_simulator.get_episode_manipulated_cells(inverted=True, criterion="unequal_init")
            info["in_goal_mean_diff"] = self.sand_simulator.get_current_mean_diff_heightfield(where="in_goal")
            info["out_goal_mean_diff"] = self.sand_simulator.get_current_mean_diff_heightfield(where = "out_goal")
            if self.reward_config is not None:
                if "gripper_to_goal_area_distance_mask_based" in self.reward_config["reward_keys"]:
                    info["goal_area_dist"] = self.sand_simulator.get_gripper_to_goal_area_distance_mask_based(
                        tool_2d_world_pos=self._eef_xpos[:2],
                        shrink_goal_area=self.reward_config["reward_keys"]["gripper_to_goal_area_distance_mask_based"]["shrink_goal_area"],
                        corner_dist=self.reward_config["reward_keys"]["gripper_to_goal_area_distance_mask_based"]["corner_dist"])
        # Increase iteration value
        self.current_iteration += 1

        # Increase episode value
        if done:
            self.current_iteration = 0
            self.current_episode += 1
        
        return reward, done, info


    def visualize(self, vis_settings):
        """
        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

    def get_variable(self, variable_name):
        value = getattr(self, variable_name, None)
        return value
    
    def set_variable(self, variable_name, variable_value):
        return setattr(self, variable_name, variable_value)

    def _check_success(self):
        if self.current_iteration >= len(self.bcpp_plan_world_coords) -1:
            self.success = True
        else:
            self.success = False
        return self.success

    def _check_terminated(self):
        """
        Check if the task has completed one way or another. The following conditions lead to termination:
            - Collision with table
            - Joint Limit reached
            - Task success

        Returns:
            bool: True if episode is terminated
        """

        terminated = False

        # Prematurely terminate if contacting the table with the arm
        if self.termination_config["terminate_on_table_contact"]:
            if self.check_contact(self.robots[0].robot_model):
                if self.verbose >= 1:
                    print(40 * "-" + " COLLIDED " + 40 * "-")
                terminated = True

        # Prematurely terminate if reaching joint limits
        if self.termination_config["terminate_on_robot_joint_limit"]:
            if self.robots[0].check_q_limits():
                if self.verbose >= 1:
                    print(40 * '-' + " JOINT LIMIT " + 40 * '-')
                terminated = True

        # Prematurely terminate if task is success
        if self.termination_config["terminate_on_success"]:
            if self._check_success():
                if self.verbose >= 1:
                    print(40 * "+" + " FINISHED TASK " + 40 * "+")
                terminated = True

        return terminated


    def _get_contacts_objects(self, model):
        """
        Checks for any contacts with @model (as defined by @model's contact_geoms) and returns the set of
        contact objects currently in contact with that model (excluding the geoms that are part of the model itself).

        Args:
            model (MujocoModel): Model to check contacts for.

        Returns:
            set: Unique contact objects containing information about contacts with this model.

        Raises:
            AssertionError: [Invalid input type]
        """
        # Make sure model is MujocoModel type
        assert isinstance(model, MujocoModel), \
            "Inputted model must be of type MujocoModel; got type {} instead!".format(type(model))
        contact_set = set()
        try:
            for contact in self.sim.data.contact[: self.sim.data.ncon]:
                # check contact geom in geoms; add to contact set if match is found
                g1, g2 = self.sim.model.geom_id2name(contact.geom1), self.sim.model.geom_id2name(contact.geom2)
                if g1 in model.contact_geoms or g2 in model.contact_geoms:
                    contact_set.add(contact)
        except:
            pass

        return contact_set
    
    def _check_gripper_contact_with_table(self):
        """
        Check if the gripper is in contact with the tabletop.

        Returns:
            bool: True if gripper is in contact with table
        """
        return self.check_contact(self.robots[0].gripper, "table_collision")

    def _check_robot_contact_with_table(self):
        """
        Check if the gripper is in contact with the tabletop.

        Returns:
            bool: True if gripper is in contact with table
        """
        self.table_contact = self.check_contact(self.robots[0].robot_model, "table_collision")
        return self.table_contact

    def _check_gripper_contact_with_sand_box_walls(self):
        """
        Check if the gripper is in contact with the tabletop.

        Returns:
            bool: True if gripper is in contact with table
        """
        return self.check_contact(self.robots[0].gripper, self.sand_box_walls_geom_name_list)

    def _check_robot_contact_with_sand_box_walls(self):
        """
        Check if the gripper is in contact with the tabletop.

        Returns:
            bool: True if gripper is in contact with table
        """
        self.sand_box_walls_contact = self.check_contact(self.robots[0].robot_model, self.sand_box_walls_geom_name_list)
        return self.sand_box_walls_contact

    @property
    def _get_gripper_xpos(self):
        """
        Grabs tool center position

        Returns:
            np.array: sand box pos (x,y,z)
        """
        return np.array(self.sim.data.body_xpos[self.gripper_body_id])
    
    def set_sim_robot_joints(self, joints):
        self.robots[0].set_robot_joint_positions(joints)
        self.sim.forward()
        self.sync_mujoco_passive_viewer()
