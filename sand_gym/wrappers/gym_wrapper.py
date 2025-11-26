import numpy as np
import gymnasium as gym
from robosuite.wrappers import Wrapper
from collections import OrderedDict

class IndexEnvWrapper(Wrapper, gym.Env):
    def __init__(self, env, index, keys=None):
        super().__init__(env=env)
        self.index = index

    def reset(self, **kwargs):
        # Pass all arguments to the underlying environment's reset method
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["env_index"] = self.index  # Include environment index in info
        return obs, reward, terminated, truncated, info
    
class GymWrapperDictObs(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    
    def __init__(self, env, keys=None, info_keys=None, norm_obs=False, norm_limits=[-1.0, 1.0], camera_config=None, verbose=0):
        # Run super method
        super().__init__(env=env)

        self.verbose = verbose
        self.check_norm = False

        self.camera_config = camera_config

        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        self.norm_obs = norm_obs
        self.norm_limits = norm_limits

        self.observation_keys = keys
        self.info_observation_keys = info_keys

        if keys is not None:
            self.keys = self.create_key_list_from_mapping(self.observation_keys)
        
        if info_keys is not None:
            self.info_keys = self.create_key_list_from_mapping(self.info_observation_keys)

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()

        if (any("depth_seg" in sub for sub in self.info_observation_keys)
            or any("depth_seg" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.camera import add_depth_seg_to_obs
            depth_seg_observations = add_depth_seg_to_obs(observation=obs, robosuite_env=self.env.sim, camera_config=self.camera_config)
            obs.update(depth_seg_observations)

        if (any("reconstructed_heightmap_current" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_current" in sub for sub in self.observation_keys.keys())
            or any("reconstructed_heightmap_diff" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_diff" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.pointcloud import add_reconstructed_heightmap_to_obs
            reconstructed_heightmap_observations = add_reconstructed_heightmap_to_obs(obs=obs, env=self.env, camera_config=self.camera_config, verbose=self.verbose)
            obs.update(reconstructed_heightmap_observations)
    
        if (any("reconstructed_heightmap_diff" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_diff" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.pointcloud import add_reconstructed_heightmap_diff_to_obs
            reconstructed_heightmap_diff = add_reconstructed_heightmap_diff_to_obs(obs=obs, env=self.env, goal_mask=self.observation_keys["reconstructed_heightmap_diff"].get("goal_mask", False), verbose=self.verbose)
            obs.update(reconstructed_heightmap_diff)

        if (any("eef_pos_prev" in sub for sub in self.info_observation_keys)
            or any("eef_pos_prev" in sub for sub in self.observation_keys.keys())):
            self.eef_pos_previous = np.copy(obs["robot0_eef_pos"])
            obs["eef_pos_prev"] = self.eef_pos_previous
        
        self.modality_dims = {obs_key: obs[obs_key].shape for obs_key in obs}

        observation_space = OrderedDict()

        if type(self.keys) == dict:
            for key, value in self.keys.items():
                shape = self.modality_dims[key]
                if self.norm_obs:
                    low, high = self.norm_limits[0], self.norm_limits[1]
                else:
                    low, high = value["limits"]
                observation_space[key] = self.build_obs_space(shape=shape, low=low, high=high)
        elif type(self.keys) == list:
            for key in self.keys:
                shape = self.modality_dims[key]
                observation_space[key] = self.build_obs_space(shape=shape, low=-np.inf, high=np.inf)
        self.observation_space = gym.spaces.Dict(observation_space)

        low, high = self.env.action_spec
        self.action_space = gym.spaces.Box(low, high)

    def create_key_list_from_mapping(self, keys):
        if type(keys) == dict:
            parsed_keys = {}
            for key, value in keys.items():
                mapped_keys = self.key_mapping(key)
                for mapped_key in mapped_keys:
                    parsed_keys[mapped_key] = {}
                    if self.norm_obs:
                        parsed_keys[mapped_key]["limits"] = value["limits"]
                    else:
                        parsed_keys[mapped_key]["limits"] = [-np.inf, np.inf]
            return parsed_keys
        elif type(keys) == list:
            temp_list = []
            for key in keys:
                temp_list.extend(self.key_mapping(key))
            return temp_list
    
    def key_mapping(self, key):
        temp_list = []
        # Add object obs if requested
        if key == "object":
            temp_list.append("object-state")
        # Add image obs if requested
        elif key == "camera":
            for cam_name in self.env.camera_names:
                temp_list.append(f"{cam_name}_image")
        # Iterate over all robots to add to state
        elif key == "robot_proprio":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_proprio-state".format(idx))
        # Iterate over all robots to add to state
        elif key == "eef_pos":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_eef_pos".format(idx))
        elif key == "eef_quat":
            for idx in range(len(self.env.robots)):
                temp_list.append("robot{}_eef_quat".format(idx))
        else:
            temp_list.append(key)
        
        return temp_list

    def concat_obs(self, obs_dict, verbose=False):
        ob_lst = []
        for key in obs_dict.keys():
            ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)
    
    def filter_obs_dict_by_keys(self, obs_dict, keys):
        from sand_gym.utils.common import normalize_dict
        observations = OrderedDict()
        for key in keys:
            observations[key] = obs_dict[key]
        if self.norm_obs:
            observations = normalize_dict(observations, keys, self.norm_limits, check_norm=self.check_norm)
        return observations

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
    
    def build_obs_space(self, shape, low, high):
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()

        if (any("depth_seg" in sub for sub in self.info_observation_keys)
            or any("depth_seg" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.camera import add_depth_seg_to_obs
            depth_seg_observations = add_depth_seg_to_obs(observation=ob_dict, robosuite_env=self.env.sim, camera_config=self.camera_config)
            ob_dict.update(depth_seg_observations)

        if (any("reconstructed_heightmap_current" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_current" in sub for sub in self.observation_keys.keys())
            or any("reconstructed_heightmap_diff" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_diff" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.pointcloud import add_reconstructed_heightmap_to_obs
            reconstructed_heightmap_observations = add_reconstructed_heightmap_to_obs(obs=ob_dict, env=self.env, camera_config=self.camera_config, verbose=self.verbose)
            ob_dict.update(reconstructed_heightmap_observations)

        if (any("reconstructed_heightmap_diff" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_diff" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.pointcloud import add_reconstructed_heightmap_diff_to_obs
            reconstructed_heightmap_diff = add_reconstructed_heightmap_diff_to_obs(obs=ob_dict, env=self.env, goal_mask=self.observation_keys["reconstructed_heightmap_diff"].get("goal_mask", False), verbose=self.verbose)
            ob_dict.update(reconstructed_heightmap_diff)

        if (any("eef_pos_prev" in sub for sub in self.info_observation_keys)
            or any("eef_pos_prev" in sub for sub in self.observation_keys.keys())):
            self.eef_pos_previous = np.copy(ob_dict["robot0_eef_pos"])
            ob_dict["eef_pos_prev"] = self.eef_pos_previous

        self.check_dict_for_nan(ob_dict)
        observations = self.filter_obs_dict_by_keys(ob_dict, self.keys)

        info = OrderedDict()
        
        return observations, info

    def step(self, action):
        ob_dict, reward, terminated, info = self.env.step(action)

        if (any("depth_seg" in sub for sub in self.info_observation_keys)
            or any("depth_seg" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.camera import add_depth_seg_to_obs
            depth_seg_observations = add_depth_seg_to_obs(observation=ob_dict, robosuite_env=self.env.sim, camera_config=self.camera_config)
            ob_dict.update(depth_seg_observations)

        if (any("reconstructed_heightmap_current" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_current" in sub for sub in self.observation_keys.keys())
            or any("reconstructed_heightmap_diff" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_diff" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.pointcloud import add_reconstructed_heightmap_to_obs
            reconstructed_heightmap_observations = add_reconstructed_heightmap_to_obs(obs=ob_dict, env=self.env, camera_config=self.camera_config, verbose=self.verbose)
            ob_dict.update(reconstructed_heightmap_observations)

        if (any("reconstructed_heightmap_diff" in sub for sub in self.info_observation_keys)
            or any("reconstructed_heightmap_diff" in sub for sub in self.observation_keys.keys())):
            from sand_gym.utils.pointcloud import add_reconstructed_heightmap_diff_to_obs
            reconstructed_heightmap_diff = add_reconstructed_heightmap_diff_to_obs(obs=ob_dict, env=self.env, goal_mask=self.observation_keys["reconstructed_heightmap_diff"].get("goal_mask", False), verbose=self.verbose)
            ob_dict.update(reconstructed_heightmap_diff)

        if (any("eef_pos_prev" in sub for sub in self.info_observation_keys)
            or any("eef_pos_prev" in sub for sub in self.observation_keys.keys())):
            ob_dict["eef_pos_prev"] = self.eef_pos_previous
            self.eef_pos_previous = np.copy(ob_dict["robot0_eef_pos"])

        self.check_dict_for_nan(ob_dict)
        observations = self.filter_obs_dict_by_keys(ob_dict, self.keys)
        
        return observations, reward, terminated, False, info
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.env.reward()
        return reward