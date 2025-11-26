import numpy as np

import sand_gym.utils.math_function as math_function
import sand_gym.utils.common as common
from sand_gym.utils.plot_window import PlotGraphWindow

class Reward():
    def __init__(self, config, sim):
        self.config = config
        self.sim = sim

        # Normalize rewards
        self.normalize_reward = self.config["normalize_reward"]

        # Initialize Rewards
        self.reward = 0.0
        self.reward_keys_config = self.config["reward_keys"]
        self.rewards = dict.fromkeys(self.reward_keys_config, 0.0)
        # MARK
        if "progressive_heightfield_difference" in self.reward_keys_config:
            self.clip_progressive_heightfield_difference_reward = self.reward_keys_config["progressive_heightfield_difference"].get("only_positive", True)
        self.reset_episode_rewards()

        self.tensorboard_reward = self.reward
        self.tensorboard_rewards = self.rewards.copy()

        self.episode_reward = self.reward
        self.episode_rewards = self.rewards.copy()
        self.cumulated_episode_rewards = []
        self.current_episode = 0
        self.previous_episode = 0

        if self.sim.verbose >= 4:
            labels = list(self.reward_keys_config.keys())
            x_offset = 2
            # y_offset = 100
            plot_size = [[0-x_offset,self.sim.horizon+x_offset]]
            self.plot_reward_window = PlotGraphWindow(name="Reward graph", plot_size=plot_size, axis_labels=["Time (iterations)", "Reward value"], legend_labels=labels)

    def calculate_reward_component(self, reward_key, input_value=None, condition=True):
        if reward_key in self.reward_keys_config and condition:
            reward_component_config = self.reward_keys_config[reward_key]
            type = reward_component_config["type"]
            magnitude = reward_component_config["magnitude"]
            
            if "slope" in reward_component_config:
                slope = reward_component_config["slope"]
            
            if "limits" in reward_component_config:
                limits = reward_component_config["limits"]

            if type != "scalar":
                # Normalize partial reward
                if self.normalize_reward:
                    input_value = common.normalize_value(input_value, limits[0], limits[1])
                # Calculate partial continuous reward depeding on function type
                if type == "linear":
                    partial_reward = math_function.linear(input_value, slope, magnitude)
                elif type == "linear_greater":
                    partial_reward = math_function.linear_greater(input_value, magnitude)
                elif type == "positive_tanh":
                    partial_reward = math_function.positive_tanh(input_value, slope, magnitude)
                elif type == "negative_tanh":
                    partial_reward = math_function.negative_tanh(input_value, slope, magnitude)
            else:
                # Calculate partial sparse reward
                partial_reward = magnitude

            self.rewards[reward_key] = partial_reward

    def compute_reward(self, achieved_goal=None, desired_goal=None):
        # Reset all rewards
        self.reward = 0.0
        for key in self.rewards.keys():
            self.rewards[key] = 0.0
        
        if "progressive_heightfield_difference" in self.reward_keys_config:
            # At the start of each episode
            mean_heightfield_difference = self.sim.sand_simulator.get_current_mean_diff_heightfield(goal_mask=self.reward_keys_config["progressive_heightfield_difference"].get("goal_mask", False), as_np=True, cut_off=self.reward_keys_config["progressive_heightfield_difference"].get("cut_off", False))
        
            progressive_mean_diff = np.asarray(self.min_reached_mean_heightfield_diff - mean_heightfield_difference, dtype=np.float64)
            if "clip_mean_diff" in self.reward_keys_config["progressive_heightfield_difference"]:
                if np.any(progressive_mean_diff != 0.0) and np.any(np.abs(progressive_mean_diff) < float(self.reward_keys_config["progressive_heightfield_difference"]["clip_mean_diff"])):
                    progressive_mean_diff = np.where(np.abs(progressive_mean_diff) < float(self.reward_keys_config["progressive_heightfield_difference"]["clip_mean_diff"]), 0.0, progressive_mean_diff)
            if np.any(progressive_mean_diff < 0.0) and "sign_relation" in self.reward_keys_config["progressive_heightfield_difference"]:
                progressive_mean_diff = np.where(progressive_mean_diff < 0.0,progressive_mean_diff*self.reward_keys_config["progressive_heightfield_difference"]["sign_relation"], progressive_mean_diff)
            if self.clip_progressive_heightfield_difference_reward:
                progressive_mean_diff = np.clip(progressive_mean_diff, a_min=0.0, a_max=None)           
            self.calculate_reward_component("progressive_heightfield_difference", progressive_mean_diff)

            if np.any(mean_heightfield_difference < self.min_reached_mean_heightfield_diff):
                self.min_reached_mean_heightfield_diff = np.asarray(mean_heightfield_difference, dtype=np.float64)
                
            if self.sim.verbose >= 2:
                print("min_reached_mean_heightfield_diff:",self.min_reached_mean_heightfield_diff)

        if "delta_heightfield_difference" in self.reward_keys_config:
            mean_heightfield_difference = self.sim.sand_simulator.get_current_mean_diff_heightfield(goal_mask=self.reward_keys_config["delta_heightfield_difference"].get("goal_mask", False), cut_off=self.reward_keys_config["delta_heightfield_difference"].get("cut_off", False))
            if self.prev_mean_diff is None:
                mean_diff = 0.0
            else:
                mean_diff = self.prev_mean_diff - mean_heightfield_difference
            if "clip_mean_diff" in self.reward_keys_config["delta_heightfield_difference"]:
                if mean_diff != 0.0 and abs(mean_diff) < float(self.reward_keys_config["delta_heightfield_difference"]["clip_mean_diff"]):
                    mean_diff = 0.0
            if mean_diff < 0.0 and "sign_relation" in self.reward_keys_config["delta_heightfield_difference"]:
                mean_diff = mean_diff*self.reward_keys_config["delta_heightfield_difference"]["sign_relation"]
            if self.reward_keys_config["delta_heightfield_difference"]["only_positive"]:
                mean_diff = np.clip(mean_diff, a_min=0.0, a_max=None)
            self.calculate_reward_component("delta_heightfield_difference", mean_diff)
            self.prev_mean_diff = mean_heightfield_difference

        if "gripper_to_goal_area_distance_mask_based" in self.reward_keys_config:
            gripper_to_goal_area_distance_mask_based = self.sim.sand_simulator.get_gripper_to_goal_area_distance_mask_based(
                tool_2d_world_pos=self.sim._eef_xpos[:2],
                shrink_goal_area=self.reward_keys_config["gripper_to_goal_area_distance_mask_based"]["shrink_goal_area"],
                corner_dist=self.reward_keys_config["gripper_to_goal_area_distance_mask_based"]["corner_dist"])

        if "gripper_to_goal_area_distance_mask_based" in self.reward_keys_config:
            self.calculate_reward_component("gripper_to_goal_area_distance_mask_based", gripper_to_goal_area_distance_mask_based)

        if "goal_area_reached_mask_based" in self.reward_keys_config:
            shrink_goal_area = self.reward_keys_config["goal_area_reached_mask_based"]["shrink_goal_area"]
            goal_area_reached_mask_based_condition = self.sim.sand_simulator.tool_in_goal_area(shrink_goal_area=shrink_goal_area)
            if goal_area_reached_mask_based_condition and self.reward_keys_config["goal_area_reached_mask_based"].get("once_per_ep", False):
                goal_area_reached_mask_based_condition = goal_area_reached_mask_based_condition and not self.reached_goal_area_mask_based
                self.reached_goal_area_mask_based = True
                
            self.calculate_reward_component("goal_area_reached_mask_based", condition=goal_area_reached_mask_based_condition)

        if "wrong_cell" in self.reward_keys_config:
            inverted_mean_heightfield_difference = self.sim.sand_simulator.get_current_mean_diff_heightfield(inverted_goal_mask=True, as_np=True, cut_off=self.reward_keys_config["wrong_cell"].get("cut_off", False))
            inverted_mean_diff = self.max_reached_inverted_mean_heightfield_difference - inverted_mean_heightfield_difference    
            if inverted_mean_diff != 0.0 and np.abs(inverted_mean_diff) < float(self.reward_keys_config["wrong_cell"]["clip_mean_diff"]):
                inverted_mean_diff = 0.0
            inverted_mean_diff = np.clip(inverted_mean_diff, a_min=None, a_max=0.0)
            self.calculate_reward_component("wrong_cell", inverted_mean_diff)         
            if np.any(inverted_mean_heightfield_difference > self.max_reached_inverted_mean_heightfield_difference):
                self.max_reached_inverted_mean_heightfield_difference = inverted_mean_heightfield_difference
        
        self.reward = sum(self.rewards.values())

        if self.sim.verbose >= 2:
            print("Reward values:", self.rewards)
            print("Reward:", self.reward)

        # Episode rewards
        self.episode_reward += self.reward
        for key in self.episode_rewards.keys():
            self.episode_rewards[key] += self.rewards[key]
        self.cumulated_episode_rewards.append(self.episode_rewards.copy())
        
        if self.sim.verbose >= 4:
            self.plot_episode_rewards()
        
        return self.reward, self.rewards

    def reset_episode_rewards(self):
        self.episode_reward = 0.0
        self.episode_rewards = dict.fromkeys(self.reward_keys_config, 0.0)
        self.cumulated_episode_rewards = []
        if "progressive_heightfield_difference" in self.reward_keys_config and self.sim.sand_simulator is not None:
            self.min_reached_mean_heightfield_diff = self.sim.sand_simulator.get_initial_mean_diff_heightfield(goal_mask=self.sim.observations_config["goal_mask"], as_np=True)
        if "delta_heightfield_difference" in self.reward_keys_config and self.sim.sand_simulator is not None:
            self.prev_mean_diff = None
        if "goal_area_reached_mask_based" in self.reward_keys_config:
            self.reached_goal_area_mask_based = False
        if "wrong_cell" in self.reward_keys_config:
            self.max_reached_inverted_mean_heightfield_difference = self.sim.sand_simulator.get_current_mean_diff_heightfield(inverted_goal_mask=True, as_np=True)

    def get_episode_reward(self):
        if type(self.episode_reward) == np.ndarray:
            return np.mean(self.episode_reward)
        else:
            return self.episode_reward

    def get_episode_rewards(self):
        episode_rewards = {}
        for key, reward in self.episode_rewards.items():
            if type(reward) == np.ndarray:
                episode_rewards[key] = np.mean(reward)
            else:
                episode_rewards[key] = reward
        return episode_rewards
    
    def get_cumulated_episode_rewards(self):
        return self.cumulated_episode_rewards
    
    def plot_episode_rewards(self):
        iteration = len(self.cumulated_episode_rewards)
        x_data = (np.linspace(0, iteration, iteration, endpoint=False, dtype=np.int64)).tolist()
        y_data = []
        for iter in self.cumulated_episode_rewards:
            y_data.append(list(iter.values()))
        self.plot_reward_window.update(x_data, y_data)