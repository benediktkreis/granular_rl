import cv2
import mujoco as mj
import numpy as np
import random
from scipy import ndimage
from scipy.ndimage import binary_erosion, distance_transform_edt, binary_dilation
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import transforms3d.quaternions as tq
import sand_gym.utils.common as common
from sand_gym.utils.plot_window import PlotSandWindow

# CONST
flow_rate = 0.125
fragment_counter = 0
fragment_force_threshold = 0.005
kernel = np.ones((3, 3))

class SandSimulator:

    def __init__(
            self,
            # Simulation
            sim,
            # Sand box parameters
            field_size, # list [nrow, ncol] or int [nrow,nrow] (if nrow=ncol) of hfield elevation defined in sand_shaping_arena.xml
            sand_range_in_m, # Max hfield elevation defined in sand_shaping_arena.xml
            sand_geom_name,
            sand_box_offset=(0.0,0.0,0.0),
            init_height_in_mm=75,
            angle_of_repose=34,
            height_perturbation_range_in_mm=5,
            # Tool parameters
            tool_body_name="gripper0_gripper_base",
            tool_geom_name="gripper0_gripper_interaction",
            dataset_config=None,
            # Function parameters
            render_callback=None,
            correct_render_config=None,
            step_simulation_callback=None,
            # Debug parameters
            add_visual_element_callback = None,
            goal_randomization = False,
            env = None,
            verbose=0,
            ):

        self.angle_of_repose_deg = angle_of_repose
        self.angle_of_repose_rad = np.deg2rad(self.angle_of_repose_deg)

        # MUJOCO
        self.heightfield = None
        self.sim = sim
        self.env = env
        self.add_visual_element_callback = add_visual_element_callback
        self.model = self.sim.model._model
        self.data = self.sim.data._data
        self.render_callback = render_callback
        self.step_simulation_callback = step_simulation_callback

        # SIMULATION
        self.tool_body_name = tool_body_name
        self.tool_body_id = self.data.body(self.tool_body_name).id
        self.tool_geom_name = tool_geom_name
        self.tool_geom_id = self.data.geom(tool_geom_name).id
        self.sand_geom_name = sand_geom_name
        self.sand_geom_id = self.data.geom(self.sand_geom_name).id
        self.last_intersection_mask = None
        self.sand_box_offset = np.array(sand_box_offset, dtype=np.float32)

        # RENDER
        self.correct_render_config = correct_render_config
        if self.correct_render_config is not None:
            self.correct_render_fill_method = correct_render_config["fill_method"]
            self.correct_render_fill_threshold = correct_render_config["fill_threshold"] 
        else:
            self.correct_render_fill_method = "griddata"
            self.correct_render_fill_threshold = 10.0

        # field_size is grid cells
        if type(field_size) != list:
            self.field_size = np.array([field_size, field_size], dtype=np.int32)
        else:
            self.field_size = np.asarray(field_size, dtype=np.int32)
        self.init_height_in_mm = init_height_in_mm
        self.initialize_heightfield(self.init_height_in_mm)
        self.height_perturbation_range_in_mm = height_perturbation_range_in_mm
        self.sand_range_in_m = sand_range_in_m
        self.sand_range_in_mm = self.convert_m_to_mm(self.sand_range_in_m)
        self.unstable_cells_threshold = 1e-3

        self.sand_box_center = self.get_geom_center_in_sand_coords(self.sand_geom_name)
        self.sand_box_half_extends = self.get_half_extents(self.sand_geom_id)
        self.sand_box_2d_dims_in_mm = (self.sand_box_half_extends[:2]*1000)*2
        self.sand_box_2d_dims_in_m = self.convert_mm_to_m(self.sand_box_2d_dims_in_mm)
        self.sand_box_3d_bb_in_m = np.append(self.sand_box_2d_dims_in_m, self.sand_range_in_m)
        temp_grid_cell_size_in_mm = self.sand_box_2d_dims_in_mm/np.flip(self.field_size)
        if temp_grid_cell_size_in_mm[0] != temp_grid_cell_size_in_mm[1]:
            print("A sand grid cell has to be quadratic (length=width). The current value in mm is ", temp_grid_cell_size_in_mm)
            raise ValueError
        self.grid_cell_size_in_mm = temp_grid_cell_size_in_mm[0]
        self.grid_cell_size_in_m = self.convert_mm_to_m(self.grid_cell_size_in_mm)
        self.half_grid_cell_size_in_m = self.grid_cell_size_in_m/2
        self.array_grid_cell_size_in_mm = np.ones(self.field_size)*self.grid_cell_size_in_mm

        x = np.linspace(-self.sand_box_half_extends[0], self.sand_box_half_extends[0], self.field_size[1])
        y = np.linspace(-self.sand_box_half_extends[1], self.sand_box_half_extends[1], self.field_size[0])
        xv, yv = np.meshgrid(x, y)
        self.sand_xy = np.stack((xv, yv), axis=-1)

        # dataset
        self.dataset_config = dataset_config
        self.goal_field_size = None
        self.observation_field_size = None
        if self.dataset_config is not None:
            self.goal_heightfield_2d_dict = {}
            self.goal_field_size = np.array(self.dataset_config.get("goal_grid_size", self.field_size.tolist()), dtype=np.int32)
            self.observation_field_size = np.array(self.dataset_config.get("observation_grid_size", self.field_size.tolist()), dtype=np.int32)
            if type(self.dataset_config["id"]) == list:
                dataset_selection_list = self.dataset_config.get("dataset_selection", [None]*len(self.dataset_config["id"]))
                for dataset_id, dataset_selection in zip(self.dataset_config["id"], dataset_selection_list):
                    temp_goal_heightfield_2d_dict = common.load_dataset(dataset_id, self.field_size, self.goal_field_size, self.init_height_in_mm, self.angle_of_repose_deg, dataset_selection=dataset_selection)
                    if len(temp_goal_heightfield_2d_dict) == 0:
                        print(f"Could not load dataset: {dataset_id}_{common.get_dataset_name_by_id(dataset_id)}")
                        raise ValueError
                    self.goal_heightfield_2d_dict = self.goal_heightfield_2d_dict | temp_goal_heightfield_2d_dict
            else:
                dataset_selection = self.dataset_config.get("dataset_selection", None)
                self.goal_heightfield_2d_dict = common.load_dataset(self.dataset_config["id"], self.field_size, self.goal_field_size, self.init_height_in_mm, self.angle_of_repose_deg, dataset_selection=dataset_selection)
            self.goal_heightfield_2d_key_list = list(self.goal_heightfield_2d_dict.keys())
            equalize_datasets = self.dataset_config.get("equalize_datasets", False)
            if equalize_datasets:
                self.goal_heightfield_2d_key_list = common.equalize_dataset_key_list(self.goal_heightfield_2d_key_list)
            self.goal_heightfield_key = None
            self.goal_heightfield_2d = None
            self.goal_heightfield_flat = None
            self.sample_strategy = dataset_config.get("sample_strategy", "random")
            self.shift_max = dataset_config.get("shift_max", int((np.min(self.field_size)/2)-4))  # (32/2)-4=12 # 4=goal_area_width/2

        # Evaluation
        self.goal_randomization = goal_randomization

        # Debug
        self.verbose = verbose
        if self.verbose >= 3:
            self.plot_window_diff = PlotSandWindow(name="Sand simulator difference heightfield", field_size_grid=self.field_size, field_size_real=self.sand_box_2d_dims_in_m, grid_cell_size=self.grid_cell_size_in_m)

    def get_goal_heighfield(self, flat=False, convert_to_obs_shape=False):
        goal_heightfield_2d = self.goal_heightfield_2d
        if convert_to_obs_shape:
            goal_heightfield_2d = self.convert_heightfield_to_observation(goal_heightfield_2d)
        if flat:
            return goal_heightfield_2d.flatten()
        else:
            return goal_heightfield_2d
    
    def reset_goal_mask(self):
        self.goal_mask = np.where(self.goal_heightfield_2d == self.init_height_in_mm, 0.0, 1.0)
        
    def update_goal_heighfield(self, key="random"):
        if len(self.goal_heightfield_2d_key_list) == 0:
            goal_heightfield = self.randomize_heightfield(np.copy(self.heightfield),
                                                        10.0,
                                                        self.init_height_in_mm)
            goal_heightfield_2d = np.reshape(np.copy(goal_heightfield), self.field_size)
        else:
            if key=="random" or "random_shift" or self.goal_randomization == False:
                self.goal_heightfield_key = random.choice(self.goal_heightfield_2d_key_list)
            else:
                self.goal_heightfield_key = key
            
            goal_heightfield_2d = self.goal_heightfield_2d_dict[self.goal_heightfield_key]

            if key=="random_shift" or self.goal_randomization:
                
                x_shift = np.random.randint(-self.shift_max,self.shift_max+1)
                y_shift = np.random.randint(-self.shift_max,self.shift_max+1)

                goal_heightfield_2d = np.roll(goal_heightfield_2d, (x_shift, y_shift), axis=(1,0))

        self.goal_heightfield_2d = goal_heightfield_2d
        self.goal_heightfield_flat = goal_heightfield_2d.flatten()
    
    def debug_plots(self, current_zed_image=None,current_zed_depth_image=None, initial_goal_heightmap=None, before=False, after=False):
            # Get tool pos
            tool_2d_world_pos = self.get_lowest_tool_point()[:2]
            tool_2d_grid_pos = self.convert_world_to_grid(tool_2d_world_pos, check_norm=False)

            # Plot difference heightfield
            current_difference_heightfield = self.get_current_difference_heightfield(flat=False, abs=False)

            tool_mask = self.get_tool_mask()
            if "goal_mask" in self.env.observation_keys:
                goal_mask = self.get_goal_mask(shrink_goal_area=self.env.observation_keys["goal_mask"]["shrink_goal_area"])
            else:
                goal_mask = self.get_goal_mask()
            self.plot_window_diff.update([current_difference_heightfield, tool_2d_grid_pos, goal_mask.astype(int), tool_mask.astype(int)])

    def step_sand_simulator(self):
        self.simulate_sand()
        self.step_simulation_callback()
        if self.render_callback is not None:
            self.render_callback()
        if self.verbose >= 3:
            self.debug_plots()
  
    # 2D check (x,y) if tool is outside the sand box
    def tool_in_sand_2d_check(self, tool_2d_pos=None):
        if tool_2d_pos is None:
            tool_2d_pos = self.get_geom_center_in_sand_coords(self.tool_geom_name)
        tool_half_extents = self.get_half_extents(self.tool_geom_id)

        # Check if the tool's half extents in x and y are within the sand_box's half extents
        within_x = bool(tool_half_extents[0] <= self.sand_box_half_extends[0] - abs(tool_2d_pos[0] - self.sand_box_center[0]))
        within_y = bool(tool_half_extents[1] <= self.sand_box_half_extends[1] - abs(tool_2d_pos[1] - self.sand_box_center[1]))

        # Return true if both x and y conditions are satisfied
        return within_x and within_y

    # 3D check (x, y, z) if tool is outside the sand box
    def tool_in_sand_3d_check(self, tool_heights=None, heightfield=None):
        if tool_heights is None:
            tool_heights = self.get_heights_by_polygon(self.tool_geom_name)
        # Calculate the overlap of how much the object overlaps with the sand ( negative cells indicate overlap )
        if heightfield is None:
            heightfield = self.get_grid_heightfield_copy()
        overlap = tool_heights - heightfield
        within = bool(np.any(overlap <= 0.0))
        return within
    
    def tool_in_sand_check(self, tool_2d_pos=None, tool_heights=None, heightfield=None):
        within = False
        within_2d = self.tool_in_sand_2d_check(tool_2d_pos)
        if within_2d:
            within_3d = self.tool_in_sand_3d_check(tool_heights, heightfield)
            if within_3d:
                within = True
        return within

    def check_mask_within_mask(self, mask_1: np.ndarray, mask_2: np.ndarray) -> bool:
        # Ensure both masks have the same shape
        if mask_1.shape != mask_2.shape:
            raise ValueError("Both masks must have the same shape.")

        # Check if all 1.0 values in mask_1 are within the 1.0 values of mask_2
        within = np.all((mask_1 == 1.0) <= (mask_2 == 1.0))
        return bool(within)

    def shrink_mask(self, mask, shrink_factor):
        # Ensure the mask is binary (0s and 1s)
        binary_mask = (mask > 0).astype(np.int32)
        
        # Apply binary erosion to shrink the goal area
        shrunk_mask = binary_erosion(binary_mask, iterations=shrink_factor)
        
        # Convert the mask back to the original type (e.g., float)
        return shrunk_mask.astype(mask.dtype)

    def tool_in_goal_area(self, shrink_goal_area=0):
        within = False
        tool_mask = self.get_tool_mask()
        goal_mask = self.get_goal_mask(shrink_goal_area=shrink_goal_area)
        # Check if tool mask is within sand bed. If not all mask values are 0.0
        if self.tool_in_sand_2d_check():
            within = self.check_mask_within_mask(tool_mask, goal_mask)
        if self.verbose >= 2:
            if within:
                print("Tool in goal area")
        return within

    def get_tool_bounding_box(self, reference="grid", tool_center_world_pos=None, tool_center_rotation_matrix=None):
        if tool_center_world_pos is None:
            if reference == "grid":
                tool_center_world_pos = self.get_geom_center_in_sand_coords(self.tool_geom_name)
            elif reference == "world":
                tool_center_world_pos = self.get_geom_center_in_world_coords(self.tool_geom_name)
        min_aabb, max_aabb = self.generate_aabb(tool_center_world_pos, self.tool_body_id, self.tool_geom_id, rotation_matrix=tool_center_rotation_matrix)
        tool_bb = np.array([min_aabb, max_aabb], dtype=np.float32)
        return tool_bb
        
    def get_tool_base_rectangle(self, reference="grid", tool_center_world_pos=None, tool_center_rotation_matrix=None):
        if tool_center_world_pos is None:
            tool_center_world_pos = self.get_geom_center_in_sand_coords(self.tool_geom_name)
        min_aabb, max_aabb = self.generate_aabb(tool_center_world_pos, self.tool_body_id, self.tool_geom_id, rotation_matrix=tool_center_rotation_matrix)
        if reference == "world":
            return min_aabb[:2], max_aabb[:2]
        elif reference == "grid":
            min_grid = self.convert_world_to_grid(np.array(min_aabb[:2]), check_norm=False).astype(int)
            max_grid = self.convert_world_to_grid(np.array(max_aabb[:2]), check_norm=False).astype(int)
            return min_grid, max_grid

    def get_lowest_tool_point(self):
        tool_center = self.get_geom_center_in_sand_coords(self.tool_geom_name)
        half_extent_tool_z = self.get_half_extents(self.tool_geom_id)[2]
        tool_min = np.copy(tool_center)
        tool_min[2] = tool_center[2] - half_extent_tool_z
        return tool_min
    
    def get_sand_bottom_in_world_coords(self):
        sand_geom_bottom = self.get_geom_center_in_world_coords(self.sand_geom_name)
        return sand_geom_bottom
    
    def get_sand_bounding_box(self):
        sand_geom_bottom = self.get_sand_bottom_in_world_coords()
        sand_geom_center = sand_geom_bottom + np.array([0.0, 0.0, self.sand_range_in_m/2], dtype=np.float32)
        sand_bounding_box = np.array([sand_geom_center-(self.sand_box_3d_bb_in_m/2), sand_geom_center+(self.sand_box_3d_bb_in_m/2)], dtype=np.float32)

        return sand_bounding_box

    def get_z_dist_tool_to_sand_bottom(self):
        tool_z_min = self.get_lowest_tool_point()[2]
        sand_z_min = self.get_sand_bottom_in_world_coords()[2]
        # Calculate the minimum Z distance between geom1 and geom2
        z_distance = tool_z_min - sand_z_min
        return z_distance
    
    def get_half_extents(self, geom_id):
        # Get the correctly sorted half extends for any geom
        half_extents_aabb = self.model.geom_aabb[geom_id][3:]
        is_mesh = self.model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_MESH
        # Meshes are y-axis up, but mujoco expects z-axis up
        if is_mesh:
            half_extents_aabb = np.array([half_extents_aabb[2], half_extents_aabb[1], half_extents_aabb[0]], dtype=np.float32)
        return half_extents_aabb

    def reset(self):
        # Reset the heightfield and the object

        self.initialize_heightfield(self.init_height_in_mm)
        self.update_goal_heighfield(key=self.sample_strategy)
        self.reset_goal_mask()

        if self.verbose >= 1:
            print("Goal heightmap:", self.goal_heightfield_key)

        self.step_simulation_callback()
        if self.render_callback is not None:
            self.render_callback()

    def get_gripper_to_goal_area_distance_mask_based(self, tool_2d_world_pos, dist_3d=False, shrink_goal_area=1, corner_dist=True):
        goal_mask_shrinked = self.get_goal_mask(shrink_goal_area=shrink_goal_area)

        sand_diff_xy = self.sand_xy - tool_2d_world_pos
        sand_diff_norm = np.linalg.norm(sand_diff_xy, axis=-1) 

        add = (goal_mask_shrinked == 0.0).astype(float) * 100000.0
        sand_diff_norm += add

        min_index = np.argmin(sand_diff_norm)
        min_cell_position = self.sand_xy.reshape(-1,2)[min_index]

        if corner_dist:
            corners = np.array([min_cell_position + np.array([self.half_grid_cell_size_in_m, self.half_grid_cell_size_in_m]),
                                min_cell_position + np.array([-self.half_grid_cell_size_in_m, self.half_grid_cell_size_in_m]),
                                min_cell_position + np.array([self.half_grid_cell_size_in_m, -self.half_grid_cell_size_in_m]),
                                min_cell_position + np.array([-self.half_grid_cell_size_in_m, -self.half_grid_cell_size_in_m])])
            corners_diff_xy = corners - np.repeat(np.expand_dims(tool_2d_world_pos, axis=0), 4, axis=0)
            corners_diff_norm = np.linalg.norm(corners_diff_xy, axis=-1)  # 4
            corners_min_diff_index = np.argmin(corners_diff_norm, axis=0)
            
            dist_3d_value = corners_diff_xy[corners_min_diff_index]
            dist = np.min(corners_diff_norm, axis=0)
        
        else:
            dist_3d_value = min_cell_position - tool_2d_world_pos
            dist = np.linalg.norm(dist_3d_value)

        goal_mask = self.get_goal_mask()
        tool_mask = self.get_tool_mask()
        within = self.check_mask_within_mask(tool_mask, goal_mask)
        if within:
            dist = 0.0
            dist_3d_value = np.zeros(2,dtype=np.float32)

        if self.verbose >= 2:
            print("gripper_to_goal_area_distance_mask_based:",dist)
            print("gripper_to_goal_area_distance_mask_based_3d:",dist_3d_value)
        return dist_3d_value if dist_3d else dist

    def get_episode_height_diff(self, goal_mask=False):
        initial_mean = self.get_initial_mean_diff_heightfield(goal_mask=goal_mask)
        current_mean = self.get_current_mean_diff_heightfield(goal_mask=goal_mask)
        episode_height_diff = 1-(current_mean/initial_mean)
        return episode_height_diff
    
    def get_episode_manipulated_cells(self, inverted=False, criterion="below_init"):
        mask = self.get_goal_mask().astype(bool)
        # If inverted, counts outside goal_area
        if inverted:
            mask = ~mask
        cell_count_in_mask = np.sum(mask)
        current_heightfield = self.get_current_heightfield()
        if criterion == "below_init":
            cell_count_manipulated = np.sum(mask & (current_heightfield < self.init_height_in_mm))
        elif criterion == "unequal_init":
            cell_count_manipulated = np.sum(mask & (current_heightfield != self.init_height_in_mm))
        manipulated_cells_in_mask = cell_count_manipulated/cell_count_in_mask
        return manipulated_cells_in_mask
    
    def get_initial_mean_diff_heightfield(self, goal_mask=False, tool_mask=False, as_np=False, where="all"):
        initial_difference_heightmap = self.get_initial_difference_heightfield(abs=True, goal_mask=goal_mask, tool_mask=tool_mask)
        if where == "all":
            mean_initial_height_difference = np.mean(initial_difference_heightmap)
        elif where == "in_goal":
            goal_mask = self.get_goal_mask().astype(bool)
            mean_initial_height_difference = np.mean(initial_difference_heightmap, where=goal_mask)
        elif where == "out_goal":
            goal_mask = self.get_goal_mask().astype(bool)
            inverted_goal_mask = ~goal_mask
            mean_initial_height_difference = np.mean(initial_difference_heightmap, where=inverted_goal_mask)
        if as_np:
            return np.asarray(mean_initial_height_difference, dtype=np.float32)
        else:
            return mean_initial_height_difference
    
    def get_current_mean_diff_heightfield(self, goal_mask=False, tool_mask=False, as_np=False, inverted_goal_mask=False, cut_off=False, where="all"):
        current_difference_heightmap = self.get_current_difference_heightfield(abs=True, goal_mask=goal_mask, tool_mask=tool_mask, inverted_goal_mask=inverted_goal_mask, cut_off=cut_off)
        if where == "all":
            mean_current_height_difference = np.mean(current_difference_heightmap)
        elif where == "in_goal":
            goal_mask = self.get_goal_mask().astype(bool)
            mean_current_height_difference = np.mean(current_difference_heightmap, where=goal_mask)
        elif where == "out_goal":
            goal_mask = self.get_goal_mask().astype(bool)
            inverted_goal_mask = ~goal_mask
            mean_current_height_difference = np.mean(current_difference_heightmap, where=inverted_goal_mask)
        if self.verbose >= 2:
            print("mean_current_height_difference:",mean_current_height_difference)
        if as_np:
            return np.asarray(mean_current_height_difference, dtype=np.float32)
        else:
            return mean_current_height_difference

    def get_initial_difference_heightfield(self, flat=False, goal_mask=False, tool_mask=False, abs=False):
        goal_heightfield = self.get_goal_heighfield(flat=flat)
        initial_heightfield = self.get_initial_heightfield(flat=flat, goal_mask=goal_mask)
        difference_heightfield = goal_heightfield - initial_heightfield
        if abs:
            difference_heightfield = np.abs(difference_heightfield)
        return difference_heightfield

    def get_current_difference_heightfield(self, flat=False, abs=False, goal_mask=False, tool_mask=False, shrink_goal_area=0, inverted_goal_mask=False, convert_to_obs_shape=False, cut_off=False):
        goal_heightfield = self.get_goal_heighfield(flat=flat, convert_to_obs_shape=convert_to_obs_shape)
        current_heightfield = self.get_current_heightfield(flat=flat, goal_mask=goal_mask, tool_mask=tool_mask, shrink_goal_area=shrink_goal_area, inverted_goal_mask=inverted_goal_mask, convert_to_obs_shape=convert_to_obs_shape)
        if cut_off:
            current_heightfield = np.where(current_heightfield > self.init_height_in_mm, self.init_height_in_mm, current_heightfield)
        difference_heightmap = goal_heightfield - current_heightfield
        if abs:
            difference_heightmap = np.abs(difference_heightmap)
        return difference_heightmap
    
    def convert_heightfield_to_observation(self, input_array, pad_value=None):
        if pad_value is None:
            pad_value = self.init_height_in_mm
        arr = input_array.copy()
        output_array = common.get_array_center(arr, self.goal_field_size)
        output_array = common.pad_array(output_array, self.observation_field_size, pad_value)
        return output_array

    def convert_mask_to_observation(self, input_array, pad_value=0.0):
        arr = input_array.copy()
        output_array = common.get_array_center(arr, self.goal_field_size)
        output_array = common.pad_array(output_array, self.observation_field_size, pad_value)
        return output_array
    
    def get_initial_heightfield(self, flat=False, goal_mask=False):
        heightfield_2d = self.get_initial_heightfield_copy()
        if goal_mask:
            heightfield = np.where(self.goal_heightfield_2d == self.init_height_in_mm, self.goal_heightfield_2d, heightfield_2d)
        else:
            heightfield = heightfield_2d

        if flat:
            return heightfield.flatten()
        else:
            return heightfield
    
    def get_tool_mask(self, flat=False, convert_to_obs_shape=False, binary=False, tool_center_world_pos=None, tool_center_rotation_matrix=None):
        tool_min, tool_max = self.get_tool_base_rectangle(reference="grid", tool_center_world_pos=tool_center_world_pos, tool_center_rotation_matrix=tool_center_rotation_matrix)
        tool_mask = np.full(self.field_size, 0.0)
        cv2.rectangle(tool_mask, (tool_min[0], tool_min[1]), (tool_max[0], tool_max[1]), (1, 0, 0), -1)
        if convert_to_obs_shape:
            tool_mask = self.convert_mask_to_observation(tool_mask)
        if binary:
            tool_mask = tool_mask.astype(bool)
        if flat:
            return tool_mask.flatten()
        else:
            return tool_mask
    
    def get_goal_mask(self, shrink_goal_area=0, flat=False, convert_to_obs_shape=False):
        goal_mask = np.copy(self.goal_mask)
        if shrink_goal_area>0:
            goal_mask = self.shrink_mask(goal_mask, shrink_goal_area)
        if convert_to_obs_shape:
            goal_mask = self.convert_mask_to_observation(goal_mask)
        if flat:
            return goal_mask.flatten()
        else:
            return goal_mask

    def get_current_heightfield(self, flat=False, goal_mask=False, tool_mask=False, shrink_goal_area=0, inverted_goal_mask=False, convert_to_obs_shape=False):
        heightfield_2d = self.get_grid_heightfield_copy()
        if goal_mask:
            mask_of_goal = self.get_goal_mask(shrink_goal_area=shrink_goal_area)
            heightfield = np.where(mask_of_goal == 1.0, heightfield_2d, self.goal_heightfield_2d)
        elif inverted_goal_mask:
            mask_of_goal = self.get_goal_mask(shrink_goal_area=shrink_goal_area)
            heightfield = np.where(mask_of_goal == 0.0, heightfield_2d, self.goal_heightfield_2d)
        else:
            heightfield = heightfield_2d
        
        if tool_mask:
            mask_of_tool = self.get_tool_mask()
            heightfield = np.where(mask_of_tool == 1.0, heightfield, self.goal_heightfield_2d)
        
        if convert_to_obs_shape:
            heightfield = self.convert_heightfield_to_observation(heightfield)
        
        if flat:
            return heightfield.flatten()
        else:
            return heightfield

    def fill_tool_area_mean(self, hf, tool_mask):
        tool_bool = tool_mask.astype(bool)

        # Dilate the tool_mask: this will include the tool and its immediate neighbors.
        struct = np.ones((3, 3), dtype=bool)
        dilated_mask = binary_dilation(tool_bool, structure=struct)
        
        border_mask = dilated_mask & ~tool_bool
        border_values = hf[border_mask]
        
        if border_values.size > 0:
            fill_value = border_values.mean()
        else:
            fill_value = 0
        
        # Clip to zero if any new point in the tool mask is below 0
        if fill_value < 0.0:
            fill_value = 0.0

        hf_filled = hf.copy()
        hf_filled[tool_bool] = fill_value
        
        return hf_filled

    def fill_tool_area_griddata(self, hf, tool_mask, method='cubic'):
        tool_bool = tool_mask.astype(bool)
        
        # Create a coordinate grid for the heightfield.
        y_indices, x_indices = np.indices(hf.shape)
        
        # The known points are all the pixels not in the tool region.
        known_points = np.column_stack((x_indices[~tool_bool], y_indices[~tool_bool]))
        known_values = hf[~tool_bool]
        
        # The points that need to be interpolated (inside the tool mask).
        missing_points = np.column_stack((x_indices[tool_bool], y_indices[tool_bool]))
        interpolated_values = griddata(known_points, known_values, missing_points, method=method)

        # Clip to zero if any new point in the tool mask is below 0
        if np.any(interpolated_values < 0.0):
             interpolated_values = np.clip(interpolated_values, a_min=0.0, a_max=None)
        
        hf_filled = hf.copy()
        hf_filled[tool_bool] = interpolated_values
        
        return hf_filled
    
    def fill_tool_area(self, hf, tool_mask, fill_method="griddata"):
        if fill_method == "mean":
            hf_filled = self.fill_tool_area_mean(hf, tool_mask)
        elif fill_method == "griddata":
            hf_filled = self.fill_tool_area_griddata(hf, tool_mask)

        return hf_filled
    
    def max_edge_difference(self, hf, tool_mask):
        # Compute the largest absolute difference in height between a cell on the edge
        # of the tool mask and an adjacent cell outside the tool mask
        tool_bool = tool_mask.astype(bool)
        
        # Compute the inner border of the tool mask
        eroded = binary_erosion(tool_bool, structure=np.ones((3, 3)))
        inner_edge = tool_bool & ~eroded

        max_diff = 0.0
        h, w = hf.shape

        # Define the 8-connected neighbor offsets.
        neighbor_offsets = [(-1, -1), (-1, 0), (-1, 1),
                            ( 0, -1),          ( 0, 1),
                            ( 1, -1), ( 1, 0), ( 1, 1)]
        
        # Loop over the edge cells.
        edge_indices = np.argwhere(inner_edge)
        for i, j in edge_indices:
            for di, dj in neighbor_offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w:
                    if not tool_bool[ni, nj]:
                        diff = abs(hf[i, j] - hf[ni, nj])
                        if diff > max_diff:
                            max_diff = diff
        return max_diff

    def get_normed_heightfield(self, flat=False, goal_mask=False, correct_render=False):
        heightfield_in_mm = self.get_current_heightfield(goal_mask=goal_mask)
        if correct_render and self.tool_in_sand_check():
            hf = np.copy(heightfield_in_mm)
            tool_mask = self.get_tool_mask().astype(bool)
            max_tool_edge_difference = self.max_edge_difference(hf, tool_mask)
            # Only use the fill method if the differnce between the "tool hole"
            # and the adjacent cells is greater than 1cm 
            if max_tool_edge_difference > self.correct_render_fill_threshold:
                heightfield_in_mm = self.fill_tool_area(hf, tool_mask, self.correct_render_fill_method)

        heightfield_in_m = self.convert_mm_to_m(heightfield_in_mm)
        normed_heightfield = common.normalize_value(heightfield_in_m, 0.0, self.sand_range_in_m, key="sand_simulator: current_heightfield")
        if flat:
            return normed_heightfield.flatten()
        else:
            return normed_heightfield
    
    def convert_m_to_mm(self, value):
        return value * 1000

    def convert_mm_to_m(self, value):
        return value / 1000

    def get_raw_heightfield(self):
        # Get the unscaled heightfield, the scale is mm
        return self.heightfield

    def displace_sand_for_tool(self, heightfield_copy):
        # Displace the sand for the 'tool' object
        # Returns the intersection mask (Where the tool touches the sand) and the new heightfield
        object_heights = self.get_heights_by_polygon(self.tool_geom_name)
        i = 0

        while True:
            i += 1

            # Calculate the overlap of how much the object overlaps with the sand ( negative cells indicate overlap )
            overlap = object_heights - heightfield_copy

            # Mask of the cells in which the object is intersecting with the sand
            intersection_mask = overlap <= 0.0

            dilated_intersection_mask = ndimage.binary_dilation(intersection_mask, structure=kernel, iterations=1)

            dilated_last_intersection_mask = np.full_like(dilated_intersection_mask, False)
            if self.last_intersection_mask is not None:
                dilated_last_intersection_mask = ndimage.binary_dilation(self.last_intersection_mask, structure=kernel,
                                                                         iterations=1)

            # Edges are always "occupied" so the sand doesnt flow out of the box
            bounded_intersection_mask = self.set_boundary_occupancy_mask(intersection_mask)

            # To account for movement of the sand, take the difference between the last occupancy mask and the newest one
            # (e.g. if the tool is moving from left to right, the sand should be displaced towards the right only)
            last_movement_mask = np.invert(np.logical_and(dilated_last_intersection_mask,
                                                          np.logical_not(dilated_intersection_mask)))

            # The possible cells where sand can be displaced into.
            # dilated_too_occupancy_mask && last_movement_mask && not intersection mask
            # Sand can be placed into cells that are 1. around/in the intersection of the tool and sand 2. have not been
            # covered by the tool in the last step ( indicating movement in a direction ) 3. are not in the
            # intersection mask i.e. the tool doesn't block them

            dilated_movement_mask = np.invert(ndimage.binary_dilation(np.invert(last_movement_mask), structure=kernel,
                                                                      iterations=1))

            displacement_mask = np.logical_and(np.logical_and(dilated_intersection_mask, dilated_movement_mask),
                                               np.invert(bounded_intersection_mask))

            # Positive values should be clipped, we are only interested in cells that need displacement
            sand_to_displace = np.clip(overlap, -np.infty, 0.)

            # Remove the nan here to make calculations easier
            sand_to_displace = np.nan_to_num(sand_to_displace, nan=0)

            if np.min(sand_to_displace) >= 0.0 or i > 10:
                break

            # Remove the sand from the heightfield that needs to be displaced
            reduced_heightfield = heightfield_copy + sand_to_displace

            # How much sand has been displaced ?
            displaced_amount = np.sum(heightfield_copy - reduced_heightfield)
            if np.sum(displacement_mask) > 0:
                displacement_increment = displacement_mask * (displaced_amount / np.sum(displacement_mask))
                # Only displace sand if the change is at least 1mm
                if np.max(displacement_increment) < 1.0:
                    break
                heightfield_copy = reduced_heightfield + displacement_increment
            else:
                # raise Exception('No cells found to displace sand to')
                if self.verbose > 1:
                    print('No cells found to displace sand to')
            
        self.last_intersection_mask = intersection_mask
        return intersection_mask, heightfield_copy

    def get_heights_by_polygon(self, geom_name):
        # Prerequisite is, that the bottom face of the geom is exactly parallel to the sand and also a rectangle

        tool_pos = self.get_geom_center_in_sand_coords(geom_name)
        geom_id = self.data.geom(geom_name).id
        half_extents_sand = self.get_half_extents(self.sand_geom_id)
        min_aabb, max_aabb = self.generate_aabb(tool_pos, self.tool_body_id, geom_id)
        min_aabb, max_aabb = self.find_intersection_rectangle(((min_aabb[0], min_aabb[1]), (max_aabb[0], max_aabb[1])),
                                                              ((-half_extents_sand[0], -half_extents_sand[1]),
                                                               (half_extents_sand[0], half_extents_sand[1])))
        
        min_grid = self.convert_world_to_grid(np.array(min_aabb)).astype(int)
        max_grid = self.convert_world_to_grid(np.array(max_aabb)).astype(int)

        grid = np.full(self.field_size, np.nan)
        cv2.rectangle(grid, (min_grid[0], min_grid[1]), (max_grid[0], max_grid[1]), (1, 0, 0), -1)
        tool_height = self.get_z_dist_tool_to_sand_bottom()
        grid = grid * tool_height * 1000
        return grid

    def generate_aabb(self, center_aabb, body_index_to_use, geom_id, rotation_matrix=None):
        # Generate the correctly oriented bounding box for this geom
        if rotation_matrix is None:
            rotation_matrix = tq.quat2mat(self.data.xquat[body_index_to_use])
        half_extents = self.get_half_extents(geom_id)
        min_point = center_aabb - np.abs(rotation_matrix @ half_extents)
        max_point = center_aabb + np.abs(rotation_matrix @ half_extents)
        return min_point, max_point

    def get_geom_center_in_sand_coords(self, geom_name):
        geom_position = self.get_geom_center_in_world_coords(geom_name) + self.sand_box_offset
        return np.array(geom_position, dtype=np.float32)

    def get_geom_center_in_world_coords(self, geom_name):
        geom_position = self.data.geom(geom_name).xpos
        return np.array(geom_position, dtype=np.float32)

    def find_intersection_rectangle(self, r1, r2):
        # Helper to find the intersection between 2 rectangles
        (x1_1, y1_1), (x2_1, y2_1) = r1
        (x1_2, y1_2), (x2_2, y2_2) = r2

        x1 = max(min(x1_1, x2_1), min(x1_2, x2_2))
        y1 = max(min(y1_1, y2_1), min(y1_2, y2_2))
        x2 = min(max(x1_1, x2_1), max(x1_2, x2_2))
        y2 = min(max(y1_1, y2_1), max(y1_2, y2_2))

        if x1 < x2 and y1 < y2:
            return [x1, y1], [x2, y2]
        else:
            return None, None
    
    def convert_world_to_grid(self, coordinates_world, check_norm=True):
        sand_diff_xy = self.sand_xy - coordinates_world
        sand_diff_norm = np.linalg.norm(sand_diff_xy, axis=-1) 
        min_index = np.argmin(sand_diff_norm)
        row, col = np.unravel_index(min_index, sand_diff_norm.shape)
        return np.array((col, row), dtype=np.int32)

    def convert_grid_to_world(self, coordinates_grid, check_norm=True):
        # Helper to convert grid coordinates to world coordinate
        col, row = coordinates_grid
        coordinate_world = self.sand_xy[row, col]
        return np.asarray(coordinate_world, dtype=np.float32)

    def get_random_2d_gripper_pos_within_goal(self, reference="world", shrink_goal_area=1):
        temp_goal_mask = self.get_goal_mask(shrink_goal_area=shrink_goal_area) 
        true_positions = np.argwhere(temp_goal_mask == 1.0)  
        idx = np.random.choice(len(true_positions))
        row, col = true_positions[idx]
        sand_coords = [col, row]
        if reference == "grid":
            return sand_coords
        if reference == "world":
            world_coords = self.convert_grid_to_world(sand_coords)
            return world_coords
        
    def initialize_heightfield(self, init_height, randomize_height=False):
        # Initialize the heightfield data
        self.heightfield = np.ones(self.field_size).flatten() * init_height
        if randomize_height:
            self.heightfield = self.randomize_heightfield(self.heightfield,
                                                        self.height_perturbation_range_in_mm,
                                                        init_height)

        heightfield_copy = np.reshape(np.copy(self.heightfield), self.field_size)

        self.heightfield = heightfield_copy.flatten()
        self.initial_heightfield = np.copy(self.heightfield)

    def randomize_heightfield(self, heightfield, perturbation_range, init_height):
        # Extract the number of rows and columns
        nrows, ncols = self.field_size

        # Iterate over 2x2 blocks
        for x in range(nrows // 2):  # Each x corresponds to two rows
            for y in range(ncols // 2):  # Each y corresponds to two columns
                # Generate a random height for this block
                height = random.uniform(init_height, init_height + perturbation_range)
                
                # Compute indices for the 2x2 block in the flattened array
                top_left = 2 * y + (2 * x) * ncols
                top_right = (2 * y + 1) + (2 * x) * ncols
                bottom_left = 2 * y + (2 * x + 1) * ncols
                bottom_right = (2 * y + 1) + (2 * x + 1) * ncols

                # Set all four cells in the block to the same random height
                heightfield[top_left] = height
                heightfield[top_right] = height
                heightfield[bottom_left] = height
                heightfield[bottom_right] = height

        return heightfield

    def get_initial_heightfield_copy(self):
        # Return a correctly shaped copy of the heightfield
        return np.reshape(np.copy(self.initial_heightfield), self.field_size)
    
    def get_grid_heightfield_copy(self):
        # Return a correctly shaped copy of the heightfield
        return np.reshape(np.copy(self.heightfield), self.field_size)

    def simulate_sand(self):
        global flow_rate

        flow_rate = 0.125
        # Assume the heightfield is unstable
        is_unstable = True
        heightfield_grid = self.get_grid_heightfield_copy()

        assert self.tool_body_id >= 0, f'No tool body index'
        assert self.tool_geom_name is not None and self.tool_geom_name != "", f'No tool geom name'

        iteration_idx = 0
        while is_unstable:
            if self.tool_in_sand_2d_check():
                # Displace the sand where the tool is (The 'tool' body could also be a mesh for a fragment)
                occupancy_mask, heightfield_grid = self.displace_sand_for_tool(
                    heightfield_grid)
            else:
                occupancy_mask = np.full(self.field_size, False, dtype="bool")

            iteration_idx += 1
            delta_h = np.array(self.get_unstable_cells(heightfield_grid, occupancy_mask)).astype(float)

            # If there are no cells that need adjusting, the field is stable
            abs_delta_h_sum = np.sum(np.array(abs(delta_h)))
            is_unstable = abs_delta_h_sum >= self.unstable_cells_threshold

            # Update the heightfield with the calculated delta_h
            heightfield_grid = heightfield_grid + delta_h

        # Only forward the simulation result if all values are valid
        # Setting negative values would result in a change of sand volume,
        # that is why it is not feasible
        heightfield_grid_flat = heightfield_grid.flatten()
        if np.all((heightfield_grid_flat >= 0.0) & (heightfield_grid_flat <= self.sand_range_in_mm)):
            self.heightfield = heightfield_grid_flat

    def set_boundary_occupancy_mask(self, occupancy_mask):
        occupancy_mask_copy = np.copy(occupancy_mask)
        occupancy_mask_copy[0, :] = 1
        occupancy_mask_copy[:, 0] = 1
        occupancy_mask_copy[self.field_size[0] - 1, :] = 1
        occupancy_mask_copy[:, self.field_size[1] - 1] = 1
        return occupancy_mask_copy

    def np_divide(self, a, b):
        #a/b
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0.0)

    def get_unstable_cells(self, heightfield_copy, occupancy_mask):
        # Returns the delta_h of each cell
        bounded_occupancy_mask = self.set_boundary_occupancy_mask(occupancy_mask)

        # Roll the array so each cell moves 1 step in the indicated direction. This is used to later calculated the#
        # differences between the cells in each direction
        north = np.roll(heightfield_copy, -1, axis=0)
        northeast = np.roll(north, 1, axis=1)
        northwest = np.roll(north, -1, axis=1)

        south = np.roll(heightfield_copy, 1, axis=0)
        southeast = np.roll(south, 1, axis=1)
        southwest = np.roll(south, -1, axis=1)

        west = np.roll(heightfield_copy, -1, axis=1)
        east = np.roll(heightfield_copy, 1, axis=1)

        # Calculate the occupancy for each direction, i.e. if a cell is occupied by the tool
        occ_north = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, -1, axis=0), bounded_occupancy_mask))
        occ_north_east = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, -1, axis=0), 1, axis=1), bounded_occupancy_mask))
        occ_north_west = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, -1, axis=0), -1, axis=1), bounded_occupancy_mask))

        occ_south = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, 1, axis=0), bounded_occupancy_mask))
        occ_south_east = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, 1, axis=0), 1, axis=1), bounded_occupancy_mask))
        occ_south_west = np.invert(
            np.logical_or(np.roll(np.roll(bounded_occupancy_mask, 1, axis=0), -1, axis=1), bounded_occupancy_mask))

        occ_west = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, -1, axis=1), bounded_occupancy_mask))
        occ_east = np.invert(np.logical_or(np.roll(bounded_occupancy_mask, 1, axis=1), bounded_occupancy_mask))

        # Compute the differences between the cell and its neighbors and multiply with the occupancy masks so cell
        # differences to occupied cells are not taken into account
        diff_down = (heightfield_copy - north) * occ_north
        diff_down_right = (heightfield_copy - northwest) * occ_north_west
        diff_down_left = (heightfield_copy - northeast) * occ_north_east

        diff_up = (heightfield_copy - south) * occ_south
        diff_up_right = (heightfield_copy - southwest) * occ_south_west
        diff_up_left = (heightfield_copy - southeast) * occ_south_east

        diff_right = (heightfield_copy - west) * occ_west
        diff_left = (heightfield_copy - east) * occ_east

        # Calculate the slope from the difference between cells
        # !! This only works because the cells are assumed to be square !!
        # This comes out as radians
        down_value = self.np_divide(diff_down, self.array_grid_cell_size_in_mm)
        slope_down = -np.arctan(down_value)
        down_right_value = self.np_divide(diff_down_right, self.array_grid_cell_size_in_mm)
        slope_down_right = -np.arctan(down_right_value)
        down_left_value = self.np_divide(diff_down_left, self.array_grid_cell_size_in_mm)
        slope_down_left = -np.arctan(down_left_value)

        up_value = self.np_divide(diff_up, self.array_grid_cell_size_in_mm)
        slope_up = -np.arctan(up_value)
        up_right_value = self.np_divide(diff_up_right, self.array_grid_cell_size_in_mm)
        slope_up_right = -np.arctan(up_right_value)
        up_left_value = self.np_divide(diff_up_left, self.array_grid_cell_size_in_mm)
        slope_up_left = -np.arctan(up_left_value)

        right_value = self.np_divide(diff_right, self.array_grid_cell_size_in_mm)
        slope_right = -np.arctan(right_value)
        left_value = self.np_divide(diff_left, self.array_grid_cell_size_in_mm)
        slope_left = -np.arctan(left_value)

        # Calculate the q-value as indicated in the paper
        q_down = self.get_q(slope_down)
        q_down_right = self.get_q(slope_down_right)
        q_down_left = self.get_q(slope_down_left)

        q_up = self.get_q(slope_up)
        q_up_right = self.get_q(slope_up_right)
        q_up_left = self.get_q(slope_up_left)

        q_right = self.get_q(slope_right)
        q_left = self.get_q(slope_left)

        # Calculate as is done in the paper
        delta_h = q_down + q_down_right + q_down_left + q_up + q_up_right + q_up_left + q_right + q_left
        delta_h /= 8  # A in the paper
        
        # Convert to degrees
        delta_h = np.rad2deg(delta_h)
        return delta_h

    def get_q(self, input_array):
        q = np.where(input_array < -self.angle_of_repose_rad, (input_array + self.angle_of_repose_rad) * flow_rate, 0)
        q += np.where(input_array > self.angle_of_repose_rad, (input_array - self.angle_of_repose_rad) * flow_rate, 0)

        # Round this to the 15th decimal place, because otherwise it results in weird floating point errors.
        return np.round(q, 15)
