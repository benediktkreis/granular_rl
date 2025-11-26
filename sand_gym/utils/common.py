import numpy as np
import os
import json
from pathlib import Path
import shutil
import glob
import yaml
import torch
import random
import re

from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING, FixedBaseRobot
from robosuite.environments.base import register_env

from sand_gym.models.grippers import SandShapingGripperQuadratic
from sand_gym.environments import SandShaping, CoveragePathPlanner

from decimal import Decimal, ROUND_HALF_UP
from typing import List, Optional
from collections import OrderedDict

from open3d.io import read_triangle_mesh

from scipy.stats import mannwhitneyu, ttest_ind, shapiro, kstest

def insert_after(odict, ref_key, new_items):
    out = []
    for k, v in odict.items():
        out.append((k, v))
        if k == ref_key:
            out.extend(new_items.items())
    odict.clear()
    odict.update(out)

def deep_dict_update(dest, src):
    """
    Recursively updates dest with keys and values from src.
    
    For each key in src:
      - If the key is in dest and both dest[key] and src[key] are dictionaries,
        then update dest[key] recursively.
      - Otherwise, set dest[key] to src[key] (adding it if it wasn’t there).
    
    Returns the updated dest dictionary.
    """
    for key, value in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            deep_dict_update(dest[key], value)
        else:
            dest[key] = value
    return dest

def get_integer_in_string(input_string):
    match = re.search(r'-?\d+', input_string)  # This pattern matches an optional '-' followed by digits
    if match:
        number = int(match.group())
        return number
    else:
        print("No integer found.")
        return None

def register_gripper(gripper_class):
    """
    Register @gripper_class in GRIPPER_MAPPING.

    Args:
        gripper_class (GripperModel): Gripper class which should be registered
    """
    GRIPPER_MAPPING[gripper_class.__name__] = gripper_class

def register_robot(robot_class):
    """
    Register @robot_class in ROBOT_CLASS_MAPPING.

    Args:
        robot_class (RobotModel): Robot class which should be registered
    """
    REGISTERED_ROBOTS[robot_class.__name__] = robot_class
    ROBOT_CLASS_MAPPING[robot_class.__name__] = FixedBaseRobot

def get_number_of_elements_in_obs(obs):
    """
    Counts the number of elements in an environment's observation space

    Args:
        obs (dict): Observation space

    """
    num_el = 0
    for key in obs:
        num_el += obs[key].size
    return int(num_el / 2) # every observation is added twice (default robosuite)

def register_models(env_id="SandShaping", robots="UR5e", gripper_types="SandShapingGripperQuadratic"):
    # Register additional environments
    if "SandShaping" in env_id:
        if env_id == "SandShaping":
            register_env(SandShaping)
    if "CoveragePathPlanner" in env_id:
        if env_id == "CoveragePathPlanner":
            register_env(CoveragePathPlanner)

    # Register additional gripper types
    if "SandShapingGripperQuadratic" in gripper_types:
        if gripper_types == "SandShapingGripperQuadratic":
            register_gripper(SandShapingGripperQuadratic)

def create_folder(folder_path):
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError as e:
        pass

def read_yaml(path):
    path = os.path.abspath(path)
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(path, data):
    path = os.path.abspath(path)
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False, indent=2, sort_keys=False, width=float("inf"))

def get_config(config_name):
    yaml_path = "./configs/" + config_name + ".yaml"
    config = read_yaml(yaml_path)
    return config
    
def print_dict(dict, header):
    print(header)
    for key, value in dict.items():
        entry = str(key) + " = " + str(value)
        print(entry)
    print("")

def read_json_file(path: str) -> dict:
    path = os.path.abspath(path)
    try:
        with open(path) as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {path} – {e}")
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return None

def save_json(path, data):
    path = os.path.abspath(path)
    with open(path, "w") as outfile:
        json.dump(data, outfile, indent=4)

def save_numpy_array(path, data):
    np.savetxt(path, data, fmt="%f")

# unit of file
# mesh is returned in meter
# default unit: meter
def read_mesh(path, unit="m"):
    mesh = read_triangle_mesh(path)
    if unit != "m":
        if unit == "mm":
            mesh.scale(1.0/1000.0, center=mesh.get_center())
        elif unit == "cm":
            mesh.scale(1.0/100.0, center=mesh.get_center())
        elif unit == "dm":
            mesh.scale(1.0/10.0, center=mesh.get_center())
    return mesh

def get_dataset_name_by_id(dataset_id, load_store_dir="datasets"):
    if type(dataset_id) == str:
        dataset_id = int(dataset_id)
    dataset_name = None
    datasets_path = "./" + load_store_dir +"/"
    datasets_path = os.path.abspath(datasets_path)
    datasets = os.path.join(datasets_path, "*")
    dataset_dir_paths = glob.glob(datasets)
    for dir_path in dataset_dir_paths:
        id = numerical_key(dir_path)
        if id == dataset_id:
            dataset_dir_name = Path(dir_path).name
            dataset_name = dataset_dir_name.split('_', 1)[1]
            return dataset_name
    if dataset_name is None:
        print("Could not find dataset name by id")
        raise ValueError

def dataset_path_construction(dataset_id, dataset_name, resolution, initial_sand_surface_height, angle_of_repose, load_store_dir="datasets"):
    if type(resolution) != list and type(resolution) != np.ndarray:
        resolution = [resolution,resolution]
    resolution_dir_name = "grid_"+str(resolution[1])+"x"+str(resolution[0])+"x"+str(initial_sand_surface_height)+"_"+str(angle_of_repose)+"°"
    dataset_dir_name = str(dataset_id)+"_"+dataset_name
    dataset_path = "./"+load_store_dir+"/"+dataset_dir_name+"/"+resolution_dir_name+"/"
    return dataset_path

def get_dataset_path(dataset_id, resolution=32, initial_sand_surface_height=100, angle_of_repose=34, load_store_dir = "datasets"):
    dataset_mapping = get_config("dataset")["dataset_mapping"]
    dataset_name = dataset_mapping[dataset_id]["name"]
    dataset_path = dataset_path_construction(dataset_id, dataset_name, resolution, initial_sand_surface_height, angle_of_repose, load_store_dir)
    return dataset_path

def equalize_dataset_key_list(key_list, shuffle=True):
    """
    Given a list of strings formatted as "<dataset_id>_<rest>",
    returns a new list where each dataset_id appears equally often
    by sampling with replacement, then shuffles the result.

    Args:
        key_list (List[str]): e.g. ['1_I', '1_L', '100_0', '100_1', ...]

    Returns:
        List[str]: extended & shuffled list with equal dataset weighting
    """
    # 1) Bucket keys by dataset_id
    dataset_key_map = {}
    for key in key_list:
        ds_id = key.split("_", 1)[0]
        dataset_key_map.setdefault(ds_id, []).append(key)

    # 2) Find the maximum bucket size
    max_size = max(len(keys) for keys in dataset_key_map.values())

    # 3) Extend each bucket up to max_size by sampling with replacement
    extended = []
    for keys in dataset_key_map.values():
        if len(keys) < max_size:
            extras = random.choices(keys, k=max_size - len(keys))
            extended.extend(keys + extras)
        else:
            extended.extend(keys)

    # 4) Shuffle for random interleaving
    if shuffle:
        random.shuffle(extended)
    return extended

def get_array_center(input_array: np.ndarray, center_shape: np.ndarray) -> np.ndarray:
    """
    Extracts the center of a 2D array based on the specified center_shape.
    
    Parameters:
    - input_array (np.ndarray): A 2D NumPy array.
    - center_shape (tuple): A tuple (target_rows, target_columns) representing the shape of the center to extract.
    
    Returns:
    - np.ndarray: A center sub-array of input_array with shape center_shape.
    
    Raises:
    - ValueError: If center_shape is larger than input_array in any dimension.
    """
    # Validate that the center shape is smaller or equal in each dimension
    if center_shape[0] > input_array.shape[0] or center_shape[1] > input_array.shape[1]:
        raise ValueError("center_shape must be smaller than or equal to the dimensions of input_array")
    
    # Calculate the starting indices for rows and columns
    start_row = (input_array.shape[0] - center_shape[0]) // 2
    start_col = (input_array.shape[1] - center_shape[1]) // 2
    
    # Slice the input_array to extract the center
    output_array = input_array[start_row:start_row + center_shape[0], start_col:start_col + center_shape[1]]
    return output_array

# Centers input_array in target_shape and fills the missing values with pad_value
def pad_array(input_array, target_shape, pad_value):
    # Ensure target_shape has the same number of dimensions as the input input_array.
    if len(target_shape) != input_array.ndim:
        raise ValueError("Target shape must have the same number of dimensions as the input input_array.")

    pad_width = []
    # Compute pad widths for each dimension.
    for current_dim, target_dim in zip(input_array.shape, target_shape):
        if target_dim < current_dim:
            raise ValueError("Target shape dimensions must be greater than or equal to the input dimensions.")
        total_padding = target_dim - current_dim
        # The padding is distributed evenly: before and after the input_array.
        pad_before = total_padding // 2
        pad_after = total_padding - pad_before
        pad_width.append((pad_before, pad_after))

    # Apply constant padding using pad_value as the const<nt value.
    padded_input_array = np.pad(input_array, pad_width=tuple(pad_width),
                            mode='constant', constant_values=pad_value)
    return padded_input_array

def load_dataset(dataset_id, sand_resolution, goal_resolution, initial_sand_surface_height, angle_of_repose=34, load_store_dir = "datasets", dataset_selection=None):
    dataset_path = get_dataset_path(int(dataset_id), goal_resolution, initial_sand_surface_height, angle_of_repose, load_store_dir)
    dataset_file_paths = glob.glob(dataset_path+'*.txt')
    if len(dataset_file_paths) == 0:
        print("No dataset files found in the specified path.")
    dataset_file_paths_sorted = sorted(dataset_file_paths, key=numerical_key)
    heightfield_2d_dict = {}
    for file_path in dataset_file_paths_sorted:
        file_name = Path(file_path).stem
        heightfield_2d_dict[str(dataset_id)+"_"+file_name] = np.loadtxt(file_path)
    if np.array_equal(goal_resolution, sand_resolution) == False:
        for goal_key, goal_heightfield in heightfield_2d_dict.items():
            heightfield_2d_dict[goal_key] = pad_array(goal_heightfield, sand_resolution, initial_sand_surface_height)
    
    if dataset_selection is not None: # No subset is selected
        if len(dataset_selection) > 0: # All items of this subset enter the subset 
            dataset_prefix = str(dataset_id) + "_"
            if type(dataset_selection) == list:
                dataset_selection = [dataset_prefix + str(item) for item in dataset_selection]
            elif type(dataset_selection) == str:
                dataset_selection = [dataset_prefix + dataset_selection]
            else:
                print("Dataset selection error when loading dataset!")
                raise ValueError
            heightfield_2d_dict_temp = {}
            for selection_key in dataset_selection:
                if selection_key in heightfield_2d_dict:
                    heightfield_2d_dict_temp[selection_key] = heightfield_2d_dict[selection_key]
                else:
                    print("Invalid dataset_selection!!!")
                    raise ValueError                    
            heightfield_2d_dict = heightfield_2d_dict_temp

    return heightfield_2d_dict

def numerical_key(path):
    file_name = os.path.basename(path)
    # Extract numbers from the file name
    numbers = re.findall(r'\d+', file_name)
    return int(numbers[0]) if numbers else float('inf')  # Handle files without numbers
    

def norm_check(arr, lo, hi, label, key):
    """
    arr:        the values to test
    lo, hi:     corresponding lower/upper bounds
    label:      either "input" or "output" for messaging
    key:        what is checked
    """
    comp_lo = arr < lo
    comp_hi = arr > hi

    # scalar case
    if comp_lo.ndim == 0 and comp_hi.ndim == 0:
        if comp_lo:
            print(f"Incorrect normalization {label} in: {key}")
            print(f"  value: {arr.item()} < min: {lo.item()}")
        elif comp_hi:
            print(f"Incorrect normalization {label} in: {key}")
            print(f"  value: {arr.item()} > max: {hi.item()}")
        return

    # array case: broadcast everything to the same shape
    bshape = np.broadcast_shapes(arr.shape, lo.shape, hi.shape)
    a_b = np.broadcast_to(arr,      bshape)
    lo_b = np.broadcast_to(lo,      bshape)
    hi_b = np.broadcast_to(hi,      bshape)
    bad_lo_idx = np.where(a_b < lo_b)
    bad_hi_idx = np.where(a_b > hi_b)

    if bad_lo_idx[0].size or bad_hi_idx[0].size:
        print(f"Incorrect normalization {label} in: {key}")
        for idx in zip(*bad_lo_idx):
            print(f"  at index {idx}: {a_b[idx]} < {lo_b[idx]}")
        for idx in zip(*bad_hi_idx):
            print(f"  at index {idx}: {a_b[idx]} > {hi_b[idx]}")

def normalize_value(value, c_min, c_max, normed_min=0.0, normed_max=1.0, key="", check_norm=True):
    value = np.asarray(value)
    c_min = np.asarray(c_min)
    c_max = np.asarray(c_max)
    normed_min = np.asarray(normed_min)
    normed_max = np.asarray(normed_max)
    if check_norm:
        norm_check(value, c_min, c_max, label="input", key=key)
        
    v_normed = (value - c_min) / (c_max - c_min)
    v_normed = v_normed * (normed_max - normed_min) + normed_min

    if check_norm:
        norm_check(v_normed, normed_min, normed_max, label="output", key=key)
    
    return v_normed

def denormalize_value(v_normed, c_min, c_max, normed_min, normed_max, key=""):
    v_normed = np.asarray(v_normed)
    c_min = np.asarray(c_min)
    c_max = np.asarray(c_max)
    normed_min = np.asarray(normed_min)
    normed_max = np.asarray(normed_max)
    value = (v_normed - normed_min)/(normed_max - normed_min)
    value = value * (c_max - c_min) + c_min
    return value

def normalize_dict(input_dict, keys, norm_limits, check_norm=True):
    output_dict = OrderedDict()
    for key, value in input_dict.items():
        # print(f"Key={key}\nValue={value}")
        low, high = keys[key]["limits"]
        normed_value = normalize_value(value, low, high, norm_limits[0], norm_limits[1], key, check_norm=check_norm)
        output_dict[key] = normed_value
    return output_dict

def denormalize_dict(input_dict, keys, norm_limits):
    output_dict = OrderedDict()
    for key, value in input_dict.items():
        # print(f"Key={key}\nValue={value}")
        low, high = keys[key]["limits"]
        output_dict[key] = denormalize_value(value, low, high, norm_limits[0], norm_limits[1], key)
    return output_dict
    
# Calculate the angle difference considering the circular nature
def circular_angle_difference(angle1, angle2):
    diff = abs(angle1 - angle2) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)

def round(value, decimals=0):
    """
    Rounds a number to a given precision in decimal digits using
    the "round half up" strategy.
    """
    d = Decimal(str(value))
    quantize_format = Decimal('1.' + '0' * decimals) if decimals > 0 else Decimal('1')
    return float(d.quantize(quantize_format, rounding=ROUND_HALF_UP))

def torch_settings(hardware):
    print("Execute code on hardware:", hardware)
    if hardware == "cuda":
        print("Cuda is available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            torch.device("cuda")
    else:
        torch.device("cpu")

    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

def set_random_seed(hardware, random_seed):
    """
    Generalizing random seed
    """
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if hardware == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        # Important for reproducability
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False     

def get_env_index(env, locals):
    for i in range(env.num_envs):
        info = locals["infos"][i]
        env_index = info.get("env_index", None)
        if env_index is not None:
            pass
        else:
            print("Could not get environment index.")
            raise ValueError
    return env_index

def get_env_variable_value(env, locals, variable_name):
    for i in range(env.num_envs):
        info = locals["infos"][i]
        env_index = info.get("env_index", None)
        if env_index is not None:
            variable_value = env.unwrapped.env_method("get_variable", str(variable_name), indices=env_index)
        else:
            print("Could not get environment index.")
            raise ValueError
    if type(variable_value) == list:
        variable_value = variable_value[0]
    return variable_value

def set_env_variable_value(env, locals, variable_name, variable_value):
    sucess = False
    for i in range(env.num_envs):
        info = locals["infos"][i]
        env_index = info.get("env_index", None)
        if env_index is not None:
            try:
                env.unwrapped.env_method(str("set_variable"), str(variable_name), variable_value, indices=env_index)
                sucess = True
            except:
                print("Could not set environment variable", variable_name)
                raise ValueError
        else:
            print("Could not get environment index.")
            raise ValueError
    return sucess

def comma_separated_list(s: str) -> Optional[List[str]]:
    if s.lower() == "none":
        res = [None]
    else:
        res = [item.strip() for item in s.split(',') if item.strip()]
    return res

def comma_separated_list_with_substrings(s: str) -> Optional[List[str]]:
    res = []
    for item in s.split('"'):
        item = item.strip()
        if len(item) > 0 and item != ",":
            inner_list = []
            for sub_item in item.split(','):
                sub_item = sub_item.strip()
                if sub_item.lower() == "none":
                    inner_list = None
                else:
                    inner_list.append(sub_item)
            res.append(inner_list)
    return res

def u_test(group1, group2, type_string):
    U, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
    significance = check_significance(p_value)
    print("p_{}:{:e} ({})".format(type_string, p_value, significance))

def t_test(group1, group2, type_string):
    t_stat, p_value = ttest_ind(group1, group2)
    significance = check_significance(p_value)
    print("p_{}:{:e} ({})".format(type_string, p_value, significance))

def check_significance(p_value):
    if p_value < 0.001:
        text = '***'
    elif p_value < 0.01:
        text = '**'
    elif p_value < 0.05:
        text = '*'
    else:
        text = 'ns'  # not significant
    return text