import os
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_real_depth_map, get_camera_extrinsic_matrix, parse_intrinsics
import open3d as o3d
import cv2
from collections import OrderedDict
import numpy as np
import functools
from scipy.ndimage import convolve, distance_transform_edt
import sand_gym.utils.camera as cam_utils
from sand_gym.utils.plot_window import PlotSandWindow, PlotEvalSandWindow

lime_green = (0.05, 0.80, 0.15) # lime-green
steel_blue = (0.15, 0.40, 0.80) # steel‑blue
crimson_red = (0.80, 0.05, 0.15)   # crimson-red

plot_window_rec_diff = None
plot_window_rec_current = None

o3d_window = None

stop_o3d_window = False
def on_key(o3d_window):
    global stop_o3d_window
    stop_o3d_window = True
    return False

def visualize_o3d(geometries, colors=None, plot_library="o3d"):
    if colors is None:
        colors = int(len(geometries))*[steel_blue]
    geoms = []
    for geom, color in zip(geometries, colors):
        if type(geom) == o3d.geometry.PointCloud:
            pass
        elif type(geom) == o3d.geometry.AxisAlignedBoundingBox:
            # Convert bbox (TriangleMesh) to PointCloud for visualization
            bbox_points = np.asarray(geom.get_box_points())  # Get the 8 corners of the bounding box
            bbox_pcd = o3d.geometry.PointCloud()
            bbox_pcd.points = o3d.utility.Vector3dVector(bbox_points)  # Convert vertices to point cloud
            geom = bbox_pcd
        geom.paint_uniform_color(color)
        geoms.append(geom)

    # Visualize the point clouds and the bounding box as points
    if plot_library=="plotly":
        o3d.visualization.draw_plotly(geoms)
    elif plot_library=="o3d":
        global o3d_window
        if o3d_window is None:
            o3d_window = o3d.visualization.VisualizerWithKeyCallback()
            o3d_window.create_window("O3D Pointcloud")
            o3d_window.poll_events()
            o3d_window.update_renderer()
            o3d_window.register_key_callback(ord('C'), on_key)

        global stop_o3d_window
        stop_o3d_window = False
        o3d_window.clear_geometries()
        for geom in geoms:
            o3d_window.add_geometry(geom)
            o3d_window.update_geometry(geom)
        while not stop_o3d_window and o3d_window.poll_events():
            o3d_window.update_renderer()

def save_pcd(path, pcd):
    o3d.io.write_point_cloud(path, pcd)
    print(f"Saved: {path}")

def read_pcd(path):
    print(f"Read point cloud from: {path}")
    pcd = o3d.io.read_point_cloud(path)
    return pcd

def alpha_shape_mesh_reconstruct(pcd, alpha=0.5, mesh_fix=False, visualize=False):
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha, tetra_mesh, pt_map)
    return mesh

def get_individual_pcd(cam_name, obs, seg_id, cam_width, cam_height, clip_range, crop_bound=None, visualize=False, env=None, save_dir=""):
    # Load the RGB-D image and camera parameters
    fx, fy, cx, cy = parse_intrinsics(get_camera_intrinsic_matrix(
        env.sim, cam_name, cam_height, cam_width))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=cam_width, height=cam_height, fx=fx, fy=fy, cx=cx, cy=cy)
    extrinsic = get_camera_extrinsic_matrix(env.sim, cam_name)

    if len(save_dir)>0:
        rgb_np = obs[f"{cam_name}_image"]
        rgb_img_name =  f"{cam_name}_{env.timestep}"
        rgb_img_save_path = os.path.join(save_dir, rgb_img_name+".jpg")
        cam_utils.save_img(rgb_img_save_path, rgb_np)

    binary_mask = (obs[f"{cam_name}_segmentation_geom"]
                   [:, :, 0] == seg_id).astype(np.uint8)
    # Apply the segmentation mask to the RGB-D image
    seg_rgb = cv2.bitwise_and(
        obs[f"{cam_name}_image"], obs[f"{cam_name}_image"], mask=binary_mask)

    if len(save_dir)>0:
        seg_rgb_img_name =  f"{cam_name}_seg_{env.timestep}"
        seg_rgb_img_save_path = os.path.join(save_dir, seg_rgb_img_name+".jpg")
        cam_utils.save_img(seg_rgb_img_save_path, seg_rgb)

    real_depth_map = get_real_depth_map(env.sim, obs[f"{cam_name}_depth"])
    real_depth_map = np.clip(real_depth_map, a_min=clip_range[0], a_max=clip_range[1])
    seg_depth = cv2.bitwise_and(
        real_depth_map, real_depth_map, mask=binary_mask)

    rgb = o3d.geometry.Image((seg_rgb))
    depth = o3d.geometry.Image(seg_depth)

    # Convert the RGB-D image to a point cloud
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb, depth, depth_scale=1.0, depth_trunc=clip_range[1], convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic)

    # Transform the point cloud to the world frame
    pcd.transform(extrinsic)

    # Define the bounding box
    if crop_bound is not None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=crop_bound[0], max_bound=crop_bound[1])
        pcd = pcd.crop(bbox)

    if visualize:
        visualize_o3d([pcd, bbox], [steel_blue, crimson_red])

    return pcd


def get_pcd_from_rgbd(obs, camera_names, desired_seg_id, clip_ranges, crop_bound=None, visualize=False, env=None, save_dir=""):

    def f(x): return [get_individual_pcd(cam_name, obs, x, cam_width, cam_height, clip_ranges[cam_name], crop_bound, visualize, env=env, save_dir=save_dir)
                      for cam_name, cam_height, cam_width in zip(camera_names, env.camera_heights, env.camera_widths)]
    pcd_object_dict = OrderedDict(
        map(lambda kv: (kv[0], f(kv[1])), desired_seg_id.items()))

    return pcd_object_dict


def merge_pcds(pcd_list):
    return functools.reduce(lambda x, y: x + y, pcd_list)


def merge_point_cloud(camera_names, obs, desired_seg_id, clip_ranges, crop_bound=None, visualize=False, env=None, save_dir=""):

    pcd_object_dict = get_pcd_from_rgbd(
        obs, camera_names, desired_seg_id, clip_ranges, crop_bound, visualize, env, save_dir)

    pcd_object_dict = OrderedDict(
        map(lambda kv: (kv[0], merge_pcds(kv[1])), pcd_object_dict.items()))

    if visualize:
        visualize_o3d(list(pcd_object_dict.values()))

    return pcd_object_dict

def preprocess_object_raw_pcd(pcd_object, preprocess_config=None, visualize=False):
    if preprocess_config is None:
        print("preprocess_config for pointcloud preprocessing is missing!")
        raise ValueError
    
    if "radius" in preprocess_config["outlier_removal_mode"]:
        outliers = None
        outlier_stat = None
        cl, inlier_ind_pcd_stat = pcd_object.remove_radius_outlier(nb_points=preprocess_config["nb_points"], radius=preprocess_config["radius"])
        pcd_stat = pcd_object.select_by_index(inlier_ind_pcd_stat)
        outlier_stat = pcd_object.select_by_index(
            inlier_ind_pcd_stat, invert=True)
        if outliers is None:
            outliers = outlier_stat
        else:
            outliers += outlier_stat

        pcd_object = pcd_stat

    if "statistical" in preprocess_config["outlier_removal_mode"]:
        rm_iter = 1
        outliers = None
        outlier_stat = None
        while rm_iter < 2:
            cl, inlier_ind_pcd_stat = pcd_object.remove_statistical_outlier(
                nb_neighbors=preprocess_config["nb_neighbors"], std_ratio=preprocess_config["std_ratio"]*rm_iter)
            pcd_stat = pcd_object.select_by_index(inlier_ind_pcd_stat)
            outlier_stat = pcd_object.select_by_index(
                inlier_ind_pcd_stat, invert=True)
            if outliers is None:
                outliers = outlier_stat
            else:
                outliers += outlier_stat

            pcd_object = pcd_stat
            rm_iter += 1

    if visualize:
        outliers.paint_uniform_color([1, 0, 0.0])
        visualize_o3d([pcd_object, outliers], title='cleaned_workspace')

    sampled_pcd = pcd_object

    return sampled_pcd

def fill_nan_with_interpolation(field, max_iter=100):
    f = field.copy().astype(float)
    
    # mask
    mask = np.isnan(f)
    if not mask.any():
        return f

    # 8-neighbor sum kernel (center weight zero)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=float)

    for _ in range(max_iter):
        # convolve sum and count of valid neighbors
        neighbor_sum   = convolve(np.nan_to_num(f), kernel, mode='mirror')
        neighbor_count = convolve((~mask).astype(float), kernel, mode='mirror')

        # positions we can fill this pass
        fill_pos = mask & (neighbor_count > 0)
        if not fill_pos.any():
            break

        # fill them with neighbor mean
        f[fill_pos] = neighbor_sum[fill_pos] / neighbor_count[fill_pos]
        # update mask
        mask = np.isnan(f)
        if not mask.any():
            break

    # if any NaNs remain (e.g. isolated), do simple nearest-neighbor fill
    if mask.any():
        # distance-transform to nearest valid pixel
        from scipy.ndimage import distance_transform_edt
        # get indices of nearest valid
        _, inds = distance_transform_edt(mask, return_indices=True)
        f[mask] = f[tuple(inds[:, mask])]

    return f

def fill_nan_nearest(field):
    f = np.array(field, copy=True)

    # Mask of holes
    nan_mask = np.isnan(f)
    if not nan_mask.any():
        return f

    # Compute the indices of the nearest zero‐pixels
    indices = distance_transform_edt(
        nan_mask,
        return_distances=False,
        return_indices=True
    )

    # Replace each NaN with the value at its nearest non‐NaN
    rows = indices[0][nan_mask]
    cols = indices[1][nan_mask]
    f[nan_mask] = f[rows, cols]

    return f

def compute_height_field(pcd,
                         field_size,
                         resolution=0.01,
                         z_offset=0.0,
                         convert_m_to_mm=True,
                         heightmap_config=None,
                         env=None,
                         tool_2d_pos=None,
                         tool_heights=None):
    
    if heightmap_config is None:
        raise ValueError("heightmap_config for heightmap reconstruction is missing!")

    # Unpack
    grid_h, grid_w = field_size
    mapping = heightmap_config["point_to_cell_mapping"]
    fill_tool = heightmap_config.get("fill_tool_cells", None)
    fill_empty = heightmap_config.get("fill_empty_cells", None)

    # Extract and filter points
    pts = np.asarray(pcd.points)  # (N,3)
    if pts.size == 0:
        height_field = np.full((grid_h, grid_w), np.nan)
    else:
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2] - z_offset

        # Compute grid bounds
        x_min = - (grid_w * resolution) / 2
        x_max = - x_min
        y_min = - (grid_h * resolution) / 2
        y_max = - y_min

        # Mask to in-bounds
        m = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)
        xs, ys, zs = xs[m], ys[m], zs[m]

        # Map to integer cell indices
        ix = np.floor((xs - x_min) / resolution).astype(int)
        iy = np.floor((ys - y_min) / resolution).astype(int)
        np.clip(ix, 0, grid_w - 1, out=ix)
        np.clip(iy, 0, grid_h - 1, out=iy)

        # Flattened bin index
        flat = iy * grid_w + ix
        total_bins = grid_h * grid_w

        # Average mapping
        if mapping == "average":
            sum_flat = np.bincount(flat, weights=zs, minlength=total_bins)
            cnt_flat = np.bincount(flat, minlength=total_bins)
            # avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                hf_flat = sum_flat / cnt_flat
            hf_flat[cnt_flat == 0] = np.nan

        # Max mapping
        elif mapping == "max":
            # initialize to the identity for max
            hf_flat = np.full(total_bins, -np.inf, dtype=float)
            np.maximum.at(hf_flat, flat, zs)
            # bins that stayed at -inf had no points → mark them as empty
            hf_flat[np.isneginf(hf_flat)] = np.nan

        # Min mapping
        elif mapping == "min":
            # initialize to the identity for min
            hf_flat = np.full(total_bins,  np.inf, dtype=float)
            np.minimum.at(hf_flat, flat, zs)
            # bins that stayed at +inf had no points → mark them as empty
            hf_flat[np.isposinf(hf_flat)] = np.nan
            
        else:
            raise ValueError(f"Unknown point_to_cell_mapping: {mapping}")

        # Reshape back to 2D
        height_field = hf_flat.reshape((grid_h, grid_w))

    # Fill empty cells
    if np.isnan(height_field).any() and fill_empty is not None:
        if isinstance(fill_empty, (float, int)):
            height_field = np.nan_to_num(height_field, nan=fill_empty)
        elif fill_empty == "interpolation":
            height_field = fill_nan_with_interpolation(height_field)
        elif fill_empty == "nearest_neighbor":
            height_field = fill_nan_nearest(height_field)
        elif fill_empty == "min_valid_height":
            hmin = np.nanmin(height_field)
            height_field = np.nan_to_num(height_field, nan=hmin)
        else:
            raise ValueError(f"Invalid fill_empty_cells: {fill_empty}")

    # Fill tool cells if requested
    if fill_tool is not None:
        # tool_heights expected in meters → match our mm later
        if tool_heights is not None:
            tool_h = tool_heights * (1000 if convert_m_to_mm else 1)
        else:
            tool_h = None
        if env.sand_simulator.tool_in_sand_check(tool_2d_pos=tool_2d_pos,
                                                 tool_heights=tool_h,
                                                 heightfield=height_field*1000):
            if tool_h is None:
                tool_h = env.sand_simulator.get_heights_by_polygon(env.sand_simulator.tool_geom_name)/1000
            mask = ~np.isnan(tool_h)
            if fill_tool == "tool_values":
                height_field[mask] = tool_h[mask]
            elif isinstance(fill_tool, (float, int)):
                height_field[mask] = fill_tool
            else:
                raise ValueError(f"Invalid fill_tool_cells: {fill_tool}")
            
    # Unit conversion
    if convert_m_to_mm:
        height_field = height_field * 1000

    return height_field

def transform_pointclouds(pcd_dict, translation_vector=None, rotation_vector=None, rotation_center=[0.0,0.0,0.0]):
    transformed_pcds = OrderedDict()
    for key, pcd in pcd_dict.items():
        # 1) Rotate around the center about XYZ
        if rotation_vector is not None:
            angles = np.deg2rad(rotation_vector)
            R = pcd.get_rotation_matrix_from_xyz(angles)
            pcd.rotate(R, center=rotation_center)
        # 2) Translate
        if translation_vector is not None:
            pcd.translate(translation_vector)
        transformed_pcds[key] = pcd
    return transformed_pcds

def remove_eef_from_pointclouds(pcds, crop_bound_eef, visualize=False):
    if type(pcds) == dict or type(pcds) == OrderedDict:
        cropped_pcd = OrderedDict()
        for key, pcd in pcds.items():
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_bound_eef[0], max_bound=crop_bound_eef[1])
            inliers_indices = bbox.get_point_indices_within_bounding_box(pcd.points)
            inliers_pcd = pcd.select_by_index(inliers_indices, invert=False) # select inside points = cropped 
            outliers_pcd = pcd.select_by_index(inliers_indices, invert=True) #select outside points
            if visualize:
                num_inlier_points = len(inliers_pcd.points)
                print("Removed eef points:",num_inlier_points)
            if visualize:
                visualize_o3d([inliers_pcd, outliers_pcd, bbox], [lime_green, steel_blue, crimson_red])
            cropped_pcd[key] = outliers_pcd
    else:
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=crop_bound_eef[0], max_bound=crop_bound_eef[1])
        inliers_indices = bbox.get_point_indices_within_bounding_box(pcds.points)
        inliers_pcd = pcds.select_by_index(inliers_indices, invert=False) # select inside points = cropped 
        outliers_pcd = pcds.select_by_index(inliers_indices, invert=True) #select outside points
        if visualize:
            num_inlier_points = len(inliers_pcd.points)
            print("Removed eef points:",num_inlier_points)
        if visualize:
            visualize_o3d([inliers_pcd, outliers_pcd, bbox], [lime_green, steel_blue, crimson_red])
        cropped_pcd = outliers_pcd
    return cropped_pcd

def get_seg_pointcloud(camera_names, obs, clip_ranges, segmentation_config=None, crop_bound_sand_box=None, crop_eef=False, transform=None, preprocess_pointcloud=False, pointcloud_config=None, env=None, visualize=False, save_dir=""):
    if segmentation_config is not None:
        # Set the desired segmentation label
        sand_geom_id = segmentation_config["sand_geom_id"]
        desired_seg_id = OrderedDict([("sand_box", np.int32(sand_geom_id)),])
    
    pcd_object_dict = merge_point_cloud(
        camera_names, obs, desired_seg_id, clip_ranges=clip_ranges, crop_bound=crop_bound_sand_box, visualize=visualize, env=env, save_dir=save_dir)

    if len(save_dir)>0:
        concat_camera_names = "_".join(camera_names)
        for key, pcd in pcd_object_dict.items():
            pcd_name = f"1_pcd_{concat_camera_names}_{key}_sand_segmented_{env.timestep}.ply"
            pcd_save_path = os.path.join(save_dir, pcd_name)
            save_pcd(pcd_save_path, pcd)

    # Define the bounding box
    if crop_eef:
        # Get the tool bounding box
        crop_bound_eef = env.sand_simulator.get_tool_bounding_box(reference="world")
        # Inflate the tool bounding box
        crop_eef_inflation = segmentation_config.get("crop_eef_inflation", 0.0)
        # Set the min/max z values to the sand box crop limits to "punch" a hole into the segmented pointcloud 
        crop_bound_eef = np.array([crop_bound_eef[0]-crop_eef_inflation, crop_bound_eef[1]+crop_eef_inflation], dtype=np.float32)
        crop_bound_eef[:,2] = crop_bound_sand_box[:,2]
        pcd_object_dict = remove_eef_from_pointclouds(pcd_object_dict, crop_bound_eef, visualize)

        if len(save_dir)>0:
            concat_camera_names = "_".join(camera_names)
            for key, pcd in pcd_object_dict.items():
                pcd_name = f"2_pcd_{concat_camera_names}_{key}_eef_punched_{env.timestep}.ply"
                pcd_save_path = os.path.join(save_dir, pcd_name)
                save_pcd(pcd_save_path, pcd)

    if preprocess_pointcloud:
        if pointcloud_config is not None:
            preprocess_config = pointcloud_config["preprocess_config"]
        else:
            preprocess_config = None
        pcd_object_dict = OrderedDict(
            map(lambda kv: (kv[0], preprocess_object_raw_pcd(
                kv[1], preprocess_config=preprocess_config, visualize=visualize)), pcd_object_dict.items()))

        if len(save_dir)>0:
            for key, pcd in pcd_object_dict.items():
                pcd_name = f"3_pcd_{concat_camera_names}_{key}_processed_{env.timestep}.ply"
                pcd_save_path = os.path.join(save_dir, pcd_name)
                save_pcd(pcd_save_path, pcd)

    if transform is not None:
        pcd_object_dict = transform_pointclouds(pcd_object_dict, transform[0], transform[1], transform[2])

        if len(save_dir)>0:
            concat_camera_names = "_".join(camera_names)
            for key, pcd in pcd_object_dict.items():
                pcd_name = f"4_pcd_{concat_camera_names}_{key}_transformed_{env.timestep}.ply"
                pcd_save_path = os.path.join(save_dir, pcd_name)
                save_pcd(pcd_save_path, pcd)

    return pcd_object_dict

def add_reconstructed_heightmap_to_obs(obs, env, convert_to_obs_shape=True, camera_config=None, verbose=0, remove_eef=True):
    reconstructed_heightmap_observations = OrderedDict()

    if camera_config is None:
        print("camera_config for heightmap reconstruction is missing!")
        raise ValueError
    
    reconstruct_heightmap_config = camera_config["reconstruct_heightmap_config"]
    pointcloud_config = camera_config["pointcloud_config"]
    preprocess_pointcloud = pointcloud_config["preprocess_pointcloud"]
    
    # Cameras to use for pointcloud generation.
    # Exclude eval cameras
    camera_names = camera_config["camera_names"]
    camera_names = [s for s in camera_names if "_eval" not in s]
    
    clip_ranges = camera_config["clip_ranges"]
    segmentation_config = camera_config["segmentation_config"]
    crop_bound_sand_box = env.sand_simulator.get_sand_bounding_box()
    sand_cut_off_height = segmentation_config.get("sand_cut_off_height", None)
    if sand_cut_off_height is not None:
        crop_bound_sand_box[1][2] = crop_bound_sand_box[0][2] + sand_cut_off_height
    if remove_eef:
        crop_eef = segmentation_config.get("crop_eef", False)
    else:
        crop_eef = False
    grid_cell_size_in_m = env.sand_simulator.grid_cell_size_in_m
    field_size = env.sand_simulator.field_size
    
    # sand_box_offset = env.sand_simulator.sand_box_offset
    sand_box_bottom = env.sand_simulator.get_sand_bottom_in_world_coords()
    translation_vector = -sand_box_bottom
    rotation_vector = None
    rotation_center = None
    transform = [translation_vector, rotation_vector, rotation_center]
    seg_pointclouds = get_seg_pointcloud(camera_names, obs, clip_ranges, segmentation_config, crop_bound_sand_box, crop_eef, transform, preprocess_pointcloud=preprocess_pointcloud,pointcloud_config=pointcloud_config, env=env, visualize=(verbose==6))
    sand_pointcloud = seg_pointclouds["sand_box"]

    height_field = compute_height_field(sand_pointcloud, field_size, resolution=grid_cell_size_in_m, heightmap_config=reconstruct_heightmap_config, env=env)        

    if verbose >= 3:
        global plot_window_rec_current
        if plot_window_rec_current == None:
            plot_window_rec_current = PlotEvalSandWindow(name="", field_size_grid=[60,30], field_size_real=[0.3,0.6], grid_cell_size=0.01)
        vis_geoms = []
        vis_geoms.append(np.copy(height_field))
        plot_window_rec_current.update(vis_geoms)

    if convert_to_obs_shape:
        height_field = env.sand_simulator.convert_heightfield_to_observation(height_field)

    reconstructed_heightmap_observations["reconstructed_heightmap_current"] = height_field
    return reconstructed_heightmap_observations

def add_reconstructed_heightmap_diff_to_obs(obs, env=None, goal_mask=False, verbose=0):
    reconstructed_heightmap_diff_observations = OrderedDict()
    reconstructed_heightmap_diff = obs["heightmap_goal"] - obs["reconstructed_heightmap_current"]

    if goal_mask:
        if "goal_mask" in obs:
            mask = obs["goal_mask"]
        elif "goal_mask" not in obs and env is not None:
            mask = env.sand_simulator.get_goal_mask()
        reconstructed_heightmap_diff = np.where(mask==1.0, reconstructed_heightmap_diff, 0.0)

    if verbose >= 3:
        global plot_window_rec_diff
        if plot_window_rec_diff == None:
            plot_window_rec_diff = PlotSandWindow(name="Reconstructed difference heightfield observation", field_size_grid=[32,32], field_size_real=[0.3,0.6], grid_cell_size=0.01)
        vis_geoms = []
        vis_geoms.append(np.copy(reconstructed_heightmap_diff))
        if "robot0_eef_pos" in obs:
            tool_2d_world_pos = np.copy(obs["robot0_eef_pos"])[:2]
            tool_2d_world_pos[0] =  tool_2d_world_pos[0] - 0.55
            tool_2d_grid_pos = env.sand_simulator.convert_world_to_grid(tool_2d_world_pos, check_norm=False)
            tool_2d_grid_pos[0] = tool_2d_grid_pos[0] + 1
            tool_2d_grid_pos[1] = tool_2d_grid_pos[1] - 14
            vis_geoms.append(tool_2d_grid_pos)
        vis_geoms.append(np.copy(mask).astype(int))
        if "tool_mask" in obs:
            tool_mask = np.copy(obs["tool_mask"]).astype(int)
            vis_geoms.append(tool_mask)
        plot_window_rec_diff.update(vis_geoms)
    reconstructed_heightmap_diff_observations["reconstructed_heightmap_diff"] = reconstructed_heightmap_diff
    return reconstructed_heightmap_diff_observations

def heightmap_to_pcd(heightmap, cell, zscale=1.0, mask=None):
    i_idx, j_idx = np.indices(heightmap.shape)
    if mask is None:
        mask_bool = np.ones_like(heightmap, dtype=bool)
    else:
        mask_bool = mask != 0.0

    x = (i_idx.ravel()[mask_bool.ravel()] * cell).astype(np.float32)
    y = (j_idx.ravel()[mask_bool.ravel()] * cell).astype(np.float32)
    z = (heightmap.ravel()[mask_bool.ravel()] * zscale).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x, y, z)))
    return pcd