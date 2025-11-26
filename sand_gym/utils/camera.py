import cv2
import numpy as np
import matplotlib.cm as cm
import sand_gym.utils.common as common
from collections import OrderedDict
from robosuite.utils.camera_utils import get_real_depth_map

def save_img(path, img_np):
    cam_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, cam_img)
    print(f"Saved: {path}")

def get_depth_clip_range(sim):
    extent = sim.model.stat.extent
    far = sim.model.vis.map.zfar * extent
    near = sim.model.vis.map.znear * extent
    return [near, far]

def convert_real_depth_image_to_opencv(img, clip_range):
    # Normalize the clipped depth image to the range [0, 255]:
    # Subtract the minimum, divide by the range, then scale to 255.
    min_clip = clip_range[0]
    max_clip = clip_range[1]
    depth_clipped = np.clip(img, a_min=min_clip, a_max=max_clip)
    depth_normalized = ((depth_clipped - min_clip) / (max_clip - min_clip)) * 255.0

    # Convert the normalized image to 8-bit (required for cv2.applyColorMap and imshow)
    depth_normalized = depth_normalized.astype(np.uint8)
    return depth_normalized

def get_mask_for_class(img, class_value):
    """
    Returns a boolean mask for the specified class_value in the segmentation image.

    Parameters:
        img (np.ndarray): The original segmentation image. Expected to be either 2D (H, W)
                          or 3D with a singleton channel (H, W, 1).
        class_value (int): The class value for which to generate the mask (e.g., 13).

    Returns:
        np.ndarray: A boolean mask of shape (H, W) where True indicates the pixel equals class_value.
    """
    # Convert image to 2D if it's (H, W, 1)
    if img.ndim == 3 and img.shape[-1] == 1:
        img_2d = np.squeeze(img, axis=2)
    else:
        img_2d = img

    # Return a boolean mask where the segmentation image equals class_value
    mask = (img_2d == class_value)
    return mask

def mask_depth_img(depth_img, mask, mask_value):   
    # Since depth_img is a 3D array with shape (H, W, 1), we access its 2D version using depth_img[..., 0].
    # The expression ~mask inverts the boolean mask so that we target pixels where the mask is False
    depth_img[..., 0][~mask] = mask_value

    return depth_img

def show_cv_img(img, type, clip_range=None, name=None, overlay_labels=True):
    if type == "mask":
        if name is None:
            name="Mask Image"
        img = img.astype(np.uint8) * 255
    elif type == "depth":
        if name is None:
            name="Depth Image"
        if clip_range is None:
            print("Clip range is required to display depth images")
            raise ValueError
        img = convert_real_depth_image_to_opencv(img, clip_range)
        # Apply a colormap for better visualization JET colormap
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif type == "seg_depth":
        if name is None:
            name="Segmented Depth Image"
        if clip_range is None:
            print("Clip range is required to display depth images")
            raise ValueError
        img = convert_real_depth_image_to_opencv(img, clip_range)
        # Apply a colormap for better visualization JET colormap
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif type == "normed_seg_depth":
        if name is None:
            name="Normed Segmented Depth Image"
        img = (img * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    elif type == "rgb":
       if name is None:
        name="RGB Image" 
       img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif type == "bgr":
       if name is None:
        name="BGR Image"
    elif type == "segmentation":
        if name is None:
            name="Segmentation Image"
        # Step 1: Find the min and max class values
        min_class = img.min()
        max_class = img.max()

        # Step 2: Generate a colormap
        # Use a colormap with distinct colors (e.g., 'viridis', 'tab20', or 'rainbow')
        num_classes = max_class - min_class + 1
        colormap = cm.get_cmap('tab10', num_classes)  # You can change 'tab10' to other colormaps

        # Convert colormap to RGB values in range [0, 255]
        colors = (colormap(np.arange(num_classes))[:, :3] * 255).astype(np.uint8)

        # Step 3: Create a blank image and map class values to colors
        height, width, _ = img.shape
        segmentation_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Convert image to 2D if it's (height, width, 1)
        img_2d = img.squeeze(axis=2) if (img.ndim == 3 and img.shape[-1] == 1) else img

        for class_value in range(min_class, max_class + 1):
            mask = img_2d == class_value  # mask is now (height, width)
            segmentation_image[mask] = colors[class_value - min_class]
        
        if overlay_labels:
            for class_value in range(min_class, max_class + 1):
                mask = img_2d == class_value
                if np.any(mask):  # Only add a label if the class is present
                    # Find all pixel coordinates for the current class.
                    coords = np.argwhere(mask)
                    # Compute the center of these coordinates.
                    center = coords.mean(axis=0).astype(int)
                    # OpenCV's putText expects (x, y), so swap (row, col)
                    center_xy = (center[1], center[0])
                    cv2.putText(segmentation_image,
                                str(class_value),      # Text label is the class value
                                center_xy,               # Position (x, y)
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,                     # Font scale
                                (255, 255, 255),         # Text color (white)
                                2,                       # Thickness
                                cv2.LINE_AA)
        img = segmentation_image


    # Display the image in a window
    cv2.imshow(name, img)
    
    return name

def loop_cv_window(names=None):
    # Instead of using waitKey(0), use a loop to check if the window is still open
    window_closed = False
    try:
        while window_closed == False:
            # Check if the window is still open; if not, break the loop.
            for name in names:
                prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
                # print(f"Window property (visible): {prop}")
                if prop < 1:
                    # print("Window closed by user.")
                    window_closed = True
            # You can also allow a key press to break out of the loop (e.g., 'q')
            if cv2.waitKey(100) & 0xFF == ord('q'):
                window_closed = True
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

def add_depth_seg_to_obs(observation, norm_range = None, camera_config=None, robosuite_env=None, show=False):
    depth_seg_observations = OrderedDict()

    if camera_config is None:
        print("segmentation_config for image segmentation is missing!")
        raise ValueError

    camera_names = camera_config["camera_names"]
    clip_ranges = camera_config["clip_ranges"]
    
    segmentation_config = camera_config["segmentation_config"]
    sand_geom_id = segmentation_config["sand_geom_id"]
    robot_geom_ids = segmentation_config.get("robot_geom_ids", [])

    for cam_name in camera_names:
        if show:
            window_names = []

        ####################
        ### Depth images ###
        ####################
        clip_range = clip_ranges[cam_name]
        
        depth_name = cam_name + "_depth"
        depth_img_np = observation[depth_name]
        if robosuite_env is not None:
            depth_img_np = get_real_depth_map(sim=robosuite_env, depth_map=depth_img_np)

        depth_img_clipped_np = np.clip(depth_img_np, a_min=clip_range[0], a_max=clip_range[1])

        ###########################
        ### Segmentation images ###
        ###########################
        seg_name = cam_name + "_segmentation_geom"
        seg_img_np = observation[seg_name]

        sand_mask = get_mask_for_class(seg_img_np, sand_geom_id)
        if show:
            window_names.append(show_cv_img(sand_mask, "mask", name="Sand Mask"))
        robot_mask = np.full((depth_img_np.shape[0], depth_img_np.shape[1]), False, dtype="bool")
        for rob_geom_id in robot_geom_ids:
            temp_mask = get_mask_for_class(seg_img_np, rob_geom_id)
            if np.any(temp_mask > 0):
                robot_mask = np.logical_or(robot_mask, temp_mask)
        if show:
            window_names.append(show_cv_img(robot_mask, "mask", name="Robot Mask"))
        sand_gripper_union_mask = np.logical_or(sand_mask, robot_mask)

        # Apply morphological closing to fill small gaps (use a small kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed_mask_cv = cv2.morphologyEx(sand_gripper_union_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # Convert to boolean mask
        closed_mask = closed_mask_cv > 0
        if show:
            window_names.append(show_cv_img(closed_mask, "mask", name="closed_mask"))

        c_min=clip_range[0]
        if "mask_value" not in segmentation_config:
            mask_value = clip_range[1]
            c_max=clip_range[1]
        else:
            c_max=segmentation_config["mask_value"]
        seg_depth_img_np = mask_depth_img(depth_img_clipped_np, closed_mask, mask_value)
        if show:
            window_names.append(show_cv_img(seg_depth_img_np, "seg_depth", clip_range))
        
        if norm_range is not None:
            seg_depth_img_np = common.normalize_value(seg_depth_img_np, c_min=c_min, c_max=c_max, normed_min=norm_range[0], normed_max=norm_range[1], key="depth conversion")
            if show:
                window_names.append(show_cv_img(seg_depth_img_np, "normed_seg_depth"))

        if show:
            loop_cv_window(names=window_names)

        img_name = cam_name + "_depth_seg"
        depth_seg_observations[img_name] = seg_depth_img_np

    return depth_seg_observations