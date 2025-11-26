import numpy as np
from sand_gym.robotics_toolbox.UR5e import UR5e
from spatialmath import SE3
from robosuite.utils.transform_utils import quat2mat, mat2euler

def get_initial_qpos(robot, offset, pos, quat):
    """
    Calculates the initial joint position for the robot based on the initial desired pose (self.goal_pos, self.goal_quat).
    If self.initial_gripper_pos_randomization is True, Guassian noise is added to the initial position of the gripper.

    Args:

    Returns:
        (np.array): n joint positions 
    """
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)

    position = pos + offset
    ori_euler = mat2euler(quat2mat(quat))

    # desired pose
    T = SE3(position) * SE3.RPY(ori_euler)
    
    if robot.name == "UR5e":
        robot = UR5e()
        # Set an initial guess that biases both the wrist pointing outwards and the elbow pointing upwards:
        # - Joint indices: 0: base, 1: shoulder, 2: elbow, 3-5: wrist joints.
        q0 = np.array([
            0.0,           # Base joint
            -np.pi/2,      # Shoulder joint (elbow up bias)
            np.pi/2,       # Elbow joint (elbow up bias)
            -np.pi/2,      # Wrist joint 1 (wrist outwards)
            -np.pi/2,      # Wrist joint 2 (wrist outwards)
            0.0            # Wrist joint 3
        ])
        sol = robot.ikine_LM(T, q0=q0)
        return sol.q
    else:
        print("Unknown robot!")