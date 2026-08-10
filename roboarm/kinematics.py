"""Kinematics functions for the robot arm."""

from typing import List, Tuple
import numpy as np

from .arm import RobotArm


def forward_kinematics(arm: RobotArm) -> Tuple[np.ndarray, np.ndarray]:
    """Compute forward kinematics for the robot arm.
    
    Args:
        arm: RobotArm instance with current joint angles.
        
    Returns:
        Tuple of (joint_positions, end_effector_position):
            - joint_positions: Nx2 array of joint positions (including base).
            - end_effector_position: 1x2 array of end-effector position.
    """
    num_links = len(arm.links)
    # N+1 positions: base + one for each link end
    positions = np.zeros((num_links + 1, 2))
    
    # Base position
    positions[0] = arm.base_position
    
    # Current angle relative to world frame
    current_angle = 0.0
    
    for i, link in enumerate(arm.links):
        # Accumulate angle (each joint angle is relative to previous link)
        current_angle += link.angle
        
        # Compute end position of this link
        dx = link.length * np.cos(current_angle)
        dy = link.length * np.sin(current_angle)
        
        positions[i + 1] = positions[i] + [dx, dy]
    
    end_effector = positions[-1].copy()
    
    return positions, end_effector


def compute_joint_angles(positions: np.ndarray) -> List[float]:
    """Compute absolute joint angles from positions (for visualization).
    
    Args:
        positions: Nx2 array of joint positions.
        
    Returns:
        List of absolute angles in radians.
    """
    angles = []
    for i in range(len(positions) - 1):
        dx = positions[i + 1, 0] - positions[i, 0]
        dy = positions[i + 1, 1] - positions[i, 1]
        angle = np.arctan2(dy, dx)
        angles.append(angle)
    return angles
