"""Kinematics functions for the robot arm."""

from typing import List, Tuple, Optional
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


def inverse_kinematics_ccd(
    arm: RobotArm,
    target: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    step_limit: Optional[float] = None
) -> List[float]:
    """Solve inverse kinematics using Cyclic Coordinate Descent (CCD).
    
    CCD iteratively adjusts each joint angle to minimize the distance
    between the end-effector and the target point.
    
    Args:
        arm: RobotArm instance (angles will be modified in place during iteration,
             but original angles are restored; result is returned).
        target: 2D target position as [x, y].
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance (distance to target).
        step_limit: Maximum angle change per iteration (radians). 
                    If None, no limit.
    
    Returns:
        List of joint angles that bring the end-effector close to the target.
        If the target is unreachable, returns angles that minimize the distance.
    """
    target = np.asarray(target, dtype=float)
    num_links = len(arm.links)
    
    # Store original angles to restore later
    original_angles = arm.get_angles()
    
    # Check if target is reachable at all
    total_length = arm.get_total_length()
    base_pos = np.array(arm.base_position)
    dist_to_target = np.linalg.norm(target - base_pos)
    
    # If target is beyond reach, we'll still try to get as close as possible
    # CCD naturally handles this by minimizing distance
    
    best_angles = original_angles.copy()
    best_distance = float('inf')
    
    for iteration in range(max_iterations):
        # Process joints from last to first (excluding base)
        for i in range(num_links - 1, -1, -1):
            # Compute current FK
            positions, end_effector = forward_kinematics(arm)
            
            current_dist = np.linalg.norm(end_effector - target)
            
            # Track best solution found
            if current_dist < best_distance:
                best_distance = current_dist
                best_angles = arm.get_angles().copy()
                
                # Early exit if converged
                if current_dist < tolerance:
                    arm.set_angles(original_angles)
                    return best_angles
            
            # Position of current joint
            joint_pos = positions[i]
            
            # Vector from joint to end-effector
            vec_to_end = end_effector - joint_pos
            vec_to_end_norm = np.linalg.norm(vec_to_end)
            
            if vec_to_end_norm < 1e-10:
                # End effector is at joint position, skip
                continue
            
            vec_to_end = vec_to_end / vec_to_end_norm
            
            # Vector from joint to target
            vec_to_target = target - joint_pos
            vec_to_target_norm = np.linalg.norm(vec_to_target)
            
            if vec_to_target_norm < 1e-10:
                # Target is at joint position
                continue
            
            vec_to_target = vec_to_target / vec_to_target_norm
            
            # Compute angle needed to align end-effector with target
            # Using arctan2 for robust angle computation
            angle_end = np.arctan2(vec_to_end[1], vec_to_end[0])
            angle_target = np.arctan2(vec_to_target[1], vec_to_target[0])
            
            delta_angle = angle_target - angle_end
            
            # Normalize to [-pi, pi]
            while delta_angle > np.pi:
                delta_angle -= 2 * np.pi
            while delta_angle < -np.pi:
                delta_angle += 2 * np.pi
            
            # Apply step limit if set
            if step_limit is not None:
                if delta_angle > step_limit:
                    delta_angle = step_limit
                elif delta_angle < -step_limit:
                    delta_angle = -step_limit
            
            # Update the joint angle
            # Note: arm.links[i].angle is relative to previous link
            # We need to adjust it by delta_angle
            new_angle = arm.links[i].angle + delta_angle
            
            # Apply joint limits if they exist
            if arm.links[i].min_angle is not None:
                new_angle = max(new_angle, arm.links[i].min_angle)
            if arm.links[i].max_angle is not None:
                new_angle = min(new_angle, arm.links[i].max_angle)
            
            arm.links[i].angle = new_angle
        
        # Check final distance after full cycle
        _, end_effector = forward_kinematics(arm)
        final_dist = np.linalg.norm(end_effector - target)
        
        if final_dist < best_distance:
            best_distance = final_dist
            best_angles = arm.get_angles().copy()
        
        if final_dist < tolerance:
            break
    
    # Restore original angles and return best found
    arm.set_angles(original_angles)
    return best_angles
