"""Trajectory planning and smooth motion functions."""

from typing import List, Tuple, Optional
import numpy as np

from .arm import RobotArm
from .kinematics import forward_kinematics, inverse_kinematics_ccd


def interpolate_angles(
    start_angles: np.ndarray,
    end_angles: np.ndarray,
    num_steps: int,
    profile: str = "linear"
) -> np.ndarray:
    """Generate interpolated angles between start and end configurations.
    
    Args:
        start_angles: Starting joint angles (M,).
        end_angles: Ending joint angles (M,).
        num_steps: Number of interpolation steps.
        profile: Interpolation profile - "linear" or "ease".
        
    Returns:
        NxM array of interpolated angles.
    """
    if profile == "linear":
        t = np.linspace(0, 1, num_steps)
        angles = np.zeros((num_steps, len(start_angles)))
        for i in range(num_steps):
            angles[i] = start_angles + (end_angles - start_angles) * t[i]
        return angles
    
    elif profile == "ease":
        # Ease-in-out using smoothstep
        t = np.linspace(0, 1, num_steps)
        t_smooth = t * t * (3 - 2 * t)  # Smoothstep
        angles = np.zeros((num_steps, len(start_angles)))
        for i in range(num_steps):
            angles[i] = start_angles + (end_angles - start_angles) * t_smooth[i]
        return angles
    
    else:
        raise ValueError(f"Unknown profile: {profile}")


def limit_angle_velocity(
    current_angles: np.ndarray,
    target_angles: np.ndarray,
    max_velocity: float,
    dt: float = 0.01
) -> np.ndarray:
    """Limit angle change per step to ensure smooth motion.
    
    Args:
        current_angles: Current joint angles.
        target_angles: Target joint angles.
        max_velocity: Maximum angular velocity (rad/s).
        dt: Time step (s).
        
    Returns:
        New angles respecting velocity limits.
    """
    delta = target_angles - current_angles
    max_delta = max_velocity * dt
    
    # Clip each joint's delta
    clipped_delta = np.clip(delta, -max_delta, max_delta)
    
    return current_angles + clipped_delta


def generate_line_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    num_points: int
) -> np.ndarray:
    """Generate a straight line trajectory.
    
    Args:
        start: Starting point (2,).
        end: Ending point (2,).
        num_points: Number of points on the line.
        
    Returns:
        Nx2 array of waypoints.
    """
    t = np.linspace(0, 1, num_points)
    trajectory = np.zeros((num_points, 2))
    for i in range(num_points):
        trajectory[i] = start + (end - start) * t[i]
    return trajectory


def generate_circle_trajectory(
    center: np.ndarray,
    radius: float,
    num_points: int,
    start_angle: float = 0.0,
    end_angle: float = 2 * np.pi
) -> np.ndarray:
    """Generate a circular trajectory.
    
    Args:
        center: Center of circle (2,).
        radius: Circle radius.
        num_points: Number of points.
        start_angle: Starting angle (radians).
        end_angle: Ending angle (radians).
        
    Returns:
        Nx2 array of waypoints.
    """
    angles = np.linspace(start_angle, end_angle, num_points)
    trajectory = np.zeros((num_points, 2))
    for i in range(num_points):
        trajectory[i, 0] = center[0] + radius * np.cos(angles[i])
        trajectory[i, 1] = center[1] + radius * np.sin(angles[i])
    return trajectory


def generate_figure8_trajectory(
    center: np.ndarray,
    width: float,
    height: float,
    num_points: int
) -> np.ndarray:
    """Generate a figure-8 (lemniscate) trajectory.
    
    Args:
        center: Center point (2,).
        width: Width of the figure.
        height: Height of the figure.
        num_points: Number of points.
        
    Returns:
        Nx2 array of waypoints.
    """
    t = np.linspace(0, 2 * np.pi, num_points)
    trajectory = np.zeros((num_points, 2))
    for i in range(num_points):
        # Parametric equation for figure-8
        trajectory[i, 0] = center[0] + width * np.sin(t[i])
        trajectory[i, 1] = center[1] + height * np.sin(t[i]) * np.cos(t[i])
    return trajectory


def follow_trajectory(
    arm: RobotArm,
    trajectory: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    return_angles: bool = True
) -> Optional[np.ndarray]:
    """Compute joint angles to follow a trajectory using IK.
    
    Args:
        arm: RobotArm instance.
        trajectory: Nx2 array of waypoints.
        max_iterations: Max CCD iterations per waypoint.
        tolerance: IK convergence tolerance.
        return_angles: If True, return angles sequence; otherwise modify arm in place.
        
    Returns:
        If return_angles is True: MxJ array of angles (M waypoints, J joints).
        Otherwise None (arm is modified in place for each waypoint).
    """
    num_waypoints = len(trajectory)
    num_joints = len(arm.links)
    
    original_angles = np.array(arm.get_angles())
    angles_sequence = np.zeros((num_waypoints, num_joints))
    
    for i, waypoint in enumerate(trajectory):
        angles = inverse_kinematics_ccd(
            arm, waypoint,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        angles_sequence[i] = angles
        
        # Use current solution as starting point for next waypoint
        # This helps with continuity
        arm.set_angles(angles)
    
    # Restore original angles
    arm.set_angles(original_angles.tolist())
    
    if return_angles:
        return angles_sequence
    return None


def smooth_trajectory_follow(
    arm: RobotArm,
    trajectory: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-3,
    max_velocity: float = 2.0,
    dt: float = 0.01
) -> np.ndarray:
    """Follow trajectory with velocity-limited smooth motion.
    
    Args:
        arm: RobotArm instance.
        trajectory: Nx2 array of waypoints.
        max_iterations: Max CCD iterations per waypoint.
        tolerance: IK convergence tolerance.
        max_velocity: Maximum joint angular velocity (rad/s).
        dt: Time step for velocity limiting (s).
        
    Returns:
        MxJ array of smoothed angles (M may be > N due to interpolation).
    """
    # First get raw IK solutions
    raw_angles = follow_trajectory(
        arm, trajectory,
        max_iterations=max_iterations,
        tolerance=tolerance,
        return_angles=True
    )
    
    # Now interpolate with velocity limiting
    all_angles = [raw_angles[0]]
    current_angles = raw_angles[0].copy()
    
    for i in range(1, len(raw_angles)):
        target_angles = raw_angles[i]
        
        # Interpolate between current and target with velocity limit
        while np.linalg.norm(target_angles - current_angles) > 1e-6:
            new_angles = limit_angle_velocity(
                current_angles, target_angles, max_velocity, dt
            )
            all_angles.append(new_angles)
            current_angles = new_angles
            
            if np.allclose(current_angles, target_angles, atol=1e-4):
                break
    
    return np.array(all_angles)
