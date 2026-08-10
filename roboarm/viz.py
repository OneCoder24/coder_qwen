"""Visualization functions for the robot arm."""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .arm import RobotArm
from .kinematics import forward_kinematics


def plot_arm(
    arm: RobotArm,
    target: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True
) -> plt.Figure:
    """Plot the robot arm in its current configuration.
    
    Args:
        arm: RobotArm instance.
        target: Optional (x, y) target position to display.
        ax: Optional matplotlib axes to plot on.
        show: Whether to call plt.show().
        
    Returns:
        The matplotlib figure.
    """
    positions, end_effector = forward_kinematics(arm)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Plot links
    ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Links')
    
    # Plot joints
    ax.plot(positions[:, 0], positions[:, 1], 'ko', markersize=8, label='Joints')
    
    # Plot end effector
    ax.plot(end_effector[0], end_effector[1], 'ro', markersize=10, label='End Effector')
    
    # Plot target if provided
    if target is not None:
        ax.plot(target[0], target[1], 'gx', markersize=12, markeredgewidth=2, label='Target')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Arm Configuration')
    
    # Set limits based on arm reach
    total_length = arm.get_total_length()
    margin = 0.5
    limit = total_length + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    if show:
        plt.show()
    
    return fig


def create_animation(
    arm: RobotArm,
    angles_sequence: np.ndarray,
    target: Optional[Tuple[float, float]] = None,
    interval: int = 50,
    save_path: Optional[str] = None
) -> FuncAnimation:
    """Create an animation of the arm moving through a sequence of angles.
    
    Args:
        arm: RobotArm instance.
        angles_sequence: NxM array where N is number of frames and M is number of joints.
        target: Optional target position to display.
        interval: Delay between frames in milliseconds.
        save_path: Optional path to save the animation.
        
    Returns:
        FuncAnimation object.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    total_length = arm.get_total_length()
    margin = 0.5
    limit = total_length + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Robot Arm Animation')
    
    # Initialize plot elements
    line, = ax.plot([], [], 'b-', linewidth=2, label='Links')
    joints, = ax.plot([], [], 'ko', markersize=8, label='Joints')
    end_effector, = ax.plot([], [], 'ro', markersize=10, label='End Effector')
    
    plots = [line, joints, end_effector]
    
    if target is not None:
        target_plot = ax.plot(target[0], target[1], 'gx', markersize=12, markeredgewidth=2, label='Target')[0]
        plots.append(target_plot)
    
    ax.legend()
    
    def init():
        line.set_data([], [])
        joints.set_data([], [])
        end_effector.set_data([], [])
        return tuple(plots)
    
    def update(frame):
        angles = angles_sequence[frame]
        arm.set_angles(angles.tolist())
        positions, end_eff = forward_kinematics(arm)
        
        line.set_data(positions[:, 0], positions[:, 1])
        joints.set_data(positions[:, 0], positions[:, 1])
        end_effector.set_data(end_eff[0], end_eff[1])
        
        return tuple(plots)
    
    anim = FuncAnimation(
        fig, update, frames=len(angles_sequence),
        init_func=init, blit=True, interval=interval
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
    
    return anim
