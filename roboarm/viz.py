"""Visualization functions for the robot arm."""

from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backend_bases import MouseEvent

from .arm import RobotArm
from .kinematics import forward_kinematics, inverse_kinematics_ccd


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


def interactive_arm(
    arm: Optional[RobotArm] = None,
    max_iterations: int = 50,
    tolerance: float = 1e-2,
    smoothing_factor: float = 0.1
) -> None:
    """Interactive mode: click to set target, arm follows in real-time.
    
    Args:
        arm: RobotArm instance. If None, creates default arm.
        max_iterations: Max CCD iterations per click.
        tolerance: IK convergence tolerance.
        smoothing_factor: Factor for smooth angle interpolation (0-1).
    """
    if arm is None:
        arm = RobotArm.create_default()
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    total_length = arm.get_total_length()
    margin = 0.5
    limit = total_length + margin
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('RoboArm Playground - Click to set target')
    
    # Initialize plot elements
    line, = ax.plot([], [], 'b-', linewidth=2, label='Links')
    joints, = ax.plot([], [], 'ko', markersize=8, label='Joints')
    end_effector_point, = ax.plot([], [], 'ro', markersize=10, label='End Effector')
    target_point = ax.plot([], [], 'gx', markersize=14, markeredgewidth=2.5, label='Target')[0]
    trail_line, = ax.plot([], [], 'y--', linewidth=1, alpha=0.5, label='Trail')
    
    ax.legend(loc='upper right')
    
    # Store trail points
    trail_points: List[np.ndarray] = []
    current_target: Optional[np.ndarray] = None
    
    # Current angles for smooth interpolation
    current_angles = arm.get_angles().copy()
    
    def draw_arm(angles: np.ndarray, target: Optional[np.ndarray] = None):
        """Draw the arm with given angles."""
        arm.set_angles(angles.tolist())
        positions, end_eff = forward_kinematics(arm)
        
        line.set_data(positions[:, 0], positions[:, 1])
        joints.set_data(positions[:, 0], positions[:, 1])
        end_effector_point.set_data(end_eff[0], end_eff[1])
        
        if target is not None:
            target_point.set_data(target[0], target[1])
            target_point.set_visible(True)
        else:
            target_point.set_visible(False)
        
        if trail_points:
            trail_x = [p[0] for p in trail_points]
            trail_y = [p[1] for p in trail_points]
            trail_line.set_data(trail_x, trail_y)
        else:
            trail_line.set_data([], [])
        
        return line, joints, end_effector_point, target_point, trail_line
    
    def on_click(event: MouseEvent):
        """Handle mouse click to set new target."""
        if event.inaxes != ax:
            return
        
        nonlocal current_target, current_angles
        
        # Get clicked position
        new_target = np.array([event.xdata, event.ydata])
        current_target = new_target
        
        print(f"Target set to: ({new_target[0]:.2f}, {new_target[1]:.2f})")
        
        # Compute IK
        angles = inverse_kinematics_ccd(
            arm, new_target, 
            max_iterations=max_iterations, 
            tolerance=tolerance
        )
        
        # Smooth interpolation towards computed angles
        for step in range(10):
            t = (step + 1) / 10
            interpolated = current_angles + (angles - current_angles) * t * smoothing_factor
            interpolated = np.clip(interpolated, -np.pi, np.pi)
            
            # Apply joint limits if any
            for i, link in enumerate(arm.links):
                if link.angle_limit_min is not None:
                    interpolated[i] = max(interpolated[i], link.angle_limit_min)
                if link.angle_limit_max is not None:
                    interpolated[i] = min(interpolated[i], link.angle_limit_max)
            
            current_angles = interpolated
            draw_arm(current_angles, new_target)
            fig.canvas.draw_idle()
            plt.pause(0.02)
        
        current_angles = angles.copy()
        
        # Add end effector position to trail
        _, end_eff = forward_kinematics(arm)
        trail_points.append(end_eff.copy())
        
        # Limit trail length
        if len(trail_points) > 100:
            trail_points.pop(0)
        
        draw_arm(current_angles, new_target)
        fig.canvas.draw_idle()
    
    def on_key(event):
        """Handle key presses."""
        nonlocal trail_points
        if event.key == 'c' or event.key == 'C':
            # Clear trail
            trail_points = []
            trail_line.set_data([], [])
            fig.canvas.draw_idle()
            print("Trail cleared")
        elif event.key == 'r' or event.key == 'R':
            # Reset arm to home position
            current_angles[:] = 0.0
            arm.set_angles([0.0] * len(arm.links))
            trail_points = []
            current_target = None
            draw_arm(current_angles, None)
            fig.canvas.draw_idle()
            print("Arm reset to home position")
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial draw
    arm.set_angles([0.0] * len(arm.links))
    draw_arm(arm.get_angles(), None)
    
    print("\n=== Interactive Mode ===")
    print("- Click anywhere to set a target")
    print("- Press 'C' to clear the trail")
    print("- Press 'R' to reset arm to home position")
    print("- Close window to exit\n")
    
    plt.show()
