"""Main demo script for RoboArm Playground."""

import numpy as np
from roboarm.arm import RobotArm, Link
from roboarm.kinematics import forward_kinematics, inverse_kinematics_ccd
from roboarm.viz import plot_arm, create_animation, interactive_arm
from roboarm.planner import (
    generate_circle_trajectory,
    smooth_trajectory_follow,
)


def demo_static():
    """Show static arm configuration."""
    print("=== Static Configuration Demo ===")
    
    arm = RobotArm.create_default()
    arm.set_angles([0.0, np.pi/4, -np.pi/8])
    
    positions, end_effector = forward_kinematics(arm)
    print(f"Joint positions:\n{positions}")
    print(f"End effector: {end_effector}")
    
    plot_arm(arm, target=(2.0, 1.0), show=True)


def demo_animation():
    """Show animated arm movement."""
    print("=== Animation Demo ===")
    
    arm = RobotArm.create_default()
    
    # Create a sequence of angles for smooth rotation
    num_frames = 60
    angles_sequence = []
    
    for i in range(num_frames):
        t = i / num_frames * 2 * np.pi
        angles = [
            np.sin(t) * 0.5,
            np.cos(t) * 0.3,
            np.sin(t * 2) * 0.4
        ]
        angles_sequence.append(angles)
    
    angles_sequence = np.array(angles_sequence)
    
    target = (1.5, 1.0)
    
    print(f"Creating animation with {num_frames} frames...")
    anim = create_animation(arm, angles_sequence, target=target, interval=50)
    
    print("Displaying animation (close window to exit)...")
    import matplotlib.pyplot as plt
    plt.show()


def demo_ik():
    """Demonstrate inverse kinematics."""
    print("=== Inverse Kinematics Demo ===")
    
    arm = RobotArm.create_default()
    arm.set_angles([0.0, 0.0, 0.0])
    
    # Define target point
    target = np.array([2.0, 1.0])
    print(f"Target: {target}")
    
    # Compute IK using CCD
    angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
    print(f"Computed angles: {[f'{a:.4f}' for a in angles]}")
    
    # Apply computed angles and verify
    arm.set_angles(angles)
    positions, end_effector = forward_kinematics(arm)
    
    distance = np.linalg.norm(end_effector - target)
    print(f"End effector position: {end_effector}")
    print(f"Distance to target: {distance:.6f}")
    
    # Plot result
    plot_arm(arm, target=tuple(target), show=True)
    
    print("\nIK test passed!" if distance < 0.01 else "\nIK test failed!")


def demo_interactive():
    """Run interactive mode."""
    print("Starting interactive mode...")
    interactive_arm()


def demo_trajectory():
    """Demonstrate trajectory following with smooth motion."""
    print("=== Trajectory Following Demo ===")
    
    arm = RobotArm.create_default()
    arm.set_angles([0.0, 0.0, 0.0])
    
    # Generate a circular trajectory
    center = np.array([1.5, 0.0])
    radius = 0.8
    num_points = 30
    
    print(f"Generating circle trajectory: center={center}, radius={radius}")
    trajectory = generate_circle_trajectory(center, radius, num_points)
    
    # Follow trajectory with smooth motion
    print("Computing smooth trajectory follow...")
    angles_sequence = smooth_trajectory_follow(
        arm, trajectory,
        max_iterations=50,
        tolerance=1e-3,
        max_velocity=3.0,
        dt=0.02
    )
    
    print(f"Generated {len(angles_sequence)} frames of smooth motion")
    
    # Create animation
    print("Creating animation...")
    anim = create_animation(arm, angles_sequence, target=None, interval=50)
    
    print("Displaying trajectory animation (close window to exit)...")
    import matplotlib.pyplot as plt
    plt.show()
    
    print("\nTrajectory demo complete!")


def demo_joint_limits():
    """Demonstrate joint limits enforcement."""
    print("=== Joint Limits Demo ===")
    
    # Create arm with joint limits
    links = [
        Link(length=1.0, angle=0.0, min_angle=-np.pi/3, max_angle=np.pi/3),
        Link(length=1.0, angle=0.0, min_angle=-np.pi/4, max_angle=np.pi/2),
        Link(length=1.0, angle=0.0, min_angle=-np.pi/2, max_angle=np.pi/2),
    ]
    arm = RobotArm(links=links, base_position=(0.0, 0.0))
    
    print("Joint limits:")
    for i, link in enumerate(arm.links):
        min_deg = np.degrees(link.min_angle) if link.min_angle else "N/A"
        max_deg = np.degrees(link.max_angle) if link.max_angle else "N/A"
        print(f"  Joint {i}: [{min_deg:.1f}°, {max_deg:.1f}°]")
    
    # Try to reach a target that would require exceeding limits
    target = np.array([2.0, 1.5])
    print(f"\nTarget: {target}")
    
    angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
    
    print(f"Computed angles: {[f'{np.degrees(a):.1f}°' for a in angles]}")
    
    # Verify limits are respected
    print("\nVerifying limits:")
    all_valid = True
    for i, (angle, link) in enumerate(zip(angles, arm.links)):
        min_ok = link.min_angle is None or angle >= link.min_angle - 1e-6
        max_ok = link.max_angle is None or angle <= link.max_angle + 1e-6
        valid = min_ok and max_ok
        status = "✓" if valid else "✗"
        print(f"  Joint {i}: {status} ({angle:.3f} rad)")
        if not valid:
            all_valid = False
    
    # Apply angles and show result
    arm.set_angles(angles)
    plot_arm(arm, target=tuple(target), show=True)
    
    if all_valid:
        print("\n✓ All joint limits respected!")
    else:
        print("\n✗ Some joint limits violated!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "animate":
            demo_animation()
        elif sys.argv[1] == "ik":
            demo_ik()
        elif sys.argv[1] == "interactive":
            demo_interactive()
        elif sys.argv[1] == "trajectory":
            demo_trajectory()
        elif sys.argv[1] == "limits":
            demo_joint_limits()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python main.py [animate|ik|interactive|trajectory|limits]")
    else:
        demo_static()
