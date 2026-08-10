"""Main demo script for RoboArm Playground."""

import numpy as np
from roboarm.arm import RobotArm
from roboarm.kinematics import forward_kinematics, inverse_kinematics_ccd
from roboarm.viz import plot_arm, create_animation, interactive_arm


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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "animate":
            demo_animation()
        elif sys.argv[1] == "ik":
            demo_ik()
        elif sys.argv[1] == "interactive":
            demo_interactive()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python main.py [animate|ik|interactive]")
    else:
        demo_static()
