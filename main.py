"""Main demo script for RoboArm Playground."""

import numpy as np
from roboarm.arm import RobotArm
from roboarm.kinematics import forward_kinematics
from roboarm.viz import plot_arm, create_animation


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


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "animate":
        demo_animation()
    else:
        demo_static()
