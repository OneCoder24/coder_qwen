# RoboArm Playground

A 2D simulator for planar robot manipulators with forward and inverse kinematics, visualization, trajectory planning, and interactive control.

## Features

- **Robot Arm Model**: Configurable number of links with lengths and joint angle limits
- **Forward Kinematics**: Compute end-effector position from joint angles
- **Inverse Kinematics**: Calculate joint angles to reach a target (CCD algorithm)
- **Visualization**: Static plots and animations using matplotlib
- **Interactive Mode**: Click to set targets and watch the arm move in real-time
- **Smooth Motion**: Velocity-limited interpolation between configurations
- **Trajectory Planning**: Follow lines, circles, and figure-8 patterns with smooth motion
- **Joint Limits**: Enforce minimum/maximum angles per joint

## Installation

Requires Python 3.11+

```bash
pip install -r requirements.txt
```

## Quick Start

### Run Static Demo

Display a static arm configuration:

```bash
python main.py
```

### Run Animation Demo

See the arm move through a sequence of poses:

```bash
python main.py animate
```

### Run Inverse Kinematics Demo

Compute joint angles to reach a target point:

```bash
python main.py ik
```

This will:
1. Set up a 3-link arm with all joints at 0°
2. Define a target point at (2.0, 1.0)
3. Use CCD algorithm to compute required joint angles
4. Display the result with matplotlib

### Run Interactive Mode

Open an interactive window where you can click to set targets:

```bash
python main.py interactive
```

**Controls:**
- **Click** anywhere in the window to set a new target position
- **C** key: Clear the end-effector trail
- **R** key: Reset arm to home position (all angles = 0)
- **Close window**: Exit

The arm will smoothly interpolate towards the computed IK solution, respecting joint limits if defined.

### Run Trajectory Demo

Watch the arm follow a circular trajectory with smooth motion:

```bash
python main.py trajectory
```

### Run Joint Limits Demo

See how joint limits are enforced when reaching for a target:

```bash
python main.py limits
```

### Run Tests

```bash
pytest tests/ -v
```

## Project Structure

```
roboarm-playground/
├── roboarm/
│   ├── __init__.py      # Package init
│   ├── arm.py           # Robot arm model (Link, RobotArm)
│   ├── kinematics.py    # Forward/inverse kinematics (FK, CCD-IK)
│   ├── planner.py       # Trajectory generation & smooth motion
│   └── viz.py           # Visualization & interactive mode
├── tests/
│   ├── test_kinematics.py
│   └── test_planner.py
├── main.py              # Demo entry point
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore
```

## Usage Examples

### Create a Robot Arm

```python
from roboarm.arm import RobotArm, Link

# Create default 3-link arm
arm = RobotArm.create_default()

# Or create custom arm with joint limits
links = [
    Link(length=2.0, min_angle=-np.pi/2, max_angle=np.pi/2),
    Link(length=1.5, min_angle=-np.pi, max_angle=np.pi),
    Link(length=1.0),
]
arm = RobotArm(links=links, base_position=(0, 0))
```

### Inverse Kinematics

```python
from roboarm.kinematics import inverse_kinematics_ccd

# Set initial angles
arm.set_angles([0, 0, 0])

# Define target
target = np.array([2.0, 1.0])

# Compute joint angles using CCD
angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)

# Apply and verify
arm.set_angles(angles)
positions, end_effector = forward_kinematics(arm)
print(f"Distance to target: {np.linalg.norm(end_effector - target)}")
```

**Algorithm**: Currently implements Cyclic Coordinate Descent (CCD):
- Iteratively adjusts each joint from last to first
- Minimizes distance between end-effector and target
- Respects joint angle limits if defined
- Handles unreachable targets by extending fully toward the target

### Generate Trajectories

```python
from roboarm.planner import (
    generate_circle_trajectory,
    generate_line_trajectory,
    generate_figure8_trajectory,
    smooth_trajectory_follow,
)

# Circular trajectory
center = np.array([1.5, 0.0])
radius = 0.8
trajectory = generate_circle_trajectory(center, radius, num_points=30)

# Line trajectory
start = np.array([2.0, 0.0])
end = np.array([1.0, 1.0])
line_traj = generate_line_trajectory(start, end, num_points=20)

# Figure-8 trajectory
figure8 = generate_figure8_trajectory(
    center=np.array([0.0, 0.0]),
    width=1.0,
    height=0.5,
    num_points=50
)

# Follow trajectory with smooth, velocity-limited motion
angles_sequence = smooth_trajectory_follow(
    arm, trajectory,
    max_velocity=3.0,  # rad/s
    dt=0.02            # time step
)
```

### Visualize

```python
from roboarm.viz import plot_arm

plot_arm(arm, target=(2.0, 1.0))
```

### Interactive Mode with Joint Limits

```python
from roboarm.viz import interactive_arm

# Define joint limits (min, max) for each joint in radians
joint_limits = [
    (-np.pi/3, np.pi/3),    # Joint 0: ±60°
    (-np.pi/4, np.pi/2),    # Joint 1: -45° to 90°
    (-np.pi/2, np.pi/2),    # Joint 2: ±90°
]

interactive_arm(joint_limits=joint_limits)
```

## Development

This project uses GitHub Flow:
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit
3. Push and create a Pull Request
4. Wait for review and merge

## License

MIT License
