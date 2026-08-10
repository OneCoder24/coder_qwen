# RoboArm Playground

A 2D simulator for planar robot manipulators with forward and inverse kinematics, visualization, and trajectory planning.

## Features

- **Robot Arm Model**: Configurable number of links with lengths and joint angle limits
- **Forward Kinematics**: Compute end-effector position from joint angles
- **Inverse Kinematics**: Calculate joint angles to reach a target (CCD/FABRIK)
- **Visualization**: Static plots and animations using matplotlib
- **Interactive Mode**: Click to set targets and watch the arm move
- **Smooth Motion**: Animated transitions between configurations

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
│   ├── kinematics.py    # Forward/inverse kinematics
│   ├── planner.py       # Trajectory planning (TODO)
│   └── viz.py           # Visualization functions
├── tests/
│   └── test_kinematics.py
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

# Or create custom arm
links = [
    Link(length=2.0, min_angle=-np.pi/2, max_angle=np.pi/2),
    Link(length=1.5, min_angle=-np.pi, max_angle=np.pi),
    Link(length=1.0),
]
arm = RobotArm(links=links, base_position=(0, 0))
```

### Forward Kinematics

```python
from roboarm.kinematics import forward_kinematics

arm.set_angles([0, np.pi/4, -np.pi/8])
positions, end_effector = forward_kinematics(arm)
print(f"End effector at: {end_effector}")
```

### Visualize

```python
from roboarm.viz import plot_arm

plot_arm(arm, target=(2.0, 1.0))
```

## Development

This project uses GitHub Flow:
1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit
3. Push and create a Pull Request
4. Wait for review and merge

## License

MIT License
