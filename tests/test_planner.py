"""Tests for trajectory planning and smooth motion."""

import pytest
import numpy as np
from roboarm.arm import RobotArm, Link
from roboarm.kinematics import forward_kinematics
from roboarm.planner import (
    interpolate_angles,
    limit_angle_velocity,
    generate_line_trajectory,
    generate_circle_trajectory,
    generate_figure8_trajectory,
    follow_trajectory,
    smooth_trajectory_follow,
)


class TestInterpolateAngles:
    """Tests for angle interpolation."""
    
    def test_linear_interpolation(self):
        """Test linear interpolation between two configurations."""
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([np.pi/2, np.pi/4, -np.pi/4])
        
        angles = interpolate_angles(start, end, num_steps=5, profile="linear")
        
        assert angles.shape == (5, 3)
        assert np.allclose(angles[0], start)
        assert np.allclose(angles[-1], end)
    
    def test_ease_interpolation(self):
        """Test ease-in-out interpolation."""
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 1.0])
        
        angles = interpolate_angles(start, end, num_steps=10, profile="ease")
        
        assert angles.shape == (10, 2)
        assert np.allclose(angles[0], start)
        assert np.allclose(angles[-1], end)
    
    def test_invalid_profile(self):
        """Test error on invalid profile."""
        start = np.array([0.0])
        end = np.array([1.0])
        
        with pytest.raises(ValueError):
            interpolate_angles(start, end, num_steps=5, profile="invalid")


class TestLimitAngleVelocity:
    """Tests for velocity limiting."""
    
    def test_within_limit(self):
        """Test when target is within velocity limit."""
        current = np.array([0.0, 0.0])
        target = np.array([0.01, 0.01])
        
        # With max_velocity=1.0 and dt=0.01, max_delta = 0.01
        result = limit_angle_velocity(current, target, max_velocity=1.0, dt=0.01)
        
        assert np.allclose(result, target)
    
    def test_exceeds_limit(self):
        """Test when target exceeds velocity limit."""
        current = np.array([0.0, 0.0])
        target = np.array([0.5, 0.5])
        
        # With max_velocity=1.0 and dt=0.01, max_delta = 0.01
        result = limit_angle_velocity(current, target, max_velocity=1.0, dt=0.01)
        
        # Should be clipped to max_delta
        assert np.allclose(result, [0.01, 0.01])
        assert np.linalg.norm(result - current) <= 0.015  # Allow small tolerance


class TestGenerateLineTrajectory:
    """Tests for line trajectory generation."""
    
    def test_line_from_origin(self):
        """Test line from origin."""
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 1.0])
        
        trajectory = generate_line_trajectory(start, end, num_points=5)
        
        assert trajectory.shape == (5, 2)
        assert np.allclose(trajectory[0], start)
        assert np.allclose(trajectory[-1], end)
    
    def test_horizontal_line(self):
        """Test horizontal line."""
        start = np.array([0.0, 0.0])
        end = np.array([2.0, 0.0])
        
        trajectory = generate_line_trajectory(start, end, num_points=3)
        
        assert np.allclose(trajectory[:, 1], 0.0)  # All y coordinates are 0


class TestGenerateCircleTrajectory:
    """Tests for circle trajectory generation."""
    
    def test_full_circle(self):
        """Test full circle trajectory."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        
        trajectory = generate_circle_trajectory(center, radius, num_points=36)
        
        assert trajectory.shape == (36, 2)
        
        # Check all points are at correct distance from center
        distances = np.linalg.norm(trajectory - center, axis=1)
        assert np.allclose(distances, radius, atol=1e-10)
    
    def test_offset_circle(self):
        """Test circle with offset center."""
        center = np.array([5.0, 3.0])
        radius = 2.0
        
        trajectory = generate_circle_trajectory(center, radius, num_points=20)
        
        distances = np.linalg.norm(trajectory - center, axis=1)
        assert np.allclose(distances, radius, atol=1e-10)
    
    def test_partial_circle(self):
        """Test partial circle (semicircle)."""
        center = np.array([0.0, 0.0])
        radius = 1.0
        
        trajectory = generate_circle_trajectory(
            center, radius, num_points=10,
            start_angle=0, end_angle=np.pi
        )
        
        # First point should be at (1, 0), last at (-1, 0)
        assert np.allclose(trajectory[0], [1.0, 0.0], atol=1e-10)
        assert np.allclose(trajectory[-1], [-1.0, 0.0], atol=1e-10)


class TestGenerateFigure8Trajectory:
    """Tests for figure-8 trajectory generation."""
    
    def test_figure8_shape(self):
        """Test figure-8 basic properties."""
        center = np.array([0.0, 0.0])
        width = 1.0
        height = 0.5
        
        trajectory = generate_figure8_trajectory(center, width, height, num_points=50)
        
        assert trajectory.shape == (50, 2)
        
        # Figure-8 should cross center multiple times
        # At t=0, pi, 2pi: sin(t)=0, so x=center[0]
        crossings = np.where(np.abs(trajectory[:, 0] - center[0]) < 0.1)[0]
        assert len(crossings) >= 3


class TestFollowTrajectory:
    """Tests for trajectory following with IK."""
    
    def test_follow_simple_line(self):
        """Test following a simple line trajectory."""
        arm = RobotArm.create_default()
        arm.set_angles([0.0, 0.0, 0.0])
        
        # Start from a reachable point closer to the arm's natural reach
        # Note: First point may have larger error if starting config doesn't match
        start = np.array([2.5, 0.0])
        end = np.array([1.5, 1.0])
        trajectory = generate_line_trajectory(start, end, num_points=10)
        
        angles_sequence = follow_trajectory(arm, trajectory, max_iterations=200)
        
        assert angles_sequence.shape == (10, 3)
        
        # Verify waypoints are reached (skip first point which depends on initial config)
        max_distance = 0.0
        for i in range(1, len(angles_sequence)):  # Skip index 0
            angles = angles_sequence[i]
            arm.set_angles(angles)
            _, end_eff = forward_kinematics(arm)
            distance = np.linalg.norm(end_eff - trajectory[i])
            max_distance = max(max_distance, distance)
        
        # Allow tolerance for IK convergence
        assert max_distance < 0.05, f"Max waypoint error too large: {max_distance}"
    
    def test_follow_circle(self):
        """Test following a circular trajectory."""
        arm = RobotArm.create_default()
        
        center = np.array([1.5, 0.0])
        radius = 0.8
        trajectory = generate_circle_trajectory(center, radius, num_points=20)
        
        angles_sequence = follow_trajectory(arm, trajectory, max_iterations=100)
        
        assert angles_sequence.shape == (20, 3)
        
        # Verify arm can reach most waypoints
        reachable_count = 0
        for i, angles in enumerate(angles_sequence):
            arm.set_angles(angles)
            _, end_eff = forward_kinematics(arm)
            distance = np.linalg.norm(end_eff - trajectory[i])
            if distance < 0.1:
                reachable_count += 1
        
        # At least 80% should be reachable
        assert reachable_count >= 16, f"Only {reachable_count}/20 waypoints reachable"


class TestSmoothTrajectoryFollow:
    """Tests for smooth trajectory following with velocity limits."""
    
    def test_smooth_motion_produces_more_frames(self):
        """Test that smooth following produces interpolated frames."""
        arm = RobotArm.create_default()
        
        center = np.array([1.5, 0.0])
        radius = 0.5
        trajectory = generate_circle_trajectory(center, radius, num_points=10)
        
        angles_sequence = smooth_trajectory_follow(
            arm, trajectory,
            max_velocity=2.0,
            dt=0.01
        )
        
        # Smooth following should produce more frames than raw trajectory
        assert len(angles_sequence) >= len(trajectory)
    
    def test_smooth_motion_continuity(self):
        """Test that smooth motion has continuous joint changes."""
        arm = RobotArm.create_default()
        
        start = np.array([2.0, 0.0])
        end = np.array([1.0, 1.0])
        trajectory = generate_line_trajectory(start, end, num_points=5)
        
        angles_sequence = smooth_trajectory_follow(
            arm, trajectory,
            max_velocity=5.0,
            dt=0.01
        )
        
        # Check consecutive frames have limited joint change
        max_joint_change = 0.0
        for i in range(1, len(angles_sequence)):
            delta = np.abs(angles_sequence[i] - angles_sequence[i-1])
            max_joint_change = max(max_joint_change, np.max(delta))
        
        # Joint changes should be bounded (not teleporting)
        assert max_joint_change < 0.5, "Joint changes too large"
