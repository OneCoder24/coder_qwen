"""Tests for kinematics and arm model."""

import pytest
import numpy as np
from roboarm.arm import RobotArm, Link
from roboarm.kinematics import forward_kinematics


class TestLink:
    """Tests for the Link dataclass."""
    
    def test_link_creation(self):
        """Test basic link creation."""
        link = Link(length=1.0, angle=0.5)
        assert link.length == 1.0
        assert link.angle == 0.5
    
    def test_link_with_limits(self):
        """Test link with angle limits."""
        link = Link(length=1.0, angle=0.5, min_angle=-1.0, max_angle=1.0)
        assert link.validate_angle() is True
        
        link.angle = 1.5
        assert link.validate_angle() is False
        
        link.angle = -1.5
        assert link.validate_angle() is False
    
    def test_clamp_angle(self):
        """Test angle clamping."""
        link = Link(length=1.0, angle=2.0, min_angle=-1.0, max_angle=1.0)
        link.clamp_angle()
        assert link.angle == 1.0
        
        link.angle = -2.0
        link.clamp_angle()
        assert link.angle == -1.0


class TestRobotArm:
    """Tests for the RobotArm class."""
    
    def test_create_default(self):
        """Test default arm creation."""
        arm = RobotArm.create_default()
        assert len(arm.links) == 3
        assert arm.base_position == (0.0, 0.0)
        for link in arm.links:
            assert link.length == 1.0
    
    def test_set_angles(self):
        """Test setting joint angles."""
        arm = RobotArm.create_default()
        angles = [0.5, 1.0, -0.5]
        arm.set_angles(angles)
        assert arm.get_angles() == angles
    
    def test_set_angles_wrong_count(self):
        """Test error on wrong number of angles."""
        arm = RobotArm.create_default()
        with pytest.raises(ValueError):
            arm.set_angles([0.5, 1.0])  # Only 2 angles for 3-link arm
    
    def test_total_length(self):
        """Test total length calculation."""
        arm = RobotArm.create_default()
        assert arm.get_total_length() == 3.0


class TestForwardKinematics:
    """Tests for forward kinematics."""
    
    def test_all_zeros(self):
        """Test FK with all zero angles (arm fully extended along X)."""
        arm = RobotArm.create_default()
        arm.set_angles([0.0, 0.0, 0.0])
        positions, end_effector = forward_kinematics(arm)
        
        # Base at origin
        assert positions[0, 0] == 0.0
        assert positions[0, 1] == 0.0
        
        # End effector at (3, 0) for 3 links of length 1
        assert np.isclose(end_effector[0], 3.0)
        assert np.isclose(end_effector[1], 0.0)
    
    def test_all_pi_half(self):
        """Test FK with all 90 degree angles."""
        arm = RobotArm.create_default()
        arm.set_angles([np.pi/2, np.pi/2, np.pi/2])
        positions, end_effector = forward_kinematics(arm)
        
        # Each link turns 90 degrees relative to previous
        # Link 1: angle=π/2 → (0, 1)
        # Link 2: angle=π → (-1, 1)
        # Link 3: angle=3π/2 → (-1, 0)
        assert np.isclose(positions[1, 0], 0.0)
        assert np.isclose(positions[1, 1], 1.0)
        
        assert np.isclose(positions[2, 0], -1.0)
        assert np.isclose(positions[2, 1], 1.0)
        
        assert np.isclose(end_effector[0], -1.0)
        assert np.isclose(end_effector[1], 0.0)
    
    def test_custom_base_position(self):
        """Test FK with non-zero base position."""
        links = [Link(length=1.0) for _ in range(3)]
        arm = RobotArm(links=links, base_position=(5.0, 5.0))
        arm.set_angles([0.0, 0.0, 0.0])
        positions, end_effector = forward_kinematics(arm)
        
        assert positions[0, 0] == 5.0
        assert positions[0, 1] == 5.0
        assert np.isclose(end_effector[0], 8.0)
        assert np.isclose(end_effector[1], 5.0)
    
    def test_num_positions(self):
        """Test that we get correct number of positions."""
        arm = RobotArm.create_default()
        arm.set_angles([0.1, 0.2, 0.3])
        positions, _ = forward_kinematics(arm)
        
        # Should have base + 3 link ends = 4 positions
        assert positions.shape == (4, 2)
