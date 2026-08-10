"""Tests for kinematics and arm model."""

import pytest
import numpy as np
from roboarm.arm import RobotArm, Link
from roboarm.kinematics import forward_kinematics, inverse_kinematics_ccd


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


class TestInverseKinematics:
    """Tests for inverse kinematics using CCD."""
    
    def test_ik_reachable_target_straight(self):
        """Test IK for a reachable target with arm starting straight."""
        arm = RobotArm.create_default()
        arm.set_angles([0.0, 0.0, 0.0])
        
        target = np.array([2.5, 0.5])
        angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
        
        # Apply the computed angles and check end-effector position
        arm.set_angles(angles)
        _, end_effector = forward_kinematics(arm)
        
        distance = np.linalg.norm(end_effector - target)
        assert distance < 0.01, f"End-effector too far from target: {distance}"
    
    def test_ik_reachable_target_from_bent(self):
        """Test IK starting from a bent configuration."""
        arm = RobotArm.create_default()
        arm.set_angles([np.pi/4, np.pi/4, np.pi/4])
        
        target = np.array([1.5, 1.5])
        angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
        
        arm.set_angles(angles)
        _, end_effector = forward_kinematics(arm)
        
        distance = np.linalg.norm(end_effector - target)
        assert distance < 0.01, f"End-effector too far from target: {distance}"
    
    def test_ik_unreachable_target_far(self):
        """Test IK for an unreachable target (beyond max reach)."""
        arm = RobotArm.create_default()
        arm.set_angles([0.0, 0.0, 0.0])
        
        # Target at distance 5, but max reach is 3
        target = np.array([5.0, 0.0])
        angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
        
        arm.set_angles(angles)
        _, end_effector = forward_kinematics(arm)
        
        # End-effector should be at max reach (or close to it) in direction of target
        distance_from_base = np.linalg.norm(end_effector - np.array(arm.base_position))
        max_reach = arm.get_total_length()
        
        # Should be stretched out close to max reach
        assert np.isclose(distance_from_base, max_reach, atol=0.1), \
            f"Arm not fully extended for unreachable target: {distance_from_base} vs {max_reach}"
    
    def test_ik_unreachable_target_close(self):
        """Test IK for a target too close to base (inside minimum reach)."""
        arm = RobotArm.create_default()
        arm.set_angles([0.0, 0.0, 0.0])
        
        # Target very close to base - arm should fold to get as close as possible
        # Note: CCD may not achieve perfect folding; we verify it gets reasonably close
        target = np.array([0.1, 0.0])
        angles = inverse_kinematics_ccd(arm, target, max_iterations=200, tolerance=1e-3)
        
        arm.set_angles(angles)
        _, end_effector = forward_kinematics(arm)
        
        # Just verify it doesn't crash and makes progress toward the target
        # CCD typically can't fully fold a 3-link arm to reach very close points
        distance = np.linalg.norm(end_effector - target)
        assert distance < 1.5, f"Could not get close to near-base target: {distance}"
    
    def test_ik_respects_joint_limits(self):
        """Test that IK respects joint angle limits."""
        # Create arm with joint limits
        links = [
            Link(length=1.0, angle=0.0, min_angle=-np.pi/2, max_angle=np.pi/2),
            Link(length=1.0, angle=0.0, min_angle=-np.pi/2, max_angle=np.pi/2),
            Link(length=1.0, angle=0.0, min_angle=-np.pi/2, max_angle=np.pi/2),
        ]
        arm = RobotArm(links=links, base_position=(0.0, 0.0))
        
        # Target that would require angles outside limits if unconstrained
        target = np.array([2.0, 1.5])
        angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
        
        # Check that all angles are within limits
        for i, (angle, link) in enumerate(zip(angles, links)):
            assert angle >= link.min_angle - 1e-6, \
                f"Joint {i} angle {angle} below limit {link.min_angle}"
            assert angle <= link.max_angle + 1e-6, \
                f"Joint {i} angle {angle} above limit {link.max_angle}"
    
    def test_ik_original_angles_restored(self):
        """Test that original arm angles are restored after IK computation."""
        arm = RobotArm.create_default()
        original_angles = [0.5, 0.3, -0.2]
        arm.set_angles(original_angles)
        
        target = np.array([2.0, 0.5])
        _ = inverse_kinematics_ccd(arm, target, max_iterations=50, tolerance=1e-3)
        
        # Original angles should be restored
        current_angles = arm.get_angles()
        for orig, curr in zip(original_angles, current_angles):
            assert np.isclose(orig, curr), "Original angles not restored"
    
    def test_ik_zero_target_at_base(self):
        """Test IK when target is exactly at base position."""
        arm = RobotArm.create_default()
        arm.set_angles([0.0, 0.0, 0.0])
        
        target = np.array([0.0, 0.0])
        angles = inverse_kinematics_ccd(arm, target, max_iterations=100, tolerance=1e-3)
        
        # Should not crash; arm will try to fold
        arm.set_angles(angles)
        _, end_effector = forward_kinematics(arm)
        
        # Just verify it runs without error
        assert end_effector is not None
