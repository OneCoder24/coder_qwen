"""Robot arm model definition."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Link:
    """A single link of the robot arm.
    
    Attributes:
        length: Length of the link.
        angle: Current joint angle in radians.
        min_angle: Minimum allowed angle in radians (optional).
        max_angle: Maximum allowed angle in radians (optional).
        angle_limit_min: Alias for min_angle (deprecated, use min_angle).
        angle_limit_max: Alias for max_angle (deprecated, use max_angle).
    """
    length: float
    angle: float = 0.0
    min_angle: Optional[float] = None
    max_angle: Optional[float] = None
    
    @property
    def angle_limit_min(self) -> Optional[float]:
        """Deprecated alias for min_angle."""
        return self.min_angle
    
    @property
    def angle_limit_max(self) -> Optional[float]:
        """Deprecated alias for max_angle."""
        return self.max_angle
    
    def validate_angle(self) -> bool:
        """Check if current angle is within limits."""
        if self.min_angle is not None and self.angle < self.min_angle:
            return False
        if self.max_angle is not None and self.angle > self.max_angle:
            return False
        return True
    
    def clamp_angle(self) -> None:
        """Clamp angle to valid range if limits are set."""
        if self.min_angle is not None and self.angle < self.min_angle:
            self.angle = self.min_angle
        if self.max_angle is not None and self.angle > self.max_angle:
            self.angle = self.max_angle


class RobotArm:
    """Planar robot manipulator model.
    
    Attributes:
        links: List of Link objects.
        base_position: (x, y) coordinates of the base mount point.
    """
    
    def __init__(
        self,
        links: List[Link],
        base_position: Tuple[float, float] = (0.0, 0.0)
    ):
        self.links = links
        self.base_position = tuple(base_position)
    
    @classmethod
    def create_default(cls) -> "RobotArm":
        """Create a default 3-link arm with unit lengths."""
        links = [
            Link(length=1.0, angle=0.0),
            Link(length=1.0, angle=0.0),
            Link(length=1.0, angle=0.0),
        ]
        return cls(links=links, base_position=(0.0, 0.0))
    
    def set_angles(self, angles: List[float]) -> None:
        """Set all joint angles.
        
        Args:
            angles: List of angles in radians.
            
        Raises:
            ValueError: If number of angles doesn't match number of links.
        """
        if len(angles) != len(self.links):
            raise ValueError(
                f"Expected {len(self.links)} angles, got {len(angles)}"
            )
        for link, angle in zip(self.links, angles):
            link.angle = angle
    
    def get_angles(self) -> List[float]:
        """Get current joint angles."""
        return [link.angle for link in self.links]
    
    def validate_all_angles(self) -> bool:
        """Check if all joint angles are within their limits."""
        return all(link.validate_angle() for link in self.links)
    
    def clamp_all_angles(self) -> None:
        """Clamp all joint angles to their valid ranges."""
        for link in self.links:
            link.clamp_angle()
    
    def get_total_length(self) -> float:
        """Get the maximum reach of the arm (sum of all link lengths)."""
        return sum(link.length for link in self.links)
