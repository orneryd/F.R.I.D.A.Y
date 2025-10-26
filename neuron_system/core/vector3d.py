"""
3D Vector implementation for neuron positioning.
"""

from dataclasses import dataclass
import math
from typing import Tuple


@dataclass
class Vector3D:
    """Represents a 3D position in space."""
    
    x: float
    y: float
    z: float
    
    def distance(self, other: 'Vector3D') -> float:
        """
        Calculate Euclidean distance to another vector.
        
        Args:
            other: Target vector
            
        Returns:
            Euclidean distance
        """
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )
    
    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.
        
        Returns:
            Vector magnitude
        """
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
    
    def normalize(self) -> 'Vector3D':
        """
        Return a normalized (unit length) version of this vector.
        
        Returns:
            Normalized vector
        """
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def dot(self, other: 'Vector3D') -> float:
        """
        Calculate dot product with another vector.
        
        Args:
            other: Target vector
            
        Returns:
            Dot product
        """
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def __add__(self, other: 'Vector3D') -> 'Vector3D':
        """Add two vectors."""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Vector3D') -> 'Vector3D':
        """Subtract two vectors."""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Vector3D':
        """Multiply vector by scalar."""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple representation."""
        return (self.x, self.y, self.z)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Vector3D':
        """Create Vector3D from dictionary."""
        return cls(x=data["x"], y=data["y"], z=data["z"])
