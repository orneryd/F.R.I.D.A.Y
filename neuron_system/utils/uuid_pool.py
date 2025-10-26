"""
UUID pre-allocation pool for instant ID assignment.
"""

from uuid import UUID, uuid4
from typing import List
from threading import Lock


class UUIDPool:
    """
    Pre-allocates UUIDs for instant assignment during neuron creation.
    
    Eliminates UUID generation overhead during high-frequency operations.
    """
    
    def __init__(self, initial_size: int = 10000, refill_threshold: int = 1000):
        """
        Initialize UUID pool.
        
        Args:
            initial_size: Number of UUIDs to pre-allocate initially
            refill_threshold: Refill pool when size drops below this threshold
        """
        self._pool: List[UUID] = []
        self._lock = Lock()
        self._initial_size = initial_size
        self._refill_threshold = refill_threshold
        self._allocated_count = 0
        
        # Pre-allocate initial UUIDs
        self._refill(initial_size)
    
    def _refill(self, count: int):
        """
        Refill the pool with new UUIDs.
        
        Args:
            count: Number of UUIDs to generate
        """
        new_uuids = [uuid4() for _ in range(count)]
        self._pool.extend(new_uuids)
    
    def acquire(self) -> UUID:
        """
        Acquire a UUID from the pool.
        
        Automatically refills pool when it drops below threshold.
        
        Returns:
            UUID instance
        """
        with self._lock:
            # Refill if below threshold
            if len(self._pool) < self._refill_threshold:
                self._refill(self._initial_size)
            
            # Get UUID from pool
            if self._pool:
                uuid = self._pool.pop()
                self._allocated_count += 1
                return uuid
            else:
                # Fallback: generate on demand if pool is empty
                self._allocated_count += 1
                return uuid4()
    
    def get_pool_size(self) -> int:
        """
        Get current pool size.
        
        Returns:
            Number of UUIDs in pool
        """
        with self._lock:
            return len(self._pool)
    
    def get_allocated_count(self) -> int:
        """
        Get total number of UUIDs allocated.
        
        Returns:
            Total allocation count
        """
        with self._lock:
            return self._allocated_count
    
    def clear(self):
        """Clear all UUIDs from the pool."""
        with self._lock:
            self._pool.clear()
    
    def get_stats(self) -> dict:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                "pool_size": len(self._pool),
                "allocated_count": self._allocated_count,
                "refill_threshold": self._refill_threshold,
            }
