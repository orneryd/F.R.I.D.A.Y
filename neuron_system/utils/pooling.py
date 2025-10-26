"""
Object pooling implementation for memory efficiency.
"""

from typing import Any, Callable, Generic, List, TypeVar
from threading import Lock


T = TypeVar('T')


class ObjectPool(Generic[T]):
    """
    Generic object pool for reusing objects to minimize allocations.
    
    Useful for high-frequency object creation scenarios like neuron generation.
    """
    
    def __init__(self, factory: Callable[[], T], max_size: int = 1000):
        """
        Initialize object pool.
        
        Args:
            factory: Function that creates new objects
            max_size: Maximum number of objects to keep in pool
        """
        self._factory = factory
        self._max_size = max_size
        self._pool: List[T] = []
        self._lock = Lock()
        self._created_count = 0
        self._reused_count = 0
    
    def acquire(self) -> T:
        """
        Acquire an object from the pool.
        
        Returns a pooled object if available, otherwise creates a new one.
        
        Returns:
            Object instance
        """
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._reused_count += 1
                return obj
            else:
                obj = self._factory()
                self._created_count += 1
                return obj
    
    def release(self, obj: T):
        """
        Return an object to the pool for reuse.
        
        Args:
            obj: Object to return to pool
        """
        with self._lock:
            if len(self._pool) < self._max_size:
                self._pool.append(obj)
    
    def clear(self):
        """Clear all objects from the pool."""
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
                "max_size": self._max_size,
                "created_count": self._created_count,
                "reused_count": self._reused_count,
                "reuse_rate": self._reused_count / max(1, self._created_count + self._reused_count),
            }
