"""
Retry logic and fallback mechanisms for error handling.

Provides exponential backoff, circuit breaker, and fallback strategies.
"""

import time
import logging
from typing import Callable, Any, Optional, Type
from functools import wraps
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    Prevents cascading failures by stopping requests to a failing service
    and allowing it time to recover.
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker reset to CLOSED state")
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now(timezone.utc)
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


def exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for exponential backoff retry logic.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def with_fallback(fallback_value: Any = None, log_fallback: bool = True):
    """
    Decorator to provide fallback value on exception.
    
    Args:
        fallback_value: Value to return on exception
        log_fallback: Whether to log when fallback is used
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_fallback:
                    logger.warning(
                        f"Function {func.__name__} failed, using fallback value: {e}"
                    )
                return fallback_value
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to function execution.
    
    Note: This uses a simple time-based check and may not interrupt
    long-running operations. For true interruption, use threading or multiprocessing.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            # Store timeout in kwargs for function to check (only if function accepts **kwargs)
            if '_timeout_deadline' not in kwargs:
                kwargs['_timeout_deadline'] = start_time + timeout_seconds
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if elapsed > timeout_seconds:
                    logger.warning(
                        f"Function {func.__name__} exceeded timeout "
                        f"({elapsed:.2f}s > {timeout_seconds}s)"
                    )
                
                return result
            except TypeError as e:
                # If function doesn't accept **kwargs, call without timeout_deadline
                if "_timeout_deadline" in str(e):
                    kwargs.pop('_timeout_deadline', None)
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    
                    if elapsed > timeout_seconds:
                        logger.warning(
                            f"Function {func.__name__} exceeded timeout "
                            f"({elapsed:.2f}s > {timeout_seconds}s)"
                        )
                    
                    return result
                else:
                    raise
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {elapsed:.2f}s: {e}"
                )
                raise
        
        return wrapper
    return decorator


class RetryStrategy:
    """
    Configurable retry strategy for storage operations.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0
    ):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay in seconds
            exponential_base: Base for exponential calculation
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    def execute(
        self,
        func: Callable,
        *args,
        exceptions: tuple = (Exception,),
        **kwargs
    ) -> Any:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            exceptions: Tuple of exceptions to catch and retry
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(
                        f"Function {func.__name__} failed after {self.max_retries} retries: {e}"
                    )
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.base_delay * (self.exponential_base ** attempt),
                    self.max_delay
                )
                
                logger.warning(
                    f"Function {func.__name__} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                    f"retrying in {delay:.2f}s: {e}"
                )
                
                time.sleep(delay)
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception


# Global circuit breakers for different services
_circuit_breakers = {}


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception
) -> CircuitBreaker:
    """
    Get or create a circuit breaker by name.
    
    Args:
        name: Circuit breaker name
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type to catch
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception
        )
    
    return _circuit_breakers[name]


def reset_circuit_breaker(name: str):
    """
    Reset a circuit breaker by name.
    
    Args:
        name: Circuit breaker name
    """
    if name in _circuit_breakers:
        breaker = _circuit_breakers[name]
        breaker.failure_count = 0
        breaker.state = CircuitState.CLOSED
        logger.info(f"Circuit breaker '{name}' manually reset")


def reset_all_circuit_breakers():
    """Reset all circuit breakers."""
    for name in _circuit_breakers:
        reset_circuit_breaker(name)
