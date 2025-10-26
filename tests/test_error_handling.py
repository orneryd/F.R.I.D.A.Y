"""
Tests for comprehensive error handling system.

Tests error response models, error codes, retry logic, circuit breakers,
and fallback mechanisms.
"""

import pytest
import time
import numpy as np
from datetime import datetime

from neuron_system.utils.errors import (
    ErrorCode,
    ErrorResponse,
    NeuronSystemException,
    CompressionError,
    QueryError,
    StorageError,
    ToolExecutionError,
    get_recovery_suggestions,
    create_error_response,
)
from neuron_system.utils.retry import (
    CircuitBreaker,
    CircuitState,
    exponential_backoff,
    with_fallback,
    with_timeout,
    RetryStrategy,
    get_circuit_breaker,
    reset_circuit_breaker,
)


class TestErrorResponse:
    """Test ErrorResponse dataclass."""
    
    def test_error_response_creation(self):
        """Test creating an error response."""
        error = ErrorResponse(
            error_code="E1001",
            message="Test error",
            details={"key": "value"},
            recoverable=True,
            recovery_suggestions=["Try again"]
        )
        
        assert error.error_code == "E1001"
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
        assert error.recoverable is True
        assert error.recovery_suggestions == ["Try again"]
        assert isinstance(error.timestamp, datetime)
    
    def test_error_response_to_dict(self):
        """Test converting error response to dictionary."""
        error = ErrorResponse(
            error_code="E1001",
            message="Test error",
            details={"key": "value"},
            recoverable=True,
            recovery_suggestions=["Try again"]
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "E1001"
        assert error_dict["message"] == "Test error"
        assert error_dict["details"] == {"key": "value"}
        assert error_dict["recoverable"] is True
        assert error_dict["recovery_suggestions"] == ["Try again"]
        assert "timestamp" in error_dict


class TestErrorCodes:
    """Test error code definitions."""
    
    def test_error_codes_exist(self):
        """Test that all error code categories exist."""
        # Compression errors
        assert ErrorCode.COMPRESSION_INVALID_INPUT.value == "E1001"
        assert ErrorCode.COMPRESSION_MODEL_FAILURE.value == "E1002"
        
        # Query errors
        assert ErrorCode.QUERY_NO_RESULTS.value == "E2001"
        assert ErrorCode.QUERY_TIMEOUT.value == "E2002"
        
        # Training errors
        assert ErrorCode.TRAINING_INVALID_VECTOR.value == "E3001"
        assert ErrorCode.TRAINING_WEIGHT_OUT_OF_BOUNDS.value == "E3002"
        
        # Tool errors
        assert ErrorCode.TOOL_EXECUTION_FAILURE.value == "E4001"
        assert ErrorCode.TOOL_TIMEOUT.value == "E4003"
        
        # Storage errors
        assert ErrorCode.STORAGE_CONNECTION_FAILURE.value == "E5001"
        assert ErrorCode.STORAGE_DISK_FULL.value == "E5004"
    
    def test_recovery_suggestions(self):
        """Test getting recovery suggestions for error codes."""
        suggestions = get_recovery_suggestions(ErrorCode.COMPRESSION_INVALID_INPUT)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert any("input" in s.lower() for s in suggestions)


class TestNeuronSystemException:
    """Test custom exception classes."""
    
    def test_exception_creation(self):
        """Test creating a custom exception."""
        exc = CompressionError(
            error_code=ErrorCode.COMPRESSION_INVALID_INPUT,
            message="Invalid input",
            details={"input": "test"},
            recoverable=True,
            recovery_suggestions=["Check input"]
        )
        
        assert exc.error_code == ErrorCode.COMPRESSION_INVALID_INPUT
        assert exc.message == "Invalid input"
        assert exc.details == {"input": "test"}
        assert exc.recoverable is True
        assert exc.recovery_suggestions == ["Check input"]
    
    def test_exception_to_error_response(self):
        """Test converting exception to error response."""
        exc = QueryError(
            error_code=ErrorCode.QUERY_TIMEOUT,
            message="Query timed out",
            details={"timeout": 5.0},
            recoverable=True
        )
        
        error_response = exc.to_error_response()
        
        assert isinstance(error_response, ErrorResponse)
        assert error_response.error_code == ErrorCode.QUERY_TIMEOUT.value
        assert error_response.message == "Query timed out"
        assert error_response.details == {"timeout": 5.0}


class TestCreateErrorResponse:
    """Test error response creation helper."""
    
    def test_create_error_response(self):
        """Test creating error response with helper function."""
        error = create_error_response(
            error_code=ErrorCode.STORAGE_WRITE_FAILURE,
            message="Write failed",
            details={"file": "test.db"},
            recoverable=True
        )
        
        assert isinstance(error, ErrorResponse)
        assert error.error_code == ErrorCode.STORAGE_WRITE_FAILURE.value
        assert error.message == "Write failed"
        assert len(error.recovery_suggestions) > 0
    
    def test_create_error_response_custom_suggestions(self):
        """Test creating error response with custom suggestions."""
        custom_suggestions = ["Custom suggestion 1", "Custom suggestion 2"]
        
        error = create_error_response(
            error_code=ErrorCode.TOOL_EXECUTION_FAILURE,
            message="Tool failed",
            custom_suggestions=custom_suggestions
        )
        
        assert error.recovery_suggestions == custom_suggestions


class TestExponentialBackoff:
    """Test exponential backoff retry decorator."""
    
    def test_successful_execution(self):
        """Test that successful execution doesn't retry."""
        call_count = [0]
        
        @exponential_backoff(max_retries=3, base_delay=0.1)
        def successful_func():
            call_count[0] += 1
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert call_count[0] == 1
    
    def test_retry_on_failure(self):
        """Test that function retries on failure."""
        call_count = [0]
        
        @exponential_backoff(max_retries=3, base_delay=0.1)
        def failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = failing_func()
        
        assert result == "success"
        assert call_count[0] == 3
    
    def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        call_count = [0]
        
        @exponential_backoff(max_retries=2, base_delay=0.1)
        def always_failing_func():
            call_count[0] += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            always_failing_func()
        
        assert call_count[0] == 3  # Initial + 2 retries


class TestWithFallback:
    """Test fallback decorator."""
    
    def test_successful_execution(self):
        """Test that successful execution returns result."""
        @with_fallback(fallback_value="fallback")
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"
    
    def test_fallback_on_exception(self):
        """Test that fallback value is returned on exception."""
        @with_fallback(fallback_value="fallback")
        def failing_func():
            raise ValueError("Error")
        
        result = failing_func()
        assert result == "fallback"
    
    def test_fallback_none(self):
        """Test fallback with None value."""
        @with_fallback(fallback_value=None)
        def failing_func():
            raise ValueError("Error")
        
        result = failing_func()
        assert result is None


class TestWithTimeout:
    """Test timeout decorator."""
    
    def test_fast_execution(self):
        """Test that fast execution completes normally."""
        @with_timeout(timeout_seconds=1.0)
        def fast_func():
            return "done"
        
        result = fast_func()
        assert result == "done"
    
    def test_timeout_deadline_passed(self):
        """Test that timeout deadline is passed to function."""
        deadline_received = [None]
        
        @with_timeout(timeout_seconds=1.0)
        def check_deadline_func(**kwargs):
            deadline_received[0] = kwargs.get('_timeout_deadline')
            return "done"
        
        check_deadline_func()
        
        assert deadline_received[0] is not None
        assert isinstance(deadline_received[0], float)


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_closed_initially(self):
        """Test that circuit starts in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        assert breaker.state == CircuitState.CLOSED
    
    def test_circuit_opens_after_failures(self):
        """Test that circuit opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
        
        def failing_func():
            raise ValueError("Error")
        
        # Fail 3 times to open circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_rejects_when_open(self):
        """Test that circuit rejects calls when open."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        
        def failing_func():
            raise ValueError("Error")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        # Next call should be rejected
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            breaker.call(failing_func)
    
    def test_circuit_half_open_after_timeout(self):
        """Test that circuit enters HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
        
        def failing_func():
            raise ValueError("Error")
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Next call should attempt (HALF_OPEN)
        with pytest.raises(ValueError):
            breaker.call(failing_func)
        
        # Should be back to OPEN after failure in HALF_OPEN
        assert breaker.state == CircuitState.OPEN
    
    def test_circuit_closes_on_success(self):
        """Test that circuit closes on successful call in HALF_OPEN."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.5)
        call_count = [0]
        
        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("Error")
            return "success"
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                breaker.call(sometimes_failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.6)
        
        # Successful call should close circuit
        result = breaker.call(sometimes_failing_func)
        
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED


class TestRetryStrategy:
    """Test retry strategy class."""
    
    def test_retry_strategy_success(self):
        """Test retry strategy with successful execution."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.1)
        
        def successful_func():
            return "success"
        
        result = strategy.execute(successful_func)
        assert result == "success"
    
    def test_retry_strategy_with_retries(self):
        """Test retry strategy with failures then success."""
        strategy = RetryStrategy(max_retries=3, base_delay=0.1)
        call_count = [0]
        
        def sometimes_failing_func():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = strategy.execute(sometimes_failing_func)
        
        assert result == "success"
        assert call_count[0] == 3
    
    def test_retry_strategy_max_retries(self):
        """Test retry strategy exceeding max retries."""
        strategy = RetryStrategy(max_retries=2, base_delay=0.1)
        
        def always_failing_func():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            strategy.execute(always_failing_func)


class TestGetCircuitBreaker:
    """Test circuit breaker registry."""
    
    def test_get_circuit_breaker(self):
        """Test getting circuit breaker by name."""
        breaker1 = get_circuit_breaker("test_breaker")
        breaker2 = get_circuit_breaker("test_breaker")
        
        # Should return same instance
        assert breaker1 is breaker2
    
    def test_reset_circuit_breaker(self):
        """Test resetting circuit breaker."""
        breaker = get_circuit_breaker("reset_test")
        
        # Open the circuit
        def failing_func():
            raise ValueError("Error")
        
        for _ in range(5):
            with pytest.raises(ValueError):
                breaker.call(failing_func)
        
        assert breaker.state == CircuitState.OPEN
        
        # Reset
        reset_circuit_breaker("reset_test")
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0


def test_integration_compression_with_fallback():
    """Integration test: compression with fallback."""
    from neuron_system.engines.compression import CompressionEngine
    
    engine = CompressionEngine()
    
    # Test with invalid input (should use fallback)
    vector, metadata = engine.compress("", raise_on_error=False)
    
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (384,)
    assert metadata.get('fallback', False) is True
    assert metadata.get('success', True) is False


def test_integration_error_response_logging():
    """Integration test: error response logging."""
    error = create_error_response(
        error_code=ErrorCode.QUERY_TIMEOUT,
        message="Query timed out after 5 seconds",
        details={"timeout": 5.0, "query": "test query"},
        recoverable=True
    )
    
    # Should not raise exception
    error.log(level="warning")
    
    assert error.error_code == ErrorCode.QUERY_TIMEOUT.value
    assert len(error.recovery_suggestions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
