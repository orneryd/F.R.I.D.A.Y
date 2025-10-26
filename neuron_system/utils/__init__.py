"""
Shared utility modules.
"""

from neuron_system.utils.pooling import ObjectPool
from neuron_system.utils.uuid_pool import UUIDPool
from neuron_system.utils.errors import (
    ErrorCode,
    ErrorResponse,
    NeuronSystemException,
    CompressionError,
    QueryError,
    TrainingError,
    ToolExecutionError,
    StorageError,
    NeuronError,
    SynapseError,
    SpatialIndexError,
    APIError,
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
    reset_all_circuit_breakers,
)
from neuron_system.utils.logging_config import (
    setup_logging,
    get_audit_logger,
    log_audit_event,
)

__all__ = [
    "ObjectPool",
    "UUIDPool",
    "ErrorCode",
    "ErrorResponse",
    "NeuronSystemException",
    "CompressionError",
    "QueryError",
    "TrainingError",
    "ToolExecutionError",
    "StorageError",
    "NeuronError",
    "SynapseError",
    "SpatialIndexError",
    "APIError",
    "get_recovery_suggestions",
    "create_error_response",
    "CircuitBreaker",
    "CircuitState",
    "exponential_backoff",
    "with_fallback",
    "with_timeout",
    "RetryStrategy",
    "get_circuit_breaker",
    "reset_circuit_breaker",
    "reset_all_circuit_breakers",
    "setup_logging",
    "get_audit_logger",
    "log_audit_event",
]
