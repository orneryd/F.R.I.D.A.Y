"""
Error handling utilities for the neuron system.

Provides standardized error responses, error codes, and error recovery suggestions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Standard error codes for all error categories."""
    
    # Compression errors (1xxx)
    COMPRESSION_INVALID_INPUT = "E1001"
    COMPRESSION_MODEL_FAILURE = "E1002"
    COMPRESSION_TIMEOUT = "E1003"
    COMPRESSION_DIMENSION_MISMATCH = "E1004"
    
    # Query errors (2xxx)
    QUERY_NO_RESULTS = "E2001"
    QUERY_TIMEOUT = "E2002"
    QUERY_INVALID_PARAMETERS = "E2003"
    QUERY_PROPAGATION_FAILURE = "E2004"
    QUERY_SPATIAL_INDEX_ERROR = "E2005"
    
    # Training errors (3xxx)
    TRAINING_INVALID_VECTOR = "E3001"
    TRAINING_WEIGHT_OUT_OF_BOUNDS = "E3002"
    TRAINING_NEURON_NOT_FOUND = "E3003"
    TRAINING_SYNAPSE_NOT_FOUND = "E3004"
    TRAINING_ROLLBACK_FAILURE = "E3005"
    TRAINING_VALIDATION_FAILURE = "E3006"
    
    # Tool execution errors (4xxx)
    TOOL_EXECUTION_FAILURE = "E4001"
    TOOL_INVALID_PARAMETERS = "E4002"
    TOOL_TIMEOUT = "E4003"
    TOOL_SANDBOXING_ERROR = "E4004"
    TOOL_NOT_FOUND = "E4005"
    TOOL_CLUSTER_CYCLE_DETECTED = "E4006"
    TOOL_CLUSTER_EXECUTION_FAILURE = "E4007"
    
    # Storage errors (5xxx)
    STORAGE_CONNECTION_FAILURE = "E5001"
    STORAGE_WRITE_FAILURE = "E5002"
    STORAGE_READ_FAILURE = "E5003"
    STORAGE_DISK_FULL = "E5004"
    STORAGE_INTEGRITY_ERROR = "E5005"
    STORAGE_BACKUP_FAILURE = "E5006"
    STORAGE_RESTORE_FAILURE = "E5007"
    STORAGE_TRANSACTION_FAILURE = "E5008"
    
    # Neuron errors (6xxx)
    NEURON_NOT_FOUND = "E6001"
    NEURON_INVALID_TYPE = "E6002"
    NEURON_CREATION_FAILURE = "E6003"
    NEURON_DELETION_FAILURE = "E6004"
    NEURON_INVALID_POSITION = "E6005"
    NEURON_VECTOR_INVALID = "E6006"
    
    # Synapse errors (7xxx)
    SYNAPSE_NOT_FOUND = "E7001"
    SYNAPSE_INVALID_WEIGHT = "E7002"
    SYNAPSE_CREATION_FAILURE = "E7003"
    SYNAPSE_DELETION_FAILURE = "E7004"
    SYNAPSE_REFERENTIAL_INTEGRITY = "E7005"
    
    # Spatial index errors (8xxx)
    SPATIAL_INDEX_INSERT_FAILURE = "E8001"
    SPATIAL_INDEX_QUERY_FAILURE = "E8002"
    SPATIAL_INDEX_OUT_OF_BOUNDS = "E8003"
    SPATIAL_INDEX_REBALANCE_FAILURE = "E8004"
    
    # API errors (9xxx)
    API_AUTHENTICATION_FAILURE = "E9001"
    API_AUTHORIZATION_FAILURE = "E9002"
    API_RATE_LIMIT_EXCEEDED = "E9003"
    API_INVALID_REQUEST = "E9004"
    API_INTERNAL_ERROR = "E9005"
    
    # General errors (0xxx)
    UNKNOWN_ERROR = "E0001"
    VALIDATION_ERROR = "E0002"
    CONFIGURATION_ERROR = "E0003"
    RESOURCE_EXHAUSTED = "E0004"


@dataclass
class ErrorResponse:
    """
    Standardized error response structure.
    
    Provides consistent error information across the system with
    error codes, messages, details, and recovery suggestions.
    """
    
    error_code: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recoverable: bool = True
    recovery_suggestions: list[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error response to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "recoverable": self.recoverable,
            "recovery_suggestions": self.recovery_suggestions
        }
    
    def log(self, level: str = "error"):
        """Log the error with appropriate level."""
        log_func = getattr(logger, level.lower(), logger.error)
        log_func(
            f"Error {self.error_code}: {self.message}",
            extra={
                "error_code": self.error_code,
                "details": self.details,
                "recoverable": self.recoverable
            }
        )


class NeuronSystemException(Exception):
    """Base exception for all neuron system errors."""
    
    def __init__(
        self,
        error_code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        recovery_suggestions: Optional[list[str]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.message = message
        self.details = details or {}
        self.recoverable = recoverable
        self.recovery_suggestions = recovery_suggestions or []
    
    def to_error_response(self) -> ErrorResponse:
        """Convert exception to ErrorResponse."""
        return ErrorResponse(
            error_code=self.error_code.value,
            message=self.message,
            details=self.details,
            recoverable=self.recoverable,
            recovery_suggestions=self.recovery_suggestions
        )


class CompressionError(NeuronSystemException):
    """Errors related to data compression."""
    pass


class QueryError(NeuronSystemException):
    """Errors related to query execution."""
    pass


class TrainingError(NeuronSystemException):
    """Errors related to training operations."""
    pass


class ToolExecutionError(NeuronSystemException):
    """Errors related to tool execution."""
    pass


class StorageError(NeuronSystemException):
    """Errors related to storage operations."""
    pass


class NeuronError(NeuronSystemException):
    """Errors related to neuron operations."""
    pass


class SynapseError(NeuronSystemException):
    """Errors related to synapse operations."""
    pass


class SpatialIndexError(NeuronSystemException):
    """Errors related to spatial index operations."""
    pass


class APIError(NeuronSystemException):
    """Errors related to API operations."""
    pass


# Error recovery suggestion templates
RECOVERY_SUGGESTIONS = {
    ErrorCode.COMPRESSION_INVALID_INPUT: [
        "Ensure input data is a valid string",
        "Check for null or empty input",
        "Verify data encoding is UTF-8"
    ],
    ErrorCode.COMPRESSION_MODEL_FAILURE: [
        "Check if embedding model is properly loaded",
        "Verify model files are not corrupted",
        "Try restarting the compression engine"
    ],
    ErrorCode.COMPRESSION_TIMEOUT: [
        "Reduce input data size",
        "Increase compression timeout in settings",
        "Check system resource availability"
    ],
    ErrorCode.QUERY_NO_RESULTS: [
        "Try a different query",
        "Check if neurons exist in the system",
        "Adjust query parameters (top_k, propagation_depth)"
    ],
    ErrorCode.QUERY_TIMEOUT: [
        "Reduce propagation depth",
        "Reduce top_k parameter",
        "Increase query timeout in settings",
        "Check system resource availability"
    ],
    ErrorCode.TRAINING_INVALID_VECTOR: [
        "Ensure vector has correct dimensions (auto-detected from model)",
        "Check for NaN or infinite values",
        "Verify vector is normalized"
    ],
    ErrorCode.TRAINING_WEIGHT_OUT_OF_BOUNDS: [
        "Ensure weight is between -1.0 and 1.0",
        "Check weight adjustment delta values"
    ],
    ErrorCode.TOOL_EXECUTION_FAILURE: [
        "Check tool code for errors",
        "Verify input parameters match schema",
        "Review tool execution logs"
    ],
    ErrorCode.TOOL_TIMEOUT: [
        "Increase tool execution timeout",
        "Optimize tool code for performance",
        "Check for infinite loops in tool code"
    ],
    ErrorCode.STORAGE_CONNECTION_FAILURE: [
        "Check database file permissions",
        "Verify database path is correct",
        "Ensure database is not locked by another process"
    ],
    ErrorCode.STORAGE_DISK_FULL: [
        "Free up disk space",
        "Move database to a different location",
        "Enable automatic cleanup of old data"
    ],
    ErrorCode.STORAGE_INTEGRITY_ERROR: [
        "Restore from backup",
        "Run integrity verification",
        "Check for database corruption"
    ],
    ErrorCode.API_RATE_LIMIT_EXCEEDED: [
        "Reduce request frequency",
        "Implement request batching",
        "Contact administrator to increase rate limit"
    ],
    ErrorCode.SPATIAL_INDEX_OUT_OF_BOUNDS: [
        "Check neuron position is within spatial bounds",
        "Adjust spatial bounds in settings",
        "Use automatic positioning"
    ]
}


def get_recovery_suggestions(error_code: ErrorCode) -> list[str]:
    """
    Get recovery suggestions for an error code.
    
    Args:
        error_code: The error code
        
    Returns:
        List of recovery suggestions
    """
    return RECOVERY_SUGGESTIONS.get(error_code, [
        "Check system logs for more details",
        "Contact system administrator if problem persists"
    ])


def create_error_response(
    error_code: ErrorCode,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    recoverable: bool = True,
    custom_suggestions: Optional[list[str]] = None
) -> ErrorResponse:
    """
    Create a standardized error response.
    
    Args:
        error_code: The error code
        message: Human-readable error message
        details: Additional error details
        recoverable: Whether the error is recoverable
        custom_suggestions: Custom recovery suggestions (overrides defaults)
        
    Returns:
        ErrorResponse instance
    """
    suggestions = custom_suggestions or get_recovery_suggestions(error_code)
    
    error_response = ErrorResponse(
        error_code=error_code.value,
        message=message,
        details=details or {},
        recoverable=recoverable,
        recovery_suggestions=suggestions
    )
    
    # Log the error
    error_response.log()
    
    return error_response
