"""
Integration examples for error handling utilities.

Shows how to integrate error handling, retry logic, and circuit breakers
with existing neuron system modules.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional
from uuid import UUID

from neuron_system.utils.errors import (
    ErrorCode,
    CompressionError,
    QueryError,
    StorageError,
    ToolExecutionError,
    create_error_response,
)
from neuron_system.utils.retry import (
    exponential_backoff,
    with_fallback,
    with_timeout,
    get_circuit_breaker,
    RetryStrategy,
)

logger = logging.getLogger(__name__)


# Example 1: Compression with fallback and error handling
def safe_compress_with_fallback(compression_engine, data: str) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Compress data with automatic fallback on failure.
    
    Args:
        compression_engine: CompressionEngine instance
        data: Text to compress
        
    Returns:
        Tuple of (vector, metadata)
    """
    try:
        # The compression engine already has built-in fallback
        vector, metadata = compression_engine.compress(data, raise_on_error=False)
        
        if not metadata.get('success', True):
            # Log error response
            error_response = create_error_response(
                error_code=ErrorCode.COMPRESSION_MODEL_FAILURE,
                message=f"Compression failed: {metadata.get('error', 'Unknown error')}",
                details=metadata,
                recoverable=True
            )
            logger.warning(f"Using fallback vector: {error_response.message}")
        
        return vector, metadata
        
    except Exception as e:
        # Create error response
        error_response = create_error_response(
            error_code=ErrorCode.COMPRESSION_INVALID_INPUT,
            message=f"Failed to compress data: {str(e)}",
            details={"input_length": len(data) if isinstance(data, str) else 0},
            recoverable=True
        )
        
        # Return zero vector as fallback
        fallback_vector = np.zeros(384, dtype=np.float32)
        metadata = {
            "success": False,
            "error": str(e),
            "fallback": True,
            "error_code": error_response.error_code
        }
        
        return fallback_vector, metadata


# Example 2: Storage operations with exponential backoff
@exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exceptions=(Exception,)
)
def save_neuron_with_retry(neuron_store, neuron) -> bool:
    """
    Save neuron with automatic retry on failure.
    
    Args:
        neuron_store: NeuronStore instance
        neuron: Neuron to save
        
    Returns:
        True if successful
        
    Raises:
        StorageError: If all retries fail
    """
    try:
        return neuron_store.create(neuron)
    except Exception as e:
        # Wrap in StorageError for better error tracking
        raise StorageError(
            error_code=ErrorCode.STORAGE_WRITE_FAILURE,
            message=f"Failed to save neuron {neuron.id}",
            details={"neuron_id": str(neuron.id), "error": str(e)},
            recoverable=True
        )


# Example 3: Query with timeout handling
@with_timeout(timeout_seconds=5.0)
def query_with_timeout(query_engine, query_text: str, **kwargs) -> List[Any]:
    """
    Execute query with timeout protection.
    
    Args:
        query_engine: QueryEngine instance
        query_text: Query text
        **kwargs: Additional query parameters
        
    Returns:
        List of activated neurons
        
    Raises:
        QueryError: If query fails or times out
    """
    try:
        # Check if we're approaching timeout
        timeout_deadline = kwargs.pop('_timeout_deadline', None)
        
        results = query_engine.query(query_text, **kwargs)
        
        if not results:
            error_response = create_error_response(
                error_code=ErrorCode.QUERY_NO_RESULTS,
                message="Query returned no results",
                details={"query": query_text[:100]},
                recoverable=True
            )
            logger.info(error_response.message)
        
        return results
        
    except Exception as e:
        raise QueryError(
            error_code=ErrorCode.QUERY_PROPAGATION_FAILURE,
            message=f"Query failed: {str(e)}",
            details={"query": query_text[:100]},
            recoverable=True
        )


# Example 4: Tool execution with circuit breaker
def execute_tool_with_circuit_breaker(
    tool_neuron,
    inputs: Dict[str, Any],
    circuit_breaker_name: str = "tool_execution"
) -> Dict[str, Any]:
    """
    Execute tool with circuit breaker protection.
    
    Args:
        tool_neuron: ToolNeuron instance
        inputs: Tool input parameters
        circuit_breaker_name: Name of circuit breaker to use
        
    Returns:
        Tool execution result
        
    Raises:
        ToolExecutionError: If execution fails or circuit is open
    """
    # Get or create circuit breaker for tool execution
    circuit_breaker = get_circuit_breaker(
        name=circuit_breaker_name,
        failure_threshold=5,
        recovery_timeout=60.0,
        expected_exception=Exception
    )
    
    try:
        # Execute through circuit breaker
        result = circuit_breaker.call(tool_neuron.execute, inputs)
        return result
        
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            # Circuit is open - service unavailable
            raise ToolExecutionError(
                error_code=ErrorCode.TOOL_EXECUTION_FAILURE,
                message="Tool execution service temporarily unavailable",
                details={
                    "tool_id": str(tool_neuron.id),
                    "circuit_state": "OPEN"
                },
                recoverable=True,
                recovery_suggestions=[
                    "Wait for circuit breaker to reset",
                    "Check tool execution logs",
                    "Verify tool code is not causing repeated failures"
                ]
            )
        else:
            # Execution failed
            raise ToolExecutionError(
                error_code=ErrorCode.TOOL_EXECUTION_FAILURE,
                message=f"Tool execution failed: {str(e)}",
                details={
                    "tool_id": str(tool_neuron.id),
                    "inputs": inputs,
                    "error": str(e)
                },
                recoverable=True
            )


# Example 5: Batch operations with retry strategy
def batch_create_neurons_with_retry(
    neuron_graph,
    neurons: List[Any],
    retry_strategy: Optional[RetryStrategy] = None
) -> Dict[str, Any]:
    """
    Create multiple neurons with retry logic.
    
    Args:
        neuron_graph: NeuronGraph instance
        neurons: List of neurons to create
        retry_strategy: Optional custom retry strategy
        
    Returns:
        Dictionary with creation results
    """
    if retry_strategy is None:
        retry_strategy = RetryStrategy(
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0
        )
    
    results = {
        "successful": [],
        "failed": [],
        "total": len(neurons)
    }
    
    for neuron in neurons:
        try:
            # Execute with retry
            success = retry_strategy.execute(
                neuron_graph.add_neuron,
                neuron,
                exceptions=(Exception,)
            )
            
            if success:
                results["successful"].append(str(neuron.id))
            else:
                results["failed"].append({
                    "neuron_id": str(neuron.id),
                    "error": "Unknown failure"
                })
                
        except Exception as e:
            # All retries failed
            error_response = create_error_response(
                error_code=ErrorCode.NEURON_CREATION_FAILURE,
                message=f"Failed to create neuron after retries: {str(e)}",
                details={"neuron_id": str(neuron.id)},
                recoverable=False
            )
            
            results["failed"].append({
                "neuron_id": str(neuron.id),
                "error": str(e),
                "error_code": error_response.error_code
            })
    
    logger.info(
        f"Batch creation complete: {len(results['successful'])} successful, "
        f"{len(results['failed'])} failed"
    )
    
    return results


# Example 6: Storage operations with integrity checking
def save_with_integrity_check(
    serialization_manager,
    retry_strategy: Optional[RetryStrategy] = None
) -> Dict[str, Any]:
    """
    Save data with integrity verification and retry.
    
    Args:
        serialization_manager: SerializationManager instance
        retry_strategy: Optional custom retry strategy
        
    Returns:
        Save statistics
        
    Raises:
        StorageError: If save fails after retries
    """
    if retry_strategy is None:
        retry_strategy = RetryStrategy(
            max_retries=3,
            base_delay=2.0,
            max_delay=30.0
        )
    
    try:
        # Save with retry
        stats = retry_strategy.execute(
            serialization_manager.save_incremental,
            exceptions=(Exception,)
        )
        
        # Verify integrity
        integrity_result = serialization_manager.verify_integrity()
        
        if not integrity_result.get("valid", False):
            # Integrity check failed
            raise StorageError(
                error_code=ErrorCode.STORAGE_INTEGRITY_ERROR,
                message="Data integrity verification failed after save",
                details=integrity_result,
                recoverable=True,
                recovery_suggestions=[
                    "Restore from backup",
                    "Re-save data",
                    "Check for database corruption"
                ]
            )
        
        return stats
        
    except Exception as e:
        raise StorageError(
            error_code=ErrorCode.STORAGE_WRITE_FAILURE,
            message=f"Failed to save data: {str(e)}",
            details={},
            recoverable=True
        )


# Example 7: Comprehensive error handling wrapper
class ErrorHandlingWrapper:
    """
    Wrapper class that adds error handling to any component.
    """
    
    def __init__(
        self,
        component: Any,
        enable_retry: bool = True,
        enable_circuit_breaker: bool = False,
        circuit_breaker_name: Optional[str] = None
    ):
        """
        Initialize error handling wrapper.
        
        Args:
            component: Component to wrap
            enable_retry: Enable retry logic
            enable_circuit_breaker: Enable circuit breaker
            circuit_breaker_name: Name for circuit breaker
        """
        self.component = component
        self.enable_retry = enable_retry
        self.enable_circuit_breaker = enable_circuit_breaker
        
        if enable_circuit_breaker:
            self.circuit_breaker = get_circuit_breaker(
                name=circuit_breaker_name or f"{component.__class__.__name__}_circuit",
                failure_threshold=5,
                recovery_timeout=60.0
            )
        else:
            self.circuit_breaker = None
        
        if enable_retry:
            self.retry_strategy = RetryStrategy(
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0
            )
        else:
            self.retry_strategy = None
    
    def call_method(self, method_name: str, *args, **kwargs) -> Any:
        """
        Call a method on the wrapped component with error handling.
        
        Args:
            method_name: Name of method to call
            *args: Method arguments
            **kwargs: Method keyword arguments
            
        Returns:
            Method result
        """
        method = getattr(self.component, method_name)
        
        try:
            if self.enable_circuit_breaker and self.circuit_breaker:
                # Use circuit breaker
                if self.enable_retry and self.retry_strategy:
                    # Circuit breaker + retry
                    return self.circuit_breaker.call(
                        self.retry_strategy.execute,
                        method,
                        *args,
                        **kwargs
                    )
                else:
                    # Circuit breaker only
                    return self.circuit_breaker.call(method, *args, **kwargs)
            elif self.enable_retry and self.retry_strategy:
                # Retry only
                return self.retry_strategy.execute(method, *args, **kwargs)
            else:
                # No error handling
                return method(*args, **kwargs)
                
        except Exception as e:
            # Create error response
            error_response = create_error_response(
                error_code=ErrorCode.UNKNOWN_ERROR,
                message=f"Method {method_name} failed: {str(e)}",
                details={
                    "component": self.component.__class__.__name__,
                    "method": method_name
                },
                recoverable=True
            )
            
            logger.error(f"Error in wrapped method: {error_response.to_dict()}")
            raise
