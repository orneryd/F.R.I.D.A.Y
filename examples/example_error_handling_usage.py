"""
Example usage of the comprehensive error handling system.

Demonstrates how to use error codes, retry logic, circuit breakers,
and fallback mechanisms in real-world scenarios.
"""

import numpy as np
from uuid import uuid4

from neuron_system.core.vector3d import Vector3D
from neuron_system.core.graph import NeuronGraph
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.engines.compression import CompressionEngine
from neuron_system.engines.query import QueryEngine
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore

from neuron_system.utils import (
    ErrorCode,
    ErrorResponse,
    create_error_response,
    exponential_backoff,
    with_fallback,
    with_timeout,
    get_circuit_breaker,
    RetryStrategy,
    setup_logging,
    log_audit_event,
)


def example_1_compression_with_fallback():
    """Example 1: Compression with automatic fallback."""
    print("\n=== Example 1: Compression with Fallback ===")
    
    engine = CompressionEngine()
    
    # Valid input - should succeed
    vector1, metadata1 = engine.compress("This is valid text", raise_on_error=False)
    print(f"Valid input: success={metadata1.get('success', True)}")
    
    # Invalid input - should use fallback
    vector2, metadata2 = engine.compress("", raise_on_error=False)
    print(f"Empty input: success={metadata2.get('success', True)}, fallback={metadata2.get('fallback', False)}")
    
    if not metadata2.get('success', True):
        error_response = create_error_response(
            error_code=ErrorCode.COMPRESSION_INVALID_INPUT,
            message=f"Compression failed: {metadata2.get('error')}",
            details=metadata2,
            recoverable=True
        )
        print(f"Error code: {error_response.error_code}")
        print(f"Recovery suggestions: {error_response.recovery_suggestions[:2]}")


def example_2_storage_with_retry():
    """Example 2: Storage operations with retry logic."""
    print("\n=== Example 2: Storage with Retry ===")
    
    # Simulate storage operation with retry
    retry_strategy = RetryStrategy(
        max_retries=3,
        base_delay=0.5,
        max_delay=5.0
    )
    
    attempt_count = [0]
    
    def flaky_storage_operation():
        """Simulates a storage operation that fails twice then succeeds."""
        attempt_count[0] += 1
        print(f"  Attempt {attempt_count[0]}")
        
        if attempt_count[0] < 3:
            raise Exception("Temporary storage failure")
        
        return True
    
    try:
        result = retry_strategy.execute(
            flaky_storage_operation,
            exceptions=(Exception,)
        )
        print(f"Storage operation succeeded after {attempt_count[0]} attempts")
    except Exception as e:
        error_response = create_error_response(
            error_code=ErrorCode.STORAGE_WRITE_FAILURE,
            message=f"Storage failed after retries: {str(e)}",
            recoverable=True
        )
        print(f"Error: {error_response.message}")


def example_3_query_with_timeout():
    """Example 3: Query with timeout protection."""
    print("\n=== Example 3: Query with Timeout ===")
    
    @with_timeout(timeout_seconds=2.0)
    def fast_query(**kwargs):
        """Fast query that completes within timeout."""
        return ["result1", "result2", "result3"]
    
    try:
        results = fast_query()
        print(f"Query completed successfully: {len(results)} results")
    except Exception as e:
        error_response = create_error_response(
            error_code=ErrorCode.QUERY_TIMEOUT,
            message=f"Query timed out: {str(e)}",
            recoverable=True
        )
        print(f"Error: {error_response.message}")


def example_4_circuit_breaker():
    """Example 4: Circuit breaker for tool execution."""
    print("\n=== Example 4: Circuit Breaker ===")
    
    circuit_breaker = get_circuit_breaker(
        name="example_tool_execution",
        failure_threshold=3,
        recovery_timeout=5.0
    )
    
    failure_count = [0]
    
    def flaky_tool_execution():
        """Simulates a tool that fails multiple times."""
        failure_count[0] += 1
        if failure_count[0] <= 3:
            raise Exception("Tool execution failed")
        return "success"
    
    # Try to execute multiple times
    for i in range(5):
        try:
            result = circuit_breaker.call(flaky_tool_execution)
            print(f"  Attempt {i+1}: Success - {result}")
        except Exception as e:
            if "Circuit breaker is OPEN" in str(e):
                print(f"  Attempt {i+1}: Circuit breaker is OPEN - service unavailable")
            else:
                print(f"  Attempt {i+1}: Failed - {str(e)}")
    
    print(f"Circuit breaker state: {circuit_breaker.state.value}")


def example_5_batch_operations():
    """Example 5: Batch operations with error handling."""
    print("\n=== Example 5: Batch Operations ===")
    
    # Simulate batch neuron creation
    neurons_to_create = [
        {"id": str(uuid4()), "data": f"Neuron {i}"}
        for i in range(5)
    ]
    
    results = {
        "successful": [],
        "failed": [],
        "total": len(neurons_to_create)
    }
    
    retry_strategy = RetryStrategy(max_retries=2, base_delay=0.1)
    
    for i, neuron_data in enumerate(neurons_to_create):
        try:
            # Simulate creation (fail on neuron 2)
            def create_neuron():
                if i == 2:
                    raise Exception("Simulated failure")
                return True
            
            retry_strategy.execute(create_neuron, exceptions=(Exception,))
            results["successful"].append(neuron_data["id"])
            
        except Exception as e:
            error_response = create_error_response(
                error_code=ErrorCode.NEURON_CREATION_FAILURE,
                message=f"Failed to create neuron: {str(e)}",
                details={"neuron_id": neuron_data["id"]},
                recoverable=False
            )
            results["failed"].append({
                "neuron_id": neuron_data["id"],
                "error": str(e),
                "error_code": error_response.error_code
            })
    
    print(f"Batch creation complete:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")


def example_6_audit_logging():
    """Example 6: Audit logging for important operations."""
    print("\n=== Example 6: Audit Logging ===")
    
    # Setup logging (in real usage, do this once at startup)
    setup_logging(
        log_level="INFO",
        enable_audit=True
    )
    
    # Log some audit events
    operations = [
        ("neuron_created", {"neuron_id": str(uuid4()), "type": "knowledge"}),
        ("training_operation", {"neuron_id": str(uuid4()), "operation": "adjust_vector"}),
        ("query_executed", {"query": "test query", "results": 5}),
    ]
    
    for event_type, details in operations:
        log_audit_event(
            event_type=event_type,
            details=details,
            user_id="example_user"
        )
        print(f"  Logged: {event_type}")
    
    print("Audit events logged to neuron_system_audit.log")


def example_7_error_response_handling():
    """Example 7: Creating and handling error responses."""
    print("\n=== Example 7: Error Response Handling ===")
    
    # Create various error responses
    errors = [
        create_error_response(
            error_code=ErrorCode.COMPRESSION_TIMEOUT,
            message="Compression took too long",
            details={"timeout": 100, "actual": 150},
            recoverable=True
        ),
        create_error_response(
            error_code=ErrorCode.QUERY_NO_RESULTS,
            message="Query returned no results",
            details={"query": "test query"},
            recoverable=True
        ),
        create_error_response(
            error_code=ErrorCode.STORAGE_DISK_FULL,
            message="Disk space exhausted",
            details={"available": 0, "required": 1000000},
            recoverable=False
        ),
    ]
    
    for error in errors:
        print(f"\nError: {error.error_code}")
        print(f"  Message: {error.message}")
        print(f"  Recoverable: {error.recoverable}")
        print(f"  Suggestions: {error.recovery_suggestions[:2]}")


def example_8_integration_with_compression():
    """Example 8: Real integration with compression engine."""
    print("\n=== Example 8: Integration with Compression Engine ===")
    
    engine = CompressionEngine()
    
    test_cases = [
        ("Valid text", "This is a valid text for compression"),
        ("Empty string", ""),
        ("Very long text", "x" * 20000),  # Exceeds max length
    ]
    
    for name, text in test_cases:
        vector, metadata = engine.compress(text, raise_on_error=False)
        
        print(f"\n{name}:")
        print(f"  Success: {metadata.get('success', True)}")
        print(f"  Fallback: {metadata.get('fallback', False)}")
        
        if not metadata.get('success', True):
            print(f"  Error: {metadata.get('error', 'Unknown')}")
            
            # Create error response
            error_response = create_error_response(
                error_code=ErrorCode.COMPRESSION_INVALID_INPUT,
                message=f"Compression failed for {name}",
                details=metadata,
                recoverable=True
            )
            print(f"  Error code: {error_response.error_code}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Error Handling System Examples")
    print("=" * 60)
    
    try:
        example_1_compression_with_fallback()
        example_2_storage_with_retry()
        example_3_query_with_timeout()
        example_4_circuit_breaker()
        example_5_batch_operations()
        example_6_audit_logging()
        example_7_error_response_handling()
        example_8_integration_with_compression()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nExample failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
