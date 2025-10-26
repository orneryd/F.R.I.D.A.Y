# Error Handling System Documentation

## Overview

The neuron system includes a comprehensive error handling framework that provides:

- **Standardized error responses** with error codes and recovery suggestions
- **Exponential backoff retry logic** for transient failures
- **Circuit breaker pattern** to prevent cascading failures
- **Fallback mechanisms** for graceful degradation
- **Timeout handling** for long-running operations
- **Audit logging** for tracking errors and operations

## Components

### 1. Error Response Models (`neuron_system/utils/errors.py`)

#### ErrorCode Enum

Defines standardized error codes organized by category:

- **Compression errors (1xxx)**: E1001-E1004
- **Query errors (2xxx)**: E2001-E2005
- **Training errors (3xxx)**: E3001-E3006
- **Tool execution errors (4xxx)**: E4001-E4007
- **Storage errors (5xxx)**: E5001-E5008
- **Neuron errors (6xxx)**: E6001-E6006
- **Synapse errors (7xxx)**: E7001-E7005
- **Spatial index errors (8xxx)**: E8001-E8004
- **API errors (9xxx)**: E9001-E9005
- **General errors (0xxx)**: E0001-E0004

#### ErrorResponse Dataclass

Standardized error response structure:

```python
@dataclass
class ErrorResponse:
    error_code: str
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    recoverable: bool
    recovery_suggestions: list[str]
```

#### Custom Exception Classes

Specialized exception classes for different error categories:

- `NeuronSystemException` - Base exception
- `CompressionError` - Compression failures
- `QueryError` - Query execution failures
- `TrainingError` - Training operation failures
- `ToolExecutionError` - Tool execution failures
- `StorageError` - Storage operation failures
- `NeuronError` - Neuron operation failures
- `SynapseError` - Synapse operation failures
- `SpatialIndexError` - Spatial index failures
- `APIError` - API operation failures

### 2. Retry Logic (`neuron_system/utils/retry.py`)

#### Exponential Backoff Decorator

Automatically retry failed operations with exponential backoff:

```python
@exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
def save_neuron(neuron):
    # Operation that may fail transiently
    return neuron_store.create(neuron)
```

#### Fallback Decorator

Provide fallback values when operations fail:

```python
@with_fallback(fallback_value=np.zeros(384))
def compress_data(data):
    # Operation that may fail
    return compression_engine.compress(data)
```

#### Timeout Decorator

Add timeout protection to operations:

```python
@with_timeout(timeout_seconds=5.0)
def execute_query(query_text):
    # Long-running operation
    return query_engine.query(query_text)
```

#### Circuit Breaker

Prevent cascading failures by stopping requests to failing services:

```python
circuit_breaker = get_circuit_breaker(
    name="tool_execution",
    failure_threshold=5,
    recovery_timeout=60.0
)

result = circuit_breaker.call(tool_neuron.execute, inputs)
```

Circuit breaker states:
- **CLOSED**: Normal operation
- **OPEN**: Service failing, reject requests
- **HALF_OPEN**: Testing if service recovered

#### RetryStrategy Class

Configurable retry strategy for complex scenarios:

```python
strategy = RetryStrategy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exponential_base=2.0
)

result = strategy.execute(
    storage_operation,
    arg1, arg2,
    exceptions=(StorageError,)
)
```

### 3. Logging Configuration (`neuron_system/utils/logging_config.py`)

#### Setup Logging

Configure logging with file rotation and audit logging:

```python
from neuron_system.utils import setup_logging

setup_logging(
    log_level="INFO",
    log_file="neuron_system.log",
    enable_audit=True,
    audit_log_file="neuron_system_audit.log"
)
```

#### Audit Logging

Track important operations for compliance and debugging:

```python
from neuron_system.utils import log_audit_event

log_audit_event(
    event_type="training_operation",
    details={
        "neuron_id": str(neuron.id),
        "operation": "adjust_vector",
        "learning_rate": 0.1
    },
    user_id="user123"
)
```

## Usage Examples

### Example 1: Compression with Fallback

```python
from neuron_system.engines.compression import CompressionEngine
from neuron_system.utils import create_error_response, ErrorCode

engine = CompressionEngine()

# Compression engine has built-in fallback
vector, metadata = engine.compress(data, raise_on_error=False)

if not metadata.get('success', True):
    error_response = create_error_response(
        error_code=ErrorCode.COMPRESSION_MODEL_FAILURE,
        message=f"Compression failed: {metadata.get('error')}",
        details=metadata,
        recoverable=True
    )
    # Vector is zero vector (fallback)
```

### Example 2: Storage with Retry

```python
from neuron_system.utils import exponential_backoff, StorageError, ErrorCode

@exponential_backoff(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    exceptions=(Exception,)
)
def save_neuron_with_retry(neuron_store, neuron):
    try:
        return neuron_store.create(neuron)
    except Exception as e:
        raise StorageError(
            error_code=ErrorCode.STORAGE_WRITE_FAILURE,
            message=f"Failed to save neuron {neuron.id}",
            details={"neuron_id": str(neuron.id), "error": str(e)},
            recoverable=True
        )

# Usage
try:
    save_neuron_with_retry(neuron_store, neuron)
except StorageError as e:
    error_response = e.to_error_response()
    print(f"Error: {error_response.message}")
    print(f"Suggestions: {error_response.recovery_suggestions}")
```

### Example 3: Query with Timeout

```python
from neuron_system.utils import with_timeout, QueryError, ErrorCode

@with_timeout(timeout_seconds=5.0)
def query_with_timeout(query_engine, query_text, **kwargs):
    try:
        results = query_engine.query(query_text, **kwargs)
        
        if not results:
            error_response = create_error_response(
                error_code=ErrorCode.QUERY_NO_RESULTS,
                message="Query returned no results",
                details={"query": query_text[:100]},
                recoverable=True
            )
        
        return results
    except Exception as e:
        raise QueryError(
            error_code=ErrorCode.QUERY_PROPAGATION_FAILURE,
            message=f"Query failed: {str(e)}",
            details={"query": query_text[:100]},
            recoverable=True
        )

# Usage
try:
    results = query_with_timeout(query_engine, "search query")
except QueryError as e:
    print(f"Query failed: {e.message}")
    for suggestion in e.recovery_suggestions:
        print(f"  - {suggestion}")
```

### Example 4: Tool Execution with Circuit Breaker

```python
from neuron_system.utils import get_circuit_breaker, ToolExecutionError, ErrorCode

def execute_tool_with_circuit_breaker(tool_neuron, inputs):
    circuit_breaker = get_circuit_breaker(
        name="tool_execution",
        failure_threshold=5,
        recovery_timeout=60.0
    )
    
    try:
        result = circuit_breaker.call(tool_neuron.execute, inputs)
        return result
    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            raise ToolExecutionError(
                error_code=ErrorCode.TOOL_EXECUTION_FAILURE,
                message="Tool execution service temporarily unavailable",
                details={"tool_id": str(tool_neuron.id)},
                recoverable=True,
                recovery_suggestions=[
                    "Wait for circuit breaker to reset",
                    "Check tool execution logs"
                ]
            )
        else:
            raise ToolExecutionError(
                error_code=ErrorCode.TOOL_EXECUTION_FAILURE,
                message=f"Tool execution failed: {str(e)}",
                details={"tool_id": str(tool_neuron.id), "error": str(e)},
                recoverable=True
            )

# Usage
try:
    result = execute_tool_with_circuit_breaker(tool_neuron, {"param": "value"})
except ToolExecutionError as e:
    if e.recoverable:
        print(f"Recoverable error: {e.message}")
        print("Recovery suggestions:")
        for suggestion in e.recovery_suggestions:
            print(f"  - {suggestion}")
```

### Example 5: Batch Operations with Retry Strategy

```python
from neuron_system.utils import RetryStrategy, create_error_response, ErrorCode

def batch_create_neurons_with_retry(neuron_graph, neurons):
    retry_strategy = RetryStrategy(
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0
    )
    
    results = {"successful": [], "failed": [], "total": len(neurons)}
    
    for neuron in neurons:
        try:
            retry_strategy.execute(
                neuron_graph.add_neuron,
                neuron,
                exceptions=(Exception,)
            )
            results["successful"].append(str(neuron.id))
        except Exception as e:
            error_response = create_error_response(
                error_code=ErrorCode.NEURON_CREATION_FAILURE,
                message=f"Failed to create neuron: {str(e)}",
                details={"neuron_id": str(neuron.id)},
                recoverable=False
            )
            results["failed"].append({
                "neuron_id": str(neuron.id),
                "error": str(e),
                "error_code": error_response.error_code
            })
    
    return results

# Usage
results = batch_create_neurons_with_retry(graph, neurons)
print(f"Created {len(results['successful'])} neurons")
print(f"Failed {len(results['failed'])} neurons")
```

### Example 6: Error Handling Wrapper

```python
from neuron_system.utils.error_integration import ErrorHandlingWrapper

# Wrap any component with error handling
wrapped_storage = ErrorHandlingWrapper(
    component=neuron_store,
    enable_retry=True,
    enable_circuit_breaker=True,
    circuit_breaker_name="storage_operations"
)

# Call methods with automatic error handling
try:
    wrapped_storage.call_method("create", neuron)
except Exception as e:
    print(f"Operation failed: {e}")
```

## Best Practices

### 1. Use Appropriate Error Codes

Always use the correct error code for the error category:

```python
# Good
raise CompressionError(
    error_code=ErrorCode.COMPRESSION_INVALID_INPUT,
    message="Input validation failed"
)

# Bad
raise Exception("Input validation failed")
```

### 2. Provide Detailed Error Information

Include relevant details in error responses:

```python
error_response = create_error_response(
    error_code=ErrorCode.STORAGE_WRITE_FAILURE,
    message="Failed to save neuron",
    details={
        "neuron_id": str(neuron.id),
        "neuron_type": neuron.neuron_type.value,
        "error": str(e),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
)
```

### 3. Use Retry for Transient Failures

Apply retry logic to operations that may fail transiently:

```python
# Good for: network operations, database writes, file I/O
@exponential_backoff(max_retries=3)
def save_to_database(data):
    return db.save(data)

# Bad for: validation errors, logic errors
# Don't retry operations that will always fail
```

### 4. Use Circuit Breakers for External Services

Protect against cascading failures:

```python
# Good for: tool execution, external APIs, remote services
circuit_breaker = get_circuit_breaker("external_service")
result = circuit_breaker.call(external_service.call, params)
```

### 5. Provide Fallback Values

Ensure system continues operating even when components fail:

```python
# Compression engine already has fallback
vector, metadata = engine.compress(data, raise_on_error=False)

# Custom fallback
@with_fallback(fallback_value=default_value)
def risky_operation():
    return potentially_failing_operation()
```

### 6. Log Errors Appropriately

Use audit logging for important operations:

```python
from neuron_system.utils import log_audit_event

try:
    training_engine.adjust_neuron(neuron_id, target_vector)
    log_audit_event(
        event_type="training_operation",
        details={"neuron_id": str(neuron_id), "operation": "adjust_vector"},
        user_id=current_user_id
    )
except TrainingError as e:
    error_response = e.to_error_response()
    error_response.log(level="error")
```

## Error Recovery Suggestions

The system provides automatic recovery suggestions for common errors:

| Error Code | Recovery Suggestions |
|------------|---------------------|
| E1001 (Compression Invalid Input) | Ensure input is valid string, check encoding |
| E1002 (Compression Model Failure) | Check model files, restart engine |
| E2002 (Query Timeout) | Reduce propagation depth, increase timeout |
| E3002 (Training Weight Out of Bounds) | Ensure weight is between -1.0 and 1.0 |
| E4003 (Tool Timeout) | Increase timeout, optimize tool code |
| E5001 (Storage Connection Failure) | Check database permissions, verify path |
| E5004 (Storage Disk Full) | Free up disk space, move database |
| E9003 (API Rate Limit Exceeded) | Reduce request frequency, implement batching |

## Testing

Comprehensive tests are available in `test_error_handling.py`:

```bash
# Run all error handling tests
python -m pytest test_error_handling.py -v

# Run specific test class
python -m pytest test_error_handling.py::TestCircuitBreaker -v

# Run with coverage
python -m pytest test_error_handling.py --cov=neuron_system.utils
```

## Integration with Existing Code

The error handling system is designed to integrate seamlessly with existing code:

1. **Compression Engine**: Already has built-in fallback mechanisms
2. **Storage Layer**: Can be wrapped with retry logic
3. **Query Engine**: Can be wrapped with timeout handling
4. **Tool Execution**: Can be protected with circuit breakers
5. **API Layer**: Can use error responses for consistent error handling

See `neuron_system/utils/error_integration.py` for integration examples.

## Summary

The comprehensive error handling system provides:

✅ **Standardized error responses** with codes and recovery suggestions  
✅ **Automatic retry logic** with exponential backoff  
✅ **Circuit breaker protection** against cascading failures  
✅ **Fallback mechanisms** for graceful degradation  
✅ **Timeout handling** for long-running operations  
✅ **Audit logging** for compliance and debugging  
✅ **28 passing tests** verifying all functionality  

All error handling utilities are available through:

```python
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
```
