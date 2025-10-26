# Documentation Index

Complete documentation for the 3D Synaptic Neuron System.

## Quick Links

- [Main README](../README.md) - Project overview and quick start
- [API Documentation](api/README.md) - REST API reference
- [SDK Documentation](../neuron_system/sdk/README.md) - Python SDK client

## Implementation Guides

Detailed implementation documentation for each major component:

### Core Systems

- [Storage Implementation](implementation/STORAGE_IMPLEMENTATION.md)
  - Database schema and persistence
  - Neuron and synapse stores
  - Serialization and backup
  - Incremental saves and integrity checks

- [Query Engine Implementation](implementation/QUERY_ENGINE_IMPLEMENTATION.md)
  - Semantic search with vector similarity
  - Activation propagation through synapses
  - Spatial filtering and ranking
  - Performance optimization

- [Training Engine Implementation](implementation/TRAINING_ENGINE_IMPLEMENTATION.md)
  - Hebbian-like learning
  - Weight adjustment algorithms
  - Rollback mechanisms
  - Training validation

### Advanced Features

- [Tool Neuron Implementation](implementation/TOOL_NEURON_IMPLEMENTATION.md)
  - Executable Python code as neurons
  - Input/output schemas
  - Sandboxed execution
  - Error handling

- [Tool Cluster Implementation](implementation/TOOL_CLUSTER_IMPLEMENTATION.md)
  - Directed acyclic graphs (DAGs)
  - Execution orchestration
  - Data flow between tools
  - Cluster management

- [Lifecycle Management](implementation/LIFECYCLE_MANAGEMENT_IMPLEMENTATION.md)
  - Fast neuron creation
  - Automatic positioning
  - Batch operations
  - Synapse pruning

- [Visualization Implementation](implementation/VISUALIZATION_IMPLEMENTATION.md)
  - 3D rendering data export
  - Three.js compatible format
  - Cluster visualization
  - Real-time updates

## System Documentation

### Error Handling

- [Error Handling Documentation](ERROR_HANDLING_DOCUMENTATION.md)
  - Error codes and categories
  - Retry logic with exponential backoff
  - Circuit breaker pattern
  - Fallback mechanisms
  - Recovery suggestions

### API & Authentication

- [API Implementation Summary](API_IMPLEMENTATION_SUMMARY.md)
  - REST API overview
  - Endpoint reference
  - Request/response formats
  - Rate limiting

- [Authentication Middleware](AUTH_MIDDLEWARE_DOCUMENTATION.md)
  - API key authentication
  - Key generation and validation
  - Security best practices
  - Rate limiting

- [SDK Implementation Summary](SDK_IMPLEMENTATION_SUMMARY.md)
  - Python SDK client
  - Usage examples
  - Error handling
  - Best practices

### Summaries

- [Error Handling Summary](ERROR_HANDLING_SUMMARY.md) - Quick reference
- [Visualization Summary](VISUALIZATION_SUMMARY.md) - Quick reference

## Examples

All examples are in the [`examples/`](../examples/) directory:

### Basic Usage
- `example_storage_usage.py` - Storage and persistence basics
- `example_sdk_usage.py` - Using the Python SDK client

### Advanced Features
- `example_tool_neuron_usage.py` - Creating and executing tool neurons
- `example_tool_cluster_usage.py` - Tool cluster orchestration
- `example_lifecycle_usage.py` - Neuron lifecycle management
- `example_visualization_usage.py` - Visualization data export
- `example_error_handling_usage.py` - Error handling patterns

## Testing

All tests are in the [`tests/`](../tests/) directory:

### Core Tests
- `test_basic_functionality.py` - Core neuron/synapse functionality
- `test_storage.py` - Storage layer and persistence
- `test_spatial_index.py` - 3D spatial indexing

### Engine Tests
- `test_query_engine.py` - Query and search functionality
- `test_training_engine.py` - Training and learning

### Feature Tests
- `test_tool_neuron.py` - Tool neuron execution
- `test_tool_cluster.py` - Tool cluster orchestration
- `test_lifecycle_management.py` - Lifecycle operations

### API Tests
- `test_api.py` - REST API endpoints
- `test_training_api.py` - Training API endpoints
- `test_auth_middleware.py` - Authentication middleware
- `test_visualization.py` - Visualization endpoints
- `test_sdk.py` - SDK client

### System Tests
- `test_error_handling.py` - Error handling system

### Validation Scripts
- `validate_api_structure.py` - API structure validation
- `validate_training_routes.py` - Training routes validation

## Architecture Overview

### Component Hierarchy

```
NeuronGraph (Central Hub)
├── Neurons (Knowledge, Tool)
├── Synapses (Weighted Connections)
├── Spatial Index (Octree)
├── Storage Layer
│   ├── Database Manager
│   ├── Neuron Store
│   ├── Synapse Store
│   └── Serialization Manager
├── Engines
│   ├── Compression Engine
│   ├── Query Engine
│   ├── Training Engine
│   └── Activation Engine
├── Tools
│   ├── Tool Neurons
│   ├── Tool Clusters
│   └── Tool Executor
├── API Layer
│   ├── REST Endpoints
│   ├── Authentication
│   └── Middleware
└── Utils
    ├── Error Handling
    ├── Retry Logic
    ├── Circuit Breakers
    └── Logging
```

### Data Flow

1. **Input** → Compression Engine → 384D Vector
2. **Query** → Query Engine → Semantic Search
3. **Activation** → Propagation through Synapses
4. **Training** → Weight Adjustment
5. **Storage** → Persistence to SQLite

## Configuration

System configuration is in `neuron_system/config/settings.py`:

```python
from neuron_system.config.settings import get_settings, update_settings

# Get current settings
settings = get_settings()

# Update settings
update_settings(
    vector_dimensions=384,
    default_top_k=10,
    query_timeout_seconds=5.0,
    database_path="custom.db"
)
```

### Key Configuration Options

- **Spatial**: bounds, positioning
- **Vectors**: dimensions (384), thresholds
- **Synapses**: weight ranges, pruning thresholds
- **Training**: learning rate
- **Query**: top_k, propagation depth, timeout
- **Performance**: pool sizes, batch sizes
- **Storage**: database path, auto-save
- **Compression**: model, timeout
- **Tools**: execution timeout, sandboxing
- **API**: host, port, authentication, rate limiting
- **Logging**: level, audit logging

## Best Practices

### Performance
- Use batch operations for bulk neuron/synapse creation
- Configure appropriate propagation depth (default: 3)
- Enable auto-save for persistence
- Use UUID pool for fast ID generation

### Error Handling
- Use error codes for consistent error handling
- Apply retry logic to transient failures
- Use circuit breakers for external services
- Implement fallback mechanisms

### Security
- Always use API key authentication in production
- Enable rate limiting
- Use tool sandboxing for untrusted code
- Enable audit logging

### Storage
- Regular backups with `serialization_manager.create_backup()`
- Verify integrity with `verify_integrity()`
- Use incremental saves for performance
- Prune weak synapses periodically

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt
```

**Database Locked**
- Close other connections
- Check file permissions
- Use `enable_auto_save=False` for testing

**Query Timeout**
- Reduce propagation depth
- Increase timeout in settings
- Check system resources

**Compression Failures**
- Validate input text
- Check model files
- Increase compression timeout

**Tool Execution Errors**
- Validate input schema
- Check tool code syntax
- Enable sandboxing
- Increase execution timeout

See [Error Handling Documentation](ERROR_HANDLING_DOCUMENTATION.md) for detailed error recovery.

## Getting Help

1. Check this documentation index
2. Review relevant implementation guides
3. Look at examples in `examples/`
4. Run tests to verify setup
5. Check error logs and audit logs
6. Open an issue on GitHub

## Contributing to Documentation

When adding new features:
1. Update relevant implementation guide
2. Add examples to `examples/`
3. Add tests to `tests/`
4. Update this index
5. Update main README if needed

## Version History

- **v1.0** - Initial implementation
  - Core neuron/synapse system
  - 3D spatial indexing
  - Query and training engines
  - Storage layer
  - Tool neurons and clusters
  - REST API with authentication
  - Comprehensive error handling
  - Visualization export
  - Python SDK client
