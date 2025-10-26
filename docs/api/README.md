# 3D Synaptic Neuron System API

REST API for the 3D Synaptic Neuron System - a novel knowledge representation system using 3D neural networks.

## Quick Start

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the API server:
```bash
python run_api.py
```

The API will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Authentication

The API supports two authentication methods:

### 1. API Key Authentication

Include your API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/neurons
```

**Default Development Key**: `dev-key-12345`

### 2. JWT Token Authentication

Include a JWT token in the `Authorization` header:

```bash
curl -H "Authorization: Bearer your-jwt-token" http://localhost:8000/api/v1/neurons
```

## API Endpoints

### General

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint

### Neurons

- `POST /api/v1/neurons` - Create a new neuron
- `POST /api/v1/neurons/batch` - Create multiple neurons
- `GET /api/v1/neurons/{id}` - Get neuron by ID
- `DELETE /api/v1/neurons/{id}` - Delete neuron

### Synapses

- `POST /api/v1/synapses` - Create a new synapse
- `GET /api/v1/synapses` - Query synapses (with filters)
- `GET /api/v1/synapses/{id}` - Get synapse by ID
- `DELETE /api/v1/synapses/{id}` - Delete synapse

### Query

- `POST /api/v1/query` - Execute knowledge query
- `POST /api/v1/query/spatial` - Execute spatial query
- `GET /api/v1/neurons/{id}/neighbors` - Get connected neurons

### Training

- `POST /api/v1/training/adjust-neuron` - Adjust neuron vector
- `POST /api/v1/training/adjust-synapse` - Modify synapse weight
- `POST /api/v1/training/create-tool` - Create tool neuron

## Usage Examples

### Create a Knowledge Neuron

```bash
curl -X POST "http://localhost:8000/api/v1/neurons" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "neuron_type": "knowledge",
    "source_data": "Python is a high-level programming language",
    "semantic_tags": ["programming", "python"],
    "metadata": {"category": "technology"}
  }'
```

### Create a Synapse

```bash
curl -X POST "http://localhost:8000/api/v1/synapses" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "source_neuron_id": "source-uuid",
    "target_neuron_id": "target-uuid",
    "weight": 0.8,
    "synapse_type": "KNOWLEDGE"
  }'
```

### Execute a Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "query_text": "What is Python?",
    "top_k": 10,
    "propagation_depth": 3
  }'
```

### Adjust a Neuron

```bash
curl -X POST "http://localhost:8000/api/v1/training/adjust-neuron" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "neuron_id": "neuron-uuid",
    "target_text": "Updated knowledge about Python",
    "learning_rate": 0.1
  }'
```

### Create a Tool Neuron

```bash
curl -X POST "http://localhost:8000/api/v1/training/create-tool" \
  -H "X-API-Key: dev-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Calculate sum of two numbers",
    "function_signature": "add(a: int, b: int) -> int",
    "executable_code": "def add(a, b):\n    return a + b",
    "input_schema": {
      "type": "object",
      "properties": {
        "a": {"type": "integer"},
        "b": {"type": "integer"}
      }
    },
    "output_schema": {
      "type": "integer"
    }
  }'
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Default limit**: 1000 requests per hour
- **Custom limits**: Can be configured per API key

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when the rate limit resets

When rate limit is exceeded, the API returns `429 Too Many Requests`.

## Error Handling

The API uses standard HTTP status codes:

- `200 OK` - Successful request
- `201 Created` - Resource created successfully
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required or failed
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

Error responses follow this format:

```json
{
  "error_code": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {},
  "timestamp": "2024-01-01T12:00:00",
  "recoverable": true
}
```

## Security Features

- **CORS**: Configurable cross-origin resource sharing
- **Rate Limiting**: Prevents API abuse
- **Request Size Limiting**: Maximum 10MB per request
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, etc.
- **API Key Authentication**: Secure API access
- **JWT Token Support**: Session-based authentication

## Middleware

The API includes several middleware layers:

1. **Health Check Bypass**: Fast-path for health checks
2. **Logging**: Request/response logging
3. **Rate Limiting**: Request throttling
4. **Request Size Limiting**: Prevent large payloads
5. **Security Headers**: Add security headers
6. **CORS Cache**: Cache preflight requests
7. **Error Tracking**: Monitor errors

## Development

### Running Tests

```bash
pytest test_*.py
```

### Code Formatting

```bash
black neuron_system/
```

### Type Checking

```bash
mypy neuron_system/
```

## Production Deployment

For production deployment:

1. Set a secure `SECRET_KEY` in `neuron_system/api/auth.py`
2. Use a proper database for API key storage
3. Configure CORS origins appropriately
4. Use Redis for distributed rate limiting
5. Enable HTTPS/TLS
6. Set up monitoring and alerting
7. Use a production ASGI server (e.g., Gunicorn with Uvicorn workers)

Example production command:

```bash
gunicorn neuron_system.api.app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Architecture

The API follows a layered architecture:

```
API Layer (FastAPI)
    ↓
Engine Layer (Query, Training, Compression)
    ↓
Core Layer (NeuronGraph, Neurons, Synapses)
    ↓
Storage Layer (Database, Stores)
```

## License

See LICENSE file for details.
