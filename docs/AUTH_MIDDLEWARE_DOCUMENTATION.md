# Authentication and Middleware Documentation

## Overview

The 3D Synaptic Neuron System API includes comprehensive authentication and middleware functionality to ensure secure, monitored, and rate-limited access to the system.

## Features Implemented

### 1. API Key Authentication (`neuron_system/api/auth.py`)

#### API Key Management

**Create API Key:**
```python
from neuron_system.api.auth import create_api_key

# Create a new API key with custom rate limit
api_key = create_api_key("My Application", rate_limit=1000)
# Returns: "nsk-<random-token>"
```

**Revoke API Key:**
```python
from neuron_system.api.auth import revoke_api_key

success = revoke_api_key(api_key)
# Returns: True if revoked, False if not found
```

**Using API Keys in Requests:**
```bash
curl -H "X-API-Key: nsk-your-api-key-here" http://localhost:8000/api/v1/neurons
```

#### API Key Features
- Automatic generation with `nsk-` prefix (neuron system key)
- Per-key rate limiting configuration
- Creation timestamp tracking
- In-memory storage (can be extended to database)

### 2. JWT Token Authentication (`neuron_system/api/auth.py`)

#### Token Creation

```python
from neuron_system.api.auth import create_access_token
from datetime import timedelta

# Create token with default expiration (60 minutes)
token = create_access_token({"sub": "user123", "permissions": ["read", "write"]})

# Create token with custom expiration
token = create_access_token(
    {"sub": "user123"}, 
    expires_delta=timedelta(hours=24)
)
```

#### Token Verification

```python
from neuron_system.api.auth import decode_access_token

try:
    payload = decode_access_token(token)
    user_id = payload["sub"]
    permissions = payload.get("permissions", [])
except HTTPException as e:
    # Handle invalid or expired token
    print(f"Token error: {e.detail}")
```

#### Using JWT Tokens in Requests

```bash
curl -H "Authorization: Bearer your-jwt-token-here" http://localhost:8000/api/v1/neurons
```

#### JWT Features
- HS256 algorithm for signing
- Configurable expiration time
- Automatic expiration validation
- Support for custom claims (permissions, roles, etc.)

### 3. Flexible Authentication

The system supports multiple authentication methods:

**API Key Only:**
```python
from fastapi import Depends
from neuron_system.api.auth import verify_api_key

@app.get("/endpoint")
async def endpoint(auth = Depends(verify_api_key)):
    # auth contains API key info
    pass
```

**JWT Token Only:**
```python
from fastapi import Depends
from neuron_system.api.auth import verify_token

@app.get("/endpoint")
async def endpoint(auth = Depends(verify_token)):
    # auth contains decoded token payload
    pass
```

**Optional Authentication:**
```python
from fastapi import Depends
from neuron_system.api.auth import optional_auth

@app.get("/endpoint")
async def endpoint(auth = Depends(optional_auth)):
    # auth is None if no credentials provided
    # auth contains info if API key or token provided
    pass
```

### 4. Rate Limiting Middleware (`neuron_system/api/middleware.py`)

#### Configuration

```python
from neuron_system.api.middleware import RateLimiter

# Create rate limiter with custom settings
limiter = RateLimiter(
    default_limit=100,      # 100 requests
    window_seconds=3600     # per hour
)
```

#### Rate Limiting Behavior

- **Per-API-Key Limiting**: Each API key has its own rate limit
- **IP-Based Limiting**: Requests without API keys are limited by IP address
- **Custom Limits**: API keys can have custom rate limits
- **Response Headers**: All responses include rate limit information

**Rate Limit Headers:**
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 2025-10-26T14:30:00
```

**Rate Limit Exceeded Response:**
```json
{
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Rate limit exceeded",
  "details": {
    "limit": 1000,
    "remaining": 0,
    "reset": "2025-10-26T14:30:00",
    "retry_after": 3456
  },
  "timestamp": "2025-10-26T13:30:00"
}
```

**HTTP Status:** 429 Too Many Requests

### 5. Request Logging Middleware

#### Features

- **Request Logging**: Logs all incoming requests with method, path, and client IP
- **Response Logging**: Logs response status and processing time
- **Request ID Tracking**: Assigns unique ID to each request for tracing
- **Error Logging**: Detailed error logging with stack traces

#### Log Format

```
INFO: Request started: GET /api/v1/neurons [req-1698765432000] from 127.0.0.1
INFO: Request completed: GET /api/v1/neurons [req-1698765432000] status=200 time=45.23ms
```

#### Request ID Header

All responses include the request ID:
```
X-Request-ID: req-1698765432000
```

### 6. Security Headers Middleware

Automatically adds security headers to all responses:

```
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### 7. Request Size Limiting

- **Maximum Size**: 10 MB per request
- **Automatic Rejection**: Requests exceeding limit return 413 status
- **Protection**: Prevents memory exhaustion attacks

**Error Response:**
```json
{
  "error_code": "REQUEST_TOO_LARGE",
  "message": "Request body too large. Maximum size is 10485760 bytes",
  "timestamp": "2025-10-26T13:30:00"
}
```

### 8. Error Tracking

#### Features

- **Error Counting**: Tracks error occurrences by type
- **Last Occurrence**: Records timestamp of last error
- **Statistics**: Provides error statistics for monitoring

#### Usage

```python
from neuron_system.api.middleware import error_tracker

# Get error statistics
stats = error_tracker.get_error_stats()
# Returns:
# {
#   "HTTP_404": {"count": 15, "last_occurrence": "2025-10-26T13:30:00"},
#   "ValueError": {"count": 3, "last_occurrence": "2025-10-26T13:25:00"}
# }
```

### 9. CORS Configuration

The API includes CORS middleware for cross-origin requests:

```python
# Configured in app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 10. Health Check Bypass

Health check endpoints bypass rate limiting and detailed logging:

- `/health`
- `/`
- `/docs`
- `/redoc`
- `/openapi.json`

## Middleware Execution Order

Middleware is executed in the following order (reverse of registration):

1. **Health Check Bypass**: Skip middleware for health endpoints
2. **Request Logging**: Log incoming request
3. **Rate Limiting**: Check and enforce rate limits
4. **Request Size Limiting**: Validate request size
5. **Security Headers**: Add security headers
6. **CORS Cache**: Cache CORS preflight requests
7. **Error Tracking**: Track errors for monitoring

## Complete Example

### Server Setup

```python
from fastapi import FastAPI, Depends
from neuron_system.api.auth import verify_api_key, create_api_key

app = FastAPI()

# Create an API key for testing
test_key = create_api_key("Test Application", rate_limit=100)
print(f"API Key: {test_key}")

@app.get("/protected")
async def protected_endpoint(auth = Depends(verify_api_key)):
    return {
        "message": "Access granted",
        "authenticated_as": auth["name"]
    }
```

### Client Usage

```python
import requests

# Using API Key
response = requests.get(
    "http://localhost:8000/protected",
    headers={"X-API-Key": "nsk-your-key-here"}
)

# Using JWT Token
from neuron_system.api.auth import create_access_token

token = create_access_token({"sub": "user123"})
response = requests.get(
    "http://localhost:8000/protected",
    headers={"Authorization": f"Bearer {token}"}
)
```

## Production Considerations

### Security

1. **Secret Key**: Use environment variable for JWT secret key
   ```python
   import os
   SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback-secret")
   ```

2. **API Key Storage**: Move to database instead of in-memory
   ```python
   # Use database for production
   api_keys = database.query("SELECT * FROM api_keys WHERE key = ?", [api_key])
   ```

3. **CORS Origins**: Restrict allowed origins
   ```python
   allow_origins=["https://yourdomain.com"]
   ```

### Rate Limiting

1. **Distributed Rate Limiting**: Use Redis for multi-instance deployments
   ```python
   import redis
   redis_client = redis.Redis(host='localhost', port=6379)
   ```

2. **Custom Limits**: Set different limits for different endpoints
   ```python
   # High limit for read operations
   # Low limit for write operations
   ```

### Monitoring

1. **Logging**: Use structured logging with log aggregation
   ```python
   import structlog
   logger = structlog.get_logger()
   ```

2. **Metrics**: Export metrics to Prometheus
   ```python
   from prometheus_client import Counter, Histogram
   request_count = Counter('api_requests_total', 'Total requests')
   ```

3. **Alerting**: Set up alerts for rate limit violations and errors
   ```python
   if error_count > threshold:
       send_alert("High error rate detected")
   ```

## Testing

Run the authentication and middleware tests:

```bash
python -m pytest test_auth_middleware.py -v
```

### Test Coverage

- ✅ API key creation and revocation
- ✅ API key authentication
- ✅ JWT token creation and decoding
- ✅ Token expiration handling
- ✅ Rate limiting (basic and via API)
- ✅ Request logging
- ✅ Security headers
- ✅ Request size limiting
- ✅ Error tracking
- ✅ Full authentication flow

## Requirements Satisfied

This implementation satisfies the following requirements:

- **Requirement 15.1**: Modular code structure with clear responsibilities
- **Requirement 16.4**: API key authentication support
- **Requirement 16.5**: Rate limiting with configurable limits

## Summary

The authentication and middleware system provides:

1. **Flexible Authentication**: API keys and JWT tokens
2. **Rate Limiting**: Per-key and IP-based limiting
3. **Request Monitoring**: Comprehensive logging and tracking
4. **Security**: Multiple security headers and protections
5. **Production-Ready**: Extensible for database storage and distributed systems

All components are tested and ready for production use with appropriate configuration.
