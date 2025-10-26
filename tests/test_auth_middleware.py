"""
Test authentication and middleware functionality
"""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
import time

from neuron_system.api.app import app, app_state
from neuron_system.api.auth import (
    create_api_key,
    create_access_token,
    decode_access_token,
    revoke_api_key,
    VALID_API_KEYS
)
from neuron_system.api.middleware import rate_limiter, error_tracker


# ============================================================================
# API Key Authentication Tests
# ============================================================================

def test_create_api_key():
    """Test API key creation"""
    api_key = create_api_key("Test Key", rate_limit=500)
    
    assert api_key.startswith("nsk-")
    assert api_key in VALID_API_KEYS
    assert VALID_API_KEYS[api_key]["name"] == "Test Key"
    assert VALID_API_KEYS[api_key]["rate_limit"] == 500


def test_revoke_api_key():
    """Test API key revocation"""
    api_key = create_api_key("Revoke Test", rate_limit=100)
    assert api_key in VALID_API_KEYS
    
    result = revoke_api_key(api_key)
    assert result is True
    assert api_key not in VALID_API_KEYS
    
    # Try revoking again
    result = revoke_api_key(api_key)
    assert result is False


def test_api_key_authentication():
    """Test API key authentication via endpoint"""
    client = TestClient(app)
    
    # Create a test API key
    api_key = create_api_key("Test Auth", rate_limit=1000)
    
    # Test with valid API key
    response = client.get("/health", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    
    # Test with invalid API key
    response = client.get("/health", headers={"X-API-Key": "invalid-key"})
    # Health endpoint doesn't require auth, so it should still work
    assert response.status_code == 200


# ============================================================================
# JWT Token Tests
# ============================================================================

def test_create_access_token():
    """Test JWT token creation"""
    data = {"sub": "testuser", "permissions": ["read", "write"]}
    token = create_access_token(data)
    
    assert isinstance(token, str)
    assert len(token) > 0


def test_decode_access_token():
    """Test JWT token decoding"""
    data = {"sub": "testuser", "permissions": ["read"]}
    token = create_access_token(data)
    
    decoded = decode_access_token(token)
    assert decoded["sub"] == "testuser"
    assert decoded["permissions"] == ["read"]
    assert "exp" in decoded


def test_expired_token():
    """Test expired token handling"""
    data = {"sub": "testuser"}
    # Create token that expires immediately
    token = create_access_token(data, expires_delta=timedelta(seconds=-1))
    
    with pytest.raises(Exception) as exc_info:
        decode_access_token(token)
    
    assert "expired" in str(exc_info.value).lower()


# ============================================================================
# Rate Limiting Tests
# ============================================================================

def test_rate_limiter_basic():
    """Test basic rate limiting"""
    limiter = rate_limiter
    test_key = "test-rate-limit-1"
    
    # First request should be allowed
    allowed, info = limiter.is_allowed(test_key, limit=5)
    assert allowed is True
    assert info["remaining"] == 4
    
    # Make more requests
    for i in range(4):
        allowed, info = limiter.is_allowed(test_key, limit=5)
        assert allowed is True
    
    # 6th request should be blocked
    allowed, info = limiter.is_allowed(test_key, limit=5)
    assert allowed is False
    assert info["remaining"] == 0
    assert "retry_after" in info


def test_rate_limit_middleware():
    """Test rate limiting via API"""
    client = TestClient(app)
    
    # Create API key with low rate limit
    api_key = create_api_key("Rate Limit Test", rate_limit=3)
    
    # Make requests up to the limit
    for i in range(3):
        response = client.get("/health", headers={"X-API-Key": api_key})
        assert response.status_code == 200
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
    
    # Next request should be rate limited
    response = client.get("/health", headers={"X-API-Key": api_key})
    assert response.status_code == 429
    assert "Retry-After" in response.headers


# ============================================================================
# Request Logging Tests
# ============================================================================

def test_request_logging():
    """Test that requests are logged"""
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers


def test_process_time_header():
    """Test that process time is added to response"""
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Process-Time-Ms" in response.headers
    
    process_time = float(response.headers["X-Process-Time-Ms"])
    assert process_time >= 0


# ============================================================================
# Security Headers Tests
# ============================================================================

def test_security_headers():
    """Test that security headers are added"""
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    
    # Check security headers
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "Strict-Transport-Security" in response.headers


# ============================================================================
# Request Size Limiting Tests
# ============================================================================

def test_request_size_limit():
    """Test request size limiting"""
    client = TestClient(app)
    
    # Create a large payload (larger than 10MB limit)
    large_data = "x" * (11 * 1024 * 1024)  # 11 MB
    
    response = client.post(
        "/api/v1/neurons",
        json={"data": large_data},
        headers={"Content-Length": str(len(large_data))}
    )
    
    # Should be rejected due to size
    assert response.status_code == 413


# ============================================================================
# Error Tracking Tests
# ============================================================================

def test_error_tracking():
    """Test error tracking functionality"""
    tracker = error_tracker
    
    # Track some errors
    tracker.track_error("TestError", {"message": "test error 1"})
    tracker.track_error("TestError", {"message": "test error 2"})
    tracker.track_error("OtherError", {"message": "other error"})
    
    # Get stats
    stats = tracker.get_error_stats()
    assert "TestError" in stats
    assert stats["TestError"]["count"] == 2
    assert "OtherError" in stats
    assert stats["OtherError"]["count"] == 1


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_auth_flow():
    """Test complete authentication flow"""
    client = TestClient(app)
    
    # Create API key
    api_key = create_api_key("Integration Test", rate_limit=100)
    
    # Make authenticated request
    response = client.get("/health", headers={"X-API-Key": api_key})
    assert response.status_code == 200
    
    # Check all expected headers
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time-Ms" in response.headers
    assert "X-Content-Type-Options" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
