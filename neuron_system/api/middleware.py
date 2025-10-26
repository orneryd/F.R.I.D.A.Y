"""
Middleware for request processing, rate limiting, and logging
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Tuple
import time
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter
    
    In production, use Redis or similar distributed cache
    """
    
    def __init__(self, default_limit: int = 100, window_seconds: int = 3600):
        """
        Initialize rate limiter
        
        Args:
            default_limit: Default number of requests allowed per window
            window_seconds: Time window in seconds (default: 1 hour)
        """
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def _clean_old_requests(self, key: str, now: datetime):
        """Remove requests outside the current window"""
        cutoff = now - timedelta(seconds=self.window_seconds)
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if req_time > cutoff
        ]
    
    def is_allowed(self, key: str, limit: int = None) -> Tuple[bool, dict]:
        """
        Check if request is allowed under rate limit
        
        Args:
            key: Identifier for rate limiting (e.g., API key or IP)
            limit: Optional custom limit (uses default if not provided)
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        now = datetime.now()
        limit = limit or self.default_limit
        
        # Clean old requests
        self._clean_old_requests(key, now)
        
        # Check current count
        current_count = len(self.requests[key])
        
        if current_count >= limit:
            # Rate limit exceeded
            oldest_request = min(self.requests[key])
            reset_time = oldest_request + timedelta(seconds=self.window_seconds)
            
            return False, {
                "limit": limit,
                "remaining": 0,
                "reset": reset_time.isoformat(),
                "retry_after": int((reset_time - now).total_seconds())
            }
        
        # Allow request
        self.requests[key].append(now)
        
        return True, {
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset": (now + timedelta(seconds=self.window_seconds)).isoformat()
        }


# Global rate limiter instance
rate_limiter = RateLimiter(default_limit=1000, window_seconds=3600)


async def rate_limit_middleware(request: Request, call_next):
    """
    Rate limiting middleware
    
    Limits requests based on API key or IP address
    """
    # Extract identifier for rate limiting
    api_key = request.headers.get("X-API-Key")
    identifier = api_key if api_key else request.client.host
    
    # Get custom limit from API key if available
    custom_limit = None
    if api_key:
        from neuron_system.api.auth import VALID_API_KEYS
        if api_key in VALID_API_KEYS:
            custom_limit = VALID_API_KEYS[api_key].get("rate_limit")
    
    # Check rate limit
    is_allowed, rate_info = rate_limiter.is_allowed(identifier, custom_limit)
    
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for {identifier}")
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error_code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded",
                "details": rate_info,
                "timestamp": datetime.now().isoformat()
            },
            headers={
                "X-RateLimit-Limit": str(rate_info["limit"]),
                "X-RateLimit-Remaining": str(rate_info["remaining"]),
                "X-RateLimit-Reset": rate_info["reset"],
                "Retry-After": str(rate_info["retry_after"])
            }
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers to response
    response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
    response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
    response.headers["X-RateLimit-Reset"] = rate_info["reset"]
    
    return response


# ============================================================================
# Request Logging
# ============================================================================

async def logging_middleware(request: Request, call_next):
    """
    Request logging middleware
    
    Logs all incoming requests and their processing time
    """
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", f"req-{int(start_time * 1000)}")
    
    # Log request
    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"[{request_id}] from {request.client.host}"
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"[{request_id}] status={response.status_code} time={process_time:.2f}ms"
        )
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(
            f"Request failed: {request.method} {request.url.path} "
            f"[{request_id}] error={str(e)} time={process_time:.2f}ms",
            exc_info=True
        )
        raise


# ============================================================================
# Security Headers
# ============================================================================

async def security_headers_middleware(request: Request, call_next):
    """
    Add security headers to all responses
    """
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# ============================================================================
# Request Size Limiting
# ============================================================================

MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB


async def request_size_middleware(request: Request, call_next):
    """
    Limit request body size to prevent memory exhaustion
    """
    content_length = request.headers.get("content-length")
    
    if content_length:
        content_length = int(content_length)
        if content_length > MAX_REQUEST_SIZE:
            logger.warning(
                f"Request too large: {content_length} bytes from {request.client.host}"
            )
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={
                    "error_code": "REQUEST_TOO_LARGE",
                    "message": f"Request body too large. Maximum size is {MAX_REQUEST_SIZE} bytes",
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    return await call_next(request)


# ============================================================================
# CORS Preflight Cache
# ============================================================================

async def cors_cache_middleware(request: Request, call_next):
    """
    Add cache headers for CORS preflight requests
    """
    response = await call_next(request)
    
    if request.method == "OPTIONS":
        # Cache preflight requests for 24 hours
        response.headers["Access-Control-Max-Age"] = "86400"
    
    return response


# ============================================================================
# Health Check Bypass
# ============================================================================

async def health_check_bypass_middleware(request: Request, call_next):
    """
    Bypass rate limiting and logging for health check endpoints
    """
    if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
        # Skip middleware processing for health checks
        return await call_next(request)
    
    # Continue with normal middleware chain
    return await call_next(request)


# ============================================================================
# Error Tracking
# ============================================================================

class ErrorTracker:
    """Track errors for monitoring and alerting"""
    
    def __init__(self):
        self.errors: Dict[str, list] = defaultdict(list)
    
    def track_error(self, error_type: str, details: dict):
        """Track an error occurrence"""
        self.errors[error_type].append({
            "timestamp": datetime.now(),
            "details": details
        })
        
        # Keep only last 100 errors per type
        if len(self.errors[error_type]) > 100:
            self.errors[error_type] = self.errors[error_type][-100:]
    
    def get_error_stats(self) -> dict:
        """Get error statistics"""
        stats = {}
        for error_type, occurrences in self.errors.items():
            stats[error_type] = {
                "count": len(occurrences),
                "last_occurrence": occurrences[-1]["timestamp"].isoformat() if occurrences else None
            }
        return stats


# Global error tracker
error_tracker = ErrorTracker()


async def error_tracking_middleware(request: Request, call_next):
    """
    Track errors for monitoring
    """
    try:
        response = await call_next(request)
        
        # Track 4xx and 5xx responses
        if response.status_code >= 400:
            error_tracker.track_error(
                f"HTTP_{response.status_code}",
                {
                    "path": request.url.path,
                    "method": request.method,
                    "client": request.client.host
                }
            )
        
        return response
        
    except Exception as e:
        error_tracker.track_error(
            type(e).__name__,
            {
                "path": request.url.path,
                "method": request.method,
                "error": str(e)
            }
        )
        raise
