"""
Authentication and authorization for the API
"""
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional
import jwt
import secrets
import logging

logger = logging.getLogger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

# Configuration (should be loaded from settings in production)
SECRET_KEY = secrets.token_urlsafe(32)  # Generate a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# In-memory API key storage (should use database in production)
VALID_API_KEYS = {
    "dev-key-12345": {
        "name": "Development Key",
        "created_at": datetime.now(),
        "rate_limit": 1000  # requests per hour
    }
}


# ============================================================================
# API Key Authentication
# ============================================================================

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """
    Verify API key authentication
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        dict: API key information
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    if api_key not in VALID_API_KEYS:
        logger.warning(f"Invalid API key attempted: {api_key[:10]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"}
        )
    
    key_info = VALID_API_KEYS[api_key]
    logger.debug(f"API key authenticated: {key_info['name']}")
    
    return {
        "api_key": api_key,
        "name": key_info["name"],
        "rate_limit": key_info["rate_limit"]
    }


def create_api_key(name: str, rate_limit: int = 1000) -> str:
    """
    Create a new API key
    
    Args:
        name: Name/description for the API key
        rate_limit: Rate limit in requests per hour
        
    Returns:
        str: Generated API key
    """
    api_key = f"nsk-{secrets.token_urlsafe(32)}"  # nsk = neuron system key
    
    VALID_API_KEYS[api_key] = {
        "name": name,
        "created_at": datetime.now(),
        "rate_limit": rate_limit
    }
    
    logger.info(f"Created new API key: {name}")
    return api_key


def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key
    
    Args:
        api_key: API key to revoke
        
    Returns:
        bool: True if key was revoked, False if not found
    """
    if api_key in VALID_API_KEYS:
        key_info = VALID_API_KEYS[api_key]
        del VALID_API_KEYS[api_key]
        logger.info(f"Revoked API key: {key_info['name']}")
        return True
    return False


# ============================================================================
# JWT Token Authentication
# ============================================================================

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional expiration time delta
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT access token
    
    Args:
        token: JWT token to decode
        
    Returns:
        dict: Decoded token data
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def verify_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> dict:
    """
    Verify JWT token authentication
    
    Args:
        credentials: Bearer token credentials
        
    Returns:
        dict: Decoded token data
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    payload = decode_access_token(token)
    
    logger.debug(f"Token authenticated for user: {payload.get('sub', 'unknown')}")
    return payload


# ============================================================================
# Optional Authentication (API Key or Token)
# ============================================================================

async def optional_auth(
    api_key: Optional[str] = Security(api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme)
) -> Optional[dict]:
    """
    Optional authentication - accepts either API key or JWT token
    
    Args:
        api_key: Optional API key from header
        credentials: Optional bearer token
        
    Returns:
        dict or None: Authentication info if provided, None otherwise
    """
    # Try API key first
    if api_key:
        try:
            return await verify_api_key(api_key)
        except HTTPException:
            pass
    
    # Try JWT token
    if credentials:
        try:
            return await verify_token(credentials)
        except HTTPException:
            pass
    
    # No authentication provided (allowed for optional auth)
    return None


# ============================================================================
# Helper Functions
# ============================================================================

def get_current_user(auth_info: dict) -> str:
    """
    Extract user identifier from authentication info
    
    Args:
        auth_info: Authentication information from verify_api_key or verify_token
        
    Returns:
        str: User identifier
    """
    if "api_key" in auth_info:
        return auth_info["name"]
    elif "sub" in auth_info:
        return auth_info["sub"]
    else:
        return "anonymous"


def check_permission(auth_info: dict, required_permission: str) -> bool:
    """
    Check if authenticated user has required permission
    
    Args:
        auth_info: Authentication information
        required_permission: Permission to check
        
    Returns:
        bool: True if user has permission
    """
    # Simple permission check - can be extended with role-based access control
    permissions = auth_info.get("permissions", [])
    return required_permission in permissions or "admin" in permissions
