"""
Application configuration settings.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Settings:
    """
    Application configuration settings.
    
    Centralizes all configurable parameters for the neuron system.
    """
    
    # Spatial configuration
    spatial_bounds_min: Tuple[float, float, float] = (-100.0, -100.0, -100.0)
    spatial_bounds_max: Tuple[float, float, float] = (100.0, 100.0, 100.0)
    
    # Neuron configuration
    vector_dimensions: int = 384
    default_activation_threshold: float = 0.5
    
    # Synapse configuration
    synapse_weight_min: float = -1.0
    synapse_weight_max: float = 1.0
    synapse_weak_threshold: float = 0.1
    synapse_prune_threshold: float = 0.01
    synapse_strengthen_delta: float = 0.01
    synapse_weaken_delta: float = 0.001
    
    # Training configuration
    default_learning_rate: float = 0.1
    
    # Query configuration
    default_top_k: int = 10
    default_propagation_depth: int = 3
    query_timeout_seconds: float = 5.0
    
    # Performance configuration
    uuid_pool_size: int = 10000
    uuid_pool_refill_threshold: int = 1000
    object_pool_max_size: int = 1000
    batch_size: int = 100
    
    # Storage configuration
    database_path: str = "neuron_system.db"
    enable_auto_save: bool = True
    auto_save_interval_seconds: int = 300
    
    # Compression configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    compression_timeout_ms: int = 100
    
    # Tool configuration
    tool_execution_timeout_seconds: float = 30.0
    enable_tool_sandboxing: bool = True
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key_required: bool = True
    rate_limit_requests_per_minute: int = 100
    
    # Logging configuration
    log_level: str = "INFO"
    enable_audit_logging: bool = True


# Global settings instance
_settings: Settings = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def update_settings(**kwargs):
    """
    Update global settings.
    
    Args:
        **kwargs: Settings to update
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    
    for key, value in kwargs.items():
        if hasattr(_settings, key):
            setattr(_settings, key, value)
        else:
            raise ValueError(f"Unknown setting: {key}")
