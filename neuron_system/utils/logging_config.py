"""
Logging configuration for the neuron system.

Provides centralized logging setup with audit logging support.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_audit: bool = True,
    audit_log_file: Optional[str] = None
):
    """
    Set up logging configuration for the neuron system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to main log file (optional)
        enable_audit: Whether to enable audit logging
        audit_log_file: Path to audit log file (optional)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(pathname)s:%(lineno)d',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Audit logger (if enabled)
    if enable_audit:
        audit_logger = logging.getLogger('neuron_system.audit')
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False  # Don't propagate to root logger
        
        audit_file = audit_log_file or 'neuron_system_audit.log'
        audit_path = Path(audit_file)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        audit_handler = logging.handlers.RotatingFileHandler(
            audit_file,
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10
        )
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        audit_handler.setFormatter(audit_formatter)
        audit_logger.addHandler(audit_handler)


def get_audit_logger():
    """
    Get the audit logger instance.
    
    Returns:
        Audit logger
    """
    return logging.getLogger('neuron_system.audit')


def log_audit_event(
    event_type: str,
    details: dict,
    user_id: Optional[str] = None
):
    """
    Log an audit event.
    
    Args:
        event_type: Type of event (e.g., 'training_operation', 'neuron_created')
        details: Event details
        user_id: User ID (if applicable)
    """
    audit_logger = get_audit_logger()
    
    audit_entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'event_type': event_type,
        'user_id': user_id,
        'details': details
    }
    
    audit_logger.info(str(audit_entry))
