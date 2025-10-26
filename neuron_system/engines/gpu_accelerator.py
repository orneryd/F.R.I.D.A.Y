"""
GPU Accelerator - Automatic GPU detection and acceleration for training.

Features:
- Automatic GPU detection (CUDA, MPS for Mac, CPU fallback)
- Batch processing optimization
- Memory management
- Performance monitoring
"""

import logging
import torch
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class GPUAccelerator:
    """
    GPU acceleration manager for neural network operations.
    """
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize GPU accelerator with automatic device detection.
        
        Args:
            force_cpu: Force CPU usage even if GPU is available
        """
        self.force_cpu = force_cpu
        self.device = self._detect_device()
        self.device_name = self._get_device_name()
        
        # Performance tracking
        self.total_operations = 0
        self.gpu_operations = 0
        self.cpu_operations = 0
        
        logger.info(f"GPU Accelerator initialized: {self.device_name}")
    
    def _detect_device(self) -> torch.device:
        """
        Detect best available device (CUDA > MPS > CPU).
        
        Returns:
            torch.device object
        """
        if self.force_cpu:
            logger.info("Forced CPU mode")
            return torch.device("cpu")
        
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return device
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Apple Silicon MPS detected")
            return device
        
        # Fallback to CPU
        logger.warning("No GPU detected, using CPU (this will be slower)")
        return torch.device("cpu")
    
    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.type == "cuda":
            return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        elif self.device.type == "mps":
            return "Apple Silicon MPS"
        else:
            return "CPU"
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available and being used."""
        return self.device.type in ["cuda", "mps"]
    
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get detailed device information.
        
        Returns:
            Dictionary with device details
        """
        info = {
            "device_type": self.device.type,
            "device_name": self.device_name,
            "is_gpu": self.is_gpu_available(),
            "pytorch_version": torch.__version__,
        }
        
        if self.device.type == "cuda":
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                "gpu_memory_cached_gb": torch.cuda.memory_reserved(0) / 1e9,
            })
        
        return info
    
    def optimize_batch_size(self, default_batch_size: int = 32) -> int:
        """
        Optimize batch size based on available device.
        
        Args:
            default_batch_size: Default batch size for CPU
            
        Returns:
            Optimized batch size
        """
        if self.device.type == "cuda":
            # GPU can handle larger batches
            # Scale based on GPU memory
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            if gpu_memory_gb >= 16:
                return default_batch_size * 8  # 256 for 16GB+ GPU
            elif gpu_memory_gb >= 8:
                return default_batch_size * 4  # 128 for 8GB GPU
            elif gpu_memory_gb >= 4:
                return default_batch_size * 2  # 64 for 4GB GPU
            else:
                return default_batch_size  # 32 for <4GB GPU
        
        elif self.device.type == "mps":
            # Apple Silicon - moderate batch size
            return default_batch_size * 2  # 64
        
        else:
            # CPU - keep default
            return default_batch_size
    
    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        elif self.device.type == "mps":
            # MPS doesn't have explicit cache clearing
            pass
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory stats
        """
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated(0) / 1e9,
                "cached_gb": torch.cuda.memory_reserved(0) / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated(0) / 1e9,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            }
        else:
            return {
                "device": self.device.type,
                "note": "Memory stats only available for CUDA"
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        gpu_ratio = (
            self.gpu_operations / self.total_operations
            if self.total_operations > 0
            else 0.0
        )
        
        return {
            "total_operations": self.total_operations,
            "gpu_operations": self.gpu_operations,
            "cpu_operations": self.cpu_operations,
            "gpu_usage_ratio": gpu_ratio,
            "device": self.device_name,
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.total_operations = 0
        self.gpu_operations = 0
        self.cpu_operations = 0
