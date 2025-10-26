"""
Compression Engine for converting raw data into 384-dimensional embeddings.

This module provides the CompressionEngine class that uses sentence-transformers
to compress text data into fixed-size vector representations.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Configure logger
logger = logging.getLogger(__name__)


class CompressionEngine:
    """
    Engine for compressing text data into 384-dimensional embeddings.
    
    Uses the sentence-transformers library with the all-MiniLM-L6-v2 model
    for lightweight, fast embedding generation.
    
    Attributes:
        model: The sentence transformer model instance
        model_name: Name of the embedding model
        vector_dim: Dimensionality of output vectors (384)
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_input_length: int = 10000, use_gpu: bool = True):
        """
        Initialize the CompressionEngine with a sentence-transformers model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is "all-MiniLM-L6-v2" (80MB, 384 dimensions)
            max_input_length: Maximum allowed input text length (default: 10000 chars)
            use_gpu: Whether to use GPU acceleration if available (default: True)
        
        Raises:
            ImportError: If sentence-transformers is not installed
            Exception: If model loading fails
        """
        self.model_name = model_name
        self.vector_dim = 384
        self.max_input_length = max_input_length
        self.use_gpu = use_gpu
        self._model = None
        self._model_loaded = False
        self._gpu_accelerator = None
        
        # Performance tracking
        self._compression_count = 0
        self._total_compression_time = 0.0
        self._failed_compressions = 0
        
    def _ensure_model_loaded(self):
        """Lazy load the sentence-transformers model with GPU support."""
        if self._model_loaded and self._model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            from neuron_system.engines.gpu_accelerator import GPUAccelerator
            
            # Initialize GPU accelerator
            if self.use_gpu and self._gpu_accelerator is None:
                self._gpu_accelerator = GPUAccelerator(force_cpu=not self.use_gpu)
                device_info = self._gpu_accelerator.get_device_info()
                logger.info(f"GPU Accelerator: {device_info['device_name']}")
            
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            
            # Load model
            self._model = SentenceTransformer(self.model_name)
            
            # Move model to GPU if available
            if self.use_gpu and self._gpu_accelerator and self._gpu_accelerator.is_gpu_available():
                device = self._gpu_accelerator.device
                self._model = self._model.to(device)
                logger.info(f"Model moved to {device}")
            else:
                logger.info("Model running on CPU")
            
            # Verify model is working with a test
            test_vector = self._model.encode("test", convert_to_numpy=True)
            if test_vector is None or len(test_vector) == 0:
                raise Exception("Model loaded but failed to generate embeddings")
            
            self._model_loaded = True
            logger.info(f"Model {self.model_name} loaded successfully (dim: {len(test_vector)})")
            
        except ImportError:
            error_msg = (
                "sentence-transformers is required for compression. "
                "Install it with: pip install sentence-transformers"
            )
            logger.error(error_msg)
            self._model_loaded = False
            self._model = None
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            logger.error(error_msg)
            self._model_loaded = False
            self._model = None
            raise Exception(error_msg)
    
    def validate_input(self, data: str) -> Tuple[bool, Optional[str]]:
        """
        Validate input data format before compression.
        
        Args:
            data: Text string to validate
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if data is valid, False otherwise
            - error_message: None if valid, error description if invalid
        """
        # Check type
        if not isinstance(data, str):
            return False, f"Input must be a string, got {type(data).__name__}"
        
        # Check for empty or whitespace-only
        if not data or not data.strip():
            return False, "Input cannot be empty or whitespace only"
        
        # Check length
        if len(data) > self.max_input_length:
            return False, f"Input length ({len(data)}) exceeds maximum ({self.max_input_length})"
        
        # Check for valid characters (basic check)
        try:
            data.encode('utf-8')
        except UnicodeEncodeError as e:
            return False, f"Input contains invalid characters: {str(e)}"
        
        return True, None
    
    def compress(
        self, 
        data: str,
        normalize: bool = True,
        raise_on_error: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compress a single text input into a 384-dimensional vector.
        
        Args:
            data: Text string to compress
            normalize: Whether to normalize the output vector (default: True)
            raise_on_error: If True, raise exceptions; if False, return fallback (default: False)
        
        Returns:
            Tuple of (vector, metadata) where:
                - vector: numpy array of shape (384,)
                - metadata: dict with compression stats (ratio, time, etc.)
        
        Raises:
            ValueError: If input data is invalid and raise_on_error is True
            Exception: If compression fails and raise_on_error is True
        """
        # Validate input format
        is_valid, error_msg = self.validate_input(data)
        if not is_valid:
            logger.warning(f"Input validation failed: {error_msg}")
            if raise_on_error:
                raise ValueError(error_msg)
            # Return fallback
            return self._create_fallback_vector(data, normalize, error_msg)
        
        # Ensure model is loaded
        try:
            self._ensure_model_loaded()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            if raise_on_error:
                raise
            return self._create_fallback_vector(data, normalize, str(e))
        
        # Track timing
        start_time = time.time()
        
        try:
            # Generate embedding
            vector = self._model.encode(data, convert_to_numpy=True)
            
            # Validate output vector
            if not self.validate_vector(vector):
                error_msg = "Generated vector failed validation"
                logger.error(error_msg)
                if raise_on_error:
                    raise ValueError(error_msg)
                return self._create_fallback_vector(data, normalize, error_msg)
            
            # Normalize if requested
            if normalize:
                vector = self._normalize_vector(vector)
            
            # Calculate compression ratio (characters to vector elements)
            compression_ratio = len(data) / self.vector_dim
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Update performance tracking
            self._compression_count += 1
            self._total_compression_time += elapsed_time
            
            # Build metadata
            metadata = {
                "compression_ratio": compression_ratio,
                "elapsed_time_ms": elapsed_time * 1000,
                "input_length": len(data),
                "vector_dim": self.vector_dim,
                "normalized": normalize,
                "model": self.model_name,
                "success": True
            }
            
            logger.debug(f"Compressed text ({len(data)} chars) in {elapsed_time*1000:.2f}ms")
            
            return vector, metadata
            
        except Exception as e:
            # Fallback: return zero vector on failure
            elapsed_time = time.time() - start_time
            self._failed_compressions += 1
            
            error_msg = f"Compression failed: {str(e)}"
            logger.error(error_msg)
            
            if raise_on_error:
                raise
            
            return self._create_fallback_vector(data, normalize, error_msg, elapsed_time)
    
    def _create_fallback_vector(
        self,
        data: str,
        normalize: bool,
        error: str,
        elapsed_time: float = 0.0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Create a fallback zero vector when compression fails.
        
        Args:
            data: Original input data
            normalize: Whether normalization was requested
            error: Error message
            elapsed_time: Time elapsed before failure
        
        Returns:
            Tuple of (zero_vector, metadata)
        """
        fallback_vector = np.zeros(self.vector_dim, dtype=np.float32)
        
        metadata = {
            "compression_ratio": 0.0,
            "elapsed_time_ms": elapsed_time * 1000,
            "input_length": len(data) if isinstance(data, str) else 0,
            "vector_dim": self.vector_dim,
            "normalized": normalize,
            "model": self.model_name,
            "error": error,
            "fallback": True,
            "success": False
        }
        
        return fallback_vector, metadata
    
    def batch_compress(
        self,
        data_list: List[str],
        normalize: bool = True,
        batch_size: int = None,
        raise_on_error: bool = False
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Compress multiple text inputs into 384-dimensional vectors.
        
        This method is more efficient than calling compress() multiple times
        as it processes inputs in batches. Automatically optimizes batch size for GPU.
        
        Args:
            data_list: List of text strings to compress
            normalize: Whether to normalize output vectors (default: True)
            batch_size: Number of texts to process at once (None = auto-optimize for GPU)
            raise_on_error: If True, raise exceptions; if False, return fallback (default: False)
        
        Returns:
            Tuple of (vectors, metadata_list) where:
                - vectors: numpy array of shape (n, 384)
                - metadata_list: list of dicts with compression stats for each input
        
        Raises:
            ValueError: If input data is invalid and raise_on_error is True
            Exception: If compression fails and raise_on_error is True
        """
        # Auto-optimize batch size for GPU if not specified
        if batch_size is None:
            if self._gpu_accelerator and self._gpu_accelerator.is_gpu_available():
                batch_size = self._gpu_accelerator.optimize_batch_size(default_batch_size=32)
                logger.info(f"Auto-optimized batch size for GPU: {batch_size}")
            else:
                batch_size = 32
        # Validate input list
        if not isinstance(data_list, list):
            error_msg = f"Input must be a list, got {type(data_list).__name__}"
            logger.error(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return self._create_batch_fallback(data_list, normalize, error_msg)
        
        if not data_list:
            error_msg = "Input list cannot be empty"
            logger.error(error_msg)
            if raise_on_error:
                raise ValueError(error_msg)
            return np.zeros((0, self.vector_dim), dtype=np.float32), []
        
        # Validate all items
        validation_errors = []
        for i, data in enumerate(data_list):
            is_valid, error_msg = self.validate_input(data)
            if not is_valid:
                validation_errors.append((i, error_msg))
        
        if validation_errors:
            error_summary = f"Validation failed for {len(validation_errors)} items"
            logger.warning(f"{error_summary}: {validation_errors[:3]}")  # Log first 3
            if raise_on_error:
                raise ValueError(f"{error_summary}: {validation_errors}")
            return self._create_batch_fallback(data_list, normalize, error_summary)
        
        # Ensure model is loaded
        try:
            self._ensure_model_loaded()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            if raise_on_error:
                raise
            return self._create_batch_fallback(data_list, normalize, str(e))
        
        # Track timing
        start_time = time.time()
        
        try:
            # Generate embeddings in batch
            vectors = self._model.encode(
                data_list,
                convert_to_numpy=True,
                batch_size=batch_size,
                show_progress_bar=False
            )
            
            # Normalize if requested
            if normalize:
                vectors = np.array([self._normalize_vector(v) for v in vectors])
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Update performance tracking
            self._compression_count += len(data_list)
            self._total_compression_time += elapsed_time
            
            # Build metadata for each input
            metadata_list = []
            for i, data in enumerate(data_list):
                compression_ratio = len(data) / self.vector_dim
                metadata = {
                    "compression_ratio": compression_ratio,
                    "elapsed_time_ms": (elapsed_time * 1000) / len(data_list),  # Average per item
                    "input_length": len(data),
                    "vector_dim": self.vector_dim,
                    "normalized": normalize,
                    "model": self.model_name,
                    "batch_index": i,
                    "success": True
                }
                metadata_list.append(metadata)
            
            logger.info(f"Batch compressed {len(data_list)} texts in {elapsed_time*1000:.2f}ms")
            
            return vectors, metadata_list
            
        except Exception as e:
            # Fallback: return zero vectors on failure
            elapsed_time = time.time() - start_time
            self._failed_compressions += len(data_list)
            
            error_msg = f"Batch compression failed: {str(e)}"
            logger.error(error_msg)
            
            if raise_on_error:
                raise
            
            return self._create_batch_fallback(data_list, normalize, error_msg, elapsed_time)
    
    def _create_batch_fallback(
        self,
        data_list: List[str],
        normalize: bool,
        error: str,
        elapsed_time: float = 0.0
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Create fallback zero vectors when batch compression fails.
        
        Args:
            data_list: Original input data list
            normalize: Whether normalization was requested
            error: Error message
            elapsed_time: Time elapsed before failure
        
        Returns:
            Tuple of (zero_vectors, metadata_list)
        """
        fallback_vectors = np.zeros((len(data_list), self.vector_dim), dtype=np.float32)
        
        metadata_list = []
        for i, data in enumerate(data_list):
            input_len = len(data) if isinstance(data, str) else 0
            metadata = {
                "compression_ratio": 0.0,
                "elapsed_time_ms": (elapsed_time * 1000) / max(len(data_list), 1),
                "input_length": input_len,
                "vector_dim": self.vector_dim,
                "normalized": normalize,
                "model": self.model_name,
                "batch_index": i,
                "error": error,
                "fallback": True,
                "success": False
            }
            metadata_list.append(metadata)
        
        return fallback_vectors, metadata_list
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        This ensures the vector stays within valid embedding space boundaries
        and improves cosine similarity calculations.
        
        Args:
            vector: Input vector to normalize
        
        Returns:
            Normalized vector with L2 norm = 1
        """
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm
    
    def validate_vector(self, vector: np.ndarray) -> bool:
        """
        Validate that a vector is in the correct format and range.
        
        Args:
            vector: Vector to validate
        
        Returns:
            True if vector is valid, False otherwise
        """
        if not isinstance(vector, np.ndarray):
            return False
        
        if vector.shape != (self.vector_dim,):
            return False
        
        if not np.isfinite(vector).all():
            return False
        
        return True
    
    def get_compression_stats(self, text: str, vector: np.ndarray) -> Dict[str, Any]:
        """
        Calculate compression statistics for a text-vector pair.
        
        Args:
            text: Original text
            vector: Compressed vector
        
        Returns:
            Dictionary with compression statistics
        """
        return {
            "input_bytes": len(text.encode('utf-8')),
            "input_chars": len(text),
            "vector_bytes": vector.nbytes,
            "vector_dim": len(vector),
            "compression_ratio": len(text) / len(vector),
            "bytes_ratio": len(text.encode('utf-8')) / vector.nbytes
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the compression engine.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_time = (
            self._total_compression_time / self._compression_count
            if self._compression_count > 0
            else 0.0
        )
        
        success_rate = (
            (self._compression_count - self._failed_compressions) / self._compression_count
            if self._compression_count > 0
            else 0.0
        )
        
        stats = {
            "total_compressions": self._compression_count,
            "failed_compressions": self._failed_compressions,
            "success_rate": success_rate,
            "total_time_seconds": self._total_compression_time,
            "average_time_ms": avg_time * 1000,
            "model": self.model_name,
            "vector_dim": self.vector_dim
        }
        
        # Add GPU stats if available
        if self._gpu_accelerator:
            stats["gpu_info"] = self._gpu_accelerator.get_device_info()
            stats["gpu_performance"] = self._gpu_accelerator.get_performance_stats()
        
        return stats
    
    def reset_stats(self):
        """Reset performance tracking statistics."""
        self._compression_count = 0
        self._total_compression_time = 0.0
        self._failed_compressions = 0
        logger.info("Performance statistics reset")
