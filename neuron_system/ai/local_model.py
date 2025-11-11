"""
Local Model Integration for Friday.

Downloads and runs Qwen3-0.6B locally without external dependencies.
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class LocalModelManager:
    """
    Manages local language models for Friday.
    
    Downloads, caches, and runs models locally.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize local model manager.
        
        Args:
            model_dir: Directory to store models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.model_name = None
    
    def download_model(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        force_download: bool = False
    ) -> bool:
        """
        Download model from HuggingFace.
        
        Args:
            model_name: HuggingFace model identifier
            force_download: Force re-download even if exists
            
        Returns:
            True if successful
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Create model-specific directory
            model_folder = self.model_dir / model_name.replace("/", "_")
            model_folder.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Downloading model: {model_name}")
            logger.info(f"Target directory: {model_folder}")
            
            # Check if already downloaded
            if not force_download and (model_folder / "config.json").exists():
                logger.info("Model already downloaded, loading from cache...")
            else:
                logger.info("Downloading model files (this may take a few minutes)...")
            
            # Download tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(model_folder),
                trust_remote_code=True
            )
            
            # Download model
            logger.info("Loading model...")
            import torch
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(model_folder),
                trust_remote_code=True,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self.model = self.model.to(device)
            
            self.model_name = model_name
            
            logger.info(f"✓ Model loaded successfully: {model_name}")
            logger.info(f"✓ Model size: ~600MB")
            logger.info(f"✓ Location: {model_folder}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            logger.info("\nTrying to install required packages...")
            
            try:
                import subprocess
                subprocess.run(
                    ["pip", "install", "transformers", "torch", "accelerate"],
                    check=True
                )
                logger.info("Packages installed. Please run again.")
            except Exception as install_error:
                logger.error(f"Failed to install packages: {install_error}")
            
            return False
    
    def load_model(self, model_name: str = "Qwen/Qwen3-0.6B") -> bool:
        """
        Load a previously downloaded model.
        
        Args:
            model_name: Model to load
            
        Returns:
            True if successful
        """
        model_folder = self.model_dir / model_name.replace("/", "_")
        
        # Try to find the actual model files in the cache structure
        possible_paths = [
            model_folder / "models--Qwen--Qwen3-0.6B" / "snapshots" / "c1899de289a04d12100db370d81485cdf75e47ca",
            model_folder,
        ]
        
        model_path = None
        for path in possible_paths:
            if (path / "config.json").exists():
                model_path = path
                break
        
        if model_path is None:
            logger.info("Model not found locally, downloading...")
            return self.download_model(model_name)
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading model from: {model_path}")
            
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Move to device
            self.model = self.model.to(device)
            
            self.model_name = model_name
            logger.info(f"✓ Model loaded: {model_name}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_response(
        self,
        prompt: str,
        max_length: int = 150,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50
    ) -> str:
        """
        Generate text response.
        
        Args:
            prompt: Input prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Call load_model() first.")
            return ""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "name": self.model_name,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "location": str(self.model_dir / self.model_name.replace("/", "_"))
        }


class LocalGenerativeSynthesizer:
    """
    Generative synthesizer using local Qwen model.
    
    Replaces Ollama/HuggingFace pipeline with direct model access.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B"):
        """
        Initialize synthesizer.
        
        Args:
            model_name: Model to use
        """
        self.model_manager = LocalModelManager()
        self.model_name = model_name
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize the model."""
        if self._initialized:
            return True
        
        logger.info("Initializing local generative model...")
        success = self.model_manager.load_model(self.model_name)
        
        if success:
            self._initialized = True
            info = self.model_manager.get_model_info()
            logger.info(f"✓ Model ready: {info['name']}")
            logger.info(f"✓ Device: {info['device']}")
            logger.info(f"✓ Parameters: {info['parameters']:,}")
        
        return success
    
    def synthesize_response(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]],
        max_length: int = 150
    ) -> str:
        """
        Generate response based on query and knowledge.
        
        Args:
            query: User's question
            knowledge_pieces: Retrieved knowledge
            max_length: Maximum response length
            
        Returns:
            Generated response
        """
        if not self._initialized:
            if not self.initialize():
                return "Model initialization failed."
        
        # Build context
        context = self._build_context(knowledge_pieces)
        
        # Create prompt
        prompt = self._create_prompt(query, context)
        
        # Generate
        response = self.model_manager.generate_response(
            prompt=prompt,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        # Clean response
        return self._clean_response(response)
    
    def _build_context(self, knowledge_pieces: List[Dict[str, Any]]) -> str:
        """Build context from knowledge pieces."""
        if not knowledge_pieces:
            return ""
        
        top_pieces = sorted(
            knowledge_pieces,
            key=lambda x: x.get('activation', 0),
            reverse=True
        )[:3]
        
        context_parts = []
        for piece in top_pieces:
            text = piece.get('text', '').strip()
            if text:
                context_parts.append(text)
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for model."""
        if context:
            # Simple, direct prompt works best for small models
            return f"""{context}

Question: {query}
Answer:"""
        else:
            return f"Question: {query}\nAnswer:"
    
    def _clean_response(self, response: str) -> str:
        """Clean generated response."""
        response = response.strip()
        
        # Remove prompt artifacts
        artifacts = [
            "Based on this information, answer the question concisely:",
            "Based on the information,",
            "The answer to the question is:",
            "The answer is:",
            "Q:", "Question:", "A:", "Answer:",
            "__________"
        ]
        
        for artifact in artifacts:
            if artifact in response:
                parts = response.split(artifact)
                # Take the part after the artifact if it's longer
                if len(parts) > 1 and len(parts[1].strip()) > len(parts[0].strip()):
                    response = parts[1].strip()
                else:
                    response = parts[0].strip()
        
        # Remove duplicate sentences
        sentences = [s.strip() for s in response.split('. ') if s.strip()]
        unique_sentences = []
        seen = set()
        
        for sent in sentences:
            sent_lower = sent.lower()
            if sent_lower not in seen:
                unique_sentences.append(sent)
                seen.add(sent_lower)
        
        # Take first 2-3 unique sentences
        if len(unique_sentences) > 3:
            unique_sentences = unique_sentences[:3]
        
        response = '. '.join(unique_sentences)
        if response and not response.endswith('.'):
            response += '.'
        
        return response
