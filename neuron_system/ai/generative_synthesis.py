"""
Generative Response Synthesis for Friday.

Uses a local language model to generate natural responses
based on retrieved knowledge, instead of just returning stored text.
"""

import logging
from typing import List, Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class GenerativeSynthesizer:
    """
    Generates natural language responses using a local LLM.
    
    Combines retrieved knowledge with generative AI to create
    original, contextual responses instead of just returning stored text.
    """
    
    def __init__(self, model_type: str = "auto"):
        """
        Initialize the generative synthesizer.
        
        Args:
            model_type: Type of model to use:
                - "auto": Try local model first, then Ollama, then HuggingFace
                - "local": Use local Qwen3-0.6B model (best option!)
                - "ollama": Use Ollama (requires ollama running locally)
                - "huggingface": Use HuggingFace transformers
                - "openai": Use OpenAI API (requires API key)
                - "fallback": Use simple template-based generation
        """
        self.model_type = model_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the generative model based on type."""
        if self.model_type == "auto":
            # Try local model first (best option - fast, private, no external deps)
            if self._try_local():
                return
            # Try Ollama (good quality, requires external service)
            if self._try_ollama():
                return
            # Try HuggingFace (slower but works)
            if self._try_huggingface():
                return
            # Fallback to template-based
            logger.warning("No generative model available, using template fallback")
            self.model_type = "fallback"
        
        elif self.model_type == "local":
            if not self._try_local():
                raise RuntimeError("Local model not available. Run: python setup_model.py")
        
        elif self.model_type == "ollama":
            if not self._try_ollama():
                raise RuntimeError("Ollama not available. Install: https://ollama.ai")
        
        elif self.model_type == "huggingface":
            if not self._try_huggingface():
                raise RuntimeError("HuggingFace model failed to load")
        
        elif self.model_type == "openai":
            if not self._try_openai():
                raise RuntimeError("OpenAI API not configured")
    
    def _try_local(self) -> bool:
        """Try to initialize local Qwen model."""
        try:
            from neuron_system.ai.local_model import LocalGenerativeSynthesizer
            
            self.model = LocalGenerativeSynthesizer(model_name="Qwen/Qwen3-0.6B")
            
            # Try to initialize (will load model if available)
            if self.model.initialize():
                self.model_type = "local"
                logger.info("Using local Qwen3-0.6B model (best option!)")
                return True
            else:
                logger.debug("Local model not available")
                return False
        
        except Exception as e:
            logger.debug(f"Local model not available: {e}")
            return False
    
    def _try_ollama(self) -> bool:
        """Try to initialize Ollama."""
        try:
            import requests
            # Check if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                self.model_type = "ollama"
                # Use Qwen 3:1.7b - excellent quality and speed!
                self.model = "qwen2.5:1.5b"  # Try qwen2.5 first
                
                # Check if qwen is available, otherwise try alternatives
                models_data = response.json()
                available_models = [m.get('name', '') for m in models_data.get('models', [])]
                
                if any('qwen' in m.lower() for m in available_models):
                    # Use any qwen model available
                    qwen_models = [m for m in available_models if 'qwen' in m.lower()]
                    self.model = qwen_models[0]
                    logger.info(f"Using Ollama model: {self.model} (Qwen - excellent quality!)")
                elif any('llama3.2' in m for m in available_models):
                    self.model = "llama3.2:1b"
                    logger.info(f"Using Ollama model: {self.model}")
                else:
                    # Use first available model
                    self.model = available_models[0] if available_models else "llama3.2:1b"
                    logger.info(f"Using Ollama model: {self.model}")
                
                return True
        except Exception as e:
            logger.debug(f"Ollama not available: {e}")
        return False
    
    def _try_huggingface(self) -> bool:
        """Try to initialize HuggingFace model."""
        try:
            from transformers import pipeline
            # Use a small, fast model for text generation
            self.model = pipeline(
                "text2text-generation",
                model="google/flan-t5-small",  # 80MB, fast
                device=-1  # CPU
            )
            self.model_type = "huggingface"
            logger.info("Using HuggingFace model: flan-t5-small")
            return True
        except Exception as e:
            logger.debug(f"HuggingFace not available: {e}")
        return False
    
    def _try_openai(self) -> bool:
        """Try to initialize OpenAI."""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                openai.api_key = api_key
                self.model = "gpt-3.5-turbo"
                self.model_type = "openai"
                logger.info("Using OpenAI model: gpt-3.5-turbo")
                return True
        except Exception as e:
            logger.debug(f"OpenAI not available: {e}")
        return False
    
    def synthesize_response(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]],
        max_length: int = 150
    ) -> str:
        """
        Generate a natural response based on query and knowledge.
        
        Args:
            query: User's question
            knowledge_pieces: Retrieved knowledge from neurons
            max_length: Maximum response length
            
        Returns:
            Generated natural language response
        """
        # Build context from knowledge
        context = self._build_context(knowledge_pieces)
        
        # Generate response based on model type
        if self.model_type == "local":
            return self._generate_local(query, context, max_length)
        elif self.model_type == "ollama":
            return self._generate_ollama(query, context, max_length)
        elif self.model_type == "huggingface":
            return self._generate_huggingface(query, context, max_length)
        elif self.model_type == "openai":
            return self._generate_openai(query, context, max_length)
        else:
            return self._generate_fallback(query, context)
    
    def _build_context(self, knowledge_pieces: List[Dict[str, Any]]) -> str:
        """Build context string from knowledge pieces."""
        if not knowledge_pieces:
            return ""
        
        # Take top 3 most relevant pieces
        top_pieces = sorted(
            knowledge_pieces,
            key=lambda x: x.get('activation', 0),
            reverse=True
        )[:3]
        
        context_parts = []
        for i, piece in enumerate(top_pieces, 1):
            text = piece.get('text', '').strip()
            if text:
                context_parts.append(f"{i}. {text}")
        
        return "\n".join(context_parts)
    
    def _generate_local(self, query: str, context: str, max_length: int) -> str:
        """Generate response using local Qwen model."""
        try:
            # Build knowledge pieces format
            knowledge_pieces = []
            if context:
                for line in context.split('\n'):
                    if line.strip():
                        knowledge_pieces.append({
                            'text': line.strip(),
                            'activation': 1.0
                        })
            
            # Generate using local model
            response = self.model.synthesize_response(
                query=query,
                knowledge_pieces=knowledge_pieces,
                max_length=max_length
            )
            
            if response and len(response) > 10:
                return response
            else:
                logger.warning("Local model returned empty response")
                return self._generate_fallback(query, context)
        
        except Exception as e:
            logger.error(f"Local model generation failed: {e}")
            return self._generate_fallback(query, context)
    
    def _generate_ollama(self, query: str, context: str, max_length: int) -> str:
        """Generate response using Ollama."""
        try:
            import requests
            
            prompt = self._create_prompt(query, context)
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 512,  # Much more tokens for Qwen's thinking + response
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=60  # Longer timeout for thinking models
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Qwen 3 has a thinking mode - it thinks first, then answers
                answer = result.get("response", "").strip()
                thinking = result.get("thinking", "").strip()
                
                # For Qwen reasoning models:
                # - "thinking" contains the reasoning process
                # - "response" contains the final answer
                # We want the final answer, not the thinking!
                
                if answer:
                    # Perfect! We have the final answer
                    logger.debug("Using final response from Qwen")
                    cleaned = self._clean_response(answer)
                elif thinking:
                    # Qwen only gave us thinking, no final answer
                    # This happens when token limit is reached during thinking
                    logger.debug("Qwen only provided thinking, extracting answer")
                    
                    # Try to extract the actual answer from thinking
                    # Look for patterns like "So the answer is..." or "In summary..."
                    answer_markers = [
                        "So the answer is:",
                        "In summary:",
                        "To answer:",
                        "The answer is:",
                        "Simply put:",
                        "In short:",
                        "Basically:"
                    ]
                    
                    extracted = None
                    for marker in answer_markers:
                        if marker in thinking:
                            parts = thinking.split(marker, 1)
                            if len(parts) > 1:
                                extracted = parts[1].strip()
                                break
                    
                    if extracted:
                        cleaned = self._clean_response(extracted)
                    else:
                        # No clear answer marker, use last sentence of thinking
                        sentences = thinking.split('. ')
                        if sentences:
                            cleaned = self._clean_response(sentences[-1])
                        else:
                            logger.warning("Could not extract answer from thinking")
                            return self._generate_fallback(query, context)
                else:
                    logger.warning("Empty response from Ollama")
                    return self._generate_fallback(query, context)
                
                # If answer is too similar to context, it means LLM just copied
                # In that case, try to reformulate
                if context and self._is_too_similar(cleaned, context):
                    logger.debug("Response too similar to context, trying reformulation")
                    # Try a more forceful prompt
                    reformulate_prompt = f"""Rewrite this answer in completely different words, but keep the same meaning:

{cleaned}

Rewritten answer (use different words and sentence structure):"""
                    
                    response2 = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": self.model,
                            "prompt": reformulate_prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.9,
                                "num_predict": max_length
                            }
                        },
                        timeout=30
                    )
                    
                    if response2.status_code == 200:
                        result2 = response2.json()
                        reformulated = result2.get("response", "").strip()
                        if reformulated and len(reformulated) > 10:
                            return self._clean_response(reformulated)
                
                return cleaned
        
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
        
        return self._generate_fallback(query, context)
    
    def _is_too_similar(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """Check if two texts are too similar (simple word overlap check)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        similarity = overlap / max(len(words1), len(words2))
        
        return similarity > threshold
    
    def _generate_huggingface(self, query: str, context: str, max_length: int) -> str:
        """Generate response using HuggingFace."""
        try:
            prompt = self._create_prompt(query, context)
            
            result = self.model(
                prompt,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7
            )
            
            if result and len(result) > 0:
                answer = result[0]['generated_text'].strip()
                return self._clean_response(answer)
        
        except Exception as e:
            logger.error(f"HuggingFace generation failed: {e}")
        
        return self._generate_fallback(query, context)
    
    def _generate_openai(self, query: str, context: str, max_length: int) -> str:
        """Generate response using OpenAI."""
        try:
            import openai
            
            prompt = self._create_prompt(query, context)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are Friday, a helpful AI assistant. Answer concisely based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            return self._clean_response(answer)
        
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
        
        return self._generate_fallback(query, context)
    
    def _generate_fallback(self, query: str, context: str) -> str:
        """Simple template-based generation as fallback."""
        if not context:
            return "I don't have enough information to answer that question."
        
        # Extract first relevant piece
        lines = context.split('\n')
        if lines:
            first_piece = lines[0].replace('1. ', '').strip()
            
            # Simple reformulation based on query type
            query_lower = query.lower()
            
            if any(w in query_lower for w in ['what is', 'what are', 'define']):
                return first_piece
            elif 'how' in query_lower:
                return f"Based on my knowledge: {first_piece}"
            elif 'why' in query_lower:
                return f"The reason is: {first_piece}"
            else:
                return first_piece
        
        return "I'm not sure how to answer that."
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create a prompt for the LLM."""
        if context:
            # Simpler prompt that works better with reasoning models like Qwen
            return f"""Based on this information, answer the question concisely:

{context}

Q: {query}
A:"""
        else:
            return f"""Q: {query}
A:"""
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove common artifacts
        response = response.strip()
        
        # Remove "Answer:" prefix if present
        if response.startswith("Answer:"):
            response = response[7:].strip()
        
        # Remove "Instructions:" and everything after
        if "Instructions:" in response:
            response = response.split("Instructions:")[0].strip()
        
        # Remove "Context:" and everything after
        if "Context:" in response:
            response = response.split("Context:")[0].strip()
        
        # Remove "Question:" and everything after
        if "Question:" in response:
            response = response.split("Question:")[0].strip()
        
        # Remove quotes if the entire response is quoted
        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1].strip()
        
        # Remove duplicate sentences (sometimes LLMs repeat)
        sentences = response.split('. ')
        unique_sentences = []
        seen = set()
        for sent in sentences:
            sent_clean = sent.strip().lower()
            if sent_clean and sent_clean not in seen:
                unique_sentences.append(sent.strip())
                seen.add(sent_clean)
        
        response = '. '.join(unique_sentences)
        if response and not response.endswith('.'):
            response += '.'
        
        # Limit to 2-3 sentences for conciseness
        sentences = response.split('. ')
        if len(sentences) > 3:
            response = '. '.join(sentences[:3])
            if not response.endswith('.'):
                response += '.'
        
        return response
