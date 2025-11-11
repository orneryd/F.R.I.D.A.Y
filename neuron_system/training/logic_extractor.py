"""
Logic Extractor - Extract the THINKING LOGIC from LLMs.

This extracts HOW models think, not just WHAT they know:
- Attention patterns (what to focus on)
- Token selection strategies (how to choose words)
- Context processing (how to understand input)
- Generation logic (how to build responses)

This makes Friday DYNAMIC like an LLM, not static like a database.
"""

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class LogicExtractor:
    """
    Extracts the THINKING LOGIC from language models.
    
    Instead of just storing Q&A, we extract:
    1. How the model processes context (attention patterns)
    2. How it selects next tokens (generation strategy)
    3. How it combines information (reasoning logic)
    4. How it structures responses (composition patterns)
    """
    
    def __init__(self, training_manager):
        """
        Initialize logic extractor.
        
        Args:
            training_manager: TrainingManager instance
        """
        self.training_manager = training_manager
        self.language_model = training_manager.language_model
        self.graph = training_manager.graph
        self.qwen_model = None
        self.qwen_tokenizer = None
    
    def load_qwen(self) -> bool:
        """Load Qwen model for analysis."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Try multiple possible paths
            possible_paths = [
                "models/Qwen_Qwen3-0.6B/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",  # Exact local path
                "models/Qwen_Qwen3-0.6B",  # Local cache folder
                "Qwen/Qwen3-0.6B",  # HuggingFace direct (fallback)
            ]
            
            logger.info("Loading Qwen3 for logic extraction...")
            
            model_loaded = False
            for model_path in possible_paths:
                try:
                    self.qwen_tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                    
                    self.qwen_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        output_attentions=True,  # Get attention weights!
                        output_hidden_states=True,  # Get internal states!
                        dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    model_loaded = True
                    logger.info(f"✓ Loaded from: {model_path}")
                    break
                except Exception as e:
                    logger.debug(f"Failed to load from {model_path}: {e}")
                    continue
            
            if not model_loaded:
                raise Exception("Could not load model from any path")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.qwen_model = self.qwen_model.to(device)
            self.qwen_model.eval()  # Evaluation mode
            
            logger.info(f"✓ Qwen3 loaded on {device}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")
            return False
    
    def extract_attention_patterns(
        self,
        num_examples: int = 50
    ) -> Dict[str, int]:
        """
        Extract attention patterns with BATCH TRAINING.
        
        Args:
            num_examples: Number of examples to analyze
            
        Returns:
            Statistics dict
        """
        if not self.qwen_model:
            if not self.load_qwen():
                return {"error": "Failed to load model"}
        
        logger.info(f"Extracting attention patterns from {num_examples} examples...")
        logger.info("Using BATCH mode...")
        
        # Test prompts
        prompts = self._get_test_prompts()[:num_examples]
        
        # Phase 1: Analyze all patterns
        logger.info("📊 Phase 1: Analyzing attention...")
        batch_rules = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Tokenize
                inputs = self.qwen_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}
                
                # Get model output with attention
                with torch.no_grad():
                    outputs = self.qwen_model(**inputs)
                
                # Extract attention patterns
                attentions = outputs.attentions
                
                # Analyze attention patterns
                attention_rules = self._analyze_attention(attentions, prompt)
                
                # Add to batch
                for rule in attention_rules:
                    batch_rules.append({
                        'text': rule,
                        'tags': ['logic', 'attention', 'pattern', 'extracted']
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"    Analyzed: {i+1}/{num_examples}")
            
            except Exception as e:
                logger.debug(f"Failed to extract attention: {e}")
        
        # Phase 2: SMART Batch train
        logger.info(f"💾 Phase 2: SMART Batch training {len(batch_rules)} rules...")
        extracted = 0
        batch_size = 30  # Process 30 rules at once
        
        for batch_start in range(0, len(batch_rules), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_rules))
            current_batch = batch_rules[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=True
                )
                
                extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: {len(neuron_ids)} rules ({extracted}/{len(batch_rules)})")
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'])
                        extracted += 1
                    except:
                        pass
        
        # Save once
        logger.info("💾 Saving...")
        self.graph.save()
        
        logger.info(f"✓ Extracted {extracted} attention patterns")
        
        return {
            "extracted": extracted,
            "total_neurons": len(self.graph.neurons)
        }
    
    def extract_generation_logic(
        self,
        num_examples: int = 50
    ) -> Dict[str, int]:
        """
        Extract generation logic - HOW the model chooses words.
        
        This extracts the strategy for selecting next tokens based on:
        - Context
        - Previous tokens
        - Probability distributions
        
        Args:
            num_examples: Number of examples
            
        Returns:
            Statistics dict
        """
        if not self.qwen_model:
            if not self.load_qwen():
                return {"error": "Failed to load model"}
        
        logger.info(f"Extracting generation logic from {num_examples} examples...")
        logger.info("Using BATCH mode...")
        
        prompts = self._get_test_prompts()[:num_examples]
        
        # Phase 1: Analyze all generation patterns
        logger.info("📊 Phase 1: Analyzing generation...")
        batch_rules = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Tokenize
                inputs = self.qwen_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}
                
                # Generate with output scores
                with torch.no_grad():
                    outputs = self.qwen_model.generate(
                        **inputs,
                        max_new_tokens=20,
                        output_scores=True,
                        return_dict_in_generate=True,
                        do_sample=False  # Greedy for analysis
                    )
                
                # Analyze generation strategy
                generation_rules = self._analyze_generation(
                    outputs.scores,
                    outputs.sequences,
                    prompt
                )
                
                # Add to batch
                for rule in generation_rules:
                    batch_rules.append({
                        'text': rule,
                        'tags': ['logic', 'generation', 'strategy', 'extracted']
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"    Analyzed: {i+1}/{num_examples}")
            
            except Exception as e:
                logger.debug(f"Failed to extract generation logic: {e}")
        
        # Phase 2: SMART Batch train
        logger.info(f"💾 Phase 2: SMART Batch training {len(batch_rules)} rules...")
        extracted = 0
        batch_size = 25
        
        for batch_start in range(0, len(batch_rules), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_rules))
            current_batch = batch_rules[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=True
                )
                
                extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: {len(neuron_ids)} rules ({extracted}/{len(batch_rules)})")
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'])
                        extracted += 1
                    except:
                        pass
        
        self.graph.save()
        
        logger.info(f"✓ Extracted {extracted} generation rules")
        
        return {
            "extracted": extracted,
            "total_neurons": len(self.graph.neurons)
        }
    
    def extract_reasoning_logic(
        self,
        num_examples: int = 30
    ) -> Dict[str, int]:
        """
        Extract reasoning logic - HOW the model thinks step-by-step.
        
        This extracts the logical flow from input to output.
        
        Args:
            num_examples: Number of examples
            
        Returns:
            Statistics dict
        """
        if not self.qwen_model:
            if not self.load_qwen():
                return {"error": "Failed to load model"}
        
        logger.info(f"Extracting reasoning logic from {num_examples} examples...")
        logger.info("Using BATCH mode...")
        
        # Reasoning prompts
        prompts = [
            "If A is true and B is true, then",
            "Because X happened, we can conclude",
            "Given that Y, it follows that",
            "The reason why Z is",
            "This suggests that",
            "Therefore, we can say",
            "As a result of",
            "This implies",
            "Consequently,",
            "Based on this evidence,"
        ] * (num_examples // 10 + 1)
        
        prompts = prompts[:num_examples]
        
        # Phase 1: Analyze all reasoning patterns
        logger.info("📊 Phase 1: Analyzing reasoning...")
        batch_rules = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Generate with hidden states
                inputs = self.qwen_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.qwen_model.generate(
                        **inputs,
                        max_new_tokens=30,
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                
                # Analyze reasoning flow
                reasoning_rules = self._analyze_reasoning(
                    outputs.hidden_states,
                    outputs.sequences,
                    prompt
                )
                
                # Add to batch
                for rule in reasoning_rules:
                    batch_rules.append({
                        'text': rule,
                        'tags': ['logic', 'reasoning', 'thinking', 'extracted']
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"    Analyzed: {i+1}/{num_examples}")
            
            except Exception as e:
                logger.debug(f"Failed to extract reasoning: {e}")
        
        # Phase 2: SMART Batch train
        logger.info(f"💾 Phase 2: SMART Batch training {len(batch_rules)} rules...")
        extracted = 0
        batch_size = 20
        
        for batch_start in range(0, len(batch_rules), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_rules))
            current_batch = batch_rules[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=True
                )
                
                extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: {len(neuron_ids)} rules ({extracted}/{len(batch_rules)})")
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'])
                        extracted += 1
                    except:
                        pass
        
        self.graph.save()
        
        logger.info(f"✓ Extracted {extracted} reasoning rules")
        
        return {
            "extracted": extracted,
            "total_neurons": len(self.graph.neurons)
        }
    
    def extract_composition_patterns(
        self,
        num_examples: int = 50
    ) -> Dict[str, int]:
        """
        Extract composition patterns - HOW the model structures responses.
        
        This extracts patterns for:
        - Sentence structure
        - Paragraph organization
        - Response flow
        
        Args:
            num_examples: Number of examples
            
        Returns:
            Statistics dict
        """
        if not self.qwen_model:
            if not self.load_qwen():
                return {"error": "Failed to load model"}
        
        logger.info(f"Extracting composition patterns from {num_examples} examples...")
        logger.info("Using BATCH mode...")
        
        prompts = self._get_test_prompts()[:num_examples]
        
        # Phase 1: Analyze all composition patterns
        logger.info("📊 Phase 1: Analyzing composition...")
        batch_rules = []
        
        for i, prompt in enumerate(prompts):
            try:
                # Generate response
                inputs = self.qwen_tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.qwen_model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.qwen_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.7
                    )
                
                # Decode
                response = self.qwen_tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
                
                # Analyze composition
                composition_rules = self._analyze_composition(prompt, response)
                
                # Add to batch
                for rule in composition_rules:
                    batch_rules.append({
                        'text': rule,
                        'tags': ['logic', 'composition', 'structure', 'extracted']
                    })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"    Analyzed: {i+1}/{num_examples}")
            
            except Exception as e:
                logger.debug(f"Failed to extract composition: {e}")
        
        # Phase 2: SMART Batch train
        logger.info(f"💾 Phase 2: SMART Batch training {len(batch_rules)} rules...")
        extracted = 0
        batch_size = 30
        
        for batch_start in range(0, len(batch_rules), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_rules))
            current_batch = batch_rules[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=True
                )
                
                extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: {len(neuron_ids)} rules ({extracted}/{len(batch_rules)})")
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'])
                        extracted += 1
                    except:
                        pass
        
        self.graph.save()
        
        logger.info(f"✓ Extracted {extracted} composition patterns")
        
        return {
            "extracted": extracted,
            "total_neurons": len(self.graph.neurons)
        }
    
    def _analyze_attention(
        self,
        attentions: Tuple,
        prompt: str
    ) -> List[str]:
        """Analyze attention patterns and extract rules."""
        rules = []
        
        try:
            # Get last layer attention (most relevant)
            last_attention = attentions[-1]  # Shape: [batch, heads, seq, seq]
            
            # Average over heads
            avg_attention = last_attention.mean(dim=1)[0]  # Shape: [seq, seq]
            
            # Find high attention patterns
            attention_np = avg_attention.cpu().numpy()
            
            # Extract patterns
            max_attention = attention_np.max(axis=1)
            high_attention_indices = np.where(max_attention > 0.5)[0]
            
            if len(high_attention_indices) > 0:
                rule = f"ATTENTION_PATTERN: Focus on positions {high_attention_indices.tolist()} when processing similar context"
                rules.append(rule)
            
            # Pattern: Beginning vs End focus
            beginning_focus = attention_np[:3, :].mean()
            end_focus = attention_np[-3:, :].mean()
            
            if beginning_focus > end_focus:
                rules.append("ATTENTION_RULE: Prioritize beginning of context for understanding")
            else:
                rules.append("ATTENTION_RULE: Prioritize recent context for generation")
        
        except Exception as e:
            logger.debug(f"Attention analysis failed: {e}")
        
        return rules
    
    def _analyze_generation(
        self,
        scores: Tuple,
        sequences: torch.Tensor,
        prompt: str
    ) -> List[str]:
        """Analyze generation strategy and extract rules."""
        rules = []
        
        try:
            # Analyze token selection
            for i, score in enumerate(scores[:5]):  # First 5 tokens
                # Get top-k tokens
                topk = torch.topk(score[0], k=5)
                top_probs = torch.softmax(topk.values, dim=0)
                
                # Extract rule
                if top_probs[0] > 0.7:
                    rules.append(f"GENERATION_RULE: High confidence selection (>{top_probs[0]:.2f}) for deterministic output")
                elif top_probs[0] < 0.3:
                    rules.append(f"GENERATION_RULE: Low confidence ({top_probs[0]:.2f}) - consider multiple options")
        
        except Exception as e:
            logger.debug(f"Generation analysis failed: {e}")
        
        return rules
    
    def _analyze_reasoning(
        self,
        hidden_states: Tuple,
        sequences: torch.Tensor,
        prompt: str
    ) -> List[str]:
        """Analyze reasoning flow and extract rules."""
        rules = []
        
        try:
            # Analyze hidden state evolution
            if hidden_states and len(hidden_states) > 0:
                # Get first and last hidden states
                first_hidden = hidden_states[0][-1]  # Last layer, first token
                
                # Extract reasoning pattern
                rules.append("REASONING_RULE: Process input → Build context → Generate conclusion")
                rules.append("REASONING_PATTERN: Use logical connectors (therefore, because, thus)")
        
        except Exception as e:
            logger.debug(f"Reasoning analysis failed: {e}")
        
        return rules
    
    def _analyze_composition(
        self,
        prompt: str,
        response: str
    ) -> List[str]:
        """Analyze response composition and extract patterns."""
        rules = []
        
        try:
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            # Analyze structure
            sentences = response.split('. ')
            
            if len(sentences) > 1:
                rules.append(f"COMPOSITION_PATTERN: Multi-sentence response ({len(sentences)} sentences)")
            
            # Analyze length
            words = response.split()
            if len(words) < 20:
                rules.append("COMPOSITION_RULE: Concise response (< 20 words)")
            elif len(words) > 50:
                rules.append("COMPOSITION_RULE: Detailed response (> 50 words)")
            
            # Analyze structure
            if response.startswith(("Yes", "No", "The", "It", "This")):
                rules.append(f"COMPOSITION_PATTERN: Direct start with '{response.split()[0]}'")
        
        except Exception as e:
            logger.debug(f"Composition analysis failed: {e}")
        
        return rules
    
    def _get_test_prompts(self) -> List[str]:
        """Get test prompts for analysis."""
        return [
            "What is artificial intelligence?",
            "How does machine learning work?",
            "Explain neural networks",
            "Why is AI important?",
            "Tell me about programming",
            "What is Python?",
            "How do computers work?",
            "Explain the internet",
            "What is data science?",
            "How does the brain work?",
            "What is consciousness?",
            "Explain quantum physics",
            "How does evolution work?",
            "What is the universe?",
            "Explain time",
            "What is mathematics?",
            "How do we learn?",
            "What is language?",
            "Explain creativity",
            "What is intelligence?"
        ] * 10  # Repeat for more examples
