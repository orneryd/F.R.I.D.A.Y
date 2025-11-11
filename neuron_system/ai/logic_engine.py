"""
Logic Engine - Execute extracted logic from assimilated models.

This engine doesn't just STORE logic as text, it EXECUTES it:
- Attention rules → Applied during context processing
- Generation rules → Applied during word selection
- Reasoning rules → Applied during thinking
- Composition rules → Applied during response building

This makes Friday truly dynamic, using Qwen's thinking patterns.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class LogicEngine:
    """
    Executes logic extracted from assimilated models.
    
    Instead of just storing "ATTENTION_RULE: Focus on beginning",
    this engine APPLIES that rule during processing.
    """
    
    def __init__(self, graph, query_engine):
        """
        Initialize logic engine.
        
        Args:
            graph: NeuronGraph instance
            query_engine: QueryEngine instance
        """
        self.graph = graph
        self.query_engine = query_engine
        self.logic_cache = {
            "attention": [],
            "generation": [],
            "reasoning": [],
            "composition": []
        }
        self._load_logic_rules()
    
    def _load_logic_rules(self):
        """Load all logic rules from neurons."""
        logger.info("Loading logic rules from neurons...")
        
        # Find all logic neurons
        logic_neurons = [
            n for n in self.graph.neurons.values()
            if hasattr(n, 'semantic_tags') and 'logic' in n.semantic_tags
        ]
        
        logger.info(f"Found {len(logic_neurons)} logic neurons")
        
        # Parse and categorize rules
        for neuron in logic_neurons:
            if hasattr(neuron, 'source_data'):
                rule_text = neuron.source_data
                rule = self._parse_rule(rule_text)
                
                if rule:
                    rule_type = rule['type']
                    if rule_type in self.logic_cache:
                        self.logic_cache[rule_type].append(rule)
        
        # Log statistics
        for rule_type, rules in self.logic_cache.items():
            logger.info(f"  {rule_type.capitalize()}: {len(rules)} rules")
    
    def _parse_rule(self, rule_text: str) -> Optional[Dict[str, Any]]:
        """Parse a rule from text into executable format."""
        try:
            # Attention rules
            if "ATTENTION_RULE:" in rule_text or "ATTENTION_PATTERN:" in rule_text:
                return {
                    "type": "attention",
                    "text": rule_text,
                    "executable": self._make_attention_rule(rule_text)
                }
            
            # Generation rules
            elif "GENERATION_RULE:" in rule_text:
                return {
                    "type": "generation",
                    "text": rule_text,
                    "executable": self._make_generation_rule(rule_text)
                }
            
            # Reasoning rules
            elif "REASONING_RULE:" in rule_text or "REASONING_PATTERN:" in rule_text:
                return {
                    "type": "reasoning",
                    "text": rule_text,
                    "executable": self._make_reasoning_rule(rule_text)
                }
            
            # Composition rules
            elif "COMPOSITION_RULE:" in rule_text or "COMPOSITION_PATTERN:" in rule_text:
                return {
                    "type": "composition",
                    "text": rule_text,
                    "executable": self._make_composition_rule(rule_text)
                }
        
        except Exception as e:
            logger.debug(f"Failed to parse rule: {e}")
        
        return None
    
    def _make_attention_rule(self, rule_text: str) -> callable:
        """Create executable attention rule."""
        if "beginning" in rule_text.lower():
            # Focus on beginning
            def attention_func(context_pieces):
                # Boost first pieces
                for i, piece in enumerate(context_pieces[:3]):
                    piece['activation'] = piece.get('activation', 0.5) * 1.2
                return context_pieces
            return attention_func
        
        elif "recent" in rule_text.lower() or "end" in rule_text.lower():
            # Focus on recent/end
            def attention_func(context_pieces):
                # Boost last pieces
                for i, piece in enumerate(context_pieces[-3:]):
                    piece['activation'] = piece.get('activation', 0.5) * 1.2
                return context_pieces
            return attention_func
        
        else:
            # Default: no modification
            return lambda x: x
    
    def _make_generation_rule(self, rule_text: str) -> callable:
        """Create executable generation rule."""
        # Extract confidence threshold if present
        confidence_match = re.search(r'(\d+\.?\d*)', rule_text)
        
        if "high confidence" in rule_text.lower():
            threshold = float(confidence_match.group(1)) if confidence_match else 0.7
            
            def generation_func(candidates):
                # If top candidate has high confidence, use it
                if candidates and candidates[0].get('score', 0) > threshold:
                    return candidates[0]
                return candidates[0] if candidates else None
            return generation_func
        
        elif "low confidence" in rule_text.lower():
            def generation_func(candidates):
                # Consider multiple options
                return candidates[:3] if len(candidates) >= 3 else candidates
            return generation_func
        
        else:
            return lambda x: x[0] if x else None
    
    def _make_reasoning_rule(self, rule_text: str) -> callable:
        """Create executable reasoning rule."""
        if "process input" in rule_text.lower():
            # Multi-step reasoning
            def reasoning_func(query, context):
                steps = []
                steps.append(f"Input: {query}")
                steps.append(f"Context: {len(context)} pieces")
                steps.append("Conclusion: [to be generated]")
                return steps
            return reasoning_func
        
        elif "logical connector" in rule_text.lower():
            # Add logical connectors
            def reasoning_func(query, context):
                connectors = ["therefore", "because", "thus", "hence"]
                return {"use_connectors": connectors}
            return reasoning_func
        
        else:
            return lambda q, c: []
    
    def _make_composition_rule(self, rule_text: str) -> callable:
        """Create executable composition rule."""
        if "concise" in rule_text.lower():
            # Keep response short
            def composition_func(response):
                words = response.split()
                if len(words) > 20:
                    return ' '.join(words[:20]) + '...'
                return response
            return composition_func
        
        elif "detailed" in rule_text.lower():
            # Allow longer response
            def composition_func(response):
                return response  # No truncation
            return composition_func
        
        elif "multi-sentence" in rule_text.lower():
            # Ensure multiple sentences
            def composition_func(response):
                if '. ' not in response and len(response) > 30:
                    # Add sentence break
                    mid = len(response) // 2
                    return response[:mid] + '. ' + response[mid:]
                return response
            return composition_func
        
        else:
            return lambda x: x
    
    def apply_attention(
        self,
        context_pieces: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply attention rules to context pieces.
        
        Args:
            context_pieces: List of context pieces with activations
            
        Returns:
            Modified context pieces with attention applied
        """
        if not self.logic_cache["attention"]:
            return context_pieces
        
        # Apply all attention rules
        for rule in self.logic_cache["attention"]:
            try:
                context_pieces = rule["executable"](context_pieces)
            except Exception as e:
                logger.debug(f"Attention rule failed: {e}")
        
        # Re-sort by activation
        context_pieces.sort(key=lambda x: x.get('activation', 0), reverse=True)
        
        return context_pieces
    
    def apply_generation(
        self,
        candidates: List[Dict[str, Any]]
    ) -> Any:
        """
        Apply generation rules to select best candidate.
        
        Args:
            candidates: List of candidate responses with scores
            
        Returns:
            Selected candidate(s)
        """
        if not self.logic_cache["generation"]:
            return candidates[0] if candidates else None
        
        # Apply first applicable generation rule
        for rule in self.logic_cache["generation"]:
            try:
                result = rule["executable"](candidates)
                if result:
                    return result
            except Exception as e:
                logger.debug(f"Generation rule failed: {e}")
        
        return candidates[0] if candidates else None
    
    def apply_reasoning(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply reasoning rules to build logical flow.
        
        Args:
            query: User query
            context: Context pieces
            
        Returns:
            Reasoning structure
        """
        reasoning = {
            "steps": [],
            "connectors": [],
            "structure": "linear"
        }
        
        if not self.logic_cache["reasoning"]:
            return reasoning
        
        # Apply all reasoning rules
        for rule in self.logic_cache["reasoning"]:
            try:
                result = rule["executable"](query, context)
                
                if isinstance(result, list):
                    reasoning["steps"].extend(result)
                elif isinstance(result, dict):
                    reasoning.update(result)
            
            except Exception as e:
                logger.debug(f"Reasoning rule failed: {e}")
        
        return reasoning
    
    def apply_composition(
        self,
        response: str
    ) -> str:
        """
        Apply composition rules to structure response.
        
        Args:
            response: Raw response text
            
        Returns:
            Composed response
        """
        if not self.logic_cache["composition"]:
            return response
        
        # Apply all composition rules
        for rule in self.logic_cache["composition"]:
            try:
                response = rule["executable"](response)
            except Exception as e:
                logger.debug(f"Composition rule failed: {e}")
        
        return response
    
    def generate_with_logic(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response using extracted logic.
        
        This is the main method that applies ALL logic types
        to generate a response like Qwen would.
        
        Args:
            query: User query
            knowledge_pieces: Retrieved knowledge
            
        Returns:
            Generated response
        """
        # Step 1: Apply attention to focus on relevant context
        focused_context = self.apply_attention(knowledge_pieces.copy())
        
        # Step 2: Apply reasoning to build logical flow
        reasoning = self.apply_reasoning(query, focused_context)
        
        # Step 3: Generate candidate responses
        candidates = self._generate_candidates(query, focused_context, reasoning)
        
        # Step 4: Apply generation rules to select best
        selected = self.apply_generation(candidates)
        
        # Step 5: Apply composition rules to structure
        if isinstance(selected, dict):
            response = selected.get('text', '')
        elif isinstance(selected, list):
            response = selected[0].get('text', '') if selected else ''
        else:
            response = str(selected) if selected else ''
        
        final_response = self.apply_composition(response)
        
        return final_response
    
    def _generate_candidates(
        self,
        query: str,
        context: List[Dict[str, Any]],
        reasoning: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate candidate responses."""
        candidates = []
        
        # Candidate 1: Direct from top context
        if context:
            top_text = context[0].get('text', '')
            candidates.append({
                'text': top_text,
                'score': context[0].get('activation', 0.5),
                'method': 'direct'
            })
        
        # Candidate 2: Combined from top 2
        if len(context) >= 2:
            combined = f"{context[0].get('text', '')} {context[1].get('text', '')}"
            avg_score = (context[0].get('activation', 0.5) + context[1].get('activation', 0.5)) / 2
            candidates.append({
                'text': combined,
                'score': avg_score * 0.9,  # Slightly lower for combined
                'method': 'combined'
            })
        
        # Candidate 3: With reasoning connectors
        if context and reasoning.get('use_connectors'):
            connector = reasoning['use_connectors'][0]
            text_with_connector = f"{context[0].get('text', '')} {connector}"
            candidates.append({
                'text': text_with_connector,
                'score': context[0].get('activation', 0.5) * 0.95,
                'method': 'reasoning'
            })
        
        # Sort by score
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates
    
    def get_logic_stats(self) -> Dict[str, int]:
        """Get statistics about loaded logic."""
        return {
            "attention_rules": len(self.logic_cache["attention"]),
            "generation_rules": len(self.logic_cache["generation"]),
            "reasoning_rules": len(self.logic_cache["reasoning"]),
            "composition_rules": len(self.logic_cache["composition"]),
            "total_rules": sum(len(rules) for rules in self.logic_cache.values())
        }
