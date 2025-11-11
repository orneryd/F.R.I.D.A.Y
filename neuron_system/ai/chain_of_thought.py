"""
Chain-of-Thought (CoT) Reasoning System.

Generates structured reasoning traces in <think> tags without requiring
a generative LLM. Uses heuristics and templates for fast, transparent reasoning.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ReasoningTrace:
    """Represents a chain-of-thought reasoning trace."""
    understand: str  # Restatement of the question
    question_type: str  # Type of question (causal, procedural, etc.)
    plan: str  # Brief plan for answering
    final_answer: str  # The actual answer
    used_neuron_ids: List[str]  # IDs of neurons used
    
    def to_text(self, include_think_tags: bool = True) -> str:
        """Convert to formatted text output."""
        if include_think_tags:
            think_block = (
                f"<think>\n"
                f"Understand: {self.understand}\n"
                f"Type: {self.question_type}\n"
                f"Plan: {self.plan}\n"
                f"</think>\n"
            )
            return think_block + self.final_answer
        else:
            return self.final_answer


class ChainOfThoughtGenerator:
    """
    Generates chain-of-thought reasoning traces.
    
    This is a lightweight, heuristic-based system that doesn't require
    a generative LLM. It uses templates and patterns to create structured
    reasoning traces.
    """
    
    def __init__(self):
        """Initialize CoT generator."""
        logger.info("Chain-of-Thought Generator initialized")
    
    def generate_reasoning_trace(
        self,
        query: str,
        question_type: str,
        context: Dict,
        final_answer: str,
        activated_neurons: List = None
    ) -> ReasoningTrace:
        """
        Generate a reasoning trace for the query.
        
        Args:
            query: The user's question
            question_type: Type of question (definition, how, why, etc.)
            context: Context from activated neurons
            final_answer: The generated answer
            activated_neurons: List of activated neurons
            
        Returns:
            ReasoningTrace object
        """
        # Step 1: Understand - restate the question
        understand = self._generate_understand(query, question_type)
        
        # Step 2: Classify question type
        qtype = self._classify_question_type(question_type, query)
        
        # Step 3: Generate plan
        plan = self._generate_plan(question_type, context)
        
        # Step 4: Extract neuron IDs
        neuron_ids = []
        if activated_neurons:
            neuron_ids = [str(n.neuron.id)[:8] for n in activated_neurons[:5]]
        
        return ReasoningTrace(
            understand=understand,
            question_type=qtype,
            plan=plan,
            final_answer=final_answer,
            used_neuron_ids=neuron_ids
        )
    
    def _generate_understand(self, query: str, question_type: str) -> str:
        """Generate 'Understand' step - restate the question."""
        # Clean up query
        query = query.strip()
        if not query.endswith('?'):
            query += '?'
        
        # For different question types, rephrase slightly
        if question_type == 'definition':
            # "What is X?" -> "Define X"
            match = re.match(r'what\s+is\s+(.+?)\??$', query, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                return f"Define {subject}"
        
        elif question_type in ['how', 'why']:
            # Keep as is but make it a statement
            return query.replace('?', '')
        
        # Default: just return the query
        return query
    
    def _classify_question_type(self, question_type: str, query: str) -> str:
        """Classify the question into a reasoning category."""
        query_lower = query.lower()
        
        # Map internal types to CoT types
        type_mapping = {
            'definition': 'definitional',
            'definition_plural': 'definitional',
            'how': 'procedural',
            'why': 'causal',
            'who': 'factual',
            'when': 'temporal',
            'where': 'spatial',
            'greeting': 'conversational',
            'question': 'general',
        }
        
        cot_type = type_mapping.get(question_type, 'general')
        
        # Check for comparison
        if any(word in query_lower for word in ['difference', 'compare', 'versus', 'vs', 'better']):
            cot_type = 'comparison'
        
        # Check for analysis
        if any(word in query_lower for word in ['analyze', 'evaluate', 'assess']):
            cot_type = 'analytical'
        
        return cot_type
    
    def _generate_plan(self, question_type: str, context: Dict) -> str:
        """Generate a brief plan for answering."""
        # Get context info
        tags = context.get('tags', set())
        keywords = context.get('keywords', set())
        
        # Generate plan based on question type
        if question_type == 'definition':
            return "Provide clear definition with key characteristics"
        
        elif question_type == 'how':
            return "Explain process step-by-step with mechanisms"
        
        elif question_type == 'why':
            return "Identify causes and explain causal relationships"
        
        elif question_type in ['who', 'when', 'where']:
            return "Retrieve specific factual information"
        
        elif question_type == 'greeting':
            return "Respond appropriately to greeting"
        
        else:
            # General plan based on available context
            if tags:
                tag_list = ', '.join(list(tags)[:3])
                return f"Synthesize information from {tag_list} knowledge"
            elif keywords:
                kw_list = ', '.join(list(keywords)[:3])
                return f"Combine relevant concepts: {kw_list}"
            else:
                return "Synthesize available knowledge into coherent answer"


class SelfReflection:
    """
    Lightweight self-reflection system.
    
    Validates and potentially enhances the generated answer without
    requiring a generative LLM.
    """
    
    def __init__(self):
        """Initialize self-reflection system."""
        logger.info("Self-Reflection system initialized")
    
    def reflect(
        self,
        query: str,
        answer: str,
        question_type: str,
        context: Dict
    ) -> Tuple[str, bool]:
        """
        Reflect on the answer and potentially enhance it.
        
        Args:
            query: The original question
            answer: The generated answer
            question_type: Type of question
            context: Context from neurons
            
        Returns:
            Tuple of (enhanced_answer, was_modified)
        """
        # Check for obvious issues
        issues = self._check_answer_quality(query, answer, question_type)
        
        if not issues:
            return answer, False
        
        # Try to fix issues
        enhanced = self._enhance_answer(answer, issues, context)
        
        return enhanced, (enhanced != answer)
    
    def _check_answer_quality(
        self,
        query: str,
        answer: str,
        question_type: str
    ) -> List[str]:
        """Check for quality issues in the answer."""
        issues = []
        
        # Check 1: Answer too short
        if len(answer) < 20:
            issues.append('too_short')
        
        # Check 2: Answer is just the question
        if query.lower().replace('?', '') in answer.lower():
            if len(answer) < len(query) + 20:
                issues.append('just_question')
        
        # Check 3: Answer contains meta-text
        meta_patterns = ['<think>', '</think>', 'Question:', 'Answer:']
        if any(pattern in answer for pattern in meta_patterns):
            issues.append('meta_text')
        
        # Check 4: Definition questions should have "is" or "are"
        if question_type == 'definition':
            if not any(word in answer.lower() for word in [' is ', ' are ', ' was ', ' were ']):
                issues.append('missing_definition')
        
        return issues
    
    def _enhance_answer(
        self,
        answer: str,
        issues: List[str],
        context: Dict
    ) -> str:
        """Enhance the answer to fix issues."""
        enhanced = answer
        
        # Fix meta-text
        if 'meta_text' in issues:
            enhanced = re.sub(r'<think>.*?</think>', '', enhanced, flags=re.DOTALL)
            enhanced = re.sub(r'Question:\s*', '', enhanced, flags=re.IGNORECASE)
            enhanced = re.sub(r'Answer:\s*', '', enhanced, flags=re.IGNORECASE)
            enhanced = enhanced.strip()
        
        # Fix "just question" issue
        if 'just_question' in issues:
            # Try to find a better sentence from context
            sentences = context.get('sentences', [])
            if sentences:
                # Take the best sentence
                enhanced = sentences[0].get('text', answer)
        
        # Fix too short
        if 'too_short' in issues and len(enhanced) < 20:
            # Add a generic enhancement
            enhanced = enhanced + " (based on available knowledge)"
        
        return enhanced.strip()
