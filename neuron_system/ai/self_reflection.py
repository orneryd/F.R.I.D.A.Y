"""
Self-Reflection System for AI to validate and improve its own responses.

This system enables the AI to:
- Question its own answers
- Detect nonsensical responses
- Apply reasoning to validate responses
- Self-correct when needed
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SelfReflection:
    """
    Self-reflection system for response validation and improvement.
    """
    
    # Patterns that indicate a bad response
    BAD_RESPONSE_PATTERNS = [
        "= = ",  # Wiki formatting artifacts
        "[truncated",  # Truncated content
        "When asked",  # Meta-instructions
        "When someone",  # Meta-instructions
        "means asking",  # Meta-instructions
        "means requesting",  # Meta-instructions
        "respond with",  # Meta-instructions
        "should respond",  # Meta-instructions
    ]
    
    # Questions the AI should ask itself
    SELF_QUESTIONS = [
        "Does this answer make sense?",
        "Is this answer relevant to the question?",
        "Is this answer complete?",
        "Does this answer contain artifacts or meta-text?",
        "Would a human understand this answer?",
        "Is this answer too short or too vague?",
        "Does this answer actually answer the question?",
    ]
    
    def __init__(self, language_model):
        """
        Initialize self-reflection system.
        
        Args:
            language_model: LanguageModel instance for reasoning
        """
        self.language_model = language_model
        self.reflection_count = 0
        self.correction_count = 0
    
    def validate_response(
        self,
        query: str,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validate a response and potentially improve it.
        
        Args:
            query: Original query
            response: Generated response
            knowledge_pieces: Knowledge used to generate response
            
        Returns:
            Tuple of (is_valid, improved_response, reasoning)
        """
        self.reflection_count += 1
        
        # Step 1: Check for obvious bad patterns
        for pattern in self.BAD_RESPONSE_PATTERNS:
            if pattern in response:
                logger.warning(f"Bad pattern detected: {pattern}")
                return False, None, f"Response contains artifact: {pattern}"
        
        # Step 2: Check if response is too short
        if len(response.strip()) < 10:
            logger.warning("Response too short")
            return False, None, "Response is too short to be meaningful"
        
        # Step 3: Check if response is just a question back
        if response.strip().endswith('?') and len(response) < 100:
            # It's okay to ask clarifying questions, but check if it's appropriate
            if not self._is_clarifying_question_appropriate(query, response):
                logger.warning("Inappropriate question-only response")
                return False, None, "Response is just a question without attempting to answer"
        
        # Step 4: Check relevance using reasoning
        is_relevant = self._check_relevance(query, response)
        if not is_relevant:
            logger.warning("Response not relevant to query")
            # Try to find a better response
            improved = self._find_better_response(query, knowledge_pieces)
            if improved:
                self.correction_count += 1
                return False, improved, "Found more relevant response"
            return False, None, "Response not relevant to query"
        
        # Step 5: Check for completeness
        is_complete = self._check_completeness(query, response)
        if not is_complete:
            logger.info("Response could be more complete")
            # Try to enhance response
            enhanced = self._enhance_response(query, response, knowledge_pieces)
            if enhanced and enhanced != response:
                self.correction_count += 1
                return True, enhanced, "Enhanced response for completeness"
        
        # Response is valid
        return True, None, "Response validated successfully"
    
    def _is_clarifying_question_appropriate(self, query: str, response: str) -> bool:
        """
        Check if asking a clarifying question is appropriate.
        
        Args:
            query: Original query
            response: Response (a question)
            
        Returns:
            True if appropriate, False otherwise
        """
        # If query is vague or ambiguous, clarifying questions are good
        vague_indicators = [
            'what about', 'how about', 'tell me about',
            'what if', 'suppose', 'imagine'
        ]
        
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in vague_indicators)
    
    def _check_relevance(self, query: str, response: str) -> bool:
        """
        Check if response is relevant to query using reasoning.
        LENIENT VERSION: Don't be too strict, many good answers don't repeat query words.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            True if relevant, False otherwise
        """
        # LENIENT: If response is substantial, assume it's relevant
        if len(response.strip()) >= 30:
            # Long responses are usually relevant
            return True
        
        # Simple heuristic: check if key words from query appear in response
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'can', 'could', 'should', 'may', 'might', 'must', 'i', 'you',
                       'he', 'she', 'it', 'we', 'they', 'what', 'when', 'where',
                       'who', 'why', 'how', 'which', 'this', 'that', 'these', 'those',
                       'my', 'your', 'me', 'to', 'of', 'in', 'on', 'at', 'for', 'with'}
        
        query_words -= common_words
        response_words -= common_words
        
        # Check overlap
        if not query_words:
            return True  # Query too short to judge
        
        overlap = len(query_words & response_words)
        relevance_ratio = overlap / len(query_words)
        
        # LENIENT: Only 10% overlap needed (was 20%)
        # Many good answers don't repeat query words
        return relevance_ratio >= 0.1
    
    def _check_completeness(self, query: str, response: str) -> bool:
        """
        Check if response is complete enough.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            True if complete, False otherwise
        """
        # Check if response is too short for the query complexity
        query_words = len(query.split())
        response_words = len(response.split())
        
        # For complex queries (>10 words), expect substantial responses
        if query_words > 10 and response_words < 15:
            return False
        
        # For simple queries, shorter responses are okay
        if query_words <= 10 and response_words < 5:
            return False
        
        return True
    
    def _find_better_response(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Try to find a better response from knowledge pieces.
        
        Args:
            query: Original query
            knowledge_pieces: Available knowledge
            
        Returns:
            Better response or None
        """
        # Look for Q&A format responses
        for piece in knowledge_pieces:
            text = piece['text']
            if 'Question:' in text and 'Answer:' in text:
                # Extract answer
                answer_start = text.find('Answer:') + 7
                answer = text[answer_start:].strip()
                
                # Check if it's better (no bad patterns, reasonable length)
                if len(answer) >= 15 and not any(p in answer for p in self.BAD_RESPONSE_PATTERNS):
                    return answer
        
        # Look for direct statements
        for piece in knowledge_pieces:
            text = piece['text'].strip()
            # Skip meta-instructions
            if any(p in text for p in self.BAD_RESPONSE_PATTERNS):
                continue
            # Use if it's substantial
            if len(text) >= 20 and len(text) < 200:
                return text
        
        return None
    
    def _enhance_response(
        self,
        query: str,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Try to enhance response with additional information.
        
        Args:
            query: Original query
            response: Current response
            knowledge_pieces: Available knowledge
            
        Returns:
            Enhanced response or None
        """
        # Find additional relevant information
        additional_info = []
        
        for piece in knowledge_pieces:
            text = piece['text'].strip()
            # Skip if already in response
            if text in response:
                continue
            # Skip meta-instructions
            if any(p in text for p in self.BAD_RESPONSE_PATTERNS):
                continue
            # Add if relevant and not too long
            if len(text) >= 20 and len(text) < 150:
                additional_info.append(text)
                if len(additional_info) >= 2:
                    break
        
        if not additional_info:
            return None
        
        # Combine response with additional info
        enhanced = response
        for info in additional_info:
            # Add with natural connector
            if not enhanced.endswith('.'):
                enhanced += '.'
            enhanced += f" Additionally, {info.lower() if info[0].isupper() else info}"
        
        return enhanced
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-reflection statistics."""
        return {
            'total_reflections': self.reflection_count,
            'total_corrections': self.correction_count,
            'correction_rate': (
                self.correction_count / self.reflection_count
                if self.reflection_count > 0 else 0.0
            )
        }


class ChainOfThought:
    """
    Chain-of-Thought reasoning for complex queries.
    
    Breaks down complex queries into steps and reasons through them.
    """
    
    def __init__(self, language_model):
        """
        Initialize chain-of-thought system.
        
        Args:
            language_model: LanguageModel instance
        """
        self.language_model = language_model
    
    def reason_through(
        self,
        query: str,
        max_steps: int = 3
    ) -> Tuple[str, List[str]]:
        """
        Reason through a query step by step.
        
        Args:
            query: Complex query
            max_steps: Maximum reasoning steps
            
        Returns:
            Tuple of (final_answer, reasoning_steps)
        """
        reasoning_steps = []
        
        # Step 1: Understand what's being asked
        step1 = f"Understanding the question: {query}"
        reasoning_steps.append(step1)
        
        # Step 2: Break down the query
        query_lower = query.lower()
        
        # Identify query type
        if any(word in query_lower for word in ['why', 'because', 'reason']):
            step2 = "This is a causal question requiring explanation of reasons"
        elif any(word in query_lower for word in ['how', 'process', 'method']):
            step2 = "This is a procedural question requiring step-by-step explanation"
        elif any(word in query_lower for word in ['what', 'define', 'explain']):
            step2 = "This is a definitional question requiring clear explanation"
        elif any(word in query_lower for word in ['compare', 'difference', 'similar']):
            step2 = "This is a comparison question requiring analysis of similarities/differences"
        else:
            step2 = "This is a general question requiring informative response"
        
        reasoning_steps.append(step2)
        
        # Step 3: Find relevant knowledge
        step3 = "Searching for relevant knowledge to answer this question"
        reasoning_steps.append(step3)
        
        # Generate answer using the language model
        answer = self.language_model.generate_response(
            query,
            context_size=5,
            min_activation=0.1,
            use_reasoning=True
        )
        
        return answer, reasoning_steps
