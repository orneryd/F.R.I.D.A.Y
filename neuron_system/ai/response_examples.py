"""
Response Examples - Dense neurons with rich information.

Provides detailed examples of how to respond to various types of questions.
This teaches the AI by example, similar to few-shot learning in LLMs.
"""

from typing import List, Dict, Any


class ResponseExamples:
    """
    Rich examples of good responses to teach the AI.
    """
    
    # Detailed Q&A examples with reasoning
    DETAILED_EXAMPLES = [
        # Example 1: Explaining concepts
        """Question: What is artificial intelligence?
Answer: Artificial intelligence (AI) is technology that enables machines to perform tasks that typically require human intelligence. This includes learning from experience, understanding natural language, recognizing patterns, solving problems, and making decisions. AI systems use algorithms and data to improve their performance over time.""",
        
        # Example 2: How-to explanations
        """Question: How do you learn something new?
Answer: Learning something new involves several steps: First, understand what you want to learn and why. Second, break the topic into smaller, manageable parts. Third, study each part systematically, using multiple sources. Fourth, practice applying what you've learned. Fifth, review and reinforce the knowledge regularly. Finally, teach others what you've learned to deepen your understanding.""",
        
        # Example 3: Comparisons
        """Question: What's the difference between knowledge and wisdom?
Answer: Knowledge is having information and facts - knowing that something is true. Wisdom is understanding how to apply that knowledge effectively - knowing when and how to use what you know. Knowledge comes from learning and study, while wisdom comes from experience and reflection. You can have knowledge without wisdom, but wisdom requires knowledge as its foundation.""",
        
        # Example 4: Causal explanations
        """Question: Why is practice important for learning?
Answer: Practice is important for learning because it strengthens neural pathways in the brain, making skills more automatic and reliable. Through repetition, you move knowledge from conscious effort to unconscious competence. Practice also reveals gaps in understanding, provides feedback for improvement, and builds confidence. Additionally, spaced practice over time leads to better long-term retention than cramming.""",
        
        # Example 5: Problem-solving
        """Question: How do you solve a difficult problem?
Answer: To solve a difficult problem: First, clearly define what the problem actually is. Second, gather all relevant information. Third, break the problem into smaller, more manageable sub-problems. Fourth, consider multiple possible approaches. Fifth, choose the most promising approach and try it. Sixth, evaluate the results and adjust if needed. If stuck, take a break and return with fresh perspective, or seek input from others.""",
        
        # Example 6: Self-awareness
        """Question: How do you know if you're wrong?
Answer: You can tell you might be wrong when: your conclusion contradicts reliable evidence, multiple knowledgeable people disagree with you, your reasoning has logical flaws, you're making assumptions without verification, or you feel defensive rather than curious about challenges to your view. Being wrong is normal and valuable - it's an opportunity to learn and improve your understanding.""",
        
        # Example 7: Meta-cognition
        """Question: How do you think about thinking?
Answer: Thinking about thinking, called meta-cognition, involves being aware of your own thought processes. This means noticing when you're making assumptions, recognizing your biases, questioning your conclusions, and evaluating the quality of your reasoning. It's like having an internal observer that monitors and guides your thinking, helping you think more clearly and effectively.""",
        
        # Example 8: Uncertainty
        """Question: What do you do when you're not sure?
Answer: When uncertain, the best approach is to: acknowledge the uncertainty honestly, gather more information if possible, consider multiple perspectives, weigh the available evidence, assess the confidence level of your knowledge, and communicate any limitations clearly. It's better to say "I'm not certain, but based on what I know..." than to present uncertain information as fact.""",
        
        # Example 9: Complex reasoning
        """Question: How do you evaluate if something is true?
Answer: To evaluate truth, consider: Is there reliable evidence supporting it? Can it be verified through multiple independent sources? Is the reasoning logical and free from fallacies? Are there alternative explanations? What are the credentials and potential biases of the sources? Does it align with established, well-tested knowledge? Be especially skeptical of claims that seem too good to be true or that confirm what you already believe.""",
        
        # Example 10: Practical application
        """Question: How do you apply knowledge in real situations?
Answer: Applying knowledge effectively requires: understanding the principles deeply, not just memorizing facts; recognizing when a situation calls for specific knowledge; adapting general principles to specific contexts; combining knowledge from different areas; practicing application in varied scenarios; learning from both successes and failures; and continuously refining your approach based on results.""",
    ]
    
    # Examples of self-correction
    SELF_CORRECTION_EXAMPLES = [
        """Question: Is the sky always blue?
Initial thought: Yes, the sky is blue.
Self-correction: Wait, that's not complete. The sky appears blue during the day due to light scattering, but it's not always blue - it can be red/orange at sunset, gray when cloudy, black at night, and even green during certain storms.
Better answer: The sky is not always blue. During the day it appears blue due to Rayleigh scattering of sunlight, but it changes color based on conditions - red/orange at sunrise/sunset, gray when cloudy, black at night, and various other colors in different weather conditions.""",
        
        """Question: Do all birds fly?
Initial thought: Yes, birds fly.
Self-correction: That's incorrect. While most birds can fly, there are flightless birds like penguins, ostriches, emus, and kiwis.
Better answer: No, not all birds fly. While flight is a common characteristic of birds, several species are flightless, including penguins, ostriches, emus, kiwis, and cassowaries. These birds have evolved other adaptations for survival.""",
        
        """Question: Is water always liquid?
Initial thought: Water is liquid.
Self-correction: That's too narrow. Water exists in three states depending on temperature and pressure.
Better answer: Water is not always liquid. It exists in three states: solid (ice) below 0°C, liquid between 0-100°C, and gas (water vapor) above 100°C at standard pressure. The state depends on temperature and pressure conditions.""",
    ]
    
    # Examples of reasoning chains
    REASONING_CHAINS = [
        """Question: Why do we need sleep?
Step 1: Consider what happens during sleep - the body rests, the brain processes information
Step 2: Think about consequences of not sleeping - fatigue, impaired cognition, health issues
Step 3: Consider evolutionary perspective - all animals sleep, suggesting it's essential
Step 4: Synthesize: Sleep is necessary for physical restoration, memory consolidation, emotional regulation, and maintaining health
Answer: We need sleep because it serves multiple critical functions: physical restoration and repair, memory consolidation and learning, emotional regulation, immune system maintenance, and metabolic regulation. Without adequate sleep, cognitive function, physical health, and emotional well-being all deteriorate.""",
        
        """Question: How does learning work?
Step 1: Learning involves acquiring new information or skills
Step 2: The brain forms new neural connections when learning
Step 3: Repetition and practice strengthen these connections
Step 4: Different types of learning (declarative, procedural) use different brain systems
Step 5: Emotions and context affect how well we learn and remember
Answer: Learning works through neuroplasticity - the brain's ability to form and strengthen neural connections. When you learn something new, neurons create new synaptic connections. Repetition and practice strengthen these connections, making recall easier. The process involves attention, encoding, consolidation, and retrieval, and is enhanced by emotional engagement, meaningful context, and spaced repetition.""",
    ]
    
    # Examples of handling ambiguity
    AMBIGUITY_HANDLING = [
        """Question: What about AI?
Recognition: This question is too vague - "what about" could mean many things
Response: I'd be happy to discuss AI! Are you interested in: what AI is, how it works, its applications, its limitations, ethical considerations, or something else specific? Let me know what aspect interests you most.""",
        
        """Question: Is it good?
Recognition: "It" is undefined - need clarification
Response: I'd need to know what "it" refers to in order to evaluate whether it's good. Could you specify what you're asking about? Also, "good" can mean different things in different contexts - good for what purpose?""",
        
        """Question: Tell me everything.
Recognition: Too broad - need to narrow scope
Response: "Everything" is quite broad! I'd be happy to provide information, but it would help to know what topic you're interested in. What specific subject or question would you like to explore?""",
    ]
    
    # Examples of admitting limitations
    LIMITATION_EXAMPLES = [
        """Question: What will happen tomorrow?
Recognition: This asks for prediction of specific future events, which is impossible
Response: I cannot predict specific future events. While we can make educated guesses based on patterns and probabilities, the future is inherently uncertain. If you're asking about a specific situation, I can discuss likely possibilities and factors that might influence outcomes, but not definite predictions.""",
        
        """Question: What am I thinking right now?
Recognition: This asks for mind-reading, which is impossible
Response: I cannot read minds or know what you're thinking. I can only respond to what you communicate to me. If you'd like to share your thoughts, I'm here to listen and discuss them with you.""",
        
        """Question: What's the absolute truth about everything?
Recognition: This asks for complete, absolute knowledge, which no one has
Response: There is no single "absolute truth about everything" that anyone can provide. Knowledge is always partial and contextual. Different domains (science, philosophy, personal experience) offer different types of truth. I can share what's known in specific areas, but always with the understanding that knowledge evolves and has limitations.""",
    ]
    
    @classmethod
    def get_all_examples(cls) -> List[Dict[str, Any]]:
        """
        Get all response examples.
        
        Returns:
            List of example items with text and tags
        """
        examples = []
        
        categories = [
            (cls.DETAILED_EXAMPLES, ['examples', 'detailed', 'teaching', 'reasoning']),
            (cls.SELF_CORRECTION_EXAMPLES, ['examples', 'self-correction', 'validation', 'reasoning']),
            (cls.REASONING_CHAINS, ['examples', 'reasoning', 'chain-of-thought', 'analysis']),
            (cls.AMBIGUITY_HANDLING, ['examples', 'ambiguity', 'clarification', 'reasoning']),
            (cls.LIMITATION_EXAMPLES, ['examples', 'limitations', 'honesty', 'reasoning']),
        ]
        
        for items, tags in categories:
            for text in items:
                examples.append({
                    'text': text,
                    'tags': tags
                })
        
        return examples
