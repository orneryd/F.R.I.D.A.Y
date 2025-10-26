"""
Reasoning Knowledge Base for teaching the AI logical tasks.

Provides examples and patterns for:
- Counting letters in words
- Basic reasoning tasks
- Pattern recognition
"""

from typing import List, Dict, Any


class ReasoningKnowledge:
    """
    Knowledge base for reasoning and logical tasks.
    """
    
    # Letter counting examples
    LETTER_COUNTING = [
        # Direct answer format - specific letter counts
        "The letter 'r' appears 3 times in strawberry",
        "There are 3 r's in the word strawberry",
        "Strawberry contains 3 letter r",
        "The letter 'p' appears 2 times in apple",
        "There are 2 p's in the word apple",
        "Apple contains 2 letter p",
        "The letter 'o' appears 2 times in book",
        "There are 2 o's in the word book",
        "Book contains 2 letter o",
        "The letter 'e' appears 2 times in tree",
        "There are 2 e's in the word tree",
        "Tree contains 2 letter e",
        "The letter 'l' appears 2 times in hello",
        "There are 2 l's in the word hello",
        "Hello contains 2 letter l",
        
        # Complete breakdown format
        "The word 'apple' has 1 letter 'a', 2 letters 'p', 1 letter 'l', and 1 letter 'e'",
        "The word 'book' has 1 letter 'b', 2 letters 'o', and 1 letter 'k'",
        "The word 'tree' has 1 letter 't', 2 letters 'r', and 2 letters 'e'",
        "The word 'hello' has 1 letter 'h', 1 letter 'e', 2 letters 'l', and 1 letter 'o'",
        
        # Strawberry detailed
        "The word 'strawberry' has 1 letter 's', 1 letter 't', 3 letters 'r', 1 letter 'a', 1 letter 'w', 1 letter 'b', 1 letter 'e', and 1 letter 'y'",
        "The word 'strawberry' contains the letter 'r' three times: st-r-awbe-rr-y",
        "To count letters in 'strawberry': s-t-r-a-w-b-e-r-r-y has 3 r's",
        "In 'strawberry', the letter 'r' appears 3 times: position 3, position 8, and position 9",
        
        # More examples with repeated letters
        "The word 'letter' has 1 letter 'l', 2 letters 'e', 2 letters 't', and 1 letter 'r'",
        "The letter 'e' appears 2 times in letter",
        "The letter 't' appears 2 times in letter",
        "The word 'banana' has 1 letter 'b', 3 letters 'a', and 2 letters 'n'",
        "The letter 'a' appears 3 times in banana",
        "The letter 'n' appears 2 times in banana",
        "The word 'mississippi' has 1 letter 'm', 4 letters 'i', 4 letters 's', and 2 letters 'p'",
        "The letter 'i' appears 4 times in mississippi",
        "The letter 's' appears 4 times in mississippi",
        "The letter 'p' appears 2 times in mississippi",
        "The word 'coffee' has 1 letter 'c', 1 letter 'o', 2 letters 'f', and 2 letters 'e'",
        "The letter 'f' appears 2 times in coffee",
        "The letter 'e' appears 2 times in coffee",
        
        # Teaching the method
        "To count a letter in a word, go through each letter one by one and count how many times it appears",
        "When counting letters, treat uppercase and lowercase as the same letter",
        "Letter counting is case-insensitive: 'A' and 'a' are the same letter",
    ]
    
    # Reasoning patterns
    REASONING_PATTERNS = [
        "To solve a counting problem, break it down into individual steps",
        "When analyzing a word, examine each character separately",
        "Pattern recognition helps identify repeated elements",
        "Systematic analysis requires going through data methodically",
        "Breaking complex problems into smaller parts makes them easier to solve",
    ]
    
    # Number and counting concepts
    COUNTING_CONCEPTS = [
        "Counting means determining the total number of items",
        "To count accurately, check each item exactly once",
        "Zero means there are no items of that type",
        "One means there is a single item",
        "Two means there are a pair of items",
        "Three means there are three items",
        "Multiple means more than one",
        "Repeated means appearing more than once",
    ]
    
    @classmethod
    def get_all_knowledge(cls) -> List[Dict[str, Any]]:
        """
        Get all reasoning knowledge with categories.
        
        Returns:
            List of knowledge items with text and tags
        """
        knowledge = []
        
        # Add all categories
        categories = [
            (cls.LETTER_COUNTING, ['counting', 'letters', 'analysis', 'reasoning']),
            (cls.REASONING_PATTERNS, ['reasoning', 'logic', 'problem-solving']),
            (cls.COUNTING_CONCEPTS, ['counting', 'numbers', 'concepts']),
            (cls.MATH_REASONING, ['math', 'arithmetic', 'reasoning']),
            (cls.LOGICAL_REASONING, ['logic', 'reasoning', 'deduction']),
            (cls.COMPARISON_REASONING, ['comparison', 'analysis', 'reasoning']),
            (cls.CAUSE_EFFECT, ['causation', 'reasoning', 'analysis']),
            (cls.PROBLEM_SOLVING, ['problem-solving', 'strategy', 'reasoning']),
            (cls.PATTERN_RECOGNITION, ['patterns', 'recognition', 'reasoning']),
            (cls.CLASSIFICATION, ['classification', 'categorization', 'reasoning']),
            (cls.TEMPORAL_REASONING, ['time', 'temporal', 'reasoning']),
            (cls.SPATIAL_REASONING, ['space', 'spatial', 'reasoning']),
            (cls.PROBABILITY_REASONING, ['probability', 'uncertainty', 'reasoning']),
            (cls.DEDUCTIVE_REASONING, ['deduction', 'logic', 'reasoning']),
            (cls.INDUCTIVE_REASONING, ['induction', 'generalization', 'reasoning']),
            (cls.ANALOGICAL_REASONING, ['analogy', 'comparison', 'reasoning']),
            (cls.CRITICAL_THINKING, ['critical', 'analysis', 'reasoning']),
            (cls.ABSTRACT_REASONING, ['abstract', 'concepts', 'reasoning']),
        ]
        
        for items, tags in categories:
            for text in items:
                knowledge.append({
                    'text': text,
                    'tags': tags
                })
        
        return knowledge

    
    # Mathematical reasoning
    MATH_REASONING = [
        "Addition means combining two or more numbers to get a sum",
        "Subtraction means taking one number away from another",
        "Multiplication means repeated addition of the same number",
        "Division means splitting a number into equal parts",
        "2 plus 2 equals 4",
        "5 minus 3 equals 2",
        "3 times 4 equals 12",
        "10 divided by 2 equals 5",
        "Even numbers are divisible by 2",
        "Odd numbers are not divisible by 2",
        "Prime numbers are only divisible by 1 and themselves",
        "Zero is neither positive nor negative",
        "Negative numbers are less than zero",
        "Positive numbers are greater than zero",
        "Fractions represent parts of a whole",
        "Percentages are fractions out of 100",
    ]
    
    # Logical reasoning
    LOGICAL_REASONING = [
        "If A is true and B is true, then both A and B are true",
        "If A is true or B is true, then at least one is true",
        "If A implies B, and A is true, then B must be true",
        "If A implies B, and B is false, then A must be false",
        "Correlation does not imply causation",
        "All humans are mortal, Socrates is human, therefore Socrates is mortal",
        "If it rains, the ground gets wet. The ground is wet, but it might not have rained",
        "Necessary conditions must be present for something to occur",
        "Sufficient conditions guarantee that something will occur",
        "Contradictions cannot both be true at the same time",
        "Tautologies are always true by definition",
        "Valid arguments have conclusions that follow from premises",
        "Sound arguments are valid and have true premises",
    ]
    
    # Comparison and analysis
    COMPARISON_REASONING = [
        "Greater than means one value is larger than another",
        "Less than means one value is smaller than another",
        "Equal means two values are the same",
        "Similar means having common characteristics",
        "Different means not the same",
        "Identical means exactly the same in every way",
        "Comparable means able to be compared",
        "Incomparable means cannot be meaningfully compared",
        "Maximum is the largest value in a set",
        "Minimum is the smallest value in a set",
        "Average is the sum divided by the count",
        "Median is the middle value when sorted",
        "Mode is the most frequently occurring value",
    ]
    
    # Cause and effect
    CAUSE_EFFECT = [
        "Cause is what makes something happen",
        "Effect is what happens as a result",
        "Direct causes immediately lead to effects",
        "Indirect causes lead to effects through intermediaries",
        "Multiple causes can lead to a single effect",
        "A single cause can lead to multiple effects",
        "Correlation means things occur together",
        "Causation means one thing causes another",
        "Necessary causes must be present for an effect",
        "Sufficient causes guarantee an effect will occur",
    ]
    
    # Problem-solving strategies
    PROBLEM_SOLVING = [
        "Define the problem clearly before attempting to solve it",
        "Break complex problems into smaller, manageable parts",
        "Identify what information you have and what you need",
        "Consider multiple approaches to solving a problem",
        "Test your solution to verify it works",
        "Learn from mistakes to improve future problem-solving",
        "Ask clarifying questions when information is unclear",
        "Use analogies to relate new problems to familiar ones",
        "Work backwards from the desired outcome",
        "Eliminate impossible solutions to narrow options",
        "Prioritize the most important aspects of a problem",
        "Document your reasoning process for future reference",
    ]
    
    # Pattern recognition
    PATTERN_RECOGNITION = [
        "Patterns are regularities that repeat",
        "Sequences follow a specific order",
        "Cycles repeat in a predictable manner",
        "Trends show general directions of change",
        "Anomalies are deviations from expected patterns",
        "Symmetry means balanced proportions",
        "Repetition means occurring multiple times",
        "Periodicity means occurring at regular intervals",
        "Fibonacci sequence: each number is the sum of the two preceding ones",
        "Arithmetic sequences increase by a constant difference",
        "Geometric sequences multiply by a constant ratio",
        "Recognizing patterns helps predict future occurrences",
    ]
    
    # Classification and categorization
    CLASSIFICATION = [
        "Classification means grouping similar items together",
        "Categories are groups with shared characteristics",
        "Hierarchies organize items from general to specific",
        "Taxonomy is a system of classification",
        "Supersets contain subsets",
        "Subsets are contained within supersets",
        "Mutually exclusive categories don't overlap",
        "Overlapping categories share some members",
        "Binary classification divides into two groups",
        "Multi-class classification divides into many groups",
        "Attributes are characteristics used for classification",
        "Labels identify which category an item belongs to",
    ]
    
    # Temporal reasoning
    TEMPORAL_REASONING = [
        "Before means earlier in time",
        "After means later in time",
        "During means at the same time as",
        "Simultaneous means happening at the same time",
        "Sequential means happening in order",
        "Concurrent means happening together",
        "Past refers to time before now",
        "Present refers to the current time",
        "Future refers to time after now",
        "Duration is how long something lasts",
        "Frequency is how often something occurs",
        "Intervals are spaces between occurrences",
    ]
    
    # Spatial reasoning
    SPATIAL_REASONING = [
        "Above means higher in position",
        "Below means lower in position",
        "Left means to the left side",
        "Right means to the right side",
        "Inside means within boundaries",
        "Outside means beyond boundaries",
        "Near means close in distance",
        "Far means distant",
        "Adjacent means next to",
        "Opposite means across from",
        "Parallel means running alongside without meeting",
        "Perpendicular means at right angles",
    ]
    
    # Probability and uncertainty
    PROBABILITY_REASONING = [
        "Probability measures likelihood of occurrence",
        "Certain means will definitely happen",
        "Impossible means cannot happen",
        "Likely means has a high probability",
        "Unlikely means has a low probability",
        "Random means without predictable pattern",
        "Expected value is the average outcome",
        "Risk is the potential for loss",
        "Uncertainty means lack of complete knowledge",
        "Confidence indicates degree of certainty",
    ]
    
    # Deductive reasoning
    DEDUCTIVE_REASONING = [
        "Deduction moves from general principles to specific conclusions",
        "All mammals are warm-blooded, whales are mammals, therefore whales are warm-blooded",
        "If all A are B, and C is A, then C is B",
        "Syllogisms are forms of deductive reasoning",
        "Valid deductions guarantee true conclusions if premises are true",
        "Deductive reasoning is certain when premises are true",
        "Modus ponens: If P then Q, P is true, therefore Q is true",
        "Modus tollens: If P then Q, Q is false, therefore P is false",
    ]
    
    # Inductive reasoning
    INDUCTIVE_REASONING = [
        "Induction moves from specific observations to general conclusions",
        "The sun has risen every day, therefore it will rise tomorrow",
        "Inductive reasoning provides probable but not certain conclusions",
        "Generalizations are formed from multiple observations",
        "Sample size affects reliability of inductive conclusions",
        "Inductive reasoning is strengthened by more evidence",
        "Counterexamples can disprove inductive generalizations",
        "Statistical inference uses inductive reasoning",
    ]
    
    # Analogical reasoning
    ANALOGICAL_REASONING = [
        "Analogies compare similar relationships",
        "A is to B as C is to D shows analogical relationships",
        "Analogical reasoning transfers knowledge from familiar to unfamiliar",
        "Metaphors use analogical thinking",
        "Similarities between situations enable analogical reasoning",
        "Analogies help explain complex concepts",
        "Weak analogies have few relevant similarities",
        "Strong analogies have many relevant similarities",
    ]
    
    # Critical thinking
    CRITICAL_THINKING = [
        "Critical thinking means analyzing information objectively",
        "Question assumptions before accepting conclusions",
        "Evaluate evidence for reliability and relevance",
        "Consider alternative explanations",
        "Identify biases in reasoning",
        "Distinguish facts from opinions",
        "Recognize logical fallacies",
        "Assess credibility of sources",
        "Look for hidden assumptions",
        "Consider context when evaluating information",
    ]
    
    # Abstract reasoning
    ABSTRACT_REASONING = [
        "Abstract thinking deals with concepts rather than concrete objects",
        "Generalization creates abstract concepts from specific examples",
        "Abstraction removes specific details to focus on essentials",
        "Concepts are abstract ideas",
        "Principles are abstract rules",
        "Theories are abstract explanations",
        "Models are abstract representations",
        "Symbols represent abstract concepts",
    ]
