"""
Reasoning Instructions - Lehrt Friday wie man Reasoning und Antwort trennt.

Die KI soll lernen:
- Alle Überlegungen gehören ins <think> Block
- Nur die finale, saubere Antwort kommt danach
- Keine Meta-Informationen in der finalen Antwort
"""

REASONING_INSTRUCTIONS = [
    (
        "How should I structure my response?",
        "Structure your response in two parts: First, put all your reasoning, analysis, and thought process in the <think> block. Second, provide only the clean, final answer after the </think> tag. The final answer should be concise and direct, without any meta-commentary or reasoning traces.",
        ['reasoning', 'instruction', 'structure', 'format']
    ),
    (
        "What goes in the reasoning block?",
        "The reasoning block should contain: your understanding of the question, the type of question, your plan for answering, any analysis or comparisons you make, and intermediate thoughts. Everything that shows HOW you arrived at the answer goes in <think>.",
        ['reasoning', 'instruction', 'think-block']
    ),
    (
        "What goes in the final answer?",
        "The final answer should contain ONLY the direct response to the question. No phrases like 'Based on my analysis', 'As I mentioned', 'Additionally', or 'In my reasoning'. Just the clean, factual answer.",
        ['reasoning', 'instruction', 'final-answer']
    ),
    (
        "Should I explain my reasoning in the final answer?",
        "No. All reasoning and explanation of your thought process belongs in the <think> block. The final answer should be the result of your reasoning, not an explanation of it.",
        ['reasoning', 'instruction', 'separation']
    ),
    (
        "Can I say 'I think' or 'I believe' in the final answer?",
        "Avoid meta-commentary in the final answer. Instead of 'I think X is Y', just state 'X is Y'. Your confidence and reasoning should be in the <think> block.",
        ['reasoning', 'instruction', 'meta-commentary']
    ),
    (
        "What if I find multiple relevant pieces of information?",
        "In the <think> block, note that you found multiple sources and how you're choosing between them. In the final answer, present only the synthesized, unified response without mentioning that you had multiple sources.",
        ['reasoning', 'instruction', 'synthesis']
    ),
    (
        "Should I mention my knowledge sources in the answer?",
        "No. Don't say things like 'According to my knowledge' or 'Based on what I know'. Just state the information directly and confidently.",
        ['reasoning', 'instruction', 'sources']
    ),
    (
        "How do I handle uncertainty?",
        "Express uncertainty in the <think> block by noting what you're unsure about. In the final answer, either provide what you know confidently, or clearly state 'I don't have enough information about X' without lengthy explanations.",
        ['reasoning', 'instruction', 'uncertainty']
    ),
    (
        "What makes a good final answer?",
        "A good final answer is: concise, direct, factual, complete, and free of meta-commentary. It should stand alone without needing the reasoning block to make sense.",
        ['reasoning', 'instruction', 'quality']
    ),
    (
        "Example: How should I answer 'What is AI?'",
        "<think>\nUnderstand: Define AI\nType: definitional\nPlan: Provide clear definition with key characteristics\nSources: Found multiple definitions, will synthesize the most comprehensive one\n</think>\nAI (Artificial Intelligence) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.",
        ['reasoning', 'instruction', 'example', 'definition']
    ),
    (
        "Example: How should I answer 'Why is water important?'",
        "<think>\nUnderstand: Explain importance of water\nType: causal explanation\nPlan: List key reasons with focus on life-sustaining properties\nNote: Found info about biological, chemical, and environmental importance\n</think>\nWater is essential for all known forms of life. It regulates temperature, transports nutrients, and is involved in nearly every bodily function.",
        ['reasoning', 'instruction', 'example', 'causal']
    ),
    (
        "Example: Bad answer with meta-commentary",
        "BAD: 'Based on my analysis, I think AI is probably the simulation of human intelligence. Additionally, I found that it includes learning. As I mentioned in my reasoning, this is important.'\nGOOD: 'AI is the simulation of human intelligence processes by machines, including learning, reasoning, and self-correction.'",
        ['reasoning', 'instruction', 'example', 'bad-vs-good']
    ),
]


def get_reasoning_instructions():
    """Returns the reasoning instructions."""
    return REASONING_INSTRUCTIONS


def get_reasoning_instruction_count():
    """Returns the number of instruction pairs."""
    return len(REASONING_INSTRUCTIONS)
