"""
Additional Knowledge Base for comprehensive AI training.

Provides knowledge about:
- Science and Technology
- History and Culture
- Education and Learning
- Communication and Language
- General Knowledge
"""

from typing import List, Dict, Any


class AdditionalKnowledge:
    """
    Additional knowledge base for AI training.
    """
    
    # Science and Technology
    SCIENCE = [
        "Science is the systematic study of the natural world through observation and experimentation",
        "Physics is the study of matter, energy, and the fundamental forces of nature",
        "Chemistry is the study of substances, their properties, and how they interact",
        "Biology is the study of living organisms and life processes",
        "Technology is the application of scientific knowledge for practical purposes",
        "A computer is an electronic device that processes data and performs calculations",
        "The internet is a global network connecting millions of computers worldwide",
        "Artificial intelligence is the simulation of human intelligence by machines",
        "Mathematics is the study of numbers, quantities, shapes, and patterns",
        "An experiment is a scientific procedure to test a hypothesis or demonstrate a fact",
    ]
    
    # History and Culture
    HISTORY = [
        "History is the study of past events and human civilization",
        "Culture refers to the customs, beliefs, and practices of a society",
        "Ancient civilizations include Egypt, Greece, Rome, and Mesopotamia",
        "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century",
        "World War II was a global conflict from 1939 to 1945",
        "Democracy is a system of government where citizens have the power to vote",
        "Art is the expression of human creativity through various mediums",
        "Music is the arrangement of sounds in time to produce beauty or expression",
        "Literature consists of written works, especially those considered to have artistic merit",
        "Philosophy is the study of fundamental questions about existence, knowledge, and ethics",
    ]
    
    # Education and Learning
    EDUCATION = [
        "Education is the process of acquiring knowledge, skills, and values",
        "Learning is the acquisition of knowledge or skills through study or experience",
        "A school is an institution for educating children and young people",
        "A teacher is a person who instructs students and facilitates learning",
        "A student is a person who is studying at a school or university",
        "Reading is the process of interpreting written words to gain meaning",
        "Writing is the act of forming letters and words to communicate ideas",
        "A book is a written or printed work consisting of pages bound together",
        "A library is a collection of books and other resources for reading and study",
        "Knowledge is information, understanding, and skills acquired through experience or education",
    ]
    
    # Communication and Language
    COMMUNICATION = [
        "Communication is the exchange of information between individuals",
        "Language is a system of communication using words and grammar",
        "Speaking is the act of expressing thoughts through spoken words",
        "Listening is the act of paying attention to sounds and understanding their meaning",
        "Conversation is an informal exchange of ideas through spoken words",
        "A question is a sentence used to request information or clarification",
        "An answer is a response to a question providing requested information",
        "Understanding means comprehending the meaning or significance of something",
        "Explanation is the act of making something clear by describing it in detail",
        "Information is facts or knowledge provided or learned about something",
    ]
    
    # General Knowledge
    GENERAL = [
        "The Earth is the third planet from the Sun in our solar system",
        "Water is essential for all known forms of life",
        "The human body consists of various organs and systems working together",
        "Food provides energy and nutrients necessary for survival",
        "Health is a state of physical, mental, and social well-being",
        "Exercise is physical activity that improves health and fitness",
        "Sleep is a natural state of rest essential for physical and mental recovery",
        "Weather refers to atmospheric conditions like temperature, rain, and wind",
        "A city is a large human settlement with extensive infrastructure",
        "Transportation is the movement of people or goods from one place to another",
    ]
    
    # Problem Solving and Thinking
    THINKING = [
        "Thinking is the process of using one's mind to consider or reason",
        "A problem is a matter or situation requiring a solution",
        "A solution is the means of solving a problem or dealing with a difficulty",
        "Logic is reasoning conducted according to strict principles of validity",
        "Creativity is the use of imagination to create something new or original",
        "Analysis is the detailed examination of elements or structure of something",
        "Decision-making is the process of choosing between alternatives",
        "Memory is the faculty by which information is stored and recalled",
        "Attention is the mental focus on a particular object or task",
        "Understanding requires both knowledge and the ability to apply it appropriately",
    ]
    
    # Work and Society
    SOCIETY = [
        "Society is a group of people living together in an organized community",
        "Work is activity involving mental or physical effort to achieve a result",
        "A job is a paid position of regular employment",
        "Money is a medium of exchange used to purchase goods and services",
        "Economy refers to the system of production and consumption of goods and services",
        "Government is the system by which a state or community is governed",
        "Law is a system of rules created and enforced through social institutions",
        "Rights are moral or legal entitlements to have or do something",
        "Responsibility is the state of being accountable for something",
        "Cooperation is working together toward a common goal",
    ]
    
    # Emotions and Relationships
    RELATIONSHIPS = [
        "Love is a strong feeling of affection and care for someone",
        "Friendship is a relationship of mutual affection between people",
        "Trust is the firm belief in the reliability or truth of someone",
        "Respect is admiration for someone based on their qualities or achievements",
        "Kindness is the quality of being friendly, generous, and considerate",
        "Empathy is the ability to understand and share the feelings of others",
        "Honesty is the quality of being truthful and sincere",
        "Patience is the capacity to accept delay or suffering without getting angry",
        "Gratitude is the quality of being thankful and showing appreciation",
        "Forgiveness is the act of pardoning someone for a mistake or wrongdoing",
    ]
    
    @classmethod
    def get_all_knowledge(cls) -> List[Dict[str, Any]]:
        """
        Get all additional knowledge with categories.
        
        Returns:
            List of knowledge items with text and tags
        """
        knowledge = []
        
        # Add all categories
        categories = [
            (cls.SCIENCE, ['science', 'technology', 'knowledge']),
            (cls.HISTORY, ['history', 'culture', 'society']),
            (cls.EDUCATION, ['education', 'learning', 'knowledge']),
            (cls.COMMUNICATION, ['communication', 'language', 'interaction']),
            (cls.GENERAL, ['general', 'world', 'life']),
            (cls.THINKING, ['thinking', 'cognition', 'problem-solving']),
            (cls.SOCIETY, ['society', 'work', 'community']),
            (cls.RELATIONSHIPS, ['relationships', 'emotions', 'social']),
        ]
        
        for items, tags in categories:
            for text in items:
                knowledge.append({
                    'text': text,
                    'tags': tags
                })
        
        return knowledge
