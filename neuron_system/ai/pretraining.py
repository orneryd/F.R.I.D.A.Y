"""
Pre-training module for loading initial knowledge into the neuron system.

Provides pre-trained knowledge bases for language understanding.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.vector3d import Vector3D
from neuron_system.neuron_types.knowledge_neuron import KnowledgeNeuron
from neuron_system.engines.compression import CompressionEngine
from neuron_system.ai.language_model import LanguageModel

logger = logging.getLogger(__name__)


class EnglishKnowledgeBase:
    """
    Pre-defined English language knowledge base.
    
    Contains fundamental English language knowledge including:
    - Common words and their meanings
    - Grammar rules
    - Common phrases
    - Basic facts
    """
    
    # Basic vocabulary and definitions
    VOCABULARY = [
        "Hello means a greeting used when meeting someone",
        "Goodbye means a farewell expression when parting",
        "Thank you expresses gratitude or appreciation",
        "Please is used when making a polite request",
        "Yes indicates agreement or affirmation",
        "No indicates disagreement or negation",
        "Help means to assist or aid someone",
        "Question is an inquiry seeking information",
        "Answer is a response to a question",
        "Understand means to comprehend or grasp the meaning",
    ]
    
    # Grammar and language structure
    GRAMMAR = [
        "A sentence is a group of words expressing a complete thought",
        "A noun is a word that names a person, place, thing, or idea",
        "A verb is a word that expresses an action or state of being",
        "An adjective is a word that describes or modifies a noun",
        "An adverb is a word that modifies a verb, adjective, or another adverb",
        "A pronoun is a word that takes the place of a noun",
        "A preposition shows the relationship between a noun and another word",
        "A conjunction connects words, phrases, or clauses",
        "Subject-verb agreement means the verb must match the subject in number",
        "Past tense indicates an action that happened in the past",
    ]
    
    # Common phrases and expressions
    PHRASES = [
        "How are you is a common greeting asking about someone's wellbeing",
        "I'm fine thank you is a polite response to how are you",
        "Nice to meet you is said when meeting someone for the first time",
        "Excuse me is used to politely get someone's attention",
        "I'm sorry expresses regret or apology",
        "You're welcome is a response to thank you",
        "What's your name asks for someone's name",
        "My name is introduces yourself by stating your name",
        "How can I help you offers assistance to someone",
        "I don't understand indicates lack of comprehension",
    ]
    
    # Basic facts and general knowledge
    FACTS = [
        "English is a West Germanic language that originated in England",
        "The alphabet consists of 26 letters from A to Z",
        "A question typically ends with a question mark",
        "A statement typically ends with a period",
        "Capital letters are used at the beginning of sentences",
        "Proper nouns like names are capitalized",
        "Communication is the exchange of information between people",
        "Language is a system of communication using words and grammar",
        "Vocabulary refers to the words used in a language",
        "Grammar is the set of rules governing language structure",
    ]
    
    # Numbers and counting
    NUMBERS = [
        "One is the first natural number",
        "Two is the number after one",
        "Three is the number after two",
        "Four is the number after three",
        "Five is the number after four",
        "Ten is the number after nine",
        "Zero represents nothing or the absence of quantity",
        "Hundred is ten times ten",
        "Thousand is ten times hundred",
        "Numbers are used for counting and measuring",
    ]
    
    # Time and dates
    TIME = [
        "Today refers to the current day",
        "Yesterday refers to the day before today",
        "Tomorrow refers to the day after today",
        "Morning is the early part of the day",
        "Afternoon is the middle part of the day",
        "Evening is the later part of the day",
        "Night is the time when it is dark",
        "Week consists of seven days",
        "Month is approximately 30 days",
        "Year consists of 12 months or 365 days",
    ]
    
    # Colors
    COLORS = [
        "Red is a primary color like blood or fire",
        "Blue is a primary color like the sky or ocean",
        "Yellow is a primary color like the sun",
        "Green is the color of grass and leaves",
        "Black is the darkest color, absence of light",
        "White is the lightest color, presence of all colors",
        "Orange is a mix of red and yellow",
        "Purple is a mix of red and blue",
        "Brown is the color of wood or earth",
        "Gray is between black and white",
    ]
    
    # Actions and verbs
    ACTIONS = [
        "Walk means to move by putting one foot in front of the other",
        "Run means to move quickly on foot",
        "Eat means to consume food",
        "Drink means to consume liquid",
        "Sleep means to rest with eyes closed and reduced consciousness",
        "Think means to use your mind to consider or reason",
        "Speak means to say words aloud",
        "Listen means to pay attention to sounds",
        "Read means to look at and understand written words",
        "Write means to form letters and words on a surface",
    ]
    
    @classmethod
    def get_all_knowledge(cls) -> List[Dict[str, Any]]:
        """
        Get all pre-defined knowledge with categories.
        
        Returns:
            List of knowledge items with text and tags
        """
        knowledge = []
        
        # Add vocabulary
        for text in cls.VOCABULARY:
            knowledge.append({
                'text': text,
                'tags': ['vocabulary', 'basic', 'english']
            })
        
        # Add grammar
        for text in cls.GRAMMAR:
            knowledge.append({
                'text': text,
                'tags': ['grammar', 'language', 'english']
            })
        
        # Add phrases
        for text in cls.PHRASES:
            knowledge.append({
                'text': text,
                'tags': ['phrases', 'conversation', 'english']
            })
        
        # Add facts
        for text in cls.FACTS:
            knowledge.append({
                'text': text,
                'tags': ['facts', 'knowledge', 'english']
            })
        
        # Add numbers
        for text in cls.NUMBERS:
            knowledge.append({
                'text': text,
                'tags': ['numbers', 'counting', 'math']
            })
        
        # Add time
        for text in cls.TIME:
            knowledge.append({
                'text': text,
                'tags': ['time', 'dates', 'temporal']
            })
        
        # Add colors
        for text in cls.COLORS:
            knowledge.append({
                'text': text,
                'tags': ['colors', 'visual', 'descriptive']
            })
        
        # Add actions
        for text in cls.ACTIONS:
            knowledge.append({
                'text': text,
                'tags': ['actions', 'verbs', 'activities']
            })
        
        return knowledge


class PreTrainingLoader:
    """
    Loader for pre-training the neuron system with knowledge.
    """
    
    def __init__(
        self,
        language_model: LanguageModel,
        spatial_distribution: str = "grid"
    ):
        """
        Initialize pre-training loader.
        
        Args:
            language_model: Language model to load knowledge into
            spatial_distribution: How to distribute neurons spatially
                                 ('grid', 'random', 'clustered')
        """
        self.language_model = language_model
        self.spatial_distribution = spatial_distribution
        
        logger.info(f"PreTrainingLoader initialized with {spatial_distribution} distribution")
    
    def load_english_knowledge(
        self,
        create_connections: bool = True,
        batch_size: int = 10
    ) -> int:
        """
        Load English language knowledge base.
        
        Args:
            create_connections: Whether to create synapses between related neurons
            batch_size: Number of neurons to process before logging progress
            
        Returns:
            Number of neurons created
        """
        logger.info("Loading English knowledge base...")
        
        knowledge_items = EnglishKnowledgeBase.get_all_knowledge()
        total_items = len(knowledge_items)
        
        logger.info(f"Loading {total_items} knowledge items...")
        
        created_count = 0
        for i, item in enumerate(knowledge_items):
            # Generate position based on distribution strategy
            position = self._generate_position(i, total_items)
            
            # Learn the knowledge
            self.language_model.learn(
                text=item['text'],
                position=position,
                tags=item['tags'],
                create_connections=False  # We'll create connections after all neurons are loaded
            )
            
            created_count += 1
            
            # Log progress
            if (i + 1) % batch_size == 0:
                logger.info(f"Loaded {i + 1}/{total_items} knowledge items...")
        
        logger.info(f"Loaded {created_count} knowledge neurons")
        
        # Create connections between related neurons
        if create_connections:
            logger.info("Creating connections between related neurons...")
            self._create_all_connections()
            logger.info("Connections created")
        
        return created_count
    
    def _generate_position(self, index: int, total: int) -> Vector3D:
        """
        Generate position for a neuron based on distribution strategy.
        
        Args:
            index: Index of the neuron
            total: Total number of neurons
            
        Returns:
            Generated position
        """
        bounds = self.language_model.graph.bounds
        if bounds:
            min_bound, max_bound = bounds
        else:
            min_bound = Vector3D(-100, -100, -100)
            max_bound = Vector3D(100, 100, 100)
        
        if self.spatial_distribution == "grid":
            # Distribute in a 3D grid
            import math
            grid_size = math.ceil(total ** (1/3))  # Cube root for 3D grid
            
            x_idx = index % grid_size
            y_idx = (index // grid_size) % grid_size
            z_idx = index // (grid_size * grid_size)
            
            x = min_bound.x + (max_bound.x - min_bound.x) * x_idx / grid_size
            y = min_bound.y + (max_bound.y - min_bound.y) * y_idx / grid_size
            z = min_bound.z + (max_bound.z - min_bound.z) * z_idx / grid_size
            
            return Vector3D(x, y, z)
        
        elif self.spatial_distribution == "random":
            # Random distribution
            import numpy as np
            x = np.random.uniform(min_bound.x, max_bound.x)
            y = np.random.uniform(min_bound.y, max_bound.y)
            z = np.random.uniform(min_bound.z, max_bound.z)
            
            return Vector3D(x, y, z)
        
        else:  # clustered
            # Cluster by category (simplified)
            import numpy as np
            cluster_id = index % 8  # 8 clusters for 8 categories
            
            # Define cluster centers
            cluster_centers = [
                Vector3D(-50, -50, -50),  # vocabulary
                Vector3D(50, -50, -50),   # grammar
                Vector3D(-50, 50, -50),   # phrases
                Vector3D(50, 50, -50),    # facts
                Vector3D(-50, -50, 50),   # numbers
                Vector3D(50, -50, 50),    # time
                Vector3D(-50, 50, 50),    # colors
                Vector3D(50, 50, 50),     # actions
            ]
            
            center = cluster_centers[cluster_id]
            
            # Add some randomness around the center
            x = center.x + np.random.uniform(-20, 20)
            y = center.y + np.random.uniform(-20, 20)
            z = center.z + np.random.uniform(-20, 20)
            
            return Vector3D(x, y, z)
    
    def _create_all_connections(self):
        """Create connections between all related neurons."""
        neurons = list(self.language_model.graph.neurons.values())
        
        for i, neuron in enumerate(neurons):
            if isinstance(neuron, KnowledgeNeuron):
                self.language_model._create_connections(neuron, top_k=3)
            
            # Log progress every 10 neurons
            if (i + 1) % 10 == 0:
                logger.info(f"Created connections for {i + 1}/{len(neurons)} neurons...")
    
    def load_from_file(
        self,
        file_path: str,
        create_connections: bool = True
    ) -> int:
        """
        Load knowledge from a JSON file.
        
        File format:
        [
            {"text": "...", "tags": ["tag1", "tag2"]},
            ...
        ]
        
        Args:
            file_path: Path to JSON file
            create_connections: Whether to create connections
            
        Returns:
            Number of neurons created
        """
        logger.info(f"Loading knowledge from file: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            knowledge_items = json.load(f)
        
        created_count = 0
        for i, item in enumerate(knowledge_items):
            position = self._generate_position(i, len(knowledge_items))
            
            self.language_model.learn(
                text=item['text'],
                position=position,
                tags=item.get('tags', []),
                create_connections=False
            )
            
            created_count += 1
        
        if create_connections:
            self._create_all_connections()
        
        logger.info(f"Loaded {created_count} knowledge items from file")
        return created_count
    
    def save_knowledge_base(self, file_path: str):
        """
        Save current knowledge base to a JSON file.
        
        Args:
            file_path: Path to save JSON file
        """
        logger.info(f"Saving knowledge base to: {file_path}")
        
        knowledge_items = []
        for neuron in self.language_model.graph.neurons.values():
            if isinstance(neuron, KnowledgeNeuron) and hasattr(neuron, 'source_data'):
                knowledge_items.append({
                    'text': neuron.source_data,
                    'tags': getattr(neuron, 'semantic_tags', [])
                })
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(knowledge_items, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(knowledge_items)} knowledge items")
