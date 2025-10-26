"""
Extended English Knowledge Base for Language AI

Provides comprehensive English language knowledge including:
- Expanded vocabulary (500+ words)
- Grammar rules and examples
- Common phrases and idioms
- General knowledge
- Conversational patterns
"""

from typing import List, Dict, Any


class ExtendedEnglishKnowledge:
    """
    Comprehensive English language knowledge base.
    
    Contains extensive vocabulary, grammar, phrases, and general knowledge
    to enable natural language understanding and generation.
    """
    
    # Expanded Greetings and Social Interactions
    GREETINGS = [
        "Hello is a common greeting used when meeting someone or starting a conversation",
        "Hi is an informal greeting similar to hello but more casual",
        "Good morning is a greeting used before noon",
        "Good afternoon is a greeting used between noon and evening",
        "Good evening is a greeting used after sunset",
        "Goodbye means farewell when parting from someone",
        "See you later means goodbye with expectation of meeting again",
        "Take care is a friendly way to say goodbye wishing someone well",
        "How are you asks about someone's wellbeing or current state",
        "I'm fine thank you is a polite response indicating you are well",
        "Nice to meet you expresses pleasure at meeting someone new",
        "Pleased to meet you is a formal way to greet someone new",
        "Welcome is used to greet someone arriving or to show hospitality",
    ]
    
    # Colors with detailed descriptions
    COLORS = [
        "Red is a primary color associated with fire, blood, passion, and energy",
        "Blue is a primary color associated with sky, ocean, calmness, and trust",
        "Yellow is a primary color associated with sun, happiness, and optimism",
        "Green is a secondary color made from blue and yellow, associated with nature and growth",
        "Orange is a secondary color made from red and yellow, associated with warmth and enthusiasm",
        "Purple is a secondary color made from red and blue, associated with royalty and creativity",
        "Black is the darkest color representing absence of light, associated with elegance and mystery",
        "White is the lightest color representing purity, cleanliness, and simplicity",
        "Gray is a neutral color between black and white, associated with balance and neutrality",
        "Brown is an earth tone associated with wood, soil, and stability",
        "Pink is a light red color associated with gentleness and romance",
        "Turquoise is a blue-green color associated with tropical waters and tranquility",
    ]
    
    # Numbers and Mathematics
    NUMBERS = [
        "Zero represents nothing or the absence of quantity, written as 0",
        "One is the first natural number representing a single unit",
        "Two is the number after one representing a pair or couple",
        "Three is the number after two representing a trio or triple",
        "Four is the number after three representing a quartet",
        "Five is the number after four, half of ten",
        "Six is the number after five, half a dozen",
        "Seven is the number after six, considered lucky in many cultures",
        "Eight is the number after seven, representing infinity when turned sideways",
        "Nine is the number after eight, the last single digit",
        "Ten is the first two-digit number, the base of our decimal system",
        "Hundred is ten times ten, written as 100",
        "Thousand is ten times hundred, written as 1000",
        "Million is one thousand thousand, written as 1000000",
        "Addition is the mathematical operation of combining numbers to get a sum",
        "Subtraction is the mathematical operation of taking away to find the difference",
        "Multiplication is repeated addition to find a product",
        "Division is splitting into equal parts to find a quotient",
    ]
    
    # Time and Temporal Concepts
    TIME = [
        "Time is the ongoing sequence of events from past through present to future",
        "Second is the base unit of time, about one heartbeat",
        "Minute is sixty seconds, a short period of time",
        "Hour is sixty minutes, one twenty-fourth of a day",
        "Day is twenty-four hours, one rotation of Earth",
        "Week is seven days from Monday to Sunday",
        "Month is approximately thirty days, one twelfth of a year",
        "Year is twelve months or 365 days, one orbit around the sun",
        "Today refers to the current day, this very day",
        "Yesterday refers to the day before today, the recent past",
        "Tomorrow refers to the day after today, the near future",
        "Morning is the early part of the day from sunrise to noon",
        "Afternoon is the middle part of the day from noon to evening",
        "Evening is the later part of the day from late afternoon to night",
        "Night is the time when it is dark, from sunset to sunrise",
        "Dawn is the first light of day, sunrise",
        "Dusk is the last light of day, sunset",
        "Past refers to time that has already happened",
        "Present refers to the current moment, now",
        "Future refers to time that has not yet happened",
        "Always means at all times, without exception",
        "Never means at no time, not ever",
        "Sometimes means occasionally, not always but not never",
        "Often means frequently, many times",
        "Rarely means seldom, not often",
    ]
    
    # Actions and Verbs (Expanded)
    ACTIONS = [
        "Walk means to move by putting one foot in front of the other at a moderate pace",
        "Run means to move quickly on foot faster than walking",
        "Jump means to push off the ground with your legs to go up in the air",
        "Sit means to rest with your weight on your buttocks",
        "Stand means to be upright on your feet",
        "Lie means to be in a horizontal position, to recline",
        "Eat means to consume food by chewing and swallowing",
        "Drink means to consume liquid by swallowing",
        "Sleep means to rest with eyes closed and reduced consciousness",
        "Wake means to stop sleeping and become conscious",
        "Think means to use your mind to consider, reason, or remember",
        "Feel means to experience an emotion or physical sensation",
        "See means to perceive with your eyes, to look at",
        "Hear means to perceive sound with your ears, to listen",
        "Speak means to say words aloud, to talk",
        "Listen means to pay attention to sounds or speech",
        "Read means to look at and understand written words",
        "Write means to form letters and words on a surface",
        "Draw means to make pictures with a pen, pencil, or other tool",
        "Paint means to apply color to a surface with a brush",
        "Sing means to make musical sounds with your voice",
        "Dance means to move rhythmically to music",
        "Play means to engage in activity for enjoyment",
        "Work means to do tasks or labor for a purpose or payment",
        "Study means to learn about a subject through reading and practice",
        "Teach means to help someone learn by giving information",
        "Learn means to gain knowledge or skill through study or experience",
        "Help means to assist or aid someone in need",
        "Give means to transfer something to someone else",
        "Take means to get hold of something or receive it",
        "Buy means to purchase something by paying money",
        "Sell means to exchange something for money",
        "Make means to create or produce something",
        "Build means to construct something by putting parts together",
        "Break means to separate into pieces or damage",
        "Fix means to repair something that is broken",
        "Open means to make accessible or not closed",
        "Close means to shut or make not open",
        "Start means to begin or commence an action",
        "Stop means to cease or end an action",
        "Continue means to keep going without stopping",
        "Finish means to complete or bring to an end",
    ]
    
    # Emotions and Feelings
    EMOTIONS = [
        "Happy means feeling joy, pleasure, or contentment",
        "Sad means feeling sorrow, unhappiness, or grief",
        "Angry means feeling strong displeasure or rage",
        "Afraid means feeling fear or anxiety about something",
        "Excited means feeling enthusiastic and eager",
        "Calm means feeling peaceful and relaxed",
        "Nervous means feeling worried or anxious",
        "Proud means feeling satisfaction in achievements",
        "Ashamed means feeling embarrassed or guilty",
        "Surprised means feeling astonishment at something unexpected",
        "Confused means feeling uncertain or unable to understand",
        "Confident means feeling sure of yourself and your abilities",
        "Lonely means feeling sad from being alone",
        "Grateful means feeling thankful and appreciative",
        "Jealous means feeling envious of someone else",
        "Love is a deep affection and care for someone",
        "Hate is an intense dislike or aversion",
        "Hope is a feeling of expectation and desire for something",
        "Fear is an unpleasant emotion caused by threat or danger",
        "Joy is a feeling of great happiness and delight",
    ]
    
    # Common Objects and Things
    OBJECTS = [
        "Book is a written or printed work consisting of pages bound together",
        "Pen is a writing instrument that uses ink",
        "Pencil is a writing instrument with a graphite core",
        "Paper is a thin material made from wood pulp used for writing",
        "Table is a piece of furniture with a flat top and legs",
        "Chair is a piece of furniture for one person to sit on",
        "Door is a movable barrier used to close an entrance",
        "Window is an opening in a wall with glass to let in light",
        "House is a building where people live",
        "Car is a vehicle with four wheels used for transportation",
        "Phone is a device used for communication over distance",
        "Computer is an electronic device for processing data",
        "Television is a device for receiving broadcast images and sound",
        "Clock is a device for measuring and showing time",
        "Watch is a small clock worn on the wrist",
        "Bag is a container made of flexible material for carrying things",
        "Shoe is footwear that covers and protects the foot",
        "Clothes are items worn to cover the body",
        "Food is any substance consumed to provide nutrition",
        "Water is a clear liquid essential for life",
        "Money is a medium of exchange used to buy goods and services",
        "Key is a device used to open locks",
        "Light is electromagnetic radiation visible to the human eye",
        "Sun is the star at the center of our solar system providing light and heat",
        "Moon is Earth's natural satellite visible in the night sky",
    ]
    
    # Grammar and Language Structure (Expanded)
    GRAMMAR = [
        "A sentence is a group of words expressing a complete thought with subject and predicate",
        "A noun is a word that names a person, place, thing, or idea",
        "A verb is a word that expresses an action, occurrence, or state of being",
        "An adjective is a word that describes or modifies a noun",
        "An adverb is a word that modifies a verb, adjective, or another adverb",
        "A pronoun is a word that takes the place of a noun like he, she, it, they",
        "A preposition shows the relationship between a noun and another word like in, on, at",
        "A conjunction connects words, phrases, or clauses like and, but, or",
        "An article is a word like a, an, or the that introduces a noun",
        "Subject is the person or thing performing the action in a sentence",
        "Predicate is the part of a sentence containing the verb and describing the subject",
        "Object is the person or thing receiving the action of the verb",
        "Clause is a group of words containing a subject and verb",
        "Phrase is a group of words without both subject and verb",
        "Present tense indicates an action happening now",
        "Past tense indicates an action that happened before now",
        "Future tense indicates an action that will happen later",
        "Singular refers to one person or thing",
        "Plural refers to more than one person or thing",
        "Question is a sentence that asks for information and ends with a question mark",
        "Statement is a sentence that declares something and ends with a period",
        "Exclamation is a sentence expressing strong emotion and ends with exclamation mark",
        "Capital letter is an uppercase letter used at the start of sentences and names",
        "Punctuation marks are symbols used to organize and clarify writing",
        "Period is a punctuation mark indicating the end of a statement",
        "Comma is a punctuation mark indicating a pause or separating items",
        "Question mark is a punctuation mark at the end of a question",
        "Exclamation mark shows strong feeling or emphasis",
    ]
    
    # Common Phrases and Expressions
    PHRASES = [
        "Thank you expresses gratitude or appreciation for something",
        "You're welcome is a polite response to thank you",
        "Please is used when making a polite request",
        "Excuse me is used to politely get attention or apologize for interruption",
        "I'm sorry expresses regret, apology, or sympathy",
        "No problem means something is not difficult or you're happy to help",
        "Of course means certainly or naturally, expressing agreement",
        "I don't know means lacking information or understanding about something",
        "I understand means comprehending what has been said",
        "I agree means having the same opinion",
        "I disagree means having a different opinion",
        "What's your name asks for someone's name",
        "My name is introduces yourself by stating your name",
        "How can I help you offers assistance to someone",
        "Can you help me requests assistance from someone",
        "I need help indicates requiring assistance",
        "Where is asks for the location of something",
        "What time is it asks for the current time",
        "How much does it cost asks for the price of something",
        "I would like expresses a desire or preference politely",
        "May I asks permission politely",
        "Could you is a polite way to make a request",
        "Would you mind asks if someone objects to something",
        "It doesn't matter means something is not important",
        "Never mind means forget about it or it's not important",
        "I see means understanding or acknowledging information",
        "That makes sense means something is logical or understandable",
        "Good idea expresses approval of a suggestion",
        "I think so expresses tentative agreement or belief",
        "I don't think so expresses tentative disagreement or doubt",
    ]
    
    # General Knowledge and Facts
    FACTS = [
        "English is a West Germanic language that originated in medieval England",
        "The alphabet consists of 26 letters from A to Z",
        "Vowels are the letters A, E, I, O, U and sometimes Y",
        "Consonants are all letters that are not vowels",
        "Communication is the exchange of information between people",
        "Language is a system of communication using words and grammar",
        "Vocabulary refers to the words used in a language",
        "Grammar is the set of rules governing language structure",
        "Reading is the process of understanding written text",
        "Writing is the process of creating written text",
        "Speaking is verbal communication using words",
        "Listening is receiving and understanding spoken communication",
        "Conversation is an exchange of spoken words between people",
        "Question is an inquiry seeking information or clarification",
        "Answer is a response providing information to a question",
        "Information is facts or knowledge about something",
        "Knowledge is understanding gained through experience or education",
        "Learning is the process of acquiring knowledge or skills",
        "Teaching is the process of helping others learn",
        "Understanding is comprehending the meaning or nature of something",
        "Meaning is the significance or definition of something",
        "Definition is an explanation of what a word or concept means",
        "Example is a specific instance illustrating a general rule",
        "Explanation is making something clear by describing it",
        "Description is giving details about something's characteristics",
    ]
    
    # Family and Relationships
    FAMILY = [
        "Family is a group of people related by blood or marriage",
        "Mother is a female parent who gave birth to or raised a child",
        "Father is a male parent who helped create or raised a child",
        "Parent is a mother or father who raises a child",
        "Child is a young person, son or daughter of parents",
        "Son is a male child in relation to parents",
        "Daughter is a female child in relation to parents",
        "Brother is a male sibling sharing the same parents",
        "Sister is a female sibling sharing the same parents",
        "Grandmother is the mother of one's parent",
        "Grandfather is the father of one's parent",
        "Aunt is the sister of one's parent",
        "Uncle is the brother of one's parent",
        "Cousin is the child of one's aunt or uncle",
        "Friend is a person you know well and like",
        "Friendship is a relationship of mutual affection between people",
    ]
    
    # Places and Locations
    PLACES = [
        "Home is the place where one lives permanently",
        "School is an institution for education and learning",
        "Office is a place where professional work is done",
        "Store is a place where goods are sold to customers",
        "Restaurant is a place where meals are prepared and served",
        "Hospital is a place where sick or injured people receive medical treatment",
        "Library is a place where books and information are available",
        "Park is an outdoor area for recreation and nature",
        "City is a large town with many people and buildings",
        "Town is a populated area smaller than a city",
        "Village is a small community in a rural area",
        "Country is a nation with its own government and territory",
        "World is the earth and all people and things on it",
        "Street is a public road in a city or town",
        "Building is a structure with walls and a roof",
    ]
    
    # Nature and Environment
    NATURE = [
        "Nature is the physical world including plants, animals, and landscapes",
        "Tree is a large plant with a wooden trunk and branches",
        "Flower is the colorful part of a plant that produces seeds",
        "Grass is a green plant with narrow leaves covering the ground",
        "Animal is a living creature that can move and sense its environment",
        "Bird is an animal with feathers and wings that usually flies",
        "Fish is an animal that lives in water and breathes through gills",
        "Dog is a domesticated animal often kept as a pet",
        "Cat is a small domesticated animal often kept as a pet",
        "Sky is the space above the earth where clouds and sun appear",
        "Cloud is a visible mass of water droplets floating in the sky",
        "Rain is water falling from clouds in drops",
        "Snow is frozen water falling from clouds as white flakes",
        "Wind is moving air that you can feel",
        "Weather is the state of the atmosphere at a particular time and place",
        "Season is one of the four divisions of the year: spring, summer, fall, winter",
        "Spring is the season when plants begin to grow",
        "Summer is the warmest season of the year",
        "Fall or autumn is the season when leaves change color and fall",
        "Winter is the coldest season of the year",
    ]
    
    @classmethod
    def get_all_knowledge(cls) -> List[Dict[str, Any]]:
        """
        Get all extended knowledge with categories.
        
        Returns:
            List of knowledge items with text and tags
        """
        knowledge = []
        
        # Add all categories
        categories = [
            (cls.GREETINGS, ['greetings', 'social', 'conversation']),
            (cls.COLORS, ['colors', 'visual', 'descriptive']),
            (cls.NUMBERS, ['numbers', 'math', 'counting']),
            (cls.TIME, ['time', 'temporal', 'dates']),
            (cls.ACTIONS, ['actions', 'verbs', 'activities']),
            (cls.EMOTIONS, ['emotions', 'feelings', 'psychology']),
            (cls.OBJECTS, ['objects', 'things', 'nouns']),
            (cls.GRAMMAR, ['grammar', 'language', 'structure']),
            (cls.PHRASES, ['phrases', 'expressions', 'conversation']),
            (cls.FACTS, ['facts', 'knowledge', 'information']),
            (cls.FAMILY, ['family', 'relationships', 'people']),
            (cls.PLACES, ['places', 'locations', 'geography']),
            (cls.NATURE, ['nature', 'environment', 'world']),
        ]
        
        for items, tags in categories:
            for text in items:
                knowledge.append({
                    'text': text,
                    'tags': tags + ['english']
                })
        
        return knowledge
