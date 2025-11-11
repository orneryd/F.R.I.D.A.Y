"""
Comprehensive Q&A Knowledge Base for Friday.

100+ Q&A pairs covering various topics.
"""

QA_DATA = [
    # === TECHNOLOGY & AI ===
    ("What is AI?", "AI (Artificial Intelligence) is the simulation of human intelligence processes by machines, especially computer systems. These processes include learning, reasoning, and self-correction.", ['ai', 'definition', 'technology']),
    ("What is machine learning?", "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.", ['ml', 'ai', 'technology']),
    ("What is deep learning?", "Deep learning is a subset of machine learning that uses neural networks with multiple layers to progressively extract higher-level features from raw input.", ['deep-learning', 'ai', 'technology']),
    ("What is a neural network?", "A neural network is a computing system inspired by biological neural networks that constitute animal brains. It consists of interconnected nodes (neurons) that process information.", ['neural-network', 'ai', 'technology']),
    ("What is natural language processing?", "Natural language processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language.", ['nlp', 'ai', 'technology']),
    ("What is a computer?", "A computer is an electronic device that manipulates information or data. It has the ability to store, retrieve, and process data.", ['computer', 'definition', 'technology']),
    ("What is programming?", "Programming is the process of creating a set of instructions that tell a computer how to perform a task. It involves writing code in programming languages.", ['programming', 'definition', 'technology']),
    ("What is the internet?", "The internet is a global network of billions of computers and other electronic devices that allows access to information and communication from anywhere in the world.", ['internet', 'definition', 'technology']),
    ("What is a database?", "A database is an organized collection of structured information or data, typically stored electronically in a computer system.", ['database', 'definition', 'technology']),
    ("What is cloud computing?", "Cloud computing is the delivery of computing services including servers, storage, databases, networking, software, and analytics over the internet.", ['cloud', 'definition', 'technology']),
    
    # === HOW QUESTIONS - TECHNOLOGY ===
    ("How does AI work?", "AI works by processing large amounts of data, identifying patterns, and making decisions based on those patterns. It uses algorithms and neural networks to learn from experience.", ['ai', 'how', 'technology']),
    ("How does machine learning work?", "Machine learning works by training algorithms on data, allowing them to identify patterns and make predictions without being explicitly programmed for each task.", ['ml', 'how', 'technology']),
    ("How does the internet work?", "The internet works by connecting millions of computers worldwide through a network of cables and wireless signals, using standardized protocols to transmit data.", ['internet', 'how', 'technology']),
    ("How does a computer work?", "A computer works by processing binary data (0s and 1s) through its central processing unit (CPU), which executes instructions stored in memory.", ['computer', 'how', 'technology']),
    
    # === SCIENCE & NATURE ===
    ("What is water?", "Water is a transparent, tasteless, odorless chemical substance composed of hydrogen and oxygen (H2O). It is essential for all known forms of life.", ['water', 'definition', 'science']),
    ("What is gravity?", "Gravity is a natural phenomenon by which all things with mass or energy are brought toward one another. On Earth, it gives weight to physical objects.", ['gravity', 'definition', 'science']),
    ("What is energy?", "Energy is the capacity to do work. It exists in various forms including kinetic, potential, thermal, electrical, chemical, and nuclear energy.", ['energy', 'definition', 'science']),
    ("What is light?", "Light is electromagnetic radiation that is visible to the human eye. It travels in waves and behaves both as a wave and as a particle (photon).", ['light', 'definition', 'science']),
    ("What is electricity?", "Electricity is the flow of electrical charge. It is a form of energy resulting from the movement of electrons through a conductor.", ['electricity', 'definition', 'science']),
    ("What is DNA?", "DNA (Deoxyribonucleic acid) is a molecule that carries the genetic instructions for the development, functioning, growth, and reproduction of all known organisms.", ['dna', 'definition', 'biology']),
    ("What is evolution?", "Evolution is the process by which different kinds of living organisms develop and diversify from earlier forms during the history of the earth.", ['evolution', 'definition', 'biology']),
    ("What is photosynthesis?", "Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water, producing oxygen as a byproduct.", ['photosynthesis', 'definition', 'biology']),
    
    # === MARITIME & TRANSPORTATION ===
    ("What is a ship?", "A ship is a large watercraft that travels the world's oceans and other sufficiently deep waterways, carrying goods or passengers.", ['ship', 'definition', 'maritime']),
    ("What is a boat?", "A boat is a watercraft of various types and sizes, generally smaller than a ship, designed for travel on water.", ['boat', 'definition', 'maritime']),
    ("How does a ship work?", "A ship works by displacing water equal to its weight, creating buoyancy that keeps it afloat. Propulsion systems like engines or sails move it through the water.", ['ship', 'how', 'maritime']),
    ("What is an airplane?", "An airplane is a powered flying vehicle with fixed wings that is heavier than air and capable of sustained flight.", ['airplane', 'definition', 'transportation']),
    ("What is a car?", "A car is a wheeled motor vehicle used for transportation. It typically has four wheels and can carry a small number of passengers.", ['car', 'definition', 'transportation']),
    
    # === CHEMISTRY & MATERIALS ===
    ("What is gold?", "Gold is a chemical element with the symbol Au and atomic number 79. It is a bright, slightly reddish yellow, dense, soft, malleable, and ductile metal.", ['gold', 'definition', 'chemistry']),
    ("What is silver?", "Silver is a chemical element with the symbol Ag and atomic number 47. It is a soft, white, lustrous transition metal with the highest electrical and thermal conductivity of any metal.", ['silver', 'definition', 'chemistry']),
    ("What is iron?", "Iron is a chemical element with the symbol Fe and atomic number 26. It is a metal that is strong, hard, and widely used in construction and manufacturing.", ['iron', 'definition', 'chemistry']),
    ("What is oxygen?", "Oxygen is a chemical element with the symbol O and atomic number 8. It is a colorless, odorless gas essential for most life on Earth.", ['oxygen', 'definition', 'chemistry']),
    ("What is carbon?", "Carbon is a chemical element with the symbol C and atomic number 6. It is the basis of all known life and forms more compounds than any other element.", ['carbon', 'definition', 'chemistry']),
    
    # === WHY QUESTIONS ===
    ("Why is gold valuable?", "Gold is valuable because it is rare, doesn't corrode, is easily worked, and has been used as currency and jewelry for thousands of years.", ['gold', 'why', 'economics']),
    ("Why is AI important?", "AI is important because it can process vast amounts of data quickly, automate complex tasks, and solve problems that are difficult for humans to handle alone.", ['ai', 'why', 'technology']),
    ("Why is water important?", "Water is important because it is essential for all known forms of life. It regulates temperature, transports nutrients, and is involved in nearly every bodily function.", ['water', 'why', 'science']),
    ("Why do we need oxygen?", "We need oxygen because it is essential for cellular respiration, the process by which our cells produce energy from food.", ['oxygen', 'why', 'biology']),
    ("Why is the internet important?", "The internet is important because it enables global communication, access to information, online commerce, and has transformed how we work, learn, and interact.", ['internet', 'why', 'technology']),
    
    # === MATHEMATICS ===
    ("What is mathematics?", "Mathematics is the study of numbers, quantities, shapes, and patterns. It is used to describe and analyze the world around us.", ['math', 'definition', 'science']),
    ("What is algebra?", "Algebra is a branch of mathematics that uses symbols and letters to represent numbers and quantities in formulas and equations.", ['algebra', 'definition', 'math']),
    ("What is geometry?", "Geometry is a branch of mathematics concerned with the properties and relations of points, lines, surfaces, and solids.", ['geometry', 'definition', 'math']),
    ("What is calculus?", "Calculus is a branch of mathematics that studies continuous change. It includes differential calculus and integral calculus.", ['calculus', 'definition', 'math']),
    
    # === HISTORY & CULTURE ===
    ("What is history?", "History is the study of past events, particularly in human affairs. It helps us understand how societies, cultures, and civilizations have developed over time.", ['history', 'definition', 'humanities']),
    ("What is culture?", "Culture is the social behavior, norms, knowledge, beliefs, arts, laws, customs, and habits of a particular group of people.", ['culture', 'definition', 'humanities']),
    ("What is language?", "Language is a system of communication used by humans, consisting of structured sounds, words, and grammar to convey meaning.", ['language', 'definition', 'humanities']),
    ("What is art?", "Art is the expression or application of human creative skill and imagination, typically in visual form such as painting or sculpture.", ['art', 'definition', 'humanities']),
    
    # === GEOGRAPHY ===
    ("What is a mountain?", "A mountain is a large natural elevation of the earth's surface rising abruptly from the surrounding level.", ['mountain', 'definition', 'geography']),
    ("What is an ocean?", "An ocean is a very large expanse of sea, particularly each of the main areas into which the sea is divided geographically.", ['ocean', 'definition', 'geography']),
    ("What is a river?", "A river is a large natural stream of water flowing in a channel to the sea, a lake, or another river.", ['river', 'definition', 'geography']),
    ("What is a desert?", "A desert is a barren area of landscape with little precipitation and consequently living conditions are hostile for plant and animal life.", ['desert', 'definition', 'geography']),
    
    # === COMPARISONS ===
    ("What is the difference between AI and machine learning?", "AI is the broader concept of machines being able to carry out tasks in a smart way, while machine learning is a specific subset of AI that trains machines to learn from data.", ['ai', 'ml', 'comparison']),
    ("What is the difference between a ship and a boat?", "A ship is generally larger than a boat and designed for ocean travel, while a boat is smaller and typically used in rivers, lakes, or coastal waters.", ['ship', 'boat', 'comparison']),
    ("What is the difference between weather and climate?", "Weather refers to short-term atmospheric conditions in a specific place, while climate refers to long-term patterns of temperature and precipitation in a region.", ['weather', 'climate', 'comparison']),
    
    # === CONVERSATIONAL ===
    ("Hello", "Hello! I'm Friday, your AI assistant. How can I help you today?", ['greeting', 'conversation']),
    ("Hi", "Hi there! I'm here to help. What would you like to know?", ['greeting', 'conversation']),
    ("How are you?", "I'm functioning well, thank you! I'm ready to help you with any questions you have.", ['greeting', 'conversation']),
    ("Good morning", "Good morning! How can I assist you today?", ['greeting', 'conversation']),
    ("Good evening", "Good evening! What can I help you with?", ['greeting', 'conversation']),
    ("Thank you", "You're welcome! Feel free to ask if you need anything else.", ['gratitude', 'conversation']),
    ("Thanks", "You're welcome! Happy to help.", ['gratitude', 'conversation']),
    ("Goodbye", "Goodbye! Feel free to come back if you have more questions.", ['farewell', 'conversation']),
    ("Bye", "Bye! Have a great day!", ['farewell', 'conversation']),
    ("Can you help me?", "Yes, I'm here to help! I can answer questions, provide information, and assist you with various topics. What would you like to know?", ['help', 'conversation']),
    ("What can you do?", "I can answer questions, provide information on various topics like science, technology, history, and more. I can explain concepts, compare things, and have conversations. What would you like to know about?", ['capabilities', 'conversation']),
    ("Who are you?", "I'm Friday, an AI assistant built on a unique 3D neuron architecture. I'm here to help answer your questions and provide information.", ['identity', 'conversation']),
    ("What are you?", "I'm an AI assistant called Friday. I use a 3D neuron system to store and retrieve knowledge, allowing me to answer questions and have conversations.", ['identity', 'conversation']),
    ("Tell me about yourself", "I'm Friday, an AI assistant with a unique architecture based on 3D neurons. I can learn from data, answer questions, and help with various topics. What would you like to know?", ['identity', 'conversation']),
    
    # === PRACTICAL QUESTIONS ===
    ("How can I learn programming?", "You can learn programming by starting with beginner-friendly languages like Python, using online tutorials and courses, practicing regularly, and building small projects.", ['programming', 'how', 'advice']),
    ("What is the best programming language?", "There's no single 'best' programming language. Python is great for beginners and data science, JavaScript for web development, Java for enterprise applications, and C++ for performance-critical systems.", ['programming', 'advice', 'technology']),
    ("How do I stay healthy?", "To stay healthy, maintain a balanced diet, exercise regularly, get adequate sleep, stay hydrated, manage stress, and have regular medical checkups.", ['health', 'advice', 'lifestyle']),
    ("What is a healthy diet?", "A healthy diet includes a variety of fruits, vegetables, whole grains, lean proteins, and healthy fats while limiting processed foods, sugar, and excessive salt.", ['health', 'diet', 'lifestyle']),
    
    # === SPACE & ASTRONOMY ===
    ("What is the sun?", "The sun is the star at the center of our solar system. It is a nearly perfect sphere of hot plasma that provides light and heat to Earth.", ['sun', 'definition', 'astronomy']),
    ("What is a planet?", "A planet is a celestial body that orbits a star, is massive enough to be rounded by its own gravity, and has cleared its orbital path of debris.", ['planet', 'definition', 'astronomy']),
    ("What is the moon?", "The moon is Earth's only natural satellite. It orbits Earth and reflects sunlight, appearing to change shape in a monthly cycle.", ['moon', 'definition', 'astronomy']),
    ("What is a star?", "A star is a luminous sphere of plasma held together by its own gravity. Stars produce energy through nuclear fusion in their cores.", ['star', 'definition', 'astronomy']),
    
    # === PHYSICS ===
    ("What is force?", "Force is a push or pull upon an object resulting from its interaction with another object. It can cause an object to accelerate, decelerate, or change direction.", ['force', 'definition', 'physics']),
    ("What is velocity?", "Velocity is the rate of change of an object's position with respect to time. It is a vector quantity that includes both speed and direction.", ['velocity', 'definition', 'physics']),
    ("What is acceleration?", "Acceleration is the rate of change of velocity with respect to time. It occurs when an object speeds up, slows down, or changes direction.", ['acceleration', 'definition', 'physics']),
    ("What is momentum?", "Momentum is the quantity of motion of a moving body, measured as a product of its mass and velocity.", ['momentum', 'definition', 'physics']),
    
    # === ECONOMICS & BUSINESS ===
    ("What is economics?", "Economics is the social science that studies how people interact with value, particularly the production, distribution, and consumption of goods and services.", ['economics', 'definition', 'social-science']),
    ("What is money?", "Money is a medium of exchange that facilitates transactions for goods and services. It serves as a unit of account and a store of value.", ['money', 'definition', 'economics']),
    ("What is inflation?", "Inflation is the rate at which the general level of prices for goods and services rises, causing purchasing power to fall.", ['inflation', 'definition', 'economics']),
    ("What is a market?", "A market is a place or system where buyers and sellers interact to exchange goods, services, or information.", ['market', 'definition', 'economics']),
    
    # === PSYCHOLOGY ===
    ("What is psychology?", "Psychology is the scientific study of the mind and behavior. It explores how people think, feel, and act.", ['psychology', 'definition', 'social-science']),
    ("What is memory?", "Memory is the faculty of the mind by which information is encoded, stored, and retrieved. It allows us to recall past experiences and learned information.", ['memory', 'definition', 'psychology']),
    ("What is learning?", "Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, or preferences through experience, study, or teaching.", ['learning', 'definition', 'psychology']),
    
    # === ENVIRONMENT ===
    ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities that increase greenhouse gas emissions.", ['climate-change', 'definition', 'environment']),
    ("What is pollution?", "Pollution is the introduction of harmful substances or contaminants into the natural environment, causing adverse changes.", ['pollution', 'definition', 'environment']),
    ("What is renewable energy?", "Renewable energy is energy from sources that are naturally replenishing, such as solar, wind, hydro, and geothermal power.", ['renewable-energy', 'definition', 'environment']),
]

def get_qa_data():
    """Returns the Q&A data."""
    return QA_DATA

def get_qa_count():
    """Returns the number of Q&A pairs."""
    return len(QA_DATA)
