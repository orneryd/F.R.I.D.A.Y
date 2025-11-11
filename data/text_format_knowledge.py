"""
Text Format Knowledge Base - Wie Text strukturiert und formatiert ist.

Lehrt Friday über:
- Satzstruktur und Grammatik
- Absätze und Formatierung
- Interpunktion und Stil
- Verschiedene Texttypen
"""

TEXT_FORMAT_DATA = [
    # === SENTENCE STRUCTURE ===
    (
        "What is a sentence?",
        "A sentence is a grammatical unit that expresses a complete thought. It typically contains a subject and a predicate, and ends with punctuation like a period, question mark, or exclamation point.",
        ['grammar', 'text-format', 'sentence', 'definition']
    ),
    (
        "What are the parts of a sentence?",
        "The main parts of a sentence are the subject (who or what the sentence is about) and the predicate (what the subject does or is). Additional parts include objects, complements, and modifiers.",
        ['grammar', 'text-format', 'sentence', 'structure']
    ),
    (
        "What is a paragraph?",
        "A paragraph is a group of related sentences that develop a single main idea. It typically starts with a topic sentence, followed by supporting sentences, and may end with a concluding sentence.",
        ['text-format', 'paragraph', 'structure', 'definition']
    ),
    
    # === PUNCTUATION ===
    (
        "What is punctuation?",
        "Punctuation consists of marks used in writing to separate sentences and clarify meaning. Common punctuation marks include periods, commas, question marks, exclamation points, colons, and semicolons.",
        ['grammar', 'text-format', 'punctuation', 'definition']
    ),
    (
        "When do you use a period?",
        "A period is used at the end of a declarative sentence (a statement) or an imperative sentence (a command). It signals the end of a complete thought.",
        ['grammar', 'text-format', 'punctuation', 'period']
    ),
    (
        "When do you use a comma?",
        "A comma is used to separate items in a list, separate independent clauses joined by conjunctions, set off introductory elements, and separate non-essential information from the main clause.",
        ['grammar', 'text-format', 'punctuation', 'comma']
    ),
    (
        "When do you use a question mark?",
        "A question mark is used at the end of an interrogative sentence (a direct question). It indicates that the sentence is asking for information.",
        ['grammar', 'text-format', 'punctuation', 'question-mark']
    ),
    
    # === TEXT TYPES ===
    (
        "What is narrative text?",
        "Narrative text tells a story or describes a sequence of events. It typically includes characters, setting, plot, conflict, and resolution. Narratives can be fiction or non-fiction.",
        ['text-format', 'text-type', 'narrative', 'definition']
    ),
    (
        "What is expository text?",
        "Expository text explains or informs about a topic using facts, definitions, and examples. It aims to educate the reader and is typically organized logically with clear structure.",
        ['text-format', 'text-type', 'expository', 'definition']
    ),
    (
        "What is persuasive text?",
        "Persuasive text aims to convince the reader to accept a particular viewpoint or take a specific action. It uses arguments, evidence, and rhetorical techniques to influence opinion.",
        ['text-format', 'text-type', 'persuasive', 'definition']
    ),
    (
        "What is descriptive text?",
        "Descriptive text uses sensory details and vivid language to create a mental image for the reader. It focuses on describing people, places, objects, or experiences.",
        ['text-format', 'text-type', 'descriptive', 'definition']
    ),
    
    # === FORMATTING ===
    (
        "What is text formatting?",
        "Text formatting refers to the visual presentation of text, including font style, size, color, alignment, spacing, and emphasis (bold, italic, underline). It helps organize information and guide the reader's attention.",
        ['text-format', 'formatting', 'visual', 'definition']
    ),
    (
        "What is bold text used for?",
        "Bold text is used to emphasize important words or phrases, highlight headings and titles, or draw attention to key information. It makes text stand out visually.",
        ['text-format', 'formatting', 'bold', 'emphasis']
    ),
    (
        "What is italic text used for?",
        "Italic text is used for emphasis, titles of books and publications, foreign words, technical terms, or to indicate thoughts in narrative writing.",
        ['text-format', 'formatting', 'italic', 'emphasis']
    ),
    (
        "What are headings?",
        "Headings are titles or labels that organize content into sections. They create a hierarchy of information, making text easier to scan and navigate. Headings are typically larger or bolder than body text.",
        ['text-format', 'formatting', 'headings', 'structure']
    ),
    
    # === WRITING STYLE ===
    (
        "What is writing style?",
        "Writing style is the way an author expresses ideas through word choice, sentence structure, tone, and literary devices. It reflects the author's personality and purpose, and can be formal, informal, academic, creative, or technical.",
        ['text-format', 'style', 'writing', 'definition']
    ),
    (
        "What is formal writing?",
        "Formal writing uses standard grammar, complete sentences, and professional language. It avoids contractions, slang, and personal pronouns. It's used in academic papers, business documents, and official communications.",
        ['text-format', 'style', 'formal', 'definition']
    ),
    (
        "What is informal writing?",
        "Informal writing is conversational and relaxed. It may use contractions, colloquialisms, and personal pronouns. It's used in emails to friends, personal blogs, and casual communications.",
        ['text-format', 'style', 'informal', 'definition']
    ),
    (
        "What is tone in writing?",
        "Tone is the attitude or emotion conveyed by the writer toward the subject or audience. It can be serious, humorous, sarcastic, optimistic, pessimistic, formal, or informal, among others.",
        ['text-format', 'style', 'tone', 'definition']
    ),
    
    # === GRAMMAR BASICS ===
    (
        "What is a noun?",
        "A noun is a word that names a person, place, thing, or idea. Nouns can be concrete (physical objects) or abstract (concepts), and can be singular or plural.",
        ['grammar', 'text-format', 'noun', 'definition']
    ),
    (
        "What is a verb?",
        "A verb is a word that expresses an action, occurrence, or state of being. Verbs are essential to sentences as they indicate what the subject does or is.",
        ['grammar', 'text-format', 'verb', 'definition']
    ),
    (
        "What is an adjective?",
        "An adjective is a word that describes or modifies a noun or pronoun. It provides information about qualities, quantities, or characteristics.",
        ['grammar', 'text-format', 'adjective', 'definition']
    ),
    (
        "What is an adverb?",
        "An adverb is a word that modifies a verb, adjective, or another adverb. It typically describes how, when, where, or to what extent something happens.",
        ['grammar', 'text-format', 'adverb', 'definition']
    ),
    
    # === DOCUMENT STRUCTURE ===
    (
        "What is an introduction?",
        "An introduction is the opening section of a text that presents the topic, provides context, and often includes a thesis statement or main idea. It engages the reader and sets the direction for the content.",
        ['text-format', 'structure', 'introduction', 'definition']
    ),
    (
        "What is a conclusion?",
        "A conclusion is the closing section of a text that summarizes main points, reinforces the thesis, and provides closure. It may also suggest implications or future directions.",
        ['text-format', 'structure', 'conclusion', 'definition']
    ),
    (
        "What is a thesis statement?",
        "A thesis statement is a sentence that expresses the main idea or argument of an essay or paper. It typically appears in the introduction and guides the entire piece of writing.",
        ['text-format', 'structure', 'thesis', 'definition']
    ),
    
    # === LISTS AND ORGANIZATION ===
    (
        "What is a bulleted list?",
        "A bulleted list is a series of items presented with bullet points (•) rather than numbers. It's used for items that don't need to be in a specific order.",
        ['text-format', 'formatting', 'list', 'bullets']
    ),
    (
        "What is a numbered list?",
        "A numbered list is a series of items presented with numbers (1, 2, 3...). It's used for sequential steps, ranked items, or when order matters.",
        ['text-format', 'formatting', 'list', 'numbers']
    ),
    (
        "What is white space?",
        "White space (or negative space) is the empty area around text and other elements. It improves readability, creates visual hierarchy, and makes content less overwhelming.",
        ['text-format', 'formatting', 'whitespace', 'design']
    ),
    
    # === QUOTATIONS AND CITATIONS ===
    (
        "What are quotation marks used for?",
        "Quotation marks are used to indicate direct speech, quotations from sources, titles of short works, or to highlight specific words or phrases. They show that the enclosed text is someone else's exact words.",
        ['grammar', 'text-format', 'punctuation', 'quotation']
    ),
    (
        "What is a citation?",
        "A citation is a reference to a source of information used in writing. It gives credit to the original author and allows readers to locate the source. Citations follow specific formats like APA, MLA, or Chicago style.",
        ['text-format', 'academic', 'citation', 'definition']
    ),
    
    # === CLARITY AND CONCISENESS ===
    (
        "What is clear writing?",
        "Clear writing communicates ideas effectively using simple, direct language. It avoids ambiguity, uses concrete examples, and organizes information logically so readers can easily understand the message.",
        ['text-format', 'style', 'clarity', 'definition']
    ),
    (
        "What is concise writing?",
        "Concise writing expresses ideas using the fewest words necessary without sacrificing clarity or completeness. It eliminates redundancy, wordiness, and unnecessary details.",
        ['text-format', 'style', 'concise', 'definition']
    ),
    (
        "What is active voice?",
        "Active voice is a sentence structure where the subject performs the action. For example: 'The cat chased the mouse.' It's generally more direct and engaging than passive voice.",
        ['grammar', 'text-format', 'voice', 'active']
    ),
    (
        "What is passive voice?",
        "Passive voice is a sentence structure where the subject receives the action. For example: 'The mouse was chased by the cat.' It's used when the actor is unknown or less important than the action.",
        ['grammar', 'text-format', 'voice', 'passive']
    ),
]


def get_text_format_data():
    """Returns the text format data."""
    return TEXT_FORMAT_DATA


def get_text_format_count():
    """Returns the number of text format pairs."""
    return len(TEXT_FORMAT_DATA)
