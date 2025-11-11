"""
Generative Language Model - Lernt aus Neuronen und generiert neue Antworten.

Statt nur gespeicherte Texte zurückzugeben, lernt dieses Modell aus den
Neuron-Mustern und generiert komplett neue, kontextbezogene Antworten.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class GenerativeLanguageModel:
    """
    Generatives Sprachmodell das aus Neuronen lernt.
    
    Statt Retrieval nutzt es die Neuronen-Aktivierungen um:
    1. Muster zu erkennen
    2. Kontext zu verstehen
    3. Neue Antworten zu generieren
    """
    
    def __init__(self, language_model):
        """
        Initialize generative model.
        
        Args:
            language_model: Base language model with graph access
        """
        self.language_model = language_model
        self.graph = language_model.graph
        self.compression_engine = language_model.compression_engine
        self.query_engine = language_model.query_engine
        
        # Lern-Statistiken
        self.word_patterns = defaultdict(lambda: defaultdict(int))
        self.concept_patterns = defaultdict(list)
        self.response_templates = []
        
        logger.info("Generative Language Model initialized")
    
    def learn_from_neurons(self, max_neurons: int = None):
        """
        Lernt Muster aus allen Neuronen.
        
        Args:
            max_neurons: Maximum number of neurons to learn from
        """
        logger.info("Learning patterns from neurons...")
        
        neurons = list(self.graph.neurons.values())
        if max_neurons:
            neurons = neurons[:max_neurons]
        
        for neuron in neurons:
            if not hasattr(neuron, 'source_data') or not neuron.source_data:
                continue
            
            text = neuron.source_data
            tags = getattr(neuron, 'semantic_tags', [])
            
            # Lerne Wort-Muster
            self._learn_word_patterns(text, tags)
            
            # Lerne Konzept-Muster
            self._learn_concept_patterns(text, tags)
        
        logger.info(f"Learned patterns from {len(neurons)} neurons")
        logger.info(f"Word patterns: {len(self.word_patterns)}")
        logger.info(f"Concept patterns: {len(self.concept_patterns)}")
    
    def _learn_word_patterns(self, text: str, tags: List[str]):
        """Lernt Wort-Folge-Muster."""
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Lerne Bi-Gramme (Wort-Paare)
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i + 1]
            self.word_patterns[current][next_word] += 1
        
        # Lerne Tag-assoziierte Wörter
        for tag in tags:
            for word in words[:10]:  # Erste 10 Wörter sind oft wichtig
                self.word_patterns[f"tag:{tag}"][word] += 1
    
    def _learn_concept_patterns(self, text: str, tags: List[str]):
        """Lernt Konzept-Muster."""
        # Extrahiere Sätze
        sentences = re.split(r'[.!?]+', text)
        
        for tag in tags:
            for sentence in sentences[:3]:  # Erste 3 Sätze
                sentence = sentence.strip()
                if len(sentence) > 20:
                    self.concept_patterns[tag].append(sentence)
    
    def generate_response(
        self,
        query: str,
        context_size: int = 5,
        creativity: float = 0.7
    ) -> str:
        """
        Generiert eine neue Antwort basierend auf gelernten Mustern.
        
        Args:
            query: Die Frage
            context_size: Anzahl relevanter Neuronen
            creativity: 0.0 = konservativ, 1.0 = kreativ
            
        Returns:
            Generierte Antwort
        """
        logger.info(f"Generating response for: {query[:50]}...")
        
        # 1. Finde relevante Neuronen
        results = self.query_engine.query(
            query_text=query,
            top_k=context_size,
            propagation_depth=0
        )
        
        if not results:
            return self._generate_fallback_response(query)
        
        # 2. Extrahiere Kontext aus Neuronen
        context = self._extract_context(results)
        
        # 3. Analysiere Frage
        question_type = self._analyze_question(query)
        
        # 4. Generiere Antwort basierend auf Kontext und Mustern
        response = self._generate_from_context(
            query, question_type, context, creativity
        )
        
        return response
    
    def _extract_context(self, results: List) -> Dict[str, Any]:
        """Extrahiert Kontext aus aktivierten Neuronen."""
        context = {
            'texts': [],
            'tags': set(),
            'key_words': defaultdict(int),
            'concepts': []
        }
        
        for result in results:
            neuron = result.neuron
            
            if hasattr(neuron, 'source_data') and neuron.source_data:
                context['texts'].append(neuron.source_data)
                
                # Extrahiere wichtige Wörter
                words = re.findall(r'\b\w+\b', neuron.source_data.lower())
                for word in words:
                    if len(word) > 3:  # Nur längere Wörter
                        context['key_words'][word] += result.activation
            
            if hasattr(neuron, 'semantic_tags'):
                context['tags'].update(neuron.semantic_tags or [])
        
        # Sortiere Wörter nach Wichtigkeit
        context['key_words'] = dict(
            sorted(context['key_words'].items(), 
                   key=lambda x: x[1], reverse=True)[:20]
        )
        
        return context
    
    def _analyze_question(self, query: str) -> str:
        """Analysiert den Fragetyp."""
        query_lower = query.lower().strip()
        
        # Questions first (before greetings, da "hello what is" sonst als greeting erkannt wird)
        if 'what is' in query_lower or 'was ist' in query_lower or 'what are' in query_lower:
            return 'definition'
        elif 'how does' in query_lower or 'how do' in query_lower or 'wie funktioniert' in query_lower:
            return 'how'
        elif 'why' in query_lower or 'warum' in query_lower:
            return 'why'
        elif 'who is' in query_lower or 'who are' in query_lower or 'wer ist' in query_lower:
            return 'who'
        elif 'when' in query_lower or 'wann' in query_lower:
            return 'when'
        elif 'where' in query_lower or 'wo' in query_lower:
            return 'where'
        
        # Greetings (nur wenn KEINE Frage)
        if any(word == query_lower or query_lower.startswith(word + ' ') or query_lower.startswith(word + '!') 
               for word in ['hello', 'hi', 'hey', 'hallo', 'guten tag']):
            return 'greeting'
        
        # General questions
        if query_lower.endswith('?') or any(word in query_lower for word in ['what', 'how', 'why', 'who', 'when', 'where']):
            return 'question'
        
        return 'statement'
    
    def _generate_from_context(
        self,
        query: str,
        question_type: str,
        context: Dict[str, Any],
        creativity: float
    ) -> str:
        """Generiert Antwort aus Kontext."""
        
        # Spezielle Behandlung für Greetings
        if question_type == 'greeting':
            return self._generate_greeting_response()
        
        # Für Fragen: Generiere basierend auf Kontext
        if question_type in ['definition', 'how', 'why', 'who', 'when', 'where', 'question']:
            return self._generate_knowledge_response(
                query, question_type, context, creativity
            )
        
        # Fallback
        return self._generate_conversational_response(context)
    
    def _generate_greeting_response(self) -> str:
        """Generiert eine Begrüßung."""
        greetings = [
            "Hello! I'm Friday, your AI assistant. How can I help you today?",
            "Hi there! I'm here to help. What would you like to know?",
            "Hello! I'm Friday. What can I do for you?",
            "Hey! I'm ready to assist you. What's on your mind?",
        ]
        return np.random.choice(greetings)
    
    def _generate_knowledge_response(
        self,
        query: str,
        question_type: str,
        context: Dict[str, Any],
        creativity: float
    ) -> str:
        """Generiert Wissens-Antwort."""
        
        if not context['key_words']:
            return "I don't have enough information to answer that question."
        
        # Hole die wichtigsten Wörter
        top_words = list(context['key_words'].keys())[:5]
        
        # Generiere Antwort basierend auf Wort-Mustern
        if creativity > 0.5:
            # Kreative Generierung
            response = self._generate_creative_response(
                question_type, top_words, context
            )
        else:
            # Konservative Generierung (näher an Originaltexten)
            response = self._generate_conservative_response(
                question_type, context
            )
        
        return response
    
    def _generate_creative_response(
        self,
        question_type: str,
        key_words: List[str],
        context: Dict[str, Any]
    ) -> str:
        """Generiert kreative Antwort aus Mustern."""
        
        # Starte mit einem passenden Anfang
        starters = {
            'definition': ['This refers to', 'It is', 'This is', 'It describes'],
            'how': ['It works by', 'The process involves', 'This happens through'],
            'why': ['This is because', 'The reason is', 'This occurs due to'],
            'who': ['This refers to', 'This person', 'This entity'],
            'when': ['This occurred', 'This happened', 'The time was'],
            'where': ['This is located', 'This takes place', 'This is found'],
        }
        
        starter = np.random.choice(starters.get(question_type, ['This is about']))
        
        # Generiere Satz aus Wort-Mustern
        generated_words = [starter]
        current_word = key_words[0] if key_words else 'information'
        
        for _ in range(10):  # Max 10 Wörter generieren
            if current_word in self.word_patterns:
                # Wähle nächstes Wort basierend auf Wahrscheinlichkeiten
                next_words = self.word_patterns[current_word]
                if next_words:
                    # Gewichtete Auswahl
                    words = list(next_words.keys())
                    weights = list(next_words.values())
                    total = sum(weights)
                    probs = [w/total for w in weights]
                    
                    current_word = np.random.choice(words, p=probs)
                    generated_words.append(current_word)
                else:
                    break
            else:
                # Wähle aus key_words
                if key_words:
                    current_word = np.random.choice(key_words)
                    generated_words.append(current_word)
                else:
                    break
        
        response = ' '.join(generated_words)
        
        # Cleanup
        response = response.capitalize()
        if not response.endswith('.'):
            response += '.'
        
        return response
    
    def _generate_conservative_response(
        self,
        question_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Generiert konservative Antwort (näher an Originaltexten)."""
        
        if not context['texts']:
            return "I don't have enough information about that."
        
        # Finde relevantesten Text
        best_text = context['texts'][0]
        
        # Extrahiere relevanten Teil
        sentences = re.split(r'[.!?]+', best_text)
        
        # Wähle beste Sätze basierend auf key_words
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Score basierend auf key_words
            score = 0
            for word in context['key_words']:
                if word in sentence.lower():
                    score += context['key_words'][word]
            
            scored_sentences.append((score, sentence))
        
        if not scored_sentences:
            return best_text[:200] + '...'
        
        # Sortiere und nimm beste
        scored_sentences.sort(reverse=True)
        best_sentence = scored_sentences[0][1]
        
        # Füge Kontext hinzu wenn nötig
        if question_type == 'definition':
            if not best_sentence.lower().startswith(('it is', 'this is', 'it refers')):
                best_sentence = f"It is {best_sentence.lower()}"
        
        return best_sentence
    
    def _generate_conversational_response(
        self,
        context: Dict[str, Any]
    ) -> str:
        """Generiert konversationelle Antwort."""
        
        if context['tags']:
            tags_str = ', '.join(list(context['tags'])[:3])
            return f"I can tell you about {tags_str}. What would you like to know?"
        
        return "I'm here to help! What would you like to know?"
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generiert Fallback-Antwort wenn kein Kontext."""
        
        question_type = self._analyze_question(query)
        
        if question_type == 'greeting':
            return self._generate_greeting_response()
        
        return "I don't have enough information to answer that question. Could you rephrase it or ask something else?"
