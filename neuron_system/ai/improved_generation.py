"""
Improved Generation - Bessere Antwort-Generierung mit mehr Kontext.

Statt zufälliger Wort-Muster nutzt dieses Modell:
1. Sentence-Level Patterns (ganze Sätze)
2. Template-basierte Generierung
3. Kontext-bewusste Synthese
"""

import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class ImprovedGenerator:
    """
    Verbesserte Antwort-Generierung mit mehr Kontext-Verständnis.
    """
    
    def __init__(self, language_model):
        """Initialize improved generator."""
        self.language_model = language_model
        self.graph = language_model.graph
        self.compression_engine = language_model.compression_engine
        self.query_engine = language_model.query_engine
        
        # Sentence patterns (ganze Sätze statt nur Wörter)
        self.sentence_patterns = defaultdict(list)
        self.topic_sentences = defaultdict(list)
        
        logger.info("Improved Generator initialized")
    
    def learn_from_neurons(self, max_neurons: int = None):
        """
        Lernt Satz-Muster aus Neuronen.
        
        Args:
            max_neurons: Maximum number of neurons to learn from
        """
        logger.info("Learning sentence patterns from neurons...")
        
        neurons = list(self.graph.neurons.values())
        if max_neurons:
            neurons = neurons[:max_neurons]
        
        for neuron in neurons:
            if not hasattr(neuron, 'source_data') or not neuron.source_data:
                continue
            
            text = neuron.source_data
            tags = getattr(neuron, 'semantic_tags', [])
            
            # Extrahiere Sätze
            # Spezial-Behandlung für Q&A Format
            if "Question:" in text and "Answer:" in text:
                # Extrahiere nur die Antwort
                answer_start = text.find("Answer:") + 7
                answer_text = text[answer_start:].strip()
                sentences = self._extract_sentences(answer_text)
            else:
                sentences = self._extract_sentences(text)
            
            # Speichere nach Tags
            for tag in tags:
                for sentence in sentences:
                    if len(sentence) > 30:  # Nur sinnvolle Sätze
                        self.topic_sentences[tag].append(sentence)
            
            # Speichere nach Schlüsselwörtern
            for sentence in sentences:
                keywords = self._extract_keywords(sentence)
                for keyword in keywords:
                    self.sentence_patterns[keyword].append(sentence)
        
        logger.info(f"Learned {len(self.sentence_patterns)} keyword patterns")
        logger.info(f"Learned {len(self.topic_sentences)} topic patterns")
    
    def generate_response(
        self,
        query: str,
        context_size: int = 5
    ) -> str:
        """
        Generiert verbesserte Antwort.
        
        Args:
            query: Die Frage
            context_size: Anzahl relevanter Neuronen
            
        Returns:
            Generierte Antwort
        """
        logger.info(f"Generating improved response for: {query[:50]}...")
        
        # 1. Finde relevante Neuronen
        results = self.query_engine.query(
            query_text=query,
            top_k=context_size,
            propagation_depth=0
        )
        
        if not results:
            return self._generate_fallback(query)
        
        # 2. Analysiere Frage
        question_type = self._analyze_question(query)
        
        # 3. Extrahiere besten Kontext
        best_context = self._extract_best_context(results, query)
        
        # 4. Generiere basierend auf Fragetyp
        response = self._generate_by_type(
            query, question_type, best_context
        )
        
        return response
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extrahiert Sätze aus Text."""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            # Filter out too short, headers, or meta-text
            if (len(sent) > 20 and 
                not sent.startswith('=') and
                not sent.startswith('@') and
                any(c.isalpha() for c in sent)):
                cleaned.append(sent)
        
        return cleaned
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrahiert Schlüsselwörter."""
        # Tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stopwords and short words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                     'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
                     'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
                     'does', 'did', 'will', 'would', 'could', 'should', 'may',
                     'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        
        # Return top keywords (by frequency)
        from collections import Counter
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(10)]
    
    def _analyze_question(self, query: str) -> str:
        """Analysiert Fragetyp."""
        query_lower = query.lower().strip()
        
        # Greetings
        greetings = ['hello', 'hi', 'hey', 'hallo', 'guten tag', 'good morning', 'good evening']
        if any(query_lower == g or query_lower.startswith(g + ' ') or query_lower.startswith(g + '!') 
               for g in greetings):
            return 'greeting'
        
        # Questions
        if 'what is' in query_lower or 'was ist' in query_lower:
            return 'definition'
        elif 'what are' in query_lower or 'was sind' in query_lower:
            return 'definition_plural'
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
        elif query_lower.endswith('?'):
            return 'question'
        
        return 'statement'
    
    def _extract_best_context(self, results: List, query: str) -> Dict[str, Any]:
        """Extrahiert besten Kontext aus Neuronen."""
        context = {
            'sentences': [],
            'tags': set(),
            'keywords': set(),
            'best_text': None,
            'best_activation': 0.0
        }
        
        for result in results[:5]:  # Top 5
            neuron = result.neuron
            activation = result.activation
            
            if hasattr(neuron, 'source_data') and neuron.source_data:
                text = neuron.source_data
                
                # Extrahiere Sätze
                sentences = self._extract_sentences(text)
                
                # Score sentences by relevance to query
                query_words = set(query.lower().split())
                for sentence in sentences:
                    sentence_words = set(sentence.lower().split())
                    overlap = len(query_words & sentence_words)
                    
                    if overlap > 0:
                        context['sentences'].append({
                            'text': sentence,
                            'score': overlap * activation,
                            'activation': activation
                        })
                
                # Track best text
                if activation > context['best_activation']:
                    context['best_text'] = text
                    context['best_activation'] = activation
                
                # Collect keywords
                keywords = self._extract_keywords(text)
                context['keywords'].update(keywords[:5])
            
            if hasattr(neuron, 'semantic_tags'):
                context['tags'].update(neuron.semantic_tags or [])
        
        # Sort sentences by score
        context['sentences'].sort(key=lambda x: x['score'], reverse=True)
        
        return context
    
    def _generate_by_type(
        self,
        query: str,
        question_type: str,
        context: Dict[str, Any]
    ) -> str:
        """Generiert Antwort basierend auf Typ."""
        
        if question_type == 'greeting':
            return self._generate_greeting()
        
        if question_type in ['definition', 'definition_plural']:
            return self._generate_definition(query, context)
        
        if question_type == 'how':
            return self._generate_how(query, context)
        
        if question_type == 'why':
            return self._generate_why(query, context)
        
        if question_type in ['who', 'when', 'where']:
            return self._generate_factual(query, context)
        
        if question_type == 'question':
            return self._generate_general(query, context)
        
        return self._generate_conversational(context)
    
    def _generate_greeting(self) -> str:
        """Generiert Begrüßung."""
        greetings = [
            "Hello! I'm Friday, your AI assistant. How can I help you today?",
            "Hi there! I'm here to help. What would you like to know?",
            "Hello! What can I do for you?",
            "Hey! I'm ready to assist you. What's on your mind?",
        ]
        return np.random.choice(greetings)
    
    def _generate_definition(self, query: str, context: Dict[str, Any]) -> str:
        """Generiert Definition."""
        # Finde beste Sätze
        if not context['sentences']:
            return "I don't have enough information to define that."
        
        # Extrahiere Subjekt aus Frage
        subject = self._extract_subject(query)
        if not subject:
            subject = query.lower().replace('what is', '').replace('?', '').strip()
        
        subject_lower = subject.lower()
        
        # Suche nach Sätzen die das Subjekt definieren
        for sent_data in context['sentences'][:15]:  # Check top 15
            sentence = sent_data['text']
            sentence_lower = sentence.lower()
            
            # Skip zu spezifische Sätze
            if any(word in sentence_lower for word in ['midshipman', 'dabney', 'atlanta', 'monitors']):
                continue
            
            # WICHTIG: Prüfe ob Satz das Subjekt enthält
            if subject_lower not in sentence_lower:
                continue
            
            # Bevorzuge Sätze die mit dem Subjekt starten und "is" enthalten
            # z.B. "AI is...", "Water is...", "Gold is..."
            if sentence_lower.startswith(subject_lower) and ' is ' in sentence_lower:
                return self._cleanup_sentence(sentence)
            
            # Bevorzuge Sätze mit "X is a/an"
            patterns = [
                f'{subject_lower} is a ',
                f'{subject_lower} is an ',
                f'{subject_lower} is the ',
            ]
            if any(pattern in sentence_lower for pattern in patterns):
                return self._cleanup_sentence(sentence)
            
            # Bevorzuge Sätze die mit "is", "was", "are" starten
            if any(sentence_lower.startswith(s) for s in ['it is', 'it was', 'this is', 'this was', 'they are', 'they were']):
                # Aber nur wenn Subjekt im Satz ist
                if subject_lower in sentence_lower:
                    return self._cleanup_sentence(sentence)
        
        # Zweiter Durchlauf: Weniger streng
        for sent_data in context['sentences'][:10]:
            sentence = sent_data['text']
            sentence_lower = sentence.lower()
            
            # Skip zu spezifische Sätze
            if any(word in sentence_lower for word in ['midshipman', 'dabney', 'atlanta', 'monitors']):
                continue
            
            # Wenn Subjekt im Satz und "is" vorhanden
            if subject_lower in sentence_lower and ' is ' in sentence_lower:
                return self._cleanup_sentence(sentence)
        
        # Fallback: Nimm besten Satz der das Subjekt enthält
        for sent_data in context['sentences']:
            sentence = sent_data['text']
            if subject_lower in sentence.lower():
                sentence = self._cleanup_sentence(sentence)
                if len(sentence) < 200:
                    return sentence
        
        # Letzter Fallback
        return "I found information about that, but I'm not sure how to explain it clearly. Could you rephrase your question?"
    
    def _cleanup_sentence(self, sentence: str) -> str:
        """Bereinigt Satz von technischen Details."""
        original = sentence
        
        # Entferne Q&A Marker (mehrfach anwenden falls nötig)
        sentence = re.sub(r'^Question:\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'^Answer:\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\bQuestion:\s*', '', sentence, flags=re.IGNORECASE)
        sentence = re.sub(r'\bAnswer:\s*', '', sentence, flags=re.IGNORECASE)
        
        # Entferne Fragen am Anfang wenn gefolgt von Antwort
        # Beispiel: "what is the capital of germany Berlin is the Capital"
        # Wir wollen nur: "Berlin is the Capital"
        
        # Prüfe ob Satz mit Fragewort beginnt
        sentence_lower = sentence.lower()
        starts_with_question = any(
            sentence_lower.startswith(q + ' ') 
            for q in ['what', 'who', 'where', 'when', 'why', 'how', 'was', 'ist', 'wer', 'wo', 'wann', 'warum', 'wie']
        )
        
        if starts_with_question:
            # Suche nach einem Großbuchstaben-Wort das einen neuen Satz beginnt
            parts = sentence.split()
            
            # Finde den ersten Teil der mit Großbuchstaben beginnt und ein Verb folgt
            for i, word in enumerate(parts):
                if i > 0 and len(word) > 0 and word[0].isupper() and word.lower() not in ['i', 'a']:
                    # Prüfe ob das nächste Wort ein Verb ist (is, are, was, etc.)
                    if i + 1 < len(parts) and parts[i + 1].lower() in ['is', 'are', 'was', 'were', 'has', 'have', 'had']:
                        # Das ist wahrscheinlich der Beginn der Antwort
                        sentence = ' '.join(parts[i:])
                        break
        
        # Entferne @-@ Marker
        sentence = re.sub(r'\s*@-@\s*', ' ', sentence)
        sentence = re.sub(r'\s*@\s*', ' ', sentence)
        
        # Entferne Klammern mit Zahlen/Details
        sentence = re.sub(r'\s*\([^)]*\d[^)]*\)', '', sentence)
        
        # Entferne mehrfache Leerzeichen
        sentence = re.sub(r'\s+', ' ', sentence)
        
        # Trim
        sentence = sentence.strip()
        
        # Wenn Satz zu kurz oder leer, nimm Original
        if len(sentence) < 10:
            sentence = original.strip()
        
        # Stelle sicher dass Satz mit Punkt endet
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        return sentence
    
    def _generate_how(self, query: str, context: Dict[str, Any]) -> str:
        """Generiert How-Antwort."""
        if not context['sentences']:
            return "I don't have enough information to explain how that works."
        
        # Suche nach Sätzen mit Prozess-Wörtern
        process_words = ['by', 'through', 'using', 'with', 'via', 'when', 'after']
        
        for sent_data in context['sentences']:
            sentence = sent_data['text']
            if any(word in sentence.lower() for word in process_words):
                return self._cleanup_sentence(sentence)
        
        # Fallback: Bester Satz
        return self._cleanup_sentence(context['sentences'][0]['text'])
    
    def _generate_why(self, query: str, context: Dict[str, Any]) -> str:
        """Generiert Why-Antwort."""
        if not context['sentences']:
            return "I don't have enough information to explain why."
        
        # Suche nach Sätzen mit Begründungen
        reason_words = ['because', 'since', 'due to', 'as', 'therefore', 'thus']
        
        for sent_data in context['sentences']:
            sentence = sent_data['text']
            if any(word in sentence.lower() for word in reason_words):
                return self._cleanup_sentence(sentence)
        
        # Fallback: Füge "This is because" hinzu
        best = self._cleanup_sentence(context['sentences'][0]['text'])
        return f"This is because {best.lower()}"
    
    def _generate_factual(self, query: str, context: Dict[str, Any]) -> str:
        """Generiert faktische Antwort (who/when/where)."""
        if not context['sentences']:
            return "I don't have that information."
        
        # Nimm besten Satz
        return self._cleanup_sentence(context['sentences'][0]['text'])
    
    def _generate_general(self, query: str, context: Dict[str, Any]) -> str:
        """Generiert allgemeine Antwort."""
        if not context['sentences']:
            return "I don't have enough information to answer that."
        
        # Kombiniere top 2 Sätze wenn beide gut sind
        if len(context['sentences']) >= 2:
            first = context['sentences'][0]
            second = context['sentences'][1]
            
            if second['score'] > first['score'] * 0.7:  # Zweiter ist auch relevant
                first_clean = self._cleanup_sentence(first['text'])
                second_clean = self._cleanup_sentence(second['text'])
                return f"{first_clean} {second_clean}"
        
        return self._cleanup_sentence(context['sentences'][0]['text'])
    
    def _generate_conversational(self, context: Dict[str, Any]) -> str:
        """Generiert konversationelle Antwort."""
        if context['tags']:
            tags = list(context['tags'])[:3]
            return f"I can tell you about {', '.join(tags)}. What would you like to know?"
        
        return "I'm here to help! What would you like to know?"
    
    def _generate_fallback(self, query: str) -> str:
        """Generiert Fallback wenn kein Kontext."""
        question_type = self._analyze_question(query)
        
        if question_type == 'greeting':
            return self._generate_greeting()
        
        return "I don't have enough information to answer that question. Could you rephrase it or ask something else?"
    
    def _extract_subject(self, query: str) -> Optional[str]:
        """Extrahiert Subjekt aus Frage."""
        # "What is X?" → "X"
        patterns = [
            r'what is (?:a |an |the )?(.+?)\??$',
            r'was ist (?:ein |eine |der |die |das )?(.+?)\??$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                subject = match.group(1).strip()
                # Capitalize first letter
                return subject.capitalize()
        
        return None
