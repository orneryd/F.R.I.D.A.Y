"""
Intelligent Response Synthesis - Versteht und kombiniert Neuron-Daten intelligent.

Dieses Modul analysiert aktivierte Neuronen und generiert daraus sinnvolle,
kontextbezogene Antworten statt nur Text-Snippets zurückzugeben.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class IntelligentSynthesizer:
    """
    Intelligente Response-Synthese die Neuron-Daten wirklich versteht.
    
    Statt nur Text-Snippets zurückzugeben, analysiert dieser Synthesizer:
    - Den Kontext der Frage
    - Die Bedeutung der aktivierten Neuronen
    - Beziehungen zwischen Informationen
    - Und generiert daraus kohärente Antworten
    """
    
    def __init__(self):
        """Initialize the intelligent synthesizer."""
        self.logger = logging.getLogger(__name__)
        
        # Pattern für verschiedene Fragetypen
        self.question_patterns = {
            'definition': r'\b(was ist|what is|define|definition|bedeutet|means)\b',
            'how': r'\b(wie|how|auf welche weise)\b',
            'why': r'\b(warum|why|weshalb|wieso)\b',
            'when': r'\b(wann|when)\b',
            'where': r'\b(wo|where)\b',
            'who': r'\b(wer|who)\b',
            'comparison': r'\b(unterschied|difference|vergleich|compare|vs|versus)\b',
            'list': r'\b(liste|list|nenne|name|welche|which)\b',
            'yes_no': r'\b(ist|is|kann|can|hat|has|gibt es|is there)\b',
        }
    
    def synthesize_response(
        self,
        query: str,
        knowledge_pieces: List[Dict[str, Any]],
        min_confidence: float = 0.3
    ) -> str:
        """
        Generiert eine intelligente Antwort aus Neuron-Daten.
        
        Args:
            query: Die ursprüngliche Frage
            knowledge_pieces: Liste von aktivierten Neuronen mit ihren Daten
            min_confidence: Minimale Konfidenz für Antworten
            
        Returns:
            Synthetisierte, intelligente Antwort
        """
        if not knowledge_pieces:
            return "Ich habe leider keine relevanten Informationen zu dieser Frage."
        
        # 1. Analysiere die Frage
        question_type = self._analyze_question_type(query)
        self.logger.debug(f"Question type detected: {question_type}")
        
        # 2. Extrahiere und verstehe die Informationen aus Neuronen
        understood_info = self._understand_knowledge(knowledge_pieces, query)
        
        if not understood_info:
            return "Ich konnte die Informationen nicht richtig verstehen."
        
        # 3. Filtere nach Konfidenz
        confident_info = [
            info for info in understood_info
            if info['confidence'] >= min_confidence
        ]
        
        if not confident_info:
            return "Ich bin mir nicht sicher genug, um diese Frage zu beantworten."
        
        # 4. Generiere Antwort basierend auf Fragetyp
        response = self._generate_typed_response(
            query, question_type, confident_info
        )
        
        return response
    
    def _analyze_question_type(self, query: str) -> str:
        """
        Analysiert den Typ der Frage.
        
        Args:
            query: Die Frage
            
        Returns:
            Fragetyp (definition, how, why, etc.)
        """
        query_lower = query.lower()
        
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, query_lower):
                return q_type
        
        return 'general'
    
    def _understand_knowledge(
        self,
        knowledge_pieces: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Versteht und strukturiert die Informationen aus Neuronen.
        
        Args:
            knowledge_pieces: Rohe Neuron-Daten
            query: Die ursprüngliche Frage
            
        Returns:
            Liste von verstandenen Informationen mit Kontext
        """
        understood = []
        
        for piece in knowledge_pieces:
            text = piece.get('text', '').strip()
            activation = piece.get('activation', 0.0)
            tags = piece.get('tags', [])
            
            if not text:
                continue
            
            # Extrahiere strukturierte Information
            info = self._extract_structured_info(text, query)
            
            if info:
                info['activation'] = activation
                info['tags'] = tags
                info['confidence'] = self._calculate_confidence(info, query, activation)
                understood.append(info)
        
        # Sortiere nach Konfidenz
        understood.sort(key=lambda x: x['confidence'], reverse=True)
        
        return understood
    
    def _extract_structured_info(
        self,
        text: str,
        query: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extrahiert strukturierte Information aus Neuron-Text.
        
        Args:
            text: Roher Text aus Neuron
            query: Die Frage (für Kontext)
            
        Returns:
            Strukturierte Information oder None
        """
        # Q&A Format
        if "Question:" in text and "Answer:" in text:
            return self._parse_qa_format(text)
        
        # Definitions-Format (z.B. "AI ist...")
        if self._is_definition(text):
            return self._parse_definition(text)
        
        # Listen-Format
        if self._is_list(text):
            return self._parse_list(text)
        
        # Einfacher Fakt
        return {
            'type': 'fact',
            'content': text,
            'raw_text': text
        }
    
    def _parse_qa_format(self, text: str) -> Dict[str, Any]:
        """Parst Q&A Format."""
        try:
            # Extrahiere Frage und Antwort
            question_match = re.search(r'Question:\s*(.+?)(?=Answer:|$)', text, re.DOTALL)
            answer_match = re.search(r'Answer:\s*(.+?)$', text, re.DOTALL)
            
            question = question_match.group(1).strip() if question_match else ""
            answer = answer_match.group(1).strip() if answer_match else text
            
            return {
                'type': 'qa',
                'question': question,
                'answer': answer,
                'content': answer,
                'raw_text': text
            }
        except Exception as e:
            self.logger.warning(f"Failed to parse Q&A format: {e}")
            return {'type': 'fact', 'content': text, 'raw_text': text}
    
    def _parse_definition(self, text: str) -> Dict[str, Any]:
        """Parst Definitions-Format."""
        # Suche nach "X ist Y" oder "X is Y" Pattern
        patterns = [
            r'(.+?)\s+(?:ist|is|bedeutet|means)\s+(.+)',
            r'(.+?):\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                return {
                    'type': 'definition',
                    'term': term,
                    'definition': definition,
                    'content': definition,
                    'raw_text': text
                }
        
        return {'type': 'fact', 'content': text, 'raw_text': text}
    
    def _parse_list(self, text: str) -> Dict[str, Any]:
        """Parst Listen-Format."""
        # Suche nach Listen-Markern
        lines = text.split('\n')
        items = []
        
        for line in lines:
            line = line.strip()
            # Bullet points, Nummern, etc.
            if re.match(r'^[\-\*\•\d]+[\.\)]\s+', line):
                item = re.sub(r'^[\-\*\•\d]+[\.\)]\s+', '', line)
                items.append(item)
        
        if items:
            return {
                'type': 'list',
                'items': items,
                'content': ', '.join(items),
                'raw_text': text
            }
        
        return {'type': 'fact', 'content': text, 'raw_text': text}
    
    def _is_definition(self, text: str) -> bool:
        """Prüft ob Text eine Definition ist."""
        definition_markers = [
            r'\b(?:ist|is|bedeutet|means|refers to|defined as)\b',
            r':\s*[A-Z]',  # "Term: Definition"
        ]
        return any(re.search(marker, text, re.IGNORECASE) for marker in definition_markers)
    
    def _is_list(self, text: str) -> bool:
        """Prüft ob Text eine Liste ist."""
        list_markers = [
            r'^\s*[\-\*\•]',  # Bullet points
            r'^\s*\d+[\.\)]',  # Numbered lists
        ]
        lines = text.split('\n')
        return any(
            any(re.match(marker, line) for marker in list_markers)
            for line in lines
        )
    
    def _calculate_confidence(
        self,
        info: Dict[str, Any],
        query: str,
        activation: float
    ) -> float:
        """
        Berechnet Konfidenz für eine Information.
        
        Args:
            info: Strukturierte Information
            query: Die Frage
            activation: Neuron-Aktivierung
            
        Returns:
            Konfidenz-Score (0.0 - 1.0)
        """
        confidence = activation
        
        # Bonus für Q&A Format (direktere Antworten)
        if info.get('type') == 'qa':
            confidence *= 1.2
        
        # Bonus für Definitionen bei Definition-Fragen
        if info.get('type') == 'definition' and 'was ist' in query.lower():
            confidence *= 1.15
        
        # Bonus für relevante Keywords
        content = info.get('content', '').lower()
        query_words = set(query.lower().split())
        content_words = set(content.split())
        overlap = len(query_words & content_words)
        
        if overlap > 0:
            confidence *= (1.0 + overlap * 0.05)
        
        # Penalty für zu kurze Antworten
        if len(content) < 20:
            confidence *= 0.8
        
        # Penalty für Meta-Instruktionen
        if any(phrase in content for phrase in ['when asked', 'means asking', 'means requesting']):
            confidence *= 0.3
        
        return min(1.0, confidence)
    
    def _generate_typed_response(
        self,
        query: str,
        question_type: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """
        Generiert Antwort basierend auf Fragetyp.
        
        Args:
            query: Die Frage
            question_type: Typ der Frage
            info_list: Liste von verstandenen Informationen
            
        Returns:
            Generierte Antwort
        """
        if not info_list:
            return "Ich habe keine passenden Informationen gefunden."
        
        # Hole die beste Information
        best_info = info_list[0]
        
        # Spezielle Behandlung nach Fragetyp
        if question_type == 'definition':
            return self._generate_definition_response(query, info_list)
        
        elif question_type == 'how':
            return self._generate_how_response(query, info_list)
        
        elif question_type == 'why':
            return self._generate_why_response(query, info_list)
        
        elif question_type == 'comparison':
            return self._generate_comparison_response(query, info_list)
        
        elif question_type == 'list':
            return self._generate_list_response(query, info_list)
        
        elif question_type == 'yes_no':
            return self._generate_yes_no_response(query, info_list)
        
        else:
            # Generelle Antwort
            return self._generate_general_response(query, info_list)
    
    def _generate_definition_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert Antwort für Definitions-Fragen."""
        best = info_list[0]
        
        if best['type'] == 'definition':
            return best['definition']
        
        # Kombiniere mehrere Definitionen wenn verfügbar
        definitions = [
            info['content'] for info in info_list[:2]
            if info['type'] in ['definition', 'qa', 'fact']
        ]
        
        if len(definitions) == 1:
            return definitions[0]
        elif len(definitions) > 1:
            # Kombiniere mit "und"
            return f"{definitions[0]}. {definitions[1]}"
        
        return best['content']
    
    def _generate_how_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert Antwort für How-Fragen."""
        # Suche nach prozess-orientierten Informationen
        process_info = [
            info for info in info_list
            if any(word in info['content'].lower() 
                   for word in ['durch', 'by', 'mittels', 'using', 'via'])
        ]
        
        if process_info:
            return process_info[0]['content']
        
        return info_list[0]['content']
    
    def _generate_why_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert Antwort für Why-Fragen."""
        # Suche nach Begründungen
        reason_info = [
            info for info in info_list
            if any(word in info['content'].lower() 
                   for word in ['weil', 'because', 'da', 'since', 'deshalb', 'therefore'])
        ]
        
        if reason_info:
            return reason_info[0]['content']
        
        # Füge "weil" hinzu wenn nicht vorhanden
        content = info_list[0]['content']
        if 'weil' not in content.lower() and 'because' not in content.lower():
            return f"Das liegt daran, dass {content.lower()}"
        
        return content
    
    def _generate_comparison_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert Antwort für Vergleichs-Fragen."""
        # Kombiniere mehrere Informationen für Vergleich
        if len(info_list) >= 2:
            return f"{info_list[0]['content']}. Im Vergleich dazu: {info_list[1]['content']}"
        
        return info_list[0]['content']
    
    def _generate_list_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert Antwort für Listen-Fragen."""
        best = info_list[0]
        
        if best['type'] == 'list':
            items = best['items']
            if len(items) <= 3:
                return ', '.join(items)
            else:
                return ', '.join(items[:3]) + f' und {len(items)-3} weitere'
        
        # Kombiniere mehrere Fakten als Liste
        if len(info_list) > 1:
            items = [info['content'] for info in info_list[:3]]
            return '. '.join(items)
        
        return best['content']
    
    def _generate_yes_no_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert Antwort für Ja/Nein-Fragen."""
        content = info_list[0]['content'].lower()
        
        # Suche nach Ja/Nein Indikatoren
        yes_indicators = ['ja', 'yes', 'kann', 'can', 'ist', 'is', 'hat', 'has']
        no_indicators = ['nein', 'no', 'nicht', 'not', 'kein', 'none']
        
        if any(word in content for word in yes_indicators):
            return f"Ja, {info_list[0]['content']}"
        elif any(word in content for word in no_indicators):
            return f"Nein, {info_list[0]['content']}"
        
        return info_list[0]['content']
    
    def _generate_general_response(
        self,
        query: str,
        info_list: List[Dict[str, Any]]
    ) -> str:
        """Generiert allgemeine Antwort."""
        best = info_list[0]
        
        # WICHTIG: Nimm NUR die beste Antwort, keine Kombinationen!
        # Alle Überlegungen gehören ins Reasoning, nicht in die finale Antwort.
        # Kombinationen mit "Zusätzlich:" verwirren und machen die Antwort unrein.
        
        return best['content']
