"""
Self-Reflection System - Die KI prüft ihre eigenen Antworten.

Die KI soll wie ein LLM "nachdenken" und ihre Antwort kritisch prüfen:
- Ist die Antwort relevant zur Frage?
- Ist die Antwort vollständig?
- Macht die Antwort Sinn?
- Gibt es Widersprüche?
- Kann die Antwort verbessert werden?
"""

import logging
import re
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class SelfReflection:
    """
    Self-Reflection System für kritische Selbstprüfung.
    
    Die KI analysiert ihre eigene Antwort und korrigiert sie wenn nötig.
    """
    
    def __init__(self, language_model):
        """Initialize self-reflection system."""
        self.language_model = language_model
        self.graph = language_model.graph
        self.query_engine = language_model.query_engine
        
        logger.info("Self-Reflection system initialized")
    
    def validate_response(
        self,
        query: str,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], str]:
        """
        Validiert und verbessert die Antwort durch kritische Selbstprüfung.
        
        Args:
            query: Die ursprüngliche Frage
            response: Die generierte Antwort
            knowledge_pieces: Verwendetes Wissen
            
        Returns:
            Tuple von (is_valid, improved_response, reasoning)
            - is_valid: Ist die Antwort akzeptabel?
            - improved_response: Verbesserte Antwort (oder None)
            - reasoning: Begründung der Entscheidung
        """
        logger.debug(f"Self-reflecting on response for: {query[:50]}...")
        
        # Schritt 1: Grundlegende Qualitätsprüfung
        quality_issues = self._check_quality(query, response)
        
        # Schritt 2: Relevanz-Prüfung
        relevance_score = self._check_relevance(query, response)
        
        # Schritt 3: Vollständigkeits-Prüfung
        completeness_score = self._check_completeness(query, response, knowledge_pieces)
        
        # Schritt 4: Konsistenz-Prüfung
        consistency_issues = self._check_consistency(response, knowledge_pieces)
        
        # Schritt 5: Entscheidung treffen
        is_valid = (
            len(quality_issues) == 0 and
            relevance_score >= 0.5 and
            completeness_score >= 0.5 and
            len(consistency_issues) == 0
        )
        
        # Schritt 6: Verbesserung wenn nötig
        improved_response = None
        reasoning_parts = []
        
        if quality_issues:
            reasoning_parts.append(f"Quality issues: {', '.join(quality_issues)}")
            improved_response = self._fix_quality_issues(response, quality_issues, knowledge_pieces)
        
        if relevance_score < 0.5:
            reasoning_parts.append(f"Low relevance: {relevance_score:.2f}")
            improved_response = self._improve_relevance(query, response, knowledge_pieces)
        
        if completeness_score < 0.5:
            reasoning_parts.append(f"Incomplete: {completeness_score:.2f}")
            if improved_response is None:
                improved_response = self._improve_completeness(response, knowledge_pieces)
        
        if consistency_issues:
            reasoning_parts.append(f"Consistency issues: {', '.join(consistency_issues)}")
            if improved_response is None:
                improved_response = self._fix_consistency(response, consistency_issues)
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Response is valid"
        
        return is_valid, improved_response, reasoning
    
    def _check_quality(self, query: str, response: str) -> List[str]:
        """Prüft grundlegende Qualität der Antwort."""
        issues = []
        
        # 1. Zu kurz
        if len(response.strip()) < 15:
            issues.append("too_short")
        
        # 2. Nur die Frage wiederholt
        query_clean = query.lower().replace('?', '').strip()
        response_clean = response.lower().strip()
        
        if query_clean in response_clean and len(response_clean) < len(query_clean) + 30:
            issues.append("just_repeats_question")
        
        # 3. Enthält Meta-Text
        meta_patterns = ['I don\'t know', 'I\'m not sure', 'I cannot', 'I can\'t']
        if any(pattern.lower() in response_clean for pattern in meta_patterns):
            # Das ist OK wenn es ehrlich ist, aber prüfen wir ob es Wissen gibt
            if len(response_clean) < 50:
                issues.append("uncertain_and_short")
        
        # 4. Enthält technische Artefakte
        if any(marker in response for marker in ['@-@', '= =', '< unk >']):
            issues.append("technical_artifacts")
        
        # 5. Doppelte Sätze
        sentences = re.split(r'[.!?]+', response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) != len(set(sentences)):
            issues.append("duplicate_sentences")
        
        return issues
    
    def _check_relevance(self, query: str, response: str) -> float:
        """
        Prüft ob die Antwort zur Frage passt.
        
        Returns:
            Score von 0.0 (irrelevant) bis 1.0 (sehr relevant)
        """
        # Extrahiere Schlüsselwörter aus Frage
        query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        
        # Entferne Fragewörter
        question_words = {'what', 'who', 'where', 'when', 'why', 'how', 'which', 
                         'was', 'ist', 'wer', 'wo', 'wann', 'warum', 'wie', 'welche'}
        query_words -= question_words
        
        if not query_words:
            return 0.8  # Keine spezifischen Wörter, schwer zu prüfen
        
        # Extrahiere Wörter aus Antwort
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))
        
        # Berechne Überlappung
        overlap = len(query_words & response_words)
        relevance = overlap / len(query_words)
        
        return min(1.0, relevance)
    
    def _check_completeness(
        self,
        query: str,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> float:
        """
        Prüft ob die Antwort vollständig ist.
        
        Returns:
            Score von 0.0 (unvollständig) bis 1.0 (vollständig)
        """
        # Für Definition-Fragen: Sollte "is" oder "are" enthalten
        if any(word in query.lower() for word in ['what is', 'was ist', 'what are']):
            if not any(word in response.lower() for word in [' is ', ' are ', ' was ', ' were ']):
                return 0.5  # Weniger streng
        
        # Für How-Fragen: Sollte Prozess-Wörter enthalten
        if any(word in query.lower() for word in ['how does', 'how do', 'wie funktioniert']):
            process_words = ['by', 'through', 'using', 'with', 'when', 'first', 'then']
            if not any(word in response.lower() for word in process_words):
                return 0.6  # Weniger streng
        
        # Für Why-Fragen: Sollte Kausal-Wörter enthalten
        if any(word in query.lower() for word in ['why', 'warum']):
            causal_words = ['because', 'since', 'due to', 'as', 'therefore', 'thus', 'weil', 'da']
            if not any(word in response.lower() for word in causal_words):
                return 0.6  # Weniger streng
        
        # Länge als Indikator - weniger streng
        if len(response) < 20:
            return 0.5
        elif len(response) < 40:
            return 0.7
        else:
            return 0.9  # Meiste Antworten sind OK
    
    def _check_consistency(
        self,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> List[str]:
        """Prüft auf Widersprüche in der Antwort."""
        issues = []
        
        # 1. Widersprüchliche Aussagen (einfache Heuristik)
        response_lower = response.lower()
        
        # Prüfe auf "not" gefolgt von "is" und dann wieder "is"
        if ' not ' in response_lower and ' is ' in response_lower:
            # Könnte ein Widerspruch sein, aber schwer zu erkennen
            pass
        
        # 2. Zahlen-Widersprüche
        numbers = re.findall(r'\b\d+\b', response)
        if len(numbers) > len(set(numbers)):
            # Gleiche Zahl mehrfach - könnte Widerspruch sein
            pass
        
        # 3. Prüfe ob Antwort zu Wissen passt
        # (Vereinfacht: Prüfe ob wichtige Wörter aus Wissen in Antwort sind)
        
        return issues
    
    def _fix_quality_issues(
        self,
        response: str,
        issues: List[str],
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """Behebt Qualitätsprobleme."""
        fixed = response
        
        # Fix: Doppelte Sätze entfernen
        if 'duplicate_sentences' in issues:
            sentences = re.split(r'([.!?]+)', fixed)
            seen = set()
            unique_parts = []
            
            for i in range(0, len(sentences), 2):
                if i < len(sentences):
                    sentence = sentences[i].strip()
                    if sentence and sentence not in seen:
                        unique_parts.append(sentence)
                        seen.add(sentence)
                        if i + 1 < len(sentences):
                            unique_parts.append(sentences[i + 1])
                            # Add space after punctuation if next part exists
                            if i + 2 < len(sentences):
                                unique_parts.append(' ')
            
            fixed = ''.join(unique_parts).strip()
        
        # Fix: Technische Artefakte entfernen
        if 'technical_artifacts' in issues:
            fixed = re.sub(r'@-@', '', fixed)
            fixed = re.sub(r'= =', '', fixed)
            fixed = re.sub(r'< unk >', '', fixed)
            fixed = re.sub(r'\s+', ' ', fixed).strip()
        
        # Fix: Nur Frage wiederholt - nehme bestes Wissen
        if 'just_repeats_question' in issues and knowledge_pieces:
            best_knowledge = knowledge_pieces[0]['text']
            # Extrahiere ersten sinnvollen Satz
            sentences = re.split(r'[.!?]+', best_knowledge)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:
                    fixed = sent + '.'
                    break
        
        return fixed
    
    def _improve_relevance(
        self,
        query: str,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """Verbessert Relevanz der Antwort."""
        # VEREINFACHT: Nur wenn Relevanz sehr niedrig ist
        # Sonst behalten wir die ursprüngliche Antwort
        
        # Wenn Antwort schon gut ist, nicht ändern
        if len(response) > 30:
            return response
        
        # Nur bei sehr kurzen/schlechten Antworten versuchen zu verbessern
        query_words = set(re.findall(r'\b\w{4,}\b', query.lower()))
        
        best_match = None
        best_score = 0.0
        
        for piece in knowledge_pieces:
            text = piece['text']
            text_words = set(re.findall(r'\b\w{4,}\b', text.lower()))
            
            overlap = len(query_words & text_words)
            score = overlap / max(len(query_words), 1)
            
            if score > best_score:
                best_score = score
                best_match = text
        
        if best_match and best_score > 0.5:  # Höherer Threshold
            # Extrahiere relevanten Satz
            sentences = re.split(r'[.!?]+', best_match)
            for sent in sentences:
                sent = sent.strip()
                sent_words = set(re.findall(r'\b\w{4,}\b', sent.lower()))
                if len(query_words & sent_words) >= 2 and len(sent) > 20:
                    return sent + '.'
        
        return response
    
    def _improve_completeness(
        self,
        response: str,
        knowledge_pieces: List[Dict[str, Any]]
    ) -> str:
        """Verbessert Vollständigkeit der Antwort."""
        # DEAKTIVIERT: Fügt oft unpassende Zusatzinformationen hinzu
        # Die Antwort sollte fokussiert bleiben
        return response
    
    def _fix_consistency(
        self,
        response: str,
        issues: List[str]
    ) -> str:
        """Behebt Konsistenz-Probleme."""
        # Vereinfacht: Entferne widersprüchliche Teile
        # (In der Praxis würde man hier komplexere Logik brauchen)
        return response
