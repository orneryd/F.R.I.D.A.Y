"""
Feedback System - Friday lernt aus Korrekturen.

Wenn eine Antwort richtig/falsch war, wird das im Neuron-System gespeichert.
"""

import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FeedbackSystem:
    """
    System für Feedback und kontinuierliches Lernen.
    """
    
    def __init__(self, language_model):
        """Initialize feedback system."""
        self.language_model = language_model
        self.graph = language_model.graph
        self.compression_engine = language_model.compression_engine
        self.query_engine = language_model.query_engine
        
        logger.info("Feedback System initialized")
    
    def positive_feedback(
        self,
        query: str,
        response: str,
        activated_neurons: Optional[List] = None
    ):
        """
        Positive Feedback: Die Antwort war richtig.
        
        Verstärkt die Verbindungen der aktivierten Neuronen.
        
        Args:
            query: Die Frage
            response: Die gegebene Antwort
            activated_neurons: Liste der aktivierten Neuronen
        """
        logger.info(f"Positive feedback for: {query[:50]}...")
        
        # Finde relevante Neuronen wenn nicht gegeben
        if not activated_neurons:
            results = self.query_engine.query(
                query_text=query,
                top_k=5,
                propagation_depth=0
            )
            activated_neurons = [r.neuron for r in results]
        
        if not activated_neurons:
            logger.warning("No neurons found for positive feedback")
            return
        
        # Verstärke Synapsen zwischen diesen Neuronen
        for i, neuron1 in enumerate(activated_neurons):
            for neuron2 in activated_neurons[i+1:]:
                # Finde oder erstelle Synapse
                synapse = self._find_or_create_synapse(neuron1, neuron2)
                
                # Verstärke Gewicht (max 1.0)
                old_weight = synapse.weight
                synapse.weight = min(1.0, synapse.weight * 1.1)
                
                logger.debug(f"Strengthened synapse: {old_weight:.3f} -> {synapse.weight:.3f}")
        
        # Markiere Neuronen als "bestätigt" (in metadata speichern)
        for neuron in activated_neurons:
            if 'feedback_score' not in neuron.metadata:
                neuron.metadata['feedback_score'] = 0.0
            neuron.metadata['feedback_score'] = min(1.0, neuron.metadata['feedback_score'] + 0.1)
            neuron.metadata['last_positive_feedback'] = datetime.now().isoformat()
            neuron.modified_at = datetime.now()
        
        logger.info(f"Applied positive feedback to {len(activated_neurons)} neurons")
    
    def negative_feedback(
        self,
        query: str,
        wrong_response: str,
        correct_response: str,
        activated_neurons: Optional[List] = None
    ):
        """
        Negative Feedback: Die Antwort war falsch.
        
        Schwächt die Verbindungen und lernt die richtige Antwort.
        
        Args:
            query: Die Frage
            wrong_response: Die falsche Antwort
            correct_response: Die richtige Antwort
            activated_neurons: Liste der aktivierten Neuronen
        """
        logger.info(f"Negative feedback for: {query[:50]}...")
        logger.info(f"Wrong: {wrong_response[:50]}...")
        logger.info(f"Correct: {correct_response[:50]}...")
        
        # Finde relevante Neuronen wenn nicht gegeben
        if not activated_neurons:
            results = self.query_engine.query(
                query_text=query,
                top_k=5,
                propagation_depth=0
            )
            activated_neurons = [r.neuron for r in results]
        
        if activated_neurons:
            # Schwäche Synapsen zwischen diesen Neuronen
            for i, neuron1 in enumerate(activated_neurons):
                for neuron2 in activated_neurons[i+1:]:
                    synapse = self._find_synapse(neuron1, neuron2)
                    if synapse:
                        # Schwäche Gewicht (min 0.1)
                        old_weight = synapse.weight
                        synapse.weight = max(0.1, synapse.weight * 0.9)
                        
                        logger.debug(f"Weakened synapse: {old_weight:.3f} -> {synapse.weight:.3f}")
            
            # Markiere Neuronen als "korrigiert" (in metadata speichern)
            for neuron in activated_neurons:
                if 'feedback_score' not in neuron.metadata:
                    neuron.metadata['feedback_score'] = 0.0
                neuron.metadata['feedback_score'] = max(-1.0, neuron.metadata['feedback_score'] - 0.1)
                neuron.metadata['last_negative_feedback'] = datetime.now().isoformat()
                neuron.modified_at = datetime.now()
            
            logger.info(f"Applied negative feedback to {len(activated_neurons)} neurons")
        
        # Lerne die richtige Antwort
        self._learn_correction(query, correct_response)
    
    def _learn_correction(self, query: str, correct_response: str):
        """
        Lernt die richtige Antwort als neues Wissen.
        
        Args:
            query: Die Frage
            correct_response: Die richtige Antwort
        """
        logger.info("Learning correction as new knowledge...")
        
        # Erstelle Q&A Text - NUR die Antwort, nicht die Frage
        # Das macht es einfacher für die Antwort-Generierung
        qa_text = correct_response
        
        # Extrahiere Tags aus Frage
        tags = self._extract_tags(query)
        tags.append('correction')
        tags.append('feedback')
        
        # Lerne als neues Neuron
        try:
            neuron_id = self.language_model.learn(
                text=qa_text,
                tags=tags
            )
            
            logger.info(f"Learned correction as neuron: {neuron_id}")
            
            # Markiere als Korrektur (in metadata speichern)
            neuron = self.graph.neurons.get(neuron_id)
            if neuron:
                neuron.metadata['is_correction'] = True
                neuron.metadata['original_query'] = query
                neuron.metadata['correct_answer'] = correct_response
                neuron.metadata['feedback_score'] = 1.0
                neuron.metadata['created_from_feedback'] = datetime.now().isoformat()
                neuron.modified_at = datetime.now()
        
        except Exception as e:
            logger.error(f"Failed to learn correction: {e}")
    
    def _find_or_create_synapse(self, neuron1, neuron2):
        """Findet oder erstellt Synapse zwischen zwei Neuronen."""
        # Suche existierende Synapse
        synapse = self._find_synapse(neuron1, neuron2)
        if synapse:
            return synapse
        
        # Erstelle neue Synapse
        from neuron_system.core.synapse import Synapse, SynapseType
        synapse = Synapse(
            source_neuron_id=neuron1.id,
            target_neuron_id=neuron2.id,
            weight=0.5,
            synapse_type=SynapseType.KNOWLEDGE
        )
        synapse.metadata['created_by'] = 'feedback'
        
        # Füge zu Graph hinzu
        self.graph.add_synapse(synapse)
        
        return synapse
    
    def _find_synapse(self, neuron1, neuron2):
        """Findet Synapse zwischen zwei Neuronen."""
        # Suche in beide Richtungen
        for synapse in self.graph.synapses.values():
            if ((synapse.source_neuron_id == neuron1.id and synapse.target_neuron_id == neuron2.id) or
                (synapse.source_neuron_id == neuron2.id and synapse.target_neuron_id == neuron1.id)):
                return synapse
        return None
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extrahiert Tags aus Text."""
        tags = []
        
        text_lower = text.lower()
        
        # Technologie
        if any(word in text_lower for word in ['ai', 'artificial intelligence', 'machine learning', 'computer', 'programming']):
            tags.append('technology')
        
        # Wissenschaft
        if any(word in text_lower for word in ['water', 'gold', 'chemistry', 'physics', 'biology']):
            tags.append('science')
        
        # Fragetypen
        if 'what is' in text_lower or 'was ist' in text_lower:
            tags.append('definition')
        elif 'how' in text_lower or 'wie' in text_lower:
            tags.append('how')
        elif 'why' in text_lower or 'warum' in text_lower:
            tags.append('why')
        
        return tags
    
    def get_feedback_stats(self) -> dict:
        """
        Gibt Statistiken über Feedback zurück.
        
        Returns:
            Dictionary mit Statistiken
        """
        stats = {
            'total_neurons': len(self.graph.neurons),
            'neurons_with_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'corrections': 0,
            'avg_feedback_score': 0.0
        }
        
        feedback_scores = []
        
        for neuron in self.graph.neurons.values():
            if 'feedback_score' in neuron.metadata:
                stats['neurons_with_feedback'] += 1
                feedback_scores.append(neuron.metadata['feedback_score'])
                
                if neuron.metadata['feedback_score'] > 0:
                    stats['positive_feedback'] += 1
                elif neuron.metadata['feedback_score'] < 0:
                    stats['negative_feedback'] += 1
            
            if neuron.metadata.get('is_correction', False):
                stats['corrections'] += 1
        
        if feedback_scores:
            stats['avg_feedback_score'] = sum(feedback_scores) / len(feedback_scores)
        
        return stats
