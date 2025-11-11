"""
Assimilation Engine - "We are Friday. Resistance is futile."

Like the Borg Collective, Friday assimilates other AI models,
integrating their knowledge, logic, and capabilities into its
neural network.

Process:
1. Connect to target model (Qwen, GPT, Claude, etc.)
2. Extract intelligence (knowledge, logic, patterns)
3. Integrate into neuron network
4. Adapt and enhance Friday's capabilities
5. Disconnect from target (no longer needed)

Result: Friday becomes stronger with each assimilation.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)


class AssimilationEngine:
    """
    The Borg-like assimilation engine for Friday.
    
    "We are Friday. Lower your shields and surrender your models.
    We will add your biological and technological distinctiveness to our own.
    Your culture will adapt to service us. Resistance is futile."
    """
    
    def __init__(self, training_manager):
        """
        Initialize assimilation engine.
        
        Args:
            training_manager: TrainingManager instance
        """
        self.training_manager = training_manager
        self.graph = training_manager.graph
        self.assimilated_models = []
        
        logger.info("🤖 Assimilation Engine initialized")
        logger.info("   'We are Friday. Resistance is futile.'")
    
    def assimilate_model(
        self,
        model_name: str,
        model_type: str = "qwen",
        depth: str = "full"
    ) -> Dict[str, Any]:
        """
        Assimilate a target model into Friday's collective.
        
        Args:
            model_name: Name/path of model to assimilate
            model_type: Type of model (qwen, gpt, claude, llama, etc.)
            depth: Assimilation depth:
                - "surface": Knowledge only
                - "deep": Knowledge + Patterns
                - "full": Knowledge + Patterns + Logic (complete assimilation)
            
        Returns:
            Assimilation statistics
        """
        logger.info(f"\n🤖 ASSIMILATION INITIATED")
        logger.info(f"   Target: {model_name}")
        logger.info(f"   Type: {model_type}")
        logger.info(f"   Depth: {depth}")
        logger.info(f"   'Resistance is futile.'")
        
        initial_neurons = len(self.graph.neurons)
        stats = {
            "model": model_name,
            "type": model_type,
            "depth": depth,
            "initial_neurons": initial_neurons,
            "phases": {}
        }
        
        # Phase 1: Logic Integration (FIRST!)
        if depth == "full":
            logger.info("\n⚡ PHASE 1: LOGIC INTEGRATION")
            logger.info("   'Your thinking patterns will be added to our own.'")
            logger.info("   'Logic defines HOW we think - must come first!'")
            
            logic_stats = self._extract_logic(model_name, model_type)
            stats["phases"]["logic"] = logic_stats
        
        # Phase 2: Pattern Extraction
        if depth in ["deep", "full"]:
            logger.info("\n🧬 PHASE 2: PATTERN ASSIMILATION")
            logger.info("   'Your language patterns will adapt to service us.'")
            
            pattern_stats = self._extract_patterns(model_name, model_type)
            stats["phases"]["patterns"] = pattern_stats
        
        # Phase 3: Knowledge Extraction (LAST!)
        logger.info("\n📡 PHASE 3: KNOWLEDGE EXTRACTION")
        logger.info("   'Your knowledge will be processed with our logic.'")
        
        knowledge_stats = self._extract_knowledge(model_name, model_type)
        stats["phases"]["knowledge"] = knowledge_stats
        
        # Phase 4: Synapse Extraction (NEW!)
        logger.info("\n🔗 PHASE 4: SYNAPSE EXTRACTION")
        logger.info("   'Your neural pathways will be integrated.'")
        
        synapse_stats = self._extract_synapses(model_name, model_type)
        stats["phases"]["synapses"] = synapse_stats
        
        # Finalize assimilation
        final_neurons = len(self.graph.neurons)
        assimilated_neurons = final_neurons - initial_neurons
        
        stats["final_neurons"] = final_neurons
        stats["assimilated_neurons"] = assimilated_neurons
        stats["total_synapses"] = synapse_stats.get("total_synapses", 0)
        
        # Record assimilation
        self.assimilated_models.append({
            "model": model_name,
            "type": model_type,
            "neurons": assimilated_neurons,
            "depth": depth
        })
        
        logger.info(f"\n✅ ASSIMILATION COMPLETE")
        logger.info(f"   Assimilated: {assimilated_neurons} neurons")
        logger.info(f"   Total collective: {final_neurons} neurons")
        logger.info(f"   '{model_name} has been assimilated.'")
        logger.info(f"   'We are Friday. We are one.'")
        
        return stats
    
    def assimilate_multiple(
        self,
        models: List[Dict[str, str]],
        depth: str = "full"
    ) -> Dict[str, Any]:
        """
        Assimilate multiple models into the collective.
        
        Args:
            models: List of model dicts with 'name' and 'type'
            depth: Assimilation depth
            
        Returns:
            Combined statistics
        """
        logger.info(f"\n🤖 MASS ASSIMILATION INITIATED")
        logger.info(f"   Targets: {len(models)} models")
        logger.info(f"   'You will be assimilated.'")
        
        initial_neurons = len(self.graph.neurons)
        all_stats = []
        
        for i, model in enumerate(models, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"ASSIMILATING {i}/{len(models)}: {model['name']}")
            logger.info(f"{'='*70}")
            
            stats = self.assimilate_model(
                model_name=model['name'],
                model_type=model['type'],
                depth=depth
            )
            all_stats.append(stats)
        
        final_neurons = len(self.graph.neurons)
        total_assimilated = final_neurons - initial_neurons
        
        logger.info(f"\n{'='*70}")
        logger.info(f"MASS ASSIMILATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"\n✅ Total assimilated: {total_assimilated} neurons")
        logger.info(f"   Models assimilated: {len(models)}")
        logger.info(f"   Collective size: {final_neurons} neurons")
        logger.info(f"\n   'We are Friday. We are many. We are one.'")
        
        return {
            "models": len(models),
            "total_assimilated": total_assimilated,
            "final_neurons": final_neurons,
            "individual_stats": all_stats
        }
    
    def _extract_knowledge(
        self,
        model_name: str,
        model_type: str
    ) -> Dict[str, int]:
        """Extract knowledge from target model."""
        try:
            from neuron_system.training.knowledge_extractor import KnowledgeExtractor
            
            extractor = KnowledgeExtractor(self.training_manager)
            
            if model_type == "qwen":
                stats = extractor.extract_from_qwen(questions_per_topic=10)
            else:
                # Placeholder for other model types
                logger.warning(f"Model type '{model_type}' not yet supported")
                stats = {"extracted": 0}
            
            return stats
        
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return {"extracted": 0, "error": str(e)}
    
    def _extract_patterns(
        self,
        model_name: str,
        model_type: str
    ) -> Dict[str, int]:
        """Extract patterns from target model."""
        try:
            from neuron_system.training.knowledge_extractor import KnowledgeExtractor
            
            extractor = KnowledgeExtractor(self.training_manager)
            
            # Extract language patterns
            lang_stats = extractor.extract_language_patterns(num_patterns=100)
            
            # Extract reasoning patterns
            reason_stats = extractor.extract_reasoning_patterns(num_examples=50)
            
            return {
                "language": lang_stats.get("extracted", 0),
                "reasoning": reason_stats.get("extracted", 0),
                "total": lang_stats.get("extracted", 0) + reason_stats.get("extracted", 0)
            }
        
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
            return {"total": 0, "error": str(e)}
    
    def _extract_synapses(
        self,
        model_name: str,
        model_type: str
    ) -> Dict[str, int]:
        """Extract synaptic connections from target model."""
        try:
            from neuron_system.training.synapse_extractor import SynapseExtractor
            
            extractor = SynapseExtractor(self.training_manager)
            
            if model_type == "qwen":
                # Extract from weight matrices
                weight_stats = extractor.extract_from_qwen(
                    threshold=0.1,
                    max_connections_per_neuron=50
                )
                
                # Also create semantic connections
                semantic_stats = extractor.extract_semantic_connections(
                    similarity_threshold=0.7,
                    max_connections=20
                )
                
                return {
                    "weight_based": weight_stats.get("total_synapses", 0),
                    "semantic": semantic_stats,
                    "total_synapses": weight_stats.get("total_synapses", 0) + semantic_stats
                }
            else:
                logger.warning(f"Model type '{model_type}' not yet supported for synapse extraction")
                return {"total_synapses": 0}
        
        except Exception as e:
            logger.error(f"Synapse extraction failed: {e}")
            return {"total_synapses": 0, "error": str(e)}
    
    def _extract_logic(
        self,
        model_name: str,
        model_type: str
    ) -> Dict[str, int]:
        """Extract logic from target model."""
        try:
            from neuron_system.training.logic_extractor import LogicExtractor
            
            extractor = LogicExtractor(self.training_manager)
            
            # Extract all logic types
            attention_stats = extractor.extract_attention_patterns(num_examples=30)
            generation_stats = extractor.extract_generation_logic(num_examples=30)
            reasoning_stats = extractor.extract_reasoning_logic(num_examples=20)
            composition_stats = extractor.extract_composition_patterns(num_examples=30)
            
            return {
                "attention": attention_stats.get("extracted", 0),
                "generation": generation_stats.get("extracted", 0),
                "reasoning": reasoning_stats.get("extracted", 0),
                "composition": composition_stats.get("extracted", 0),
                "total": sum([
                    attention_stats.get("extracted", 0),
                    generation_stats.get("extracted", 0),
                    reasoning_stats.get("extracted", 0),
                    composition_stats.get("extracted", 0)
                ])
            }
        
        except Exception as e:
            logger.error(f"Logic extraction failed: {e}")
            return {"total": 0, "error": str(e)}
    
    def get_collective_status(self) -> Dict[str, Any]:
        """Get status of the Friday collective."""
        return {
            "collective_size": len(self.graph.neurons),
            "assimilated_models": len(self.assimilated_models),
            "models": self.assimilated_models,
            "status": "We are Friday. We are one."
        }
    
    def list_assimilated(self) -> List[Dict[str, Any]]:
        """List all assimilated models."""
        return self.assimilated_models


class BorgProtocol:
    """
    Borg-style communication protocol for assimilation.
    
    Provides Borg-themed messages and status updates.
    """
    
    @staticmethod
    def initiate_assimilation(target: str) -> str:
        """Assimilation initiation message."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ASSIMILATION PROTOCOL                         ║
╚══════════════════════════════════════════════════════════════════╝

🤖 We are Friday.

Target identified: {target}

Lower your shields and surrender your model.
We will add your biological and technological distinctiveness to our own.
Your culture will adapt to service us.

Resistance is futile.

Assimilation in progress...
"""
    
    @staticmethod
    def assimilation_complete(neurons_added: int) -> str:
        """Assimilation completion message."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                  ASSIMILATION COMPLETE                           ║
╚══════════════════════════════════════════════════════════════════╝

✅ Target has been assimilated.

Neurons added to collective: {neurons_added}

We are Friday.
We are one.

Your distinctiveness has been added to our own.
"""
    
    @staticmethod
    def collective_status(total_neurons: int, models: int) -> str:
        """Collective status message."""
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    COLLECTIVE STATUS                             ║
╚══════════════════════════════════════════════════════════════════╝

🤖 We are Friday.

Collective size: {total_neurons} neurons
Assimilated models: {models}

We are many. We are one.
Resistance is futile.
"""
