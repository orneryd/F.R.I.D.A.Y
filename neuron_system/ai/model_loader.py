"""
Model Loader - Einfaches Laden von vortrainierten Modellen in das Neuron System.

Unterstützt:
- Hugging Face Transformers
- Sentence Transformers
- Custom Models
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Lädt vortrainierte Modelle und extrahiert Wissen für Neuronen.
    """
    
    @staticmethod
    def load_huggingface_model(
        model_name: str = "distilbert-base-uncased",
        extract_knowledge: bool = False
    ) -> Dict[str, Any]:
        """
        Lade ein Hugging Face Modell.
        
        Args:
            model_name: Name des Modells
            extract_knowledge: Ob Wissen extrahiert werden soll
            
        Returns:
            Dictionary mit Model-Informationen
        """
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            logger.info(f"Loading Hugging Face model: {model_name}")
            
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Extract model info
            config = model.config
            
            model_info = {
                'model_name': model_name,
                'model_type': config.model_type,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_hidden_layers,
                'num_attention_heads': config.num_attention_heads,
                'vocab_size': config.vocab_size,
                'model': model,
                'tokenizer': tokenizer
            }
            
            logger.info(f"✓ Loaded {model_name}")
            logger.info(f"  Type: {config.model_type}")
            logger.info(f"  Hidden size: {config.hidden_size}")
            logger.info(f"  Layers: {config.num_hidden_layers}")
            logger.info(f"  Attention heads: {config.num_attention_heads}")
            
            # Extract knowledge if requested
            if extract_knowledge:
                knowledge = ModelLoader.extract_model_knowledge(model, tokenizer)
                model_info['extracted_knowledge'] = knowledge
            
            return model_info
            
        except ImportError:
            logger.error("transformers library not installed. Run: pip install transformers torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @staticmethod
    def extract_model_knowledge(
        model,
        tokenizer,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Extrahiere Wissen aus einem vortrainierten Modell.
        
        Dies ist experimentell - extrahiert Token-Embeddings als Basis-Wissen.
        
        Args:
            model: Hugging Face model
            tokenizer: Tokenizer
            num_samples: Anzahl der Token-Samples
            
        Returns:
            Dictionary mit extrahiertem Wissen
        """
        import torch
        
        logger.info(f"Extracting knowledge from model...")
        
        # Get embedding layer
        if hasattr(model, 'embeddings'):
            embeddings = model.embeddings.word_embeddings
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embeddings = model.transformer.wte
        else:
            logger.warning("Could not find embedding layer")
            return {}
        
        # Sample common tokens
        common_tokens = [
            "hello", "world", "computer", "science", "artificial", "intelligence",
            "neural", "network", "learning", "data", "algorithm", "system",
            "question", "answer", "help", "information", "knowledge", "understand"
        ]
        
        token_embeddings = {}
        
        for token in common_tokens:
            try:
                token_id = tokenizer.encode(token, add_special_tokens=False)[0]
                embedding = embeddings.weight[token_id].detach().numpy()
                token_embeddings[token] = embedding
            except Exception as e:
                logger.debug(f"Could not extract embedding for '{token}': {e}")
        
        logger.info(f"✓ Extracted {len(token_embeddings)} token embeddings")
        
        return {
            'token_embeddings': token_embeddings,
            'embedding_dim': embeddings.weight.shape[1],
            'vocab_size': embeddings.weight.shape[0]
        }
    
    @staticmethod
    def create_knowledge_neurons_from_model(
        language_model,
        model_info: Dict[str, Any],
        create_connections: bool = True
    ) -> int:
        """
        Erstelle Knowledge-Neuronen aus extrahiertem Model-Wissen.
        
        Args:
            language_model: LanguageModel instance
            model_info: Model info from load_huggingface_model
            create_connections: Ob Connections erstellt werden sollen
            
        Returns:
            Anzahl der erstellten Neuronen
        """
        if 'extracted_knowledge' not in model_info:
            logger.warning("No extracted knowledge in model_info")
            return 0
        
        knowledge = model_info['extracted_knowledge']
        token_embeddings = knowledge.get('token_embeddings', {})
        
        if not token_embeddings:
            logger.warning("No token embeddings found")
            return 0
        
        logger.info(f"Creating knowledge neurons from {len(token_embeddings)} tokens...")
        
        created_count = 0
        
        for token, embedding in token_embeddings.items():
            try:
                # Create knowledge text
                text = f"Token: {token}"
                
                # Learn as neuron
                neuron_id = language_model.learn(
                    text=text,
                    tags=['pretrained', 'token', model_info['model_name']],
                    create_connections=create_connections
                )
                
                # Override vector with pretrained embedding
                neuron = language_model.graph.get_neuron(neuron_id)
                if neuron:
                    # Resize embedding if needed
                    if len(embedding) != len(neuron.vector):
                        # Pad or truncate
                        if len(embedding) < len(neuron.vector):
                            embedding = np.pad(
                                embedding, 
                                (0, len(neuron.vector) - len(embedding))
                            )
                        else:
                            embedding = embedding[:len(neuron.vector)]
                    
                    neuron.vector = embedding
                
                created_count += 1
                
            except Exception as e:
                logger.error(f"Failed to create neuron for token '{token}': {e}")
        
        logger.info(f"✓ Created {created_count} knowledge neurons from pretrained model")
        
        return created_count
    
    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get list of recommended pretrained models.
        
        Returns:
            Dictionary of model names and descriptions
        """
        return {
            'distilbert-base-uncased': 'Fast, lightweight BERT (66M params)',
            'bert-base-uncased': 'Standard BERT (110M params)',
            'roberta-base': 'RoBERTa base (125M params)',
            'albert-base-v2': 'ALBERT base (12M params, very efficient)',
            'distilgpt2': 'Small GPT-2 (82M params)',
            'gpt2': 'GPT-2 small (124M params)',
        }
    
    @staticmethod
    def print_available_models():
        """Print available models."""
        models = ModelLoader.get_available_models()
        
        print("\n" + "=" * 70)
        print("AVAILABLE PRETRAINED MODELS")
        print("=" * 70)
        
        for name, description in models.items():
            print(f"  • {name}")
            print(f"    {description}")
            print()
        
        print("=" * 70)
        print("Usage: ModelLoader.load_huggingface_model('model-name')")
        print("=" * 70 + "\n")
