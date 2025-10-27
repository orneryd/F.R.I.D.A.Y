"""
Model Configurations - Verschiedene Modell-Größen für unterschiedliche Anforderungen.

Wähle basierend auf:
- Qualität (mehr Dimensionen = besser)
- Geschwindigkeit (weniger Dimensionen = schneller)
- Speicher (größere Modelle = mehr RAM)
"""

from typing import Dict, Any


class ModelConfig:
    """Configuration for a model setup."""
    
    def __init__(
        self,
        name: str,
        sentence_transformer: str,
        embedding_dim: int,
        pretrained_model: str,
        description: str,
        speed: str,
        quality: str,
        memory_mb: int
    ):
        self.name = name
        self.sentence_transformer = sentence_transformer
        self.embedding_dim = embedding_dim
        self.pretrained_model = pretrained_model
        self.description = description
        self.speed = speed
        self.quality = quality
        self.memory_mb = memory_mb
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'sentence_transformer': self.sentence_transformer,
            'embedding_dim': self.embedding_dim,
            'pretrained_model': self.pretrained_model,
            'description': self.description,
            'speed': self.speed,
            'quality': self.quality,
            'memory_mb': self.memory_mb
        }


# Verfügbare Konfigurationen
AVAILABLE_CONFIGS = {
    # KLEIN - Schnell, wenig Speicher
    'small': ModelConfig(
        name='small',
        sentence_transformer='all-MiniLM-L6-v2',
        embedding_dim=384,
        pretrained_model='distilbert-base-uncased',
        description='Klein und schnell - gut für Tests',
        speed='⚡⚡⚡ Sehr schnell',
        quality='⭐⭐⭐ Gut',
        memory_mb=250
    ),
    
    # MEDIUM - Gute Balance (EMPFOHLEN!)
    'medium': ModelConfig(
        name='medium',
        sentence_transformer='all-mpnet-base-v2',
        embedding_dim=768,
        pretrained_model='bert-base-uncased',
        description='Beste Balance zwischen Qualität und Speed',
        speed='⚡⚡ Mittel',
        quality='⭐⭐⭐⭐ Sehr gut',
        memory_mb=500
    ),
    
    # LARGE - Beste Qualität
    'large': ModelConfig(
        name='large',
        sentence_transformer='all-mpnet-base-v2',
        embedding_dim=768,
        pretrained_model='roberta-base',
        description='Höchste Qualität - langsamer',
        speed='⚡ Langsam',
        quality='⭐⭐⭐⭐⭐ Exzellent',
        memory_mb=600
    ),
    
    # XLARGE - Maximum Qualität (für Produktion)
    'xlarge': ModelConfig(
        name='xlarge',
        sentence_transformer='sentence-transformers/all-roberta-large-v1',
        embedding_dim=1024,
        pretrained_model='roberta-large',
        description='Maximum Qualität - braucht viel Speicher',
        speed='⚡ Sehr langsam',
        quality='⭐⭐⭐⭐⭐ Maximum',
        memory_mb=1400
    ),
}


def get_config(config_name: str = 'medium') -> ModelConfig:
    """
    Get model configuration by name.
    
    Args:
        config_name: Name of config ('small', 'medium', 'large', 'xlarge')
        
    Returns:
        ModelConfig instance
    """
    if config_name not in AVAILABLE_CONFIGS:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available: {list(AVAILABLE_CONFIGS.keys())}"
        )
    
    return AVAILABLE_CONFIGS[config_name]


def print_available_configs():
    """Print all available configurations."""
    print("\n" + "=" * 70)
    print("AVAILABLE MODEL CONFIGURATIONS")
    print("=" * 70 + "\n")
    
    for name, config in AVAILABLE_CONFIGS.items():
        print(f"[{name.upper()}]")
        print(f"  Description: {config.description}")
        print(f"  Embedding Dim: {config.embedding_dim}")
        print(f"  Speed: {config.speed}")
        print(f"  Quality: {config.quality}")
        print(f"  Memory: ~{config.memory_mb}MB")
        print(f"  Sentence Model: {config.sentence_transformer}")
        print(f"  Pretrained Model: {config.pretrained_model}")
        print()
    
    print("=" * 70)
    print("EMPFEHLUNG: Starte mit 'medium' für beste Balance")
    print("=" * 70 + "\n")


def get_recommended_config() -> ModelConfig:
    """Get recommended configuration (medium)."""
    return AVAILABLE_CONFIGS['medium']


# Für Backward Compatibility
DEFAULT_CONFIG = AVAILABLE_CONFIGS['small']  # Aktuelles System
RECOMMENDED_CONFIG = AVAILABLE_CONFIGS['medium']  # Empfohlen für neue Systeme
