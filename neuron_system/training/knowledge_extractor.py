"""
Knowledge Extractor - Extract intelligence from LLMs into Friday's neurons.

This system extracts knowledge, language patterns, and reasoning from
external models (like Qwen3) and stores them as neurons, making Friday
independent of external models.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    """
    Extracts knowledge from LLMs and stores in neuron system.
    
    Process:
    1. Generate responses from LLM for various topics
    2. Extract patterns, knowledge, and reasoning
    3. Store as neurons in Friday's brain
    4. Friday becomes independent of external models
    """
    
    def __init__(self, training_manager):
        """
        Initialize knowledge extractor.
        
        Args:
            training_manager: TrainingManager instance
        """
        self.training_manager = training_manager
        self.language_model = training_manager.language_model
        self.graph = training_manager.graph
    
    def extract_from_qwen(
        self,
        topics: Optional[List[str]] = None,
        questions_per_topic: int = 10
    ) -> Dict[str, int]:
        """
        Extract knowledge from Qwen3 model with BATCH TRAINING.
        
        Args:
            topics: List of topics to extract knowledge about
            questions_per_topic: Number of questions per topic
            
        Returns:
            Statistics dict
        """
        if topics is None:
            topics = self._get_default_topics()
        
        logger.info(f"Extracting knowledge from Qwen3 for {len(topics)} topics...")
        logger.info("Using BATCH mode for faster training...")
        
        # Load Qwen model
        try:
            from neuron_system.ai.local_model import LocalModelManager
            qwen = LocalModelManager(model_dir="models")
            
            if not qwen.load_model("Qwen/Qwen3-0.6B"):
                logger.error("Failed to load Qwen model")
                return {"error": "Model not loaded"}
            
            logger.info("✓ Qwen3 model loaded")
        
        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")
            return {"error": str(e)}
        
        # Extract knowledge in BATCHES
        total_extracted = 0
        batch_data = []  # Collect all data first
        
        # Phase 1: Generate all Q&A pairs (fast)
        logger.info("\n📊 Phase 1: Generating Q&A pairs...")
        for topic in topics:
            logger.info(f"  Topic: {topic}")
            
            # Generate questions about topic
            questions = self._generate_questions(topic, questions_per_topic)
            
            # Get answers from Qwen
            for question in questions:
                try:
                    # Generate answer
                    prompt = f"Question: {question}\nAnswer:"
                    answer = qwen.generate_response(
                        prompt=prompt,
                        max_length=100,
                        temperature=0.7
                    )
                    
                    if answer and len(answer) > 20:
                        # Add to batch
                        qa_text = f"Question: {question}\nAnswer: {answer}"
                        tags = ['extracted', 'qwen', topic.lower(), 'qa']
                        
                        batch_data.append({
                            'text': qa_text,
                            'tags': tags
                        })
                        
                        logger.debug(f"    ✓ Generated: {question[:50]}...")
                
                except Exception as e:
                    logger.debug(f"    ✗ Failed: {e}")
        
        # Phase 2: SMART Batch training (adaptive batch sizes)
        logger.info(f"\n💾 Phase 2: SMART Batch training {len(batch_data)} items...")
        
        # Process in adaptive batches
        batch_size = 50  # Start with 50 items per batch
        total_extracted = 0
        
        for batch_start in range(0, len(batch_data), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_data))
            current_batch = batch_data[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                # Learn batch at once
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=False  # No connections during training
                )
                
                total_extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: Trained {len(neuron_ids)} neurons ({batch_start+len(neuron_ids)}/{len(batch_data)})")
                
            except Exception as e:
                logger.warning(f"Batch training failed, trying individual: {e}")
                
                # Fallback: train individually for this batch only
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'], create_connections=False)
                        total_extracted += 1
                    except Exception as e2:
                        logger.debug(f"    Failed: {e2}")
        
        logger.info(f"✓ Extracted {total_extracted} knowledge pieces")
        
        return {
            "extracted": total_extracted,
            "topics": len(topics),
            "total_neurons": len(self.graph.neurons)
        }
    
    def extract_language_patterns(
        self,
        num_patterns: int = 100
    ) -> Dict[str, int]:
        """
        Extract language patterns with BATCH TRAINING.
        
        Args:
            num_patterns: Number of patterns to extract
            
        Returns:
            Statistics dict
        """
        logger.info(f"Extracting {num_patterns} language patterns from Qwen...")
        logger.info("Using BATCH mode...")
        
        # Load Qwen
        try:
            from neuron_system.ai.local_model import LocalModelManager
            qwen = LocalModelManager(model_dir="models")
            qwen.load_model("Qwen/Qwen3-0.6B")
        except Exception as e:
            logger.error(f"Failed to load Qwen: {e}")
            return {"error": str(e)}
        
        # Phase 1: Generate all patterns
        logger.info("📊 Phase 1: Generating patterns...")
        prompts = self._get_conversation_prompts()
        batch_data = []
        
        for prompt in prompts[:num_patterns]:
            try:
                response = qwen.generate_response(
                    prompt=prompt,
                    max_length=80,
                    temperature=0.8
                )
                
                if response and len(response) > 15:
                    batch_data.append({
                        'text': response,
                        'tags': ['pattern', 'conversation', 'extracted']
                    })
            
            except Exception as e:
                logger.debug(f"Failed to generate pattern: {e}")
        
        # Phase 2: SMART Batch train
        logger.info(f"💾 Phase 2: SMART Batch training {len(batch_data)} patterns...")
        extracted = 0
        batch_size = 25  # Process 25 patterns at once
        
        for batch_start in range(0, len(batch_data), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_data))
            current_batch = batch_data[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=True  # Create connections for patterns
                )
                
                extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: {len(neuron_ids)} patterns ({extracted}/{len(batch_data)})")
                
            except Exception as e:
                logger.warning(f"Batch failed, using individual: {e}")
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'])
                        extracted += 1
                    except:
                        pass
        
        # Save once at the end
        logger.info("💾 Saving...")
        self.graph.save()
        
        return {
            "extracted": extracted,
            "total_neurons": len(self.graph.neurons)
        }
    
    def extract_reasoning_patterns(
        self,
        num_examples: int = 50
    ) -> Dict[str, int]:
        """
        Extract reasoning patterns from Qwen.
        
        Args:
            num_examples: Number of reasoning examples
            
        Returns:
            Statistics dict
        """
        logger.info(f"Extracting {num_examples} reasoning patterns...")
        
        # Load Qwen
        try:
            from neuron_system.ai.local_model import LocalModelManager
            qwen = LocalModelManager(model_dir="models")
            qwen.load_model("Qwen/Qwen3-0.6B")
        except Exception as e:
            return {"error": str(e)}
        
        # Phase 1: Generate all reasoning examples
        logger.info("📊 Phase 1: Generating reasoning examples...")
        reasoning_prompts = self._get_reasoning_prompts()
        batch_data = []
        
        for prompt in reasoning_prompts[:num_examples]:
            try:
                response = qwen.generate_response(
                    prompt=prompt,
                    max_length=150,
                    temperature=0.7
                )
                
                if response and len(response) > 30:
                    batch_data.append({
                        'text': response,
                        'tags': ['reasoning', 'logic', 'extracted', 'qwen']
                    })
            
            except Exception as e:
                logger.debug(f"Failed: {e}")
        
        # Phase 2: SMART Batch train
        logger.info(f"💾 Phase 2: SMART Batch training {len(batch_data)} reasoning patterns...")
        extracted = 0
        batch_size = 20  # Process 20 reasoning patterns at once
        
        for batch_start in range(0, len(batch_data), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_data))
            current_batch = batch_data[batch_start:batch_end]
            
            try:
                texts = [item['text'] for item in current_batch]
                tags_list = [item['tags'] for item in current_batch]
                
                neuron_ids = self.language_model.learn_batch(
                    texts=texts,
                    tags_list=tags_list,
                    create_connections=True
                )
                
                extracted += len(neuron_ids)
                logger.info(f"✓ Batch {batch_start//batch_size + 1}: {len(neuron_ids)} patterns ({extracted}/{len(batch_data)})")
                
            except Exception as e:
                logger.warning(f"Batch failed: {e}")
                for item in current_batch:
                    try:
                        self.language_model.learn(text=item['text'], tags=item['tags'])
                        extracted += 1
                    except:
                        pass
        
        self.graph.save()
        
        return {
            "extracted": extracted,
            "total_neurons": len(self.graph.neurons)
        }
    
    def _get_default_topics(self) -> List[str]:
        """Get default topics for knowledge extraction."""
        return [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "programming",
            "python",
            "data science",
            "mathematics",
            "physics",
            "chemistry",
            "biology",
            "history",
            "geography",
            "literature",
            "philosophy",
            "psychology",
            "technology",
            "science",
            "computers",
            "internet",
            "software"
        ]
    
    def _generate_questions(self, topic: str, count: int) -> List[str]:
        """Generate questions about a topic."""
        question_templates = [
            f"What is {topic}?",
            f"How does {topic} work?",
            f"Why is {topic} important?",
            f"Explain {topic} in simple terms",
            f"What are the benefits of {topic}?",
            f"What are the challenges of {topic}?",
            f"How can I learn {topic}?",
            f"What are examples of {topic}?",
            f"Who invented {topic}?",
            f"When was {topic} created?",
            f"Where is {topic} used?",
            f"What is the history of {topic}?",
            f"What is the future of {topic}?",
            f"How is {topic} different from similar concepts?",
            f"What are the applications of {topic}?"
        ]
        
        return question_templates[:count]
    
    def _get_conversation_prompts(self) -> List[str]:
        """Get prompts for extracting conversation patterns."""
        return [
            "Hello! How are you?",
            "Can you help me?",
            "Thank you!",
            "I don't understand",
            "Could you explain that?",
            "That's interesting!",
            "I agree",
            "I disagree because",
            "What do you think about",
            "Tell me more about",
            "How can I",
            "Why should I",
            "What if",
            "I'm confused about",
            "That makes sense",
            "Good point!",
            "I see what you mean",
            "Let me think about that",
            "That's a good question",
            "Here's what I think"
        ]
    
    def _get_reasoning_prompts(self) -> List[str]:
        """Get prompts for extracting reasoning patterns."""
        return [
            "If A is true and B is true, then",
            "The reason why this happens is",
            "We can conclude that",
            "This suggests that",
            "Based on this evidence,",
            "Therefore,",
            "It follows that",
            "This implies",
            "As a result,",
            "Consequently,",
            "Given that",
            "Assuming that",
            "If we consider",
            "Taking into account",
            "From this we can infer"
        ]
