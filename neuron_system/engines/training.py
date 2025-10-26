"""
Training Engine for live learning and neuron/synapse modification.

This module provides the TrainingEngine class that handles direct manipulation
of neurons and synapses without GPU-intensive training processes.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np

from neuron_system.core.graph import NeuronGraph
from neuron_system.core.neuron import Neuron
from neuron_system.core.synapse import Synapse
from neuron_system.core.vector3d import Vector3D

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class TrainingOperation:
    """
    Represents a single training operation for audit trail and rollback.
    
    Attributes:
        operation_id: Unique identifier for the operation
        operation_type: Type of operation (adjust_neuron, strengthen_synapse, etc.)
        timestamp: When the operation was performed
        target_id: ID of the neuron or synapse being modified
        before_state: State before the operation
        after_state: State after the operation
        metadata: Additional operation metadata
    """
    operation_id: int
    operation_type: str
    timestamp: datetime
    target_id: UUID
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainingEngine:
    """
    Engine for training the neuron network through direct manipulation.
    
    Provides methods for adjusting neuron vectors, modifying synapse weights,
    and learning from usage patterns without GPU-intensive processes.
    
    Attributes:
        graph: The neuron graph to train
        operation_log: History of all training operations
        learning_rate: Default learning rate for vector adjustments
        decay_rate: Default decay rate for synapse weakening
    """
    
    def __init__(
        self,
        neuron_graph: NeuronGraph,
        learning_rate: float = 0.1,
        decay_rate: float = 0.001
    ):
        """
        Initialize TrainingEngine.
        
        Args:
            neuron_graph: The neuron graph to train
            learning_rate: Default learning rate for adjustments (0.0 to 1.0)
            decay_rate: Default decay rate for synapse weakening
            
        Requirements: 4.1, 4.2, 4.3, 4.5
        """
        self.graph = neuron_graph
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # Operation logging for audit trail
        self.operation_log: List[TrainingOperation] = []
        self._operation_counter = 0
        
        # Configuration
        self._max_log_size = 10000  # Maximum operations to keep in memory
        self._enable_logging = True
        
        # Validation settings
        self._vector_dimension = 384  # Expected vector dimension
        self._min_vector_value = -10.0  # Minimum allowed vector value
        self._max_vector_value = 10.0  # Maximum allowed vector value
        
        logger.info("TrainingEngine initialized")
    
    def adjust_neuron(
        self,
        neuron_id: UUID,
        target_vector: np.ndarray,
        learning_rate: Optional[float] = None,
        validate: bool = True
    ) -> bool:
        """
        Incrementally adjust a neuron's vector toward a target.
        
        Uses vector arithmetic to move the neuron vector closer to the target:
        new_vector = current_vector + learning_rate * (target - current)
        
        Args:
            neuron_id: UUID of the neuron to adjust
            target_vector: Target vector to move toward
            learning_rate: Learning rate (uses default if None)
            validate: Whether to validate the update before applying
            
        Returns:
            True if adjustment was successful, False otherwise
            
        Requirements: 4.1, 4.2, 4.3, 4.5
        """
        # Get neuron
        neuron = self.graph.get_neuron(neuron_id)
        if neuron is None:
            logger.warning(f"Neuron not found: {neuron_id}")
            return False
        
        # Check if neuron has a vector
        if neuron.vector is None or len(neuron.vector) == 0:
            logger.warning(f"Neuron {neuron_id} has no vector to adjust")
            return False
        
        # Use default learning rate if not provided
        if learning_rate is None:
            learning_rate = self.learning_rate
        
        # Validate learning rate
        if learning_rate < 0.0 or learning_rate > 1.0:
            logger.error(f"Invalid learning rate: {learning_rate}, must be in [0.0, 1.0]")
            return False
        
        # Store before state for logging
        before_state = {
            "vector": neuron.vector.copy(),
            "modified_at": neuron.modified_at
        }
        
        try:
            # Validate target vector dimensions
            if not self._validate_vector_dimensions(target_vector):
                logger.error(
                    f"Invalid target vector dimensions: {len(target_vector)}, "
                    f"expected {self._vector_dimension}"
                )
                return False
            
            # Calculate new vector
            current_vector = neuron.vector
            delta = target_vector - current_vector
            new_vector = current_vector + learning_rate * delta
            
            # Validate new vector is within bounds
            if validate and not self._validate_vector_bounds(new_vector):
                logger.error("New vector exceeds embedding space bounds")
                return False
            
            # Apply the adjustment
            neuron.vector = new_vector
            neuron.modified_at = datetime.now()
            
            # Store after state
            after_state = {
                "vector": neuron.vector.copy(),
                "modified_at": neuron.modified_at
            }
            
            # Log operation
            self._log_operation(
                operation_type="adjust_neuron",
                target_id=neuron_id,
                before_state=before_state,
                after_state=after_state,
                metadata={
                    "learning_rate": learning_rate,
                    "delta_magnitude": float(np.linalg.norm(delta))
                }
            )
            
            logger.debug(
                f"Adjusted neuron {neuron_id} with learning_rate={learning_rate}, "
                f"delta_magnitude={np.linalg.norm(delta):.4f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to adjust neuron {neuron_id}: {str(e)}", exc_info=True)
            return False
    
    def _validate_vector_dimensions(self, vector: np.ndarray) -> bool:
        """
        Validate that vector has correct dimensions.
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if dimensions are valid
        """
        if vector is None:
            return False
        
        if len(vector) != self._vector_dimension:
            return False
        
        return True
    
    def _validate_vector_bounds(self, vector: np.ndarray) -> bool:
        """
        Validate that vector values are within embedding space bounds.
        
        Args:
            vector: Vector to validate
            
        Returns:
            True if all values are within bounds
        """
        if vector is None:
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            logger.error("Vector contains NaN or Inf values")
            return False
        
        # Check bounds
        if np.any(vector < self._min_vector_value) or np.any(vector > self._max_vector_value):
            logger.warning(
                f"Vector values outside bounds [{self._min_vector_value}, {self._max_vector_value}]"
            )
            return False
        
        return True
    
    def _log_operation(
        self,
        operation_type: str,
        target_id: UUID,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ):
        """
        Log a training operation for audit trail.
        
        Args:
            operation_type: Type of operation
            target_id: ID of the target neuron or synapse
            before_state: State before the operation
            after_state: State after the operation
            metadata: Additional metadata
        """
        if not self._enable_logging:
            return
        
        operation = TrainingOperation(
            operation_id=self._operation_counter,
            operation_type=operation_type,
            timestamp=datetime.now(),
            target_id=target_id,
            before_state=before_state,
            after_state=after_state,
            metadata=metadata or {}
        )
        
        self.operation_log.append(operation)
        self._operation_counter += 1
        
        # Trim log if it exceeds max size
        if len(self.operation_log) > self._max_log_size:
            # Remove oldest 10% of operations
            trim_count = self._max_log_size // 10
            self.operation_log = self.operation_log[trim_count:]
            logger.debug(f"Trimmed operation log, removed {trim_count} oldest operations")
    
    def get_operation_log(
        self,
        limit: Optional[int] = None,
        operation_type: Optional[str] = None
    ) -> List[TrainingOperation]:
        """
        Get training operation history.
        
        Args:
            limit: Maximum number of operations to return (most recent)
            operation_type: Filter by operation type
            
        Returns:
            List of training operations
        """
        operations = self.operation_log
        
        # Filter by type if specified
        if operation_type:
            operations = [op for op in operations if op.operation_type == operation_type]
        
        # Limit to most recent if specified
        if limit:
            operations = operations[-limit:]
        
        return operations
    
    def clear_operation_log(self):
        """Clear the operation log."""
        self.operation_log.clear()
        logger.info("Operation log cleared")
    
    def set_learning_rate(self, learning_rate: float):
        """
        Set the default learning rate.
        
        Args:
            learning_rate: New learning rate (0.0 to 1.0)
        """
        if learning_rate < 0.0 or learning_rate > 1.0:
            raise ValueError(f"Learning rate must be in [0.0, 1.0], got {learning_rate}")
        
        self.learning_rate = learning_rate
        logger.info(f"Learning rate set to {learning_rate}")
    
    def set_decay_rate(self, decay_rate: float):
        """
        Set the default decay rate.
        
        Args:
            decay_rate: New decay rate
        """
        if decay_rate < 0.0:
            raise ValueError(f"Decay rate must be non-negative, got {decay_rate}")
        
        self.decay_rate = decay_rate
        logger.info(f"Decay rate set to {decay_rate}")
    
    def enable_logging(self, enabled: bool = True):
        """
        Enable or disable operation logging.
        
        Args:
            enabled: Whether to enable logging
        """
        self._enable_logging = enabled
        logger.info(f"Operation logging {'enabled' if enabled else 'disabled'}")
    
    def strengthen_synapse(
        self,
        synapse_id: UUID,
        delta: Optional[float] = None,
        validate: bool = True
    ) -> bool:
        """
        Strengthen a synapse by increasing its weight (Hebbian learning).
        
        Increases the synapse weight, clamping to maximum of 1.0.
        
        Args:
            synapse_id: UUID of the synapse to strengthen
            delta: Amount to increase weight (uses 0.01 if None)
            validate: Whether to validate the update before applying
            
        Returns:
            True if strengthening was successful, False otherwise
            
        Requirements: 9.2, 9.3, 9.4
        """
        # Get synapse
        synapse = self.graph.get_synapse(synapse_id)
        if synapse is None:
            logger.warning(f"Synapse not found: {synapse_id}")
            return False
        
        # Use default delta if not provided
        if delta is None:
            delta = 0.01
        
        # Validate delta
        if delta < 0.0:
            logger.error(f"Delta must be non-negative for strengthening, got {delta}")
            return False
        
        # Store before state
        before_state = {
            "weight": synapse.weight,
            "modified_at": synapse.modified_at
        }
        
        try:
            # Calculate new weight
            new_weight = min(1.0, synapse.weight + delta)
            
            # Apply the change
            synapse.weight = new_weight
            synapse.modified_at = datetime.now()
            
            # Store after state
            after_state = {
                "weight": synapse.weight,
                "modified_at": synapse.modified_at
            }
            
            # Log operation
            self._log_operation(
                operation_type="strengthen_synapse",
                target_id=synapse_id,
                before_state=before_state,
                after_state=after_state,
                metadata={
                    "delta": delta,
                    "weight_change": new_weight - before_state["weight"]
                }
            )
            
            logger.debug(
                f"Strengthened synapse {synapse_id}: "
                f"{before_state['weight']:.4f} -> {new_weight:.4f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to strengthen synapse {synapse_id}: {str(e)}", exc_info=True)
            return False
    
    def weaken_synapse(
        self,
        synapse_id: UUID,
        delta: Optional[float] = None,
        validate: bool = True
    ) -> bool:
        """
        Weaken a synapse by decreasing its weight (decay).
        
        Decreases the synapse weight, clamping to minimum of -1.0.
        If weight reaches 0.0, the synapse is marked for deletion.
        
        Args:
            synapse_id: UUID of the synapse to weaken
            delta: Amount to decrease weight (uses decay_rate if None)
            validate: Whether to validate the update before applying
            
        Returns:
            True if weakening was successful, False otherwise
            
        Requirements: 9.2, 9.3, 9.4
        """
        # Get synapse
        synapse = self.graph.get_synapse(synapse_id)
        if synapse is None:
            logger.warning(f"Synapse not found: {synapse_id}")
            return False
        
        # Use default delta if not provided
        if delta is None:
            delta = self.decay_rate
        
        # Validate delta
        if delta < 0.0:
            logger.error(f"Delta must be non-negative for weakening, got {delta}")
            return False
        
        # Store before state
        before_state = {
            "weight": synapse.weight,
            "modified_at": synapse.modified_at
        }
        
        try:
            # Calculate new weight
            new_weight = max(-1.0, synapse.weight - delta)
            
            # Apply the change
            synapse.weight = new_weight
            synapse.modified_at = datetime.now()
            
            # Check if synapse should be deleted (weight near zero)
            should_delete = abs(new_weight) < 0.01
            
            # Store after state
            after_state = {
                "weight": synapse.weight,
                "modified_at": synapse.modified_at,
                "marked_for_deletion": should_delete
            }
            
            # Log operation
            self._log_operation(
                operation_type="weaken_synapse",
                target_id=synapse_id,
                before_state=before_state,
                after_state=after_state,
                metadata={
                    "delta": delta,
                    "weight_change": new_weight - before_state["weight"],
                    "should_delete": should_delete
                }
            )
            
            logger.debug(
                f"Weakened synapse {synapse_id}: "
                f"{before_state['weight']:.4f} -> {new_weight:.4f}"
                f"{' (marked for deletion)' if should_delete else ''}"
            )
            
            # Automatically delete if weight is effectively zero
            if should_delete:
                self._delete_synapse(synapse_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to weaken synapse {synapse_id}: {str(e)}", exc_info=True)
            return False
    
    def _delete_synapse(self, synapse_id: UUID) -> bool:
        """
        Delete a synapse from the graph.
        
        Args:
            synapse_id: UUID of the synapse to delete
            
        Returns:
            True if deletion was successful
        """
        synapse = self.graph.get_synapse(synapse_id)
        if synapse is None:
            return False
        
        # Store before state
        before_state = {
            "synapse": synapse.to_dict()
        }
        
        # Remove from graph
        success = self.graph.remove_synapse(synapse_id)
        
        if success:
            # Log operation
            self._log_operation(
                operation_type="delete_synapse",
                target_id=synapse_id,
                before_state=before_state,
                after_state={"deleted": True},
                metadata={
                    "reason": "weight_reached_zero",
                    "final_weight": synapse.weight
                }
            )
            
            logger.info(f"Deleted synapse {synapse_id} (weight reached zero)")
        
        return success
    
    def adjust_synapse_weight(
        self,
        synapse_id: UUID,
        new_weight: float,
        validate: bool = True
    ) -> bool:
        """
        Directly set a synapse weight to a specific value.
        
        Args:
            synapse_id: UUID of the synapse to modify
            new_weight: New weight value (-1.0 to 1.0)
            validate: Whether to validate the weight
            
        Returns:
            True if adjustment was successful, False otherwise
            
        Requirements: 9.2, 9.3, 9.4
        """
        # Get synapse
        synapse = self.graph.get_synapse(synapse_id)
        if synapse is None:
            logger.warning(f"Synapse not found: {synapse_id}")
            return False
        
        # Validate weight
        if validate and (new_weight < -1.0 or new_weight > 1.0):
            logger.error(f"Invalid weight: {new_weight}, must be in [-1.0, 1.0]")
            return False
        
        # Store before state
        before_state = {
            "weight": synapse.weight,
            "modified_at": synapse.modified_at
        }
        
        try:
            # Clamp weight to valid range
            new_weight = max(-1.0, min(1.0, new_weight))
            
            # Apply the change
            synapse.weight = new_weight
            synapse.modified_at = datetime.now()
            
            # Check if synapse should be deleted
            should_delete = abs(new_weight) < 0.01
            
            # Store after state
            after_state = {
                "weight": synapse.weight,
                "modified_at": synapse.modified_at,
                "marked_for_deletion": should_delete
            }
            
            # Log operation
            self._log_operation(
                operation_type="adjust_synapse_weight",
                target_id=synapse_id,
                before_state=before_state,
                after_state=after_state,
                metadata={
                    "weight_change": new_weight - before_state["weight"],
                    "should_delete": should_delete
                }
            )
            
            logger.debug(
                f"Adjusted synapse {synapse_id} weight: "
                f"{before_state['weight']:.4f} -> {new_weight:.4f}"
            )
            
            # Automatically delete if weight is effectively zero
            if should_delete:
                self._delete_synapse(synapse_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to adjust synapse {synapse_id}: {str(e)}", exc_info=True)
            return False
    
    def apply_usage_based_learning(
        self,
        synapse_id: UUID,
        strengthen_delta: float = 0.01
    ) -> bool:
        """
        Apply automatic weight strengthening based on synapse usage.
        
        Called when a synapse is traversed during activation propagation.
        Increments usage counter and strengthens the synapse.
        
        Args:
            synapse_id: UUID of the synapse that was traversed
            strengthen_delta: Amount to strengthen the synapse
            
        Returns:
            True if learning was applied successfully
            
        Requirements: 9.1, 9.5
        """
        # Get synapse
        synapse = self.graph.get_synapse(synapse_id)
        if synapse is None:
            logger.warning(f"Synapse not found: {synapse_id}")
            return False
        
        try:
            # Mark synapse as traversed (increments usage counter)
            synapse.traverse()
            
            # Strengthen based on usage
            success = self.strengthen_synapse(synapse_id, delta=strengthen_delta)
            
            if success:
                logger.debug(
                    f"Applied usage-based learning to synapse {synapse_id}, "
                    f"usage_count={synapse.usage_count}"
                )
            
            return success
            
        except Exception as e:
            logger.error(
                f"Failed to apply usage-based learning to synapse {synapse_id}: {str(e)}",
                exc_info=True
            )
            return False
    
    def apply_time_based_decay(
        self,
        time_threshold_seconds: float = 86400,  # 24 hours
        decay_delta: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply time-based decay to unused synapses.
        
        Weakens synapses that haven't been traversed recently.
        
        Args:
            time_threshold_seconds: Time threshold for considering a synapse unused
            decay_delta: Amount to decay (uses decay_rate if None)
            
        Returns:
            Dictionary with decay statistics
            
        Requirements: 9.1, 9.5
        """
        if decay_delta is None:
            decay_delta = self.decay_rate
        
        current_time = datetime.now()
        decayed_count = 0
        deleted_count = 0
        skipped_count = 0
        
        # Iterate through all synapses
        for synapse_id, synapse in list(self.graph.synapses.items()):
            # Skip if synapse was never traversed
            if synapse.last_traversed is None:
                skipped_count += 1
                continue
            
            # Calculate time since last traversal
            time_since_traversal = (current_time - synapse.last_traversed).total_seconds()
            
            # Apply decay if threshold exceeded
            if time_since_traversal > time_threshold_seconds:
                # Store weight before decay
                weight_before = synapse.weight
                
                # Weaken the synapse
                success = self.weaken_synapse(synapse_id, delta=decay_delta)
                
                if success:
                    decayed_count += 1
                    
                    # Check if synapse was deleted
                    if self.graph.get_synapse(synapse_id) is None:
                        deleted_count += 1
                        logger.debug(
                            f"Synapse {synapse_id} deleted due to time-based decay "
                            f"(unused for {time_since_traversal:.0f}s)"
                        )
        
        stats = {
            "decayed_count": decayed_count,
            "deleted_count": deleted_count,
            "skipped_count": skipped_count,
            "time_threshold_seconds": time_threshold_seconds,
            "decay_delta": decay_delta
        }
        
        logger.info(
            f"Applied time-based decay: {decayed_count} synapses decayed, "
            f"{deleted_count} deleted, {skipped_count} skipped"
        )
        
        return stats
    
    def apply_batch_usage_learning(
        self,
        synapse_ids: List[UUID],
        strengthen_delta: float = 0.01
    ) -> Dict[str, Any]:
        """
        Apply usage-based learning to multiple synapses in batch.
        
        More efficient than calling apply_usage_based_learning repeatedly.
        
        Args:
            synapse_ids: List of synapse UUIDs that were traversed
            strengthen_delta: Amount to strengthen each synapse
            
        Returns:
            Dictionary with batch learning statistics
            
        Requirements: 9.1, 9.5
        """
        success_count = 0
        failed_count = 0
        
        for synapse_id in synapse_ids:
            success = self.apply_usage_based_learning(synapse_id, strengthen_delta)
            if success:
                success_count += 1
            else:
                failed_count += 1
        
        stats = {
            "total_synapses": len(synapse_ids),
            "success_count": success_count,
            "failed_count": failed_count,
            "strengthen_delta": strengthen_delta
        }
        
        logger.info(
            f"Applied batch usage learning: {success_count} successful, "
            f"{failed_count} failed"
        )
        
        return stats
    
    def configure_automatic_learning(
        self,
        enable_usage_learning: bool = True,
        usage_strengthen_delta: float = 0.01,
        enable_time_decay: bool = True,
        time_decay_threshold: float = 86400,
        time_decay_delta: Optional[float] = None
    ):
        """
        Configure automatic learning parameters.
        
        Args:
            enable_usage_learning: Whether to enable usage-based strengthening
            usage_strengthen_delta: Delta for usage-based strengthening
            enable_time_decay: Whether to enable time-based decay
            time_decay_threshold: Time threshold for decay (seconds)
            time_decay_delta: Delta for time-based decay (uses decay_rate if None)
        """
        self._auto_learning_config = {
            "enable_usage_learning": enable_usage_learning,
            "usage_strengthen_delta": usage_strengthen_delta,
            "enable_time_decay": enable_time_decay,
            "time_decay_threshold": time_decay_threshold,
            "time_decay_delta": time_decay_delta or self.decay_rate
        }
        
        logger.info(f"Configured automatic learning: {self._auto_learning_config}")
    
    def get_automatic_learning_config(self) -> Dict[str, Any]:
        """
        Get current automatic learning configuration.
        
        Returns:
            Dictionary with automatic learning settings
        """
        if not hasattr(self, '_auto_learning_config'):
            # Return default configuration
            return {
                "enable_usage_learning": True,
                "usage_strengthen_delta": 0.01,
                "enable_time_decay": True,
                "time_decay_threshold": 86400,
                "time_decay_delta": self.decay_rate
            }
        
        return self._auto_learning_config.copy()
    
    def rollback_operation(self, operation_id: int) -> bool:
        """
        Rollback a specific training operation.
        
        Restores the state before the operation was applied.
        
        Args:
            operation_id: ID of the operation to rollback
            
        Returns:
            True if rollback was successful, False otherwise
            
        Requirements: 4.4, 4.5
        """
        # Find the operation
        operation = None
        for op in self.operation_log:
            if op.operation_id == operation_id:
                operation = op
                break
        
        if operation is None:
            logger.warning(f"Operation not found: {operation_id}")
            return False
        
        try:
            # Rollback based on operation type
            if operation.operation_type == "adjust_neuron":
                return self._rollback_adjust_neuron(operation)
            elif operation.operation_type in ["strengthen_synapse", "weaken_synapse", "adjust_synapse_weight"]:
                return self._rollback_synapse_weight(operation)
            elif operation.operation_type == "delete_synapse":
                return self._rollback_delete_synapse(operation)
            else:
                logger.warning(f"Unknown operation type for rollback: {operation.operation_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to rollback operation {operation_id}: {str(e)}", exc_info=True)
            return False
    
    def _rollback_adjust_neuron(self, operation: TrainingOperation) -> bool:
        """
        Rollback a neuron vector adjustment.
        
        Args:
            operation: The operation to rollback
            
        Returns:
            True if rollback was successful
        """
        neuron = self.graph.get_neuron(operation.target_id)
        if neuron is None:
            logger.warning(f"Cannot rollback: neuron {operation.target_id} not found")
            return False
        
        # Restore previous vector
        before_vector = operation.before_state.get("vector")
        if before_vector is None:
            logger.error("Cannot rollback: no before state vector")
            return False
        
        neuron.vector = np.array(before_vector)
        neuron.modified_at = operation.before_state.get("modified_at", datetime.now())
        
        logger.info(f"Rolled back neuron adjustment for {operation.target_id}")
        return True
    
    def _rollback_synapse_weight(self, operation: TrainingOperation) -> bool:
        """
        Rollback a synapse weight modification.
        
        Args:
            operation: The operation to rollback
            
        Returns:
            True if rollback was successful
        """
        synapse = self.graph.get_synapse(operation.target_id)
        if synapse is None:
            logger.warning(f"Cannot rollback: synapse {operation.target_id} not found")
            return False
        
        # Restore previous weight
        before_weight = operation.before_state.get("weight")
        if before_weight is None:
            logger.error("Cannot rollback: no before state weight")
            return False
        
        synapse.weight = before_weight
        synapse.modified_at = operation.before_state.get("modified_at", datetime.now())
        
        logger.info(
            f"Rolled back synapse weight modification for {operation.target_id}: "
            f"weight={before_weight:.4f}"
        )
        return True
    
    def _rollback_delete_synapse(self, operation: TrainingOperation) -> bool:
        """
        Rollback a synapse deletion by recreating it.
        
        Args:
            operation: The operation to rollback
            
        Returns:
            True if rollback was successful
        """
        # Get synapse data from before state
        synapse_data = operation.before_state.get("synapse")
        if synapse_data is None:
            logger.error("Cannot rollback: no synapse data in before state")
            return False
        
        try:
            # Recreate the synapse
            synapse = Synapse.from_dict(synapse_data)
            self.graph.add_synapse(synapse)
            
            logger.info(f"Rolled back synapse deletion for {operation.target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recreate synapse: {str(e)}", exc_info=True)
            return False
    
    def rollback_last_n_operations(self, n: int) -> Dict[str, Any]:
        """
        Rollback the last N operations.
        
        Args:
            n: Number of operations to rollback
            
        Returns:
            Dictionary with rollback statistics
            
        Requirements: 4.4, 4.5
        """
        if n <= 0:
            return {"rolled_back": 0, "failed": 0}
        
        # Get last n operations
        operations_to_rollback = self.operation_log[-n:]
        
        success_count = 0
        failed_count = 0
        
        # Rollback in reverse order (most recent first)
        for operation in reversed(operations_to_rollback):
            success = self.rollback_operation(operation.operation_id)
            if success:
                success_count += 1
            else:
                failed_count += 1
        
        stats = {
            "rolled_back": success_count,
            "failed": failed_count,
            "total_attempted": n
        }
        
        logger.info(
            f"Rolled back {success_count} operations, {failed_count} failed"
        )
        
        return stats
    
    def begin_transaction(self) -> 'TrainingTransaction':
        """
        Begin a training transaction for batch operations.
        
        Returns:
            TrainingTransaction context manager
            
        Requirements: 4.4, 4.5
        """
        return TrainingTransaction(self)
    
    def validate_operation(
        self,
        operation_type: str,
        target_id: UUID,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a training operation before applying it.
        
        Args:
            operation_type: Type of operation to validate
            target_id: ID of the target neuron or synapse
            **kwargs: Additional operation parameters
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Requirements: 4.4, 4.5
        """
        if operation_type == "adjust_neuron":
            neuron = self.graph.get_neuron(target_id)
            if neuron is None:
                return False, f"Neuron not found: {target_id}"
            
            target_vector = kwargs.get("target_vector")
            if target_vector is None:
                return False, "Missing target_vector parameter"
            
            if not self._validate_vector_dimensions(target_vector):
                return False, f"Invalid vector dimensions: {len(target_vector)}"
            
            if not self._validate_vector_bounds(target_vector):
                return False, "Target vector exceeds embedding space bounds"
            
            return True, None
            
        elif operation_type in ["strengthen_synapse", "weaken_synapse", "adjust_synapse_weight"]:
            synapse = self.graph.get_synapse(target_id)
            if synapse is None:
                return False, f"Synapse not found: {target_id}"
            
            if operation_type == "adjust_synapse_weight":
                new_weight = kwargs.get("new_weight")
                if new_weight is None:
                    return False, "Missing new_weight parameter"
                
                if new_weight < -1.0 or new_weight > 1.0:
                    return False, f"Invalid weight: {new_weight}, must be in [-1.0, 1.0]"
            
            return True, None
            
        else:
            return False, f"Unknown operation type: {operation_type}"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training statistics
        """
        operation_counts = {}
        for op in self.operation_log:
            operation_counts[op.operation_type] = operation_counts.get(op.operation_type, 0) + 1
        
        return {
            "total_operations": len(self.operation_log),
            "operation_counts": operation_counts,
            "learning_rate": self.learning_rate,
            "decay_rate": self.decay_rate,
            "logging_enabled": self._enable_logging,
            "log_size": len(self.operation_log),
            "max_log_size": self._max_log_size
        }
    
    def create_tool_neuron(
        self,
        description: str,
        code: str,
        position: Optional[Vector3D] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        auto_connect: bool = True,
        connection_threshold: float = 0.7
    ) -> UUID:
        """
        Create a new tool neuron from description and code.
        
        Dynamically creates a tool neuron, validates its code, and optionally
        connects it to relevant knowledge neurons based on semantic similarity.
        
        Args:
            description: Natural language description of the tool's functionality
            code: Python code to execute (must set 'result' variable)
            position: Optional 3D position (auto-calculated if None)
            input_schema: Optional JSON Schema for input validation
            output_schema: Optional JSON Schema for output validation
            auto_connect: Whether to automatically connect to relevant neurons
            connection_threshold: Similarity threshold for auto-connections
            
        Returns:
            UUID of the created tool neuron
            
        Raises:
            ValueError: If code validation fails
            
        Requirements: 10.5, 12.1, 12.2
        """
        from neuron_system.neuron_types.tool_neuron import ToolNeuron
        from neuron_system.core.synapse import Synapse, SynapseType
        from neuron_system.engines.compression import CompressionEngine
        from uuid import uuid4
        
        logger.info(f"Creating tool neuron: {description[:50]}")
        
        # Validate the code
        validation_errors = self._validate_tool_code(code)
        if validation_errors:
            raise ValueError(f"Code validation failed: {', '.join(validation_errors)}")
        
        # Parse function signature from description
        function_signature = self._parse_function_signature(description, input_schema)
        
        # Compress description to vector
        compression_engine = CompressionEngine()
        vector, compression_meta = compression_engine.compress(description, normalize=True)
        
        if not compression_meta.get("success", False):
            logger.warning("Failed to compress tool description, using zero vector")
            vector = np.zeros(384)
        
        # Calculate position if not provided
        if position is None:
            # Position near similar neurons if auto-connect is enabled
            if auto_connect:
                position = self.graph.position_neuron(
                    None,  # We'll create the neuron after
                    strategy="random"
                )
            else:
                position = self.graph.position_neuron(None, strategy="random")
        
        # Create the tool neuron
        tool_neuron = ToolNeuron(
            function_signature=function_signature,
            executable_code=code,
            input_schema=input_schema or {},
            output_schema=output_schema or {},
            execution_count=0,
            average_execution_time=0.0,
            activation_threshold=0.5
        )
        
        tool_neuron.id = uuid4()
        tool_neuron.position = position
        tool_neuron.vector = vector
        tool_neuron.metadata = {
            "description": description,
            "created_by": "training_engine",
            "auto_connected": auto_connect
        }
        
        # Add to graph
        self.graph.add_neuron(tool_neuron)
        
        # Auto-connect to relevant knowledge neurons
        if auto_connect:
            connected_count = self._auto_connect_tool_neuron(
                tool_neuron,
                threshold=connection_threshold
            )
            logger.info(
                f"Auto-connected tool neuron {tool_neuron.id} to "
                f"{connected_count} knowledge neurons"
            )
        
        # Log operation
        self._log_operation(
            operation_type="create_tool_neuron",
            target_id=tool_neuron.id,
            before_state={},
            after_state={
                "neuron_id": str(tool_neuron.id),
                "description": description,
                "position": position.to_dict()
            },
            metadata={
                "auto_connected": auto_connect,
                "function_signature": function_signature
            }
        )
        
        logger.info(f"Created tool neuron {tool_neuron.id}")
        return tool_neuron.id
    
    def _validate_tool_code(self, code: str) -> List[str]:
        """
        Validate tool code for safety and correctness.
        
        Performs basic validation to ensure the code is safe to execute
        and follows expected patterns.
        
        Args:
            code: Python code to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check if code is empty
        if not code or not code.strip():
            errors.append("Code cannot be empty")
            return errors
        
        # Check for dangerous imports/operations
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'import socket',
            'import requests',
            'import urllib',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',
            'file(',
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                errors.append(f"Dangerous operation detected: {pattern}")
        
        # Check if code sets 'result' variable
        if 'result' not in code:
            errors.append("Code must set 'result' variable")
        
        # Try to compile the code
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            errors.append(f"Syntax error: {str(e)}")
        
        return errors
    
    def _parse_function_signature(
        self,
        description: str,
        input_schema: Optional[Dict[str, Any]]
    ) -> str:
        """
        Parse a function signature from description and input schema.
        
        Args:
            description: Tool description
            input_schema: Input schema with parameter definitions
            
        Returns:
            Function signature string
        """
        # Extract function name from description (first few words)
        words = description.split()
        func_name = '_'.join(words[:3]).lower()
        func_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in func_name)
        
        # Build parameter list from input schema
        params = []
        if input_schema and 'properties' in input_schema:
            for param_name, param_info in input_schema['properties'].items():
                param_type = param_info.get('type', 'any')
                params.append(f"{param_name}: {param_type}")
        
        if params:
            signature = f"{func_name}({', '.join(params)})"
        else:
            signature = f"{func_name}()"
        
        return signature
    
    def _auto_connect_tool_neuron(
        self,
        tool_neuron: Neuron,
        threshold: float = 0.7
    ) -> int:
        """
        Automatically connect a tool neuron to relevant knowledge neurons.
        
        Finds knowledge neurons with high semantic similarity to the tool's
        description and creates TOOL_INPUT synapses.
        
        Args:
            tool_neuron: The tool neuron to connect
            threshold: Minimum similarity for creating connections
            
        Returns:
            Number of connections created
            
        Requirements: 10.5, 12.1, 12.2
        """
        from neuron_system.core.neuron import NeuronType
        from neuron_system.core.synapse import Synapse, SynapseType
        from uuid import uuid4
        
        if tool_neuron.vector is None or len(tool_neuron.vector) == 0:
            logger.warning("Tool neuron has no vector, cannot auto-connect")
            return 0
        
        connections_created = 0
        
        # Find similar knowledge neurons
        for neuron in self.graph.neurons.values():
            # Skip non-knowledge neurons
            if neuron.neuron_type != NeuronType.KNOWLEDGE:
                continue
            
            # Skip if neuron has no vector
            if neuron.vector is None or len(neuron.vector) == 0:
                continue
            
            # Calculate similarity
            similarity = self._calculate_cosine_similarity(
                tool_neuron.vector,
                neuron.vector
            )
            
            # Create connection if similarity exceeds threshold
            if similarity >= threshold:
                synapse = Synapse(
                    id=uuid4(),
                    source_neuron_id=neuron.id,
                    target_neuron_id=tool_neuron.id,
                    weight=similarity,
                    synapse_type=SynapseType.TOOL_INPUT,
                    metadata={
                        "auto_created": True,
                        "similarity": similarity,
                        "parameter_name": "context"  # Default parameter name
                    }
                )
                
                self.graph.add_synapse(synapse)
                connections_created += 1
                
                logger.debug(
                    f"Connected knowledge neuron {neuron.id} to tool neuron "
                    f"{tool_neuron.id} with similarity {similarity:.3f}"
                )
        
        return connections_created
    
    def _calculate_cosine_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Ensure vectors are the same length
        if len(vec1) != len(vec2):
            max_len = max(len(vec1), len(vec2))
            if len(vec1) < max_len:
                vec1 = np.pad(vec1, (0, max_len - len(vec1)))
            if len(vec2) < max_len:
                vec2 = np.pad(vec2, (0, max_len - len(vec2)))
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Clamp to [-1, 1]
        return np.clip(similarity, -1.0, 1.0)
    
    def update_tool_neuron(
        self,
        tool_neuron_id: UUID,
        new_code: Optional[str] = None,
        new_description: Optional[str] = None,
        new_input_schema: Optional[Dict[str, Any]] = None,
        new_output_schema: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update an existing tool neuron's code or metadata.
        
        Allows modification of tool neurons through natural language instructions
        by updating their code, description, or schemas.
        
        Args:
            tool_neuron_id: UUID of the tool neuron to update
            new_code: Optional new code to replace existing code
            new_description: Optional new description
            new_input_schema: Optional new input schema
            new_output_schema: Optional new output schema
            
        Returns:
            True if update was successful, False otherwise
            
        Requirements: 12.3
        """
        from neuron_system.core.neuron import NeuronType
        
        # Get the tool neuron
        neuron = self.graph.get_neuron(tool_neuron_id)
        if neuron is None:
            logger.warning(f"Tool neuron not found: {tool_neuron_id}")
            return False
        
        if neuron.neuron_type != NeuronType.TOOL:
            logger.error(f"Neuron {tool_neuron_id} is not a tool neuron")
            return False
        
        # Store before state
        before_state = {
            "executable_code": neuron.executable_code,
            "function_signature": neuron.function_signature,
            "input_schema": neuron.input_schema.copy(),
            "output_schema": neuron.output_schema.copy(),
            "metadata": neuron.metadata.copy()
        }
        
        try:
            # Update code if provided
            if new_code is not None:
                validation_errors = self._validate_tool_code(new_code)
                if validation_errors:
                    raise ValueError(f"Code validation failed: {', '.join(validation_errors)}")
                neuron.executable_code = new_code
            
            # Update description and recompress if provided
            if new_description is not None:
                from neuron_system.engines.compression import CompressionEngine
                compression_engine = CompressionEngine()
                vector, compression_meta = compression_engine.compress(
                    new_description,
                    normalize=True
                )
                if compression_meta.get("success", False):
                    neuron.vector = vector
                neuron.metadata["description"] = new_description
                
                # Update function signature
                neuron.function_signature = self._parse_function_signature(
                    new_description,
                    new_input_schema or neuron.input_schema
                )
            
            # Update schemas if provided
            if new_input_schema is not None:
                neuron.input_schema = new_input_schema
            
            if new_output_schema is not None:
                neuron.output_schema = new_output_schema
            
            # Update modified timestamp
            neuron.modified_at = datetime.now()
            
            # Store after state
            after_state = {
                "executable_code": neuron.executable_code,
                "function_signature": neuron.function_signature,
                "input_schema": neuron.input_schema.copy(),
                "output_schema": neuron.output_schema.copy(),
                "metadata": neuron.metadata.copy()
            }
            
            # Log operation
            self._log_operation(
                operation_type="update_tool_neuron",
                target_id=tool_neuron_id,
                before_state=before_state,
                after_state=after_state,
                metadata={
                    "updated_code": new_code is not None,
                    "updated_description": new_description is not None,
                    "updated_schemas": (new_input_schema is not None or new_output_schema is not None)
                }
            )
            
            logger.info(f"Updated tool neuron {tool_neuron_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update tool neuron {tool_neuron_id}: {str(e)}", exc_info=True)
            return False


class TrainingTransaction:
    """
    Context manager for batch training operations with rollback support.
    
    Allows multiple operations to be grouped together and rolled back
    as a single unit if any operation fails.
    
    Requirements: 4.4, 4.5
    """
    
    def __init__(self, training_engine: TrainingEngine):
        """
        Initialize transaction.
        
        Args:
            training_engine: The training engine to use
        """
        self.engine = training_engine
        self.operations: List[int] = []
        self.start_operation_count = len(training_engine.operation_log)
        self.committed = False
    
    def __enter__(self):
        """Enter transaction context."""
        logger.debug("Started training transaction")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit transaction context.
        
        If an exception occurred, rollback all operations.
        """
        if exc_type is not None:
            # Exception occurred, rollback
            logger.warning(f"Transaction failed with {exc_type.__name__}, rolling back")
            self.rollback()
            return False  # Re-raise the exception
        
        if not self.committed:
            # Transaction not explicitly committed, rollback
            logger.warning("Transaction not committed, rolling back")
            self.rollback()
        
        return True
    
    def adjust_neuron(self, neuron_id: UUID, target_vector: np.ndarray, **kwargs) -> bool:
        """
        Adjust a neuron within the transaction.
        
        Args:
            neuron_id: UUID of the neuron
            target_vector: Target vector
            **kwargs: Additional arguments
            
        Returns:
            True if successful
        """
        success = self.engine.adjust_neuron(neuron_id, target_vector, **kwargs)
        if success:
            # Track the operation
            if len(self.engine.operation_log) > 0:
                self.operations.append(self.engine.operation_log[-1].operation_id)
        return success
    
    def strengthen_synapse(self, synapse_id: UUID, **kwargs) -> bool:
        """
        Strengthen a synapse within the transaction.
        
        Args:
            synapse_id: UUID of the synapse
            **kwargs: Additional arguments
            
        Returns:
            True if successful
        """
        success = self.engine.strengthen_synapse(synapse_id, **kwargs)
        if success:
            if len(self.engine.operation_log) > 0:
                self.operations.append(self.engine.operation_log[-1].operation_id)
        return success
    
    def weaken_synapse(self, synapse_id: UUID, **kwargs) -> bool:
        """
        Weaken a synapse within the transaction.
        
        Args:
            synapse_id: UUID of the synapse
            **kwargs: Additional arguments
            
        Returns:
            True if successful
        """
        success = self.engine.weaken_synapse(synapse_id, **kwargs)
        if success:
            if len(self.engine.operation_log) > 0:
                self.operations.append(self.engine.operation_log[-1].operation_id)
        return success
    
    def commit(self):
        """Commit the transaction."""
        self.committed = True
        logger.info(f"Committed transaction with {len(self.operations)} operations")
    
    def rollback(self):
        """Rollback all operations in the transaction."""
        rollback_count = 0
        
        # Rollback in reverse order
        for operation_id in reversed(self.operations):
            success = self.engine.rollback_operation(operation_id)
            if success:
                rollback_count += 1
        
        logger.info(f"Rolled back {rollback_count} operations in transaction")
        self.operations.clear()
