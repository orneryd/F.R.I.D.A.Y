"""
Synapse storage operations with CRUD functionality.

Provides persistent storage for synapses with batch operations and referential integrity.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID

from neuron_system.core.synapse import Synapse
from neuron_system.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class SynapseStore:
    """
    Handles CRUD operations for synapses in the database.
    
    Maintains referential integrity with neurons through foreign key constraints.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize synapse store.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def create(self, synapse: Synapse) -> bool:
        """
        Create a new synapse in the database.
        
        Args:
            synapse: Synapse instance to store
            
        Returns:
            True if successful
            
        Raises:
            Exception: If source or target neuron doesn't exist
        """
        try:
            # Validate referential integrity
            self._validate_neuron_references(synapse)
            
            with self.db.transaction() as conn:
                data = self._synapse_to_row(synapse)
                
                placeholders = ', '.join(['?' for _ in data])
                columns = ', '.join(data.keys())
                
                conn.execute(
                    f"INSERT INTO synapses ({columns}) VALUES ({placeholders})",
                    list(data.values())
                )
            
            logger.debug(f"Created synapse {synapse.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create synapse {synapse.id}: {e}")
            raise
    
    def get(self, synapse_id: UUID) -> Optional[Synapse]:
        """
        Retrieve a synapse by ID.
        
        Args:
            synapse_id: Synapse UUID
            
        Returns:
            Synapse instance or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM synapses WHERE id = ?",
                    (str(synapse_id),)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                return self._row_to_synapse(dict(row))
        
        except Exception as e:
            logger.error(f"Failed to get synapse {synapse_id}: {e}")
            raise
    
    def update(self, synapse: Synapse) -> bool:
        """
        Update an existing synapse.
        
        Args:
            synapse: Synapse instance with updated data
            
        Returns:
            True if successful
        """
        try:
            with self.db.transaction() as conn:
                data = self._synapse_to_row(synapse)
                
                # Remove id from update data
                synapse_id = data.pop('id')
                
                set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
                values = list(data.values()) + [synapse_id]
                
                cursor = conn.execute(
                    f"UPDATE synapses SET {set_clause} WHERE id = ?",
                    values
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"Synapse {synapse_id} not found for update")
                    return False
            
            logger.debug(f"Updated synapse {synapse.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update synapse {synapse.id}: {e}")
            raise
    
    def delete(self, synapse_id: UUID) -> bool:
        """
        Delete a synapse by ID.
        
        Args:
            synapse_id: Synapse UUID
            
        Returns:
            True if successful
        """
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM synapses WHERE id = ?",
                    (str(synapse_id),)
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"Synapse {synapse_id} not found for deletion")
                    return False
            
            logger.debug(f"Deleted synapse {synapse_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete synapse {synapse_id}: {e}")
            raise
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Synapse]:
        """
        List all synapses with optional pagination.
        
        Args:
            limit: Maximum number of synapses to return
            offset: Number of synapses to skip
            
        Returns:
            List of Synapse instances
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM synapses ORDER BY created_at"
                
                if limit is not None:
                    query += f" LIMIT {limit} OFFSET {offset}"
                
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                
                return [self._row_to_synapse(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to list synapses: {e}")
            raise
    
    def list_by_source(self, source_neuron_id: UUID) -> List[Synapse]:
        """
        List all synapses originating from a neuron.
        
        Args:
            source_neuron_id: Source neuron UUID
            
        Returns:
            List of Synapse instances
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM synapses WHERE source_neuron_id = ? ORDER BY weight DESC",
                    (str(source_neuron_id),)
                )
                rows = cursor.fetchall()
                
                return [self._row_to_synapse(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to list synapses by source {source_neuron_id}: {e}")
            raise
    
    def list_by_target(self, target_neuron_id: UUID) -> List[Synapse]:
        """
        List all synapses targeting a neuron.
        
        Args:
            target_neuron_id: Target neuron UUID
            
        Returns:
            List of Synapse instances
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM synapses WHERE target_neuron_id = ? ORDER BY weight DESC",
                    (str(target_neuron_id),)
                )
                rows = cursor.fetchall()
                
                return [self._row_to_synapse(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to list synapses by target {target_neuron_id}: {e}")
            raise
    
    def list_by_neurons(self, source_neuron_id: UUID, target_neuron_id: UUID) -> List[Synapse]:
        """
        List synapses between two specific neurons.
        
        Args:
            source_neuron_id: Source neuron UUID
            target_neuron_id: Target neuron UUID
            
        Returns:
            List of Synapse instances
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    """SELECT * FROM synapses 
                       WHERE source_neuron_id = ? AND target_neuron_id = ?
                       ORDER BY created_at""",
                    (str(source_neuron_id), str(target_neuron_id))
                )
                rows = cursor.fetchall()
                
                return [self._row_to_synapse(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to list synapses between neurons: {e}")
            raise
    
    def batch_create(self, synapses: List[Synapse]) -> int:
        """
        Create multiple synapses in a single transaction.
        
        Args:
            synapses: List of Synapse instances
            
        Returns:
            Number of synapses created
        """
        if not synapses:
            return 0
        
        try:
            with self.db.transaction() as conn:
                count = 0
                for synapse in synapses:
                    # Validate referential integrity
                    self._validate_neuron_references(synapse)
                    
                    data = self._synapse_to_row(synapse)
                    
                    placeholders = ', '.join(['?' for _ in data])
                    columns = ', '.join(data.keys())
                    
                    conn.execute(
                        f"INSERT INTO synapses ({columns}) VALUES ({placeholders})",
                        list(data.values())
                    )
                    count += 1
            
            logger.info(f"Batch created {count} synapses")
            return count
        
        except Exception as e:
            logger.error(f"Failed to batch create synapses: {e}")
            raise
    
    def batch_update(self, synapses: List[Synapse]) -> int:
        """
        Update multiple synapses in a single transaction.
        
        Args:
            synapses: List of Synapse instances
            
        Returns:
            Number of synapses updated
        """
        if not synapses:
            return 0
        
        try:
            with self.db.transaction() as conn:
                count = 0
                for synapse in synapses:
                    data = self._synapse_to_row(synapse)
                    synapse_id = data.pop('id')
                    
                    set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
                    values = list(data.values()) + [synapse_id]
                    
                    cursor = conn.execute(
                        f"UPDATE synapses SET {set_clause} WHERE id = ?",
                        values
                    )
                    count += cursor.rowcount
            
            logger.info(f"Batch updated {count} synapses")
            return count
        
        except Exception as e:
            logger.error(f"Failed to batch update synapses: {e}")
            raise
    
    def get_modified_since(self, timestamp: str) -> List[Synapse]:
        """
        Get synapses modified since a specific timestamp.
        
        Args:
            timestamp: ISO format timestamp
            
        Returns:
            List of modified Synapse instances
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM synapses WHERE modified_at > ? ORDER BY modified_at",
                    (timestamp,)
                )
                rows = cursor.fetchall()
                
                return [self._row_to_synapse(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to get modified synapses: {e}")
            raise
    
    def delete_weak_synapses(self, threshold: float = 0.01) -> int:
        """
        Delete synapses with weight below threshold.
        
        Args:
            threshold: Weight threshold for deletion
            
        Returns:
            Number of synapses deleted
        """
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM synapses WHERE ABS(weight) < ?",
                    (threshold,)
                )
                count = cursor.rowcount
            
            logger.info(f"Deleted {count} weak synapses")
            return count
        
        except Exception as e:
            logger.error(f"Failed to delete weak synapses: {e}")
            raise
    
    def _validate_neuron_references(self, synapse: Synapse):
        """
        Validate that source and target neurons exist.
        
        Args:
            synapse: Synapse to validate
            
        Raises:
            ValueError: If neurons don't exist
        """
        with self.db.get_connection() as conn:
            # Check source neuron
            cursor = conn.execute(
                "SELECT id FROM neurons WHERE id = ?",
                (str(synapse.source_neuron_id),)
            )
            if cursor.fetchone() is None:
                raise ValueError(f"Source neuron {synapse.source_neuron_id} does not exist")
            
            # Check target neuron
            cursor = conn.execute(
                "SELECT id FROM neurons WHERE id = ?",
                (str(synapse.target_neuron_id),)
            )
            if cursor.fetchone() is None:
                raise ValueError(f"Target neuron {synapse.target_neuron_id} does not exist")
    
    def _synapse_to_row(self, synapse: Synapse) -> Dict[str, Any]:
        """
        Convert synapse to database row format.
        
        Args:
            synapse: Synapse instance
            
        Returns:
            Dictionary with database column values
        """
        synapse_dict = synapse.to_dict()
        
        return {
            'id': synapse_dict['id'],
            'source_neuron_id': synapse_dict['source_neuron_id'],
            'target_neuron_id': synapse_dict['target_neuron_id'],
            'weight': synapse_dict['weight'],
            'usage_count': synapse_dict['usage_count'],
            'last_traversed': synapse_dict['last_traversed'],
            'synapse_type': synapse_dict['synapse_type'],
            'metadata': json.dumps(synapse_dict.get('metadata', {})),
            'created_at': synapse_dict['created_at'],
            'modified_at': synapse_dict['modified_at'],
        }
    
    def _row_to_synapse(self, row: Dict[str, Any]) -> Synapse:
        """
        Convert database row to synapse instance.
        
        Args:
            row: Database row as dictionary
            
        Returns:
            Synapse instance
        """
        # Deserialize JSON fields
        if row.get('metadata'):
            row['metadata'] = json.loads(row['metadata'])
        
        return Synapse.from_dict(row)
