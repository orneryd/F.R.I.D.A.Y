"""
Neuron storage operations with CRUD functionality.

Provides persistent storage for neurons with batch operations and type-specific handling.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import numpy as np

from neuron_system.core.neuron import Neuron, NeuronTypeRegistry
from neuron_system.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class NeuronStore:
    """
    Handles CRUD operations for neurons in the database.
    
    Supports all registered neuron types through the NeuronTypeRegistry.
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize neuron store.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
    
    def create(self, neuron: Neuron) -> bool:
        """
        Create a new neuron in the database.
        
        Args:
            neuron: Neuron instance to store
            
        Returns:
            True if successful
        """
        try:
            with self.db.transaction() as conn:
                data = self._neuron_to_row(neuron)
                
                placeholders = ', '.join(['?' for _ in data])
                columns = ', '.join(data.keys())
                
                conn.execute(
                    f"INSERT INTO neurons ({columns}) VALUES ({placeholders})",
                    list(data.values())
                )
            
            logger.debug(f"Created neuron {neuron.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create neuron {neuron.id}: {e}")
            raise
    
    def get(self, neuron_id: UUID) -> Optional[Neuron]:
        """
        Retrieve a neuron by ID.
        
        Args:
            neuron_id: Neuron UUID
            
        Returns:
            Neuron instance or None if not found
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM neurons WHERE id = ?",
                    (str(neuron_id),)
                )
                row = cursor.fetchone()
                
                if row is None:
                    return None
                
                return self._row_to_neuron(dict(row))
        
        except Exception as e:
            logger.error(f"Failed to get neuron {neuron_id}: {e}")
            raise
    
    def update(self, neuron: Neuron) -> bool:
        """
        Update an existing neuron.
        
        Args:
            neuron: Neuron instance with updated data
            
        Returns:
            True if successful
        """
        try:
            with self.db.transaction() as conn:
                data = self._neuron_to_row(neuron)
                
                # Remove id from update data
                neuron_id = data.pop('id')
                
                set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
                values = list(data.values()) + [neuron_id]
                
                cursor = conn.execute(
                    f"UPDATE neurons SET {set_clause} WHERE id = ?",
                    values
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"Neuron {neuron_id} not found for update")
                    return False
            
            logger.debug(f"Updated neuron {neuron.id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update neuron {neuron.id}: {e}")
            raise
    
    def delete(self, neuron_id: UUID) -> bool:
        """
        Delete a neuron by ID.
        
        Cascades to delete associated synapses due to foreign key constraints.
        
        Args:
            neuron_id: Neuron UUID
            
        Returns:
            True if successful
        """
        try:
            with self.db.transaction() as conn:
                cursor = conn.execute(
                    "DELETE FROM neurons WHERE id = ?",
                    (str(neuron_id),)
                )
                
                if cursor.rowcount == 0:
                    logger.warning(f"Neuron {neuron_id} not found for deletion")
                    return False
            
            logger.debug(f"Deleted neuron {neuron_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete neuron {neuron_id}: {e}")
            raise
    
    def list_all(self, limit: Optional[int] = None, offset: int = 0) -> List[Neuron]:
        """
        List all neurons with optional pagination.
        
        Args:
            limit: Maximum number of neurons to return
            offset: Number of neurons to skip
            
        Returns:
            List of Neuron instances
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM neurons ORDER BY created_at"
                
                if limit is not None:
                    query += f" LIMIT {limit} OFFSET {offset}"
                
                cursor = conn.execute(query)
                rows = cursor.fetchall()
                
                return [self._row_to_neuron(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to list neurons: {e}")
            raise
    
    def list_by_type(self, neuron_type: str, limit: Optional[int] = None) -> List[Neuron]:
        """
        List neurons by type.
        
        Args:
            neuron_type: Type of neurons to retrieve
            limit: Maximum number of neurons to return
            
        Returns:
            List of Neuron instances
        """
        try:
            with self.db.get_connection() as conn:
                query = "SELECT * FROM neurons WHERE neuron_type = ? ORDER BY created_at"
                
                if limit is not None:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query, (neuron_type,))
                rows = cursor.fetchall()
                
                return [self._row_to_neuron(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to list neurons by type {neuron_type}: {e}")
            raise
    
    def batch_create(self, neurons: List[Neuron]) -> int:
        """
        Create multiple neurons in a single transaction.
        
        Args:
            neurons: List of Neuron instances
            
        Returns:
            Number of neurons created
        """
        if not neurons:
            return 0
        
        try:
            with self.db.transaction() as conn:
                count = 0
                for neuron in neurons:
                    data = self._neuron_to_row(neuron)
                    
                    placeholders = ', '.join(['?' for _ in data])
                    columns = ', '.join(data.keys())
                    
                    conn.execute(
                        f"INSERT INTO neurons ({columns}) VALUES ({placeholders})",
                        list(data.values())
                    )
                    count += 1
            
            logger.info(f"Batch created {count} neurons")
            return count
        
        except Exception as e:
            logger.error(f"Failed to batch create neurons: {e}")
            raise
    
    def batch_update(self, neurons: List[Neuron]) -> int:
        """
        Update multiple neurons in a single transaction.
        
        Args:
            neurons: List of Neuron instances
            
        Returns:
            Number of neurons updated
        """
        if not neurons:
            return 0
        
        try:
            with self.db.transaction() as conn:
                count = 0
                for neuron in neurons:
                    data = self._neuron_to_row(neuron)
                    neuron_id = data.pop('id')
                    
                    set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
                    values = list(data.values()) + [neuron_id]
                    
                    cursor = conn.execute(
                        f"UPDATE neurons SET {set_clause} WHERE id = ?",
                        values
                    )
                    count += cursor.rowcount
            
            logger.info(f"Batch updated {count} neurons")
            return count
        
        except Exception as e:
            logger.error(f"Failed to batch update neurons: {e}")
            raise
    
    def get_modified_since(self, timestamp: str) -> List[Neuron]:
        """
        Get neurons modified since a specific timestamp.
        
        Args:
            timestamp: ISO format timestamp
            
        Returns:
            List of modified Neuron instances
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM neurons WHERE modified_at > ? ORDER BY modified_at",
                    (timestamp,)
                )
                rows = cursor.fetchall()
                
                return [self._row_to_neuron(dict(row)) for row in rows]
        
        except Exception as e:
            logger.error(f"Failed to get modified neurons: {e}")
            raise
    
    def _neuron_to_row(self, neuron: Neuron) -> Dict[str, Any]:
        """
        Convert neuron to database row format.
        
        Args:
            neuron: Neuron instance
            
        Returns:
            Dictionary with database column values
        """
        # Get base neuron data
        neuron_dict = neuron.to_dict()
        
        # Extract position
        position = neuron_dict.get('position', {})
        
        # Serialize vector as bytes
        vector_bytes = None
        if neuron_dict.get('vector'):
            vector_array = np.array(neuron_dict['vector'], dtype=np.float32)
            vector_bytes = vector_array.tobytes()
        
        # Base row data
        row = {
            'id': neuron_dict['id'],
            'position_x': position.get('x', 0.0),
            'position_y': position.get('y', 0.0),
            'position_z': position.get('z', 0.0),
            'vector': vector_bytes,
            'neuron_type': neuron_dict['type'],
            'metadata': json.dumps(neuron_dict.get('metadata', {})),
            'activation_level': neuron_dict.get('activation_level', 0.0),
            'created_at': neuron_dict['created_at'],
            'modified_at': neuron_dict['modified_at'],
        }
        
        # Add type-specific fields
        if neuron_dict['type'] == 'knowledge':
            row.update({
                'source_data': neuron_dict.get('source_data'),
                'compression_ratio': neuron_dict.get('compression_ratio'),
                'semantic_tags': json.dumps(neuron_dict.get('semantic_tags', [])),
            })
        elif neuron_dict['type'] == 'tool':
            row.update({
                'function_signature': neuron_dict.get('function_signature'),
                'executable_code': neuron_dict.get('executable_code'),
                'input_schema': json.dumps(neuron_dict.get('input_schema', {})),
                'output_schema': json.dumps(neuron_dict.get('output_schema', {})),
                'execution_count': neuron_dict.get('execution_count', 0),
                'average_execution_time': neuron_dict.get('average_execution_time', 0.0),
                'activation_threshold': neuron_dict.get('activation_threshold', 0.5),
            })
        
        return row
    
    def _row_to_neuron(self, row: Dict[str, Any]) -> Neuron:
        """
        Convert database row to neuron instance.
        
        Args:
            row: Database row as dictionary
            
        Returns:
            Neuron instance
        """
        # Deserialize vector
        if row.get('vector'):
            vector_array = np.frombuffer(row['vector'], dtype=np.float32)
            row['vector'] = vector_array.tolist()
        
        # Reconstruct position
        row['position'] = {
            'x': row.pop('position_x'),
            'y': row.pop('position_y'),
            'z': row.pop('position_z'),
        }
        
        # Deserialize JSON fields
        if row.get('metadata'):
            row['metadata'] = json.loads(row['metadata'])
        
        # Map neuron_type to type for registry
        row['type'] = row.pop('neuron_type')
        
        # Type-specific deserialization
        if row['type'] == 'knowledge':
            if row.get('semantic_tags'):
                row['semantic_tags'] = json.loads(row['semantic_tags'])
        elif row['type'] == 'tool':
            if row.get('input_schema'):
                row['input_schema'] = json.loads(row['input_schema'])
            if row.get('output_schema'):
                row['output_schema'] = json.loads(row['output_schema'])
        
        # Use registry to deserialize
        return NeuronTypeRegistry.deserialize(row)
