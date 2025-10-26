"""
Serialization and data integrity management for the neuron system.

Provides incremental save/load, change tracking, checksums, and backup/restore functionality.
"""

import json
import hashlib
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from uuid import UUID

from neuron_system.core.neuron import Neuron
from neuron_system.core.synapse import Synapse
from neuron_system.storage.database import DatabaseManager
from neuron_system.storage.neuron_store import NeuronStore
from neuron_system.storage.synapse_store import SynapseStore

logger = logging.getLogger(__name__)


class ChangeTracker:
    """
    Tracks modifications to neurons and synapses for incremental saves.
    """
    
    def __init__(self):
        """Initialize change tracker."""
        self.modified_neurons: Set[UUID] = set()
        self.modified_synapses: Set[UUID] = set()
        self.deleted_neurons: Set[UUID] = set()
        self.deleted_synapses: Set[UUID] = set()
        self.last_save_time: Optional[datetime] = None
    
    def mark_neuron_modified(self, neuron_id: UUID):
        """
        Mark a neuron as modified.
        
        Args:
            neuron_id: Neuron UUID
        """
        self.modified_neurons.add(neuron_id)
        # Remove from deleted if it was there
        self.deleted_neurons.discard(neuron_id)
    
    def mark_synapse_modified(self, synapse_id: UUID):
        """
        Mark a synapse as modified.
        
        Args:
            synapse_id: Synapse UUID
        """
        self.modified_synapses.add(synapse_id)
        # Remove from deleted if it was there
        self.deleted_synapses.discard(synapse_id)
    
    def mark_neuron_deleted(self, neuron_id: UUID):
        """
        Mark a neuron as deleted.
        
        Args:
            neuron_id: Neuron UUID
        """
        self.deleted_neurons.add(neuron_id)
        # Remove from modified if it was there
        self.modified_neurons.discard(neuron_id)
    
    def mark_synapse_deleted(self, synapse_id: UUID):
        """
        Mark a synapse as deleted.
        
        Args:
            synapse_id: Synapse UUID
        """
        self.deleted_synapses.add(synapse_id)
        # Remove from modified if it was there
        self.modified_synapses.discard(synapse_id)
    
    def clear(self):
        """Clear all tracked changes."""
        self.modified_neurons.clear()
        self.modified_synapses.clear()
        self.deleted_neurons.clear()
        self.deleted_synapses.clear()
        self.last_save_time = datetime.now()
    
    def has_changes(self) -> bool:
        """
        Check if there are any tracked changes.
        
        Returns:
            True if there are changes
        """
        return bool(
            self.modified_neurons or 
            self.modified_synapses or 
            self.deleted_neurons or 
            self.deleted_synapses
        )
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about tracked changes.
        
        Returns:
            Dictionary with change counts
        """
        return {
            "modified_neurons": len(self.modified_neurons),
            "modified_synapses": len(self.modified_synapses),
            "deleted_neurons": len(self.deleted_neurons),
            "deleted_synapses": len(self.deleted_synapses),
        }


class SerializationManager:
    """
    Manages serialization, deserialization, and data integrity for the neuron system.
    
    Provides incremental saves, checksums, and backup/restore functionality.
    """
    
    def __init__(self, db_manager: DatabaseManager, backup_dir: str = "backups"):
        """
        Initialize serialization manager.
        
        Args:
            db_manager: Database manager instance
            backup_dir: Directory for backup files
        """
        self.db = db_manager
        self.neuron_store = NeuronStore(db_manager)
        self.synapse_store = SynapseStore(db_manager)
        self.change_tracker = ChangeTracker()
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def save_all(self, neurons: Dict[UUID, Neuron], synapses: Dict[UUID, Synapse]) -> Dict[str, Any]:
        """
        Save all neurons and synapses to database.
        
        Args:
            neurons: Dictionary of neurons by ID
            synapses: Dictionary of synapses by ID
            
        Returns:
            Dictionary with save statistics
        """
        start_time = datetime.now()
        
        try:
            # Save neurons
            neuron_count = self.neuron_store.batch_create(list(neurons.values()))
            
            # Save synapses
            synapse_count = self.synapse_store.batch_create(list(synapses.values()))
            
            # Clear change tracker
            self.change_tracker.clear()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            stats = {
                "neurons_saved": neuron_count,
                "synapses_saved": synapse_count,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"Saved all data: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to save all data: {e}")
            raise
    
    def save_incremental(self, neurons: Dict[UUID, Neuron], synapses: Dict[UUID, Synapse]) -> Dict[str, Any]:
        """
        Save only modified neurons and synapses.
        
        Args:
            neurons: Dictionary of all neurons by ID
            synapses: Dictionary of all synapses by ID
            
        Returns:
            Dictionary with save statistics
        """
        if not self.change_tracker.has_changes():
            logger.debug("No changes to save")
            return {
                "neurons_saved": 0,
                "synapses_saved": 0,
                "duration_seconds": 0,
                "timestamp": datetime.now().isoformat(),
            }
        
        start_time = datetime.now()
        
        try:
            neuron_count = 0
            synapse_count = 0
            
            # Save modified neurons
            modified_neurons = [
                neurons[nid] for nid in self.change_tracker.modified_neurons 
                if nid in neurons
            ]
            if modified_neurons:
                neuron_count = self.neuron_store.batch_update(modified_neurons)
            
            # Save modified synapses
            modified_synapses = [
                synapses[sid] for sid in self.change_tracker.modified_synapses 
                if sid in synapses
            ]
            if modified_synapses:
                synapse_count = self.synapse_store.batch_update(modified_synapses)
            
            # Handle deletions (already handled by store operations)
            # Just clear the tracking
            
            # Clear change tracker
            self.change_tracker.clear()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            stats = {
                "neurons_saved": neuron_count,
                "synapses_saved": synapse_count,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"Incremental save completed: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to save incremental changes: {e}")
            raise
    
    def load_all(self) -> Dict[str, Any]:
        """
        Load all neurons and synapses from database.
        
        Returns:
            Dictionary with neurons, synapses, and load statistics
        """
        start_time = datetime.now()
        
        try:
            # Load all neurons
            neuron_list = self.neuron_store.list_all()
            neurons = {neuron.id: neuron for neuron in neuron_list}
            
            # Load all synapses
            synapse_list = self.synapse_store.list_all()
            synapses = {synapse.id: synapse for synapse in synapse_list}
            
            # Clear change tracker since we just loaded
            self.change_tracker.clear()
            
            duration = (datetime.now() - start_time).total_seconds()
            
            stats = {
                "neurons_loaded": len(neurons),
                "synapses_loaded": len(synapses),
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
            }
            
            logger.info(f"Loaded all data: {stats}")
            
            return {
                "neurons": neurons,
                "synapses": synapses,
                "stats": stats,
            }
        
        except Exception as e:
            logger.error(f"Failed to load all data: {e}")
            raise
    
    def calculate_checksum(self, data: Any) -> str:
        """
        Calculate SHA256 checksum for data integrity verification.
        
        Args:
            data: Data to checksum (will be JSON serialized)
            
        Returns:
            Hexadecimal checksum string
        """
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify data integrity by checking database consistency.
        
        Returns:
            Dictionary with integrity check results
        """
        issues = []
        
        try:
            with self.db.get_connection() as conn:
                # Check for orphaned synapses (source neuron doesn't exist)
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM synapses s
                    LEFT JOIN neurons n ON s.source_neuron_id = n.id
                    WHERE n.id IS NULL
                """)
                orphaned_sources = cursor.fetchone()['count']
                if orphaned_sources > 0:
                    issues.append(f"{orphaned_sources} synapses with missing source neurons")
                
                # Check for orphaned synapses (target neuron doesn't exist)
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM synapses s
                    LEFT JOIN neurons n ON s.target_neuron_id = n.id
                    WHERE n.id IS NULL
                """)
                orphaned_targets = cursor.fetchone()['count']
                if orphaned_targets > 0:
                    issues.append(f"{orphaned_targets} synapses with missing target neurons")
                
                # Check for invalid synapse weights
                cursor = conn.execute("""
                    SELECT COUNT(*) as count FROM synapses
                    WHERE weight < -1.0 OR weight > 1.0
                """)
                invalid_weights = cursor.fetchone()['count']
                if invalid_weights > 0:
                    issues.append(f"{invalid_weights} synapses with invalid weights")
            
            result = {
                "valid": len(issues) == 0,
                "issues": issues,
                "timestamp": datetime.now().isoformat(),
            }
            
            if result["valid"]:
                logger.info("Data integrity check passed")
            else:
                logger.warning(f"Data integrity issues found: {issues}")
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to verify integrity: {e}")
            raise
    
    def create_backup(self, backup_name: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_name: Optional custom backup name
            
        Returns:
            Path to backup file
        """
        try:
            if backup_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"neuron_system_backup_{timestamp}.db"
            
            backup_path = self.backup_dir / backup_name
            
            # Copy database file
            shutil.copy2(self.db.db_path, backup_path)
            
            # Create metadata file
            metadata = {
                "backup_time": datetime.now().isoformat(),
                "original_db": str(self.db.db_path),
                "stats": self.db.get_stats(),
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore database from a backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if successful
        """
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Close current database connections
            self.db.close()
            
            # Create backup of current database before restoring
            current_backup = self.create_backup("pre_restore_backup.db")
            logger.info(f"Created safety backup: {current_backup}")
            
            # Restore from backup
            shutil.copy2(backup_file, self.db.db_path)
            
            # Reinitialize database
            self.db._initialized = False
            self.db._initialize_schema()
            
            # Clear change tracker
            self.change_tracker.clear()
            
            logger.info(f"Restored from backup: {backup_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            raise
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups.
        
        Returns:
            List of backup information dictionaries
        """
        backups = []
        
        for backup_file in self.backup_dir.glob("*.db"):
            metadata_file = backup_file.with_suffix('.json')
            
            backup_info = {
                "path": str(backup_file),
                "name": backup_file.name,
                "size_bytes": backup_file.stat().st_size,
                "created": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
            }
            
            # Load metadata if available
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    backup_info["metadata"] = metadata
            
            backups.append(backup_info)
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x["created"], reverse=True)
        
        return backups
    
    def export_to_json(self, output_path: str) -> Dict[str, Any]:
        """
        Export entire database to JSON format.
        
        Args:
            output_path: Path to output JSON file
            
        Returns:
            Export statistics
        """
        try:
            # Load all data
            data = self.load_all()
            
            # Convert to serializable format
            export_data = {
                "version": 1,
                "export_time": datetime.now().isoformat(),
                "neurons": [neuron.to_dict() for neuron in data["neurons"].values()],
                "synapses": [synapse.to_dict() for synapse in data["synapses"].values()],
                "stats": data["stats"],
            }
            
            # Calculate checksum
            export_data["checksum"] = self.calculate_checksum({
                "neurons": export_data["neurons"],
                "synapses": export_data["synapses"],
            })
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            stats = {
                "neurons_exported": len(export_data["neurons"]),
                "synapses_exported": len(export_data["synapses"]),
                "file_path": output_path,
                "file_size_bytes": Path(output_path).stat().st_size,
            }
            
            logger.info(f"Exported to JSON: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            raise
    
    def import_from_json(self, input_path: str, verify_checksum: bool = True) -> Dict[str, Any]:
        """
        Import database from JSON format.
        
        Args:
            input_path: Path to input JSON file
            verify_checksum: Whether to verify data integrity
            
        Returns:
            Import statistics
        """
        try:
            # Read JSON file
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            # Verify checksum if requested
            if verify_checksum and "checksum" in import_data:
                calculated_checksum = self.calculate_checksum({
                    "neurons": import_data["neurons"],
                    "synapses": import_data["synapses"],
                })
                
                if calculated_checksum != import_data["checksum"]:
                    raise ValueError("Checksum verification failed - data may be corrupted")
            
            # Import neurons
            from neuron_system.core.neuron import NeuronTypeRegistry
            neurons = {}
            for neuron_data in import_data["neurons"]:
                neuron = NeuronTypeRegistry.deserialize(neuron_data)
                neurons[neuron.id] = neuron
            
            # Import synapses
            synapses = {}
            for synapse_data in import_data["synapses"]:
                synapse = Synapse.from_dict(synapse_data)
                synapses[synapse.id] = synapse
            
            # Save to database
            save_stats = self.save_all(neurons, synapses)
            
            stats = {
                "neurons_imported": len(neurons),
                "synapses_imported": len(synapses),
                "save_stats": save_stats,
            }
            
            logger.info(f"Imported from JSON: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Failed to import from JSON: {e}")
            raise
