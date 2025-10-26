"""
Database schema and connection management for the neuron system.

Provides SQLite database setup with proper schema, indexes, and connection pooling.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages SQLite database connections and schema.
    
    Implements connection pooling and provides schema migration support.
    """
    
    # Schema version for migration tracking
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = "neuron_system.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._initialized = False
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._initialize_schema()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get thread-local database connection.
        
        Returns:
            SQLite connection for current thread
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            # Enable foreign keys
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            # Use WAL mode for better concurrency
            self._local.connection.execute("PRAGMA journal_mode = WAL")
            # Row factory for dict-like access
            self._local.connection.row_factory = sqlite3.Row
        
        return self._local.connection
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            SQLite connection
        """
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        
        Automatically commits on success, rolls back on error.
        
        Yields:
            SQLite connection
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction error: {e}")
            raise
    
    def _initialize_schema(self):
        """Initialize database schema if not exists."""
        with self._lock:
            if self._initialized:
                return
            
            with self.transaction() as conn:
                # Create schema version table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY,
                        applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Check current version
                cursor = conn.execute("SELECT MAX(version) as version FROM schema_version")
                row = cursor.fetchone()
                current_version = row['version'] if row['version'] is not None else 0
                
                # Apply migrations
                if current_version < 1:
                    self._apply_migration_v1(conn)
                    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
                
                self._initialized = True
                logger.info(f"Database initialized at version {self.SCHEMA_VERSION}")
    
    def _apply_migration_v1(self, conn: sqlite3.Connection):
        """
        Apply version 1 schema migration.
        
        Args:
            conn: Database connection
        """
        # Create neurons table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS neurons (
                id TEXT PRIMARY KEY,
                position_x REAL NOT NULL,
                position_y REAL NOT NULL,
                position_z REAL NOT NULL,
                vector BLOB,
                neuron_type TEXT NOT NULL,
                metadata TEXT,
                activation_level REAL DEFAULT 0.0,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                
                -- Type-specific fields (stored as JSON in metadata for extensibility)
                -- KnowledgeNeuron fields
                source_data TEXT,
                compression_ratio REAL,
                semantic_tags TEXT,
                
                -- ToolNeuron fields
                function_signature TEXT,
                executable_code TEXT,
                input_schema TEXT,
                output_schema TEXT,
                execution_count INTEGER DEFAULT 0,
                average_execution_time REAL DEFAULT 0.0,
                activation_threshold REAL DEFAULT 0.5
            )
        """)
        
        # Create synapses table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS synapses (
                id TEXT PRIMARY KEY,
                source_neuron_id TEXT NOT NULL,
                target_neuron_id TEXT NOT NULL,
                weight REAL NOT NULL CHECK(weight >= -1.0 AND weight <= 1.0),
                usage_count INTEGER DEFAULT 0,
                last_traversed TEXT,
                synapse_type TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                modified_at TEXT NOT NULL,
                
                FOREIGN KEY (source_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE,
                FOREIGN KEY (target_neuron_id) REFERENCES neurons(id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for neurons
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_neurons_position 
            ON neurons(position_x, position_y, position_z)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_neurons_type 
            ON neurons(neuron_type)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_neurons_modified 
            ON neurons(modified_at)
        """)
        
        # Create indexes for synapses
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_synapses_source 
            ON synapses(source_neuron_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_synapses_target 
            ON synapses(target_neuron_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_synapses_weight 
            ON synapses(weight)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_synapses_modified 
            ON synapses(modified_at)
        """)
        
        logger.info("Applied schema migration v1")
    
    def close(self):
        """Close all database connections."""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    def vacuum(self):
        """Optimize database by running VACUUM."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with neuron and synapse counts
        """
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM neurons")
            neuron_count = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT COUNT(*) as count FROM synapses")
            synapse_count = cursor.fetchone()['count']
            
            cursor = conn.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()['size']
            
            return {
                "neuron_count": neuron_count,
                "synapse_count": synapse_count,
                "database_size_bytes": db_size,
            }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager(db_path: str = "neuron_system.db") -> DatabaseManager:
    """
    Get or create global database manager instance.
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        DatabaseManager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager(db_path)
    return _db_manager


def close_database():
    """Close global database manager."""
    global _db_manager
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None
