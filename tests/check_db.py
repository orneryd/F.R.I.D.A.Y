"""
Check database contents directly.
"""

import sqlite3
import os

db_path = "comprehensive_ai.db"

if not os.path.exists(db_path):
    print(f"Database not found: {db_path}")
    exit(1)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:")
for table in tables:
    print(f"  - {table[0]}")
print()

# Check neurons
cursor.execute("SELECT COUNT(*) FROM neurons;")
neuron_count = cursor.fetchone()[0]
print(f"Neurons in database: {neuron_count}")

# Check synapses
cursor.execute("SELECT COUNT(*) FROM synapses;")
synapse_count = cursor.fetchone()[0]
print(f"Synapses in database: {synapse_count}")

# Sample some neurons
if neuron_count > 0:
    print("\nSample neurons:")
    cursor.execute("SELECT id, neuron_type, data FROM neurons LIMIT 5;")
    for row in cursor.fetchall():
        print(f"  ID: {row[0]}, Type: {row[1]}, Data length: {len(row[2])}")

conn.close()
