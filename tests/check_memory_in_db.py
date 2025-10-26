"""
Check if memory neurons are actually in the database.
"""
import sqlite3

db_path = "comprehensive_ai.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check neurons table
cursor.execute("SELECT COUNT(*) FROM neurons WHERE neuron_type = 'memory'")
memory_count = cursor.fetchone()[0]

print(f"Memory neurons in database: {memory_count}")

# Get some examples
cursor.execute("SELECT id, neuron_type, created_at FROM neurons WHERE neuron_type = 'memory' LIMIT 5")
rows = cursor.fetchall()

if rows:
    print("\nExample memory neurons:")
    for row in rows:
        print(f"  ID: {row[0]}, Type: {row[1]}, Created: {row[2]}")
else:
    print("\nNo memory neurons found in database!")

# Check all neuron types
cursor.execute("SELECT neuron_type, COUNT(*) FROM neurons GROUP BY neuron_type")
types = cursor.fetchall()

print("\nAll neuron types in database:")
for neuron_type, count in types:
    print(f"  {neuron_type}: {count}")

conn.close()
