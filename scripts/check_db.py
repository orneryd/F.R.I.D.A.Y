import sqlite3

conn = sqlite3.connect('data/test_persistence.db')
cursor = conn.cursor()

# Check tables
cursor.execute('SELECT name FROM sqlite_master WHERE type="table"')
tables = [row[0] for row in cursor.fetchall()]
print(f'Tables: {tables}')

# Check neurons
cursor.execute('SELECT COUNT(*) FROM neurons')
neuron_count = cursor.fetchone()[0]
print(f'Neurons in DB: {neuron_count}')

# Check synapses
cursor.execute('SELECT COUNT(*) FROM synapses')
synapse_count = cursor.fetchone()[0]
print(f'Synapses in DB: {synapse_count}')

conn.close()
