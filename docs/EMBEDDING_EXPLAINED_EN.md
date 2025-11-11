# Embeddings Simply Explained

## 🎯 The Problem

Computers don't understand words, only numbers!

```
Computer sees:
"Ship" → ??? (What is this?)
"Boat" → ??? (What is this?)
"Car"  → ??? (What is this?)
```

## ✨ The Solution: Embeddings

Embeddings convert words into numbers that capture the **meaning**:

```
"Ship" → [0.2, 0.5, 0.3, 0.1, ...]  (384 numbers)
"Boat" → [0.21, 0.48, 0.31, 0.09, ...]  (384 numbers)
"Car"  → [0.8, -0.2, 0.1, 0.5, ...]  (384 numbers)
```

## 🔍 Why is This Useful?

### Example 1: Recognizing Similarity

```python
# Without embeddings (word comparison):
"Ship" == "Boat"  → False ❌
"Ship" == "Ship"  → True ✓

# With embeddings (meaning comparison):
similarity("Ship", "Boat")  → 0.95 (95% similar!) ✓
similarity("Ship", "Car")   → 0.23 (23% similar)
```

### Example 2: Different Languages

```python
# Without embeddings:
"Ship" == "Schiff"  → False ❌

# With embeddings:
similarity("Ship", "Schiff")  → 0.98 (98% similar!) ✓
# Same meaning = Similar vectors!
```

### Example 3: Recognizing Synonyms

```python
# Without embeddings:
"big" == "huge"  → False ❌

# With embeddings:
similarity("big", "huge")  → 0.87 (87% similar!) ✓
```

## 📐 How Does It Work?

### Step 1: Training (already done!)

The embedding model was trained on millions of texts:

```
Training data:
"The ship sails on the sea"
"A boat floats in the water"
"The car drives on the road"
...millions more sentences...

→ Model learns:
  - "Ship" and "Boat" appear in similar contexts
  - "Ship" and "Sea" belong together
  - "Car" and "Road" belong together
```

### Step 2: Encoding (what we use!)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Text → Vector
vector = model.encode("A ship sails on the sea")
# → [0.23, -0.45, 0.67, ..., 0.12]  # 384 numbers
```

### Step 3: Comparing

```python
import numpy as np

# Convert two texts to vectors
v1 = model.encode("A ship")
v2 = model.encode("A boat")

# Calculate cosine-similarity
similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
# → 0.95 (very similar!)
```

## 🎨 Visualization

Imagine each word is a point in 3D space:

```
        Z
        ↑
        │
        │    ● Ship
        │   ╱│
        │  ╱ │● Boat
        │ ╱  │╱
        │╱___│________→ X
       ╱     │
      ╱      │
     ╱       │
    ↙        ↓
   Y      ● Car
```

- **Ship** and **Boat** are close together (similar meaning)
- **Car** is far away (different meaning)

**In reality:** 384 dimensions instead of just 3!

## 🔢 The Numbers in Detail

### What Do the 384 Numbers Mean?

Each dimension captures an aspect of meaning:

```
Dimension 1: "Is it a vehicle?" → 0.8 (yes!)
Dimension 2: "Is it on water?" → 0.9 (yes!)
Dimension 3: "Is it modern?" → 0.3 (maybe)
Dimension 4: "Is it large?" → 0.7 (rather yes)
...
Dimension 384: "Is it historical?" → 0.5 (neutral)
```

**Important:** These dimensions are not defined by humans, but learned by the model!

## 🚀 How Friday Uses This

### 1. Training: Storing Knowledge

```
Text: "SMS Zrínyi was a ship built in 1910"
  ↓ Embedding Model
Vector: [0.23, -0.45, 0.67, ..., 0.12]
  ↓
Create Neuron #42 with this vector
```

### 2. Query: Finding Knowledge

```
Question: "What is a ship?"
  ↓ Embedding Model
Query-Vector: [0.21, -0.43, 0.65, ..., 0.11]
  ↓
Compare with all neurons:
  Neuron #42: Similarity = 0.95 ✓ (very relevant!)
  Neuron #87: Similarity = 0.89 ✓ (relevant)
  Neuron #123: Similarity = 0.12 ✗ (not relevant)
  ↓
Use Neurons #42 and #87 for answer
```

## 💡 Why is This Better Than Word Search?

### Word Search (old):

```
Question: "What is a ship?"
Search for: "ship"

Finds:
✓ "The ship sails..."
✗ "A boat floats..." (boat ≠ ship)
✗ "The ship sails..." (ship ≠ ship in German)
```

### Embedding Search (new):

```
Question: "What is a ship?"
Embedding: [0.2, 0.5, 0.3, ...]

Finds:
✓ "The ship sails..." (0.98 similar)
✓ "A boat floats..." (0.95 similar - synonym!)
✓ "Das Schiff fährt..." (0.97 similar - translation!)
```

## 🎓 Technical Details

### Cosine-Similarity

How to calculate similarity between two vectors:

```python
def cosine_similarity(v1, v2):
    # Dot product
    dot_product = sum(a * b for a, b in zip(v1, v2))
    
    # Vector lengths
    length_v1 = sqrt(sum(a * a for a in v1))
    length_v2 = sqrt(sum(b * b for b in v2))
    
    # Cosine-Similarity
    return dot_product / (length_v1 * length_v2)
```

**Result:**
- 1.0 = Identical
- 0.9 = Very similar
- 0.5 = Somewhat similar
- 0.0 = Not similar
- -1.0 = Opposite

### Why 384 Dimensions?

```
Fewer dimensions (e.g. 50):
  ✓ Faster
  ✗ Less precise

More dimensions (e.g. 1024):
  ✓ More precise
  ✗ Slower
  ✗ More memory

384 dimensions:
  ✓ Good compromise!
  ✓ Fast enough
  ✓ Precise enough
```

## 🔬 Example Code

### Creating Embeddings

```python
from sentence_transformers import SentenceTransformer

# Load model (once)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert texts to vectors
texts = [
    "A ship sails on the sea",
    "A boat floats in the water",
    "A car drives on the road"
]

embeddings = model.encode(texts)
# → 3 vectors with 384 numbers each
```

### Calculating Similarity

```python
from sklearn.metrics.pairwise import cosine_similarity

# Similarity between all texts
similarities = cosine_similarity(embeddings)

print(similarities)
# [[1.00, 0.95, 0.23],   # Ship vs. all
#  [0.95, 1.00, 0.21],   # Boat vs. all
#  [0.23, 0.21, 1.00]]   # Car vs. all
```

### Using in Friday

```python
# 1. Training: Store text
text = "SMS Zrínyi was a ship"
vector = compression_engine.compress(text)
neuron = KnowledgeNeuron(source_data=text, vector=vector)
graph.add_neuron(neuron)

# 2. Query: Find similar neurons
query = "What is a ship?"
query_vector = compression_engine.compress(query)
results = query_engine.query(query_vector, top_k=5)

# 3. Generate answer
response = generative_model.generate_response(query, results)
```

## 📚 Summary

**Embeddings are:**
- Numerical representations of text
- Capture meaning, not just words
- Enable semantic search
- The key to Friday's intelligence

**Without Embeddings:**
- Only word comparison possible
- No synonyms recognized
- No translations recognized
- No semantic search

**With Embeddings:**
- Meaning comparison possible ✓
- Synonyms recognized ✓
- Translations recognized ✓
- Semantic search ✓

**The Embedding Model is the foundation of Friday!**

---

**Further Resources:**
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [How Friday Works](HOW_FRIDAY_WORKS_EN.md)
- [Cosine-Similarity Explained](https://en.wikipedia.org/wiki/Cosine_similarity)
