# Friday Architecture - Simply Explained

## 🏗️ The 3 Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                            │
│                                                              │
│  "What is a ship?" → [Chat/CLI] → Answer                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              GENERATIVE MODEL (Layer 3)                      │
│                                                              │
│  • Analyzes question                                        │
│  • Learns from activated neurons                            │
│  • Generates new answer                                     │
│                                                              │
│  Input: "What is a ship?"                                   │
│  Output: "A ship is a watercraft..."                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              3D NEURON SYSTEM (Layer 2)                      │
│                                                              │
│  • Stores knowledge in neurons                              │
│  • Finds similar neurons                                    │
│  • Activates related neurons                                │
│                                                              │
│  1000 neurons in 3D space:                                  │
│    Neuron #42: "SMS Zrínyi was a ship..." (Activation: 0.95)│
│    Neuron #87: "The ship sailed..." (Activation: 0.89)     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│           EMBEDDING MODEL (Layer 1 - Foundation)             │
│                                                              │
│  • Converts text to vectors (384 numbers)                   │
│  • Enables meaning comparison                               │
│  • Basis for everything else                                │
│                                                              │
│  "Ship" → [0.2, 0.5, 0.3, ..., 0.1]                        │
│  "Boat" → [0.21, 0.48, 0.31, ..., 0.09] (similar!)         │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 The Data Flow

### Training (Adding Knowledge)

```
1. WikiText Data
   "SMS Zrínyi was a ship built in 1910..."
   
2. ↓ Embedding Model
   Vector: [0.23, -0.45, 0.67, ..., 0.12]
   
3. ↓ 3D Neuron System
   Neuron #42 created @ Position (10, 20, 30)
   Synapses to similar neurons
   
4. ↓ Database
   Stored in SQLite
```

### Query (Answering Questions)

```
1. User asks
   "What is a ship?"
   
2. ↓ Embedding Model
   Query-Vector: [0.21, -0.43, 0.65, ..., 0.11]
   
3. ↓ 3D Neuron System
   Finds similar neurons:
   - Neuron #42: 0.95 similar ✓
   - Neuron #87: 0.89 similar ✓
   
4. ↓ Generative Model
   Learns from neurons:
   - Word patterns
   - Concepts
   - Context
   
5. ↓ Answer
   "A ship is a watercraft..."
```

## 🎯 Why This Architecture?

### Layer 1: Embedding Model
**Problem:** Computers don't understand words
**Solution:** Convert text to numbers

```
Without Embeddings:
"Ship" vs "Boat" → No similarity detectable ❌

With Embeddings:
[0.2, 0.5, ...] vs [0.21, 0.48, ...] → 95% similar! ✓
```

### Layer 2: 3D Neuron System
**Problem:** How to store knowledge efficiently?
**Solution:** 3D space with neurons and synapses

```
Advantages:
✓ Similar concepts are spatially close
✓ Synapses connect related knowledge
✓ Activation spreads (like in the brain)
✓ Transparent and traceable
```

### Layer 3: Generative Model
**Problem:** How to generate new answers?
**Solution:** Learn from neurons and generate

```
Not just retrieval:
❌ "Here's the stored text"

Real generation:
✓ Learns word patterns
✓ Understands context
✓ Generates new sentences
```

## 📊 Comparison with Other Systems

### Traditional Database

```
Database:
┌─────────────┐
│ Text Search │ → Finds only exact words
└─────────────┘

Friday:
┌─────────────┐
│ Embedding   │ → Understands meaning
│ 3D Neurons  │ → Finds similar concepts
│ Generation  │ → Creates new answers
└─────────────┘
```

### ChatGPT

```
ChatGPT:
┌──────────────────┐
│ Massive Model    │ → 175 billion parameters
│ (Black Box)      │ → Not traceable
└──────────────────┘

Friday:
┌──────────────────┐
│ 1000 Neurons     │ → Manageable
│ (Transparent)    │ → Every neuron visible
└──────────────────┘
```

## 🔍 Detailed Flow

### Example: "What is a ship?"

```
Step 1: Embedding
─────────────────────
Input:  "What is a ship?"
Model:  all-MiniLM-L6-v2
Output: [0.21, -0.43, 0.65, ..., 0.11]  (384 numbers)

Step 2: Neuron Search
─────────────────────
Query-Vector: [0.21, -0.43, 0.65, ...]

Compare with all 1000 neurons:
  Neuron #42: [0.23, -0.45, 0.67, ...]
    → Cosine-Similarity: 0.95 ✓
    → Text: "SMS Zrínyi was a ship..."
    
  Neuron #87: [0.20, -0.41, 0.63, ...]
    → Cosine-Similarity: 0.89 ✓
    → Text: "The ship sailed across..."
    
  Neuron #123: [0.8, 0.2, -0.1, ...]
    → Cosine-Similarity: 0.12 ✗
    → Text: "Gold was used for coins..."

Top 5 neurons selected

Step 3: Activation
─────────────────────
Neuron #42 (0.95) activated
  → Synapse to Neuron #87 (0.8 weight)
  → Neuron #87 also activated (0.89 * 0.8 = 0.71)

Activated neurons:
  #42: 0.95
  #87: 0.89
  #91: 0.71
  #103: 0.68
  #156: 0.65

Step 4: Context Extraction
─────────────────────
From activated neurons:
  Key-Words: ship, vessel, water, sail, built
  Tags: history, wikitext, knowledge
  Concepts: maritime, transportation

Step 5: Learn Word Patterns
─────────────────────
From neuron texts:
  "ship" → "was" (0.8)
  "was" → "a" (0.9)
  "a" → "vessel" (0.7)
  "vessel" → "used" (0.6)

Step 6: Generation
─────────────────────
Question-Type: definition
Starter: "It is"

Generate with word patterns:
  "It is" + "a" + "vessel" + "used" + "for" + ...

Cleanup & formatting:
  "It is a vessel used for transportation on water."

Step 7: Output
─────────────────────
Answer: "It is a vessel used for transportation on water."
```

## 💡 The Role of Each Component

### Embedding Model (Sentence-Transformers)
```
Task:    Text → Numbers
Why:     Computers can only work with numbers
How:     Trained on millions of texts
Result:  Meaning captured in vector
```

### 3D Neuron System
```
Task:    Store and find knowledge
Why:     Efficient organization of knowledge
How:     Spatial arrangement + synapses
Result:  Similar knowledge is close together
```

### Generative Model
```
Task:    Create new answers
Why:     Not just return stored texts
How:     Learns patterns from neurons
Result:  Real generation, not just retrieval
```

## 🎓 Summary

**Friday is a 3-layer system:**

1. **Embedding Model** (Foundation)
   - Converts text to vectors
   - Enables semantic search
   - Basis for everything

2. **3D Neuron System** (Storage)
   - Organizes knowledge spatially
   - Finds similar concepts
   - Transparent and traceable

3. **Generative Model** (Intelligence)
   - Learns from neurons
   - Generates new answers
   - Real AI capabilities

**Each layer is essential!**
- Without embeddings: No semantic search
- Without neurons: No knowledge storage
- Without generation: Only retrieval, no intelligence

---

**Further Documentation:**
- [How Friday Works](HOW_FRIDAY_WORKS_EN.md)
- [Embeddings Explained](EMBEDDING_EXPLAINED_EN.md)
- [README](../README.md)
