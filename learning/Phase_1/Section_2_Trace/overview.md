# Section 2: Tracing Through a Forward Pass

## Executive Summary

A forward pass is the process of computing outputs from inputs by following the computational graph from left to right. Understanding how values flow forward through operations is essential before learning how gradients flow backward.

## What is a Forward Pass?

When you execute code like:
```python
a = Value(2)
b = Value(3)
c = a + b
d = c * 4
```

You're performing a **forward pass** - computing the final result by following the operations in sequence.

## The Step-by-Step Process

### 1. Start with Input Values
- `a = Value(2)` - creates a leaf node with data=2
- `b = Value(3)` - creates another leaf node with data=3

### 2. Apply Operations Sequentially
- `c = a + b` - creates new Value(5) with parents {a, b}
- `d = c * 4` - creates new Value(20) with parents {c, Value(4)}

### 3. Build the Computational Graph
Each operation creates a new node that remembers:
- Its computed value (`data`)
- Its parent nodes (`_prev`)
- The operation that created it (`_op`)
- How to compute gradients (`_backward`)

## Key Insights

**Forward Pass = Value Computation**
- We compute actual numeric results
- We build the graph structure as we go
- Each node stores everything needed for later gradient computation

**Graph Building is Automatic**
- Every operation creates new nodes
- Parent relationships are tracked automatically
- The graph grows dynamically as you compute

**Preparation for Backward Pass**
- Forward pass sets up everything needed for gradients
- The `_backward` functions are defined but not called yet
- Graph structure determines gradient flow path

## Next Steps

After understanding forward passes, we'll see how the same graph enables automatic gradient computation through the backward pass.
