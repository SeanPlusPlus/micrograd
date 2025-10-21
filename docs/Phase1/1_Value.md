# Understanding the Value Class

## Executive Summary

The `Value` class is a "smart number" that tracks both its numeric value and how to compute gradients. Each `Value` maintains backward pointers to its parent nodes, creating a computational graph that remembers the exact path of computation. This structure is the foundation of automatic differentiation - when you call `backward()`, it follows these pointers in reverse to backpropagate gradients through the entire computation.

## What is a Value Object?

A `Value` is like a regular number, but with superpowers:

```python
# Regular Python
a = 2
b = 3
c = a + b  # c = 5, that's it

# With Value objects
a = Value(2)
b = Value(3) 
c = a + b  # c = Value(5), but it REMEMBERS it came from a + b
```

## The Computational Graph Structure

Think of it like a **backward-pointing linked list**, but with multiple parents:

```python
a = Value(2)
b = Value(3) 
c = a + b      # c._prev = {a, b}
d = c * 4      # d._prev = {c}
```

Creates this **Directed Acyclic Graph (DAG)**:
```
a(2) ──┐
       ├─→ c(5) ──┐
b(3) ──┘          ├─→ d(20)
            4 ────┘
```

Where:
- `c = a + b = 2 + 3 = 5`
- `d = c * 4 = 5 * 4 = 20`

Key properties:
- **Multiple parents**: `c` has two parents (`a` and `b`)
- **Backward pointing**: arrows point from result back to inputs  
- **No cycles**: can't have circular dependencies
- **Computation history**: each node remembers how it was created

## What Each Value Stores

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data           # The actual number (e.g., 5.0)
        self.grad = 0             # Gradient (starts at 0)
        self._prev = set(_children)  # Parent nodes that created this
        self._op = _op            # Operation name ('+', '*', etc.)
        self._backward = lambda: None  # Function to compute gradients
```

## How Operations Build the Graph

### Addition Example
```python
def __add__(self, other):
    # Create new Value with sum
    out = Value(self.data + other.data, (self, other), '+')
    
    # Define gradient computation
    def _backward():
        self.grad += out.grad    # Gradient passes through unchanged
        other.grad += out.grad   # Both inputs get same gradient
    out._backward = _backward
    
    return out
```

**Key insight**: Addition just passes gradients through unchanged to both inputs.

### Multiplication Example  
```python
def __mul__(self, other):
    # Create new Value with product
    out = Value(self.data * other.data, (self, other), '*')
    
    # Define gradient computation using calculus: d(a*b)/da = b
    def _backward():
        self.grad += other.data * out.grad  # My gradient = other's value
        other.grad += self.data * out.grad  # Other's gradient = my value
    out._backward = _backward
    
    return out
```

**Key insight**: Multiplication gradients use the "other" input's value (basic calculus).

## Why This Matters

This structure lets us:
1. **Build complex computations** from simple operations
2. **Automatically compute gradients** for any computation
3. **Train neural networks** by following gradients to minimize loss

The same pattern scales from simple math to massive transformer models - it's all just `Value` objects connected in a graph!

## Next Steps

- Trace through a complete forward and backward pass
- See how `backward()` walks the graph
- Build a simple neuron using these operations
