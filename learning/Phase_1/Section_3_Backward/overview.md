# Section 3: Backward Pass - How Gradients Flow

## Executive Summary

The backward pass is where automatic differentiation performs its magic. After the forward pass builds the computational graph, `backward()` walks through it in reverse topological order, calling each node's `_backward()` function to compute and accumulate gradients using the chain rule.

## What is the Backward Pass?

While the forward pass computes values from inputs to outputs, the backward pass computes **gradients** from outputs back to inputs. It answers: "How much does each input need to change to affect the final output?"

## The Algorithm

### 1. Topological Sort
```python
def backward(self):
    # Build topological ordering of all nodes
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:  # Visit parents first
                build_topo(child)
            topo.append(v)        # Add self after parents
    build_topo(self)
```

**Key insight**: We must process nodes in the right order - a node can only compute its gradient after all nodes that depend on it have computed theirs.

### 2. Gradient Initialization
```python
self.grad = 1  # Output gradient is always 1
```

The final output's gradient is 1 by definition (∂output/∂output = 1).

### 3. Chain Rule Application
```python
for v in reversed(topo):  # Process in reverse topological order
    v._backward()         # Call each node's gradient function
```

Each `_backward()` function implements the chain rule for that specific operation.

## How _backward() Functions Work

### Addition: Pass Through
```python
def _backward():
    self.grad += out.grad    # ∂out/∂self = 1
    other.grad += out.grad   # ∂out/∂other = 1
```

### Multiplication: Use Other Input
```python
def _backward():
    self.grad += other.data * out.grad  # ∂(a*b)/∂a = b
    other.grad += self.data * out.grad  # ∂(a*b)/∂b = a
```

### Power: Use Calculus
```python
def _backward():
    self.grad += (n * self.data**(n-1)) * out.grad  # ∂(x^n)/∂x = n*x^(n-1)
```

## The Chain Rule in Action

For a computation like `f = (a + b) * c`:

1. **Forward**: Build graph and compute values
2. **Backward**: 
   - `f.grad = 1` (given)
   - `∂f/∂(a+b) = c`, `∂f/∂c = (a+b)` (multiplication rule)
   - `∂f/∂a = ∂f/∂(a+b) * ∂(a+b)/∂a = c * 1 = c` (chain rule)
   - `∂f/∂b = ∂f/∂(a+b) * ∂(a+b)/∂b = c * 1 = c` (chain rule)

## Key Insights

**Automatic Chain Rule**: Each `_backward()` function only needs to know the local derivative. The chain rule is applied automatically by the graph traversal.

**Gradient Accumulation**: We use `+=` not `=` because a node might be used multiple times in the computation (like `a + a`).

**Topological Order**: Critical for correctness - we must process nodes after all their dependents have been processed.

**Local + Global = Magic**: Each operation implements simple local rules, but the graph structure enables computing gradients for arbitrarily complex functions.

## Next Steps

Understanding backward pass is the key to understanding how neural networks learn - they use these gradients to adjust weights and minimize loss functions.
