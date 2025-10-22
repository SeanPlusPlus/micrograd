# Backward Pass Quiz

## Executive Summary

Test your understanding of the backward pass algorithm - how gradients flow through the computational graph in reverse topological order.

---

## Quiz: Backward Pass Algorithm

**Setup:**
```python
x = Value(3)
y = Value(2)
z = x * y     # z = 6
w = z + 4     # w = 10
w.backward()
```

**Question:** In what order does the backward pass visit the nodes, and what does each node's `_backward()` function update?

**A)** Visit order: x → y → z → w; each node updates its own gradient  
**B)** Visit order: w → z → x,y; each node updates its parents' gradients  
**C)** Visit order: x,y → z → w; each node updates its children's gradients  
**D)** Visit order: w → z → x,y; each node updates all previous nodes' gradients  

---

## Answer & Explanation

**Correct Answer: B**

**Explanation:**
- **Visit order: w → z → x,y** - Reverse topological order (output to inputs)
- **Each node updates its parents' gradients** - Local updates only

**Detailed breakdown:**
1. **Visit w**: `w._backward()` updates `z.grad` and `Value(4).grad` (w's parents)
2. **Visit z**: `z._backward()` updates `x.grad` and `y.grad` (z's parents)  
3. **Visit x,y**: Leaf nodes, their `_backward()` does nothing

**Why other answers are wrong:**
- **A**: Wrong direction (forward, not backward) and wrong update target
- **C**: Wrong direction and wrong update target (children don't exist in backward pass)
- **D**: Correct order but wrong scope (only immediate parents, not all previous nodes)

**Key Insight:** The backward pass uses reverse topological order to ensure each node's gradient is computed before it's used to compute its parents' gradients. Each `_backward()` function only updates its immediate parents using the chain rule.
