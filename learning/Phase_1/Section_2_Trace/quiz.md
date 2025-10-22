# Forward Pass Quiz

## Executive Summary

Test your understanding of how computational graphs build during forward computation. Focus on the timing and structure of graph construction.

---

## Quiz: Graph Building During Forward Pass

**Setup:**
```python
a = Value(2)
b = Value(3)
c = a * b
d = c + 5
e = d ** 2
```

**Question:** After executing `c = a * b`, how many Value objects exist in total, and what does `c._prev` contain?

**A)** 2 objects total; `c._prev = set()`  
**B)** 3 objects total; `c._prev = {a, b}`  
**C)** 3 objects total; `c._prev = {Value(2), Value(3)}`  
**D)** 4 objects total; `c._prev = {a, b}`  

---

## Answer & Explanation

**Correct Answer: B**

**Explanation:**
- **3 objects total**: `a`, `b`, and the newly created `c`
- **`c._prev = {a, b}`**: The multiplication operation creates `c` with parent pointers to the exact `a` and `b` objects (not copies)

**Key Insight:** Each operation immediately creates a new Value object that links back to its input objects. The graph builds dynamically during forward computation, not before or after.

**Why other answers are wrong:**
- **A**: `c._prev` would be empty only for leaf nodes (inputs with no parents)
- **C**: `c._prev` contains references to the actual `a` and `b` objects, not new Value objects
- **D**: Only 3 objects exist at this point; `d` and `e` haven't been created yet
