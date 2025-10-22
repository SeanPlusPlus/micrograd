# Value Objects Quiz

## Executive Summary

Test your understanding of how Value objects compute gradients through automatic differentiation. These quizzes cover the core concepts of gradient flow through addition and multiplication operations.

---

## Quiz 1: Basic Gradient Flow

**Setup:**
```python
x = Value(3)
y = Value(2)
z = x * y     # z = 6
w = z + 5     # w = 11
w.backward()
```

**Questions:**
1. What will `w.grad` be? Why?
2. What will `z.grad` be?
3. What will `x.grad` be?
4. What will `y.grad` be?
5. **Bonus**: If I want `w` to increase from 11 to 12, and I can only change `x`, what should I change `x` to?

---

## Quiz 2: Multi-Step Chain

**Setup:**
```python
a = Value(2)
b = Value(4)
c = a + b     # c = 6
d = c * 3     # d = 18
e = d + 10    # e = 28
f = e * 2     # f = 56
f.backward()
```

**DAG:**
```
a(2) ──┐
       ├─→ c(6) ──→ d(18) ──→ e(28) ──→ f(56)
b(4) ──┘      ×3        +10       ×2
```

**Questions:**
1. What's `f.grad`?
2. What's `e.grad`?
3. What's `d.grad`?
4. What's `c.grad`?
5. What are `a.grad` and `b.grad`?
6. **Challenge**: If I want `f` to increase by 6, and I can only change `a`, what should `a` become?

---

## Answers & Explanations

### Quiz 1 Answers

1. **`w.grad = 1`** - Always 1 for the final output node by definition.

2. **`z.grad = 1`** - Since `w = z + 5`, if `z` increases by 1, `w` increases by 1. Addition passes gradients through unchanged.

3. **`x.grad = 2`** - Since `z = x * y` where `y = 2`, if `x` increases by 1, `z` increases by 2, so `w` increases by 2. In multiplication, each input's gradient equals the other input's value.

4. **`y.grad = 3`** - Since `z = x * y` where `x = 3`, if `y` increases by 1, `z` increases by 3, so `w` increases by 3.

5. **`x = 3.5`** - Since `x.grad = 2`, changing `x` by 0.5 changes `w` by 1. So `x` goes from 3 to 3.5.

### Quiz 2 Answers

1. **`f.grad = 1`** - Final output node.

2. **`e.grad = 2`** - Since `f = e * 2`, if `e` increases by 1, `f` increases by 2.

3. **`d.grad = 2`** - Since `e = d + 10`, if `d` increases by 1, `e` increases by 1, then `f` increases by 2.

4. **`c.grad = 6`** - Since `d = c * 3`, if `c` increases by 1, `d` increases by 3, `e` increases by 3, `f` increases by 6.

5. **`a.grad = 6` and `b.grad = 6`** - Since `c = a + b`, both inputs get the same gradient as `c`. Addition passes gradients through unchanged to all inputs.

6. **`a = 3`** - Since `a.grad = 6`, changing `a` by 1 changes `f` by 6. So `a` goes from 2 to 3.

---

## Key Patterns

### Addition Operations
- **Rule**: Pass gradient through unchanged to all inputs
- **Example**: If `c = a + b` and `c.grad = 5`, then `a.grad = 5` and `b.grad = 5`

### Multiplication Operations  
- **Rule**: Each input's gradient = (other input's value) × (output's gradient)
- **Example**: If `c = a * b` where `a = 3, b = 2` and `c.grad = 5`, then:
  - `a.grad = 2 * 5 = 10`
  - `b.grad = 3 * 5 = 15`

### Chain Rule
Gradients flow backward through the computational graph, multiplying through multiplication operations and passing unchanged through addition operations.
