# Visualization Quiz

## Executive Summary

Test your understanding of computational graph structure and how activation functions like ReLU affect gradient flow visualization.

---

## Quiz: Graph Structure and Gradient Flow

**Setup:**
```python
x = Value(3.0)
y = Value(-2.0)
z = x * y      # z = -6.0
w = z.relu()   # w = 0.0 (since z < 0)
w.backward()
```

**Question:** In the visualization of this computational graph, how many nodes will there be, and what will the gradient of `x` be after the backward pass?

**A)** 4 nodes; x.grad = -2.0  
**B)** 4 nodes; x.grad = 0.0  
**C)** 3 nodes; x.grad = -2.0  
**D)** 3 nodes; x.grad = 0.0  

---

## Answer & Explanation

**Correct Answer: B**

**Node Count: 4 nodes**
- `x` (input node)
- `y` (input node)  
- `z` (multiplication result)
- `w` (ReLU result)

Each operation creates a new Value object, so we have 4 total nodes.

**Gradient: x.grad = 0.0**

**ReLU Behavior:**
- **Forward**: `w = relu(z) = 0` (since z = -6 < 0)
- **Backward**: ReLU's gradient function is `(out.data > 0) * out.grad`
- Since `w.data = 0` (not > 0), the gradient is **blocked**

**Key Insight:** When ReLU outputs 0, it prevents gradients from flowing backward. Even though `x` affects `z`, the gradient can't reach `x` because ReLU "cuts off" the gradient flow when its input is negative.

This is why ReLU is called a "gating" function - it can turn neurons "off" during training by blocking their gradients.
