#!/usr/bin/env python3
"""
Experiment: Backward Pass Algorithm Step by Step
Watch how gradients flow through the graph during backward pass.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from micrograd.engine import Value

def print_gradients(nodes, step_name):
    """Helper to show current gradient state"""
    print(f"\n=== {step_name} ===")
    for name, node in nodes.items():
        print(f"{name}: data={node.data}, grad={node.grad}")

print("=== Backward Pass Algorithm Experiment ===\n")

# Step 1: Build the computation graph (forward pass)
print("Step 1: Forward pass - build the graph")
a = Value(2.0, _op='input')
b = Value(3.0, _op='input') 
c = a + b  # c = 5
d = c * 4  # d = 20

nodes = {'a': a, 'b': b, 'c': c, 'd': d}
print_gradients(nodes, "After Forward Pass")

print(f"\nGraph structure:")
print(f"d._prev = {[f'Value({v.data})' for v in d._prev]}")
print(f"c._prev = {[f'Value({v.data})' for v in c._prev]}")

# Step 2: Initialize output gradient
print("\nStep 2: Initialize output gradient")
d.grad = 1.0  # ∂d/∂d = 1
print_gradients(nodes, "After Initializing d.grad = 1")

# Step 3: Process d's backward function
print("\nStep 3: Process d._backward() - multiplication rule")
print("d = c * 4, so:")
print("  c.grad += 4 * d.grad = 4 * 1 = 4")
print("  Value(4).grad += c.data * d.grad = 5 * 1 = 5")

# Manually call d's backward (to show what happens)
c.grad += 4 * d.grad  # This is what d._backward() does
print_gradients(nodes, "After d._backward()")

# Step 4: Process c's backward function  
print("\nStep 4: Process c._backward() - addition rule")
print("c = a + b, so:")
print("  a.grad += c.grad = 4")
print("  b.grad += c.grad = 4")

# Manually call c's backward
a.grad += c.grad  # This is what c._backward() does
b.grad += c.grad
print_gradients(nodes, "After c._backward()")

print("\n=== Compare with Automatic backward() ===")

# Reset gradients and use automatic backward
for node in nodes.values():
    node.grad = 0

print("Resetting gradients and calling d.backward()...")
d.backward()
print_gradients(nodes, "After Automatic d.backward()")

print("\n=== Key Insights ===")
print("1. Backward pass visits nodes in reverse topological order: d → c → a,b")
print("2. Each node updates only its immediate parents' gradients")
print("3. Chain rule happens automatically through the traversal order")
print("4. Final gradients tell us: 'change input by 1 → output changes by gradient'")
print(f"5. Example: if a increases by 0.25, d increases by 0.25 * {a.grad} = {0.25 * a.grad}")
