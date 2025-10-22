#!/usr/bin/env python3
"""
Hands-on experiment to understand Value objects.
Run this and watch how the computational graph builds up.
"""

from micrograd.engine import Value

print("=== Understanding Value Objects ===\n")

# Step 1: Create two simple values
print("Step 1: Create values")
a = Value(2.0)
b = Value(3.0)
print(f"a = {a}")
print(f"b = {b}")
print(f"a._prev = {a._prev}")  # Empty - these are leaf nodes
print(f"b._prev = {b._prev}")  # Empty - these are leaf nodes
print()

# Step 2: Add them
print("Step 2: Add a + b")
c = a + b
print(f"c = {c}")
print(f"c._prev = {c._prev}")  # Points back to a and b!
print(f"c._op = '{c._op}'")    # Shows it came from addition
print()

# Step 3: Multiply by 4
print("Step 3: Multiply c * 4")
d = c * 4
print(f"d = {d}")
print(f"d._prev = {d._prev}")  # Points back to c and Value(4)
print(f"d._op = '{d._op}'")    # Shows it came from multiplication
print()

# Step 4: The magic - compute gradients!
print("Step 4: Compute gradients with d.backward()")
print("Before backward:")
print(f"  a.grad = {a.grad}")
print(f"  b.grad = {b.grad}")
print(f"  c.grad = {c.grad}")
print(f"  d.grad = {d.grad}")

d.backward()

print("\nAfter backward:")
print(f"  a.grad = {a.grad}")  # How much d changes if we nudge a
print(f"  b.grad = {b.grad}")  # How much d changes if we nudge b
print(f"  c.grad = {c.grad}")  # How much d changes if we nudge c
print(f"  d.grad = {d.grad}")  # Always 1 for the final output

print("\n=== Key Insight ===")
print("Each Value remembers its parents through ._prev")
print("This creates a graph that backward() can walk through!")
print("The gradients tell us: 'if I want d to increase by 1, how much should each input change?'")
