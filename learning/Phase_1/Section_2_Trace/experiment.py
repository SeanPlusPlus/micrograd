#!/usr/bin/env python3
"""
Experiment: Tracing Through Forward Passes
Watch how the computational graph builds step by step during forward computation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from micrograd.engine import Value

def print_value_details(name, val):
    """Helper to print detailed info about a Value object"""
    print(f"{name}:")
    print(f"  data = {val.data}")
    print(f"  grad = {val.grad}")
    print(f"  _op = '{val._op}'")
    print(f"  _prev = {[f'Value({v.data})' for v in val._prev]}")
    print()

print("=== Forward Pass Tracing Experiment ===\n")

print("Step 1: Create input values (leaf nodes)")
a = Value(2.0)
b = Value(3.0)
print_value_details("a", a)
print_value_details("b", b)

print("Step 2: First operation - addition")
c = a + b
print(f"Executing: c = a + b = {a.data} + {b.data} = {c.data}")
print_value_details("c", c)

print("Step 3: Second operation - multiplication")
d = c * 4
print(f"Executing: d = c * 4 = {c.data} * 4 = {d.data}")
print_value_details("d", d)

print("Step 4: Third operation - power")
e = d ** 2
print(f"Executing: e = d ** 2 = {d.data} ** 2 = {e.data}")
print_value_details("e", e)

print("=== Forward Pass Complete ===")
print(f"Final result: {e.data}")
print(f"Computational path: a({a.data}) + b({b.data}) → c({c.data}) → d({d.data}) → e({e.data})")

print("\n=== Graph Structure Analysis ===")
print("Leaf nodes (no parents):", [f"a({a.data})", f"b({b.data})"])
print("Intermediate nodes:", [f"c({c.data})", f"d({d.data})"])
print("Final output:", f"e({e.data})")

print("\n=== Key Observations ===")
print("1. Each operation creates a new Value object")
print("2. Parent relationships are automatically tracked")
print("3. Operations are recorded for later gradient computation")
print("4. The graph grows dynamically as we compute")
print("5. Forward pass = building the graph + computing values")
