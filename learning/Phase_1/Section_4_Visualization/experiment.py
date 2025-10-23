#!/usr/bin/env python3
"""
Experiment: Simple Computational Graph Visualization
Create and visualize a basic computation to see the graph structure.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from micrograd.engine import Value

def draw_simple_graph(root, filename="graph"):
    """
    Simple text-based graph visualization
    Shows nodes and their connections in a readable format
    """
    nodes = []
    edges = []
    visited = set()
    
    def collect_nodes(v):
        if v not in visited:
            visited.add(v)
            nodes.append(v)
            for child in v._prev:
                edges.append((child, v))
                collect_nodes(child)
    
    collect_nodes(root)
    
    print(f"\n=== Graph Visualization: {filename} ===")
    print("\nNodes:")
    for i, node in enumerate(nodes):
        op_str = f" [{node._op}]" if node._op else " [input]"
        print(f"  {i}: Value(data={node.data:.2f}, grad={node.grad:.2f}){op_str}")
    
    print("\nConnections:")
    for parent, child in edges:
        parent_idx = nodes.index(parent)
        child_idx = nodes.index(child)
        print(f"  Node {parent_idx} → Node {child_idx}")
    
    return nodes, edges

print("=== Visualization Experiment ===\n")

# Create a simple computation
print("Step 1: Create computation")
a = Value(2.0)
b = Value(3.0)
c = a + b      # c = 5
d = c * 4      # d = 20
e = d.relu()   # e = 20 (since d > 0)

print(f"a = {a.data}")
print(f"b = {b.data}")
print(f"c = a + b = {c.data}")
print(f"d = c * 4 = {d.data}")
print(f"e = d.relu() = {e.data}")

# Visualize before gradients
draw_simple_graph(e, "before_backward")

print("\nStep 2: Compute gradients")
e.backward()

print(f"After backward pass:")
print(f"a.grad = {a.grad}")
print(f"b.grad = {b.grad}")
print(f"c.grad = {c.grad}")
print(f"d.grad = {d.grad}")
print(f"e.grad = {e.grad}")

# Visualize after gradients
draw_simple_graph(e, "after_backward")

print("\n=== Graph Analysis ===")
print("1. The graph shows how each operation creates new nodes")
print("2. Connections show the parent-child relationships")
print("3. Gradients flow backward through these same connections")
print("4. Each node stores both its value (forward) and gradient (backward)")

print("\n=== Try This ===")
print("Modify the computation above and run again to see different graphs!")
print("Examples:")
print("- Add more operations: f = e ** 2")
print("- Use different values: a = Value(-1.0)")
print("- Try different operations: tanh, sigmoid, etc.")
