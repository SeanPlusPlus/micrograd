#!/usr/bin/env python3
"""
Experiment: Building and Training a Single Neuron
Create a neuron from scratch and train it to learn a simple function.
"""

import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from micrograd.engine import Value


class Neuron:
    def __init__(self, nin):
        # Initialize random weights for each input
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        # Initialize random bias
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # Compute weighted sum: w₁*x₁ + w₂*x₂ + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Apply activation function
        out = act.relu()
        return out


print("=== Single Neuron Training Experiment ===\n")

# Step 2: Create a simple dataset
# Let's teach the neuron to learn: y = x1 + x2 (simple addition)
dataset = [
    ([1.0, 2.0], 3.0),  # 1 + 2 = 3
    ([0.0, 1.0], 1.0),  # 0 + 1 = 1
    ([-1.0, 2.0], 1.0),  # -1 + 2 = 1
    ([2.0, -1.0], 1.0),  # 2 + (-1) = 1
]

print("Dataset (trying to learn y = x1 + x2):")
for i, (inputs, target) in enumerate(dataset):
    print(f"  Example {i + 1}: {inputs} → {target}")

# Step 3: Create neuron (2 inputs since we have x1, x2)
neuron = Neuron(2)
print(f"\nCreated neuron with 2 inputs")
print(f"Initial weights: {[w.data for w in neuron.w]}")
print(f"Initial bias: {neuron.b.data}")

# Step 4: Test untrained neuron
print(f"\nUntrained neuron predictions:")
for i, (inputs, target) in enumerate(dataset):
    # Convert inputs to Value objects
    x = [Value(inp) for inp in inputs]
    prediction = neuron(x)
    print(f"  Input {inputs} → Predicted: {prediction.data:.3f}, Target: {target}")

# Step 5: Training loop
print(f"\nTraining...")
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Reset gradients
    for w in neuron.w:
        w.grad = 0
    neuron.b.grad = 0

    # Compute loss for all examples
    total_loss = Value(0)
    for inputs, target in dataset:
        x = [Value(inp) for inp in inputs]
        prediction = neuron(x)
        loss = (prediction - target) ** 2  # Mean squared error
        total_loss = total_loss + loss

    # Backward pass
    total_loss.backward()

    # Update parameters
    for w in neuron.w:
        w.data -= learning_rate * w.grad
    neuron.b.data -= learning_rate * neuron.b.grad

    # Print progress every 20 epochs
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: Loss = {total_loss.data:.4f}")

# Step 6: Test trained neuron
print(f"\nTrained neuron predictions:")
for i, (inputs, target) in enumerate(dataset):
    x = [Value(inp) for inp in inputs]
    prediction = neuron(x)
    print(f"  Input {inputs} → Predicted: {prediction.data:.3f}, Target: {target}")

print(f"\nFinal weights: {[w.data for w in neuron.w]}")
print(f"Final bias: {neuron.b.data}")
