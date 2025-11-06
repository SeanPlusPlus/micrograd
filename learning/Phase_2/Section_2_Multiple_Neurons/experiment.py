#!/usr/bin/env python3
"""
Experiment: Multiple Neurons Working Together
Building on single neuron limitations, create layers of neurons to solve complex problems.
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
    # Original examples
    ([1.0, 2.0], 3.0),
    ([0.0, 1.0], 1.0),
    ([-1.0, 2.0], 1.0),
    ([2.0, -1.0], 1.0),
    # More positive examples
    ([3.0, 1.0], 4.0),
    ([2.0, 2.0], 4.0),
    ([1.0, 3.0], 4.0),
    ([4.0, 1.0], 5.0),
    ([2.5, 1.5], 4.0),
    # More negative examples
    ([-2.0, 3.0], 1.0),
    ([-1.0, -1.0], -2.0),
    ([-3.0, 2.0], -1.0),
    ([1.0, -2.0], -1.0),
    # Zero examples
    ([0.0, 0.0], 0.0),
    ([5.0, -5.0], 0.0),
    ([-3.0, 3.0], 0.0),
    # Decimal examples
    ([0.5, 0.5], 1.0),
    ([1.2, 2.3], 3.5),
    ([0.1, 0.9], 1.0),
    ([-0.5, 1.5], 1.0),
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

print("\n" + "=" * 50)
print("SECTION 2: MULTIPLE NEURONS")
print("=" * 50)


# Simple Layer class - just multiple neurons working in parallel
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]


# Create layer with 3 neurons (more chances to learn!)
print("Creating layer with 3 neurons...")
layer = Layer(2, 3)

# Test what 3 untrained neurons predict
print("Untrained layer predictions:")
for inputs, target in dataset:
    x = [Value(inp) for inp in inputs]
    outputs = layer(x)
    combined = sum(outputs)  # Simple combination: add all outputs
    print(
        f"  {inputs} → [{outputs[0].data:.3f}, {outputs[1].data:.3f}, {outputs[2].data:.3f}] → Combined: {combined.data:.3f} (target: {target})"
    )

# Train the layer (all 3 neurons together)
print(f"\nTraining layer with 3 neurons...")
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    # Reset gradients for all neurons
    for neuron in layer.neurons:
        for w in neuron.w:
            w.grad = 0
        neuron.b.grad = 0

    # Compute loss for all examples
    total_loss = Value(0)
    for inputs, target in dataset:
        x = [Value(inp) for inp in inputs]
        outputs = layer(x)
        combined = sum(outputs)  # Combine all neuron outputs
        loss = (combined - target) ** 2
        total_loss = total_loss + loss

    # Backward pass
    total_loss.backward()

    # Update all neuron parameters
    for neuron in layer.neurons:
        for w in neuron.w:
            w.data -= learning_rate * w.grad
        neuron.b.data -= learning_rate * neuron.b.grad

    # Print progress
    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: Loss = {total_loss.data:.4f}")

# Test trained layer
print(f"\nTrained layer predictions:")
for inputs, target in dataset:
    x = [Value(inp) for inp in inputs]
    outputs = layer(x)
    combined = sum(outputs)
    print(
        f"  {inputs} → [{outputs[0].data:.3f}, {outputs[1].data:.3f}, {outputs[2].data:.3f}] → Combined: {combined.data:.3f} (target: {target})"
    )
