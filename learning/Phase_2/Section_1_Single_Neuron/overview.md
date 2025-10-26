# Section 1: Building a Single Neuron

## Executive Summary

A neuron is the fundamental building block of neural networks. Using the Value objects and automatic differentiation from Phase 1, we'll build a single neuron from scratch, understanding how it processes inputs, applies weights and biases, and produces outputs that can learn from data.

## What is a Neuron?

A neuron is a computational unit that:
1. **Takes multiple inputs** (x₁, x₂, x₃, ...)
2. **Applies weights** (w₁, w₂, w₃, ...) to each input
3. **Adds a bias** (b) 
4. **Applies an activation function** (like ReLU or tanh)
5. **Produces a single output**

**Mathematical formula:**
```
output = activation(w₁*x₁ + w₂*x₂ + w₃*x₃ + ... + b)
```

## Neuron Components

### Weights (Parameters)
- **Purpose**: Control how much each input influences the output
- **Learning**: Adjusted during training to minimize loss
- **Implementation**: `Value` objects that accumulate gradients

### Bias (Parameter)  
- **Purpose**: Allows the neuron to activate even when all inputs are zero
- **Learning**: Also adjusted during training
- **Implementation**: Single `Value` object

### Activation Function
- **Purpose**: Introduces non-linearity (enables complex patterns)
- **Common types**: ReLU, tanh, sigmoid
- **Implementation**: Methods on `Value` objects

## Building a Neuron with Value Objects

```python
class Neuron:
    def __init__(self, nin):
        # Initialize random weights for each input
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        # Initialize random bias
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, x):
        # Compute weighted sum: w₁*x₁ + w₂*x₂ + ... + b
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        # Apply activation function
        out = act.tanh()
        return out
```

## Key Insights

**Automatic Differentiation Integration**: Since weights and bias are `Value` objects, gradients flow through them automatically during backpropagation.

**Parameter Learning**: The neuron "learns" by adjusting its weights and bias based on gradients to minimize prediction errors.

**Non-linearity is Crucial**: Without activation functions, multiple neurons would just be linear combinations - no more powerful than a single linear function.

**Scalability**: The same principles that work for one neuron scale to millions of neurons in deep networks.

## From Single Neuron to Networks

Understanding a single neuron is the foundation for:
- **Layers**: Collections of neurons processing inputs in parallel
- **Multi-layer networks**: Stacking layers to learn complex patterns  
- **Deep learning**: Many layers creating hierarchical representations

## Practical Applications

**Classification**: Neuron output can represent class probabilities
**Regression**: Neuron output can predict continuous values
**Feature Detection**: Each neuron can learn to detect specific patterns in data

## Next Steps

After mastering single neurons, we'll explore how multiple neurons work together in layers, and how layers combine to form powerful neural networks capable of learning complex tasks.
