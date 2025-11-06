# Section 3: Activation Functions

## Executive Summary

Activation functions determine when and how strongly neurons "fire" in response to inputs. They're the key to making neural networks capable of learning complex, non-linear patterns. Without them, even deep networks would be equivalent to simple linear regression.

## The Linear Problem

Consider a neuron without activation:
```
output = w1*x1 + w2*x2 + b
```

No matter how many of these you stack, the result is still linear. A network of linear functions can only learn linear relationships - useless for most real-world problems.

## What Activation Functions Do

Activation functions introduce **non-linearity**, allowing networks to:
- Learn complex patterns and decision boundaries
- Approximate any continuous function (universal approximation theorem)
- Create rich internal representations

## Common Activation Functions

### ReLU (Rectified Linear Unit)
```
relu(x) = max(0, x)
```

**Pros:**
- Simple and fast to compute
- Doesn't saturate for positive values
- Sparse activation (many neurons output 0)

**Cons:**
- "Dead neuron" problem - neurons can get stuck at 0
- Not differentiable at x=0 (though we treat derivative as 0)

### Sigmoid
```
sigmoid(x) = 1 / (1 + e^(-x))
```

**Pros:**
- Smooth, differentiable everywhere
- Output bounded between 0 and 1
- Historically important

**Cons:**
- Vanishing gradient problem for extreme values
- Not zero-centered (all outputs positive)
- Computationally expensive

### Tanh (Hyperbolic Tangent)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```

**Pros:**
- Zero-centered output (-1 to 1)
- Smooth and differentiable
- Stronger gradients than sigmoid

**Cons:**
- Still suffers from vanishing gradients
- Computationally expensive

## Choosing Activation Functions

**Hidden layers:** ReLU is the default choice
- Fast, simple, works well in practice
- Consider Leaky ReLU or Swish for dead neuron issues

**Output layer:** Depends on your problem
- Binary classification: Sigmoid
- Multi-class classification: Softmax
- Regression: Linear (no activation)

## Key Insights

1. **Non-linearity is essential** - without it, deep networks collapse to linear models
2. **ReLU dominance** - simple often beats complex in deep learning
3. **Context matters** - different layers may benefit from different activations
4. **Gradient flow** - activation choice affects how well gradients propagate during training

The choice of activation function can make or break your network's ability to learn effectively.
