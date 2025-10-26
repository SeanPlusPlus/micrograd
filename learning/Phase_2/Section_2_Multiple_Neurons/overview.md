# Section 2: Multiple Neurons Working Together

## Executive Summary

Single neurons have fundamental limitations - they can only learn simple patterns and ReLU activation blocks negative values. By combining multiple neurons into layers, we create networks that can learn complex functions that individual neurons cannot handle.

## The Problem We Discovered

In Section 1, our single neuron failed to learn addition because:
- **ReLU blocks negative values** → outputs 0 for many inputs
- **Single linear boundary** → can't handle complex patterns  
- **Limited expressiveness** → one neuron = one simple decision

## The Solution: Multiple Neurons

**Key insight**: Different neurons can specialize in different parts of the problem:
- Neuron 1: Handle positive input combinations
- Neuron 2: Handle negative input combinations  
- Neuron 3: Learn different weight patterns
- **Combine their outputs** to solve the full problem

## Layer Architecture

```python
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    
    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
```

**What this gives us:**
- **Parallel processing**: All neurons see the same inputs
- **Diverse responses**: Each neuron learns different patterns
- **Increased capacity**: More parameters = more learning power

## Why Multiple Neurons Work

### Specialization
Each neuron can focus on different aspects:
- Some neurons activate for large positive sums
- Others activate for small positive sums
- Different weight combinations capture different patterns

### Redundancy  
If one neuron gets "stuck" (like our ReLU problem), others can still learn and contribute.

### Combination Power
Multiple weak learners can combine to create a strong learner - this is the foundation of deep learning.

## From Neurons to Networks

**This section bridges the gap:**
- Section 1: Single neuron limitations
- Section 2: Multiple neurons cooperation  
- Future: Deep networks with many layers

## Key Learning Goals

1. **Understand layer architecture** - how neurons work in parallel
2. **See improved learning** - multiple neurons solving what one cannot
3. **Grasp the scaling principle** - more neurons = more capability
4. **Foundation for deep learning** - layers are the building blocks

## Next Steps

After mastering multiple neurons, we'll explore how to stack layers to create deep networks that can learn hierarchical representations and solve even more complex problems.
