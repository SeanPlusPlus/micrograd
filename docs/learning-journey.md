# Learning Journey: From LLM User to Understanding Internals

## Executive Summary

This document tracks exploration of how neural networks and LLMs actually work under the hood. Starting from micrograd's automatic differentiation engine, we build understanding step by step through hands-on experiments.

## Learning Path

### Phase 1: Automatic Differentiation Fundamentals
- [ ] Understand what a `Value` object represents
- [ ] Trace through a simple forward pass
- [ ] See how gradients flow backward
- [ ] Visualize the computational graph

### Phase 2: Neural Network Basics
- [ ] Build a single neuron from scratch
- [ ] Understand weights, biases, and activation functions
- [ ] Train a simple network on toy data
- [ ] See how loss functions guide learning

### Phase 3: Scaling Up Concepts
- [ ] Multiple layers and hidden representations
- [ ] Different activation functions and their purposes
- [ ] Optimization algorithms (SGD, Adam, etc.)
- [ ] Regularization and overfitting

### Phase 4: Path to Modern LLMs
- [ ] From MLPs to attention mechanisms
- [ ] Sequence modeling and embeddings
- [ ] Transformer architecture basics
- [ ] Training at scale concepts

## Experiments & Notes

### Experiment 1: First Value Operations
See [`experiments/value_experiment.py`](../experiments/value_experiment.py) - demonstrates how Value objects build computational graphs and compute gradients.

**Key insight**: Each Value remembers its parents through `._prev`, creating a graph that `backward()` can walk through to compute gradients automatically.

### Questions to Explore
- Why do we need gradients at all?
- How does backpropagation actually work?
- What makes transformers so powerful for language?

## Resources
- [Micrograd repo](https://github.com/karpathy/micrograd) - our starting point
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [3Blue1Brown Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)

---

*This is a living document - updated as we learn and experiment.*
