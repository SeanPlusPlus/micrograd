# Learning Plan: Neural Networks from First Principles

## Executive Summary

This learning journey explores how neural networks and LLMs work under the hood, starting from micrograd's automatic differentiation engine. Each concept is broken down into digestible sections with theory, hands-on experiments, and quizzes.

## Recommended Usage

**Best experienced with a coding agent**: This learning path is designed to be explored interactively with an AI coding assistant. The agent can:
- Explain concepts and answer questions as you learn
- Help you work through the experiments step by step  
- Quiz you on concepts (answers are included in quiz files, so the agent can check your understanding)
- Adapt explanations based on your background and learning style

## Structure

Each learning phase is organized as:
```
Phase_X/
├── Section_Y_Topic/
│   ├── overview.md    # Core concepts and theory
│   ├── experiment.py  # Hands-on code to run
│   └── quiz.md       # Test understanding
```

This approach ensures:
- **Theory**: Clear explanations of concepts
- **Practice**: Executable code to see concepts in action  
- **Assessment**: Quizzes to verify understanding

## Learning Path

### Phase 1: Automatic Differentiation Fundamentals
- [x] **Section 1: Value Objects** - Understanding smart numbers that track gradients
- [x] **Section 2: Forward Pass** - Tracing computation and graph building
- [x] **Section 3: Backward Pass** - How gradients flow through the graph
- [x] **Section 4: Visualization** - Seeing the computational graph

### Phase 2: Neural Network Basics
- [ ] **Section 1: Single Neuron** - Building neurons from Value objects
- [ ] **Section 2: Weights & Biases** - Understanding parameters
- [ ] **Section 3: Activation Functions** - ReLU, sigmoid, and their purposes
- [ ] **Section 4: Loss Functions** - How networks measure performance

### Phase 3: Training Networks
- [ ] **Section 1: Gradient Descent** - Following gradients to minimize loss
- [ ] **Section 2: Backpropagation** - Training multi-layer networks
- [ ] **Section 3: Optimization** - SGD, momentum, learning rates
- [ ] **Section 4: Real Dataset** - Training on actual data

### Phase 4: Path to Modern LLMs
- [ ] **Section 1: Multi-Layer Perceptrons** - Scaling up networks
- [ ] **Section 2: Sequence Modeling** - Handling text and sequences
- [ ] **Section 3: Attention Mechanisms** - The transformer revolution
- [ ] **Section 4: Modern Architectures** - From GPT to current models

## Resources
- [Micrograd repo](https://github.com/karpathy/micrograd) - our foundation
- [Andrej Karpathy's Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [3Blue1Brown Neural Networks series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
