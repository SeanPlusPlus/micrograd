# Agent Collaboration Directives

## Project Context

This is a fork of [Karpathy's micrograd](https://github.com/karpathy/micrograd) with the goal of building out a comprehensive `learning/` directory. The target audience is software developers who have coding experience but want to understand the deep internals of neural networks from first principles.

## Core Rules

### 1. Terminal Commands
- **Sean runs all terminal commands** - no `execute_bash` from agent
- Agent can write Python code and suggest commands
- Use `pbcopy` to put suggested commands in clipboard for Sean to run

### 2. Learning-First Approach
- Focus on building intuition through hands-on experiments
- Create minimal, focused examples that demonstrate one concept clearly
- Always explain the "why" behind each step
- Structure content for progressive learning (theory → practice → assessment)

### 3. Documentation Style
- Use clear, structured markdown with executive summaries
- Save key insights and "aha moments" in permanent docs
- Build a learning journey that others can follow
- Make content shareable with other developers

## Learning Structure

Each concept follows the pattern:
```
Section_X_Topic/
├── overview.md    # Theory and concepts
├── experiment.py  # Hands-on code
└── quiz.md       # Understanding verification
```

## Current Focus
- Phase 1: Understanding automatic differentiation fundamentals
- Building from Value objects → neural networks → modern LLMs
- Creating a resource that bridges the gap between "using AI" and "understanding AI internals"
