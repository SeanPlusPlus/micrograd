# Activation Functions Quiz

Test your understanding of activation functions and their role in neural networks.

## Question 1: The Linear Problem
Why can't a network of linear neurons (no activation functions) learn complex patterns, no matter how deep?

**Your answer:**

<details>
<summary>Answer</summary>

Any composition of linear functions is still linear. A network of linear neurons can only learn linear relationships, which severely limits what problems it can solve. Even a 1000-layer network without activations is mathematically equivalent to a single linear transformation.
</details>

## Question 2: ReLU Characteristics
What are the main advantages and disadvantages of ReLU activation?

**Your answer:**

<details>
<summary>Answer</summary>

**Advantages:**
- Simple and fast to compute
- No saturation for positive values (gradients don't vanish)
- Sparse activation (many neurons output 0)
- Works well in practice

**Disadvantages:**
- Dead neuron problem (neurons can get stuck outputting 0)
- Not differentiable at x=0
- Can be sensitive to learning rate
</details>

## Question 3: Gradient Flow
Given these activation functions, which would likely have the worst vanishing gradient problem in deep networks?
- ReLU
- Sigmoid  
- Tanh

**Your answer:**

<details>
<summary>Answer</summary>

Sigmoid would have the worst vanishing gradient problem. For large positive or negative inputs, sigmoid's gradient approaches zero. When you multiply many small gradients together in deep networks, they vanish exponentially. ReLU doesn't have this problem for positive inputs, and tanh is better than sigmoid but still suffers from vanishing gradients.
</details>

## Question 4: Dead Neurons
A ReLU neuron has weights [0.1, 0.2] and bias -5.0. For typical inputs in range [0, 1], will this neuron learn effectively? Why?

**Your answer:**

<details>
<summary>Answer</summary>

No, this neuron likely won't learn effectively. With a large negative bias (-5.0) and small positive weights, the linear combination will almost always be negative. ReLU will output 0, giving zero gradient, so the weights won't update. This is the "dead neuron" problem.
</details>

## Question 5: Activation Choice
For each scenario, which activation function would you choose and why?

a) Hidden layer in a deep network for image classification
b) Output layer for binary classification (yes/no prediction)  
c) Output layer for regression (predicting house prices)

**Your answers:**

<details>
<summary>Answers</summary>

a) **ReLU** - Standard choice for hidden layers. Fast, simple, and works well in practice despite potential dead neuron issues.

b) **Sigmoid** - Output needs to be between 0 and 1 to represent probability. Sigmoid naturally provides this range.

c) **Linear (no activation)** - Regression needs to output any real number. Adding an activation would constrain the output range unnecessarily.
</details>

## Bonus Question: Universal Approximation
The universal approximation theorem states that neural networks can approximate any continuous function. What role do activation functions play in this capability?

**Your answer:**

<details>
<summary>Answer</summary>

Activation functions provide the non-linearity that makes universal approximation possible. Without them, networks can only represent linear functions regardless of depth. Non-linear activations allow networks to create complex decision boundaries and approximate arbitrarily complex continuous functions given sufficient width and depth.
</details>
