#!/usr/bin/env python3
"""
Activation Functions Experiment

Compare different activation functions and see their impact on:
1. Function shape and derivatives
2. Network expressiveness 
3. Training behavior
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from micrograd.engine import Value
import math

def relu(x):
    """ReLU: max(0, x)"""
    return x if x.data > 0 else Value(0.0)

def sigmoid(x):
    """Sigmoid: 1 / (1 + e^(-x))"""
    return Value(1.0) / (Value(1.0) + (-x).exp())

def tanh(x):
    """Tanh: (e^x - e^(-x)) / (e^x + e^(-x))"""
    ex = x.exp()
    e_neg_x = (-x).exp()
    return (ex - e_neg_x) / (ex + e_neg_x)

def test_activation_shapes():
    """Test how different activations transform inputs"""
    print("=== Activation Function Shapes ===")
    
    test_inputs = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    
    print(f"{'Input':<8} {'ReLU':<8} {'Sigmoid':<8} {'Tanh':<8}")
    print("-" * 35)
    
    for val in test_inputs:
        x = Value(val)
        
        r = relu(x)
        s = sigmoid(x)
        t = tanh(x)
        
        print(f"{val:<8.1f} {r.data:<8.3f} {s.data:<8.3f} {t.data:<8.3f}")

def test_gradients():
    """Test how gradients flow through different activations"""
    print("\n=== Gradient Flow ===")
    
    test_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0]
    
    print(f"{'Input':<8} {'ReLU_grad':<10} {'Sigmoid_grad':<12} {'Tanh_grad':<10}")
    print("-" * 45)
    
    for val in test_inputs:
        # Test ReLU gradient
        x1 = Value(val)
        y1 = relu(x1)
        y1.backward()
        relu_grad = x1.grad
        
        # Test Sigmoid gradient  
        x2 = Value(val)
        y2 = sigmoid(x2)
        y2.backward()
        sigmoid_grad = x2.grad
        
        # Test Tanh gradient
        x3 = Value(val)
        y3 = tanh(x3)
        y3.backward()
        tanh_grad = x3.grad
        
        print(f"{val:<8.1f} {relu_grad:<10.3f} {sigmoid_grad:<12.3f} {tanh_grad:<10.3f}")

def simple_neuron_comparison():
    """Compare how different activations affect a simple neuron"""
    print("\n=== Neuron with Different Activations ===")
    
    # Same inputs and weights for fair comparison
    x1, x2 = Value(0.5), Value(-0.3)
    w1, w2 = Value(0.8), Value(-0.4)
    b = Value(0.1)
    
    # Linear combination (same for all)
    linear = w1*x1 + w2*x2 + b
    print(f"Linear output: {linear.data:.3f}")
    
    # Apply different activations
    relu_out = relu(linear)
    sigmoid_out = sigmoid(linear)
    tanh_out = tanh(linear)
    
    print(f"ReLU output:    {relu_out.data:.3f}")
    print(f"Sigmoid output: {sigmoid_out.data:.3f}")
    print(f"Tanh output:    {tanh_out.data:.3f}")

def demonstrate_dead_relu():
    """Show the dead ReLU problem"""
    print("\n=== Dead ReLU Demonstration ===")
    
    # Neuron with large negative bias
    x = Value(1.0)
    w = Value(0.5)
    b = Value(-2.0)  # Large negative bias
    
    linear = w*x + b
    output = relu(linear)
    
    print(f"Input: {x.data}")
    print(f"Linear: {linear.data:.3f}")
    print(f"ReLU output: {output.data:.3f}")
    
    # Compute gradient
    output.backward()
    print(f"Weight gradient: {w.grad:.3f}")
    print("Note: Zero gradient means this neuron won't learn!")

if __name__ == "__main__":
    test_activation_shapes()
    test_gradients()
    simple_neuron_comparison()
    demonstrate_dead_relu()
    
    print("\n=== Key Observations ===")
    print("1. ReLU is simple but can 'die' (zero gradient)")
    print("2. Sigmoid squashes everything to (0,1)")
    print("3. Tanh is zero-centered, ranging (-1,1)")
    print("4. Gradients vary dramatically between functions")
    print("5. Choice affects both forward pass and learning")
