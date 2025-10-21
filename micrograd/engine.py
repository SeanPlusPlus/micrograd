
class Value:
    """ 
    A smart number that tracks its value AND how to compute gradients.
    This is the core of automatic differentiation - every math operation
    creates a new Value that remembers how it was made.
    """

    def __init__(self, data, _children=(), _op=''):
        # The actual numeric value (e.g., 2.5, -1.0, etc.)
        self.data = data
        
        # The gradient - how much the final output changes if we nudge this value
        # Starts at 0, gets filled in during backward pass
        self.grad = 0
        
        # Function that knows how to compute gradients for this specific operation
        # Gets overwritten by each math operation (__add__, __mul__, etc.)
        self._backward = lambda: None
        
        # Set of Value objects that were used to create this one
        # Forms the computational graph - these are our "parents"
        self._prev = set(_children)
        
        # String describing what operation created this Value (for debugging/visualization)
        self._op = _op

    def __add__(self, other):
        # Convert regular numbers to Value objects so we can track gradients
        other = other if isinstance(other, Value) else Value(other)
        
        # Create new Value with the sum, remembering who the parents are
        out = Value(self.data + other.data, (self, other), '+')

        # Define how gradients flow backward through addition
        # Key insight: gradient of addition just passes through unchanged to both inputs
        # If output needs to increase by X, both inputs should increase by X
        def _backward():
            self.grad += out.grad    # Add (don't overwrite!) the gradient
            other.grad += out.grad   # Both inputs get the same gradient
        out._backward = _backward

        return out

    def __mul__(self, other):
        # Convert regular numbers to Value objects
        other = other if isinstance(other, Value) else Value(other)
        
        # Create new Value with the product
        out = Value(self.data * other.data, (self, other), '*')

        # Define how gradients flow backward through multiplication
        # Key insight: gradient of multiplication uses the "other" input's value
        # If f = a * b, then df/da = b and df/db = a (basic calculus!)
        def _backward():
            self.grad += other.data * out.grad   # My gradient = other's value * output gradient
            other.grad += self.data * out.grad   # Other's gradient = my value * output gradient
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
