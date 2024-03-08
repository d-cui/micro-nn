"""
This class defines a Value, the basic building block of a mathematical expression, supporting forward and back propagation throughout the expression.
"""

from enum import Enum
import math


class Op(Enum):
    ADD = "+"
    MUL = "*"
    EXP = "exp"
    POW = "**"
    TANH = "tanh"
    NONE = ""


class Value:
    def __init__(self, data: float, _children=(), _op=Op.NONE.value, _name=""):
        """
        data: the scalar value stored.

        _children: represent the expression graph. If a and b are the children of c, then a and b combine through some mathematical expression to form c.

        _op: the operation that combines the children into the current scalar value

        _name: A readable name for the Value
        """
        self.data = data
        self._prev = set(_children)  # Set is for efficiency
        self._op = _op
        self._name = _name

        # In backward, need to use += instead of = to accumulate gradients when a value is used more than once.
        # See the multivariate chain rule
        self._backward = lambda: None
        self.grad = (
            0.0  # The gradient of the expression wrt this variable, 0 means no effect
        )

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Adds two Values.
        """
        if not isinstance(other, Value):
            # Wrap other in a Value if it's not already an instance of value, allowing for operations
            # such as Value(1.0) + 1.0
            other = Value(other)

        out = Value(self.data + other.data, _children=(self, other), _op=Op.ADD.value)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        # This allows us to do something like 2.0 + Value(1.0)
        return self + other

    def __mul__(self, other):
        """
        Multiplies two Values.
        """
        if not isinstance(other, Value):
            # Wrap other in a Value if it's not already an instance of value, allowing for operations
            # such as Value(1.0) * 2.0
            other = Value(other)

        out = Value(self.data * other.data, _children=(self, other), _op=Op.MUL.value)

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward

        return out

    def __rmul__(self, other):
        # This allows us to do something like 2.0 * Value(1.0)
        return self * other

    def __truediv__(self, other):  # self / other
        # Division can be written as a * b^-1
        return self * (other**-1)

    def __rtruediv__(self, other):  # other / self
        # Division can be written as a * b^-1
        return other * (self**-1)

    def __neg__(self):  # -self
        return self * -1

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __pow__(self, other):  # self**other
        return self.pow(other)

    def exp(self):
        out = Value(math.exp(self.data), _children=(self,), _op=Op.EXP.value)

        def _backward():
            # Derivative of e^x is just e^x
            self.grad += out.grad * out.data

        out._backward = _backward

        return out

    def pow(self, other):  # a ^ other, where other is a scalar
        assert isinstance(other, (int, float)), "Only int/float powers are supported"
        out = Value(self.data**other, _children=(self,), _op=f"{Op.POW.value}{other}")

        def _backward():
            self.grad += out.grad * other * (self.data ** (other - 1))

        out._backward = _backward

        return out

    def tanh(self):
        """
        Applies the tanh function to a value. Operations can be complex, as long as they are locally differentiable.
        """
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _op=Op.TANH.value)

        def _backward():
            self.grad += out.grad * (1 - out.data**2)

        out._backward = _backward

        return out

    def backprop(self):
        """
        Backpropagate through the entire expression. Backpropagation needs to happen starting from the root node, then layer by layer. We achieve this by doing a breadth-first traversal of the tree.
        """
        self.grad = 1  # Base case, since dself/dself = 1

        nodes = set([self])

        while len(nodes) > 0:
            node = nodes.pop()
            node._backward()

            for child in node._prev:
                nodes.add(child)
