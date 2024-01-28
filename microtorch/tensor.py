import numpy as np

class Tensor:
    """ stores a tensor and its gradient """

    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = np.zeros(self.data.shape)
        
        self.requires_grad = requires_grad

        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                # Broadcast out.grad to the shape of other.data before adding
                other.grad += np.sum(out.grad, axis=0, keepdims=True)

        out._backward = _backward

        return out

    def __mul__(self, other):
        if isinstance(other, int):
            other = float(other)
        # assert isinstance(other, float)
        if isinstance(other, Tensor):
            out_data = self.data * other.data
        else: 
            out_data = self.data * other
        out = Tensor(out_data, (self, other), '*')

        def _backward():
            self.grad += other * out.grad
            if isinstance(other, Tensor):
                other.grad += np.sum(self.data * out.grad)
        out._backward = _backward

        return out
    
    def __matmul__(self, other):

        out = Tensor(self.data.dot(other.data), (self, other), '@')

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.dot(other.data.T)
            if other.requires_grad:
                other.grad += self.data.T.dot(out.grad)
        out._backward = _backward
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * np.power(self.data, other - 1) * out.grad
        out._backward = _backward

        return out

    def mean(self):
        out_data = np.mean(self.data, keepdims=True)
        out = Tensor(out_data, (self,), 'Mean')

        def _backward():
            self.grad += out.grad*(1 / self.data.size)
        out._backward = _backward

        return out
    
    def sum(self):
        out_data = np.sum(self.data, keepdims=True)
        out = Tensor(out_data, (self,), 'Mean')

        def _backward():
            self.grad += out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        out_data = np.maximum(0, self.data)
        out = Tensor(out_data, (self,), 'ReLU')

        def _backward():
            self.grad += (out_data > 0) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self, dim=1):
        exp_data = np.exp(self.data - np.max(self.data, axis=dim, keepdims=True))
        out_data = exp_data / np.sum(exp_data, axis=dim, keepdims=True)
        out = Tensor(out_data, (self,), 'Softmax')

        def _backward():
            # Compute the derivative of softmax with respect to the input
            softmax_grad = out_data * (1 - out_data)
            # Update the gradients using the chain rule
            self.grad += np.dot(out.grad, softmax_grad)
        out._backward = _backward

        return out
    
    def shape(self):
        return self.data.shape

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
        self.grad = np.ones(self.data.shape)
        for v in reversed(topo):
            v._backward()

    def zero_grad_(self):
        self.grad = np.zeros(self.data.shape)

    def __neg__(self): # -self
        self.data *= -1
        return self

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
        return f'tensor({self.data}, grad_fn=<{self._op}>)'

    