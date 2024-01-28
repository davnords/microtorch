from tensor import Tensor
import numpy as np

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros(p.grad.shape)

    def parameters(self):
        return []
    
class ReLU(Module):

    def __init__(self):
        super().__init__()

    def __call__(self, x):
        assert isinstance(x, Tensor)
        return x.relu()

    def __repr__(self):
        return f"ReLU"
    
    def parameters(self):
        return []
    
class Linear(Module):

    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.w = Tensor(np.random.randn(nin, nout))
        self.b = Tensor(np.random.randn(1, nout))

    def __call__(self, x):
        assert x.data.shape[-1] == self.nin
        z = x@self.w+self.b
        return z

    def parameters(self):
        return [self.w] + [self.b] 

    def __repr__(self):
        return f"Linear layer"

class NeuralNetwork(Module):

    def __init__(self, nin, nout, hidden_nodes=10):
        self.layers = [Linear(nin, hidden_nodes), ReLU()] + [Linear(hidden_nodes,nout), ReLU()]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"