from tensor import Tensor
from nn import MLP
import numpy as np

def test_MLP():
    n_input = 10
    n_out = 10
    n_samples = 10000
    epochs = 100
    X = Tensor(np.random.randn(n_samples, n_input))
    y = Tensor(np.random.randn(n_samples, n_out))
    mlp = MLP(n_input, n_out)

    for _ in range(epochs):
        y_pred = mlp(X)
        loss = (y_pred-y)**2
        loss = loss.mean()

        print('Loss', loss.data.mean())

        mlp.zero_grad()
        loss.backward()

        for p in mlp.parameters():
            p.data += -0.01*p.grad

test_MLP()