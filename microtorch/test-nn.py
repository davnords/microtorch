from tensor import Tensor
from nn import NeuralNetwork
import pandas as pd
import numpy as np
import torch

def main():
    # Parameters 
    n_input = 1
    n_hidden = 3
    n_out = 1
    n_samples = 10
    epochs = 100
    learning_rate = 0.01
    # Synthetic data
    x = np.random.randn(n_samples, n_input)
    y = 2 * x + 1 + 0.1 * np.random.randn(n_samples, n_out)  # True relationship: y = 2x + 1 + noise

    # Weight and biases
    w1 = np.random.randn(n_input, n_hidden)
    b1 = np.random.randn(1, n_hidden)
    w2 = np.random.randn(n_hidden, n_out)
    b2 = np.random.randn(1, n_out)

    # PYTORCH TRAINING
    X = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    W1 = torch.tensor(w1, dtype=torch.float32, requires_grad=True)
    B1 = torch.tensor(b1, dtype=torch.float32, requires_grad=True)
    W2 = torch.tensor(w2, dtype=torch.float32, requires_grad=True)
    B2 = torch.tensor(b2, dtype=torch.float32, requires_grad=True)
    Y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
    
    #Forward test
    Z1 = X @ W1 + B1
    A1 = Z1.relu()
    Z2 = A1 @ W2 + B2
    A2 = Z2.relu()
    loss = ((A2-Y)**2).mean()
    forward_torch = loss
    loss.backward()
    #Backward test
    backward_torch = W1.grads

    # MICROTORCH TRAINING
    X = Tensor(x, requires_grad=True)
    W1 = Tensor(w1, requires_grad=True)
    B1 = Tensor(b1, requires_grad=True)
    W2 = Tensor(w2, requires_grad=True)
    B2 = Tensor(b2, requires_grad=True)
    Y = Tensor(y, requires_grad=True)
    
    Z1 = X @ W1 + B1
    A1 = Z1.relu()
    Z2 = A1 @ W2 + B2
    A2 = Z2.relu()
    loss = ((A2-Y)**2).mean()
    forward_microtorch = loss
    loss.backward()
    backward_microtorch = W1.grad

    forward_diff = (forward_torch.data.item() - forward_microtorch.data.item())
    backward_diff = (backward_torch - backward_microtorch).sum().data.item()
    print(backward_diff)

    tol = 1e-3
    if abs(forward_diff) < tol:
        print("FORWARD: OK")
    else:
        print("FORWARD: ERROR")
    if abs(backward_diff) < tol:
        print("BACKWARD: OK")
    else:
        print("BACKWARD: ERROR")

if __name__ == "__main__":
    main()