from tensor import Tensor
import torch
import numpy as np

def main():
    x = np.random.randn(1000, 5)
    w = np.random.randn(5, 3)
    b = np.random.randn(3)
    y = np.random.randn(1000, 3)

    # FORWARD TEST

    X = torch.tensor(x, dtype=torch.float32, requires_grad=False)
    W = torch.tensor(w, dtype=torch.float32, requires_grad=False)
    B = torch.tensor(b, dtype=torch.float32, requires_grad=False)
    forward_torch = X @ W + B

    X = Tensor(x, requires_grad=False)
    W = Tensor(w, requires_grad=False)
    B = Tensor(b, requires_grad=False)
    forward_microtorch = X @ W + B

    forward_diff = (forward_torch - forward_microtorch).sum().data.item()

    # BACKWARD TEST
    X = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    W = torch.tensor(w, dtype=torch.float32, requires_grad=True)
    B = torch.tensor(b, dtype=torch.float32, requires_grad=True)
    Y = torch.tensor(y, dtype=torch.float32, requires_grad=False)
    ypred = X @ W # + B
    ypred = ypred.mean()
    ypred.backward()
    # loss = (ypred-Y)**2
    # loss = loss.mean()
    # loss.backward()
    backward_torch = W.grad

    X = Tensor(x, requires_grad=True)
    W = Tensor(w, requires_grad=True)
    B = Tensor(b, requires_grad=True)
    Y = Tensor(y, requires_grad=False)
    ypred = X @ W # + B
    ypred = ypred.mean()
    ypred.backward()
    # loss = (ypred-Y)**2
    # loss = loss.mean()
    # loss.backward()
    backward_microtorch = W.grad

    backward_diff = (backward_torch - backward_microtorch).sum().data.item()

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