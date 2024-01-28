# microtorch

Welcome. This project aims to create a high performance deep learning framework with CUDA support and a similar API as that of PyTorch. This is done for educational purposes and detailed instructions for each step follows. Currently the project has implemented a fully fletched autograd engine that runs through numpy. The next step is to write our own tensor class in C/C++ with CUDA support.

### Installation

```bash
pip install microtorch
```

### Example usage

Below is a slightly contrived example showing a number of possible supported operations:

```python
from microtorch import Tensor

X = Tensor(x, requires_grad=True)
W = Tensor(w, requires_grad=True)
B = Tensor(b, requires_grad=True)
Y = Tensor(y, requires_grad=False)
ypred = X @ W + B # single layer forward pass
ypred.softmax(dim=1) # softmax as activation (ReLU is also supported)
loss = (ypred-Y)**2
loss = loss.mean()
loss.backward()
print(f'{W.grad:.4f}') # prints the full jacobian (gradiant) of the weight matrix
```

### Training a neural net

The engine supports training neural networks from scratch - as can be noted from the `test-engine.py` file, however, a more advanced example of full neural network training with benchmarking against PyTorch is in the making!

### Running tests

To run the unit tests you will have to install [PyTorch](https://pytorch.org/), which the tests use as a reference for verifying the correctness of the calculated gradients. Then simply:

```bash
python3 test-engine.py test-nn.py
```

When the autograd engine works (according to PyTorch) the following will be printed:
```bash
Forward: OK
Backward: OK
```

Verifying that both the forward and backward passes agree with the PyTorch API.

### C commands

```bash
python3 setup.py install
```

### License

MIT