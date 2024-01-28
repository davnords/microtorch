from tensor import Tensor
from nn import MLP
import pandas as pd
import numpy as np
import torch

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

def simple_Test():

   
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data_frame = pd.read_csv('./microtorch/housing.csv', header=None, delimiter=r"\s+", names=column_names)

    column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
    x = data_frame.loc[:,column_sels].to_numpy()
    y = data_frame['MEDV'].to_numpy().reshape(-1, 1)

    batch_size = 32
    n_input = 8
    n_out = 1
    n_samples = 506
    epochs = 100
    mlp = MLP(n_input, n_out)

    for _ in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        indices = np.arange(len(x))
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]

        # Iterate over batches
        for i in range(0, len(x), batch_size):
            batch_x = Tensor(x_shuffled[i:i + batch_size])
            batch_y = Tensor(y_shuffled[i:i + batch_size])
            y_pred = mlp(batch_x)
            loss = (y_pred-batch_y)**2
            loss = loss.mean()

            print('Loss', loss.data.mean())

            mlp.zero_grad()
            loss.backward()

            for p in mlp.parameters():
                p.data += -0.01*p.grad / n_samples

# test_MLP()
simple_Test()



