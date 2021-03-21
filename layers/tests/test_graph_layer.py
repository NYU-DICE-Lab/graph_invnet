import os
import sys

import numpy as np
import torch
from scipy.optimize import check_grad

path=os.path.realpath(os.path.dirname(__file__))
path_split=path.split('/')
path_limited=path_split[:-2]
path_together=''.join('/'+x for x in path_limited)
sys.path.append(path_together)


from layers.dp_layer.edge_functions import sum_squared
from layers.graph_layer import GraphLayer


def make_data():
    ranged = torch.arange(2, dtype=torch.float64).view(1, 1, -1)
    image = ranged.repeat((2, 2, 1))
    image.requires_grad = True
    solution = torch.tensor([[1, 1, 0, -1 * float('inf')],
                             [-1 * float('inf'), -1 * float('inf'), 4, 1],
                             [1, -1 * float('inf'), -1 * float('inf'), -1 * float('inf')],
                             [-1 * float('inf'), -1 * float('inf'), -1 * float('inf'), -1 * float('inf')]],
                            dtype=torch.float64)
    solution = solution.view((1, 4, 4)).repeat((2, 1, 1))
    return image, solution

def test_graph_forward():

    image,solution=make_data()
    graph_layer = GraphLayer(-float('inf'),sum_squared)
    theta = graph_layer(image)
    theta.sum().backward()

    for batch in range(2):
        for i in range(4):
            for j in range(4):
                assert theta[batch,i,j]==solution[batch,i,j]

def test_graph_backward():
    images ,_ = make_data()

    def grad(X):
        graph_layer = GraphLayer(-float('inf'),sum_squared)
        X = torch.tensor(X).detach()
        X.requires_grad = True
        X = X.reshape(images.shape)
        output=graph_layer(X)[0,0,0]
        X.retain_grad()
        output.backward()
        grad=X.grad.view(-1)
        return grad.detach().numpy()

    def func(X):
        graph_layer = GraphLayer(-float('inf'),sum_squared)
        Y = torch.tensor(X,dtype=torch.float64).detach()
        Y = Y.reshape(images.shape)
        output = graph_layer(Y)[0,0,0]
        output=np.array(output,dtype=np.float64)
        return output

    err = check_grad(func, grad, (images.detach().numpy()).ravel())
    assert err < 1e-6
