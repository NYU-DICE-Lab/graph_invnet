import os
import sys
path=os.path.realpath(os.path.dirname(__file__))
path_split=path.split('/')
path_limited=path_split[:-2]
path_together=''.join('/'+x for x in path_limited)
sys.path.append(path_together)
from layers.dp_layer.DPLayer import DPLayer

import numpy as np
import pytest
import torch
from scipy.optimize import approx_fprime
from torchvision import transforms, datasets

from microstructure import utils


_epsilon = np.sqrt(np.finfo(float).eps)
PARAMS=[('diff_squared',0),('sum_squared',0),('diff_squared',1),('sum_squared',1)]
PARAMS=[('diff_squared',1)]
def make_data():
    dir = '/data/datasets/two_phase_morph/morph_global_64_train_255.h5'
    dataset = utils.MicrostructureDataset(dir)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    data=next(iter(train_loader))
    return data.squeeze(0)

def make_mnist_data():
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3801])
    ])
    data_dir = '/data/MNIST'
    print('data_dir:', data_dir)
    mnist_data = datasets.MNIST(data_dir, download=False,
                                transform=data_transform)

    train_loader = torch.utils.data.DataLoader(mnist_data, batch_size=1, shuffle=True)
    images, _ = next(iter(train_loader))
    return images.squeeze(0)

def custom_check_grad(func, grad, x0, *args, **kwargs):
    """Check the correctness of a gradient function by comparing it against a
    (forward) finite-difference approximation of the gradient.

    Parameters
    ----------
    func : callable ``func(x0, *args)``
        Function whose derivative is to be checked.
    grad : callable ``grad(x0, *args)``
        Gradient of `func`.
    x0 : ndarray
        Points to check `grad` against forward difference approximation of grad
        using `func`.
    args : \\*args, optional
        Extra arguments passed to `func` and `grad`.
    epsilon : float, optional
        Step size used for the finite difference approximation. It defaults to
        ``sqrt(np.finfo(float).eps)``, which is approximately 1.49e-08.

    Returns
    -------
    err : float
        The square root of the sum of squares (i.e., the 2-norm) of the
        difference between ``grad(x0, *args)`` and the finite difference
        approximation of `grad` using func at the points `x0`.

    See Also
    --------
    approx_fprime

    Examples
    --------
    2.9802322387695312e-08

    """
    step = kwargs.pop('epsilon', _epsilon)
    if kwargs:
        raise ValueError("Unknown keyword arguments: %r" %
                         (list(kwargs.keys()),))

    true_grad= approx_fprime(x0, func, step, *args)
    approx_grad=grad(x0, *args)
    return np.sqrt(sum((true_grad-
                     approx_grad)**2))

@pytest.mark.parametrize("edge_fn,max_op",PARAMS)
def test_sp_backward(edge_fn,max_op):
    device = "cuda:0"
    torch.cuda.set_device(device)

    data = make_mnist_data()
    dp_layer=DPLayer(edge_fn,max_op,32,32,True)
    def grad(X):
        X = torch.tensor(X).detach()

        X = X.reshape(data.shape).type(torch.float64).to(device)
        X.requires_grad = True
        X.retain_grad()
        p1 = dp_layer(X)
        output=p1.mean()
        output.backward()
        grad=X.grad.cpu()
        return grad.view(-1).detach().numpy()

    def func(X):
        with torch.no_grad():
            Y = torch.tensor(X,dtype=torch.float64,device=device).detach()
            Y = Y.reshape(data.shape)
            output = dp_layer(Y)
            output=output.sum().cpu()
            np.array(output,dtype=np.float64)
        return output.numpy()


    err = custom_check_grad(func, grad, (data.detach().numpy()).ravel())
    assert err < 1e-6

