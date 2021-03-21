import os
import sys

import pytest
import torch
from torchvision import transforms, datasets

path=os.path.realpath(os.path.dirname(__file__))
path_split=path.split('/')
path_limited=path_split[:-2]
path_together=''.join('/'+x for x in path_limited)
sys.path.append(path_together)
from layers.dp_layer.DPLayer import DPLayer


def make_data():
    data_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3801])
    ])
    data_dir = '/data/MNIST'
    mnist_data = datasets.MNIST(data_dir, download=True,
                                transform=data_transform)
    train_data, val_data = torch.utils.data.random_split(mnist_data, [55000, 5000])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    image, _ = next(iter(train_loader))
    return image.squeeze(0)

@pytest.mark.parametrize('sign',[1,-1])
def test_grad_step(sign):
    device=torch.device('cuda')
    p1_layer = DPLayer('diff_exp', True, 32, 32, make_pos=True)

    image=make_data().to(device)
    image.requires_grad=True
    image.retain_grad()

    real_p1=p1_layer(image)
    loss=sign*real_p1
    loss.backward()
    grad=image.grad
    assert torch.linalg.norm(grad)>0

    coeff=01e-08
    while True:
        new_image=image-coeff*grad
        new_loss=sign*p1_layer(new_image)
        if new_loss.item()==loss.item():
            coeff*=10
        else:
            assert new_loss<loss
            break


