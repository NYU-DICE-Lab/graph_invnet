import os
import sys

import torch

path=os.path.realpath(os.path.dirname(__file__))
path_split=path.split('/')
path_limited=path_split[:-2]
path_together=''.join('/'+x for x in path_limited)
sys.path.append(path_together)

from layers.dp_layer.DPLayer import DPLayer


def make_data():
    a_range=torch.arange(1,3,dtype=torch.float,requires_grad=True).view((1,-1))
    a_range=torch.cat([a_range,a_range+2],dim=0).unsqueeze(0)
    a_range=a_range ** 2
    return a_range

def test_dp_forward():
    image=make_data()
    layer=DPLayer('diff_squared',True,2,2,make_pos=False)
    v_hard=layer(image)
    true_output=83
    err=abs(true_output-v_hard.item())
    assert err < 1e-6

# def test_dp_forward_no_grad():
#     image=make_data()
#     layer = DPLayer('diff_squared', False, 2, 2,make_pos=False)
#     with torch.no_grad():
#         v_hard = layer(image)
#     true_output = 83
#     err = abs(true_output - v_hard.item())
#     assert err < 1e-6