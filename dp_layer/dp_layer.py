import torch.nn as nn

from dp_layer.dp_function import DPFunction
from dp_layer.graph_layer import GraphLayer
from dp_layer.graph_layer.adjacency_utils import idx_adjacency
from dp_layer.graph_layer.edge_functions import edge_f_dict


class DPLayer(nn.Module):

    def __init__(self,edge_fn,max_op,max_i,max_j,make_pos=True,top2bottom=False):
        super(DPLayer, self).__init__()
        self.edge_f=edge_f_dict[edge_fn]
        self.max_op=max_op
        self.null = float('inf')
        if self.max_op:
            self.null *= -1
        self.graph_layer = GraphLayer(self.null,self.edge_f,make_pos,top2bottom)
        self.adj_array,self.rev_adj=idx_adjacency(max_i,max_j)

    def forward(self,images):
        dp_function = DPFunction.apply
        thetas = self.graph_layer(images)
        fake_lengths = dp_function(thetas, self.adj_array, self.rev_adj,self.max_op,self.null)
        return fake_lengths