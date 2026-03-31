import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.resolver import activation_resolver
from ._pooling import build_pooler


class ClusterModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers_pre=1,
        hidden_channels=64,
        activation='ELU',
        pooler=None,
        pool_kwargs=None,
        pooled_nodes=None,
    ):
        super().__init__()

        self.act = activation_resolver(activation)
        self.pooler = pooler

        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            self.conv_layers_pre.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels

        self.pool = build_pooler(pooler, in_channels, pooled_nodes, pool_kwargs)


    def forward(self, data):
        x = data.x
        adj = data.edge_index
        edge_weight = getattr(data, 'edge_weight', None)
        batch = data.batch

        for layer in self.conv_layers_pre:
            x = self.act(layer(x, adj))

        pool_out = self.pool(x=x, adj=adj, edge_weight=edge_weight, batch=batch)
        s = pool_out.so.s
        loss_d = pool_out.loss
        aux_loss = sum(pool_out.get_loss_value()) if loss_d is not None else x.new_zeros(())
        if s.dim() == 2:
            s = s.unsqueeze(0)

        return s, aux_loss, loss_d
