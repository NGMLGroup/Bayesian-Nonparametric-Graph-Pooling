import torch
import torch.nn.functional as F
from torch_geometric.nn import DenseGINConv, GINConv, GINEConv, MLP, global_add_pool
from torch_geometric.nn.resolver import activation_resolver
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ._pooling import build_pooler


class ClassificationModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_channels=None,
        num_layers_pre=1,
        num_layers_post=1,
        hidden_channels=64,
        activation='ELU',
        pooler=None,
        pool_kwargs=None,
        pooled_nodes=None,
        use_gine=False,
        dropout=0.5,
    ):
        super().__init__()

        self.num_classes = out_channels
        self.act = activation_resolver(activation)
        self.pooler = pooler
        self.dropout = dropout
        self.using_gine = edge_channels is not None and use_gine

        in_channels, edge_channels = self.init_encoders(in_channels, edge_channels)

        self.conv_layers_pre = torch.nn.ModuleList()
        channels = in_channels
        for _ in range(num_layers_pre):
            mlp = self.build_pre_mlp(channels, hidden_channels, activation)
            if self.using_gine:
                self.conv_layers_pre.append(GINEConv(nn=mlp, train_eps=False, edge_dim=edge_channels))
            else:
                self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
            channels = hidden_channels
        self.pre_norms = self.build_pre_norms(num_layers_pre, hidden_channels)

        self.pool = build_pooler(pooler, channels, pooled_nodes, pool_kwargs)
        self.use_dense_pool = self.pool.is_dense and not getattr(self.pool, 'sparse_output', False)
        self.use_pool_edge_features = pooler == 'kmis'
        post_channels = channels
        if pooler == 'eigen':
            post_channels *= self.pool.num_modes

        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = self.build_post_mlp(post_channels, activation)
            if self.use_dense_pool:
                self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))
            elif self.use_pool_edge_features:
                self.conv_layers_post.append(GINEConv(nn=mlp, train_eps=False, edge_dim=1))
            else:
                self.conv_layers_post.append(GINConv(nn=mlp, train_eps=False))

        self.mlp = self.build_readout_mlp(post_channels, out_channels, activation)

    def init_encoders(self, in_channels, edge_channels):
        return in_channels, edge_channels

    def encode_inputs(self, x, edge_attr):
        return x, edge_attr

    def build_pre_mlp(self, in_channels, hidden_channels, activation):
        return MLP(
            [in_channels, hidden_channels, hidden_channels],
            act=activation,
            **self.pre_mlp_kwargs(),
        )

    def pre_mlp_kwargs(self):
        return {}

    def build_pre_norms(self, num_layers_pre, hidden_channels):
        return None

    def apply_pre_layer(self, x, layer_idx):
        return x

    def build_post_mlp(self, hidden_channels, activation):
        return MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)

    def build_readout_mlp(self, hidden_channels, out_channels, activation):
        return MLP(
            [hidden_channels, hidden_channels, hidden_channels // 2, out_channels],
            act=activation,
            **self.readout_mlp_kwargs(),
        )

    def readout_mlp_kwargs(self):
        return {'norm': None, 'dropout': self.dropout}

    def apply_readout(self, x):
        return self.mlp(x)

    def forward(self, data):
        x = data.x
        adj = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        edge_weight = getattr(data, 'edge_weight', None)
        batch = data.batch
        x, edge_attr = self.encode_inputs(x, edge_attr)

        for layer_idx, layer in enumerate(self.conv_layers_pre):
            if self.using_gine:
                x = self.act(layer(x, adj, edge_attr=edge_attr))
            else:
                x = self.act(layer(x, adj))
            x = self.apply_pre_layer(x, layer_idx)

        pool_out = self.pool(x=x, adj=adj, edge_weight=edge_weight, batch=batch)
        x, adj = pool_out.x, pool_out.edge_index
        edge_weight = pool_out.edge_weight
        if pool_out.batch is not None:
            batch = pool_out.batch
        s = pool_out.so.s
        loss_d = pool_out.loss
        aux_loss = sum(pool_out.get_loss_value()) if loss_d is not None else x.new_zeros(())
        clust_occup = None
        if self.pooler == 'bnpool' and self.pool.batched:
            clust_occup = s * pool_out.so.in_mask.unsqueeze(-1)

        for layer in self.conv_layers_post:
            if self.use_pool_edge_features:
                if edge_weight is None:
                    edge_weight = x.new_ones(adj.size(1))
                x = self.act(layer(x, adj, edge_weight.unsqueeze(-1)))
            else:
                x = self.act(layer(x, adj))

        if self.use_dense_pool:
            x = torch.sum(x, dim=1)
        else:
            x = global_add_pool(x, batch)
        x = self.apply_readout(x)

        return x, aux_loss, loss_d, clust_occup


class MolhivModel(ClassificationModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_channels=None,
        num_layers_pre=1,
        num_layers_post=1,
        hidden_channels=64,
        activation='relu',
        pooler=None,
        pool_kwargs=None,
        pooled_nodes=None,
        use_gine=False,
        dropout=0.0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            edge_channels=edge_channels,
            num_layers_pre=num_layers_pre,
            num_layers_post=num_layers_post,
            hidden_channels=hidden_channels,
            activation=activation,
            pooler=pooler,
            pool_kwargs=pool_kwargs,
            pooled_nodes=pooled_nodes,
            use_gine=use_gine,
            dropout=dropout,
        )

    def init_encoders(self, in_channels, edge_channels):
        self.atom_encoder = AtomEncoder(100)
        self.bond_encoder = BondEncoder(100)
        return 100, 100

    def encode_inputs(self, x, edge_attr):
        return self.atom_encoder(x), self.bond_encoder(edge_attr)

    def pre_mlp_kwargs(self):
        return {'norm': None}

    def build_pre_norms(self, num_layers_pre, hidden_channels):
        return torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers_pre)]
        )

    def apply_pre_layer(self, x, layer_idx):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pre_norms[layer_idx](x)

    def readout_mlp_kwargs(self):
        return {'norm': None}

    def apply_readout(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mlp(x)
