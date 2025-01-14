import torch
from torch.nn import Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import (GINConv, GINEConv, MLP, DenseGINConv,
                                dense_mincut_pool, dense_diff_pool, 
                                global_add_pool, DMoNPooling, TopKPooling,
                                ASAPooling, graclus)

# Local imports
from source.layers.edgepool.edge_pool import EdgePooling
from source.layers.kmis.kmis_pool import KMISPooling
from source.layers.sum_pool import sum_pool
from source.layers.just_balance import just_balance_pool
from source.layers.bnpool import BNPool
from source.utils import cluster_to_s
from source.utils.misc import TensorsCache


class ClassificationModel(torch.nn.Module):
    """
    Torch module consisting of:
    - a stack of GIN layers before pooling
    - a pooling layer
    - a stack of GIN layers after pooling
    - a readout layer

    Args:
        in_channels (int): Size of node features
        out_channels (int): Number of classes
        edge_channels (int): Size of edge features
        num_layers_pre (int): Number of GIN layers before pooling
        num_layers_post (int): Number of GIN layers after pooling
        hidden_channels (int): Dimensionality of node embeddings
        activation (str): Activation of the MLP in GIN
        pooler (str): Name of the pooling method
        pool_kwargs (dict): Pooling method kwargs
        pooled_nodes (int): Number of nodes after pooling
        use_gine (bool): Use GINE instead of GIN
        dropout (float): Dropout of the readout
    """
    def __init__(self, 
                 in_channels,                           # Size of node features
                 out_channels,                          # Number of classes
                 edge_channels=None,                    # Size of edge features
                 num_layers_pre=1,                      # Number of GIN layers before pooling
                 num_layers_post=1,                     # Number of GIN layers after pooling
                 hidden_channels=64,                    # Dimensionality of node embeddings
                 activation='ELU',                      # Activation of the MLP in GIN 
                 pooler=None,                           # Pooling method
                 pool_kwargs=None,                      # Pooling method kwargs
                 pooled_nodes=None,                     # Number of nodes after pooling
                 use_gine=False,                        # Use GINE instead of GIN
                 dropout=0.5,                           # Dropout of the readout
                 ):
        super().__init__()
        
        self.num_classes = out_channels
        self.act = activation_resolver(activation)
        self.pooler = pooler
        if edge_channels is not None and use_gine:
            self.using_gine = True
        else:
            self.using_gine = False
        self.cache = TensorsCache(use_cache=False)

        ### Pre-pooling block            
        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act=activation)
            if self.using_gine:
                self.conv_layers_pre.append(GINEConv(nn=mlp, train_eps=False, edge_dim=edge_channels))
            else:
                self.conv_layers_pre.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
                        
        ### Pooling block
        if pooler in ['diffpool','mincut']:
            self.pool = Linear(hidden_channels, pooled_nodes)
            self.l1_weight = pool_kwargs['l1_weight']
            self.l2_weight = pool_kwargs['l2_weight']
        elif pooler=='dmon':
            self.pool = DMoNPooling(hidden_channels, pooled_nodes)
            self.l1_weight = pool_kwargs['l1_weight']
            self.l2_weight = pool_kwargs['l2_weight']
            self.l3_weight = pool_kwargs['l3_weight']
        elif pooler=='just_balance':
            self.pool = Linear(hidden_channels, pooled_nodes)
            self.normalize = pool_kwargs['normalize']
        elif pooler=='topk':
            self.pool = TopKPooling(hidden_channels, **pool_kwargs)
        elif pooler=='asapool':
            self.pool = ASAPooling(hidden_channels, **pool_kwargs)  
        elif pooler=='edgepool':
            self.pool = EdgePooling(hidden_channels)
        elif pooler in ['graclus','nopool']:
            pass
        elif pooler=='kmis':
            self.pool = KMISPooling(hidden_channels, **pool_kwargs)
        elif pooler=='bnpool':
            self.pool = BNPool(emb_size=hidden_channels, **pool_kwargs)
            self.balance_links = pool_kwargs['balance_links']

        ### Post-pooling block
        self.conv_layers_post = torch.nn.ModuleList()
        for _ in range(num_layers_post):
            mlp = MLP([hidden_channels, hidden_channels, hidden_channels], act=activation, norm=None)
            if pooler in ['diffpool','mincut','dmon','just_balance','bnpool']:
                self.conv_layers_post.append(DenseGINConv(nn=mlp, train_eps=False))
            elif pooler in ['kmis']: # 💡 Add here other methods that generate a sparse weighted pooled adj
                self.conv_layers_post.append(GINEConv(nn=mlp, train_eps=False, edge_dim=1))
            else:
                self.conv_layers_post.append(GINConv(nn=mlp, train_eps=False))

        ### Readout
        self.mlp = MLP([hidden_channels, hidden_channels, hidden_channels//2, out_channels], 
                        act=activation,
                        norm=None,
                        dropout=dropout)

    def forward(self, data):
        """
        ⏩ 
        """
        x = data.x    
        adj = data.edge_index
        ea = getattr(data, 'edge_attr', None)
        batch = data.batch

        ### Pre-pooling block
        if self.using_gine:
            for layer in self.conv_layers_pre:  
                x = self.act(layer(x, adj, edge_attr=ea))
        else:
            for layer in self.conv_layers_pre:  
                x = self.act(layer(x, adj))

        ### Pooling block
        if self.pooler in ['diffpool','mincut','dmon','just_balance', 'bnpool']:
            x, mask = to_dense_batch(x, batch)
            adj = self.cache.get_and_cache_A(adj, batch)
            if self.pooler=='diffpool':
                s = self.pool(x)
                x, adj, l1, l2 = dense_diff_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2
            elif self.pooler=='mincut':
                s = self.pool(x)
                x, adj, l1, l2 = dense_mincut_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l2_weight*l2
            elif self.pooler=='dmon':  
                s, x, adj, l1, l2, l3 = self.pool(x, adj, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2 + self.l1_weight*l3
            elif self.pooler=='just_balance':
                s = self.pool(x)
                x, adj, aux_loss = just_balance_pool(x, adj, s, mask=mask, 
                                                     normalize=self.normalize)
            elif self.pooler=='bnpool':
                pos_weight = None if not self.balance_links else self.cache.get_and_cache_pos_weight(adj, mask)
                s, x, adj, nonempty_clust, loss_d = self.pool(x, adj=adj, node_mask=mask,
                                                              pos_weight=pos_weight,
                                                              return_coarsened_graph=True)
                aux_loss = loss_d['quality'] + loss_d['kl'] + loss_d['K_prior']
        elif self.pooler=='topk':
            x, adj, _, batch, perm, _ = self.pool(x, adj, edge_attr=None, batch=batch)
            s = cluster_to_s(node_index=perm,
                             num_nodes=data.x.size(0),
                             cluster_index=torch.arange(perm.size(0), device=x.device),
                             num_clusters=perm.size(0))
        elif self.pooler=='asapool':
            x, adj, _, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooler=='edgepool':
            x, adj, batch, _ = self.pool(x, adj, batch=batch)
        elif self.pooler=='kmis':
            x, adj, edge_weight, batch, _, cluster = self.pool(x, adj, edge_attr=None, batch=batch)
            s = cluster_to_s(cluster_index=cluster)
        elif self.pooler=='graclus':
            data.x = x
            cluster = graclus(adj, num_nodes=data.x.size(0))
            data = sum_pool(cluster, data)
            _, assignment = torch.unique(cluster, sorted=True, return_inverse=True)
            s = cluster_to_s(cluster_index=assignment, 
                             num_nodes=x.size(0), 
                             num_clusters=data.x.size(0))
            x = data.x    
            adj = data.edge_index
            batch = data.batch
        elif self.pooler=='nopool':
            pass
        else:
            raise KeyError(f"unrecognized pooling method: {self.pooler}")
        
        ### Post-pooling block
        for layer in self.conv_layers_post:  
            if self.pooler in ['kmis']: # 💡 Add all methods that generate a sparse weighted pooled adj
                x = self.act(layer(x, adj, edge_weight.unsqueeze(-1)))
            else:
                x = self.act(layer(x, adj))

        ### Readout
        if self.pooler in ['diffpool','mincut','dmon','just_balance','bnpool']:
            x = torch.sum(x, dim=1)
        else:
            x = global_add_pool(x, batch)
        x = self.mlp(x)

        if 'edge_weight' not in locals():
            edge_weight=None
        if 'aux_loss' not in locals():
            aux_loss=0
        if 'loss_d' not in locals():
            loss_d=None
        if 's' not in locals():
            s=None
        if 'nonempty_clust' not in locals():
            nonempty_clust=None

        if self.pooler in ['diffpool','mincut','just_balance']:
            s = torch.softmax(s, dim=-1)

        return x, adj, edge_weight, batch, s, aux_loss, loss_d, nonempty_clust