import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.nn import dense_mincut_pool, dense_diff_pool, DMoNPooling
from torch_geometric.utils import to_dense_batch
from source.layers.just_balance import just_balance_pool
from source.layers.bnpool import BNPool
from source.utils.misc import TensorsCache

class ClusterModel(torch.nn.Module):
    """
    Torch module consisting of a stack of GCN layers followed by a pooling layer.

    Args:
        in_channels (int): Size of node features
        num_layers_pre (int): Number of GCN layers before pooling
        hidden_channels (int): Dimensionality of node embeddings
        activation (str): Activation of the MLP in GCN 
        pooler (str): Pooling method
        pool_kwargs (dict): Pooling method kwargs
        pooled_nodes (int): Number of nodes after pooling
        use_cache (bool): Cache computation of dense adjacency
    """

    def __init__(self, 
                 in_channels,                           # Size of node features
                 num_layers_pre=1,                      # Number of GCN layers before pooling
                 hidden_channels=64,                    # Dimensionality of node embeddings
                 activation='ELU',                      # Activation of the GCN 
                 pooler=None,                           # Pooling method
                 pool_kwargs=None,                      # Pooling method kwargs
                 pooled_nodes=None,                     # Number of nodes after pooling
                 use_cache=True                         # Use cache for tensors
                 ):
        super().__init__()
        
        self.num_layers_pre = num_layers_pre
        self.hidden_channels = hidden_channels
        self.act = activation_resolver(activation)
        self.pooler = pooler
        self.cache = TensorsCache(use_cache)

        # Pre-pooling block            
        self.conv_layers_pre = torch.nn.ModuleList()
        for _ in range(num_layers_pre):
            self.conv_layers_pre.append(GCNConv(in_channels, hidden_channels))
            in_channels = hidden_channels
                        
        # Pooling block
        if pooler in ['diffpool','mincut']:
            self.pool = Linear(hidden_channels, pooled_nodes)
            self.l1_weight = pool_kwargs['l1_weight']
            self.l2_weight = pool_kwargs['l2_weight']
            if pooler=='diffpool':
                self.normalize = pool_kwargs['normalize']
        elif pooler=='dmon':
            self.pool = DMoNPooling(hidden_channels, pooled_nodes)
            self.l1_weight = pool_kwargs['l1_weight']
            self.l2_weight = pool_kwargs['l2_weight']
            self.l3_weight = pool_kwargs['l3_weight']
        elif pooler=='just_balance':
            self.pool = Linear(hidden_channels, pooled_nodes)
            self.normalize = pool_kwargs['normalize']
        elif pooler=='bnpool':
            self.pool = BNPool(emb_size=hidden_channels, **pool_kwargs)
            self.balance_links = pool_kwargs['balance_links']


    def forward(self, data):
        """
        ⏩ 
        """
        x = data.x    
        adj = data.edge_index
        batch = data.batch

        ### pre-pooling block
        for layer in self.conv_layers_pre:  
            x = self.act(layer(x, adj))

        ### pooling block
        if self.pooler in ['diffpool','mincut','dmon','just_balance', 'bnpool']:
            adj = self.cache.get_and_cache_A(adj, batch)  # has shape B x N x N
            x, mask = to_dense_batch(x, batch)  # has shape B x N
            if self.pooler=='diffpool':
                s = self.pool(x)
                x_pool, adj_pool, l1, l2 = dense_diff_pool(x, adj, s, mask, normalize=self.normalize)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2
            elif self.pooler=='mincut':
                s = self.pool(x)
                x_pool, adj_pool, l1, l2 = dense_mincut_pool(x, adj, s, mask)
                aux_loss = self.l1_weight*l1 + self.l2_weight*l2
            elif self.pooler=='dmon':  
                s, x_pool, adj_pool, l1, l2, l3 = self.pool(x, adj, mask)
                aux_loss = self.l1_weight*l1 + self.l1_weight*l2 + self.l1_weight*l3
            elif self.pooler=='just_balance':
                s = self.pool(x)
                x_pool, adj_pool, aux_loss = just_balance_pool(x, adj, s, mask=mask, 
                                                               normalize=self.normalize)
            elif self.pooler=='bnpool':
                pos_weight = None if not self.balance_links else self.cache.get_and_cache_pos_weight(adj, mask)
                s, x_pool, adj_pool, _, loss_d = self.pool(x, adj=adj, node_mask=mask,
                                                           pos_weight=pos_weight,
                                                           return_coarsened_graph=True)
                aux_loss = loss_d['quality'] + loss_d['kl'] + loss_d['K_prior']
        else:
            raise KeyError("Unrecognized pooling method")

        if 'loss_d' not in locals():
            loss_d=None
            
        if self.pooler in ['diffpool','mincut','just_balance']:
            s = torch.softmax(s, dim=-1)
        
        return x_pool, adj_pool, batch, s, aux_loss, loss_d