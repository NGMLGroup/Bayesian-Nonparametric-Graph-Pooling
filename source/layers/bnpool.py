import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence, Beta


def _align_dimensions(rec_adj, adj, pos_weight):
    """
    Ensures that rec_adj, adj, and pos_weight have the correct dimensions.
    """
    # Assert that rec_adj has either 3 or 4 dimensions
    assert rec_adj.ndim in [3, 4]
    
    # If rec_adj has 3 dimensions, add an extra dimension at the beginning
    if rec_adj.ndim == 3:
        rec_adj = rec_adj.unsqueeze(0)
    
    # Unpack the shape of rec_adj
    P, B, N, N = rec_adj.shape

    # If pos_weight is not None, align its dimensions
    if pos_weight is not None:
        pos_weight = pos_weight.unsqueeze(0).expand(P, -1, -1, -1)

    # Align the dimensions of adj
    adj = adj.unsqueeze(0).expand(P, -1, -1, -1)

    # Return the aligned tensors
    return rec_adj, adj, pos_weight


class BNPool(nn.Module):
    """
    Bayesian Nonparametric Pooling layer.

    Args:
        emb_size (int): Size of the input embeddings.
        n_clusters (int): Maximum number of clusters.
        n_particles (int): Number of particles for the Stick Breaking Process.
        alpha_DP (float): Concentration parameter of the Dirichlet Process.
        sigma_K (float): Variance of the Gaussian prior for the cluster-cluster prob. matrix.
        mu_K (float): Mean of the Gaussian prior for the cluster-cluster prob. matrix.
        k_init (float): Initial value for the cluster-cluster prob. matrix.
        eta (float): Coefficient for the KL divergence loss.
        rescale_loss (bool): Whether to rescale the loss by the number of nodes.
        balance_links (bool): Whether to balance the links in the adjacency matrix.
        train_K (bool): Whether to train the cluster-cluster prob. matrix.
    """
    def __init__(self,
                 emb_size: int,
                 n_clusters: int = 50,
                 n_particles:int = 1,
                 alpha_DP: float = 10,
                 sigma_K: float = 1.0,
                 mu_K: float = 10.0,
                 k_init: float = 1.0,
                 eta: float = 1.0,
                 rescale_loss:bool = True,
                 balance_links: bool = True,
                 train_K: bool = True):
        super(BNPool, self).__init__()
        self.n_clusters = n_clusters
        self.n_particles = n_particles
        self.rescale_loss = rescale_loss
        self.balance_links = balance_links
        self.train_K = train_K
        self.eta = eta  # coefficient for the kl_loss

        # Prior for the Stick Breaking Process
        self.register_buffer('ones_C', th.ones(self.n_clusters - 1))
        self.register_buffer('alpha_DP', th.ones(self.n_clusters - 1) * alpha_DP)
        
        # Prior for the cluster-cluster prob. matrix
        self.register_buffer('sigma_K', th.tensor(sigma_K))
        self.register_buffer('mu_K', mu_K * th.eye(self.n_clusters, self.n_clusters) -
                             mu_K * (1 - th.eye(self.n_clusters, self.n_clusters)))
        
        # Posterior distributions for the sticks
        self.W = th.nn.Linear(emb_size, 2*(self.n_clusters-1), bias=False)

        # Posterior cluster-cluster prob matrix
        self.mu_tilde = th.nn.Parameter(k_init * th.eye(self.n_clusters, self.n_clusters) -
                                        k_init * (1-th.eye(self.n_clusters, self.n_clusters)), requires_grad=train_K)


    @staticmethod
    def _compute_pi_given_sticks(sticks):
        device = sticks.device
        log_v = th.concat([th.log(sticks), th.zeros(*sticks.shape[:-1], 1, device=device)], dim=-1)
        log_one_minus_v = th.concat([th.zeros(*sticks.shape[:-1], 1, device=device), th.log(1 - sticks)], dim=-1)
        pi = th.exp(log_v + th.cumsum(log_one_minus_v, dim=-1))  # has shape [n_particles, batch, n_nodes, n_clusters]
        return pi

    def get_S(self, node_embs):
        out = th.clamp(F.softplus(self.W(node_embs)), min=1e-3, max=1e3) 
        alpha_tilde, beta_tilde = th.split(out, self.n_clusters-1, dim=-1)
        self.alpha_tilde = alpha_tilde  # Stored for logging purposes
        self.beta_tilde = beta_tilde    # Stored for logging purposes
        q_pi = Beta(alpha_tilde, beta_tilde)
        stick_fractions = q_pi.rsample([self.n_particles])
        S = self._compute_pi_given_sticks(stick_fractions)
        return S, q_pi

    def _compute_dense_coarsened_graph(self, S, adj, x, mask):
        """
        Compute the coarsened graph by applying the clustering matrix S 
        to the adjacency matrix and the node embeddings.
        """
        x_pool = th.einsum('bnk,bnf->bkf', S, x)

        adj_pool = th.matmul(th.matmul(S.transpose(1, 2), adj), S) # has shape B x K x K
        nonempty_clust = ((S*mask.unsqueeze(-1)) >= 0.2).sum(axis=1) > 0 # Tracks which clusters are non-empty

        # remove element on the diagonal
        ind = th.arange(self.n_clusters, device=adj_pool.device) # [0,1,2,3,...,K-1]
        adj_pool[:, ind, ind] = 0
        deg = th.einsum('ijk->ij', adj_pool)
        deg = th.sqrt(deg+1e-4)[:, None]
        adj_pool = (adj_pool / deg) / deg.transpose(1, 2)

        return adj_pool, x_pool, nonempty_clust
    
    def forward(self, node_embs, adj, node_mask, 
                pos_weight=None,    
                return_coarsened_graph=None):

        # Compute the node assignments and the reconstructed adjacency matrix
        S, q_z = self.get_S(node_embs)  # S has shape P x B x N x K

        # Compute the losses
        rec_loss = self.dense_rec_loss(S, adj, pos_weight)   # has shape P x B x N x N
        kl_loss = self.eta * self.pi_prior_loss(q_z)  # has shape B x N

        K_prior_loss = self.K_prior_loss() if self.train_K else 0  # has shape 1
        
        # Sum losses over nodes by considering the actual number of nodes for each graph
        if not th.all(node_mask):
            edge_mask = th.einsum('bn,bm->bnm', node_mask, node_mask).unsqueeze(0)  # has shape 1 x B x N x N
            rec_loss = rec_loss * edge_mask
            kl_loss = kl_loss * node_mask
        rec_loss = rec_loss.sum((-1, -2))  # has shape P x B
        kl_loss = kl_loss.sum(-1)          # has shape B

        # Normalize the losses
        if self.rescale_loss:
            N = node_mask.sum(-1)
            rec_loss = rec_loss / N.unsqueeze(0)
            kl_loss = kl_loss / N
            K_prior_loss = K_prior_loss / N

        # Build the output dictionary
        loss_d = {'quality': rec_loss.mean(),
                  'kl': self.eta * kl_loss.mean()}
        if self.train_K:
            loss_d['K_prior'] = K_prior_loss.mean()

        S = S[0] # Take only the first particle

        if return_coarsened_graph:
            out_adj, out_x, nonempty_clust = self._compute_dense_coarsened_graph(S, adj, node_embs, node_mask)
            return S, out_x, out_adj, nonempty_clust, loss_d
        else:
            return S, loss_d

    def dense_rec_loss(self, S, adj, pos_weight):
        """
        BCE loss between the reconstructed adjacency matrix and the true one.
        We use with logits because K is no longer the indentiy matrix.
        """
        p_adj = S @ self.mu_tilde @ S.transpose(-1, -2)

        if self.balance_links and pos_weight is None:
            raise ValueError("pos_weight must be provided when balance_links is True")
        p_adj, adj, pos_weight = _align_dimensions(p_adj, adj, pos_weight)

        loss = F.binary_cross_entropy_with_logits(p_adj, adj, weight=pos_weight, reduction='none')

        return loss  # has shape P x B x N x N

    def pi_prior_loss(self, q_pi):
        """
        KL divergence between the posterior and the prior of the Stick Breaking Process.
        """
        p_pi = Beta(self.get_buffer('ones_C'), self.get_buffer('alpha_DP'))
        loss = kl_divergence(q_pi, p_pi).sum(-1)
        return loss  # has shape B x N

    def K_prior_loss(self):
        """
        KL divergence between the posterior and the prior of the cluster-cluster prob. matrix.
        """
        mu_K, sigma_K = self.get_buffer('mu_K'), self.get_buffer('sigma_K')
        K_prior_loss = (0.5 * (self.mu_tilde - mu_K) ** 2 / sigma_K).sum()
        return K_prior_loss  # has shape 1