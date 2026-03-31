import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric


class FuzzyClusterCosine(Metric):
    """
    Compute the cosine similarity between soft cluster assignments and class labels.

    This metric accumulates the soft cluster assignments and class labels over all batches,
    and computes the cosine similarity at the end.

    Note:
        - This may consume a lot of memory for large datasets.
    """
    def __init__(
            self, 
            num_clusters: int, 
            num_classes: int, 
            dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_clusters = num_clusters
        self.num_classes = num_classes

        # Initialize states as empty tensors
        self.add_state("U_all", default=torch.empty(0, num_clusters), dist_reduce_fx="cat")
        self.add_state("V_all", default=torch.empty(0, dtype=torch.long), dist_reduce_fx="cat")

    def update(self, U: torch.Tensor, V: torch.Tensor):
        """
        Update the metric states with the current batch data.

        Args:
            U (torch.Tensor): Soft cluster assignments of shape (batch_size, num_clusters).
            V (torch.Tensor): Class labels of shape (batch_size,).
        """
        # Ensure that states are on the same device as input tensors
        device = U.device

        if self.U_all.device != device:
            self.U_all = self.U_all.to(device)
            self.V_all = self.V_all.to(device)

        # Detach tensors to avoid unnecessary computation graph retention
        U = U.detach()
        V = V.detach()

        # Concatenate current batch to the accumulated tensors
        self.U_all = torch.cat([self.U_all, U], dim=0)
        self.V_all = torch.cat([self.V_all, V], dim=0)

    def compute(self):
        """
        Compute the consine similarity between soft cluster assignments 
        and one-hot representations of class labels.

        Returns:
            float: The consine similarity value between 0 and 1.
        """
        U = self.U_all          # Shape: (n_samples, n_clusters)
        V = self.V_all.long()   # Shape: (n_samples,)

        # Convert class labels to one-hot encoding
        V_one_hot = F.one_hot(V, num_classes=self.num_classes).float()

        # Compute agreement matrices
        UUT = torch.matmul(U, U.T)                  # Shape: (n_samples, n_samples)
        VVT = torch.matmul(V_one_hot, V_one_hot.T)  # Shape: (n_samples, n_samples)

        # Compute numerator and denominator
        numerator = torch.sum(UUT * VVT)
        denominator = torch.sqrt(torch.sum(UUT ** 2) * torch.sum(VVT ** 2))

        # Handle zero denominator
        if denominator == 0:
            return torch.tensor(0.0, device=U.device)

        COS = numerator / denominator

        return COS
    

class NonEmptyClusterTracker(Metric):
    """
    Track the sum of non-empty clusters over all batches.
    """
    def __init__(self, n_clusters=None):
        super().__init__()
        self.eps = None if n_clusters is None else 1.0 / n_clusters
        self.add_state("non_empty_counts", default=torch.tensor([], dtype=torch.float), dist_reduce_fx="cat")

    def update(self, nonempty_clust):
        """
        Update the metric states with the current batch data.

        Args:
            nonempty_clust (torch.Tensor): Either per-graph counts or a cluster-occupancy
            tensor with shape `(batch, nodes, clusters)`.
        """
        if not isinstance(nonempty_clust, torch.Tensor):
            nonempty_clust = torch.tensor(nonempty_clust, dtype=torch.float)

        if nonempty_clust.dim() >= 3:
            eps = 0.0 if self.eps is None else self.eps
            nonempty_clust_sum = torch.any(nonempty_clust > eps, dim=1).sum(-1)
        else:
            nonempty_clust_sum = nonempty_clust.to(dtype=torch.float)
        
        # Concatenate the incoming batch sum to the running tensor
        self.non_empty_counts = torch.cat((self.non_empty_counts, nonempty_clust_sum))

    def compute(self):
        """
        Compute the sum of non-empty clusters over all batches.

        Returns:
            torch.Tensor: The sum of non-empty clusters.
        """
        # If empty, return an empty tensor
        if self.non_empty_counts.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=self.non_empty_counts.device)
        
        # Ensure that the size is the same if running on multiple devices
        if self.non_empty_counts.dim() > 1:
            self.non_empty_counts = self.non_empty_counts.view(-1)

        return self.non_empty_counts

    def reset(self):
        """
        Reset to empty tensor.
        """
        self.non_empty_counts = self.non_empty_counts.new_empty(0)
        super().reset()


class ClusterOccupancyTracker(Metric):
    def __init__(self, n_clusters):
        super().__init__()
        self.add_state("clust_occup", default=torch.zeros(n_clusters, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, clust_occup):
        if not isinstance(clust_occup, torch.Tensor):
            clust_occup = torch.tensor(clust_occup, dtype=torch.float)

        self.clust_occup += clust_occup.mean(dim=1).sum(0)

    def compute(self):
        if self.clust_occup.numel() == 0:
            return torch.empty(0, dtype=torch.float, device=self.clust_occup.device)

        if self.clust_occup.dim() > 1:
            self.clust_occup = self.clust_occup.view(-1)

        total = self.clust_occup.sum()
        if total == 0:
            return torch.zeros_like(self.clust_occup)

        return self.clust_occup / total

    def reset(self):
        self.clust_occup = torch.zeros_like(self.clust_occup)
        super().reset()
