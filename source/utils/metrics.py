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
    def __init__(self):
        super().__init__()
        self.add_state("non_empty_counts", default=torch.tensor([], dtype=torch.float), dist_reduce_fx="cat")

    def update(self, nonempty_clust_sum):
        """
        Update the metric states with the current batch data.

        Args:
            nonempty_clust_sum (torch.Tensor): Sum of non-empty clusters in the current batch,
            obtained by summing along the last axis of the non-empty cluster tensor.
        """
        if not isinstance(nonempty_clust_sum, torch.Tensor):
            nonempty_clust_sum = torch.tensor(nonempty_clust_sum, dtype=torch.float)
        
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