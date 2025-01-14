import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, DenseGINConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

# Local imports
from source.layers.bnpool import BNPool
from source.utils import NormalizeAdjSparse_with_ea
from source.utils.misc import TensorsCache
from source.utils.data import get_train_val_test_datasets


### Get the data
dataset = TUDataset(root="data/TUDataset", name='MUTAG', pre_transform=NormalizeAdjSparse_with_ea())
train_dataset, val_dataset, test_dataset = get_train_val_test_datasets(dataset, seed=777, n_folds=10, fold_id=0)
tr_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)


### Model definition
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_features = dataset.num_features
        num_classes = dataset.num_classes
        hidden_channels = 64  
        self.cache = TensorsCache(use_cache=False)

        # First MP layer
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        # BNPool layer
        self.pool = BNPool(emb_size=hidden_channels)

        # Second MP layer
        self.conv2 = DenseGINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )

        # Readout layer
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):

        # First MP layer
        x = self.conv1(x, edge_index)

        # Transform to dense batch
        x, mask = to_dense_batch(x, batch)
        adj = self.cache.get_and_cache_A(edge_index, batch)

        # BNPool layer
        pos_weight = self.cache.get_and_cache_pos_weight(adj, mask)
        _, x, adj, _, loss_d = self.pool(x, adj=adj, node_mask=mask,
                                         pos_weight=pos_weight,
                                         return_coarsened_graph=True)
        aux_loss = loss_d['quality'] + loss_d['kl'] + loss_d['K_prior']

        # Second MP layer
        x = self.conv2(x, adj)

        # Global pooling
        x = x.mean(dim=1)

        # Readout layer
        x = self.lin(x)

        return F.log_softmax(x, dim=-1), aux_loss


### Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)


def train():
    model.train()
    loss_all = 0

    for data in tr_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        output, aux_loss = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y.view(-1)) + aux_loss
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.batch)[0].max(dim=1)[1]
        correct += int(pred.eq(data.y.view(-1)).sum())
    return correct / len(loader.dataset)


### Training loop
best_val_acc = test_acc = 0
for epoch in range(1, 151):
    train_loss = train()
    val_acc = test(val_dataloader)
    if val_acc > best_val_acc:
        test_acc = test(test_dataloader)
        best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.1f}, '
          f'Val Acc: {val_acc:.3f}, Test Acc: {test_acc:.3f}')