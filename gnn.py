import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, BatchNorm1d
from torch_geometric.data import *

class CustomEdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')
        self.mlp = Seq(Linear(2 * in_channels + 3, out_channels), ReLU(), Linear(out_channels, out_channels))
    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)
    def message(self, x_i, x_j, edge_weight):
        tmp = torch.cat([x_i, x_j - x_i, edge_weight.view(-1, 3)], dim=1)
        return self.mlp(tmp)

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = CustomEdgeConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.sigmoid(self.out(x))
        return x
    
def online_train(model, lr, new_graph_data, batch_size=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    for graph in DataLoader(new_graph_data, batch_size=batch_size):
        out = model(graph)
        loss = criterion(out, graph.y.view(-1, 1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()