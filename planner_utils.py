import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from graph_utils import *
import numpy as np
import networkx as nx


def create_plan(target_folder,model):
    target_graphs, positions = process_all_g_files(target_folder)
    target_pyg = [convert_to_pyg_data(graph) for graph in target_graphs][0]
    original_indices = list(range(target_pyg.num_nodes))
    removal_order = []
    model.eval()
    with torch.no_grad():
        while target_pyg.num_nodes > 1:
            prediction = model(target_pyg)
            prediction = prediction.squeeze()
            predicted_probs = torch.sigmoid(prediction)

            max_prob_node = predicted_probs.argmax().item()
            removal_order.append(original_indices[max_prob_node])
            target_pyg = remove_node_and_edges(target_pyg, max_prob_node)
            del original_indices[max_prob_node]
        if target_pyg.num_nodes == 1:
            remaining_node_original_index = original_indices[0]
            removal_order.append(remaining_node_original_index)

    return removal_order[::-1]

def remove_node_and_edges(data, node_idx):
    # Create a mask for the nodes to keep
    node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    node_mask[node_idx] = False
    data.x = data.x[node_mask]
    edge_mask = (data.edge_index[0] != node_idx) & (data.edge_index[1] != node_idx)
    data.edge_index = data.edge_index[:, edge_mask]
    data.edge_index[0, data.edge_index[0] > node_idx] -= 1
    data.edge_index[1, data.edge_index[1] > node_idx] -= 1
    if data.edge_attr is not None:
        data.edge_attr = data.edge_attr[edge_mask]
    return data