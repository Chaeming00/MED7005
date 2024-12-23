import numpy as np
import torch
import os
from collections import namedtuple
import torch_geometric
import torch.nn.functional as F
import sys
sys.path.insert(0, "../")
from v1.dataset import TxPDataset
import argparse


def pad_or_trim_tensor(tensor, target_size):
    """
    Pads or trims a tensor to the target size.
    For 1D or 2D tensors, the target size can be (D,) or (N, D).
    """
    if len(tensor.shape) == 2:  # Handle 2D tensors (e.g., (N, D))
        N, D = tensor.shape
        if len(target_size) == 1:  # If target_size is (D,)
            target_size = (N, target_size[0])  # Convert to (N, D)
        pad_d = max(target_size[1] - D, 0)
        padded = F.pad(tensor, (0, pad_d), mode="constant", value=0)  # Pad last dimension
        return padded[:, :target_size[1]]  # Trim if larger
    elif len(tensor.shape) == 1:  # Handle 1D tensors
        pad_d = max(target_size[0] - tensor.size(0), 0)
        padded = F.pad(tensor, (0, pad_d), mode="constant", value=0)  # Pad last dimension
        return padded[:target_size[0]]  # Trim if larger
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")


def get_edge(x):
    """
    Generates edges for window nodes using radius_graph.
    """
    r_value = x.max() * 0.1  # 좌표 범위의 10%를 반경으로 설정
    return torch_geometric.nn.radius_graph(
        x, r=r_value, batch=None, loop=False, max_num_neighbors=5
    )


def get_cross_edge(img_data):
    op_list = [op.unsqueeze(0) if op.ndim == 1 else op for op in [i[3] for i in img_data]]
    opy_list = [opy.unsqueeze(0) if opy.ndim == 1 else opy for opy in [i[4] for i in img_data]]

    # Ensure op and opy are always 3D tensors
    op_list = [
        op.unsqueeze(0) if op.ndim == 2 else op for op in op_list
    ]
    opy_list = [
        opy.unsqueeze(0) if opy.ndim == 2 else opy for opy in opy_list
    ]

    # Debugging shapes
    print("Debugging op_list and opy_list shapes:")
    for idx, op in enumerate(op_list):
        print(f"op_list[{idx}] shape: {op.shape}")
    for idx, opy in enumerate(opy_list):
        print(f"opy_list[{idx}] shape: {opy.shape}")

    max_nodes = max(opy.size(1) for opy in opy_list)

    # Padding op_list and opy_list
    padded_op_list = [
        F.pad(op, (0, 0, 0, max_nodes - op.size(1)), value=0) if op.size(1) < max_nodes else op for op in op_list
    ]
    padded_opy_list = [
        F.pad(opy, (0, 0, 0, max_nodes - opy.size(1)), value=0) if opy.size(1) < max_nodes else opy for opy in opy_list
    ]

    # Concatenate lists
    op = torch.cat(padded_op_list, dim=0)
    opy = torch.cat(padded_opy_list, dim=0)

    # Verify shapes
    if op.ndim != 3 or opy.ndim != 3:
        raise ValueError(f"Expected op and opy to have 3 dimensions, but got shapes {op.shape}, {opy.shape}")

    # Generate cross edges
    b, n, c = op.size()
    ops = torch.cat((op, opy), -1).view(b * n, -1)
    ops, inverse = torch.unique(ops, dim=0, return_inverse=True)

    unique_op = ops[:, :c]
    unique_opy = ops[:, c:]
    cross_edge = torch.stack((torch.arange(b).repeat_interleave(n), inverse))

    # Debugging results
    print(f"unique_op shape: {unique_op.shape}, unique_opy shape: {unique_opy.shape}, cross_edge shape: {cross_edge.shape}")

    return unique_op, unique_opy, cross_edge



# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--savename", required=True, type=str)
parser.add_argument("--size", required=True, type=int)
parser.add_argument("--numk", required=True, type=int)
parser.add_argument("--mdim", required=True, type=str)
parser.add_argument("--index_path", required=True, type=str)
parser.add_argument("--emb_path", required=True, type=str)
parser.add_argument("--data", required=True, type=str)
parser.add_argument("--cancer_type", required=True, type=str)
parser.add_argument("--fold", required=True, type=int)

args = parser.parse_args()

# Directory setup
savename = os.path.join(args.savename, args.cancer_type)
os.makedirs(savename, exist_ok=True)
foldername = os.path.join(savename, f"graphs_{args.numk}")
os.makedirs(foldername, exist_ok=True)

# Dataset setup
temp_arg = namedtuple("arg", ["size", "numk", "mdim", "index_path", "emb_path", "data", "fold"])
temp_arg = temp_arg(args.size, args.numk, args.mdim, args.index_path, args.emb_path, args.data, args.fold)

train_dataset = TxPDataset(
    root_dir=args.data,
    cancer_type=args.cancer_type,
    transform=None,
    args=temp_arg,
    train=True
)

print(f"Train dataset length: {len(train_dataset)}")

for iid in range(len(train_dataset.data)):
    print(f"Processing IID: {iid}")
    item = train_dataset[iid]
    pos, p, py, op, opy = item["pos"], item["p_feature"], item["count"], item["op_feature"], item["op_count"]

    op = op.unsqueeze(0) if op.dim() == 1 else op
    opy = opy.unsqueeze(0) if opy.dim() == 1 else opy
    img_data = [[pos, p, py, op, opy]]

    # Process positions, features, and count
    max_nodes = max(pos.size(0), 1)
    max_features = max(p.size(1) if p.ndim == 2 else 1, 1)

    processed_pos = [pad_or_trim_tensor(pos, (max_nodes, 2))]
    processed_x = [pad_or_trim_tensor(p, (max_nodes, max_features))]
    processed_y = [pad_or_trim_tensor(py, (max_nodes, 1))]

    # Edges
    all_pos = torch.cat(processed_pos, dim=0)
    window_edge = get_edge(all_pos)
    unique_op, unique_opy, cross_edge = get_cross_edge(img_data)

    # Build HeteroData
    data = torch_geometric.data.HeteroData()
    data["window"].pos = torch.cat(processed_pos, dim=0)
    data["window"].x = torch.cat(processed_x, dim=0)
    data["window"].y = torch.cat(processed_y, dim=0)
    data["window"].num_nodes = data["window"].x.size(0)

    example_x = torch.cat((unique_op, unique_opy), -1)
    data["example"].x = example_x
    data["example"].num_nodes = data["example"].x.size(0)

    data['window', 'near', 'window'].edge_index = window_edge
    data["example", "refer", "window"].edge_index = cross_edge[[1, 0]]
    edge_index = torch_geometric.nn.knn_graph(data["example"]["x"], k=3, loop=False)
    data["example", "close", "example"].edge_index = edge_index

    # Save data
    torch.save(data, f"{foldername}/{iid}.pt")
    print(f"Saved graph for IID {iid} to {foldername}/{iid}.pt")
