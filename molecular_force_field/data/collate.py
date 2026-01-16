"""Collate functions for batching molecular data."""

import torch


def my_collate_fn(batch_list):
    """
    Collate function for CustomDataset.
    
    Args:
        batch_list: List of samples from CustomDataset
        
    Returns:
        Batched data tuple or None if batch is empty
    """
    batch_list = [item for item in batch_list if item is not None]
    if len(batch_list) == 0:
        return None

    pos_list = []
    A_list = []
    batch_idx_list = []
    force_list = []
    target_energy_list = []
    
    edge_src_list = []
    edge_dst_list = []
    edge_shifts_list = []
    cell_list = []
    
    num_nodes_accumulated = 0

    for i, item in enumerate(batch_list):
        read_tensor, target_energy, src, dst, shifts, cell = item
        num_atoms = read_tensor.shape[0]
        
        pos = read_tensor[:, 1:4]
        atom_type = read_tensor[:, 4]
        forces = read_tensor[:, 5:8]
        
        pos_list.append(pos)
        A_list.append(atom_type)
        force_list.append(forces)
        target_energy_list.append(target_energy)
        batch_idx_list.append(torch.full((num_atoms,), i, dtype=torch.long))
        
        # Concatenate graph data (add offset)
        edge_src_list.append(src + num_nodes_accumulated)
        edge_dst_list.append(dst + num_nodes_accumulated)
        edge_shifts_list.append(shifts)
        cell_list.append(cell)
        
        num_nodes_accumulated += num_atoms

    return (
        torch.cat(pos_list, dim=0),
        torch.cat(A_list, dim=0),
        torch.cat(batch_idx_list, dim=0),
        torch.cat(force_list, dim=0),
        torch.stack(target_energy_list),
        torch.cat(edge_src_list, dim=0),
        torch.cat(edge_dst_list, dim=0),
        torch.cat(edge_shifts_list, dim=0),
        torch.stack(cell_list, dim=0)  # (B, 3, 3)
    )


def collate_fn_h5(batch_list):
    """
    Collate function specifically for H5Dataset.
    
    Args:
        batch_list: List of samples from H5Dataset
        
    Returns:
        Batched data tuple
    """
    pos_l, A_l, b_idx_l, force_l, target_l = [], [], [], [], []
    src_l, dst_l, shift_l, cell_l = [], [], [], []
    
    node_offset = 0
    for i, data in enumerate(batch_list):
        num_nodes = data['pos'].shape[0]
        
        # Basic attributes
        pos_l.append(data['pos'])
        A_l.append(data['A'])
        force_l.append(data['force'])
        target_l.append(data['y'])
        # Batch index
        b_idx_l.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        # Cell information (keep [1, 3, 3] for stacking)
        cell_l.append(data['cell'].view(1, 3, 3))
        
        # Core: Concatenate precomputed edge table (apply node offset)
        src_l.append(data['edge_src'] + node_offset)
        dst_l.append(data['edge_dst'] + node_offset)
        shift_l.append(data['edge_shifts'])
        
        node_offset += num_nodes

    return (
        torch.cat(pos_l),
        torch.cat(A_l),
        torch.cat(b_idx_l),
        torch.cat(force_l),
        torch.cat(target_l),    # [Batch_Size]
        torch.cat(src_l),       # [Total_Edges]
        torch.cat(dst_l),       # [Total_Edges]
        torch.cat(shift_l),     # [Total_Edges, 3]
        torch.cat(cell_l)       # [Batch_Size, 3, 3]
    )


def on_the_fly_collate(batch_list):
    """
    Collate function for OnTheFlyDataset.
    
    Args:
        batch_list: List of samples from OnTheFlyDataset
        
    Returns:
        Batched data tuple or None if batch is empty
    """
    if not batch_list:
        return None
    
    # Initialize lists
    pos_l, A_l, force_l, target_l, cell_l, b_idx_l = [], [], [], [], [], []
    src_l, dst_l, shift_l = [], [], []
    
    num_nodes_accum = 0
    
    for i, item in enumerate(batch_list):
        num_atoms = item['pos'].shape[0]
        
        pos_l.append(item['pos'])
        A_l.append(item['A'])
        force_l.append(item['force'])
        target_l.append(item['target'])
        cell_l.append(item['cell'])
        b_idx_l.append(torch.full((num_atoms,), i, dtype=torch.long))
        
        # Concatenate graph data (add offset)
        src_l.append(item['edge_src'] + num_nodes_accum)
        dst_l.append(item['edge_dst'] + num_nodes_accum)
        shift_l.append(item['edge_shifts'])
        
        num_nodes_accum += num_atoms

    return (
        torch.cat(pos_l),
        torch.cat(A_l),
        torch.cat(b_idx_l),
        torch.cat(force_l),
        torch.stack(target_l),
        torch.cat(src_l),
        torch.cat(dst_l),
        torch.cat(shift_l),
        torch.stack(cell_l)
    )