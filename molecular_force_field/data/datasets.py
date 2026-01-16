"""Dataset classes for molecular modeling."""

import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset
from ase import Atoms
from ase.neighborlist import neighbor_list
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Tuple, Optional


def _split_cell_and_pbc_from_cell_df(cell_df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Support both legacy cell_{prefix}.h5 formats and the newer format with pbc_x/pbc_y/pbc_z columns.

    Returns:
        cells_all: np.ndarray shape [N, 3, 3] float64
        pbcs_all: np.ndarray shape [N, 3] bool, or None if not present
    """
    cols = list(cell_df.columns)
    has_named_cell = all(c in cols for c in ['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz'])
    has_pbc = all(c in cols for c in ['pbc_x', 'pbc_y', 'pbc_z'])

    if has_named_cell:
        cell_mat = cell_df[['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']].values.astype(np.float64)
    else:
        # legacy: assume first 9 columns represent the 3x3 cell (row-major)
        cell_mat = cell_df.iloc[:, :9].values.astype(np.float64)
    cells_all = cell_mat.reshape(-1, 3, 3)

    if has_pbc:
        pbcs_all = cell_df[['pbc_x', 'pbc_y', 'pbc_z']].values.astype(bool)
    else:
        pbcs_all = None

    return cells_all, pbcs_all


def compute_graph_worker(args):
    """
    Worker function for computing graph structure in parallel.
    
    Args:
        args: Tuple of (idx, block, target, cell, pbc, max_radius)
        
    Returns:
        Dictionary with precomputed graph data
    """
    idx, block, target, cell, pbc, max_radius = args
    
    # Extract coordinates and atom types (numpy)
    pos = block[:, 1:4].numpy()
    atom_types = block[:, 4].numpy()
    
    # Determine periodicity (prefer explicit pbc if provided)
    if pbc is None:
        is_periodic = (np.abs(cell).sum() > 1e-5)
        pbc_flags = [True, True, True] if is_periodic else [False, False, False]
    else:
        pbc_flags = [bool(pbc[0]), bool(pbc[1]), bool(pbc[2])]
        is_periodic = any(pbc_flags)

    if is_periodic:
        # Safety net: if lattice is missing/zero, use a large dummy cell to keep ASE happy
        current_cell = cell if (np.abs(cell).sum() > 1e-9) else (np.eye(3) * 100.0)
    else:
        pbc_flags = [False, False, False]
        current_cell = np.eye(3) * 100.0  # Virtual large box
        
    # ASE compute neighbors
    atoms = Atoms(numbers=atom_types, positions=pos, cell=current_cell, pbc=pbc_flags)
    i, j, S = neighbor_list('ijS', atoms, max_radius)
    
    # Return cache dictionary (all converted to Tensor)
    return {
        'read_tensor': block,
        'target_energy': target,
        'edge_src': torch.tensor(i, dtype=torch.long),
        'edge_dst': torch.tensor(j, dtype=torch.long),
        'edge_shifts': torch.tensor(S, dtype=torch.float64),
        'cell': torch.tensor(current_cell, dtype=torch.float64)
    }


class CustomDataset(Dataset):
    """Custom dataset with precomputed graph structures."""
    
    def __init__(self, read_file_path, energy_file_path, cell_file_path, max_radius=5.0, num_workers=10):
        """
        Initialize custom dataset.
        
        Args:
            read_file_path: Path to read HDF5 file
            energy_file_path: Path to energy HDF5 file
            cell_file_path: Path to cell HDF5 file
            max_radius: Maximum radius for neighbor search
            num_workers: Number of worker processes for preprocessing
        """
        print(f"Loading data from {read_file_path}...")
        self.max_radius = max_radius
        
        # 1. Read base HDF5 data
        read_data = pd.read_hdf(read_file_path)
        energy_df = pd.read_hdf(energy_file_path)
        cell_df = pd.read_hdf(cell_file_path)
        
        # 2. Process energy and Cell
        if 'RawEnergy' in energy_df.columns:
            targets = torch.tensor(energy_df['RawEnergy'].values, dtype=torch.float64)
        else:
            targets = torch.tensor(energy_df.iloc[:, 0].values, dtype=torch.float64)

        cells, pbcs = _split_cell_and_pbc_from_cell_df(cell_df)
        
        # 3. Vectorized block splitting
        values = read_data.values
        stop_value = 128128.0
        is_separator = (values == stop_value).any(axis=1)
        group_ids = is_separator.cumsum()
        clean_mask = ~is_separator
        
        clean_values = values[clean_mask]
        clean_group_ids = group_ids[clean_mask]
        _, unique_indices = np.unique(clean_group_ids, return_index=True)
        
        # Original block list
        blocks = [
            torch.tensor(block, dtype=torch.float64)
            for block in np.split(clean_values, unique_indices[1:])
        ]

        # 4. Multi-process precomputation of ASE neighbor lists
        print(f"Pre-calculating ASE neighbor lists using {num_workers} workers...")
        self.cache = []
        
        # Prepare parallel task parameters
        tasks = []
        for i in range(len(blocks)):
            pbc_i = pbcs[i] if pbcs is not None and i < len(pbcs) else None
            tasks.append((i, blocks[i], targets[i], cells[i], pbc_i, self.max_radius))
        
        # Use process pool for parallel computation
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(compute_graph_worker, tasks, chunksize=20),
                total=len(tasks),
                ascii=True,
                dynamic_ncols=True,
                desc="Pre-calculating ASE"
            ))
        
        self.cache = results
        print(f"Pre-calculation complete. Cached {len(self.cache)} structures in RAM.")
    
    def restore_energy(self, normalized_energy):
        """Restore energy units (if normalized)."""
        return normalized_energy
    
    def restore_force(self, normalized_force):
        """Restore force units (if normalized)."""
        return normalized_force

    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        # Return precomputed results directly from memory list
        d = self.cache[idx]
        return (
            d['read_tensor'],
            d['target_energy'],
            d['edge_src'],
            d['edge_dst'],
            d['edge_shifts'],
            d['cell']
        )


class H5Dataset(Dataset):
    """Dataset loading from preprocessed HDF5 files."""
    
    def __init__(self, prefix, data_dir='.'):
        """
        Initialize H5 dataset.
        
        Args:
            prefix: Prefix for HDF5 files (e.g., 'train' or 'val')
            data_dir: Directory containing the HDF5 files (default: current directory)
        """
        import os
        self.file_path = os.path.join(data_dir, f'processed_{prefix}.h5')
        self._h5_file = None
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"Preprocessed data file not found: {self.file_path}\n"
                f"Please run 'mff-preprocess' first to generate the required data files."
            )
        
        with h5py.File(self.file_path, 'r') as f:
            self.num_samples = len(f.keys())
    
    def restore_energy(self, normalized_energy):
        """Restore energy units (if normalized)."""
        return normalized_energy
    
    def restore_force(self, normalized_force):
        """Restore force units (if normalized)."""
        return normalized_force

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self._h5_file is None:  # Multi-process safe initialization
            self._h5_file = h5py.File(self.file_path, 'r')
        
        g = self._h5_file[f'sample_{idx}']
        return {
            'pos': torch.from_numpy(g['pos'][:]).double(),
            'A': torch.from_numpy(g['A'][:]).long(),
            'y': torch.from_numpy(np.array([g['y'][()]])).double(),
            'force': torch.from_numpy(g['force'][:]).double(),
            'edge_src': torch.from_numpy(g['edge_src'][:]).long(),
            'edge_dst': torch.from_numpy(g['edge_dst'][:]).long(),
            'edge_shifts': torch.from_numpy(g['edge_shifts'][:]).double(),
            'cell': torch.from_numpy(g['cell'][:]).double()
        }


class OnTheFlyDataset(Dataset):
    """Dataset that computes graph structure on-the-fly during loading."""
    
    def __init__(self, read_file_path, energy_file_path, cell_file_path, max_radius=5.0):
        """
        Initialize on-the-fly dataset.
        
        Args:
            read_file_path: Path to read HDF5 file
            energy_file_path: Path to energy HDF5 file
            cell_file_path: Path to cell HDF5 file
            max_radius: Maximum radius for neighbor search
        """
        print(f"Loading raw data indices from {read_file_path}...")
        self.max_radius = max_radius
        
        # 1. Only read raw data to memory (very fast, a few seconds)
        # No ASE computation
        self.read_data = pd.read_hdf(read_file_path)
        self.energy_df = pd.read_hdf(energy_file_path)
        self.cell_df = pd.read_hdf(cell_file_path)
        
        # 2. Prepare data indices
        if 'RawEnergy' in self.energy_df.columns:
            self.targets = self.energy_df['RawEnergy'].values
        else:
            self.targets = self.energy_df.iloc[:, 0].values

        self.cells, self.pbcs = _split_cell_and_pbc_from_cell_df(self.cell_df)
        
        # 3. Vectorized block splitting (this step is fast)
        values = self.read_data.values
        stop_value = 128128.0
        is_separator = (values == stop_value).any(axis=1)
        group_ids = is_separator.cumsum()
        clean_mask = ~is_separator
        _, unique_indices = np.unique(group_ids[clean_mask], return_index=True)
        self.blocks = np.split(values[clean_mask], unique_indices[1:])

    def restore_energy(self, x):
        return x
    
    def restore_force(self, x):
        return x

    def __len__(self):
        return len(self.blocks)
    
    def __getitem__(self, idx):
        # Core of on-the-fly computation
        
        # 1. Get Numpy data
        block = self.blocks[idx]
        pos = block[:, 1:4]
        atom_types = block[:, 4]
        forces = block[:, 5:8]
        target = self.targets[idx]
        cell = self.cells[idx]
        
        # 2. Determine PBC (prefer explicit pbc flags if present)
        pbc_i = None
        if getattr(self, "pbcs", None) is not None and idx < len(self.pbcs):
            pbc_i = self.pbcs[idx]

        if pbc_i is None:
            is_periodic = (np.abs(cell).sum() > 1e-5)
            pbc_flags = [True, True, True] if is_periodic else [False, False, False]
        else:
            pbc_flags = [bool(pbc_i[0]), bool(pbc_i[1]), bool(pbc_i[2])]
            is_periodic = any(pbc_flags)

        if is_periodic:
            current_cell = cell if (np.abs(cell).sum() > 1e-9) else (np.eye(3) * 100.0)
        else:
            pbc_flags = [False, False, False]
            current_cell = np.eye(3) * 100.0
        
        # 3. Real-time ASE neighbor computation
        # This step consumes CPU, but is masked by DataLoader multiprocessing
        atoms = Atoms(numbers=atom_types, positions=pos, cell=current_cell, pbc=pbc_flags)
        i, j, S = neighbor_list('ijS', atoms, self.max_radius)
        
        # 4. Convert to Tensor and return
        return {
            'pos': torch.tensor(pos, dtype=torch.float64),
            'A': torch.tensor(atom_types, dtype=torch.float64),
            'force': torch.tensor(forces, dtype=torch.float64),
            'target': torch.tensor(target, dtype=torch.float64),
            'edge_src': torch.tensor(i, dtype=torch.long),
            'edge_dst': torch.tensor(j, dtype=torch.long),
            'edge_shifts': torch.tensor(S, dtype=torch.float64),
            'cell': torch.tensor(current_cell, dtype=torch.float64)
        }