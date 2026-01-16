"""ASE Calculator wrapper for E3NN models."""

import torch
from ase.calculators.calculator import Calculator, all_changes

from molecular_force_field.utils.graph_utils import radius_graph_pbc_gpu
from molecular_force_field.utils.tensor_utils import map_tensor_values


class MyE3NNCalculator(Calculator):
    """ASE Calculator wrapper for E3NN-based molecular force field models."""
    
    implemented_properties = ['energy', 'forces']

    def __init__(self, model, atomic_energies_dict, device, max_radius, **kwargs):
        """
        Initialize calculator.
        
        Args:
            model: E3NN model (E3_TransformerLayer_multi)
            atomic_energies_dict: Dictionary mapping atomic numbers to reference energies
            device: Device to use
            max_radius: Maximum radius for neighbor search
            **kwargs: Additional arguments for ASE Calculator
        """
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.max_radius = max_radius
        self.keys = torch.tensor(list(atomic_energies_dict.keys()), device=device)
        self.values = torch.tensor(list(atomic_energies_dict.values()), device=device)
        
        # Freeze model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        """
        Calculate energy and forces for given atoms.
        
        Args:
            atoms: ASE Atoms object
            properties: List of properties to calculate
            system_changes: System changes to check
        """
        super().calculate(atoms, properties, system_changes)
        
        # 1. Prepare data (CPU -> GPU)
        pos = torch.tensor(self.atoms.get_positions(), dtype=torch.float64, device=self.device)
        A = torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.float64, device=self.device)
        
        # Handle Cell (must be 3x3 tensor)
        # If non-periodic, use a large box to prevent computation errors
        if any(self.atoms.pbc):
            cell = torch.tensor(
                self.atoms.get_cell().array,
                dtype=torch.float64,
                device=self.device
            ).unsqueeze(0)
        else:
            cell = torch.eye(3, dtype=torch.float64, device=self.device).unsqueeze(0) * 100.0

        # 2. GPU neighbor computation
        # Replace original i, j, S = neighbor_list(...)
        edge_src, edge_dst, edge_shifts = radius_graph_pbc_gpu(pos, self.max_radius, cell)
        
        # 3. Inference
        pos.requires_grad_(True)
        batch_idx = torch.zeros(len(pos), dtype=torch.long, device=self.device)
        
        mapped_A = map_tensor_values(A, self.keys, self.values)
        E_offset = mapped_A.sum()
        
        # Call model
        atom_energies = self.model(
            pos, A, batch_idx,
            edge_src, edge_dst, edge_shifts, cell
        )
        E_total = atom_energies.sum() + E_offset
        
        grads = torch.autograd.grad(E_total, pos)[0]
        
        # 4. Output
        self.results['energy'] = E_total.item()
        self.results['forces'] = -grads.detach().cpu().numpy()