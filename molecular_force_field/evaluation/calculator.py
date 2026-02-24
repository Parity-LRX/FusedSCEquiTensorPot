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


class DDPCalculator(Calculator):
    """
    ASE Calculator 的 DDP 版本：仅在 rank 0 与 ASE 交互，每次 calculate() 时通过
    run_one_ddp_inference_from_ase_atoms 与其它 rank 协同完成推理（多卡分摊大结构）。
    需用 torchrun 启动，且非 rank 0 进程需在别处运行 worker 循环。
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, atomic_energies_dict, device, max_radius, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.model = model
        self.device = device
        self.max_radius = max_radius
        self.atomic_energies_dict = atomic_energies_dict or {}
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        from molecular_force_field.cli.inference_ddp import run_one_ddp_inference_from_ase_atoms
        dtype = next(self.model.parameters()).dtype
        energy, forces = run_one_ddp_inference_from_ase_atoms(
            self.atoms,
            self.model,
            self.max_radius,
            self.device,
            dtype,
            return_forces=True,
            atomic_energies_dict=self.atomic_energies_dict,
        )
        if energy is not None and forces is not None:
            self.results["energy"] = energy
            self.results["forces"] = forces.cpu().numpy()