"""Evaluation module for model assessment."""

import logging
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter

from molecular_force_field.utils.tensor_utils import map_tensor_values


class Evaluator:
    """Evaluator for static model evaluation."""
    
    def __init__(self, model, dataset, device, atomic_energy_keys=None,
                 atomic_energy_values=None, force_shift_value=1):
        """
        Initialize evaluator.
        
        Args:
            model: E3NN model (E3_TransformerLayer_multi)
            dataset: Dataset for evaluation
            device: Device to use
            atomic_energy_keys: Atomic energy keys tensor
            atomic_energy_values: Atomic energy values tensor
            force_shift_value: Force shift value
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.force_shift_value = force_shift_value
        
        if atomic_energy_keys is None:
            self.keys = torch.tensor([1, 6, 7, 8]).to(device)
        else:
            self.keys = atomic_energy_keys.to(device)
            
        if atomic_energy_values is None:
            self.values = torch.tensor([
                -430.53299511, -821.03326787, -1488.18856918, -2044.3509823
            ]).to(device)
        else:
            self.values = atomic_energy_values.to(device)
    
    def evaluate(self, data_loader, output_prefix='test'):
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation
            output_prefix: Prefix for output files
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        val_E_preds = []
        val_E_targets = []
        val_E_avg_preds = []
        val_E_avg_targets = []
        val_F_preds = []
        val_F_targets = []
        val_total_loss_list = []
        
        # Use SmoothL1Loss consistent with run_eval.py
        criterion = torch.nn.SmoothL1Loss(beta=0.5)
        
        for batch_idx_loader, batch in enumerate(data_loader):
            if batch is None:
                continue
            
            # Unpack
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell) = batch

            # Move to GPU
            pos = pos.to(self.device)
            A = A.to(self.device)
            batch_idx = batch_idx.to(self.device)
            force_ref = force_ref.to(self.device)
            target_energies = target_energies.to(self.device)
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_shifts = edge_shifts.to(self.device)
            cell = cell.to(self.device)
            
            pos.requires_grad = True
            
            mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
            # E_offset_val shape: [Batch_Size]
            E_offset_val = scatter(mapped_A, batch_idx, dim=0, reduce='sum')
            
            # Forward
            E_per_atom = self.model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            # E_conv_val shape: [Batch_Size, 1]
            E_conv_val = scatter(E_per_atom, batch_idx, dim=0, reduce='sum')
            
            # E_mean_val shape: [Batch_Size]
            E_mean_val = E_conv_val.view(-1) + E_offset_val
            
            # Backward (Force)
            grads = torch.autograd.grad(
                E_mean_val.sum(), pos, create_graph=False, retain_graph=False
            )[0]
            f_pred = -grads
            
            # Restore units
            fx_pred = self.dataset.restore_force(f_pred[:, 0])
            fy_pred = self.dataset.restore_force(f_pred[:, 1])
            fz_pred = self.dataset.restore_force(f_pred[:, 2])
            f_pred_final = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)  # [N_atoms, 3]
            
            f_ref_final = force_ref * self.force_shift_value
            
            # Collect data (detach and move to CPU)
            with torch.no_grad():
                # Energy [Batch_Size]
                val_E_preds.append(E_mean_val.detach().cpu())
                val_E_targets.append(target_energies.detach().cpu())
                
                # Average atomic energy [Batch_Size]
                num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
                val_E_avg_preds.append((E_mean_val / num_atoms_per_mol).detach().cpu())
                val_E_avg_targets.append((target_energies / num_atoms_per_mol).detach().cpu())
                
                # Force [N_atoms, 3] (Note: keep structure to save as CSV)
                val_F_preds.append(f_pred_final.detach().cpu())
                val_F_targets.append(f_ref_final.detach().cpu())

                # Loss (only for Log)
                batch_e_loss = criterion(E_mean_val, target_energies)
                batch_f_loss = criterion(f_pred_final.view(-1), f_ref_final.view(-1))
                val_total_loss_list.append((batch_e_loss + batch_f_loss).item())

            # Fix: take [0] when printing log to avoid error when batch_size > 1
            logging.info(f"Batch Sample -> Pred: {self.dataset.restore_energy(E_mean_val[0].item()):.4f} , True: {self.dataset.restore_energy(target_energies[0].item()):.4f}")
        
        # Process data after loop ends
        # 1. Concatenate
        all_E_preds = torch.cat(val_E_preds)  # [Total_Mols]
        all_E_targets = torch.cat(val_E_targets)  # [Total_Mols]
        all_E_avg_preds = torch.cat(val_E_avg_preds)
        all_E_avg_targets = torch.cat(val_E_avg_targets)
        
        all_F_preds = torch.cat(val_F_preds)  # [Total_Atoms, 3]
        all_F_targets = torch.cat(val_F_targets)  # [Total_Atoms, 3]
        
        # 2. Calculate global metrics (Note: Force needs view(-1) here)
        # Energy
        val_energy_rmse = torch.sqrt(F.mse_loss(all_E_preds, all_E_targets)).item()
        val_energy_rmse = self.dataset.restore_energy(val_energy_rmse)
        
        val_energy_rmse_avg = torch.sqrt(F.mse_loss(all_E_avg_preds, all_E_avg_targets)).item()
        val_energy_rmse_avg = self.dataset.restore_energy(val_energy_rmse_avg)
        
        val_energy_mae_avg = F.l1_loss(all_E_avg_preds, all_E_avg_targets).item()
        val_energy_mae_avg = self.dataset.restore_energy(val_energy_mae_avg)
        
        val_energy_loss = F.mse_loss(all_E_avg_preds, all_E_avg_targets).item()

        # Force (Flatten to calculate metrics)
        val_force_rmse = torch.sqrt(F.mse_loss(all_F_preds.view(-1), all_F_targets.view(-1))).item()
        val_force_mae = F.l1_loss(all_F_preds.view(-1), all_F_targets.view(-1)).item()

        logging.info(f"""
                        "Energy Loss Val (MSE)": {val_energy_loss},
                        "Energy RMSE Val": {val_energy_rmse},
                        "Energy RMSE avg Val": {val_energy_rmse_avg},
                        "Energy MAE avg Val": {val_energy_mae_avg},
                        "Force MAE Val": {val_force_mae},
                        "Force RMSE Val":  {val_force_rmse}
                        """)
        
        # Save results
        loss_out = [{
            "Energy_RMSE_test": val_energy_rmse,
            "Energy_RMSE_avg_test": val_energy_rmse_avg,
            "Force_RMSE_test": val_force_rmse,
        }]
        
        loss_out_df = pd.DataFrame(loss_out)
        loss_out_df.to_csv(f"{output_prefix}_loss.csv", index=False)
        
        # Save energy (one molecule per row)
        df_energy_val = pd.DataFrame({
            "Target_Energy": all_E_targets.numpy(),
            "Predicted_Energy": all_E_preds.numpy(),
            "Delta": (all_E_preds - all_E_targets).numpy()
        })
        energy_save_path = f"{output_prefix}_energy.csv"
        df_energy_val.to_csv(energy_save_path, index=False)
        
        # Save force (one atom per row, includes x, y, z components)
        # Use numpy slicing to separate components
        f_pred_np = all_F_preds.numpy()
        f_true_np = all_F_targets.numpy()
        
        df_force_val = pd.DataFrame({
            "Fx_True": f_true_np[:, 0], "Fy_True": f_true_np[:, 1], "Fz_True": f_true_np[:, 2],
            "Fx_Pred": f_pred_np[:, 0], "Fy_Pred": f_pred_np[:, 1], "Fz_Pred": f_pred_np[:, 2]
        })
        force_save_path = f"{output_prefix}_force.csv"
        df_force_val.to_csv(force_save_path, index=False)
        
        logging.info(f"Saved test details to {energy_save_path} and {force_save_path}")
        
        return {
            'energy_rmse': val_energy_rmse,
            'energy_rmse_avg': val_energy_rmse_avg,
            'energy_mae_avg': val_energy_mae_avg,
            'energy_loss': val_energy_loss,
            'force_rmse': val_force_rmse,
            'force_mae': val_force_mae,
        }