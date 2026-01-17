"""Training module for molecular force field models."""

import os
import time
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_scatter import scatter
from torch.optim.lr_scheduler import SequentialLR, LambdaLR, StepLR
import copy

from molecular_force_field.models import E3_TransformerLayer_multi, MainNet
from molecular_force_field.models.losses import RMSELoss
from molecular_force_field.utils.tensor_utils import map_tensor_values
from molecular_force_field.utils.config import ModelConfig


def log_gradient_statistics(networks, batch_idx, logger):
    """Log gradient statistics for all networks."""
    for net in networks:
        for name, param in net.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_norm = torch.norm(grad).item()
                grad_max = grad.max().item()
                grad_min = grad.min().item()
                grad_mean = grad.mean().item()
                
                logger.debug(
                    f"Gradient/Batch_{batch_idx}/Network_{net.__class__.__name__}/Layer_{name}:\n"
                    f"  Norm: {grad_norm:.6e}\n"
                    f"  Max: {grad_max:.6e}\n"
                    f"  Min: {grad_min:.6e}\n"
                    f"  Mean: {grad_mean:.6e}\n"
                )
            else:
                logger.debug(
                    f"Gradient/Batch_{batch_idx}/Network_{net.__class__.__name__}/Layer_{name}:\n"
                    "  No gradients"
                )


class Trainer:
    """Trainer class for molecular force field models."""
    
    def __init__(
        self,
        model,
        e3trans,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        device,
        config=None,
        learning_rate=2e-4,
        min_learning_rate=1e-5,
        initial_learning_rate_for_weight=0.1,
        epoch_numbers=1000,
        patience=20,
        dump_frequency=250,
        energy_log_frequency=10,
        vhat_clamp_interval=2000,
        warmup_batches=1000,
        patience_opim=1000,
        gamma_value=0.98,
        max_vhat_growth_factor=5,
        max_norm_value=0.5,
        gradient_log_interval=100,
        update_param=1000,
        force_shift_value=1,
        a=1,
        b=10,
        weight_a_growth=1.01,
        weight_b_decay=0.99,
        a_min=None,
        a_max=None,
        b_min=None,
        b_max=None,
        swa_start_epoch=None,
        swa_a=None,
        swa_b=None,
        ema_start_epoch=None,
        ema_decay=0.999,
        use_ema_for_validation=False,
        save_ema_model=False,
        checkpoint_path='combined_model.pth',
        atomic_energy_keys=None,
        atomic_energy_values=None,
        distributed=False,
        rank=0,
        world_size=1,
        train_sampler=None,
        save_val_csv=False,
        use_checkpoint_loss_weights=True,
    ):
        """
        Initialize trainer.
        
        Args:
            model: MainNet model
            e3trans: E3_TransformerLayer_multi model
            train_loader: Training data loader
            val_loader: Validation data loader
            train_dataset: Training dataset
            val_dataset: Validation dataset
            device: Device to use
            config: ModelConfig object (optional)
            learning_rate: Initial learning rate
            min_learning_rate: Minimum learning rate
            initial_learning_rate_for_weight: Initial learning rate ratio for weight
            epoch_numbers: Number of epochs
            patience: Early stopping patience
            dump_frequency: Frequency to dump validation results
            vhat_clamp_interval: Interval to clamp v_hat
            warmup_batches: Number of warmup batches
            patience_opim: Patience for optimizer
            gamma_value: Learning rate decay factor
            max_vhat_growth_factor: Maximum v_hat growth factor
            max_norm_value: Maximum gradient norm
            gradient_log_interval: Interval to log gradients
            update_param: Interval to update loss weights
            force_shift_value: Force shift value
            a: Energy loss weight
            b: Force loss weight
            swa_start_epoch: Epoch to start SWA (Stochastic Weight Averaging) for loss weights.
                After this epoch, a and b will be set to swa_a and swa_b directly.
            swa_a: Energy weight a after SWA starts
            swa_b: Force weight b after SWA starts
            ema_start_epoch: Epoch to start EMA (Exponential Moving Average) for e3trans weights
            ema_decay: EMA decay factor in (0, 1) (default: 0.999)
            use_ema_for_validation: Whether to use EMA weights for validation forward
            save_ema_model: Whether to save EMA weights into checkpoints
            checkpoint_path: Path to save checkpoints
            atomic_energy_keys: Atomic energy keys tensor
            atomic_energy_values: Atomic energy values tensor
            distributed: Whether to use distributed training
            rank: Process rank in distributed training
            world_size: Total number of processes
            train_sampler: Distributed sampler for training data
            save_val_csv: Whether to save validation energy and force predictions to CSV files
                (default: True). If False, only metrics are logged but CSV files are not saved.
            use_checkpoint_loss_weights: Whether to use loss weights (a, b) from checkpoint when loading.
                If True (default), use checkpoint values. If False, use the values passed to __init__.
        """
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.train_sampler = train_sampler
        self.is_main_process = (rank == 0)
        
        self.model = model
        self.e3trans = e3trans
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.config = config
        self.learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.initial_learning_rate_for_weight = initial_learning_rate_for_weight
        self.epoch_numbers = epoch_numbers
        self.patience = patience
        self.dump_frequency = dump_frequency
        self.energy_log_frequency = energy_log_frequency
        self.vhat_clamp_interval = vhat_clamp_interval
        self.warmup_batches = warmup_batches
        self.patience_opim = patience_opim
        self.gamma_value = gamma_value
        self.max_vhat_growth_factor = max_vhat_growth_factor
        self.max_norm_value = max_norm_value
        self.gradient_log_interval = gradient_log_interval
        self.update_param = update_param
        self.force_shift_value = force_shift_value
        self.a = a
        self.b = b
        self.weight_a_growth = weight_a_growth
        self.weight_b_decay = weight_b_decay
        self.a_min = a_min
        self.a_max = a_max
        self.b_min = b_min
        self.b_max = b_max
        self.swa_start_epoch = swa_start_epoch
        self.swa_a = swa_a
        self.swa_b = swa_b
        self.swa_applied = False  # Track if SWA has been applied
        self.ema_start_epoch = ema_start_epoch
        self.ema_decay = ema_decay
        self.use_ema_for_validation = use_ema_for_validation
        self.save_ema_model = save_ema_model
        self.ema_enabled = False
        self.e3trans_ema = None  # will be initialized when EMA starts
        self.checkpoint_path = checkpoint_path
        self.save_val_csv = save_val_csv  # Control whether to save val_energy.csv and val_force.csv
        self.use_checkpoint_loss_weights = use_checkpoint_loss_weights  # Whether to use checkpoint a/b
        
        # Initialize loss tracking for CSV
        self.loss_csv_path = 'loss.csv'
        self.loss_records = []
        
        # Initialize training state (will be updated if loading from checkpoint)
        self.start_epoch = 1
        self.batch_count = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.resumed_from_checkpoint = False
        
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
        
        # Loss functions
        self.criterion = nn.SmoothL1Loss(beta=0.5)
        self.criterion_2 = RMSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(e3trans.parameters()) + list(model.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-6,
            amsgrad=True
        )
        
        # Learning rate scheduler
        milestones = [warmup_batches]
        
        def warmup_lambda(current_step):
            progress = min(current_step / warmup_batches, 1.0)
            initial_ratio = initial_learning_rate_for_weight
            return initial_ratio + (1 - initial_ratio) * progress
        
        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
        step_scheduler = StepLR(
            self.optimizer,
            step_size=patience_opim,
            gamma=gamma_value
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=milestones
        )
        
        # Wrap models with DDP if distributed training is enabled
        if self.distributed:
            # Enable find_unused_parameters for models with conditional computation paths
            # (e.g., MainNet is not used in e3trans forward pass)
            self.e3trans = DDP(
                self.e3trans,
                device_ids=[device.index] if device.type == 'cuda' else None,
                output_device=device.index if device.type == 'cuda' else None,
                find_unused_parameters=True,  # Required: some parameters don't participate in every forward pass
                broadcast_buffers=True
            )
            # Note: model (MainNet) is not used in forward pass, so we don't wrap it
            # But if it's used elsewhere, wrap it too:
            # self.model = DDP(self.model, ...)
        
        # These will be overwritten if loading from checkpoint
        # Kept here for clarity of initialization sequence
        self.loss_out = []
        
        # Log training configuration (only on main process)
        if self.is_main_process:
            logging.info("=" * 60)
            logging.info("Training Configuration:")
            logging.info(f"  Epochs: {self.epoch_numbers}")
            logging.info(f"  Learning Rate: {self.learning_rate}")
            logging.info(f"  Min Learning Rate: {self.min_learning_rate}")
            logging.info(f"  Training Samples: {len(self.train_dataset)}")
            logging.info(f"  Validation Samples: {len(self.val_dataset)}")
            logging.info(f"  Energy Loss Weight (a): {self.a}")
            logging.info(f"  Force Loss Weight (b): {self.b}")
            if self.swa_start_epoch is not None:
                logging.info(f"  SWA enabled: Will switch to a={self.swa_a}, b={self.swa_b} at epoch {self.swa_start_epoch}")
            if self.ema_start_epoch is not None:
                logging.info(f"  EMA enabled: Will start at epoch {self.ema_start_epoch} with decay={self.ema_decay}")
                if self.use_ema_for_validation:
                    logging.info("    Using EMA model for validation")
                if self.save_ema_model:
                    logging.info("    Will save EMA model in checkpoint")
            logging.info(f"  Validation Frequency: every {self.dump_frequency} batches")
            logging.info(f"  Energy Log Frequency: every {self.energy_log_frequency} batches")
            logging.info(f"  Early Stopping Patience: {self.patience} epochs")
            if self.distributed:
                logging.info(f"  Distributed Training: {self.world_size} GPUs")
            logging.info("=" * 60)
        
        # Load checkpoint if exists
        if os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle loading into DDP model
        if self.distributed:
            self.e3trans.module.load_state_dict(checkpoint['e3trans_state_dict'])
        else:
            self.e3trans.load_state_dict(checkpoint['e3trans_state_dict'])
        
        # Load loss weights (only if use_checkpoint_loss_weights is True)
        if self.use_checkpoint_loss_weights:
            if 'a' in checkpoint:
                self.a = checkpoint['a']
            if 'b' in checkpoint:
                self.b = checkpoint['b']
            if self.is_main_process:
                logging.info(f"  Using loss weights from checkpoint: a={self.a:.4f}, b={self.b:.4f}")
        else:
            if self.is_main_process:
                logging.info(f"  Using new loss weights (ignoring checkpoint): a={self.a:.4f}, b={self.b:.4f}")
        if 'swa_applied' in checkpoint:
            self.swa_applied = checkpoint['swa_applied']
        if 'ema_enabled' in checkpoint:
            self.ema_enabled = checkpoint['ema_enabled']
        if 'e3trans_ema_state_dict' in checkpoint:
            # Lazily initialize EMA model
            if self.e3trans_ema is None:
                base = self.e3trans.module if self.distributed else self.e3trans
                self.e3trans_ema = copy.deepcopy(base).to(self.device)
                self.e3trans_ema.eval()
            self.e3trans_ema.load_state_dict(checkpoint['e3trans_ema_state_dict'])
        
        # Load training state for restart
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        if 'batch_count' in checkpoint:
            self.batch_count = checkpoint['batch_count']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'patience_counter' in checkpoint:
            self.patience_counter = checkpoint['patience_counter']
        
        self.resumed_from_checkpoint = True
        
        if self.is_main_process:
            logging.info("=" * 80)
            logging.info("Checkpoint Loaded Successfully!")
            logging.info(f"  Resuming from epoch: {self.start_epoch}")
            logging.info(f"  Batch count: {self.batch_count}")
            if self.use_checkpoint_loss_weights:
                logging.info(f"  Loss weights (from checkpoint): a={self.a:.4f}, b={self.b:.4f}")
            else:
                logging.info(f"  Loss weights (new, ignoring checkpoint): a={self.a:.4f}, b={self.b:.4f}")
            logging.info(f"  Best validation loss: {self.best_val_loss:.6f}")
            logging.info(f"  Early stopping patience counter: {self.patience_counter}/{self.patience}")
            logging.info("=" * 80)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        # Set epoch for distributed sampler
        if self.distributed and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        
        self.model.train()
        self.e3trans.train()  # DDP model's train() will call internal module's train()
        
        total_batch_loss = []
        total_batch_energy_loss = []
        total_batch_energy_rmse = []
        total_batch_energy_rmse_avg = []
        total_batch_force_loss = []
        total_batch_force_rmse = []
        
        all_nets = [self.e3trans, self.model]
        all_parameters = [param for net in all_nets for param in net.parameters()]
        
        # Log epoch start
        total_batches = len(self.train_loader)
        epoch_start_time = time.time()
        # Log epoch start only to file
        if self.is_main_process:
            logging.info(f"Epoch {epoch} started - Total batches: {total_batches}")
        
        batch_times = []
        
        for batch_idx_loader, batch_data in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            if batch_data is None:
                continue
            
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell) = batch_data
            
            # Move to device
            pos = pos.to(self.device)
            A = A.to(self.device)
            batch_idx = batch_idx.to(self.device)
            force_ref = force_ref.to(self.device)
            target_energies = target_energies.to(self.device)
            edge_src = edge_src.to(self.device)
            edge_dst = edge_dst.to(self.device)
            edge_shifts = edge_shifts.to(self.device)
            cell = cell.to(self.device)
            
            pos.requires_grad_(True)
            
            self.batch_count += 1

            # Initialize EMA when reaching ema_start_epoch
            if self.ema_start_epoch is not None and epoch >= self.ema_start_epoch and not self.ema_enabled:
                base = self.e3trans.module if self.distributed else self.e3trans
                self.e3trans_ema = copy.deepcopy(base).to(self.device)
                self.e3trans_ema.eval()
                for p in self.e3trans_ema.parameters():
                    p.requires_grad_(False)
                self.ema_enabled = True
                if self.is_main_process:
                    logging.info(f"EMA initialized at epoch {epoch} with decay={self.ema_decay}")
            
            # Update loss weights: either SWA (direct switch) or continuous growth/decay
            if self.swa_start_epoch is not None and epoch >= self.swa_start_epoch and not self.swa_applied:
                # Apply SWA: directly set a and b to SWA values
                self.a = self.swa_a
                self.b = self.swa_b
                self.swa_applied = True

                # IMPORTANT: early-stopping metric is a*val_energy_loss + b*val_force_loss.
                # When we switch (a, b), the metric scale/priority changes, so we should
                # reset early-stopping state to avoid premature stopping.
                self.patience_counter = 0
                self.best_val_loss = float('inf')

                if self.is_main_process:
                    logging.info(
                        f"SWA applied at epoch {epoch}: a={self.a:.4f}, b={self.b:.4f}. "
                        f"Early-stopping state reset (best_val_loss=inf, patience_counter=0)."
                    )
            elif self.swa_start_epoch is None or epoch < self.swa_start_epoch:
                # Continuous growth/decay (only if SWA not applied yet)
                if self.batch_count % self.update_param == 0:
                    self.a *= self.weight_a_growth
                    self.b *= self.weight_b_decay
                    # Optional clamps (enabled only if user provides bounds)
                    if self.a_min is not None:
                        self.a = max(self.a, self.a_min)
                    if self.a_max is not None:
                        self.a = min(self.a, self.a_max)
                    if self.b_min is not None:
                        self.b = max(self.b, self.b_min)
                    if self.b_max is not None:
                        self.b = min(self.b, self.b_max)
            
            self.optimizer.zero_grad()
            
            # Map atomic energies
            mapped_A = map_tensor_values(A, self.keys, self.values).to(self.device)
            E_offset_mol = scatter(mapped_A, batch_idx, dim=0, reduce='sum')
            
            # Forward pass
            E_per_atom = self.e3trans(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            E_conv_mol = scatter(E_per_atom, batch_idx, dim=0, reduce='sum').squeeze(-1)
            E_mean = E_conv_mol + E_offset_mol
            
            # Compute forces
            grads = torch.autograd.grad(
                E_mean.sum(),
                pos,
                create_graph=True,
                retain_graph=True
            )[0]
            
            fx_pred = self.train_dataset.restore_force(-grads[:, 0])
            fy_pred = self.train_dataset.restore_force(-grads[:, 1])
            fz_pred = self.train_dataset.restore_force(-grads[:, 2])
            f_pred = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)
            
            force_ref_scaled = force_ref * self.force_shift_value
            force_loss = self.criterion(f_pred.view(-1), force_ref_scaled.view(-1))
            
            with torch.no_grad():
                force_rmse = torch.sqrt(self.criterion_2(f_pred.view(-1), force_ref_scaled.view(-1)))
            
            # Energy loss
            num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
            E_avg_pred = E_mean / num_atoms_per_mol
            target_energy_avg = target_energies / num_atoms_per_mol
            energy_loss = self.criterion(E_avg_pred, target_energy_avg)
            
            # Log energy predictions every N batches (only to log file, not console, only on main process)
            if self.is_main_process and self.batch_count % self.energy_log_frequency == 0:
                # Get unique molecule indices in this batch
                unique_mol_indices = torch.unique(batch_idx).cpu().numpy()
                num_molecules = len(unique_mol_indices)
                
                # Log summary for all molecules in batch
                E_pred_np = E_mean.detach().cpu().numpy()
                E_target_np = target_energies.detach().cpu().numpy()
                E_avg_pred_np = E_avg_pred.detach().cpu().numpy()
                E_avg_target_np = target_energy_avg.detach().cpu().numpy()
                
                logging.debug(f"Batch {self.batch_count} - Energy Predictions ({num_molecules} molecules):")
                for i, mol_idx in enumerate(unique_mol_indices):
                    mol_mask = (batch_idx == mol_idx)
                    num_atoms = mol_mask.sum().item()
                    e_pred_val = E_pred_np[mol_idx]
                    e_target_val = E_target_np[mol_idx]
                    e_avg_pred_val = E_avg_pred_np[mol_idx]
                    e_avg_target_val = E_avg_target_np[mol_idx]
                    error = e_pred_val - e_target_val
                    error_avg = e_avg_pred_val - e_avg_target_val
                    
                    logging.debug(
                        f"  Mol {i+1} (idx={mol_idx}, {num_atoms} atoms): "
                        f"E_pred={e_pred_val:12.6f} eV, E_target={e_target_val:12.6f} eV, "
                        f"Error={error:8.4f} eV | "
                        f"E_avg_pred={e_avg_pred_val:10.6f} eV/atom, E_avg_target={e_avg_target_val:10.6f} eV/atom, "
                        f"Error={error_avg:8.4f} eV/atom"
                    )
            
            with torch.no_grad():
                energy_rmse = torch.sqrt(self.criterion_2(E_mean, target_energies))
                energy_rmse_avg = torch.sqrt(self.criterion_2(E_avg_pred, target_energy_avg))
            
            # Total loss
            total_loss = self.a * energy_loss + self.b * force_loss
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_parameters, max_norm=self.max_norm_value, norm_type=2.0)
            
            if self.batch_count % self.gradient_log_interval == 0:
                log_gradient_statistics(all_nets, self.batch_count, logging)
            
            self.optimizer.step()

            # Update EMA after optimizer step
            if self.ema_enabled and self.e3trans_ema is not None:
                with torch.no_grad():
                    current = self.e3trans.module if self.distributed else self.e3trans
                    for ema_param, cur_param in zip(self.e3trans_ema.parameters(), current.parameters()):
                        ema_param.data.mul_(self.ema_decay).add_(cur_param.data, alpha=1.0 - self.ema_decay)
            
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] < self.min_learning_rate:
                    param_group['lr'] = self.min_learning_rate
            
            if self.batch_count % self.vhat_clamp_interval == 0:
                for param_group in self.optimizer.param_groups:
                    for param in param_group['params']:
                        state = self.optimizer.state[param]
                        if 'v_hat' in state:
                            current_max = state['v_hat'].max().item()
                            state['v_hat'].clamp_(max=current_max * self.max_vhat_growth_factor)
            
            self.scheduler.step()
            
            total_batch_loss.append(total_loss.item())
            total_batch_energy_loss.append(energy_loss.item())
            total_batch_energy_rmse.append(energy_rmse.item())
            total_batch_energy_rmse_avg.append(energy_rmse_avg.item())
            total_batch_force_loss.append(force_loss.item())
            total_batch_force_rmse.append(force_rmse.item())
            
            # Record batch time
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            batch_times.append(batch_time)
            
            # Log training progress only to file, not console (only on main process)
            if self.is_main_process:
                progress_interval = max(1, min(10, total_batches // 20))  # Show at least 20 updates per epoch
                if (batch_idx_loader + 1) % progress_interval == 0 or (batch_idx_loader + 1) == total_batches:
                    current_energy_loss = np.mean(total_batch_energy_loss[-10:]) if len(total_batch_energy_loss) >= 10 else total_batch_energy_loss[-1] if total_batch_energy_loss else 0
                    current_force_loss = np.mean(total_batch_force_loss[-10:]) if len(total_batch_force_loss) >= 10 else total_batch_force_loss[-1] if total_batch_force_loss else 0
                    current_lr = self.optimizer.param_groups[0]['lr']
                    avg_batch_time = np.mean(batch_times[-10:]) if len(batch_times) >= 10 else np.mean(batch_times)
                    elapsed_time = batch_end_time - epoch_start_time
                    # Estimate remaining time
                    remaining_batches = total_batches - (batch_idx_loader + 1)
                    eta = remaining_batches * avg_batch_time
                    # Log to file only
                    logging.info(
                        f"Epoch {epoch} | Batch {batch_idx_loader + 1}/{total_batches} | "
                        f"Energy Loss: {current_energy_loss:.6f} | Force Loss: {current_force_loss:.6f} | "
                        f"LR: {current_lr:.2e} | Batch Count: {self.batch_count} | "
                        f"Batch Time: {batch_time:.3f}s | Avg: {avg_batch_time:.3f}s | "
                        f"Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s"
                    )
            
            if self.batch_count % self.dump_frequency == 0:
                # Calculate average training loss up to this point for logging
                current_train_energy_loss = np.mean(total_batch_energy_loss) if total_batch_energy_loss else 0
                current_train_force_loss = np.mean(total_batch_force_loss) if total_batch_force_loss else 0
                current_train_total_loss = np.mean(total_batch_loss) if total_batch_loss else 0
                current_train_energy_rmse = np.mean(total_batch_energy_rmse) if total_batch_energy_rmse else 0
                current_train_force_rmse = np.mean(total_batch_force_rmse) if total_batch_force_rmse else 0
                
                self.validate(epoch, {
                    'train_total_loss': current_train_total_loss,
                    'train_energy_loss': current_train_energy_loss,
                    'train_force_loss': current_train_force_loss,
                    'train_energy_rmse': current_train_energy_rmse,
                    'train_force_rmse': current_train_force_rmse,
                })
        
        # Log epoch summary
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        if self.is_main_process:
            logging.info(
                f"Epoch {epoch} finished - Total time: {epoch_time:.1f}s | "
                f"Avg batch time: {avg_batch_time:.3f}s | Total batches: {len(batch_times)}"
            )
        
        return {
            'total_loss': np.mean(total_batch_loss) if total_batch_loss else 0,
            'energy_loss': np.mean(total_batch_energy_loss) if total_batch_energy_loss else 0,
            'energy_rmse': np.mean(total_batch_energy_rmse) if total_batch_energy_rmse else 0,
            'energy_rmse_avg': np.mean(total_batch_energy_rmse_avg) if total_batch_energy_rmse_avg else 0,
            'force_loss': np.mean(total_batch_force_loss) if total_batch_force_loss else 0,
            'force_rmse': np.mean(total_batch_force_rmse) if total_batch_force_rmse else 0,
            'epoch_time': epoch_time,
            'avg_batch_time': avg_batch_time,
        }
    
    def _gather_variable_tensors(self, local_tensor):
        """
        Gather tensors of variable sizes from all processes.
        
        Args:
            local_tensor: Local tensor on this process (can have different size on each process)
            
        Returns:
            Concatenated tensor from all processes (on all ranks)
        """
        if not self.distributed:
            return local_tensor
        
        # Get local size
        local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=self.device)
        
        # Gather all sizes
        all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(all_sizes, local_size)
        all_sizes = [s.item() for s in all_sizes]
        max_size = max(all_sizes)
        
        # Pad local tensor to max size
        if local_tensor.dim() == 1:
            padded_tensor = torch.zeros(max_size, dtype=local_tensor.dtype, device=self.device)
            padded_tensor[:local_tensor.shape[0]] = local_tensor.to(self.device)
        else:
            # For multi-dimensional tensors (e.g., forces with shape [N, 3])
            padded_shape = list(local_tensor.shape)
            padded_shape[0] = max_size
            padded_tensor = torch.zeros(padded_shape, dtype=local_tensor.dtype, device=self.device)
            padded_tensor[:local_tensor.shape[0]] = local_tensor.to(self.device)
        
        # Gather padded tensors
        gathered_tensors = [torch.zeros_like(padded_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, padded_tensor)
        
        # Concatenate and remove padding
        result_parts = []
        for i, tensor in enumerate(gathered_tensors):
            result_parts.append(tensor[:all_sizes[i]].cpu())
        
        return torch.cat(result_parts)
    
    def validate(self, epoch, train_metrics=None):
        """Validate model.
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary containing training metrics at this batch count
        """
        self.model.eval()
        self.e3trans.eval()  # DDP model's eval() will call internal module's eval()
        
        val_E_preds = []
        val_E_targets = []
        val_E_avg_preds = []
        val_E_avg_targets = []
        val_F_preds = []
        val_F_targets = []
        
        total_batches = len(self.val_loader)
        if self.is_main_process:
            logging.info("=" * 80)
            logging.info(f"Validation Started - Epoch {epoch} | Batch {self.batch_count}")
            logging.info("=" * 80)
        
        for batch_idx_loader, batch in enumerate(self.val_loader):
            if batch is None:
                continue
            
            (pos, A, batch_idx, force_ref, target_energies,
             edge_src, edge_dst, edge_shifts, cell) = batch
            
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
            E_offset_val = scatter(mapped_A, batch_idx, dim=0, reduce='sum')

            # Choose evaluation model: EMA (optional) vs current
            if self.use_ema_for_validation and self.ema_enabled and self.e3trans_ema is not None:
                eval_model = self.e3trans_ema
            else:
                eval_model = self.e3trans.module if self.distributed else self.e3trans

            E_per_atom = eval_model(pos, A, batch_idx, edge_src, edge_dst, edge_shifts, cell)
            # E_conv_val 形状: [Batch_Size, 1] (consistent with original train-val.py)
            E_conv_val = scatter(E_per_atom, batch_idx, dim=0, reduce='sum')
            # E_mean_val 形状: [Batch_Size] (consistent with original train-val.py)
            E_mean_val = E_conv_val.view(-1) + E_offset_val
            
            # Calculate forces (need gradients, so not in no_grad context)
            grads = torch.autograd.grad(
                E_mean_val.sum(), pos, create_graph=False, retain_graph=False
            )[0]
            
            # Restore force units (consistent with training)
            fx_pred = self.val_dataset.restore_force(-grads[:, 0])
            fy_pred = self.val_dataset.restore_force(-grads[:, 1])
            fz_pred = self.val_dataset.restore_force(-grads[:, 2])
            f_pred_final = torch.stack([fx_pred, fy_pred, fz_pred], dim=1)
            
            f_ref_final = force_ref * self.force_shift_value
            
            # Detach and move to CPU (no gradients needed after force calculation)
            with torch.no_grad():
                E_pred_batch = E_mean_val.detach().cpu()
                E_target_batch = target_energies.detach().cpu()
                
                val_E_preds.append(E_pred_batch)
                val_E_targets.append(E_target_batch)
                
                num_atoms_per_mol = scatter(torch.ones_like(batch_idx), batch_idx, dim=0, reduce='sum')
                val_E_avg_preds.append((E_mean_val / num_atoms_per_mol).detach().cpu())
                val_E_avg_targets.append((target_energies / num_atoms_per_mol).detach().cpu())
                
                val_F_preds.append(f_pred_final.detach().cpu())
                val_F_targets.append(f_ref_final.detach().cpu())
                
                # Stream output: log first sample's restored energy (consistent with original train-val.py)
                # Only log on main process
                if self.is_main_process:
                    restored_pred = self.val_dataset.restore_energy(E_mean_val[0].item())
                    restored_target = self.val_dataset.restore_energy(target_energies[0].item())
                    logging.info(f"Batch {batch_idx_loader + 1}/{total_batches} Sample -> Pred: {restored_pred:.4f} , True: {restored_target:.4f}")
        
        # Check if we have any validation data on this process
        local_has_data = len(val_E_preds) > 0
        
        # In distributed mode, check if any process has no data
        if self.distributed:
            has_data_tensor = torch.tensor([1 if local_has_data else 0], device=self.device)
            dist.all_reduce(has_data_tensor, op=dist.ReduceOp.SUM)
            if has_data_tensor.item() == 0:
                if self.is_main_process:
                    logging.warning("No validation data found on any process! Skipping validation metrics.")
                return
        elif not local_has_data:
            if self.is_main_process:
                logging.warning("No validation data found! Skipping validation metrics.")
            return
        
        # If this process has no data, create empty tensors for gathering
        if not local_has_data:
            # Create empty tensors with correct dtype
            all_E_preds = torch.tensor([], dtype=torch.get_default_dtype())
            all_E_targets = torch.tensor([], dtype=torch.get_default_dtype())
            all_E_avg_preds = torch.tensor([], dtype=torch.get_default_dtype())
            all_E_avg_targets = torch.tensor([], dtype=torch.get_default_dtype())
            all_F_preds = torch.zeros((0, 3), dtype=torch.get_default_dtype())
            all_F_targets = torch.zeros((0, 3), dtype=torch.get_default_dtype())
        else:
            # Concatenate all predictions and targets
            all_E_preds = torch.cat(val_E_preds)
            all_E_targets = torch.cat(val_E_targets)
            all_E_avg_preds = torch.cat(val_E_avg_preds)
            all_E_avg_targets = torch.cat(val_E_avg_targets)
            all_F_preds = torch.cat(val_F_preds)
            all_F_targets = torch.cat(val_F_targets)
        
        # In distributed training, gather results from all processes using variable-size gather
        if self.distributed:
            all_E_preds = self._gather_variable_tensors(all_E_preds)
            all_E_targets = self._gather_variable_tensors(all_E_targets)
            all_E_avg_preds = self._gather_variable_tensors(all_E_avg_preds)
            all_E_avg_targets = self._gather_variable_tensors(all_E_avg_targets)
            all_F_preds = self._gather_variable_tensors(all_F_preds)
            all_F_targets = self._gather_variable_tensors(all_F_targets)
        
        # Verify shapes match
        assert all_E_preds.shape == all_E_targets.shape, \
            f"Energy shape mismatch: pred {all_E_preds.shape} vs target {all_E_targets.shape}"
        assert all_E_avg_preds.shape == all_E_avg_targets.shape, \
            f"Energy avg shape mismatch: pred {all_E_avg_preds.shape} vs target {all_E_avg_targets.shape}"
        assert all_F_preds.shape == all_F_targets.shape, \
            f"Force shape mismatch: pred {all_F_preds.shape} vs target {all_F_targets.shape}"
        
        # Log total number of samples (only on main process)
        if self.is_main_process:
            num_samples = len(all_E_preds)
            logging.info(f"[Validation] Processing {num_samples} samples")
        
        # Calculate all metrics (restore units consistent with original train-val.py)
        val_energy_rmse = torch.sqrt(F.mse_loss(all_E_preds, all_E_targets)).item()
        val_energy_rmse = self.val_dataset.restore_force(val_energy_rmse)  # Restore units
        
        val_energy_mae = F.l1_loss(all_E_preds, all_E_targets).item()
        val_energy_mae = self.val_dataset.restore_force(val_energy_mae)  # Restore units
        
        val_energy_rmse_avg = torch.sqrt(F.mse_loss(all_E_avg_preds, all_E_avg_targets)).item()
        val_energy_rmse_avg = self.val_dataset.restore_force(val_energy_rmse_avg)  # Restore units
        
        val_energy_mae_avg = F.l1_loss(all_E_avg_preds, all_E_avg_targets).item()
        val_energy_mae_avg = self.val_dataset.restore_force(val_energy_mae_avg)  # Restore units
        
        val_energy_loss = F.mse_loss(all_E_avg_preds, all_E_avg_targets).item()  # For logging
        
        # Force metrics (already restored in loop)
        val_force_rmse = torch.sqrt(F.mse_loss(all_F_preds.view(-1), all_F_targets.view(-1))).item()
        val_force_mae = F.l1_loss(all_F_preds.view(-1), all_F_targets.view(-1)).item()
        val_force_loss = F.mse_loss(all_F_preds.view(-1), all_F_targets.view(-1)).item()  # For early stopping
        
        # Log validation results (consistent with original train-val.py format)
        # Only log on main process
        if self.is_main_process:
            logging.info(f"""
                                "Energy Loss Val (MSE)": {val_energy_loss},
                                "Energy RMSE Val": {val_energy_rmse},
                                "Energy RMSE avg Val": {val_energy_rmse_avg},
                                "Energy MAE avg Val": {val_energy_mae_avg},
                                "Force MAE Val": {val_force_mae},
                                "Force RMSE Val":  {val_force_rmse},
                                "Current learning rate": {self.optimizer.param_groups[0]['lr']},
                                "a (energy weight)": {self.a},
                                "b (force weight)": {self.b}
                                """)

        # Save validation results (only on main process)
        if self.is_main_process:
            # Save validation CSV files if enabled
            if self.save_val_csv:
                df_energy_val = pd.DataFrame({
                    "Target_Energy": all_E_targets.numpy(),
                    "Predicted_Energy": all_E_preds.numpy(),
                    "Delta": (all_E_preds - all_E_targets).numpy()
                })
                energy_save_path = f"val_energy_epoch{epoch}_batch{self.batch_count}.csv"
                df_energy_val.to_csv(energy_save_path, index=False)
                
                f_pred_np = all_F_preds.numpy()
                f_true_np = all_F_targets.numpy()
                
                df_force_val = pd.DataFrame({
                    "Fx_True": f_true_np[:, 0], "Fy_True": f_true_np[:, 1], "Fz_True": f_true_np[:, 2],
                    "Fx_Pred": f_pred_np[:, 0], "Fy_Pred": f_pred_np[:, 1], "Fz_Pred": f_pred_np[:, 2]
                })
                force_save_path = f"val_force_epoch{epoch}_batch{self.batch_count}.csv"
                df_force_val.to_csv(force_save_path, index=False)
        
        # Early stopping - use weighted combination of energy and force loss
        # Calculate current validation loss as: a * energy_loss + b * force_loss
        current_val_loss = self.a * val_energy_loss + self.b * val_force_loss
        
        if self.distributed:
            # Broadcast current_val_loss from rank 0 to ensure consistency
            val_loss_tensor = torch.tensor([current_val_loss], device=self.device)
            dist.broadcast(val_loss_tensor, src=0)
            current_val_loss_synced = val_loss_tensor.item()
        else:
            current_val_loss_synced = current_val_loss
        
        if current_val_loss_synced < self.best_val_loss:
            self.best_val_loss = current_val_loss_synced
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Sync patience_counter across all processes
        if self.distributed:
            patience_tensor = torch.tensor([self.patience_counter], device=self.device)
            dist.broadcast(patience_tensor, src=0)
            self.patience_counter = int(patience_tensor.item())
        
        # Record validation metrics to CSV
        record = {
            'epoch': epoch,
            'batch_count': self.batch_count,
            'val_total_loss': current_val_loss_synced if self.distributed else current_val_loss,
            'val_energy_loss': val_energy_loss,
            'val_force_loss': val_force_loss,
            'val_energy_rmse': val_energy_rmse,
            'val_energy_rmse_avg': val_energy_rmse_avg,
            'val_energy_mae': val_energy_mae,
            'val_energy_mae_avg': val_energy_mae_avg,
            'val_force_rmse': val_force_rmse,
            'val_force_mae': val_force_mae,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'a': self.a,
            'b': self.b,
        }
        
        # Add training metrics if provided
        if train_metrics:
            record.update({
                'train_total_loss': train_metrics.get('train_total_loss', 0),
                'train_energy_loss': train_metrics.get('train_energy_loss', 0),
                'train_force_loss': train_metrics.get('train_force_loss', 0),
                'train_energy_rmse': train_metrics.get('train_energy_rmse', 0),
                'train_force_rmse': train_metrics.get('train_force_rmse', 0),
            })
        else:
            record.update({
                'train_total_loss': 0,
                'train_energy_loss': 0,
                'train_force_loss': 0,
                'train_energy_rmse': 0,
                'train_force_rmse': 0,
            })
        
        # Only save on main process
        if self.is_main_process:
            self.loss_records.append(record)
            
            # Save to CSV (append mode, create header if file doesn't exist)
            df = pd.DataFrame(self.loss_records)
            df.to_csv(self.loss_csv_path, index=False)
            
            # Save model (only save e3trans as MainNet is not used in inference)
            # Extract state_dict from DDP model if needed
            if self.distributed:
                e3trans_state_dict = self.e3trans.module.state_dict()
            else:
                e3trans_state_dict = self.e3trans.state_dict()
            
            checkpoint_dict = {
                'e3trans_state_dict': e3trans_state_dict,
                'a': self.a,
                'b': self.b,
                'batch_count': self.batch_count,
                'epoch': epoch,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'swa_applied': self.swa_applied,
                'ema_enabled': self.ema_enabled,
            }

            if self.save_ema_model and self.ema_enabled and self.e3trans_ema is not None:
                checkpoint_dict['e3trans_ema_state_dict'] = self.e3trans_ema.state_dict()

            torch.save(checkpoint_dict, f'combined_model_epoch{epoch}_batch_count{self.batch_count}.pth')
        
        self.model.train()
        self.e3trans.train()
    
    def run_training(self):
        """Main training loop."""
        if self.is_main_process:
            if self.resumed_from_checkpoint:
                logging.info(f"Resuming training from epoch {self.start_epoch}/{self.epoch_numbers}")
            else:
                logging.info(f"Training started - Total epochs: {self.epoch_numbers}")
            logging.info(f"Checkpoint will be saved to: {self.checkpoint_path}")
        
        # Check if training is already complete
        if self.start_epoch > self.epoch_numbers:
            if self.is_main_process:
                logging.info(f"Training already completed (start epoch {self.start_epoch} > total epochs {self.epoch_numbers})")
                logging.info(f"Model is already at epoch {self.start_epoch - 1}")
                logging.info("No further training needed.")
            return
        
        last_epoch = self.start_epoch - 1  # Track the last completed epoch
        for epoch in range(self.start_epoch, self.epoch_numbers + 1):
            last_epoch = epoch
            metrics = self.train_epoch(epoch)
            
            if self.is_main_process:
                logging.info(
                    f"Epoch {epoch}/{self.epoch_numbers} completed - "
                    f"Avg Energy Loss: {metrics['energy_loss']:.6f} | "
                    f"Avg Force Loss: {metrics['force_loss']:.6f} | "
                    f"Energy RMSE: {metrics['energy_rmse']:.6f} | "
                    f"Force RMSE: {metrics['force_rmse']:.6f} | "
                    f"Epoch Time: {metrics['epoch_time']:.1f}s | "
                    f"Avg Batch Time: {metrics['avg_batch_time']:.3f}s"
                )
            
            # Note: patience_counter is already synced in validate() method
            if self.patience_counter >= self.patience:
                if self.is_main_process:
                    logging.info(f"Early stopping triggered at epoch {epoch} (patience: {self.patience})")
                break
        
        # Save final checkpoint (only save e3trans as MainNet is not used in inference)
        if self.is_main_process:
            logging.info(f"Saving final checkpoint to {self.checkpoint_path}...")
            # Extract state_dict from DDP model if needed
            if self.distributed:
                e3trans_state_dict = self.e3trans.module.state_dict()
            else:
                e3trans_state_dict = self.e3trans.state_dict()
            
            checkpoint_dict = {
                'epoch': last_epoch,
                'e3trans_state_dict': e3trans_state_dict,
                'a': self.a,
                'b': self.b,
                'batch_count': self.batch_count,
                'best_val_loss': self.best_val_loss,
                'patience_counter': self.patience_counter,
                'swa_applied': self.swa_applied,
                'ema_enabled': self.ema_enabled,
            }
            if self.save_ema_model and self.ema_enabled and self.e3trans_ema is not None:
                checkpoint_dict['e3trans_ema_state_dict'] = self.e3trans_ema.state_dict()

            torch.save(checkpoint_dict, self.checkpoint_path)
            logging.info(f"Training completed! Final model saved to {self.checkpoint_path}")