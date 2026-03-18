"""
Main training orchestrator for ContrastiveVAE-DEC model.

This module coordinates the complete 3-stage training process:
1. VAE pretraining
2. Cluster initialization
3. Joint training with all objectives
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Any, List
import logging
from pathlib import Path
import time
import json
from datetime import datetime

from training.pretrain import VAEPretrainer
from training.finetune import JointTrainer
from models.full_model import ContrastiveVAEDEC
from losses.combined_loss import MultiObjectiveLoss

logger = logging.getLogger(__name__)


class ContrastiveVAEDECTrainer:
    """
    Complete training orchestrator for ContrastiveVAE-DEC.

    Manages the full 3-stage training pipeline with proper transitions,
    checkpointing, and monitoring throughout the process.
    """

    def __init__(self, model: ContrastiveVAEDEC, config: Dict, device: torch.device):
        """
        Initialize the complete trainer.

        Args:
            model: ContrastiveVAE-DEC model
            config: Complete training configuration
            device: Training device
        """
        self.model = model
        self.config = config
        self.device = device

        # Training stage management
        self.current_stage = 'pretrain'
        self.stage_history = []

        # Initialize loss function
        self.loss_fn = MultiObjectiveLoss(config)

        # Training components (will be initialized as needed)
        self.pretrainer = None
        self.joint_trainer = None

        # Global training state
        self.total_epochs = 0
        self.start_time = None

        # Results tracking
        self.training_results = {
            'pretrain': {},
            'cluster_init': {},
            'joint': {},
            'overall': {}
        }

        logger.info("Initialized ContrastiveVAE-DEC trainer")

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              save_dir: Path, resume_from: Optional[Path] = None) -> Dict[str, Any]:
        """
        Execute complete 3-stage training pipeline.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_dir: Directory for saving checkpoints and results
            resume_from: Optional checkpoint to resume from

        Returns:
            Complete training results
        """
        logger.info("Starting complete ContrastiveVAE-DEC training pipeline")
        self.start_time = time.time()

        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration
        self._save_config(save_dir)

        # Resume from checkpoint if specified
        if resume_from:
            self._resume_training(resume_from)

        try:
            # Stage 1: VAE Pretraining
            if self.current_stage == 'pretrain':
                self._execute_pretraining(train_loader, val_loader, save_dir)
                self._transition_to_cluster_init()

            # Stage 2: Cluster Initialization
            if self.current_stage == 'cluster_init':
                self._execute_cluster_initialization(train_loader, save_dir)
                self._transition_to_joint()

            # Stage 3: Joint Training
            if self.current_stage == 'joint':
                self._execute_joint_training(train_loader, val_loader, save_dir)

            # Finalize training
            self._finalize_training(save_dir)

        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._save_emergency_checkpoint(save_dir)
            raise

        total_time = time.time() - self.start_time
        logger.info(f"Complete training pipeline finished in {total_time:.2f} seconds")

        return self.training_results

    def _execute_pretraining(self, train_loader: DataLoader,
                           val_loader: DataLoader, save_dir: Path):
        """Execute VAE pretraining stage."""
        logger.info("=== Stage 1: VAE Pretraining ===")

        self.current_stage = 'pretrain'
        self.model.set_training_stage('pretrain')
        self.loss_fn.set_training_stage('pretrain')

        # Initialize pretrainer
        self.pretrainer = VAEPretrainer(self.model, self.config, self.device)

        # Run pretraining
        pretrain_save_dir = save_dir / 'pretrain'
        pretrain_results = self.pretrainer.pretrain(
            train_loader, val_loader, pretrain_save_dir
        )

        # Store results
        self.training_results['pretrain'] = pretrain_results
        self.total_epochs += pretrain_results['final_epoch'] + 1

        # Analysis
        analysis = self.pretrainer.analyze_pretraining()
        self.training_results['pretrain']['analysis'] = analysis

        logger.info(f"Pretraining completed. Best val loss: {pretrain_results['best_loss']:.4f}")

        # Update stage history
        self.stage_history.append({
            'stage': 'pretrain',
            'epochs': pretrain_results['final_epoch'] + 1,
            'best_loss': pretrain_results['best_loss'],
            'completion_time': time.time() - self.start_time
        })

    def _execute_cluster_initialization(self, train_loader: DataLoader, save_dir: Path):
        """Execute cluster initialization stage."""
        logger.info("=== Stage 2: Cluster Initialization ===")

        self.current_stage = 'cluster_init'
        self.model.set_training_stage('cluster_init')
        self.loss_fn.set_training_stage('cluster_init')

        # Initialize clusters using current latent representations
        cluster_init_start = time.time()

        method = self.config['clustering']['cluster_init']
        self.model.initialize_clusters(train_loader, method=method)

        cluster_init_time = time.time() - cluster_init_start

        # Validate cluster initialization
        cluster_stats = self._analyze_cluster_initialization(train_loader)

        # Store results
        self.training_results['cluster_init'] = {
            'initialization_method': method,
            'initialization_time': cluster_init_time,
            'cluster_stats': cluster_stats
        }

        logger.info(f"Cluster initialization completed in {cluster_init_time:.2f}s")
        logger.info(f"Active clusters: {cluster_stats['num_active_clusters']}")

        # Save cluster initialization checkpoint
        cluster_save_dir = save_dir / 'cluster_init'
        cluster_save_dir.mkdir(exist_ok=True)
        self._save_stage_checkpoint(cluster_save_dir / 'clusters_initialized.pt')

        # Update stage history
        self.stage_history.append({
            'stage': 'cluster_init',
            'initialization_time': cluster_init_time,
            'num_active_clusters': cluster_stats['num_active_clusters'],
            'completion_time': time.time() - self.start_time
        })

    def _execute_joint_training(self, train_loader: DataLoader,
                              val_loader: DataLoader, save_dir: Path):
        """Execute joint training stage."""
        logger.info("=== Stage 3: Joint Training ===")

        self.current_stage = 'joint'
        self.model.set_training_stage('joint')
        self.loss_fn.set_training_stage('joint')

        # Initialize joint trainer
        self.joint_trainer = JointTrainer(self.model, self.config, self.device)

        # Run joint training
        joint_save_dir = save_dir / 'joint'
        joint_results = self.joint_trainer.train(
            train_loader, val_loader, joint_save_dir,
            starting_epoch=self.total_epochs
        )

        # Store results
        self.training_results['joint'] = joint_results
        self.total_epochs += joint_results['final_epoch'] + 1

        logger.info(f"Joint training completed. Best val loss: {joint_results['best_loss']:.4f}")

        # Update stage history
        self.stage_history.append({
            'stage': 'joint',
            'epochs': joint_results['final_epoch'] + 1,
            'best_loss': joint_results['best_loss'],
            'completion_time': time.time() - self.start_time
        })

    def _transition_to_cluster_init(self):
        """Transition from pretraining to cluster initialization."""
        logger.info("Transitioning from pretraining to cluster initialization")
        self.current_stage = 'cluster_init'

        # Any necessary model updates for cluster initialization
        # (Most work happens in _execute_cluster_initialization)

    def _transition_to_joint(self):
        """Transition from cluster initialization to joint training."""
        logger.info("Transitioning from cluster initialization to joint training")
        self.current_stage = 'joint'

        # Prepare model for joint training
        if not self.model.clusters_initialized:
            logger.warning("Clusters not properly initialized before joint training")

    def _analyze_cluster_initialization(self, train_loader: DataLoader) -> Dict[str, Any]:
        """Analyze quality of cluster initialization."""
        self.model.eval()

        all_latents = []
        all_assignments = []

        with torch.no_grad():
            for batch in train_loader:
                batch = self._batch_to_device(batch)

                # Get latent representations and cluster assignments
                z = self.model.encode(batch['features'])
                q, hard_assignments = self.model.get_cluster_assignments(batch['features'])

                all_latents.append(z)
                all_assignments.append(hard_assignments)

        # Concatenate results
        latents = torch.cat(all_latents, dim=0)
        assignments = torch.cat(all_assignments, dim=0)

        # Analyze cluster quality
        cluster_sizes = torch.bincount(assignments, minlength=self.config['clustering']['num_clusters'])
        active_clusters = (cluster_sizes > 0).sum().item()

        # Basic cluster statistics
        stats = {
            'num_active_clusters': active_clusters,
            'total_clusters': self.config['clustering']['num_clusters'],
            'cluster_sizes': cluster_sizes.tolist(),
            'largest_cluster': cluster_sizes.max().item(),
            'smallest_cluster': cluster_sizes.min().item(),
            'cluster_balance': cluster_sizes.std().item() / cluster_sizes.mean().item()
        }

        return stats

    def _finalize_training(self, save_dir: Path):
        """Finalize training and save final results."""
        total_time = time.time() - self.start_time

        # Compile overall results
        self.training_results['overall'] = {
            'total_training_time': total_time,
            'total_epochs': self.total_epochs,
            'stage_history': self.stage_history,
            'final_stage': self.current_stage,
            'success': True
        }

        # Final model analysis
        final_analysis = self._analyze_final_model()
        self.training_results['overall']['final_analysis'] = final_analysis

        # Save final checkpoint
        final_checkpoint_path = save_dir / 'final_model.pt'
        self._save_final_checkpoint(final_checkpoint_path)

        # Save complete results
        results_path = save_dir / 'training_results.json'
        self._save_results(results_path)

        logger.info("Training finalization completed")

    def _analyze_final_model(self) -> Dict[str, Any]:
        """Analyze the final trained model."""
        # This would include comprehensive model analysis
        # For now, return basic information
        return {
            'model_stage': self.current_stage,
            'clusters_initialized': self.model.clusters_initialized,
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }

    def _save_config(self, save_dir: Path):
        """Save training configuration."""
        config_path = save_dir / 'training_config.json'

        # Convert config to JSON-serializable format
        json_config = self._config_to_json(self.config)

        with open(config_path, 'w') as f:
            json.dump(json_config, f, indent=2)

        logger.info(f"Saved training configuration: {config_path}")

    def _config_to_json(self, config: Dict) -> Dict:
        """Convert config to JSON-serializable format."""
        json_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                json_config[key] = self._config_to_json(value)
            elif isinstance(value, (list, tuple)):
                json_config[key] = list(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                json_config[key] = value
            else:
                json_config[key] = str(value)
        return json_config

    def _save_stage_checkpoint(self, path: Path):
        """Save checkpoint for current stage."""
        checkpoint = {
            'stage': self.current_stage,
            'total_epochs': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'stage_history': self.stage_history,
            'training_results': self.training_results,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved stage checkpoint: {path}")

    def _save_final_checkpoint(self, path: Path):
        """Save final model checkpoint."""
        checkpoint = {
            'stage': 'completed',
            'total_epochs': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'training_results': self.training_results,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved final checkpoint: {path}")

    def _save_emergency_checkpoint(self, save_dir: Path):
        """Save emergency checkpoint on training failure."""
        emergency_path = save_dir / 'emergency_checkpoint.pt'

        checkpoint = {
            'stage': self.current_stage,
            'total_epochs': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'training_results': self.training_results,
            'error_timestamp': datetime.now().isoformat()
        }

        torch.save(checkpoint, emergency_path)
        logger.info(f"Saved emergency checkpoint: {emergency_path}")

    def _save_results(self, path: Path):
        """Save training results to JSON."""
        # Convert results to JSON-serializable format
        json_results = self._results_to_json(self.training_results)

        with open(path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"Saved training results: {path}")

    def _results_to_json(self, results: Dict) -> Dict:
        """Convert results to JSON-serializable format."""
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = self._results_to_json(value)
            elif isinstance(value, list):
                json_results[key] = value
            elif isinstance(value, (int, float, str, bool, type(None))):
                json_results[key] = value
            elif hasattr(value, 'item'):  # torch.Tensor scalars
                json_results[key] = value.item()
            else:
                json_results[key] = str(value)
        return json_results

    def _resume_training(self, checkpoint_path: Path):
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore training state
        self.current_stage = checkpoint['stage']
        self.total_epochs = checkpoint['total_epochs']
        self.stage_history = checkpoint['stage_history']
        self.training_results = checkpoint['training_results']

        logger.info(f"Resumed from stage: {self.current_stage}, epoch: {self.total_epochs}")

    def _batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to training device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        if not self.training_results:
            return {'status': 'not_started'}

        summary = {
            'current_stage': self.current_stage,
            'total_epochs': self.total_epochs,
            'stages_completed': len(self.stage_history)
        }

        if self.start_time:
            summary['elapsed_time'] = time.time() - self.start_time

        # Add stage-specific summaries
        for stage_info in self.stage_history:
            stage = stage_info['stage']
            summary[f'{stage}_completed'] = True

            if 'best_loss' in stage_info:
                summary[f'{stage}_best_loss'] = stage_info['best_loss']

        return summary


def create_trainer(model: ContrastiveVAEDEC, config: Dict,
                  device: torch.device) -> ContrastiveVAEDECTrainer:
    """
    Factory function to create trainer.

    Args:
        model: ContrastiveVAE-DEC model
        config: Training configuration
        device: Training device

    Returns:
        Configured trainer
    """
    return ContrastiveVAEDECTrainer(model, config, device)