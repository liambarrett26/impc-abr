#!/usr/bin/env python3
"""
Main training script for ContrastiveVAE-DEC audiometric phenotype discovery.

This script provides a complete command-line interface for training the model
with support for configuration files, resuming from checkpoints, and comprehensive
logging and monitoring.
"""

import argparse
import sys
import yaml
import socket
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.seed import ensure_reproducibility
from utils.logging import setup_logging, TrainingLogger
from utils.checkpoint import ModelCheckpoint, load_checkpoint
from config import load_config
from data.dataset import create_abr_dataset
from data.dataloader import create_data_module
from data.preprocessor import ABRFeaturePreprocessor
from models.full_model import create_model
from losses.combined_loss import create_combined_loss
from training.trainer import ContrastiveVAEDECTrainer
from evaluation.metrics import compute_comprehensive_metrics


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train ContrastiveVAE-DEC for audiometric phenotype discovery',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml',
                       help='Path to model configuration file')
    
    # Data
    parser.add_argument('--data-path', type=str,
                       help='Path to dataset (overrides config)')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (overrides config)')
    
    # Training
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--lr', type=float,
                       help='Learning rate (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic algorithms (slower but more reproducible)')
    
    # Resuming training
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--resume-from-best', action='store_true',
                       help='Resume from best checkpoint in checkpoint directory')
    parser.add_argument('--resume-from-last', action='store_true',
                       help='Resume from last checkpoint in checkpoint directory')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Base directory for experiments')
    parser.add_argument('--experiment-name', type=str,
                       help='Name of experiment (auto-generated if not provided)')
    parser.add_argument('--run-id', type=str,
                       help='Unique run identifier (auto-generated if not provided)')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Directory for model checkpoints (overrides config)')
    
    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-dir', type=str,
                       help='Directory for log files')
    parser.add_argument('--no-console-log', action='store_true',
                       help='Disable console logging')
    
    # Model options
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, cuda:0, etc.)')
    parser.add_argument('--compile-model', action='store_true',
                       help='Use torch.compile for model optimization (PyTorch 2.0+)')
    
    # Training stages
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip VAE pretraining stage')
    parser.add_argument('--pretrain-only', action='store_true',
                       help='Run only VAE pretraining stage')
    parser.add_argument('--joint-only', action='store_true',
                       help='Run only joint training stage (requires pretrained model)')
    
    # Validation and testing
    parser.add_argument('--validate', action='store_true',
                       help='Run validation after training')
    parser.add_argument('--test', action='store_true',
                       help='Run test evaluation after training')
    
    return parser.parse_args()


def load_configs(args: argparse.Namespace) -> Dict[str, Any]:
    """Load and merge configuration files with command line overrides."""
    # Load base configurations
    with open(args.config, 'r') as f:
        training_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    # Merge configurations
    config = {
        'training': training_config,
        'model': model_config
    }
    
    # Apply command line overrides
    if args.data_path:
        config['training']['dataset']['data_path'] = args.data_path
    if args.batch_size:
        config['training']['dataset']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['training']['max_epochs'] = args.epochs
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    if args.checkpoint_dir:
        config['training']['logging']['save_dir'] = args.checkpoint_dir
    
    return config


def setup_device(device_arg: str) -> torch.device:
    """Setup computation device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_arg)
    
    logging.info(f"Using device: {device}")
    return device


def create_experiment_info(args: argparse.Namespace) -> Dict[str, str]:
    """Create experiment information with unique naming."""
    from datetime import datetime
    import socket
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Auto-generate experiment name if not provided
    if not args.experiment_name:
        stage_info = []
        if args.pretrain_only:
            stage_info.append("pretrain")
        elif args.joint_only:
            stage_info.append("joint")
        elif args.skip_pretrain:
            stage_info.append("cluster_joint")
        else:
            stage_info.append("full_pipeline")
        
        epochs_info = f"{args.epochs}ep" if args.epochs else "default"
        args.experiment_name = f"{'_'.join(stage_info)}_{epochs_info}"
    
    # Auto-generate run ID if not provided
    if not args.run_id:
        hostname = socket.gethostname()[:8]  # First 8 chars of hostname
        args.run_id = f"{timestamp}_{hostname}"
    
    return {
        'experiment_name': args.experiment_name,
        'run_id': args.run_id,
        'timestamp': timestamp
    }

def setup_output_directories(args: argparse.Namespace, experiment_info: Dict[str, str]) -> Dict[str, Path]:
    """Setup organized output directories with experiment tracking."""
    
    # Create structured hierarchy: experiments/{experiment_name}/{run_id}/
    base_dir = Path(args.output_dir) / experiment_info['experiment_name'] / experiment_info['run_id']
    
    directories = {
        'base': base_dir,
        'logs': base_dir / 'logs',
        'checkpoints': base_dir / 'checkpoints',
        'metrics': base_dir / 'metrics',
        'visualizations': base_dir / 'visualizations',
        'config': base_dir / 'config'  # Save configs used for this run
    }
    
    # Create directories
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save experiment metadata
    import json
    metadata = {
        'experiment_name': experiment_info['experiment_name'],
        'run_id': experiment_info['run_id'],
        'timestamp': experiment_info['timestamp'],
        'command_line_args': vars(args),
        'hostname': socket.gethostname(),
        'git_commit': _get_git_commit_hash()
    }
    
    with open(base_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return directories

def save_experiment_configs(args: argparse.Namespace, config: Dict[str, Any], output_dirs: Dict[str, Path]):
    """Save configuration files used for this experiment."""
    config_dir = output_dirs['config']
    
    # Copy original config files
    if Path(args.config).exists():
        shutil.copy2(args.config, config_dir / 'training_config.yaml')
    if Path(args.model_config).exists():
        shutil.copy2(args.model_config, config_dir / 'model_config.yaml')
    
    # Save merged config as used by the training
    with open(config_dir / 'merged_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Save command line arguments
    with open(config_dir / 'command_line_args.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False, indent=2)

def _get_git_commit_hash() -> str:
    """Get current git commit hash if available."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        return result.stdout.strip() if result.returncode == 0 else 'unknown'
    except:
        return 'unknown'


def load_data(config: Dict[str, Any], device: torch.device) -> Dict[str, DataLoader]:
    """Load and prepare datasets."""
    training_config = config['training']
    model_config = config['model']
    
    # Setup preprocessor
    from data.preprocessor import create_default_config
    preprocessor_config = create_default_config()
    preprocessor = ABRFeaturePreprocessor(preprocessor_config)
    
    # Create data module and setup
    data_module = create_data_module(training_config['dataset'])
    data_module.setup(training_config['dataset']['data_path'])
    
    # Get individual dataloaders
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()
    
    # Package dataloaders for compatibility
    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }
    
    logging.info(f"Loaded datasets: train={len(dataloaders['train'].dataset)}, "
                f"val={len(dataloaders['val'].dataset)}, "
                f"test={len(dataloaders['test'].dataset)}")
    
    return dataloaders


def setup_model_and_training(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Setup model, optimizer, scheduler, and loss function."""
    model_config = config['model']
    training_config = config['training']
    
    # Create model
    model = create_model(model_config)
    model = model.to(device)
    
    # Model compilation (PyTorch 2.0+)
    if hasattr(torch, 'compile') and config.get('compile_model', False):
        model = torch.compile(model)
        logging.info("Model compiled with torch.compile")
    
    # Create optimizer
    optimizer_config = training_config['optimizer']
    if optimizer_config['type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(optimizer_config['lr']),
            betas=list(optimizer_config['betas']),
            eps=float(optimizer_config['eps']),
            weight_decay=float(optimizer_config['weight_decay'])
        )
    elif optimizer_config['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config['lr']),
            betas=list(optimizer_config['betas']),
            eps=float(optimizer_config['eps']),
            weight_decay=float(optimizer_config['weight_decay'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_config['type']}")
    
    # Create scheduler
    scheduler_config = training_config['scheduler']
    if scheduler_config['type'] == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_config['T_max']),
            eta_min=float(scheduler_config['eta_min'])
        )
    elif scheduler_config['type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config['step_size']),
            gamma=float(scheduler_config['gamma'])
        )
    else:
        scheduler = None
    
    # Create loss function
    loss_fn = create_combined_loss(model_config)
    
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_fn': loss_fn
    }


def resume_from_checkpoint(checkpoint_path: Path, model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]) -> Dict[str, Any]:
    """Resume training from checkpoint."""
    logging.info(f"Resuming from checkpoint: {checkpoint_path}")
    
    checkpoint_data = load_checkpoint(
        model=model,
        filepath=checkpoint_path,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    logging.info(f"Resumed from epoch {checkpoint_data['epoch']}")
    return checkpoint_data


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create experiment information and directories
    experiment_info = create_experiment_info(args)
    output_dirs = setup_output_directories(args, experiment_info)
    
    # Setup logging
    log_dir = Path(args.log_dir) if args.log_dir else output_dirs['logs']
    logger = setup_logging(
        log_level=args.log_level,
        log_dir=log_dir,
        experiment_name=f"{experiment_info['experiment_name']}_{experiment_info['run_id']}",
        console_output=not args.no_console_log,
        file_output=True
    )
    
    # Setup reproducibility
    ensure_reproducibility(args.seed, args.deterministic)
    
    # Load configurations
    config = load_configs(args)
    
    # Save configurations for this experiment
    save_experiment_configs(args, config, output_dirs)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load data
    dataloaders = load_data(config, device)
    
    # Setup model and training components
    training_components = setup_model_and_training(config, device)
    model = training_components['model']
    optimizer = training_components['optimizer']
    scheduler = training_components['scheduler']
    loss_fn = training_components['loss_fn']
    
    # Setup checkpointing
    checkpoint_callback = ModelCheckpoint(
        checkpoint_dir=output_dirs['checkpoints'],
        monitor=config['training']['training']['monitor_checkpoint'],
        mode='min',
        save_top_k=config['training']['training']['save_top_k'],
        save_last=config['training']['training']['save_last']
    )
    
    # Setup training logger
    training_logger = TrainingLogger(
        logger=logger,
        metrics_file=output_dirs['metrics'] / 'training_metrics.json'
    )
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    training_logger.log_model_info(model, total_params, trainable_params)
    training_logger.log_hyperparameters(config)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        resume_from_checkpoint(Path(args.resume), model, optimizer, scheduler)
    elif args.resume_from_best:
        best_ckpt = checkpoint_callback.load_best_checkpoint()
        if best_ckpt:
            checkpoint_data = resume_from_checkpoint(best_ckpt, model, optimizer, scheduler)
            start_epoch = checkpoint_data['epoch'] + 1
    elif args.resume_from_last:
        last_ckpt = checkpoint_callback.load_last_checkpoint()
        if last_ckpt:
            checkpoint_data = resume_from_checkpoint(last_ckpt, model, optimizer, scheduler)
            start_epoch = checkpoint_data['epoch'] + 1
    
    # Training stages
    training_stages = config['training']['training_stages']
    
    try:
        # Stage 1: VAE Pretraining (if not skipped)
        if not args.skip_pretrain and not args.joint_only:
            logger.info("Starting Stage 1: VAE Pretraining")
            from training.pretrain import VAEPretrainer
            
            # Set model to pretraining stage
            model.set_training_stage('pretrain')
            
            # Create VAE pretrainer
            pretrainer = VAEPretrainer(model, config, device)
            
            # Run pretraining
            pretrain_results = pretrainer.pretrain(
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                save_dir=output_dirs['checkpoints'] / 'pretrain'
            )
            
            logger.info(f"Pretraining completed. Best loss: {pretrain_results['best_loss']:.4f}")
            
            if args.pretrain_only:
                logger.info("Pretraining completed. Exiting as requested.")
                return
        
        # Stage 2: Cluster Initialization
        if not args.pretrain_only:
            logger.info("Starting Stage 2: Cluster Initialization")
            
            # Set model to cluster initialization stage
            model.set_training_stage('cluster_init')
            
            # Initialize clusters
            model.initialize_clusters(dataloaders['train'])
            logger.info("Cluster initialization completed")
        
        # Stage 3: Joint Training (if not pretrain-only)
        if not args.pretrain_only:
            logger.info("Starting Stage 3: Joint Training")
            from training.finetune import JointTrainer
            
            # Set model to joint training stage
            model.set_training_stage('joint')
            
            # Create joint trainer (pass full config, not just training config)
            joint_trainer = JointTrainer(model, config, device)
            
            # Run joint training
            joint_results = joint_trainer.train(
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                save_dir=output_dirs['checkpoints'] / 'joint'
            )
            
            logger.info(f"Joint training completed. Best loss: {joint_results['best_loss']:.4f}")
        
        logger.info("Training completed successfully!")
        
        # Validation
        if args.validate:
            logger.info("Running validation evaluation")
            # TODO: Implement validation evaluation
            logger.info("Validation evaluation not yet implemented")
        
        # Testing
        if args.test:
            logger.info("Running test evaluation")
            # TODO: Implement test evaluation
            logger.info("Test evaluation not yet implemented")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == '__main__':
    main()