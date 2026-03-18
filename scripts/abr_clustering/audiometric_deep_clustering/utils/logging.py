"""
Logging configuration for ContrastiveVAE-DEC training and evaluation.

This module provides structured logging setup with file and console outputs,
proper formatting, and integration with training monitoring systems.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import json
from datetime import datetime
import os


def setup_logging(log_level: str = 'INFO',
                 log_dir: Optional[Path] = None,
                 log_filename: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 console_output: bool = True,
                 file_output: bool = True,
                 json_format: bool = False) -> logging.Logger:
    """
    Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_dir: Directory for log files (creates if doesn't exist)
        log_filename: Custom log filename (auto-generated if None)
        experiment_name: Name of experiment for log organization
        console_output: Whether to log to console
        file_output: Whether to log to file
        json_format: Whether to use JSON format for structured logging
        
    Returns:
        Configured logger instance
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Setup log directory
    if log_dir is None:
        log_dir = Path('logs')
    else:
        log_dir = Path(log_dir)
    
    if file_output:
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate log filename if not provided
    if log_filename is None and file_output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name:
            log_filename = f"{experiment_name}_{timestamp}.log"
        else:
            log_filename = f"audiometric_clustering_{timestamp}.log"
    
    # Create formatters
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Configure handlers
    handlers = []
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        handlers.append(console_handler)
    
    # File handler
    if file_output:
        file_handler = logging.FileHandler(log_dir / log_filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Get logger for this package
    logger = logging.getLogger('audiometric_clustering')
    logger.setLevel(numeric_level)
    
    # Log configuration info
    logger.info(f"Logging configured - Level: {log_level}, Console: {console_output}, File: {file_output}")
    if file_output:
        logger.info(f"Log file: {log_dir / log_filename}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with proper naming.
    
    Args:
        name: Logger name (uses calling module if None)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module's name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'audiometric_clustering')
    
    # Ensure it's under our package hierarchy
    if not name.startswith('audiometric_clustering'):
        name = f'audiometric_clustering.{name}'
    
    return logging.getLogger(name)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON objects for easier parsing and analysis.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in {'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'message'}:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class TrainingLogger:
    """
    Specialized logger for training metrics and progress.
    
    Provides methods for logging training-specific information like
    loss values, metrics, and training progress.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None,
                 metrics_file: Optional[Path] = None):
        """
        Initialize training logger.
        
        Args:
            logger: Base logger instance (creates if None)
            metrics_file: File to save training metrics
        """
        self.logger = logger or get_logger('training')
        self.metrics_file = metrics_file
        self.training_metrics = []
        
        if self.metrics_file:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  metrics: Optional[Dict[str, float]] = None,
                  lr: Optional[float] = None, extra_info: Optional[Dict[str, Any]] = None):
        """
        Log epoch training information.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
            metrics: Dictionary of additional metrics
            lr: Current learning rate
            extra_info: Additional information to log
        """
        # Create log message
        msg_parts = [f"Epoch {epoch}: train_loss={train_loss:.6f}"]
        
        if val_loss is not None:
            msg_parts.append(f"val_loss={val_loss:.6f}")
        
        if lr is not None:
            msg_parts.append(f"lr={lr:.2e}")
        
        if metrics:
            metric_strs = [f"{k}={v:.4f}" for k, v in metrics.items()]
            msg_parts.extend(metric_strs)
        
        self.logger.info(" | ".join(msg_parts))
        
        # Save to metrics file
        if self.metrics_file:
            metric_entry = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'lr': lr,
                'timestamp': datetime.now().isoformat()
            }
            
            if metrics:
                metric_entry.update(metrics)
            
            if extra_info:
                metric_entry.update(extra_info)
            
            self.training_metrics.append(metric_entry)
            self._save_metrics()
    
    def log_batch(self, epoch: int, batch_idx: int, loss: float,
                  batch_size: int, dataset_size: int, log_interval: int = 100):
        """
        Log batch-level training information.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            loss: Batch loss value
            batch_size: Size of current batch
            dataset_size: Total dataset size
            log_interval: How often to log (every N batches)
        """
        if batch_idx % log_interval == 0:
            progress = 100.0 * batch_idx * batch_size / dataset_size
            self.logger.info(
                f"Epoch {epoch} [{batch_idx * batch_size}/{dataset_size} "
                f"({progress:.0f}%)]\tLoss: {loss:.6f}"
            )
    
    def log_stage_transition(self, from_stage: str, to_stage: str,
                           epoch: int, additional_info: Optional[Dict[str, Any]] = None):
        """
        Log training stage transitions.
        
        Args:
            from_stage: Previous training stage
            to_stage: New training stage
            epoch: Current epoch
            additional_info: Additional information about the transition
        """
        msg = f"Training stage transition at epoch {epoch}: {from_stage} -> {to_stage}"
        
        if additional_info:
            info_str = ", ".join([f"{k}={v}" for k, v in additional_info.items()])
            msg += f" ({info_str})"
        
        self.logger.info(msg)
    
    def log_model_info(self, model, total_params: Optional[int] = None,
                      trainable_params: Optional[int] = None):
        """
        Log model architecture information.
        
        Args:
            model: PyTorch model
            total_params: Total number of parameters
            trainable_params: Number of trainable parameters
        """
        if total_params is None:
            total_params = sum(p.numel() for p in model.parameters())
        
        if trainable_params is None:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {model.__class__.__name__}")
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    def log_hyperparameters(self, config: Dict[str, Any]):
        """
        Log hyperparameters and configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.logger.info("Hyperparameters:")
        for section, params in config.items():
            if isinstance(params, dict):
                self.logger.info(f"  {section}:")
                for key, value in params.items():
                    self.logger.info(f"    {key}: {value}")
            else:
                self.logger.info(f"  {section}: {params}")
    
    def _save_metrics(self):
        """Save training metrics to file."""
        if self.metrics_file and self.training_metrics:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)


class ProgressLogger:
    """
    Logger for tracking long-running operations with progress indicators.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger or get_logger('progress')
        self.operations = {}
    
    def start_operation(self, operation_id: str, total_steps: int, description: str = ""):
        """
        Start tracking a long-running operation.
        
        Args:
            operation_id: Unique identifier for the operation
            total_steps: Total number of steps in the operation
            description: Description of the operation
        """
        self.operations[operation_id] = {
            'total_steps': total_steps,
            'current_step': 0,
            'start_time': datetime.now(),
            'description': description
        }
        
        msg = f"Started operation '{operation_id}'"
        if description:
            msg += f": {description}"
        msg += f" (0/{total_steps})"
        
        self.logger.info(msg)
    
    def update_progress(self, operation_id: str, step: Optional[int] = None,
                       message: Optional[str] = None, log_interval: int = 10):
        """
        Update progress for an operation.
        
        Args:
            operation_id: Operation identifier
            step: Current step (increments by 1 if None)
            message: Additional message to log
            log_interval: Log every N% progress
        """
        if operation_id not in self.operations:
            self.logger.warning(f"Operation '{operation_id}' not found")
            return
        
        op = self.operations[operation_id]
        
        if step is not None:
            op['current_step'] = step
        else:
            op['current_step'] += 1
        
        # Calculate progress percentage
        progress_pct = (op['current_step'] / op['total_steps']) * 100
        
        # Log at intervals
        if (op['current_step'] % max(1, op['total_steps'] // (100 // log_interval)) == 0 or
            op['current_step'] == op['total_steps']):
            
            elapsed = datetime.now() - op['start_time']
            msg = f"Operation '{operation_id}': {op['current_step']}/{op['total_steps']} "
            msg += f"({progress_pct:.1f}%) - Elapsed: {elapsed}"
            
            if message:
                msg += f" - {message}"
            
            self.logger.info(msg)
    
    def finish_operation(self, operation_id: str, success: bool = True,
                        final_message: Optional[str] = None):
        """
        Finish tracking an operation.
        
        Args:
            operation_id: Operation identifier
            success: Whether the operation completed successfully
            final_message: Final message to log
        """
        if operation_id not in self.operations:
            self.logger.warning(f"Operation '{operation_id}' not found")
            return
        
        op = self.operations[operation_id]
        elapsed = datetime.now() - op['start_time']
        
        status = "completed successfully" if success else "failed"
        msg = f"Operation '{operation_id}' {status} in {elapsed}"
        
        if final_message:
            msg += f" - {final_message}"
        
        if success:
            self.logger.info(msg)
        else:
            self.logger.error(msg)
        
        del self.operations[operation_id]


def configure_external_loggers(level: str = 'WARNING'):
    """
    Configure external library loggers to reduce noise.
    
    Args:
        level: Log level for external libraries
    """
    external_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'numba',
        'sklearn'
    ]
    
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(getattr(logging, level.upper()))


# Initialize logging configuration when module is imported
def _init_logging():
    """Initialize default logging if not already configured."""
    root_logger = logging.getLogger()
    
    # Only configure if no handlers exist
    if not root_logger.handlers:
        setup_logging(
            log_level='INFO',
            console_output=True,
            file_output=False,  # Don't create files by default
            json_format=False
        )
        
        # Configure external loggers
        configure_external_loggers()


# Auto-initialize when module is imported
_init_logging()