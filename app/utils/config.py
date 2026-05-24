"""
Utilities module for the Traffic Accident Detection System.
Contains common functions for logging, configuration loading, and helpers.
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime


class Config:
    """Configuration manager for the system."""
    
    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def load_config(self, config_path: str = "config/settings.yaml") -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file
            
        Returns:
            Configuration dictionary
        """
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        Example: config.get('model.yolo_model')
        
        Args:
            key: Configuration key with dot notation
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by dot-notation key.
        
        Args:
            key: Configuration key with dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


def setup_logger(
    name: str = "AccidentDetection",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger for the system.
    
    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class PerformanceMetrics:
    """Track performance metrics of the system."""
    
    def __init__(self):
        self.frame_count = 0
        self.total_time = 0.0
        self.detection_times = []
        self.tracking_times = []
        self.collision_times = []
        self.accidents_detected = 0
        self.start_time = None
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.frame_count = 0
        self.total_time = 0.0
        self.detection_times = []
        self.tracking_times = []
        self.collision_times = []
        self.accidents_detected = 0
        self.start_time = None
    
    def get_fps(self) -> float:
        """Calculate frames per second."""
        if self.total_time == 0:
            return 0.0
        return self.frame_count / self.total_time
    
    def get_avg_detection_time(self) -> float:
        """Get average detection time in milliseconds."""
        if not self.detection_times:
            return 0.0
        return np.mean(self.detection_times) * 1000
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        return {
            'total_frames': self.frame_count,
            'total_time_seconds': self.total_time,
            'fps': self.get_fps(),
            'avg_detection_time_ms': self.get_avg_detection_time(),
            'avg_tracking_time_ms': np.mean(self.tracking_times) * 1000 if self.tracking_times else 0,
            'avg_collision_detection_time_ms': np.mean(self.collision_times) * 1000 if self.collision_times else 0,
            'total_accidents_detected': self.accidents_detected
        }


def ensure_output_dir(output_dir: str = "outputs") -> str:
    """
    Ensure output directory exists and create timestamped subdirectory.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to timestamped output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_dir = os.path.join(output_dir, timestamp)
    os.makedirs(timestamped_dir, exist_ok=True)
    return timestamped_dir


def get_device() -> str:
    """
    Detect and return the available device (cuda or cpu).
    
    Returns:
        'cuda' if available, else 'cpu'
    """
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        return 'cpu'


def format_time(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_timestamp(frame_index: int, fps: int = 30) -> str:
    """
    Convert frame index to timestamp.
    
    Args:
        frame_index: Frame number
        fps: Frames per second
        
    Returns:
        Formatted timestamp
    """
    seconds = frame_index / fps
    return format_time(seconds)


# Singleton logger instance
_logger = setup_logger("AccidentDetection")


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    return _logger
