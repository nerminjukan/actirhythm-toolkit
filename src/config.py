"""Configuration file for pipeline parameters."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


# Default configuration
DEFAULT_CONFIG = {
    # Global settings
    'random_state': 42,
    'n_jobs': -1,  # Use all available cores
    
    # Data paths
    'paths': {
        'raw_data': 'data/raw',
        'processed_data': 'data/processed',
        'outputs': 'outputs',
        'logs': 'logs',
    },
    
    # Quality control
    'qc': {
        'expected_freq': '1H',
        'remove_duplicates': True,
        'fill_method': None,  # None, 'ffill', 'bfill', 'interpolate'
    },
    
    # Feature engineering
    'features': {
        'rolling_windows': [6, 12, 24],
        'rolling_stats': ['mean', 'std', 'min', 'max'],
        'lag_periods': [1, 6, 24],
        'time_features': ['hour', 'day_of_week', 'is_weekend'],
        'activity_method': 'abs',  # 'magnitude', 'abs', 'squared'
    },
    
    # HMM/HSMM
    'hmm': {
        'state_range': [2, 3, 4, 5],
        'n_iter': 100,
        'covariance_type': 'diag',  # 'spherical', 'diag', 'full', 'tied'
        'selection_criterion': 'bic',  # 'bic' or 'aic'
    },
    
    # Cosinor analysis
    'cosinor': {
        'period': 24.0,  # hours
        'test_periods': [24.0, 12.0],
        'significance_test': True,
        'n_permutations': 1000,
    },
    
    # Machine learning
    'ml': {
        'test_size': 0.2,
        'cv_folds': 5,
        
        # Random Forest
        'rf': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
        },
        
        # XGBoost
        'xgb': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        
        # GLMM
        'glmm': {
            'family': 'gaussian',
            'reml': True,
        },
    },
    
    # Logging
    'logging': {
        'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
        'log_to_file': True,
        'log_params': True,
    },
}


class Config:
    """Configuration manager for the pipeline."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Parameters
        ----------
        config_dict : dict, optional
            Custom configuration dictionary. If None, uses defaults.
        """
        if config_dict is None:
            self.config = DEFAULT_CONFIG.copy()
        else:
            self.config = self._merge_configs(DEFAULT_CONFIG, config_dict)
    
    @staticmethod
    def _merge_configs(default: dict, custom: dict) -> dict:
        """Merge custom config with defaults."""
        merged = default.copy()
        for key, value in custom.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = Config._merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
    
    @classmethod
    def from_yaml(cls, filepath: str):
        """
        Load configuration from YAML file.
        
        Parameters
        ----------
        filepath : str
            Path to YAML config file
            
        Returns
        -------
        Config
            Configuration object
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    @classmethod
    def from_default_or_env(cls, filepath: str = 'config.yaml'):
        """Load configuration from explicit path or THESIS_CONFIG_FILE override."""
        return cls.from_yaml(resolve_config_path(filepath))
    
    def save_yaml(self, filepath: str):
        """
        Save configuration to YAML file.
        
        Parameters
        ----------
        filepath : str
            Output path for YAML file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def __getitem__(self, key):
        """Access config like a dictionary."""
        return self.config[key]
    
    def __repr__(self):
        """String representation."""
        return f"Config({self.config})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def set_config(new_config: Config):
    """Set the global configuration instance."""
    global config
    config = new_config


def resolve_config_path(default_path: str = 'config.yaml') -> Path:
    """Resolve config path from environment override or a default relative path."""
    return Path(os.environ.get('THESIS_CONFIG_FILE', default_path))


def load_config_or_defaults(default_path: str = 'config.yaml') -> Config:
    """Load config from file when available, otherwise return default config."""
    config_path = resolve_config_path(default_path)
    if config_path.exists():
        return Config.from_yaml(str(config_path))
    return Config()
