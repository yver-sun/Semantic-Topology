"""工具函数模块 - 提供配置加载、路径管理、日志等功能"""
import os
import logging
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件，支持环境变量覆盖
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认路径
        
    Returns:
        配置字典
    """
    if config_path is None:
        root_dir = Path(__file__).parent.parent
        config_path = root_dir / "config" / "default_config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # 环境变量覆盖
    config = _apply_env_overrides(config)
    
    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用环境变量覆盖配置值
    
    支持格式: 
    - NLP_MODEL -> nlp.model
    - TDA_N_LANDMARKS -> tda.n_landmarks
    """
    env_mappings = {
        'PROJECT_NAME': ('project_name', str),
        'NLP_MODEL': ('nlp.model', str),
        'NLP_SPACY_MODEL': ('nlp.spacy_model', str),
        'NLP_MIN_FREQ': ('nlp.min_freq', int),
        'TDA_LANDMARK_STRATEGY': ('tda.landmark_strategy', str),
        'TDA_N_LANDMARKS': ('tda.n_landmarks', int),
        'TDA_PERSISTENCE_THRESHOLD': ('tda.persistence_threshold', float),
        'VIZ_MAPPER_NEIGHBORS': ('visualization.mapper_neighbors', int),
        'VIZ_MAPPER_OVERLAP': ('visualization.mapper_overlap', float),
        'DATA_INPUT_DIR': ('data.input_dir', str),
        'DATA_OUTPUT_DIR': ('data.output_dir', str),
        'FREQ_MIN': ('nlp.min_freq', int),  # 向后兼容
        'K_LANDMARKS': ('tda.n_landmarks', int),  # 向后兼容
    }
    
    for env_key, (config_path, type_func) in env_mappings.items():
        env_value = os.environ.get(env_key)
        if env_value is not None:
            try:
                value = type_func(env_value)
                _set_nested_value(config, config_path.split('.'), value)
            except (ValueError, TypeError) as e:
                logging.warning(f"无法解析环境变量 {env_key}={env_value}: {e}")
    
    return config


def _set_nested_value(d: Dict[str, Any], keys: list, value: Any) -> None:
    """在嵌套字典中设置值"""
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def get_path(config: Dict[str, Any], key_path: str, relative_to: Optional[Path] = None) -> Path:
    """
    从配置中获取路径，支持相对路径
    
    Args:
        config: 配置字典
        key_path: 配置键路径，如 'data.output_dir'
        relative_to: 相对路径的基准目录，默认使用项目根目录
        
    Returns:
        Path对象
    """
    if relative_to is None:
        relative_to = Path(__file__).parent.parent
    
    keys = key_path.split('.')
    value = config
    for key in keys:
        value = value.get(key, {})
    
    path = Path(value) if value else Path()
    
    if not path.is_absolute():
        path = relative_to / path
    
    return path.resolve()


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    配置日志系统
    
    Args:
        level: 日志级别
        log_file: 日志文件路径，如果为None则只输出到控制台
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers
    )


def ensure_dir(path: Path) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
        
    Returns:
        Path对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).parent.parent


def safe_filename(filename: str) -> str:
    """
    将文件名中的非法字符替换为下划线
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

