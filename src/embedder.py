"""嵌入生成模块 - BERT嵌入生成、批量处理"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import torch

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, embedding generation will fail")

from .nlp_processor import process_text, extract_words_with_context, load_spacy_model
from .utils import ensure_dir, safe_filename


logger = logging.getLogger(__name__)

# 模型缓存
_tokenizer_cache: Dict[str, Any] = {}
_model_cache: Dict[str, Any] = {}


def get_device(config: Dict) -> torch.device:
    """
    获取计算设备
    
    Args:
        config: 配置字典
        
    Returns:
        torch.device对象
    """
    device_config = config.get('embedding', {}).get('device', 'auto')
    use_gpu = config.get('embedding', {}).get('use_gpu', True)
    
    if device_config == 'auto':
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("使用CPU")
    else:
        device = torch.device(device_config)
        logger.info(f"使用设备: {device}")
    
    return device


def load_model_and_tokenizer(model_name: str, device: torch.device) -> Tuple[Any, Any]:
    """
    加载BERT模型和分词器（带缓存）
    
    Args:
        model_name: 模型名称
        device: 计算设备
        
    Returns:
        (tokenizer, model) 元组
    """
    # 检查缓存
    if model_name in _tokenizer_cache and model_name in _model_cache:
        logger.debug(f"使用缓存的模型: {model_name}")
        return _tokenizer_cache[model_name], _model_cache[model_name]
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers不可用，请先安装: pip install transformers torch")
    
    logger.info(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    # 缓存
    _tokenizer_cache[model_name] = tokenizer
    _model_cache[model_name] = model
    
    return tokenizer, model


def get_word_embeddings_batch(
    contexts: List[str],
    words: List[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    config: Dict
) -> np.ndarray:
    """
    批量生成词嵌入
    
    Args:
        contexts: 上下文列表
        words: 对应的词列表
        tokenizer: 分词器
        model: BERT模型
        device: 计算设备
        config: 配置字典
        
    Returns:
        嵌入矩阵 (N, d)
    """
    max_length = config.get('embedding', {}).get('max_length', 512)
    mean_pooling = config.get('embedding', {}).get('mean_pooling', False)
    batch_size = config.get('nlp', {}).get('batch_size', 32)
    
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(contexts), batch_size):
            batch_contexts = contexts[i:i+batch_size]
            batch_words = words[i:i+batch_size]
            
            # 分词
            encoded = tokenizer(
                batch_contexts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(device)
            
            # 前向传播
            outputs = model(**encoded)
            
            # 提取嵌入
            if mean_pooling:
                # 平均池化（排除padding）
                attention_mask = encoded['attention_mask']
                embeddings_batch = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings_batch.size()).float()
                sum_embeddings = torch.sum(embeddings_batch * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                embeddings_batch = sum_embeddings / sum_mask
            else:
                # 使用[CLS]标记
                embeddings_batch = outputs.last_hidden_state[:, 0, :]
            
            embeddings.append(embeddings_batch.cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.debug(f"处理批次: {i // batch_size + 1}/{(len(contexts) + batch_size - 1) // batch_size}")
    
    return np.vstack(embeddings)


def embed_text_file(
    text_path: Path,
    output_dir: Path,
    config: Dict,
    nlp_model: Any = None
) -> Path:
    """
    从文本文件生成嵌入并保存
    
    Args:
        text_path: 文本文件路径
        output_dir: 输出目录
        config: 配置字典
        nlp_model: Spacy模型（可选）
        
    Returns:
        输出文件路径
    """
    text_path = Path(text_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"处理文本文件: {text_path}")
    
    # 读取文本
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # NLP处理 - 提取词及其上下文
    words_with_context = extract_words_with_context(text, config, nlp_model)
    
    if not words_with_context:
        raise ValueError(f"未能从文本中提取任何词: {text_path}")
    
    words = [w[0] for w in words_with_context]
    contexts = [w[1] for w in words_with_context]
    
    logger.info(f"提取了 {len(words)} 个词")
    
    # 加载模型
    model_name = config.get('embedding', {}).get('model_name') or config.get('nlp', {}).get('model', 'bert-base-cased')
    device = get_device(config)
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    
    # 生成嵌入
    logger.info("生成嵌入...")
    embeddings = get_word_embeddings_batch(
        contexts, words, tokenizer, model, device, config
    )
    
    logger.info(f"生成嵌入形状: {embeddings.shape}")
    
    # 生成输出文件名
    base_name = text_path.stem
    safe_name = safe_filename(base_name)
    output_path = output_dir / f"{safe_name}_embeddings.npz"
    
    # 保存
    np.savez_compressed(
        output_path,
        X=embeddings,
        labels=np.array(words, dtype=object)
    )
    
    logger.info(f"保存嵌入: {output_path}")
    
    return output_path


def embed_text(
    text: str,
    config: Dict,
    nlp_model: Any = None
) -> Tuple[np.ndarray, List[str]]:
    """
    从文本字符串生成嵌入
    
    Args:
        text: 输入文本
        config: 配置字典
        nlp_model: Spacy模型（可选）
        
    Returns:
        (嵌入矩阵, 词列表) 元组
    """
    # NLP处理
    words_with_context = extract_words_with_context(text, config, nlp_model)
    
    if not words_with_context:
        return np.array([]), []
    
    words = [w[0] for w in words_with_context]
    contexts = [w[1] for w in words_with_context]
    
    # 加载模型
    model_name = config.get('embedding', {}).get('model_name') or config.get('nlp', {}).get('model', 'bert-base-cased')
    device = get_device(config)
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    
    # 生成嵌入
    embeddings = get_word_embeddings_batch(
        contexts, words, tokenizer, model, device, config
    )
    
    return embeddings, words

