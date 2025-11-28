"""语义分析模块 - 拓扑语义锚定、环纯度分析"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available, semantic analysis will fail")

from .embedder import get_device, load_model_and_tokenizer
from .utils import get_project_root


logger = logging.getLogger(__name__)


def compute_cycle_centroid(
    cycle_words: List[str],
    word_embeddings: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    计算β1环上所有词向量的质心
    
    数学公式: C = (1/n) * Σ(v_i) where v_i 是环上第i个词的嵌入向量
    
    Args:
        cycle_words: 环上的词列表
        word_embeddings: 词到嵌入向量的映射字典
        
    Returns:
        质心向量 (d,) 其中d是嵌入维度
    """
    if not cycle_words:
        raise ValueError("环词列表为空，无法计算质心")
    
    # 收集环上所有词的嵌入向量
    vectors = []
    missing_words = []
    
    for word in cycle_words:
        if word in word_embeddings:
            vectors.append(word_embeddings[word])
        else:
            missing_words.append(word)
    
    if missing_words:
        logger.warning(f"缺失嵌入向量的词: {missing_words[:10]}...")
    
    if not vectors:
        raise ValueError("无法找到任何词的嵌入向量")
    
    # 计算质心（均值）
    vectors_array = np.array(vectors)
    centroid = np.mean(vectors_array, axis=0)
    
    logger.info(f"计算环质心: {len(vectors)}/{len(cycle_words)} 个词有有效嵌入")
    
    return centroid


def get_concept_embeddings(
    concept_words: List[str],
    config: Dict,
    tokenizer: Any = None,
    model: Any = None,
    device: Any = None
) -> Dict[str, np.ndarray]:
    """
    获取概念词的嵌入向量
    
    Args:
        concept_words: 概念词列表（如 ["Void", "God", "Silence", "Technology"]）
        config: 配置字典
        tokenizer: BERT分词器（可选，如果为None则自动加载）
        model: BERT模型（可选，如果为None则自动加载）
        device: 计算设备（可选，如果为None则自动获取）
        
    Returns:
        概念词到嵌入向量的映射字典
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("transformers不可用，无法获取概念词嵌入")
    
    # 加载模型（如果需要）
    if tokenizer is None or model is None:
        model_name = config.get('embedding', {}).get('model_name') or config.get('nlp', {}).get('model', 'bert-base-cased')
        if device is None:
            device = get_device(config)
        tokenizer, model = load_model_and_tokenizer(model_name, device)
    
    if device is None:
        device = get_device(config)
    
    concept_embeddings = {}
    
    logger.info(f"获取 {len(concept_words)} 个概念词的嵌入向量")
    
    with torch.no_grad():
        for concept in concept_words:
            try:
                # 分词
                encoded = tokenizer(
                    concept,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(device)
                
                # 获取嵌入
                outputs = model(**encoded)
                # 使用[CLS]标记的嵌入，或平均池化
                mean_pooling = config.get('embedding', {}).get('mean_pooling', False)
                if mean_pooling:
                    attention_mask = encoded['attention_mask']
                    embeddings = outputs.last_hidden_state
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
                else:
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
                
                concept_embeddings[concept] = embedding
                
            except Exception as e:
                logger.error(f"获取概念词 '{concept}' 的嵌入失败: {e}")
    
    logger.info(f"成功获取 {len(concept_embeddings)} 个概念词的嵌入")
    
    return concept_embeddings


def semantic_anchoring(
    cycle_centroid: np.ndarray,
    concept_embeddings: Dict[str, np.ndarray],
    metric: str = 'euclidean'
) -> List[Tuple[str, float]]:
    """
    计算环质心与概念词的语义距离
    
    数学公式: d(C, V_concept) = ||C - V_concept||_2 (欧氏距离)
    
    Args:
        cycle_centroid: 环的质心向量
        concept_embeddings: 概念词到嵌入向量的映射
        metric: 距离度量 ('euclidean', 'cosine')
        
    Returns:
        [(概念词, 距离), ...] 列表，按距离升序排序
    """
    distances = []
    
    for concept, concept_vec in concept_embeddings.items():
        if metric == 'euclidean':
            # 欧氏距离
            dist = np.linalg.norm(cycle_centroid - concept_vec)
        elif metric == 'cosine':
            # 余弦距离
            dot_product = np.dot(cycle_centroid, concept_vec)
            norm_centroid = np.linalg.norm(cycle_centroid)
            norm_concept = np.linalg.norm(concept_vec)
            if norm_centroid > 0 and norm_concept > 0:
                cosine_sim = dot_product / (norm_centroid * norm_concept)
                dist = 1 - cosine_sim  # 转换为距离
            else:
                dist = float('inf')
        else:
            raise ValueError(f"不支持的距离度量: {metric}")
        
        distances.append((concept, dist))
    
    # 按距离排序
    distances.sort(key=lambda x: x[1])
    
    return distances


def interpret_cycle_meaning(
    cycle_words: List[str],
    word_embeddings: Dict[str, np.ndarray],
    concept_embeddings: Dict[str, np.ndarray],
    config: Dict,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    自动解释环的语义指向
    
    Args:
        cycle_words: 环上的词列表
        word_embeddings: 词到嵌入向量的映射
        concept_embeddings: 概念词到嵌入向量的映射
        config: 配置字典
        top_k: 返回前k个最接近的概念词
        
    Returns:
        解释结果字典，包含：
        - centroid: 质心向量
        - closest_concepts: 最接近的概念词列表
        - distances: 距离列表
        - interpretation: 文本解释
    """
    logger.info(f"解释环的语义指向，环包含 {len(cycle_words)} 个词")
    
    # 计算质心
    centroid = compute_cycle_centroid(cycle_words, word_embeddings)
    
    # 计算与概念词的距离
    metric = config.get('semantics', {}).get('distance_metric', 'euclidean')
    distances = semantic_anchoring(centroid, concept_embeddings, metric)
    
    # 获取前k个最接近的概念
    closest_concepts = distances[:top_k]
    
    # 生成解释文本
    interpretation = f"环质心最接近的概念是: {closest_concepts[0][0]} (距离: {closest_concepts[0][1]:.4f})"
    if len(closest_concepts) > 1:
        interpretation += f"\n其他相关概念: {', '.join([f'{c[0]} ({c[1]:.4f})' for c in closest_concepts[1:]])}"
    
    result = {
        'centroid': centroid,
        'closest_concepts': closest_concepts,
        'all_distances': distances,
        'interpretation': interpretation,
        'cycle_size': len(cycle_words)
    }
    
    logger.info(interpretation)
    
    return result


def compute_cycle_heterogeneity(
    cycle_words: List[str],
    word_embeddings: Dict[str, np.ndarray],
    metric: str = 'euclidean'
) -> float:
    """
    计算环的语义异质性
    
    异质性定义为环内词向量之间的平均距离。
    高异质性意味着环上的词在语义空间中是分散的（如技术词+神学词混合）。
    低异质性意味着环上的词在语义空间中聚集（如同义词循环）。
    
    Args:
        cycle_words: 环上的词列表
        word_embeddings: 词到嵌入向量的映射
        metric: 距离度量
        
    Returns:
        异质性值（平均距离）
    """
    if len(cycle_words) < 2:
        return 0.0
    
    # 获取所有词的嵌入向量
    vectors = []
    for word in cycle_words:
        if word in word_embeddings:
            vectors.append(word_embeddings[word])
    
    if len(vectors) < 2:
        return 0.0
    
    vectors_array = np.array(vectors)
    n = len(vectors_array)
    
    # 计算所有词对之间的距离
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            if metric == 'euclidean':
                dist = np.linalg.norm(vectors_array[i] - vectors_array[j])
            elif metric == 'cosine':
                dot_product = np.dot(vectors_array[i], vectors_array[j])
                norm_i = np.linalg.norm(vectors_array[i])
                norm_j = np.linalg.norm(vectors_array[j])
                if norm_i > 0 and norm_j > 0:
                    cosine_sim = dot_product / (norm_i * norm_j)
                    dist = 1 - cosine_sim
                else:
                    dist = 1.0
            else:
                raise ValueError(f"不支持的距离度量: {metric}")
            distances.append(dist)
    
    # 平均距离即为异质性
    heterogeneity = np.mean(distances) if distances else 0.0
    
    return float(heterogeneity)


def classify_cycle_type(
    heterogeneity: float,
    threshold: Optional[float] = None
) -> str:
    """
    基于异质性分类环的类型
    
    Args:
        heterogeneity: 异质性值
        threshold: 阈值（如果为None，则使用默认阈值0.5）
        
    Returns:
        环类型: 'high_heterogeneity' 或 'low_heterogeneity'
    """
    if threshold is None:
        threshold = 0.5  # 默认阈值
    
    if heterogeneity >= threshold:
        return 'high_heterogeneity'
    else:
        return 'low_heterogeneity'


def analyze_cycle_purity(
    cycles: List[Tuple[List[str], float, float, float]],  # [(words, birth, death, persistence), ...]
    word_embeddings: Dict[str, np.ndarray],
    config: Dict
) -> List[Dict[str, Any]]:
    """
    分析多个环的纯度（异质性）
    
    Args:
        cycles: 环列表，每个环是 (词列表, birth, death, persistence) 元组
        word_embeddings: 词到嵌入向量的映射
        config: 配置字典
        
    Returns:
        每个环的分析结果列表
    """
    results = []
    
    semantics_config = config.get('semantics', {})
    heterogeneity_threshold = semantics_config.get('heterogeneity_threshold', 0.5)
    metric = semantics_config.get('distance_metric', 'euclidean')
    
    logger.info(f"分析 {len(cycles)} 个环的纯度")
    
    for idx, (words, birth, death, persistence) in enumerate(cycles):
        heterogeneity = compute_cycle_heterogeneity(words, word_embeddings, metric)
        cycle_type = classify_cycle_type(heterogeneity, heterogeneity_threshold)
        
        result = {
            'cycle_id': idx,
            'words': words,
            'birth': birth,
            'death': death,
            'persistence': persistence,
            'heterogeneity': heterogeneity,
            'type': cycle_type,
            'size': len(words)
        }
        
        results.append(result)
        
        logger.debug(f"环 {idx}: 异质性={heterogeneity:.4f}, 类型={cycle_type}")
    
    return results

