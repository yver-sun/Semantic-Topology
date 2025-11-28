"""拓扑数据分析模块 - TDA分析、持续同调、边界词提取"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    logging.warning("ripser not available, TDA analysis will fail")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import pairwise_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, landmark selection will fail")

from .utils import ensure_dir, safe_filename


logger = logging.getLogger(__name__)


def select_landmarks(
    X: np.ndarray,
    n_landmarks: int,
    strategy: str = 'kmeans',
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    选择地标点
    
    Args:
        X: 嵌入矩阵 (N, d)
        n_landmarks: 地标数量
        strategy: 策略 ('kmeans' 或 'maxmin')
        metric: 距离度量
        
    Returns:
        地标索引数组
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn不可用，无法选择地标")
    
    n_samples = X.shape[0]
    
    if n_landmarks >= n_samples:
        logger.warning(f"地标数量({n_landmarks}) >= 样本数量({n_samples})，返回所有样本")
        return np.arange(n_samples)
    
    if strategy == 'kmeans':
        logger.info(f"使用KMeans选择 {n_landmarks} 个地标")
        kmeans = KMeans(n_clusters=n_landmarks, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # 找到每个簇的中心点（最近的点）
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        landmark_indices = []
        for i in range(n_landmarks):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) > 0:
                # 找到离中心最近的点
                distances = pairwise_distances(
                    X[cluster_points],
                    centers[i:i+1],
                    metric=metric
                ).flatten()
                closest_idx = cluster_points[np.argmin(distances)]
                landmark_indices.append(closest_idx)
            else:
                # 如果簇为空，随机选择一个点
                landmark_indices.append(np.random.choice(n_samples))
        
        return np.array(landmark_indices)
    
    elif strategy == 'maxmin':
        logger.info(f"使用MaxMin采样选择 {n_landmarks} 个地标")
        # 最远点采样
        landmark_indices = []
        
        # 第一个地标：随机选择
        current_idx = np.random.randint(n_samples)
        landmark_indices.append(current_idx)
        
        # 计算距离矩阵（只计算必要的部分）
        distances = pairwise_distances(X, metric=metric)
        
        for _ in range(n_landmarks - 1):
            # 找到离已有地标最远的点
            min_dists_to_landmarks = np.min(
                distances[np.ix_(np.arange(n_samples), landmark_indices)],
                axis=1
            )
            # 排除已有的地标
            min_dists_to_landmarks[landmark_indices] = -1
            
            next_idx = np.argmax(min_dists_to_landmarks)
            landmark_indices.append(next_idx)
        
        return np.array(landmark_indices)
    
    else:
        raise ValueError(f"未知的地标选择策略: {strategy}")


def verify_sampling_stability(
    X: np.ndarray,
    n_landmarks: int,
    strategy: str = 'kmeans',
    metric: str = 'euclidean',
    n_runs: int = 3,
    similarity_threshold: float = 0.7
) -> Tuple[bool, float, List[np.ndarray]]:
    """
    验证采样稳定性：多次采样并检查一致性
    
    Args:
        X: 嵌入矩阵
        n_landmarks: 地标数量
        strategy: 采样策略
        metric: 距离度量
        n_runs: 重复采样次数
        similarity_threshold: 相似度阈值（Jaccard相似度）
        
    Returns:
        (是否稳定, 平均相似度, 所有采样结果列表)
    """
    logger.info(f"验证采样稳定性（n_runs={n_runs}）")
    
    samples = []
    for i in range(n_runs):
        # 使用不同的随机种子
        np.random.seed(42 + i)
        sample = select_landmarks(X, n_landmarks, strategy, metric)
        samples.append(sample)
    
    # 计算两两之间的Jaccard相似度
    similarities = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            set_i = set(samples[i])
            set_j = set(samples[j])
            intersection = len(set_i & set_j)
            union = len(set_i | set_j)
            jaccard = intersection / union if union > 0 else 0.0
            similarities.append(jaccard)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    is_stable = avg_similarity >= similarity_threshold
    
    logger.info(f"采样稳定性: 平均相似度={avg_similarity:.4f}, 稳定={is_stable}")
    
    return is_stable, avg_similarity, samples


def select_landmarks_stable(
    X: np.ndarray,
    n_landmarks: int,
    config: Dict,
    verify_stability: bool = True
) -> np.ndarray:
    """
    选择地标点（带稳定性验证）
    
    Args:
        X: 嵌入矩阵
        n_landmarks: 地标数量
        config: 配置字典
        verify_stability: 是否验证稳定性
        
    Returns:
        地标索引数组
    """
    strategy = config.get('tda', {}).get('landmark_strategy', 'kmeans')
    metric = config.get('tda', {}).get('metric', 'euclidean')
    
    if verify_stability:
        n_runs = config.get('tda', {}).get('stability_n_runs', 3)
        is_stable, avg_sim, samples = verify_sampling_stability(
            X, n_landmarks, strategy, metric, n_runs
        )
        
        if not is_stable:
            logger.warning(f"采样不稳定（相似度={avg_sim:.4f}），使用第一次采样结果")
        
        # 返回第一次采样结果
        return samples[0] if samples else select_landmarks(X, n_landmarks, strategy, metric)
    else:
        return select_landmarks(X, n_landmarks, strategy, metric)


def compute_persistence(
    X: np.ndarray,
    max_dim: int = 1,
    metric: str = 'euclidean'
) -> Dict[str, Any]:
    """
    计算持续同调
    
    Args:
        X: 点云矩阵 (N, d)
        max_dim: 最大同调维数
        metric: 距离度量
        
    Returns:
        包含条码图、余循环等的字典
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("ripser不可用，无法计算持续同调")
    
    logger.info(f"计算持续同调 (max_dim={max_dim}, metric={metric})")
    
    # 计算Vietoris-Rips复形的持续同调
    result = ripser(
        X,
        maxdim=max_dim,
        metric=metric,
        coeff=2,
        do_cocycles=True
    )
    
    dgms = result['dgms']
    cocycles = result['cocycles']
    
    return {
        'dgms': dgms,
        'cocycles': cocycles,
        'num_edges': result.get('num_edges', 0)
    }


def find_top_persistent_cycle(
    dgms: List[np.ndarray],
    cocycles: Dict,
    persistence_threshold: float = 0.05,
    dim: int = 1
) -> Optional[Tuple[float, float, float, Any]]:
    """
    找到持久度最高的循环
    
    Args:
        dgms: 条码图列表
        cocycles: 余循环字典
        persistence_threshold: 持久度阈值
        dim: 同调维数
        
    Returns:
        (birth, death, persistence, cocycle) 或 None
    """
    if dim >= len(dgms) or dim < 0:
        return None
    
    diagram = dgms[dim]
    
    if len(diagram) == 0:
        logger.info(f"未检测到 {dim} 维同调类")
        return None
    
    # 找到持久度最高的循环
    # 对于有限循环，death值可能为inf
    finite_mask = np.isfinite(diagram[:, 1])
    
    if np.any(finite_mask):
        finite_diagram = diagram[finite_mask]
        persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
        top_idx = np.argmax(persistences)
        birth, death = finite_diagram[top_idx]
        persistence = persistences[top_idx]
        
        if persistence >= persistence_threshold:
            # 获取对应的余循环
            cocycle_key = (dim, top_idx)
            cocycle = cocycles.get(cocycle_key)
            return (birth, death, persistence, cocycle)
    
    return None


def extract_top_k_cycles(
    dgms: List[np.ndarray],
    cocycles: Dict,
    k: int = 5,
    persistence_threshold: float = 0.05,
    dim: int = 1
) -> List[Tuple[float, float, float, Any, int]]:
    """
    提取前K个持久度最高的循环
    
    创新点：不仅提取Top 1，而是提取多个显著环用于语义纯度分析
    
    Args:
        dgms: 条码图列表
        cocycles: 余循环字典
        k: 提取的环数量
        persistence_threshold: 持久度阈值
        dim: 同调维数
        
    Returns:
        [(birth, death, persistence, cocycle, cycle_index), ...] 列表，按持久度降序排序
    """
    if dim >= len(dgms) or dim < 0:
        return []
    
    diagram = dgms[dim]
    
    if len(diagram) == 0:
        logger.info(f"未检测到 {dim} 维同调类")
        return []
    
    # 找到所有有限循环的持久度
    finite_mask = np.isfinite(diagram[:, 1])
    
    if not np.any(finite_mask):
        return []
    
    finite_diagram = diagram[finite_mask]
    persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
    
    # 过滤低于阈值的循环
    valid_mask = persistences >= persistence_threshold
    if not np.any(valid_mask):
        logger.info(f"没有循环满足持久度阈值 {persistence_threshold}")
        return []
    
    valid_indices = np.where(valid_mask)[0]
    valid_persistences = persistences[valid_indices]
    
    # 按持久度降序排序，取前k个
    sorted_indices = np.argsort(valid_persistences)[::-1][:k]
    
    cycles = []
    for sorted_idx in sorted_indices:
        orig_idx = valid_indices[sorted_idx]
        birth, death = finite_diagram[orig_idx]
        persistence = valid_persistences[sorted_idx]
        
        # 获取对应的余循环
        # 注意：需要找到原始索引
        finite_to_original = np.where(finite_mask)[0]
        original_idx = finite_to_original[orig_idx]
        
        cocycle_key = (dim, original_idx)
        cocycle = cocycles.get(cocycle_key)
        
        cycles.append((birth, death, persistence, cocycle, original_idx))
    
    logger.info(f"提取了 {len(cycles)} 个显著环（持久度 >= {persistence_threshold}）")
    
    return cycles


def extract_cycle_words(
    cocycle: Any,
    landmark_indices: np.ndarray,
    labels: np.ndarray,
    X: np.ndarray
) -> List[str]:
    """
    从余循环中提取边界词
    
    Args:
        cocycle: 余循环
        landmark_indices: 地标索引
        labels: 词标签数组
        X: 原始嵌入矩阵
        
    Returns:
        边界词列表
    """
    if cocycle is None or len(cocycle) == 0:
        return []
    
    # 提取边（cocycle通常是sparse矩阵或列表）
    edges = set()
    
    if hasattr(cocycle, 'todense'):
        # 稀疏矩阵
        cocycle_dense = cocycle.todense()
        edge_indices = np.nonzero(cocycle_dense)
        for i, j in zip(edge_indices[0], edge_indices[1]):
            if i < j:  # 避免重复
                edges.add((i, j))
    elif isinstance(cocycle, (list, tuple)):
        # 列表格式
        for item in cocycle:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                i, j = item[0], item[1]
                if i < j:
                    edges.add((i, j))
    
    # 收集所有出现在边上的地标索引对应的词
    cycle_words = set()
    for i, j in edges:
        if i < len(landmark_indices):
            idx_i = landmark_indices[i]
            if idx_i < len(labels):
                cycle_words.add(labels[idx_i])
        if j < len(landmark_indices):
            idx_j = landmark_indices[j]
            if idx_j < len(labels):
                cycle_words.add(labels[idx_j])
    
    return list(cycle_words)


def analyze_multiple_cycles(
    embeddings_path: Path,
    output_dir: Path,
    config: Dict,
    top_k: int = 5
) -> Tuple[Path, List[Tuple[List[str], float, float, float]], Dict]:
    """
    执行拓扑数据分析，提取多个显著环
    
    Args:
        embeddings_path: 嵌入文件路径（NPZ格式）
        output_dir: 输出目录
        config: 配置字典
        top_k: 提取的环数量
        
    Returns:
        (输出文件路径, [(词列表, birth, death, persistence), ...], 元数据字典)
    """
    embeddings_path = Path(embeddings_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"分析嵌入文件（多环模式）: {embeddings_path}")
    
    # 加载嵌入
    data = np.load(embeddings_path, allow_pickle=True)
    X = data['X']
    labels = data['labels']
    
    logger.info(f"嵌入形状: {X.shape}, 标签数量: {len(labels)}")
    
    # TDA配置
    tda_config = config.get('tda', {})
    n_landmarks = tda_config.get('n_landmarks', 512)
    strategy = tda_config.get('landmark_strategy', 'kmeans')
    metric = tda_config.get('metric', 'euclidean')
    persistence_threshold = tda_config.get('persistence_threshold', 0.05)
    max_dim = tda_config.get('max_dim', 1)
    
    # 选择地标（带稳定性验证）
    verify_stability = config.get('tda', {}).get('verify_stability', True)
    landmark_indices = select_landmarks_stable(X, n_landmarks, config, verify_stability)
    X_landmarks = X[landmark_indices]
    
    logger.info(f"选择了 {len(landmark_indices)} 个地标")
    
    # 计算持续同调
    tda_result = compute_persistence(X_landmarks, max_dim=max_dim, metric=metric)
    dgms = tda_result['dgms']
    cocycles = tda_result['cocycles']
    
    # 提取前K个显著环
    cycles = extract_top_k_cycles(
        dgms,
        cocycles,
        k=top_k,
        persistence_threshold=persistence_threshold,
        dim=1  # β1
    )
    
    if not cycles:
        logger.warning("未找到显著的β1循环")
        cycles_with_words = []
        metadata = {
            'num_cycles': 0,
            'landmark_indices': landmark_indices.tolist()
        }
    else:
        # 为每个环提取边界词
        cycles_with_words = []
        for birth, death, persistence, cocycle, cycle_idx in cycles:
            cycle_words = extract_cycle_words(
                cocycle,
                landmark_indices,
                labels,
                X
            )
            cycles_with_words.append((cycle_words, birth, death, persistence))
        
        metadata = {
            'num_cycles': len(cycles_with_words),
            'landmark_indices': landmark_indices.tolist(),
            'cycles_info': [
                {'birth': b, 'death': d, 'persistence': p, 'size': len(w)}
                for w, b, d, p in cycles_with_words
            ]
        }
    
    # 保存结果
    base_name = embeddings_path.stem.replace('_embeddings', '')
    safe_name = safe_filename(base_name)
    output_path = output_dir / f"{safe_name}_beta1_multiple.npy"
    
    result_dict = {
        'dgms': dgms,
        'cocycles': cocycles,
        'landmark_indices': landmark_indices,
        'cycles': cycles_with_words,
        'metadata': metadata
    }
    
    np.save(output_path, result_dict, allow_pickle=True)
    logger.info(f"保存多环TDA结果: {output_path}")
    
    return output_path, cycles_with_words, metadata


def analyze(
    embeddings_path: Path,
    output_dir: Path,
    config: Dict
) -> Tuple[Path, List[str], Tuple[float, float, float]]:
    """
    执行拓扑数据分析
    
    Args:
        embeddings_path: 嵌入文件路径（NPZ格式）
        output_dir: 输出目录
        config: 配置字典
        
    Returns:
        (输出文件路径, 边界词列表, (birth, death, persistence))
    """
    embeddings_path = Path(embeddings_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"分析嵌入文件: {embeddings_path}")
    
    # 加载嵌入
    data = np.load(embeddings_path, allow_pickle=True)
    X = data['X']
    labels = data['labels']
    
    logger.info(f"嵌入形状: {X.shape}, 标签数量: {len(labels)}")
    
    # TDA配置
    tda_config = config.get('tda', {})
    n_landmarks = tda_config.get('n_landmarks', 512)
    strategy = tda_config.get('landmark_strategy', 'kmeans')
    metric = tda_config.get('metric', 'euclidean')
    persistence_threshold = tda_config.get('persistence_threshold', 0.05)
    max_dim = tda_config.get('max_dim', 1)
    
    # 选择地标（带稳定性验证）
    verify_stability = config.get('tda', {}).get('verify_stability', True)
    landmark_indices = select_landmarks_stable(X, n_landmarks, config, verify_stability)
    X_landmarks = X[landmark_indices]
    
    logger.info(f"选择了 {len(landmark_indices)} 个地标")
    
    # 计算持续同调
    tda_result = compute_persistence(X_landmarks, max_dim=max_dim, metric=metric)
    dgms = tda_result['dgms']
    cocycles = tda_result['cocycles']
    
    # 找到最持久的循环
    cycle_result = find_top_persistent_cycle(
        dgms,
        cocycles,
        persistence_threshold,
        dim=1  # β1
    )
    
    if cycle_result is None:
        logger.warning("未找到显著的β1循环")
        birth, death, persistence = 0.0, 0.0, 0.0
        cycle_words = []
    else:
        birth, death, persistence, cocycle = cycle_result
        logger.info(f"找到β1循环: birth={birth:.6f}, death={death:.6f}, persistence={persistence:.6f}")
        
        # 提取边界词
        cycle_words = extract_cycle_words(
            cocycle,
            landmark_indices,
            labels,
            X
        )
        logger.info(f"提取了 {len(cycle_words)} 个边界词")
    
    # 保存结果
    base_name = embeddings_path.stem.replace('_embeddings', '')
    safe_name = safe_filename(base_name)
    output_path = output_dir / f"{safe_name}_beta1.npy"
    
    result_dict = {
        'dgms': dgms,
        'cocycles': cocycles,
        'landmark_indices': landmark_indices,
        'birth': birth,
        'death': death,
        'persistence': persistence,
        'cycle_words': cycle_words,
        'top_bar': (birth, death, persistence) if cycle_result else None,
        'edges': None,  # 可以后续扩展存储边信息
    }
    
    np.save(output_path, result_dict, allow_pickle=True)
    logger.info(f"保存TDA结果: {output_path}")
    
    return output_path, cycle_words, (birth, death, persistence)

