"""动态拓扑分析模块 - 滑动窗口拓扑、时间演变分析"""
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import re

from .topology import (
    select_landmarks, compute_persistence, extract_top_k_cycles,
    extract_cycle_words
)
from .embedder import embed_text
from .nlp_processor import extract_words_with_context, load_spacy_model
from .utils import ensure_dir


logger = logging.getLogger(__name__)


def sliding_window_slicing(
    text: str,
    window_size: int = 5000,
    step_size: int = 1000
) -> List[Tuple[int, int, str]]:
    """
    将文本切分为滑动窗口
    
    Args:
        text: 输入文本
        window_size: 窗口大小（词数）
        step_size: 步长（词数）
        
    Returns:
        [(窗口索引, 开始位置, 窗口文本), ...] 列表
    """
    # 简单的分词（按空格和标点）
    words = re.findall(r'\b\S+\b', text)
    
    if len(words) < window_size:
        logger.warning(f"文本长度 ({len(words)} 词) 小于窗口大小 ({window_size})")
        return [(0, 0, text)]
    
    windows = []
    idx = 0
    
    for start in range(0, len(words) - window_size + 1, step_size):
        end = start + window_size
        window_words = words[start:end]
        window_text = ' '.join(window_words)
        
        # 计算原始文本中的位置（近似）
        char_start = text.find(window_words[0]) if window_words else 0
        char_end = text.find(window_words[-1], char_start) + len(window_words[-1]) if window_words else len(text)
        
        windows.append((idx, char_start, window_text))
        idx += 1
        
        # 如果已经到达文本末尾，停止
        if end >= len(words):
            break
    
    logger.info(f"切分为 {len(windows)} 个滑动窗口（窗口大小={window_size}, 步长={step_size}）")
    
    return windows


def compute_window_topology(
    window_text: str,
    window_idx: int,
    config: Dict,
    nlp_model: Any = None
) -> Dict[str, Any]:
    """
    计算单个窗口的拓扑特征
    
    Args:
        window_text: 窗口文本
        window_idx: 窗口索引
        config: 配置字典
        nlp_model: Spacy模型（可选）
        
    Returns:
        包含拓扑特征的字典
    """
    logger.debug(f"计算窗口 {window_idx} 的拓扑特征")
    
    try:
        # 生成嵌入
        embeddings, words = embed_text(window_text, config, nlp_model)
        
        if len(embeddings) == 0 or len(words) == 0:
            return {
                'window_idx': window_idx,
                'max_persistence': 0.0,
                'num_cycles': 0,
                'top_cycle_size': 0,
                'status': 'empty'
            }
        
        # TDA配置
        tda_config = config.get('tda', {})
        n_landmarks = min(tda_config.get('n_landmarks', 512), len(embeddings))
        strategy = tda_config.get('landmark_strategy', 'kmeans')
        metric = tda_config.get('metric', 'euclidean')
        persistence_threshold = tda_config.get('persistence_threshold', 0.05)
        max_dim = tda_config.get('max_dim', 1)
        
        if n_landmarks < 10:
            return {
                'window_idx': window_idx,
                'max_persistence': 0.0,
                'num_cycles': 0,
                'top_cycle_size': 0,
                'status': 'insufficient_data'
            }
        
        # 选择地标
        from sklearn.metrics.pairwise import pairwise_distances
        from sklearn.cluster import KMeans
        
        if strategy == 'kmeans':
            kmeans = KMeans(n_clusters=n_landmarks, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            landmark_indices = []
            for i in range(n_landmarks):
                cluster_points = np.where(labels == i)[0]
                if len(cluster_points) > 0:
                    distances = pairwise_distances(
                        embeddings[cluster_points],
                        centers[i:i+1],
                        metric=metric
                    ).flatten()
                    closest_idx = cluster_points[np.argmin(distances)]
                    landmark_indices.append(closest_idx)
                else:
                    landmark_indices.append(np.random.choice(len(embeddings)))
            
            landmark_indices = np.array(landmark_indices)
        else:
            # 简单随机采样
            landmark_indices = np.random.choice(len(embeddings), n_landmarks, replace=False)
        
        X_landmarks = embeddings[landmark_indices]
        
        # 计算持续同调
        from ripser import ripser
        result = ripser(X_landmarks, maxdim=max_dim, metric=metric, coeff=2, do_cocycles=True)
        
        dgms = result['dgms']
        cocycles = result['cocycles']
        
        # 提取显著环
        dim = 1
        if dim < len(dgms):
            diagram = dgms[dim]
            finite_mask = np.isfinite(diagram[:, 1])
            
            if np.any(finite_mask):
                finite_diagram = diagram[finite_mask]
                persistences = finite_diagram[:, 1] - finite_diagram[:, 0]
                
                valid_mask = persistences >= persistence_threshold
                if np.any(valid_mask):
                    max_persistence = np.max(persistences[valid_mask])
                    num_cycles = np.sum(valid_mask)
                    
                    # 获取最大持久度环的大小
                    top_idx = np.argmax(persistences[valid_mask])
                    valid_indices = np.where(valid_mask)[0]
                    orig_idx = valid_indices[top_idx]
                    
                    finite_to_original = np.where(finite_mask)[0]
                    original_idx = finite_to_original[orig_idx]
                    
                    cocycle_key = (dim, original_idx)
                    cocycle = cocycles.get(cocycle_key)
                    
                    if cocycle is not None:
                        cycle_words = extract_cycle_words(
                            cocycle,
                            landmark_indices,
                            np.array(words),
                            embeddings
                        )
                        top_cycle_size = len(cycle_words)
                    else:
                        top_cycle_size = 0
                    
                    return {
                        'window_idx': window_idx,
                        'max_persistence': float(max_persistence),
                        'num_cycles': int(num_cycles),
                        'top_cycle_size': top_cycle_size,
                        'status': 'success'
                    }
        
        return {
            'window_idx': window_idx,
            'max_persistence': 0.0,
            'num_cycles': 0,
            'top_cycle_size': 0,
            'status': 'no_significant_cycles'
        }
        
    except Exception as e:
        logger.error(f"窗口 {window_idx} 拓扑计算失败: {e}")
        return {
            'window_idx': window_idx,
            'max_persistence': 0.0,
            'num_cycles': 0,
            'top_cycle_size': 0,
            'status': f'error: {str(e)}'
        }


def analyze_sliding_windows(
    text_path: Path,
    config: Dict,
    nlp_model: Any = None
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, str]]]:
    """
    对文本进行滑动窗口拓扑分析
    
    Args:
        text_path: 文本文件路径
        config: 配置字典
        nlp_model: Spacy模型（可选）
        
    Returns:
        (窗口拓扑结果列表, 窗口列表)
    """
    logger.info(f"滑动窗口分析: {text_path}")
    
    # 读取文本
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 配置
    dynamics_config = config.get('dynamics', {})
    window_size = dynamics_config.get('window_size', 5000)
    step_size = dynamics_config.get('step_size', 1000)
    
    # 切分窗口
    windows = sliding_window_slicing(text, window_size, step_size)
    
    # 计算每个窗口的拓扑
    results = []
    for window_idx, char_start, window_text in windows:
        result = compute_window_topology(window_text, window_idx, config, nlp_model)
        result['char_position'] = char_start
        results.append(result)
        
        if (window_idx + 1) % 10 == 0:
            logger.info(f"已处理 {window_idx + 1}/{len(windows)} 个窗口")
    
    logger.info(f"完成滑动窗口分析，共 {len(results)} 个窗口")
    
    return results, windows


def plot_evolution_curve(
    window_results: List[Dict[str, Any]],
    output_path: Path,
    title: str = "虚无演变曲线"
) -> Path:
    """
    绘制持久度演变曲线（虚无演变曲线）
    
    Args:
        window_results: 窗口拓扑结果列表
        output_path: 输出文件路径
        title: 图表标题
        
    Returns:
        输出文件路径
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        
        # 提取数据
        window_indices = [r['window_idx'] for r in window_results]
        max_persistences = [r['max_persistence'] for r in window_results]
        num_cycles = [r['num_cycles'] for r in window_results]
        
        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 子图1：最大持久度演变
        ax1.plot(window_indices, max_persistences, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_xlabel('窗口索引', fontsize=12)
        ax1.set_ylabel('最大持久度', fontsize=12)
        ax1.set_title(f'{title} - 最大持久度演变', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(left=0)
        
        # 子图2：显著环数量演变
        ax2.plot(window_indices, num_cycles, 'r-', linewidth=2, marker='s', markersize=4)
        ax2.set_xlabel('窗口索引', fontsize=12)
        ax2.set_ylabel('显著环数量', fontsize=12)
        ax2.set_title('显著环数量演变', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(left=0)
        
        plt.tight_layout()
        
        # 保存
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"保存演变曲线: {output_path}")
        
        return output_path
        
    except ImportError:
        logger.warning("matplotlib不可用，无法绘制演变曲线")
        return None
    except Exception as e:
        logger.error(f"绘制演变曲线失败: {e}")
        return None

