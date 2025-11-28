"""可视化模块 - Mapper骨架图生成、HTML输出"""
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

try:
    import kmapper as km
    KMAPPER_AVAILABLE = True
except ImportError:
    KMAPPER_AVAILABLE = False
    logging.warning("kmapper not available, visualization will fail")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("umap-learn not available, visualization will fail")

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available, visualization will fail")

from .utils import ensure_dir, safe_filename


logger = logging.getLogger(__name__)


def create_umap_lens(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    创建UMAP降维透镜
    
    Args:
        X: 嵌入矩阵 (N, d)
        n_neighbors: 邻居数量
        min_dist: 最小距离
        n_components: 降维后的维度
        random_state: 随机种子
        
    Returns:
        降维后的坐标 (N, n_components)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn不可用，无法创建UMAP透镜")
    
    logger.info(f"创建UMAP透镜 (n_neighbors={n_neighbors}, n_components={n_components})")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=random_state,
        metric='euclidean'
    )
    
    lens = reducer.fit_transform(X)
    
    return lens


def create_mapper_graph(
    X: np.ndarray,
    lens: np.ndarray,
    n_neighbors: int = 15,
    overlap: float = 0.5,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 10
) -> Any:
    """
    创建Mapper图
    
    Args:
        X: 嵌入矩阵 (N, d)
        lens: 透镜函数值 (N, d_lens)
        n_neighbors: 邻居数量
        overlap: 重叠比例
        dbscan_eps: DBSCAN eps参数
        dbscan_min_samples: DBSCAN min_samples参数
        
    Returns:
        Mapper图对象
    """
    if not KMAPPER_AVAILABLE:
        raise ImportError("kmapper不可用，无法创建Mapper图")
    
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn不可用，无法进行聚类")
    
    logger.info("创建Mapper图")
    
    # 创建Mapper对象
    mapper = km.KeplerMapper(verbose=0)
    
    # 定义聚类器
    clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    
    # 创建Mapper图
    graph = mapper.map(
        lens,
        X,
        clusterer=clusterer,
        cover=km.Cover(n_cubes=n_neighbors, perc_overlap=overlap),
        remove_duplicate_nodes=True
    )
    
    logger.info(f"Mapper图包含 {len(graph['nodes'])} 个节点")
    
    return graph, mapper


def generate_summary(graph: Dict, labels: np.ndarray) -> str:
    """
    生成Mapper图摘要
    
    Args:
        graph: Mapper图字典
        labels: 词标签数组
        
    Returns:
        摘要文本
    """
    nodes = graph['nodes']
    links = graph['links']
    
    summary_lines = [
        "Mapper图摘要",
        "=" * 50,
        f"节点数量: {len(nodes)}",
        f"连边数量: {len(links)}",
        "",
        "节点详情:",
        "-" * 50,
    ]
    
    # 统计每个节点的信息
    for node_id, node_indices in nodes.items():
        node_size = len(node_indices)
        if node_size > 0 and node_indices[0] < len(labels):
            sample_labels = [labels[i] for i in node_indices[:10] if i < len(labels)]
            sample_text = ', '.join(sample_labels[:5])
            if len(sample_labels) > 5:
                sample_text += f", ... (+{len(sample_labels)-5})"
        else:
            sample_text = "N/A"
        
        summary_lines.append(f"节点 {node_id}: {node_size} 个点")
        summary_lines.append(f"  样本词: {sample_text}")
    
    summary_lines.extend([
        "",
        "连边详情:",
        "-" * 50,
    ])
    
    for link in links[:20]:  # 只显示前20条边
        summary_lines.append(f"{link}")
    
    if len(links) > 20:
        summary_lines.append(f"... 还有 {len(links) - 20} 条边")
    
    return "\n".join(summary_lines)


def visualize(
    embeddings_path: Path,
    output_dir: Path,
    config: Dict
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    生成Mapper可视化
    
    Args:
        embeddings_path: 嵌入文件路径（NPZ格式）
        output_dir: 输出目录
        config: 配置字典
        
    Returns:
        (HTML文件路径, 摘要文件路径) 元组
    """
    embeddings_path = Path(embeddings_path)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    logger.info(f"生成可视化: {embeddings_path}")
    
    # 加载嵌入
    data = np.load(embeddings_path, allow_pickle=True)
    X = data['X']
    labels = data['labels']
    
    logger.info(f"嵌入形状: {X.shape}, 标签数量: {len(labels)}")
    
    # 可视化配置
    viz_config = config.get('visualization', {})
    
    # 创建UMAP透镜
    umap_n_neighbors = viz_config.get('umap_n_neighbors', 15)
    umap_min_dist = viz_config.get('umap_min_dist', 0.1)
    umap_n_components = viz_config.get('umap_n_components', 2)
    
    try:
        lens = create_umap_lens(
            X,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            n_components=umap_n_components
        )
    except Exception as e:
        logger.error(f"创建UMAP透镜失败: {e}")
        return None, None
    
    # 创建Mapper图
    mapper_neighbors = viz_config.get('mapper_neighbors', 15)
    mapper_overlap = viz_config.get('mapper_overlap', 0.5)
    dbscan_eps = viz_config.get('dbscan_eps', 0.5)
    dbscan_min_samples = viz_config.get('dbscan_min_samples', 10)
    
    try:
        graph, mapper = create_mapper_graph(
            X,
            lens,
            n_neighbors=mapper_neighbors,
            overlap=mapper_overlap,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples
        )
    except Exception as e:
        logger.error(f"创建Mapper图失败: {e}")
        return None, None
    
    # 生成输出文件名
    base_name = embeddings_path.stem.replace('_embeddings', '')
    safe_name = safe_filename(base_name)
    
    # 生成HTML可视化
    html_path = None
    if KMAPPER_AVAILABLE:
        html_path = output_dir / f"{safe_name}_mapper.html"
        
        try:
            html_str = mapper.visualize(
                graph,
                color_values=lens[:, 0],  # 使用第一个UMAP维度作为颜色
                color_function_name="UMAP维度1",
                title=f"Mapper骨架图: {base_name}",
                path_html=str(html_path),
                custom_tooltips=[str(l) for l in labels] if len(labels) == len(X) else None
            )
            logger.info(f"生成HTML可视化: {html_path}")
        except Exception as e:
            logger.error(f"生成HTML失败: {e}")
            html_path = None
    
    # 生成摘要
    summary_path = None
    if viz_config.get('generate_summary', True):
        summary_path = output_dir / f"{safe_name}_mapper_summary.txt"
        
        try:
            summary_text = generate_summary(graph, labels)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            logger.info(f"生成摘要: {summary_path}")
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            summary_path = None
    
    return html_path, summary_path

