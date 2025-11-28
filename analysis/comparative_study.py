"""对比实验模块 - 打乱组、对照组生成与分析"""
import logging
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import re

from src.topology import analyze, analyze_multiple_cycles
from src.embedder import embed_text_file
from src.utils import load_config, get_path, get_project_root, ensure_dir


logger = logging.getLogger(__name__)


def generate_shuffled_text(text: str, preserve_sentences: bool = False) -> str:
    """
    生成词序打乱的文本（用于验证结构的真实性）
    
    Args:
        text: 原始文本
        preserve_sentences: 是否在句子级别打乱（True）还是在词级别打乱（False）
        
    Returns:
        打乱后的文本
    """
    if preserve_sentences:
        # 句子级别打乱：保持句子内词序不变，只打乱句子顺序
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        random.shuffle(sentences)
        shuffled_text = '. '.join(sentences)
        if sentences:
            shuffled_text += '.'
    else:
        # 词级别打乱：完全随机打乱所有词的顺序
        words = re.findall(r'\b\S+\b', text)
        random.shuffle(words)
        shuffled_text = ' '.join(words)
    
    logger.info(f"生成打乱文本（句子级别={preserve_sentences}），原始长度: {len(text)}, 打乱后长度: {len(shuffled_text)}")
    
    return shuffled_text


def analyze_shuffled_text(
    original_text_path: Path,
    output_dir: Path,
    config: Dict,
    preserve_sentences: bool = False,
    seed: int = 42
) -> Tuple[Path, Dict[str, Any]]:
    """
    分析打乱后的文本，验证拓扑结构的真实性
    
    Args:
        original_text_path: 原始文本路径
        output_dir: 输出目录
        config: 配置字典
        preserve_sentences: 是否在句子级别打乱
        seed: 随机种子
        
    Returns:
        (嵌入文件路径, 分析结果字典)
    """
    logger.info(f"分析打乱文本: {original_text_path}")
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 读取原始文本
    with open(original_text_path, 'r', encoding='utf-8') as f:
        original_text = f.read()
    
    # 生成打乱文本
    shuffled_text = generate_shuffled_text(original_text, preserve_sentences)
    
    # 保存打乱文本
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    base_name = original_text_path.stem
    shuffled_text_path = output_dir / f"{base_name}_shuffled.txt"
    
    with open(shuffled_text_path, 'w', encoding='utf-8') as f:
        f.write(shuffled_text)
    
    logger.info(f"保存打乱文本: {shuffled_text_path}")
    
    # 生成嵌入
    embeddings_dir = get_path(config, 'data.embeddings_dir', get_project_root())
    emb_path = embed_text_file(shuffled_text_path, embeddings_dir, config)
    
    # 执行TDA分析
    tda_dir = get_path(config, 'data.tda_dir', get_project_root())
    tda_path, cycle_words, metrics = analyze(emb_path, tda_dir, config)
    
    result = {
        'shuffled_text_path': shuffled_text_path,
        'embeddings_path': emb_path,
        'tda_path': tda_path,
        'cycle_words': cycle_words,
        'metrics': metrics,
        'preserve_sentences': preserve_sentences
    }
    
    return emb_path, result


def compare_with_control(
    target_text_path: Path,
    control_text_path: Path,
    config: Dict
) -> Dict[str, Any]:
    """
    对比分析目标文本与对照组文本
    
    Args:
        target_text_path: 目标文本路径（如DeLillo作品）
        control_text_path: 对照组文本路径（如海明威作品）
        config: 配置字典
        
    Returns:
        对比结果字典
    """
    logger.info(f"对比分析: {target_text_path.name} vs {control_text_path.name}")
    
    # 分析目标文本
    logger.info("分析目标文本...")
    target_emb_dir = get_path(config, 'data.embeddings_dir', get_project_root())
    target_tda_dir = get_path(config, 'data.tda_dir', get_project_root())
    
    target_emb_path = embed_text_file(target_text_path, target_emb_dir, config)
    target_tda_path, target_cycles, target_metrics = analyze(target_emb_path, target_tda_dir, config)
    
    # 分析对照组文本
    logger.info("分析对照组文本...")
    control_emb_path = embed_text_file(control_text_path, target_emb_dir, config)
    control_tda_path, control_cycles, control_metrics = analyze(control_emb_path, target_tda_dir, config)
    
    # 计算对比指标
    target_persistence = target_metrics[2]
    control_persistence = control_metrics[2]
    target_cycle_count = len(target_cycles)
    control_cycle_count = len(control_cycles)
    
    comparison = {
        'target': {
            'text_path': target_text_path,
            'embeddings_path': target_emb_path,
            'tda_path': target_tda_path,
            'cycle_words': target_cycles,
            'persistence': target_persistence,
            'cycle_count': target_cycle_count,
            'metrics': target_metrics
        },
        'control': {
            'text_path': control_text_path,
            'embeddings_path': control_emb_path,
            'tda_path': control_tda_path,
            'cycle_words': control_cycles,
            'persistence': control_persistence,
            'cycle_count': control_cycle_count,
            'metrics': control_metrics
        },
        'difference': {
            'persistence_diff': target_persistence - control_persistence,
            'cycle_count_diff': target_cycle_count - control_cycle_count,
            'persistence_ratio': target_persistence / control_persistence if control_persistence > 0 else float('inf')
        }
    }
    
    logger.info(f"对比结果: 持久度差异={comparison['difference']['persistence_diff']:.4f}, "
                f"环数量差异={comparison['difference']['cycle_count_diff']}")
    
    return comparison


def statistical_significance_test(
    original_results: List[Dict[str, Any]],
    shuffled_results: List[Dict[str, Any]],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    统计显著性检验：比较原始文本与打乱文本的拓扑特征
    
    Args:
        original_results: 原始文本分析结果列表
        shuffled_results: 打乱文本分析结果列表
        alpha: 显著性水平
        
    Returns:
        检验结果字典
    """
    from scipy import stats
    
    try:
        # 提取持久度
        original_persistences = [r['metrics'][2] for r in original_results if r['metrics'][2] > 0]
        shuffled_persistences = [r['metrics'][2] for r in shuffled_results if r['metrics'][2] > 0]
        
        if len(original_persistences) < 2 or len(shuffled_persistences) < 2:
            logger.warning("样本量不足，无法进行统计检验")
            return {
                'significant': False,
                'test_statistic': None,
                'p_value': None,
                'mean_original': np.mean(original_persistences) if original_persistences else 0,
                'mean_shuffled': np.mean(shuffled_persistences) if shuffled_persistences else 0
            }
        
        # t检验
        t_stat, p_value = stats.ttest_ind(original_persistences, shuffled_persistences)
        
        significant = p_value < alpha
        
        result = {
            'significant': significant,
            'test_statistic': float(t_stat),
            'p_value': float(p_value),
            'alpha': alpha,
            'mean_original': float(np.mean(original_persistences)),
            'mean_shuffled': float(np.mean(shuffled_persistences)),
            'std_original': float(np.std(original_persistences)),
            'std_shuffled': float(np.std(shuffled_persistences))
        }
        
        logger.info(f"统计检验结果: p={p_value:.4f}, 显著={significant} (α={alpha})")
        
        return result
        
    except ImportError:
        logger.warning("scipy不可用，无法进行统计检验")
        return {
            'significant': None,
            'error': 'scipy not available'
        }
    except Exception as e:
        logger.error(f"统计检验失败: {e}")
        return {
            'significant': None,
            'error': str(e)
        }


def generate_comparison_report(
    comparison_results: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    生成对比分析报告
    
    Args:
        comparison_results: 对比结果字典
        output_path: 输出文件路径
        
    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    target = comparison_results['target']
    control = comparison_results['control']
    diff = comparison_results['difference']
    
    report = f"""# 对比分析报告

## 目标文本: {target['text_path'].name}
- 持久度: {target['persistence']:.6f}
- 显著环数量: {target['cycle_count']}
- 边界词样本: {', '.join(target['cycle_words'][:20])}

## 对照组文本: {control['text_path'].name}
- 持久度: {control['persistence']:.6f}
- 显著环数量: {control['cycle_count']}
- 边界词样本: {', '.join(control['cycle_words'][:20])}

## 差异分析
- 持久度差异: {diff['persistence_diff']:.6f}
- 持久度比率: {diff['persistence_ratio']:.4f}
- 环数量差异: {diff['cycle_count_diff']}

## 结论
"""
    
    if diff['persistence_diff'] > 0:
        report += f"目标文本的拓扑结构更显著（持久度高出 {diff['persistence_diff']:.4f}）。\n"
    else:
        report += f"对照组文本的拓扑结构更显著。\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"生成对比报告: {output_path}")
    
    return output_path

