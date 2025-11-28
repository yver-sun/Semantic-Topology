"""主流程入口 - 完整的语义拓扑分析流水线"""
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

from src.utils import load_config, setup_logging, get_path, get_project_root, ensure_dir
from src.data_loader import extract_all
from src.embedder import embed_text_file
from src.topology import analyze
from src.visualizer import visualize
from src.nlp_processor import load_spacy_model


logger = logging.getLogger(__name__)


def write_analysis_result(
    md_path: Path,
    name: str,
    emb_path: Path,
    tda_path: Path,
    cycle_words: List[str],
    metrics: Tuple[float, float, float],
    html_path: Path = None,
    summary_path: Path = None
) -> None:
    """
    将分析结果写入Markdown文件
    
    Args:
        md_path: Markdown文件路径
        name: 文档名称
        emb_path: 嵌入文件路径
        tda_path: TDA结果文件路径
        cycle_words: 边界词列表
        metrics: (birth, death, persistence) 元组
        html_path: HTML可视化路径（可选）
        summary_path: 摘要文件路径（可选）
    """
    try:
        with open(md_path, 'a', encoding='utf-8') as md:
            md.write(f"\n## {name} 分析结果\n")
            md.write(f"- 嵌入文件：{emb_path}\n")
            md.write(f"- β1 结果文件：{tda_path}\n")
            b, d, p = metrics
            md.write(f"- β1 条码：birth={b:.6f}, death={d:.6f}, persistence={p:.6f}\n")
            if cycle_words:
                md.write(f"- 环边界词（样本≤40）：{', '.join(cycle_words[:40])}\n")
            else:
                md.write("- 当前尺度未检出显著 β1 环或词集为空。\n")
            if html_path:
                md.write(f"- Mapper 骨架：{html_path}\n")
            if summary_path:
                md.write(f"- Mapper 摘要：{summary_path}\n")
    except Exception as e:
        logger.error(f"写入分析结果失败: {e}")


def main():
    """主函数 - 执行完整的分析流水线"""
    # 加载配置
    config = load_config()
    
    # 设置日志
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper())
    log_file = config.get('logging', {}).get('log_file')
    setup_logging(level=log_level, log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("语义拓扑分析流水线启动")
    logger.info("=" * 60)
    
    # 获取路径
    root = get_project_root()
    data_config = config.get('data', {})
    
    texts_dir = get_path(config, 'data.texts_dir', root)
    embeds_dir = get_path(config, 'data.embeddings_dir', root)
    tda_dir = get_path(config, 'data.tda_dir', root)
    mapper_dir = get_path(config, 'data.mapper_dir', root)
    results_dir = get_path(config, 'data.results_dir', root)
    
    # 确保目录存在
    ensure_dir(texts_dir)
    ensure_dir(embeds_dir)
    ensure_dir(tda_dir)
    ensure_dir(mapper_dir)
    ensure_dir(results_dir)
    
    # 分析结果文件
    md_path = results_dir / 'analysis.md'
    
    # 初始化分析结果文件
    if not md_path.exists():
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# 语义拓扑分析结果\n\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 加载Spacy模型（提前加载以复用）
    nlp_model = None
    try:
        spacy_model_name = config.get('nlp', {}).get('spacy_model', 'en_core_web_sm')
        nlp_model = load_spacy_model(spacy_model_name)
        logger.info(f"已加载Spacy模型: {spacy_model_name}")
    except Exception as e:
        logger.warning(f"无法加载Spacy模型，将使用基础分词: {e}")
    
    # 步骤1: 文本提取
    logger.info("=" * 60)
    logger.info("步骤1: 文本提取")
    logger.info("=" * 60)
    
    extracted_files = extract_all(root, texts_dir, config)
    
    if not extracted_files:
        logger.warning("未找到可提取的文本文件")
        return
    
    logger.info(f"成功提取 {len(extracted_files)} 个文件")
    
    # 步骤2-4: 对每个文件进行处理
    for name, txt_path in extracted_files:
        logger.info("=" * 60)
        logger.info(f"处理文件: {name}")
        logger.info("=" * 60)
        
        try:
            # 步骤2: 生成嵌入
            logger.info("步骤2: 生成嵌入")
            emb_path = embed_text_file(txt_path, embeds_dir, config, nlp_model)
            logger.info(f"嵌入文件: {emb_path}")
            
            # 步骤3: 拓扑数据分析
            logger.info("步骤3: 拓扑数据分析")
            tda_path, cycle_words, metrics = analyze(emb_path, tda_dir, config)
            logger.info(f"β1 条码: birth={metrics[0]:.6f}, death={metrics[1]:.6f}, persistence={metrics[2]:.6f}")
            logger.info(f"边界词数量: {len(cycle_words)}")
            if cycle_words:
                logger.info(f"边界词样本: {', '.join(cycle_words[:10])}")
            
            # 步骤4: 可视化
            logger.info("步骤4: 生成可视化")
            html_path, summary_path = visualize(emb_path, mapper_dir, config)
            if html_path:
                logger.info(f"HTML可视化: {html_path}")
            if summary_path:
                logger.info(f"摘要文件: {summary_path}")
            
            # 写入分析结果
            write_analysis_result(
                md_path,
                name,
                emb_path,
                tda_path,
                cycle_words,
                metrics,
                html_path,
                summary_path
            )
            
            logger.info(f"✓ 完成: {name}")
            
        except Exception as e:
            logger.error(f"处理文件失败: {name}, 错误: {e}", exc_info=True)
            continue
    
    logger.info("=" * 60)
    logger.info("流水线完成")
    logger.info("=" * 60)
    logger.info(f"分析结果保存在: {md_path}")


if __name__ == '__main__':
    main()
