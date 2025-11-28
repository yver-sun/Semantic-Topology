"""批量运行TDA和Mapper分析"""
import logging
from pathlib import Path

from src.utils import load_config, setup_logging, get_path, get_project_root, ensure_dir
from src.topology import analyze
from src.visualizer import visualize


logger = logging.getLogger(__name__)


def main():
    """主函数 - 对嵌入目录中的所有文件进行TDA和可视化分析"""
    # 加载配置
    config = load_config()
    
    # 设置日志
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper())
    setup_logging(level=log_level)
    
    logger.info("批量运行TDA和Mapper分析")
    
    # 获取路径
    root = get_project_root()
    emb_dir = get_path(config, 'data.embeddings_dir', root)
    tda_dir = get_path(config, 'data.tda_dir', root)
    map_dir = get_path(config, 'data.mapper_dir', root)
    results_dir = get_path(config, 'data.results_dir', root)
    
    ensure_dir(tda_dir)
    ensure_dir(map_dir)
    ensure_dir(results_dir)
    
    res_md = results_dir / 'analysis.md'
    
    # 处理所有嵌入文件
    for name in sorted(emb_dir.glob('*_embeddings.npz')):
        if not name.name.endswith('_embeddings.npz'):
            continue
        
        logger.info(f"处理: {name.name}")
        
        # TDA分析
        try:
            tda_path, cycle_words, metrics = analyze(name, tda_dir, config)
            logger.info(f"TDA分析成功")
        except Exception as e:
            tda_path, cycle_words, metrics = Path(""), [], (0.0, 0.0, 0.0)
            logger.error(f"TDA分析失败: {e}", exc_info=True)
        
        # 可视化
        try:
            html_path, summary_path = visualize(name, map_dir, config)
            logger.info(f"可视化成功")
        except Exception as e:
            html_path, summary_path = None, None
            logger.error(f"可视化失败: {e}", exc_info=True)
        
        # 写入结果
        base_name = name.stem.replace('_embeddings', '')
        try:
            with open(res_md, 'a', encoding='utf-8') as md:
                md.write(f"\n## {base_name} 分析结果\n")
                md.write(f"- 嵌入文件：{name}\n")
                if tda_path:
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
            logger.error(f"写入结果失败: {e}")


if __name__ == '__main__':
    main()
