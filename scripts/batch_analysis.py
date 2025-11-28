"""批量分析文本文件"""
import logging
from pathlib import Path

from src.utils import load_config, setup_logging, get_path, get_project_root, ensure_dir
from src.embedder import embed_text_file
from src.topology import analyze
from src.visualizer import visualize
from src.nlp_processor import load_spacy_model


logger = logging.getLogger(__name__)


def main():
    """主函数 - 批量处理文本文件"""
    # 加载配置
    config = load_config()
    
    # 设置日志
    log_level = getattr(logging, config.get('logging', {}).get('level', 'INFO').upper())
    setup_logging(level=log_level)
    
    logger.info("批量分析文本文件")
    
    # 获取路径
    root = get_project_root()
    texts_dir = get_path(config, 'data.texts_dir', root)
    emb_dir = get_path(config, 'data.embeddings_dir', root)
    tda_dir = get_path(config, 'data.tda_dir', root)
    map_dir = get_path(config, 'data.mapper_dir', root)
    results_dir = get_path(config, 'data.results_dir', root)
    
    ensure_dir(emb_dir)
    ensure_dir(tda_dir)
    ensure_dir(map_dir)
    ensure_dir(results_dir)
    
    res_md = results_dir / 'analysis.md'
    
    # 目标文本文件列表
    texts = [
        texts_dir / '2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library).txt',
        texts_dir / '2016 Zero K  a novel (DeLillo, Don, author) (Z-Library).txt',
        texts_dir / '2020 The Silence (Don DeLillo) (Z-Library).txt',
    ]
    
    # 加载Spacy模型
    nlp_model = None
    try:
        spacy_model_name = config.get('nlp', {}).get('spacy_model', 'en_core_web_sm')
        nlp_model = load_spacy_model(spacy_model_name)
    except Exception as e:
        logger.warning(f"无法加载Spacy模型: {e}")
    
    for txt in texts:
        if not txt.exists():
            logger.warning(f"文件不存在: {txt}")
            continue
        
        logger.info(f"处理: {txt.name}")
        
        # 生成嵌入
        try:
            emb = embed_text_file(txt, emb_dir, config, nlp_model)
            logger.info(f"嵌入文件: {emb}")
        except Exception as e:
            logger.error(f"生成嵌入失败: {e}", exc_info=True)
            continue
        
        # TDA分析
        try:
            tda_path, cycle_words, metrics = analyze(emb, tda_dir, config)
            logger.info(f"TDA分析成功")
        except Exception as e:
            tda_path, cycle_words, metrics = Path(""), [], (0.0, 0.0, 0.0)
            logger.error(f"TDA分析失败: {e}", exc_info=True)
        
        # 可视化
        try:
            html_path, summary_path = visualize(emb, map_dir, config)
            logger.info(f"可视化成功")
        except Exception as e:
            html_path, summary_path = None, None
            logger.error(f"可视化失败: {e}", exc_info=True)
        
        # 写入结果
        name = txt.stem
        try:
            with open(res_md, 'a', encoding='utf-8') as md:
                md.write(f"\n## {name} 分析结果\n")
                md.write(f"- 嵌入文件：{emb}\n")
                if tda_path:
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
