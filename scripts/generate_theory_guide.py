"""生成理论构建辅助文档"""
import logging
from pathlib import Path

from src.utils import load_config, setup_logging, get_path, get_project_root
from src.theory_builder import (
    generate_search_guide,
    create_theory_template,
    generate_ai_search_prompts
)


logger = logging.getLogger(__name__)


def main():
    """生成理论构建辅助文档"""
    config = load_config()
    setup_logging()
    
    root = get_project_root()
    results_dir = get_path(config, 'data.results_dir', root)
    
    logger.info("生成理论构建辅助文档")
    
    # 生成文献检索指南
    search_guide_path = results_dir / 'theory_search_guide.md'
    generate_search_guide(search_guide_path)
    
    # 生成AI检索提示词
    ai_prompts_path = results_dir / 'ai_search_prompts.md'
    generate_ai_search_prompts(ai_prompts_path)
    
    # 创建理论观点整理模板
    template_dir = results_dir / 'theory_notes'
    template_dir.mkdir(exist_ok=True)
    template_path = template_dir / 'theory_template.md'
    create_theory_template(template_path)
    
    logger.info("理论构建辅助文档已生成：")
    logger.info(f"  - 文献检索指南: {search_guide_path}")
    logger.info(f"  - AI检索提示词: {ai_prompts_path}")
    logger.info(f"  - 理论观点模板: {template_path}")


if __name__ == '__main__':
    main()

