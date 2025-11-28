"""数据加载模块 - PDF/EPUB文本提取，支持OCR回退"""
import os
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import ebooklib
from ebooklib import epub

try:
    from pdfminer.high_level import extract_text as pdf_extract
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logging.warning("pdfminer.six not available, PDF extraction will fail")

try:
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logging.warning("Tesseract/PIL not available, OCR fallback disabled")

from .utils import get_path, ensure_dir, safe_filename, get_project_root


logger = logging.getLogger(__name__)


def extract_pdf(pdf_path: Path, config: Dict[str, Any]) -> Optional[str]:
    """
    从PDF文件提取文本
    
    Args:
        pdf_path: PDF文件路径
        config: 配置字典
        
    Returns:
        提取的文本，如果失败则返回None
    """
    if not PDFMINER_AVAILABLE:
        logger.error("pdfminer.six不可用，无法提取PDF")
        return None
    
    try:
        logger.info(f"提取PDF: {pdf_path}")
        text = pdf_extract(str(pdf_path))
        
        if not text or len(text.strip()) < 100:
            logger.warning(f"PDF提取的文本过短，尝试OCR回退: {pdf_path}")
            if config.get('data', {}).get('enable_ocr_fallback', False):
                return extract_pdf_ocr(pdf_path, config)
        
        return clean_text(text, config)
    
    except Exception as e:
        logger.error(f"PDF提取失败: {pdf_path}, 错误: {e}")
        if config.get('data', {}).get('enable_ocr_fallback', False):
            logger.info(f"尝试OCR回退: {pdf_path}")
            return extract_pdf_ocr(pdf_path, config)
        return None


def extract_pdf_ocr(pdf_path: Path, config: Dict[str, Any]) -> Optional[str]:
    """
    使用OCR从PDF提取文本（回退方案）
    
    Args:
        pdf_path: PDF文件路径
        config: 配置字典
        
    Returns:
        提取的文本，如果失败则返回None
    """
    if not TESSERACT_AVAILABLE:
        logger.error("Tesseract不可用，无法进行OCR")
        return None
    
    try:
        logger.info(f"使用OCR提取PDF: {pdf_path}")
        
        # 配置Tesseract路径（如果指定）
        tesseract_path = config.get('data', {}).get('tesseract_path')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # 将PDF转换为图像
        images = convert_from_path(str(pdf_path))
        
        # 限制处理页数（避免过长时间）
        max_pages = 50
        images = images[:max_pages]
        
        texts = []
        for i, image in enumerate(images):
            logger.debug(f"OCR处理第 {i+1}/{len(images)} 页")
            text = pytesseract.image_to_string(image, lang='eng+chi_sim')
            texts.append(text)
        
        full_text = '\n'.join(texts)
        return clean_text(full_text, config)
    
    except Exception as e:
        logger.error(f"OCR提取失败: {pdf_path}, 错误: {e}")
        return None


def extract_epub(epub_path: Path, config: Dict[str, Any]) -> Optional[str]:
    """
    从EPUB文件提取文本
    
    Args:
        epub_path: EPUB文件路径
        config: 配置字典
        
    Returns:
        提取的文本，如果失败则返回None
    """
    try:
        logger.info(f"提取EPUB: {epub_path}")
        book = epub.read_epub(str(epub_path))
        
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8')
                # 简单的HTML标签清理
                text = re.sub(r'<[^>]+>', '', content)
                texts.append(text)
        
        full_text = '\n'.join(texts)
        return clean_text(full_text, config)
    
    except Exception as e:
        logger.error(f"EPUB提取失败: {epub_path}, 错误: {e}")
        return None


def clean_text(text: str, config: Dict[str, Any]) -> str:
    """
    清理文本
    
    Args:
        text: 原始文本
        config: 配置字典
        
    Returns:
        清理后的文本
    """
    extract_config = config.get('text_extraction', {})
    
    # 移除多余的空白字符
    if extract_config.get('cleanup', True):
        # 合并多个空格为单个空格
        text = re.sub(r' +', ' ', text)
        # 合并多个换行符
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除特殊字符（保留常见标点）
        text = re.sub(r'[^\w\s\.,;:!?\'"()\-—–]', ' ', text)
    
    # 处理换行符
    if not extract_config.get('preserve_linebreaks', False):
        text = text.replace('\n', ' ')
        text = re.sub(r' +', ' ', text)
    
    return text.strip()


def extract_from_file(file_path: Path, config: Dict[str, Any]) -> Optional[str]:
    """
    根据文件类型自动选择提取方法
    
    Args:
        file_path: 文件路径
        config: 配置字典
        
    Returns:
        提取的文本，如果失败则返回None
    """
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return extract_pdf(file_path, config)
    elif suffix == '.epub':
        return extract_epub(file_path, config)
    elif suffix == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return clean_text(f.read(), config)
        except Exception as e:
            logger.error(f"读取文本文件失败: {file_path}, 错误: {e}")
            return None
    else:
        logger.warning(f"不支持的文件类型: {suffix}")
        return None


def extract_all(root_dir: Path, output_dir: Path, config: Dict[str, Any]) -> List[Tuple[str, Path]]:
    """
    从根目录提取所有支持的文档，保存到输出目录
    
    Args:
        root_dir: 根目录路径
        output_dir: 输出目录路径
        config: 配置字典
        
    Returns:
        [(文件名, 输出路径), ...] 列表
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    
    extracted = []
    supported_extensions = {'.pdf', '.epub', '.txt'}
    max_sentences = config.get('text_extraction', {}).get('max_sentences', 0)
    
    # 搜索根目录下的所有支持文件
    for ext in supported_extensions:
        for file_path in root_dir.rglob(f'*{ext}'):
            logger.info(f"处理文件: {file_path}")
            
            # 跳过输出目录中的文件
            try:
                if file_path.resolve().is_relative_to(output_dir.resolve()):
                    continue
            except ValueError:
                pass
            
            text = extract_from_file(file_path, config)
            
            if text:
                # 限制句子数
                if max_sentences > 0:
                    sentences = re.split(r'[.!?]+\s+', text)
                    text = '. '.join(sentences[:max_sentences])
                    if sentences:
                        text += '.'
                
                # 生成输出文件名
                base_name = file_path.stem
                safe_name = safe_filename(base_name)
                output_path = output_dir / f"{safe_name}.txt"
                
                # 保存文本
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    extracted.append((base_name, output_path))
                    logger.info(f"成功提取: {base_name} -> {output_path}")
                except Exception as e:
                    logger.error(f"保存文件失败: {output_path}, 错误: {e}")
            else:
                logger.warning(f"提取失败: {file_path}")
    
    return extracted

