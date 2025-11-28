"""NLP处理模块 - 词性过滤、停用词、词频统计"""
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import re

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    STOP_WORDS = set()
    logging.warning("spacy not available, using basic tokenization")

from .utils import get_project_root
from typing import Any


logger = logging.getLogger(__name__)

# Spacy模型缓存
_nlp_model: Optional[Any] = None


def load_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """
    加载Spacy模型（带缓存）
    
    Args:
        model_name: Spacy模型名称
        
    Returns:
        Spacy nlp对象
    """
    global _nlp_model
    
    if not SPACY_AVAILABLE:
        raise ImportError("spacy不可用，请先安装: pip install spacy")
    
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load(model_name)
            logger.info(f"加载Spacy模型: {model_name}")
        except OSError:
            raise ImportError(
                f"Spacy模型 '{model_name}' 未安装。"
                f"请运行: python -m spacy download {model_name}"
            )
    
    return _nlp_model


def basic_tokenize(text: str) -> List[str]:
    """
    基础分词（当Spacy不可用时使用）
    
    Args:
        text: 输入文本
        
    Returns:
        词列表
    """
    # 简单的分词：按空格和标点分割
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return words


def filter_by_pos(
    text: str,
    nlp_model: Any,
    keep_pos: List[str],
    extra_stopwords: List[str] = None,
    min_word_length: int = 3
) -> List[Tuple[str, str]]:
    """
    根据词性过滤文本，返回(词, 词性)列表
    
    Args:
        text: 输入文本
        nlp_model: Spacy模型
        keep_pos: 保留的词性列表（如 ['NOUN', 'VERB']）
        extra_stopwords: 额外的停用词列表
        min_word_length: 最小词长度
        
    Returns:
        [(词, 词性), ...] 列表
    """
    if extra_stopwords is None:
        extra_stopwords = []
    
    stop_words = set(STOP_WORDS) | set(w.lower() for w in extra_stopwords)
    keep_pos_set = set(keep_pos)
    
    # 处理文本
    doc = nlp_model(text)
    
    filtered = []
    for token in doc:
        # 检查条件
        if (token.pos_ in keep_pos_set and
            not token.is_stop and
            token.text.lower() not in stop_words and
            len(token.text) >= min_word_length and
            token.is_alpha):
            filtered.append((token.text.lower(), token.pos_))
    
    return filtered


def count_word_frequency(
    words: List[Tuple[str, str]],
    min_freq: int = 1
) -> Dict[str, int]:
    """
    统计词频，过滤低频词
    
    Args:
        words: (词, 词性)列表
        min_freq: 最小词频
        
    Returns:
        词频字典 {词: 频次}
    """
    word_counter = Counter(word for word, _ in words)
    
    # 过滤低频词
    filtered = {
        word: count
        for word, count in word_counter.items()
        if count >= min_freq
    }
    
    return filtered


def process_text(
    text: str,
    config: Dict,
    nlp_model: Any = None
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """
    处理文本：分词、词性过滤、词频统计
    
    Args:
        text: 输入文本
        config: 配置字典
        nlp_model: Spacy模型（如果为None则自动加载）
        
    Returns:
        (所有词列表, 词标签列表, 词频字典)
    """
    nlp_config = config.get('nlp', {})
    
    # 加载模型
    if nlp_model is None:
        if SPACY_AVAILABLE:
            model_name = nlp_config.get('spacy_model', 'en_core_web_sm')
            nlp_model = load_spacy_model(model_name)
        else:
            # 回退到基础分词
            words = basic_tokenize(text)
            word_freq = Counter(words)
            min_freq = nlp_config.get('min_freq', 5)
            filtered_words = [w for w in words if word_freq[w] >= min_freq]
            labels = filtered_words.copy()
            freq_dict = {w: word_freq[w] for w in filtered_words if word_freq[w] >= min_freq}
            return filtered_words, labels, freq_dict
    
    # 检查是否启用依存句法分析
    use_dependency = nlp_config.get('use_dependency_analysis', True)
    
    # 词性+依存关系双重过滤
    filtered = filter_by_pos_and_dependency(
        text,
        nlp_model,
        config,
        use_dependency=use_dependency
    )
    
    # 提取所有词（用于嵌入）
    all_words = [word for word, _ in filtered]
    
    # 词频统计
    min_freq = nlp_config.get('min_freq', 5)
    word_freq = count_word_frequency(filtered, min_freq)
    
    # 过滤低频词
    filtered_words = [word for word in all_words if word_freq.get(word, 0) >= min_freq]
    
    # 生成标签（保持与词一一对应）
    labels = filtered_words.copy()
    
    return filtered_words, labels, word_freq


def extract_words_with_context(
    text: str,
    config: Dict,
    nlp_model: Any = None
) -> List[Tuple[str, str, int, int]]:
    """
    提取词及其上下文位置
    
    Args:
        text: 输入文本
        config: 配置字典
        nlp_model: Spacy模型
        
    Returns:
        [(词, 上下文, 开始位置, 结束位置), ...]
    """
    nlp_config = config.get('nlp', {})
    
    if nlp_model is None:
        model_name = nlp_config.get('spacy_model', 'en_core_web_sm')
        if SPACY_AVAILABLE:
            nlp_model = load_spacy_model(model_name)
        else:
            return []
    
    keep_pos = nlp_config.get('keep_pos', ['NOUN', 'PROPN', 'ADJ', 'VERB'])
    extra_stopwords = nlp_config.get('extra_stopwords', [])
    min_word_length = nlp_config.get('min_word_length', 3)
    min_freq = nlp_config.get('min_freq', 5)
    
    stop_words = set(STOP_WORDS) | set(w.lower() for w in extra_stopwords)
    keep_pos_set = set(keep_pos)
    
    # 检查是否启用依存句法分析
    use_dependency = nlp_config.get('use_dependency_analysis', True)
    
    doc = nlp_model(text)
    
    # 确定保留的依存关系或词性
    if use_dependency:
        keep_dep = nlp_config.get('keep_dependencies', None)
        if keep_dep is None:
            keep_dep = ['nsubj', 'nsubjpass', 'dobj', 'pobj', 'ROOT', 
                       'amod', 'acomp', 'attr', 'nmod', 'compound']
        keep_dep_set = set(keep_dep)
        keep_pos_set = None  # 不使用词性过滤，只使用依存关系
    else:
        keep_dep_set = None
        keep_pos_set = set(keep_pos)
    
    # 先统计词频
    word_counter = Counter()
    for token in doc:
        match = False
        if use_dependency:
            # 基于依存关系
            if (keep_dep_set and token.dep_ in keep_dep_set and
                not token.is_stop and
                token.text.lower() not in stop_words and
                len(token.text) >= min_word_length and
                token.is_alpha and
                token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']):
                match = True
        else:
            # 基于词性
            if (token.pos_ in keep_pos_set and
                not token.is_stop and
                token.text.lower() not in stop_words and
                len(token.text) >= min_word_length and
                token.is_alpha):
                match = True
        
        if match:
            word_counter[token.text.lower()] += 1
    
    # 提取词及其上下文
    words_with_context = []
    for token in doc:
        match = False
        if use_dependency:
            # 基于依存关系
            if (keep_dep_set and token.dep_ in keep_dep_set and
                not token.is_stop and
                token.text.lower() not in stop_words and
                len(token.text) >= min_word_length and
                token.is_alpha and
                token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']):
                match = True
        else:
            # 基于词性
            if (token.pos_ in keep_pos_set and
                not token.is_stop and
                token.text.lower() not in stop_words and
                len(token.text) >= min_word_length and
                token.is_alpha):
                match = True
        
        if match and word_counter[token.text.lower()] >= min_freq:
            word = token.text.lower()
            context_start = max(0, token.idx - 50)
            context_end = min(len(text), token.idx + len(token.text) + 50)
            context = text[context_start:context_end]
            
            words_with_context.append((
                word,
                context,
                token.idx,
                token.idx + len(token.text)
            ))
    
    return words_with_context


def extract_syntactic_core(
    text: str,
    nlp_model: Any,
    keep_dep: Optional[List[str]] = None,
    extra_stopwords: List[str] = None,
    min_word_length: int = 3
) -> List[Tuple[str, str, str]]:
    """
    基于依存句法分析提取语义核心词（主语、宾语、核心谓语、修饰性形容词）
    
    创新点：只提取承载语义核心的词汇，过滤叙述性噪音
    
    Args:
        text: 输入文本
        nlp_model: Spacy模型（必须支持依存分析）
        keep_dep: 保留的依存关系列表，默认保留：
            - nsubj: 主语
            - dobj: 直接宾语
            - pobj: 介词宾语
            - ROOT: 根节点（核心谓语）
            - amod: 形容词修饰语
            - acomp: 形容词补语
            - attr: 属性
        extra_stopwords: 额外的停用词列表
        min_word_length: 最小词长度
        
    Returns:
        [(词, 词性, 依存关系), ...] 列表
    """
    if not SPACY_AVAILABLE:
        logger.warning("Spacy不可用，无法进行依存句法分析")
        return []
    
    if keep_dep is None:
        # 默认保留的依存关系：主语、宾语、核心谓语、修饰性形容词
        keep_dep = ['nsubj', 'nsubjpass', 'dobj', 'pobj', 'ROOT', 
                   'amod', 'acomp', 'attr', 'nmod', 'compound']
    
    if extra_stopwords is None:
        extra_stopwords = []
    
    stop_words = set(STOP_WORDS) | set(w.lower() for w in extra_stopwords)
    keep_dep_set = set(keep_dep)
    
    # 扩展的文学叙述性噪音停用词
    literary_noise = {
        'said', 'says', 'say', 'told', 'tell', 'asked', 'ask',
        'looked', 'look', 'looks', 'saw', 'see', 'sees',
        'went', 'go', 'goes', 'came', 'come', 'comes',
        'thought', 'think', 'thinks', 'felt', 'feel', 'feels',
        'know', 'knew', 'knows', 'seemed', 'seem', 'seems'
    }
    stop_words.update(literary_noise)
    
    doc = nlp_model(text)
    
    filtered = []
    for token in doc:
        # 检查条件：
        # 1. 依存关系在保留列表中
        # 2. 不是停用词
        # 3. 满足最小长度
        # 4. 是字母字符
        # 5. 词性是实词（名词、动词、形容词）
        if (token.dep_ in keep_dep_set and
            not token.is_stop and
            token.text.lower() not in stop_words and
            len(token.text) >= min_word_length and
            token.is_alpha and
            token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV']):
            filtered.append((token.text.lower(), token.pos_, token.dep_))
    
    return filtered


def filter_by_pos_and_dependency(
    text: str,
    nlp_model: Any,
    config: Dict,
    use_dependency: bool = True
) -> List[Tuple[str, str]]:
    """
    词性+依存关系的双重过滤机制
    
    Args:
        text: 输入文本
        nlp_model: Spacy模型
        config: 配置字典
        use_dependency: 是否使用依存关系过滤（默认True）
        
    Returns:
        [(词, 词性), ...] 列表
    """
    nlp_config = config.get('nlp', {})
    
    if use_dependency:
        # 使用依存句法分析
        keep_dep = nlp_config.get('keep_dependencies', None)
        extra_stopwords = nlp_config.get('extra_stopwords', [])
        min_word_length = nlp_config.get('min_word_length', 3)
        
        syntactic_core = extract_syntactic_core(
            text, nlp_model, keep_dep, extra_stopwords, min_word_length
        )
        
        # 转换为(词, 词性)格式
        return [(word, pos) for word, pos, dep in syntactic_core]
    else:
        # 回退到原有的词性过滤
        keep_pos = nlp_config.get('keep_pos', ['NOUN', 'PROPN', 'ADJ', 'VERB'])
        extra_stopwords = nlp_config.get('extra_stopwords', [])
        min_word_length = nlp_config.get('min_word_length', 3)
        
        return filter_by_pos(
            text, nlp_model, keep_pos, extra_stopwords, min_word_length
        )

