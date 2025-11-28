"""理论构建辅助模块 - 文献检索指南、理论观点整理"""
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

from .utils import ensure_dir, get_project_root


logger = logging.getLogger(__name__)


# 预设的搜索提示词和关键词
SEARCH_QUERIES = {
    'english': [
        "How does Don DeLillo depict technology as a form of religion or theology?",
        "What is the relationship between the technological sublime and apophatic theology in contemporary literature?",
        "Does Don DeLillo's late fiction focus on silence and the void?",
        "DeLillo technology apophatic theology",
        "DeLillo late style minimalism theology",
        "technological sublime apophatic theology DeLillo",
        "DeLillo silence void technology"
    ],
    'chinese': [
        "唐·德利洛 技术崇高 否定神学",
        "德利洛 晚期作品 技术 神学",
        "德利洛 极简主义 虚无",
        "技术神学 德利洛"
    ],
    'keywords': {
        'english': [
            "Don DeLillo technology divine",
            "DeLillo technological sublime",
            "apophatic theology DeLillo",
            "DeLillo silence void",
            "late DeLillo minimalism",
            "DeLillo technology religion"
        ],
        'chinese': [
            "唐·德利洛 技术崇高",
            "否定神学 德利洛",
            "技术神学 德利洛",
            "德利洛 晚期作品",
            "极简主义 虚无"
        ]
    }
}

# 核心理论框架
THEORETICAL_FRAMEWORK = {
    'core_thesis': "德利洛晚期作品中的技术被描绘为一种新形式的否定神学，通过'虚无'的拓扑结构体现",
    
    'key_concepts': {
        'technological_sublime': {
            'definition': "技术崇高：技术带来的超越性体验",
            'keywords': ['technology', 'sublime', 'transcendence', 'screen', 'digital'],
            'theoretical_support': []
        },
        'apophatic_theology': {
            'definition': "否定神学：通过否定来接近神圣的不可言说性",
            'keywords': ['void', 'silence', 'nothingness', 'absence', 'negation'],
            'theoretical_support': []
        },
        'late_style': {
            'definition': "晚期风格：极简主义的语言形式，指向意义的缺失",
            'keywords': ['minimalism', 'sparse', 'reduction', 'absence'],
            'theoretical_support': []
        }
    },
    
    'research_questions': [
        "德利洛如何将技术描绘为一种神学形式？",
        "技术崇高与否定神学在当代文学中的关系是什么？",
        "德利洛晚期作品是否聚焦于沉默与虚无？",
        "拓扑结构如何体现'技术即神学'的主题？"
    ],
    
    'mainstream_views': {
        'consensus_1': {
            'view': "德利洛晚期作品展现了技术作为一种新宗教/神学形式",
            'supporters': [],
            'evidence': [],
            'citation': ""
        },
        'consensus_2': {
            'view': "晚期风格与极简主义指向意义的缺失和虚无",
            'supporters': [],
            'evidence': [],
            'citation': ""
        },
        'consensus_3': {
            'view': "技术设备（屏幕、数字界面）在德利洛作品中具有神学维度",
            'supporters': [],
            'evidence': [],
            'citation': ""
        }
    },
    
    'opposing_views': {
        'critique_1': {
            'view': "技术只是物质性的，不涉及神学维度",
            'counter_evidence': [],
            'response': ""
        }
    }
}


def generate_search_guide(output_path: Path) -> Path:
    """
    生成文献检索指南文档
    
    Args:
        output_path: 输出文件路径
        
    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    content = f"""# 理论构建：文献检索指南

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 研究核心问题

**核心论题**：{THEORETICAL_FRAMEWORK['core_thesis']}

### 关键研究问题

"""
    
    for i, question in enumerate(THEORETICAL_FRAMEWORK['research_questions'], 1):
        content += f"{i}. {question}\n"
    
    content += """
## 英文搜索提示词（推荐用于AI辅助检索，如Consensus, Perplexity等）

### 核心问题式查询

"""
    
    for query in SEARCH_QUERIES['english'][:3]:
        content += f"- {query}\n"
    
    content += """
### 关键词组合查询

"""
    
    for keyword in SEARCH_QUERIES['keywords']['english']:
        content += f"- `{keyword}`\n"
    
    content += """
## 中文搜索提示词

### 关键词组合

"""
    
    for keyword in SEARCH_QUERIES['chinese']:
        content += f"- {keyword}\n"
    
    content += """
### 中文关键词

"""
    
    for keyword in SEARCH_QUERIES['keywords']['chinese']:
        content += f"- `{keyword}`\n"
    
    content += """
## 核心理论概念

### 1. 技术崇高 (Technological Sublime)

**定义**：{technological_sublime['definition']}

**关键词**：{', '.join(technological_sublime['keywords'])}

### 2. 否定神学 (Apophatic Theology)

**定义**：{apophatic_theology['definition']}

**关键词**：{', '.join(apophatic_theology['keywords'])}

### 3. 晚期风格 (Late Style)

**定义**：{late_style['definition']}

**关键词**：{', '.join(late_style['keywords'])}

## 主流观点框架（待填充）

### 共识观点1：技术作为新神学形式
**观点**：{mainstream_views['consensus_1']['view']}

**需要检索的内容**：
- 支持此观点的学者和文献
- 具体证据和文本分析
- 相关理论框架

### 共识观点2：晚期风格与虚无
**观点**：{mainstream_views['consensus_2']['view']}

### 共识观点3：技术设备的神学维度
**观点**：{mainstream_views['consensus_3']['view']}

## 对立观点（需要回应）

### 批判观点1：技术的纯粹物质性
**观点**：{opposing_views['critique_1']['view']}

**需要准备的回应**：
- TDA分析的拓扑证据
- 语义锚定的数学证明

## 检索策略建议

1. **数据库选择**：
   - JSTOR
   - Project MUSE
   - MLA International Bibliography
   - Google Scholar

2. **检索步骤**：
   - 使用英文提示词在AI工具（如Consensus）中初步检索
   - 根据初步结果在学术数据库中精炼检索
   - 追踪重要文献的引用关系

3. **关键词组合策略**：
   - DeLillo + technology + theology/sublime
   - DeLillo + late style + minimalism
   - apophatic theology + contemporary literature

## TDA分析结果的理论对话点

### 语义锚定结果 → 理论验证

如果语义锚定显示环质心最接近"Silence"或"Void"：
- 这与否定神学的理论框架一致
- 可以用拓扑证据支持"虚无"的主题

### 多环纯度分析 → 理论区分

如果检测到高异质性环（技术词+神学词混合）：
- 这证明了"技术即神学"的并置
- 可以回应"技术纯粹物质性"的批判

### 滑动窗口演变 → 叙事分析

如果"虚无演变曲线"在特定情节处上升：
- 可以将拓扑变化与叙事转折点关联
- 支持"技术神学"是文本固有结构的论点

---
**注意**：此文档是动态的，应在文献检索过程中不断更新和完善。

""".format(
        technological_sublime=THEORETICAL_FRAMEWORK['key_concepts']['technological_sublime'],
        apophatic_theology=THEORETICAL_FRAMEWORK['key_concepts']['apophatic_theology'],
        late_style=THEORETICAL_FRAMEWORK['key_concepts']['late_style'],
        mainstream_views=THEORETICAL_FRAMEWORK['mainstream_views'],
        opposing_views=THEORETICAL_FRAMEWORK['opposing_views']
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"生成文献检索指南: {output_path}")
    
    return output_path


def create_theory_template(output_path: Path) -> Path:
    """
    创建理论观点整理模板
    
    Args:
        output_path: 输出文件路径
        
    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    content = """# 理论观点整理模板

## 文献信息

**标题**：
**作者**：
**发表年份**：
**来源**：
**DOI/链接**：

## 核心观点

（简要总结作者的主要论点）

## 与本研究的相关性

- [ ] 支持"技术即神学"的观点
- [ ] 讨论否定神学与文学的关系
- [ ] 分析德利洛晚期风格
- [ ] 其他相关性：___________

## 关键引文

> （摘录支持本研究的重要引文）

## 理论对话点

### 可以用于支撑TDA分析的哪些发现？

- [ ] 语义锚定结果
- [ ] 多环纯度分析
- [ ] 滑动窗口演变
- [ ] 对比实验结果

### 如何与TDA结果对话？

（描述如何将此文献观点与拓扑分析结果结合）

---

**记录日期**：
**状态**：[ ]待深入阅读 [ ]已提取要点 [ ]已整合到论文

---

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"创建理论观点整理模板: {output_path}")
    
    return output_path


def generate_ai_search_prompts(output_path: Path) -> Path:
    """
    生成用于AI辅助检索的提示词文件（如Consensus）
    
    Args:
        output_path: 输出文件路径
        
    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    content = """# AI辅助文献检索提示词（适用于Consensus等工具）

## 核心问题式查询（推荐优先使用）

"""
    
    for i, query in enumerate(SEARCH_QUERIES['english'][:3], 1):
        content += f"""
### 查询 {i}
```
{query}
```

**预期检索内容**：关于德利洛技术神学/技术崇高的学术观点和文献

**后续精炼**：可以追问"有哪些学者支持这一观点？"或"请提供具体文献引用"

---

"""
    
    content += """## 关键词组合查询

"""
    
    for keyword in SEARCH_QUERIES['keywords']['english']:
        content += f"- `{keyword}`\n"
    
    content += """
## 中文查询（Consensus对中文支持尚可）

"""
    
    for keyword in SEARCH_QUERIES['chinese']:
        content += f"- {keyword}\n"
    
    content += """
## 使用建议

1. **初始检索**：使用问题式查询获得概览
2. **深度挖掘**：对感兴趣的观点追问文献来源
3. **交叉验证**：用不同关键词组合验证结果
4. **引用追踪**：找到关键文献后，追踪其引用和被引用关系

## 检索后的整理步骤

1. 将重要文献记录到理论观点整理模板中
2. 提取支持/反对核心论点的观点
3. 建立理论框架与TDA分析结果的对应关系

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"生成AI检索提示词文件: {output_path}")
    
    return output_path


def compare_theory_with_tda(
    theoretical_view: str,
    tda_results: Dict[str, Any],
    output_path: Path
) -> Path:
    """
    对比理论观点与TDA分析结果
    
    Args:
        theoretical_view: 理论观点文本
        tda_results: TDA分析结果字典（包含语义锚定、环纯度等）
        output_path: 输出文件路径
        
    Returns:
        输出文件路径
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    
    content = f"""# 理论观点与TDA分析结果对比

生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 理论观点

{theoretical_view}

## TDA分析结果

### 语义锚定结果
"""
    
    if 'semantic_anchoring' in tda_results:
        anchoring = tda_results['semantic_anchoring']
        content += f"""
- 环质心最接近的概念：{anchoring.get('closest_concept', 'N/A')}
- 距离：{anchoring.get('distance', 'N/A'):.4f}
- 解释：{anchoring.get('interpretation', 'N/A')}
"""
    
    if 'cycle_purity' in tda_results:
        purity = tda_results['cycle_purity']
        content += f"""
### 环纯度分析
- 异质性：{purity.get('heterogeneity', 'N/A'):.4f}
- 环类型：{purity.get('type', 'N/A')}
- 环大小：{purity.get('size', 'N/A')}
"""
    
    content += """
## 对应关系分析

### 理论观点如何被TDA结果支持？

（描述TDA分析如何提供数学证据支持或反驳理论观点）

### TDA结果的文学/哲学解读

（将拓扑结构转化为文学理论语言）

### 创新点

（TDA分析如何超越现有理论，提供新的证据或视角）

"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"生成理论-TDA对比文件: {output_path}")
    
    return output_path

