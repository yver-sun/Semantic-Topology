# 理论构建指南

## 概述

理论构建模块帮助您：
1. **找到"靶子"**：定位学界对德利洛"技术神学"的主流观点
2. **建立对话**：将TDA分析结果与理论观点进行对比
3. **构建框架**：系统整理理论支撑和证据链

## 快速开始

### 1. 生成检索指南

```bash
python scripts/generate_theory_guide.py
```

这将生成三个文档：
- `artifacts/results/theory_search_guide.md` - 完整的文献检索指南
- `artifacts/results/ai_search_prompts.md` - AI工具检索提示词
- `artifacts/results/theory_notes/theory_template.md` - 文献整理模板

### 2. 使用AI辅助检索

#### 推荐工具：Consensus

1. 打开 `ai_search_prompts.md`
2. 复制核心问题式查询，例如：
   ```
   How does Don DeLillo depict technology as a form of religion or theology?
   ```
3. 在Consensus中搜索，获取学术观点概览
4. 追问文献来源和具体引用

#### 关键词搜索

使用以下关键词组合在学术数据库中检索：
- `Don DeLillo technology divine`
- `DeLillo technological sublime`
- `apophatic theology DeLillo`

### 3. 整理理论观点

对每个重要文献：
1. 复制 `theory_template.md` 创建一个新文件
2. 填写文献信息和核心观点
3. 标记与本研究的相关性
4. 提取关键引文
5. 建立与TDA分析结果的对应关系

## 核心理论框架

### 三个核心概念

1. **技术崇高 (Technological Sublime)**
   - 技术带来的超越性体验
   - 关键词：technology, sublime, transcendence, screen

2. **否定神学 (Apophatic Theology)**
   - 通过否定来接近神圣的不可言说性
   - 关键词：void, silence, nothingness, absence

3. **晚期风格 (Late Style)**
   - 极简主义的语言形式，指向意义的缺失
   - 关键词：minimalism, sparse, reduction

### 主流观点（待填充）

1. **技术作为新神学形式**
   - 需要找到：支持此观点的学者和文献
   
2. **晚期风格与虚无**
   - 需要找到：讨论德利洛晚期极简主义的文献
   
3. **技术设备的神学维度**
   - 需要找到：分析屏幕、数字界面等技术的文献

## TDA结果与理论的对话

### 语义锚定结果 → 理论验证

如果TDA分析显示环质心最接近"Silence"或"Void"：
- 这支持否定神学的理论框架
- 可以用拓扑证据证明"虚无"主题

### 多环纯度分析 → 理论区分

如果检测到高异质性环（技术词+神学词混合）：
- 这证明了"技术即神学"的并置
- 可以回应"技术纯粹物质性"的批判

### 滑动窗口演变 → 叙事分析

如果"虚无演变曲线"在特定情节处上升：
- 可以将拓扑变化与叙事转折点关联
- 支持"技术神学"是文本固有结构的论点

## 检索策略

### 英文检索（推荐）

使用提供的英文提示词在以下工具中检索：
- **Consensus** - AI辅助学术检索，对英文支持最好
- **Perplexity** - 另一个AI检索工具
- **Elicit** - 专门用于学术研究

### 中文检索

关键词（Consensus对中文支持尚可）：
- "唐·德利洛 技术崇高 否定神学"
- "德利洛 技术神学"

### 传统数据库

- JSTOR
- Project MUSE
- MLA International Bibliography
- Google Scholar

## 理论对话模板

当您找到理论观点后，可以：

1. **提取核心论点**
2. **与TDA结果对比**：
   - TDA支持/反驳哪些观点？
   - TDA提供了哪些新证据？
   - TDA如何超越现有理论？

3. **构建论证链**：
   - 理论观点 → TDA证据 → 新结论

## 示例工作流

1. **初始检索**（使用AI工具）
   ```
   "How does Don DeLillo depict technology as a form of religion?"
   ```
   
2. **深度挖掘**（追问文献）
   ```
   "What are the key academic works that support this view?"
   ```
   
3. **整理观点**（使用模板）
   - 记录文献信息
   - 提取核心论点
   - 标记相关性
   
4. **理论对话**（对比TDA结果）
   - 语义锚定是否支持该观点？
   - 环纯度分析提供什么新证据？
   - 如何整合到论文中？

## 注意事项

- **动态更新**：检索指南是起点，应在研究过程中不断更新
- **交叉验证**：用不同关键词组合验证结果
- **引用追踪**：找到关键文献后，追踪其引用关系
- **理论对话**：始终思考TDA结果如何与理论观点对话

---

更多详细信息，请查看生成的文件：
- `artifacts/results/theory_search_guide.md` - 完整指南
- `artifacts/results/ai_search_prompts.md` - 提示词库

