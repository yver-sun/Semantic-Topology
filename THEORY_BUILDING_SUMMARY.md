# 理论构建辅助功能总结

## 已实现的功能

我已经为您创建了完整的理论构建辅助模块，帮助您进行"文学与神学"的理论构建工作。

### 1. 核心模块

**文件位置**: `src/theory_builder.py`

包含以下功能：
- `generate_search_guide()` - 生成文献检索指南
- `create_theory_template()` - 创建理论观点整理模板
- `generate_ai_search_prompts()` - 生成AI检索提示词
- `compare_theory_with_tda()` - 对比理论观点与TDA结果

### 2. 预设内容

模块中已经内置了您提供的所有搜索提示词和关键词：

#### 英文搜索提示词（推荐用于Consensus等AI工具）
- "How does Don DeLillo depict technology as a form of religion or theology?"
- "What is the relationship between the technological sublime and apophatic theology in contemporary literature?"
- "Does Don DeLillo's late fiction focus on silence and the void?"

#### 关键词组合
- "Don DeLillo technology divine"
- "DeLillo technological sublime"
- "apophatic theology DeLillo"

#### 中文关键词
- "唐·德利洛 技术崇高 否定神学"
- "德利洛 技术神学"

### 3. 生成脚本

**文件位置**: `scripts/generate_theory_guide.py`

运行方式：
```bash
python scripts/generate_theory_guide.py
```

将生成以下文档：
1. `artifacts/results/theory_search_guide.md` - 完整的文献检索指南
2. `artifacts/results/ai_search_prompts.md` - AI工具检索提示词
3. `artifacts/results/theory_notes/theory_template.md` - 文献整理模板

### 4. 文档说明

**文件位置**: `docs/THEORY_BUILDING.md`

详细的使用指南，包括：
- 快速开始步骤
- 核心理论框架说明
- TDA结果与理论的对话方法
- 检索策略建议
- 理论对话模板

## 使用方法

### 第一步：生成检索指南

```bash
python scripts/generate_theory_guide.py
```

### 第二步：使用AI工具检索

1. 打开生成的文件 `artifacts/results/ai_search_prompts.md`
2. 复制英文提示词到Consensus或其他AI检索工具
3. 获取学术观点和文献来源

### 第三步：整理理论观点

1. 使用 `theory_template.md` 为每个重要文献创建记录
2. 填写文献信息、核心观点、关键引文
3. 标记与TDA分析结果的相关性

### 第四步：建立理论对话

使用 `compare_theory_with_tda()` 函数对比：
- 理论观点 ↔ TDA语义锚定结果
- 理论观点 ↔ 多环纯度分析
- 理论观点 ↔ 滑动窗口演变曲线

## 核心理论框架

模块中预设了三个核心概念：

1. **技术崇高 (Technological Sublime)**
   - 技术带来的超越性体验
   - 关键词：technology, sublime, transcendence, screen

2. **否定神学 (Apophatic Theology)**
   - 通过否定来接近神圣的不可言说性
   - 关键词：void, silence, nothingness, absence

3. **晚期风格 (Late Style)**
   - 极简主义的语言形式，指向意义的缺失
   - 关键词：minimalism, sparse, reduction

## TDA与理论的对话点

模块中已预设了如何将TDA分析结果与理论观点对接：

### 语义锚定 → 理论验证
如果环质心最接近"Silence"或"Void"，这支持否定神学的理论框架。

### 多环纯度 → 理论区分
如果检测到高异质性环（技术词+神学词混合），这证明了"技术即神学"的并置。

### 滑动窗口 → 叙事分析
如果"虚无演变曲线"在特定情节处上升，可以将拓扑变化与叙事转折点关联。

## 后续工作建议

1. **运行脚本生成指南**
   ```bash
   python scripts/generate_theory_guide.py
   ```

2. **使用Consensus等AI工具**
   - 复制提示词进行检索
   - 获取学术观点和文献

3. **整理文献观点**
   - 使用模板记录每个重要文献
   - 提取支持/反对核心论点的观点

4. **建立对应关系**
   - 将理论观点与TDA结果连接
   - 构建"理论→证据→结论"的论证链

## 注意事项

- 所有预设内容都可以根据您的实际研究需求进行调整
- 检索指南是动态的，应在研究过程中不断更新
- 建议先使用AI工具（如Consensus）获得概览，再在学术数据库中精炼检索

---

**文件位置索引**：
- 模块代码：`src/theory_builder.py`
- 生成脚本：`scripts/generate_theory_guide.py`
- 使用指南：`docs/THEORY_BUILDING.md`
- 生成文档：`artifacts/results/` 目录

