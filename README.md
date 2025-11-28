# 语义拓扑分析项目 (Semantic Topology Analysis)

用于分析文学作品语义拓扑结构的完整工具链，通过BERT嵌入、拓扑数据分析（TDA）和Mapper可视化来发现文本的语义结构特征。

## 项目简介

本项目实现了一套完整的无监督语义拓扑分析框架，用于从文本中发现语义结构。项目基于以下假设：

- **流形假设**：自然语言语义分布嵌入于高维流形，几何结构编码语义信息
- **无监督方法**：不预设词表，使文本的语义结构以数学形式自发涌现

### 技术路线

```
全量上下文嵌入 → 语义点云 → Witness 复形 → 持续同调（β1） → 
同调生成元逆向映射 → Mapper 骨架图
```

## 功能特性

- 📄 **多格式文本提取**：支持PDF、EPUB、TXT，带OCR回退支持
- 🔤 **NLP预处理**：词性过滤、停用词过滤、词频统计
- 🧠 **上下文感知嵌入**：基于BERT的语义嵌入生成
- 🔬 **拓扑数据分析**：持续同调（β1条码）、边界词提取
- 📊 **交互式可视化**：Mapper骨架图（HTML）、UMAP降维

## 安装指南

### 系统要求

- Python 3.8+
- 8GB+ RAM（推荐16GB用于大型文本）
- GPU（可选，用于加速BERT嵌入）

### 1. 克隆仓库

```bash
git clone <repository-url>
cd Semantic-Topology
```

### 2. 安装Python依赖

```bash
pip install -r requirements.txt
```

### 3. 安装Spacy语言模型

```bash
python -m spacy download en_core_web_sm
```

### 4. （可选）安装Tesseract OCR

如果要处理扫描版PDF，需要安装Tesseract OCR：

**Windows:**
- 下载安装程序：https://github.com/UB-Mannheim/tesseract/wiki
- 安装后，在配置文件中设置路径：`config/default_config.yaml`

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim
```

**macOS:**
```bash
brew install tesseract
```

## 配置说明

配置文件位于 `config/default_config.yaml`，主要配置项包括：

### 数据路径

```yaml
data:
  input_dir: "artifacts/raw_texts"      # 原始文件目录
  texts_dir: "artifacts/texts"          # 提取的文本目录
  embeddings_dir: "artifacts/embeddings" # 嵌入文件目录
  tda_dir: "artifacts/tda"              # TDA结果目录
  mapper_dir: "artifacts/mapper"        # Mapper可视化目录
  results_dir: "artifacts/results"      # 分析结果目录
```

### NLP处理

```yaml
nlp:
  model: "bert-base-cased"              # BERT模型名称
  spacy_model: "en_core_web_sm"         # Spacy模型
  keep_pos: ["NOUN", "PROPN", "ADJ", "VERB"]  # 保留的词性
  min_freq: 5                           # 最小词频
```

### 拓扑数据分析

```yaml
tda:
  landmark_strategy: "kmeans"           # 地标选择策略: 'kmeans' 或 'maxmin'
  n_landmarks: 512                      # 地标数量
  persistence_threshold: 0.05           # 持久度阈值
```

### 可视化

```yaml
visualization:
  mapper_neighbors: 15                  # Mapper邻居数
  mapper_overlap: 0.5                   # Mapper重叠比例
  umap_n_neighbors: 15                  # UMAP邻居数
  umap_min_dist: 0.1                    # UMAP最小距离
```

### 环境变量覆盖

所有配置项都可以通过环境变量覆盖：

```bash
export NLP_MODEL="prajjwal1/bert-tiny"  # 使用更小的模型加速
export TDA_N_LANDMARKS=256              # 减少地标数以加快计算
export FREQ_MIN=1                       # 降低词频阈值
```

## 使用方法

### 完整流水线

运行完整的分析流水线（文本提取 → 嵌入生成 → TDA分析 → 可视化）：

```bash
python run_pipeline.py
```

### 单独运行各步骤

#### 1. 文本提取

```python
from src.data_loader import extract_all
from src.utils import load_config, get_path, get_project_root

config = load_config()
root = get_project_root()
texts_dir = get_path(config, 'data.texts_dir', root)

extracted = extract_all(root, texts_dir, config)
```

#### 2. 生成嵌入

```python
from src.embedder import embed_text_file
from pathlib import Path

emb_path = embed_text_file(
    Path("artifacts/texts/example.txt"),
    Path("artifacts/embeddings"),
    config
)
```

#### 3. 拓扑数据分析

```python
from src.topology import analyze

tda_path, cycle_words, metrics = analyze(
    Path("artifacts/embeddings/example_embeddings.npz"),
    Path("artifacts/tda"),
    config
)

birth, death, persistence = metrics
print(f"β1条码: birth={birth:.6f}, death={death:.6f}, persistence={persistence:.6f}")
print(f"边界词: {cycle_words[:10]}")
```

#### 4. 可视化

```python
from src.visualizer import visualize

html_path, summary_path = visualize(
    Path("artifacts/embeddings/example_embeddings.npz"),
    Path("artifacts/mapper"),
    config
)
```

### 使用脚本

#### 批量分析嵌入文件

```bash
python scripts/run_tda_mapper.py
```

#### 分析特定英文文件

```bash
python scripts/analyze_en.py
```

#### 批量处理文本文件

```bash
python scripts/batch_analysis.py
```

#### 查看嵌入文件形状

```bash
python scripts/report_shapes.py
```

## 输出文件说明

### 文本文件

- **位置**: `artifacts/texts/*.txt`
- **格式**: 纯文本，UTF-8编码

### 嵌入文件

- **位置**: `artifacts/embeddings/*_embeddings.npz`
- **格式**: NumPy压缩文件
- **内容**:
  - `X`: 嵌入矩阵 (N, d)
  - `labels`: 词标签数组 (N,)

### TDA结果

- **位置**: `artifacts/tda/*_beta1.npy`
- **格式**: NumPy文件
- **内容**:
  - `dgms`: 条码图
  - `cocycles`: 余循环
  - `birth`, `death`, `persistence`: β1条码参数
  - `cycle_words`: 边界词列表

### 可视化文件

- **HTML**: `artifacts/mapper/*_mapper.html` - 交互式Mapper骨架图
- **摘要**: `artifacts/mapper/*_mapper_summary.txt` - 节点和连边统计

### 分析报告

- **位置**: `artifacts/results/analysis.md`
- **格式**: Markdown
- **内容**: 所有分析结果的汇总报告

## 项目结构

```
Semantic-Topology/
├── config/
│   └── default_config.yaml      # 配置文件
├── src/
│   ├── __init__.py
│   ├── utils.py                 # 工具函数
│   ├── data_loader.py           # 文本提取
│   ├── nlp_processor.py         # NLP处理
│   ├── embedder.py              # 嵌入生成
│   ├── topology.py              # TDA分析
│   └── visualizer.py            # 可视化
├── scripts/
│   ├── analyze_en.py            # 分析英文文件
│   ├── batch_analysis.py        # 批量分析
│   ├── run_tda_mapper.py        # TDA和Mapper分析
│   └── report_shapes.py         # 报告嵌入形状
├── artifacts/                   # 输出目录
│   ├── texts/                   # 提取的文本
│   ├── embeddings/              # 嵌入文件
│   ├── tda/                     # TDA结果
│   ├── mapper/                  # Mapper可视化
│   └── results/                 # 分析报告
├── notebooks/                   # Jupyter notebooks
├── run_pipeline.py              # 主流水线
├── requirements.txt             # Python依赖
└── README.md                    # 本文档
```

## 示例

### 分析德利洛作品

项目包含了分析Don DeLillo晚期三部作品的示例：

1. **Point Omega (2010)**
2. **Zero K (2016)**
3. **The Silence (2020)**

运行完整流水线：

```bash
python run_pipeline.py
```

查看结果：

```bash
cat artifacts/results/analysis.md
```

打开可视化：

```bash
# 在浏览器中打开
open artifacts/mapper/2010_Point_Omega_mapper.html
```

## 常见问题

### Q: 如何加速嵌入生成？

A: 有几种方法：
1. 使用更小的BERT模型（如 `prajjwal1/bert-tiny`）
2. 设置环境变量 `USE_GPU=false` 如果GPU内存不足
3. 减小 `batch_size` 配置

### Q: OCR提取失败怎么办？

A: 
1. 确保已安装Tesseract OCR
2. 在配置文件中设置正确的 `tesseract_path`
3. 检查PDF是否为扫描版（需要OCR）

### Q: 内存不足错误

A:
1. 减少地标数量（`n_landmarks`）
2. 限制处理的最大句子数（`max_sentences`）
3. 使用CPU而非GPU（减少GPU内存占用）

### Q: 如何调整词频阈值？

A: 修改配置文件中的 `nlp.min_freq`，或设置环境变量：
```bash
export FREQ_MIN=1  # 包含所有词
```

## 技术栈

- **文本处理**: pdfminer.six, ebooklib, pytesseract
- **NLP**: spaCy, transformers
- **机器学习**: scikit-learn, PyTorch
- **拓扑数据分析**: ripser
- **可视化**: kmapper, umap-learn
- **配置管理**: PyYAML

## 参考文献

1. Singh, G., Mémoli, F., & Carlsson, G. (2007). Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition. SODA 2007.

2. Edelsbrunner, H., & Harer, J. (2010). Computational Topology: An Introduction.

3. Chazal, F., Michel, B., & Rieck, B. (2021). An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists.

## 许可证

[添加许可证信息]

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

[添加联系方式]

---

**注意**: 本项目仍在持续开发中。如有问题或建议，请通过Issue反馈。

