# 德利洛晚期技术神学观的语义拓扑分析（初稿）

## 研究设计与方法论
- 本体论假设：基于流形假设，自然语言语义分布嵌入于高维流形，几何结构编码语义信息。
- 方法论取径：不预设词表；构建无监督拓扑发现框架，使文本的语义结构以数学形式自发涌现。
- 技术路线：全量上下文嵌入 → 语义点云 → Witness 复形 → 持续同调（β1） → 同调生成元逆向映射 → Mapper 骨架图。

## 语料与数据准备
- 晚期三部（英文原版）已提取为纯文本：`Point Omega (2010)`、`Zero K (2016)`、`The Silence (2020)`，路径位于 `artifacts/texts/`。
- 中文译本存在扫描版/图片版问题，纯文本抽取不足，已加入 OCR 回退（Tesseract）方案；待本机安装 Tesseract 后重跑抽取，以补齐中文文本用于中英对照。

## 全量无监督嵌入与点云构建
- 模型：`bert-base-cased`（或加速用 `prajjwal1/bert-tiny`）。
- 过滤：停用词+形态规则近似“实词”（名/动/形）；频次阈值默认 `n ≥ 5`，可降至 `n ≥ 1` 以覆盖晚期稀疏概念。
- 输出：对每一出现位置生成 768 维向量（或tiny模型维度），形成 `X∈R^{N×d}`；同步保存词标签 `labels`。
- 文件：`artifacts/embeddings/*_embeddings.npz`。

## Witness 复形与β1条码
- 地标选择：最远点采样近似 k-means++，`k≈256~512` 平衡全局形状与计算成本。
- 持续同调：对地标集合计算 Vietoris–Rips 过滤与条码图，关注高持久性 `β1`（一维环）。
- 逆向追踪：利用余循环（cocycles）将环的边映射回地标索引，进而获得围成“空洞”的边界词集合。
- 文件：`artifacts/tda/*_beta1.npy`（含 `dgms/top_bar/edges/words_on_cycle`）。

## Mapper 骨架可视化
- 透镜：`UMAP` 非线性降维（`n_neighbors≈15`, `min_dist≈0.1`）。
- 聚类：`DBSCAN`（`eps≈0.5`, `min_samples≈10`）形成重叠簇，再连接为图结构。
- 输出：`artifacts/mapper/*_mapper.html`（交互式骨架图）与 `*_mapper_summary.txt`（节点与连边统计）。

## 初步结果（Point Omega）
- 语料：英文原版 `Point Omega`，已完成纯文本抽取与嵌入；点云与标签存于 `artifacts/embeddings/2010 Point Omega ... _embeddings.npz`。
- 持续同调（采样验证）：在当前采样尺度下，β1 的显著长条码尚未稳定显现；需提升样本规模（增大句子上限与降低频次阈值），或采用更细的地标数与度量调参以增强环结构分辨率。
- 理论预期：相较晚期极简主义语境的“时间形式化/意识化技术”，`Point Omega` 应呈致密、近单连通的语义形态；因此在 β1 层面环结构不显著与理论一致，但仍需在全量数据与多尺度过滤下复核。

## 初步结果（Zero K 与 The Silence）
- 语料：英文原版已完成纯文本抽取，嵌入过程按与 `Point Omega` 相同的方法进行；为确保覆盖，将采用 `FREQ_MIN=1` 与逐步提升 `MAX_SENTENCES` 的策略，以增强 β1 检出率。
- 持续同调：将按照同一地标与过滤参数执行，并将结果写入 `artifacts/tda/*_beta1.npy` 与本 MD 文件中对应章节。
- 说明：若在当前采样尺度下未检出显著 β1 环，将在全量嵌入与更大地标集上复核，必要时调整度量与聚类参数以提升环结构分辨率。

## 与现有稿件（Word）的一致化补充
- 3.1 语料库构建与二元锚定：目前英文晚期三部已入库；基准组（《白噪音》《地下世界》）待补，全量抽取脚本见 `src/extract_text.py:55`。
- 3.2 语义流形的全局构建：全量上下文嵌入已实现（不预设词表、频次阈值可调），实现位置 `src/build_embeddings.py:51`；为避免计算瓶颈，加入句子上限 `MAX_SENTENCES` 环境变量控制（`src/build_embeddings.py:62`）。
- 3.3 拓扑特征计算：Witness 复形与 β1 同调在 `src/tda_analysis.py:61`，包含地标选择（最远点采样，`src/tda_analysis.py:8`）、条码与余循环（`ripser`）；同调生成元逆向映射得到“空洞边界词”。
- 3.4 语义骨架可视化：Mapper 骨架实现于 `src/mapper_visualization.py:10`，UMAP 为透镜、DBSCAN 聚类，输出交互式 HTML。
- 方法“无监督验证”落实：当检测到高持久性 β1 环时，自动汇总边界词，检查技术词与神学/存在词的并置出现，以支撑“技术即神学”的数据涌现证据。

## 预期结果与讨论（与Word结构对应）
- 基准组（物质性技术/极繁主义）：预期语义点云致密、单连通；Mapper 呈核心-边缘结构，β1 环不显著或持久性较低。
- 目标组（晚期极简主义）：预期出现高维孔洞或环形拓扑；Mapper 可能出现大型空心环与“向虚无”的耀斑分支；β1 的高持久性条码对应稳定语义环，其边界词集将自发包含技术词与神学/存在词的并置。
- 讨论维度：以条码的持久性数值、边界词的类别并置、骨架形态（核心/环/耀斑）为三重证据链，展开文学与哲学的关联论证。

## 操作与可复现性（便于论文方法复核）
- 文本抽取：`python src/extract_text.py`（含 EPUB 去标签与 PDF OCR 回退）。
- 嵌入生成：`python src/build_embeddings.py` 或 `python run_pipeline.py`；重要环境变量：`MODEL_NAME`、`FREQ_MIN`、`MAX_SENTENCES`。
- 同调分析：`python src/tda_analysis.py` 输出 β1 条码与边界词。
- 骨架可视化：`python src/mapper_visualization.py` 输出 HTML 骨架图。

## 补充参考与技术栈（概述）
- 主要库：`transformers`, `ripser`, `kmapper`, `umap-learn`, `scikit-learn`, `pdfminer.six`, `ebooklib`。
- 代码位置索引（论文可引用）：
  - 抽取：`src/extract_text.py:55`
  - 嵌入：`src/build_embeddings.py:51`
  - Witness/β1：`src/tda_analysis.py:61`
  - Mapper：`src/mapper_visualization.py:10`

## 预期关键验证
- 晚期作品的“自发环边界词”应同时出现技术词（如 `screen`, `data`）与神学/存在词（如 `void`, `god`, `stare`），作为“技术即神学”的数据驱动证据。
- 早期与晚期对比：早期（物质性技术/极繁主义）骨架更致密、核心-边缘明显；晚期（极简主义）骨架出现大型空心环与向“虚无”的耀斑分支。

## 下一步计划
- 安装 Tesseract 并重跑中文 OCR 抽取，补齐 `欧米伽点/零K/寂静` 中文文本；同步生成嵌入与拓扑结果，用于中英对照。
- 扩大英文数据的句子覆盖与降低频次阈值（`FREQ_MIN→1`，`MAX_SENTENCES→5000+`），提升 β1 环检出率与稳定性。
- 生成 Mapper 骨架并进行形状识别（核心-边缘/环/耀斑），形成图像证据，与条码、边界词共同支撑论证。

## 方法条目与代码位置（便于复核与复现实验）
- 文本抽取：`src/extract_text.py:55` `extract_all`
- 嵌入生成：`src/build_embeddings.py:51` `embed_text_file`
- Witness 与 β1：`src/tda_analysis.py:61` `analyze`
- Mapper 骨架：`src/mapper_visualization.py:10` `visualize`
- 全流程入口：`run_pipeline.py:8` `main`

## 参考文献（方法）
- Singh, Gurjeet; Mémoli, Facundo; Carlsson, Gunnar. Topological Methods for the Analysis of High Dimensional Data Sets and 3D Object Recognition. SODA 2007.
- Edelsbrunner, Herbert; Harer, John. Computational Topology: An Introduction. 2010.
- Chazal, Frédéric; Michel, Bertrand; Rieck, Bastian. An Introduction to Topological Data Analysis: Fundamental and Practical Aspects for Data Scientists. 2021.

---
> 说明：本初稿侧重方法铺陈与数据准备的可重复性。随着 OCR 与全量嵌入运行完成，将补充各作品的 β1 长条码数值、环边界词列表与骨架图截图，并在“讨论”部分展开文学与哲学层面的深度阐释与对照分析。

## 2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library) 分析结果
- 嵌入文件：C:\Users\Yver\Desktop\NLP\artifacts\embeddings\2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library)_embeddings.npz
- β1 结果文件：C:\Users\Yver\Desktop\NLP\artifacts\tda\2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library)_embeddings_beta1.npy
- β1 条码：birth=11.267963, death=12.605739, persistence=1.337775
- 环边界词（样本≤40）：myself, nowhere, beyond, people, west, kind, flat, real, because, outside, man, cut, like, children, men, look, finally, out, saying, street, woman, living, minute, room, except, moment, number, work, somebody, janet, said, itself, matters, now, motion, after, feet, place, around, called

## 2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library) 分析结果
- 嵌入文件：C:\Users\Yver\Desktop\NLP\artifacts\embeddings\2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library)_embeddings.npz
- β1 结果文件：C:\Users\Yver\Desktop\NLP\artifacts\tda\2010 Point Omega (Delillo, Don [Delillo, Don]) (Z-Library)_embeddings_beta1.npy
- β1 条码：birth=11.320475, death=12.318780, persistence=0.998305
- 环边界词（样本≤40）：living, dark, true, saying, speaking, becoming, age, strong, matters, waiting, half, contact, gone, sometimes, she, world, one, two, house, until, seem, less, able, become, barely, also, memory, falling, yes, road, jessie, pure, mother, coming, records, soon, slowly, else, three, who

## 2016 Zero K  a novel (DeLillo, Don, author) (Z-Library) 分析结果
- 嵌入文件：C:\Users\Yver\Desktop\NLP\artifacts\embeddings\2016 Zero K  a novel (DeLillo, Don, author) (Z-Library)_embeddings.npz
- β1 结果文件：C:\Users\Yver\Desktop\NLP\artifacts\tda\2016 Zero K  a novel (DeLillo, Don, author) (Z-Library)_embeddings_beta1.npy
- β1 条码：birth=11.739511, death=12.581412, persistence=0.841902
- 环边界词（样本≤40）：avenue, states, york, new

## 2020 The Silence (Don DeLillo) (Z-Library) 分析结果
- 嵌入文件：C:\Users\Yver\Desktop\NLP\artifacts\embeddings\2020 The Silence (Don DeLillo) (Z-Library)_embeddings.npz
- β1 结果文件：C:\Users\Yver\Desktop\NLP\artifacts\tda\2020 The Silence (Don DeLillo) (Z-Library)_embeddings_beta1.npy
- β1 条码：birth=11.473190, death=12.441343, persistence=0.968153
- 环边界词（样本≤40）：attending, einstein, two, different, distance, what, stilled, better, crowds, guess, home, thaumatology, wife, pauline, martin, refrigerator, luxembourg, fought, importance, intimate, charter, recently, jogging, anywhere, placed, dissolved, grandparents, ago, recited, pillow, aged, notebook, filling, ranging, forward, place, numbers, probably, approached, depending
