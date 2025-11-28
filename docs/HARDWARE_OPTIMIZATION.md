# 硬件配置优化指南

## 您的硬件配置分析

### 配置概览

- **CPU**: AMD Ryzen 7 9700X (Zen 5, 8核16线程) ⭐ **性能强劲**
- **GPU**: RX590 (AMD, 12nm) ⚠️ **注意GPU兼容性**
- **内存**: 32GB DDR5 6000MHz ✅ **非常充足**
- **存储**: 2TB PCIe 4.0 + 1TB NVMe ✅ **高速存储**

### 性能评估

| 组件 | 性能评级 | 对本项目的适用性 |
|------|---------|----------------|
| CPU | ⭐⭐⭐⭐⭐ | 优秀 - 8核16线程，BERT推理速度很快 |
| GPU | ⚠️⭐⭐ | 受限 - AMD GPU需要ROCm，PyTorch默认不支持 |
| 内存 | ⭐⭐⭐⭐⭐ | 优秀 - 32GB足够处理大型文本 |
| 存储 | ⭐⭐⭐⭐⭐ | 优秀 - PCIe 4.0高速，缓存加载快 |

## 关键配置建议

### 1. GPU配置（重要）

**问题**：PyTorch默认使用CUDA（NVIDIA），您的RX590是AMD显卡。

**解决方案**：

#### 方案A：使用CPU（推荐）
您的Ryzen 7 9700X性能很强，CPU推理速度足够快。

```yaml
# config/default_config.yaml
embedding:
  device: "cpu"  # 强制使用CPU
  use_gpu: false
```

#### 方案B：尝试AMD ROCm（高级用户）
如果您想使用GPU，需要安装ROCm支持：

```bash
# 注意：ROCm对RX590的支持可能不完善
# 建议先测试，如果不行就使用CPU
```

**当前代码会自动检测**：
- 如果`device: "auto"`，代码会检测CUDA可用性
- 由于RX590不支持CUDA，会自动回退到CPU
- 这样配置是完全安全的

### 2. 内存优化

**优势**：32GB内存非常充足，可以：
- 处理更大的文本文件
- 增加批处理大小
- 同时运行多个分析任务

**建议配置**：
```yaml
nlp:
  batch_size: 64  # 可以从32增加到64（内存充足）

tda:
  n_landmarks: 1024  # 可以从512增加到1024
```

### 3. 存储优化

**优势**：PCIe 4.0高速存储，建议：
- 将项目放在PCIe 4.0的2TB盘上（`TiPlus 7100`）
- 模型缓存会更快加载
- 大型嵌入文件读写速度快

**建议目录结构**：
```
TiPlus 7100 (2TB)/
  └── Semantic-Topology/  # 项目主目录
      ├── artifacts/       # 数据输出（在高速盘上）
      └── ~/.cache/huggingface/  # 模型缓存（自动）
```

### 4. CPU多核利用

**优势**：8核16线程，可以并行处理：
- 文本提取（多文件并行）
- TDA分析（可考虑并行化多个文本）

**当前支持**：
- BERT推理会使用CPU的所有核心
- Spacy NLP处理会利用多核
- NumPy/SciPy计算会自动并行

## 针对您的配置的推荐设置

### 配置文件优化（`config/default_config.yaml`）

```yaml
embedding:
  model_name: "bert-base-cased"
  device: "cpu"  # 明确使用CPU（避免GPU检测开销）
  use_gpu: false
  max_length: 512
  mean_pooling: false

nlp:
  batch_size: 48  # 利用32GB内存，增加批处理
  # 其他配置保持不变

tda:
  n_landmarks: 768  # 可以增加到768（内存和CPU都足够）
  landmark_strategy: "kmeans"
  verify_stability: true

# 利用大内存，可以处理更大的文本
text_extraction:
  max_sentences: 0  # 不限制，利用32GB内存
```

### 性能优化技巧

1. **增加批处理大小**
   ```python
   # 在config/default_config.yaml中
   nlp:
     batch_size: 48  # 从32增加到48
   ```

2. **利用多进程处理多个文本**
   - 当前代码顺序处理
   - 可以考虑并行处理多个文本文件（需要修改代码）

3. **预加载模型到内存**
   - 代码已有模型缓存机制
   - 首次运行后，模型会保留在内存中

## 性能预期

### CPU推理速度（Ryzen 7 9700X）

- **BERT嵌入生成**：约 100-200 tokens/秒
- **一本小说的处理时间**：
  - Point Omega (~50,000词)：约 5-10 分钟
  - Zero K (~70,000词)：约 7-15 分钟
  - The Silence (~30,000词)：约 3-7 分钟

### 内存使用

- **BERT模型**：~400MB
- **嵌入矩阵**：每10,000词约 30-50MB
- **TDA分析**：地标矩阵约 100-200MB
- **峰值内存**：单个文本约 1-2GB（您有32GB，完全足够）

## 环境变量优化

创建 `.env` 文件或设置环境变量：

```bash
# 强制使用CPU（避免GPU检测）
export USE_GPU=false

# 增加NumPy线程数（利用多核）
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Spacy使用所有CPU核心
export SPACY_NUM_JOBS=16
```

## 监控性能

### 查看CPU使用率

```bash
# Windows
taskmgr  # 任务管理器

# Linux/Mac
htop  # 或 top
```

### 查看内存使用

代码会自动记录日志，查看：
- `artifacts/results/analysis.md` - 处理时间
- 日志文件 - 详细的性能信息

## 故障排除

### 如果GPU相关错误

**症状**：出现CUDA相关错误或警告

**解决**：
```yaml
# 在config/default_config.yaml中明确禁用GPU
embedding:
  device: "cpu"
  use_gpu: false
```

### 如果内存不足（理论上不会，但以防万一）

**症状**：内存使用接近32GB

**解决**：
- 减小 `batch_size` 到 24 或 16
- 减小 `n_landmarks` 到 512
- 限制 `max_sentences`

## 最佳实践

1. **首次运行**：使用较小配置测试
   ```yaml
   nlp:
     batch_size: 32
   tda:
     n_landmarks: 512
   ```

2. **确认稳定后**：增加配置利用硬件优势
   ```yaml
   nlp:
     batch_size: 48-64
   tda:
     n_landmarks: 768-1024
   ```

3. **监控系统资源**：确保CPU和内存使用合理

## 对比GPU版本（参考）

如果未来升级到NVIDIA GPU：
- RTX 4060 (8GB)：约 3-5 倍加速
- RTX 4070 (12GB)：约 5-8 倍加速

但您的CPU性能已经很好，对于本项目（推理而非训练），CPU已经足够快。

---

**总结**：您的配置非常适合本项目，CPU性能强劲，内存充足。建议使用CPU模式，可以获得稳定、高效的性能。

