# AMD平台快速配置指南

## 您的配置

- **CPU**: AMD Ryzen 7 9700X (8核16线程) ✅
- **GPU**: RX590 (AMD) ⚠️
- **内存**: 32GB DDR5 ✅

## 快速配置步骤

### 1. 确认配置文件

配置文件 `config/default_config.yaml` 已经针对您的硬件进行了优化：

```yaml
embedding:
  device: "cpu"  # 使用CPU（AMD GPU不支持CUDA）
  use_gpu: false

nlp:
  batch_size: 48  # 利用32GB大内存

tda:
  n_landmarks: 768  # 利用强大CPU和内存
```

### 2. 运行硬件检测

```bash
python scripts/check_hardware.py
```

这将显示：
- CPU信息（应该显示8核16线程）
- 内存信息（应该显示32GB）
- CUDA状态（应该显示不可用，这是正常的）
- 优化建议

### 3. 设置环境变量（可选但推荐）

创建 `.env` 文件或设置环境变量以充分利用多核CPU：

```bash
# Windows PowerShell
$env:OMP_NUM_THREADS="16"
$env:MKL_NUM_THREADS="16"
$env:SPACY_NUM_JOBS="16"

# Linux/Mac
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export SPACY_NUM_JOBS=16
```

### 4. 运行分析

```bash
python run_pipeline.py
```

## 性能预期

基于您的硬件配置：

- **BERT嵌入生成**：约 100-200 tokens/秒（CPU模式）
- **完整分析时间**：
  - Point Omega: ~5-10 分钟
  - Zero K: ~7-15 分钟
  - The Silence: ~3-7 分钟

## 为什么使用CPU？

1. **AMD GPU限制**：
   - PyTorch默认使用CUDA（NVIDIA专用）
   - AMD GPU需要ROCm，但支持不完善
   - RX590可能不支持最新ROCm版本

2. **CPU性能优势**：
   - Ryzen 7 9700X性能非常强（8核16线程）
   - 对于推理任务（不是训练），CPU已经足够快
   - 更稳定，不需要额外的GPU驱动配置

3. **实际体验**：
   - CPU模式在本项目中的表现已经很好
   - 内存充足（32GB）可以增大批处理
   - 稳定性更高

## 常见问题

### Q: 我的RX590不能加速吗？

A: 
- 理论上可以，但需要安装ROCm（AMD的CUDA替代）
- ROCm配置复杂，对RX590支持可能不完善
- **建议**：使用CPU模式，您的9700X已经很快了

### Q: 未来升级到NVIDIA GPU会更快吗？

A: 
- 是的，NVIDIA GPU (RTX 4060+) 可以加速3-5倍
- 但对于本项目（推理而非训练），当前CPU性能已经足够
- 升级GPU的收益不是必需的

### Q: 如何最大化利用我的硬件？

A:
1. ✅ 使用优化后的配置文件（已经设置好）
2. ✅ 设置环境变量充分利用多核
3. ✅ 利用32GB内存，可以处理更大的文本
4. ✅ 使用高速PCIe 4.0存储，模型加载更快

## 配置文件说明

当前的优化配置：

```yaml
# 使用CPU，明确且稳定
embedding:
  device: "cpu"
  use_gpu: false

# 增加批处理，利用32GB内存
nlp:
  batch_size: 48  # 从32增加到48

# 增加地标数量，利用强大CPU
tda:
  n_landmarks: 768  # 从512增加到768
```

## 监控性能

运行分析时，查看：
- CPU使用率应该在60-80%（多核利用）
- 内存使用应该远低于32GB
- 处理速度应该稳定

如果遇到问题：
- CPU使用率低 → 检查环境变量设置
- 内存不足 → 减小batch_size或n_landmarks
- 速度慢 → 检查是否意外使用了GPU检测（应该直接使用CPU）

## 总结

✅ **您的配置非常适合本项目**
- CPU性能强劲
- 内存非常充足
- 存储速度快

✅ **推荐配置**
- 使用CPU模式（稳定、快速）
- 增大批处理和地标数量（利用硬件优势）
- 设置环境变量（充分利用多核）

✅ **预期性能**
- 单个文本分析：5-15分钟
- 完整流水线：30-60分钟（三个文本）

---

详细优化指南：`docs/HARDWARE_OPTIMIZATION.md`

