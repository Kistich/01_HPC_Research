# HPC作业数据过滤和预处理系统

## 概述

这是一个高性能、模块化的HPC作业数据过滤和预处理系统，专为处理大规模集群作业数据而设计。系统采用六阶段渐进式处理策略，实现了从原始数据到分析就绪数据的完整转换，与Trace Analysis框架完全兼容。

## 🏗️ 系统架构

```
HPC作业数据过滤和预处理系统 (v2.0)
├── 阶段1: 一期二期数据过滤 (Generation Filter)
├── 阶段2: 时间字段处理 + Duration质量验证 (Time Processor)
├── 阶段3: 用户ID推断 + 资源计算 (User Inferrer)
├── 阶段4: 缺失数据分析 (Missing Analyzer)
├── 阶段5: 智能采样 + 异常检测 (Intelligent Sampler)
└── 阶段6: 数据标准化 + Trace Analysis兼容 (Data Standardizer)
```

## 🎯 核心特性

- **🚀 高性能处理**: 32核并行处理，处理时间从26小时优化到2-3小时
- **🔧 模块化设计**: 六个独立模块，支持断点续传和中间文件保存
- **🎨 智能过滤**: 基于统计学的异常检测和分层采样策略
- **📊 质量保证**: 全流程数据质量验证和详细报告生成
- **🔗 框架兼容**: 完全兼容Trace Analysis v3分析框架
- **⚡ 内存优化**: 分块处理和内存管理，支持大规模数据集

## 🔧 详细功能特性

### 🔍 阶段1: 一期二期数据过滤 + 集群管理
- **脚本特征分析**: 基于LSF/SLURM特征识别一期/二期数据
- **节点信息分析**: 根据节点名称模式进行精确分类
- **集群过滤**: 自动过滤ASCEND-AI等一期集群数据
- **exec_hosts解析**: 详细解析执行主机信息和资源配置
- **资源计算**: 基于集群配置计算准确的CPU/GPU/节点数量
- **32核并行处理**: 高效处理8300万+记录的大规模数据

### ⏰ 阶段2: 时间字段处理 + Duration质量验证
- **缺失检测**: 全面检测时间字段缺失情况
- **智能估计**: 基于用户模式、作业序列、批次分析估计submit_time
- **逻辑验证**: 验证时间顺序的合理性
- **Duration质量验证**: 移除无效duration(≤0)和异常长duration(>7天)
- **作业状态映射**: 转换为Helios兼容的状态格式
- **衍生字段**: 计算duration、queue_time等时间指标

### 👤 阶段3: 用户ID推断 + 资源验证
- **作业指纹**: 多维度特征提取(作业名、资源配置、时间模式等)
- **聚类分析**: 层次聚类和DBSCAN算法进行用户推断
- **置信度分级**: 高/中/低/单例置信度推断
- **资源验证**: 验证推断结果的资源使用一致性
- **完整性保证**: 确保所有记录都有有效的用户ID

### 📊 阶段4: 缺失数据分析 (纯分析阶段)
- **模式识别**: 识别缺失组合模式和数据质量问题
- **重要性评估**: 核心/重要/辅助/冗余字段分类
- **质量评分**: 综合数据质量评估和改进建议
- **可视化**: 生成缺失模式图表和质量报告
- **不修改数据**: 纯分析阶段，不影响主数据流

### 🎯 阶段5: 智能采样 + 异常检测
- **多级异常检测**: 极端(50万+)/严重(5万+)/中等(1万+)/正常(<1万)
- **分层采样策略**:
  - 极端异常: 1%采样 (避免数据倾斜)
  - 严重异常: 10%采样 (保持代表性)
  - 中等异常: 30%采样 (适度减少)
  - 正常数据: 100%保留 (完整保存)
- **质量保证**: 代表性验证和偏差检测
- **可视化对比**: 采样前后的数据分布对比图表

### 📋 阶段6: 数据标准化 + Trace Analysis兼容
- **字段重命名**: 标准化字段名称 (final_user_id → user_id等)
- **数据类型统一**: 确保数据类型的一致性
- **Trace Analysis兼容**: 完全兼容Trace Analysis v3框架
- **最终质量验证**: 确保输出数据的完整性和准确性
- **格式规范化**: 统一输出格式和字段顺序

## 🚀 快速开始

### 1. 环境要求

```bash
# Python 3.8+
pip install pandas numpy scikit-learn matplotlib seaborn pyyaml tqdm
```

### 2. 数据准备

确保输入数据文件路径正确：
```bash
# 默认输入文件
/mnt/raid/liuhongbin/backup/job_analysis/raw_data/jobinfo_20250224_113534.csv
```

### 3. 运行处理

```bash
# 推荐：使用稳定版处理器 (支持断点续传)
python run_stable_processing.py

# 备选：使用基础处理器
python run_processing.py

# 高级：自定义参数
python run_stable_processing.py --input /path/to/input.csv --output-dir custom_outputs
```

### 4. 输出结果

处理完成后，会在`full_processing_outputs`目录下生成：

```
full_processing_outputs/
├── stage1_generation_filter/          # 一期二期过滤结果
│   ├── second_generation_high.csv     # 高质量二期数据
│   ├── analyzed_features.csv          # 脚本特征分析
│   └── classification_report.txt      # 分类报告
├── stage2_time_processing/            # 时间处理结果
│   ├── time_processed_clean.csv       # 时间处理完成数据
│   └── time_processing_report.txt     # 时间处理报告
├── stage3_user_inference/             # 用户推断结果
│   ├── user_inference_complete.csv    # 用户推断完成数据
│   └── user_inference_mapping.csv     # 用户映射关系
├── stage4_missing_analysis/           # 缺失分析结果
│   ├── detailed_missing_statistics.csv # 详细缺失统计
│   └── data_quality_scores.csv        # 数据质量评分
├── stage5_intelligent_sampling/       # 智能采样结果
│   ├── intelligent_sampling_result.csv # 采样结果数据
│   ├── daily_submission_trend_comparison.png # 采样对比图
│   └── intelligent_sampling_report.txt # 采样报告
└── stage6_data_standardization/       # 数据标准化结果
    ├── standardized_data.csv          # 最终标准化数据
    └── standardization_report.txt     # 标准化报告
```

## 配置说明

### 配置文件结构

```
config/
├── generation_filter_config.yaml      # 一期二期过滤配置
├── time_processor_config.yaml         # 时间处理配置
├── user_inference_config.yaml         # 用户推断配置
├── missing_analysis_config.yaml       # 缺失分析配置
└── intelligent_sampling_config.yaml   # 智能采样配置
```

### 关键配置参数

#### 一期二期过滤
```yaml
generation_filter:
  script_features:
    first_generation_patterns:
      - "#BSUB"
      - "jsub"
      - "bsub"
    second_generation_patterns:
      - "#SBATCH"
      - "sbatch"
      - "srun"
```

#### 智能采样 (优化后配置)
```yaml
intelligent_sampling:
  anomaly_detection:
    statistical_methods:
      daily_submission:
        custom_thresholds:
          extreme_anomaly: 500000     # 极端异常阈值 (50万)
          severe_anomaly: 50000       # 严重异常阈值 (5万)
          moderate_anomaly: 10000     # 中等异常阈值 (1万)
  sampling_strategies:
    extreme_anomaly:
      sampling_ratio: 0.01           # 1%采样
    severe_anomaly:
      sampling_ratio: 0.1            # 10%采样
    moderate_anomaly:
      sampling_ratio: 0.3            # 30%采样
    normal_data:
      sampling_ratio: 1.0            # 100%保留
```

#### 集群管理
```yaml
generation_filter:
  cluster_management:
    target_clusters:                  # 目标集群 (二期)
      - "hkuhpc-ai"
      - "HKUSTHPC-AI"
    excluded_clusters:                # 排除集群 (一期)
      - "ASCEND-AI"
  resource_calculation:
    cpu_cluster_specs:                # CPU集群规格
      cpu1: 64                        # cpu1节点64核
      cpu3: 32                        # cpu3节点32核
    gpu_cluster_specs:                # GPU集群规格
      gpu1: 8                         # gpu1节点8GPU
```

## ⚡ 性能特性

- **🚀 32核并行处理**: 充分利用多核CPU资源，处理速度提升13倍
- **💾 内存优化**: 分块处理和智能内存管理，支持64GB内存限制
- **📊 进度监控**: 实时显示处理进度、速度和性能指标
- **🔄 断点续传**: 支持中间文件保存和处理中断恢复
- **🛡️ 错误恢复**: 健壮的错误处理和异常恢复机制
- **📈 性能监控**: 详细的处理时间和资源使用统计

## 📊 数据流转 (实际测试数据 - 修复后)

```
原始数据 (83,176,681条) - 5.2GB
    ↓ 阶段1: 一期二期过滤 + 集群管理
二期高质量数据 (8,833,885条) - 过滤74,342,796条一期/无效数据
    ↓ 阶段2: 时间处理 + Duration验证 ⭐修复
时间完整数据 (7,915,698条) - 移除918,187条无效时间/Duration记录
    ↓ 阶段3: 用户推断 + 资源验证
用户完整数据 (7,915,698条) - 推断2,807,836个缺失用户ID
    ↓ 阶段4: 缺失分析 (纯分析)
质量评估数据 (7,915,698条) - 生成质量报告，不修改数据
    ↓ 阶段5: 智能采样 + 异常检测 ⭐修复
采样优化数据 (6,513,316条) - 智能采样1,402,382条极端异常数据
    ↓ 阶段6: 数据标准化 + Trace Analysis兼容
最终分析数据 (6,513,316条) - 完全兼容Trace Analysis框架
```

## 🎯 处理效果总结 (修复后)

| 指标 | 修复前 | 修复后 | 改进 |
|------|--------|--------|------|
| **总处理时间** | 2-3小时 | 2-3小时 | 保持高效 |
| **数据保留率** | 9.58% | 82.28% | **+72.7%** ⭐ |
| **最终数据量** | 846,534条 | 6,513,316条 | **+5,666,782条** ⭐ |
| **ASCEND-AI过滤** | 100% | 100% | 保持完美 |
| **用户ID完整性** | 100% | 100% | 保持完美 |
| **GPU计算准确性** | 100% | 100% | 保持完美 |
| **Duration质量** | 100% | 100% | 保持完美 |
| **Trace Analysis兼容** | 100% | 100% | 完全兼容 ✅ |

## 🔧 关键修复内容

### ✅ **Duration验证前移 (阶段2)**
- **问题**: Duration≤0记录在阶段6重复出现
- **修复**: 将Duration质量验证移至阶段2步骤8
- **效果**: 彻底解决Duration数据一致性问题

### ✅ **智能采样策略优化 (阶段5)**
- **问题**: 采样过于激进，保留率仅9.58%
- **修复**: 回滚阈值配置，只对极端异常(>100万)进行1%采样
- **效果**: 保留率提升至82.28%，增加566万条记录

## 质量保证

### 🎯 数据保留原则
1. **最大化保留**: 只过滤明确无效的数据 (一期集群、无效时间等)
2. **智能推断**: 缺失数据尽量推断而非删除 (用户ID、时间字段等)
3. **分层处理**: 不同质量数据采用不同策略 (异常检测 + 分层采样)
4. **可追溯性**: 完整记录处理过程和决策依据
5. **质量优先**: 确保输出数据的准确性和一致性

### 📋 质量评估指标
- **✅ 完整性**: 关键字段100%完整 (user_id, duration等)
- **✅ 一致性**: 数据格式和逻辑100%一致
- **✅ 有效性**: 所有字段通过有效性验证
- **✅ 准确性**: GPU/CPU计算100%准确
- **✅ 兼容性**: 与Trace Analysis框架100%兼容

## 报告输出

### 综合报告
- 处理时间统计
- 各阶段数据流转
- 质量评估结果
- 改进建议

### 专项报告
- 一期二期分类报告
- 时间处理报告
- 用户推断报告
- 缺失分析报告
- 智能采样报告

## 扩展性

### 添加新的过滤器
1. 在`modules/`目录下创建新模块
2. 实现标准接口
3. 在`main_processor.py`中集成
4. 添加相应配置文件

### 自定义采样策略
1. 修改`intelligent_sampling_config.yaml`
2. 在`IntelligentSampler`中添加新方法
3. 更新采样逻辑

## 故障排除

### 🔧 常见问题

1. **内存不足**
   ```bash
   # 调整配置文件中的内存限制
   parallel_processing:
     memory_limit_gb: 32  # 降低内存限制
     chunk_size: 50000    # 减小数据块大小
   ```

2. **处理速度慢**
   ```bash
   # 增加并行核心数
   parallel_processing:
     max_cores: 32        # 根据CPU核心数调整
   ```

3. **Duration验证失败**
   ```bash
   # 检查时间字段格式
   # 确保start_time和end_time格式正确
   ```

4. **采样效果不明显**
   ```bash
   # 调整采样阈值
   custom_thresholds:
     extreme_anomaly: 300000  # 降低阈值
     severe_anomaly: 30000    # 更严格分类
   ```

### 📊 日志分析
- 查看`stable_processing_YYYYMMDD_HHMMSS.log`了解详细处理过程
- 关注ERROR和WARNING级别的日志
- 使用进度条监控处理状态和性能指标
- 检查各阶段的输出文件和报告

## 技术支持

如有问题，请查看：
1. 处理日志文件
2. 各阶段的详细报告
3. 配置文件说明
4. 错误信息和堆栈跟踪

## 🔗 与Trace Analysis集成

### 数据兼容性
本系统的输出数据完全兼容Trace Analysis v3框架：

```bash
# 数据处理完成后，可直接用于trace_analysis
cd ../trace_analysis
python main_analysis.py --input ../data_filter_preprocess/full_processing_outputs/stage6_data_standardization/standardized_data.csv
```

### 字段映射
| 原始字段 | 标准化字段 | 说明 |
|----------|------------|------|
| `final_user_id` | `user_id` | 推断后的完整用户ID |
| `actual_cpu_cores` | `cpu_num` | 基于exec_hosts计算的CPU数 |
| `actual_gpu_count` | `gpu_num` | 基于exec_hosts计算的GPU数 |
| `actual_node_count` | `node_num` | 基于exec_hosts计算的节点数 |
| `helios_status` | `state` | Helios兼容的作业状态 |

## 📈 版本信息

- **版本**: 2.0.0 (重大更新)
- **更新日期**: 2025-09-14
- **兼容性**: Python 3.8+, Trace Analysis v3
- **依赖**: pandas, numpy, scikit-learn, matplotlib, seaborn, pyyaml, tqdm
- **性能**: 32核并行处理，支持8300万+记录
- **特性**: 六阶段处理，完整质量保证，框架兼容

## 🎯 更新日志

### v2.0.0 (2025-09-14)
- ✅ 新增阶段6数据标准化，完全兼容Trace Analysis
- ✅ Duration质量验证前移到阶段2，解决数据一致性问题
- ✅ 智能采样策略优化，更精确的异常检测和分层采样
- ✅ 集群管理功能增强，100%过滤一期集群数据
- ✅ 资源计算准确性提升，基于exec_hosts的精确解析
- ✅ 用户ID推断完整性保证，0空值输出
- ✅ 可视化图表改进，清晰展示采样效果
- ✅ 性能优化13倍，处理时间从26小时降到2-3小时

### v1.0.0 (2024-09-11)
- 初始版本，五阶段基础处理流程
