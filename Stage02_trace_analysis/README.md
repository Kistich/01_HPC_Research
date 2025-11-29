# HPC Workload Analysis Framework (Helios-Compatible)

## 项目概述

本项目是一个**严格按照Helios项目标准**的HPC集群工作负载分析系统，针对HPC集群的作业提交数据进行全面的特征分析和可视化。系统完全复现Helios的分析方法、数据格式和可视化风格，确保分析结果的可比性和一致性。

## 🆕 Helios兼容特性

### **完整的Helios数据格式支持**
- ✅ `cluster_log.csv` - 作业级别数据（7,956,871条记录）
- ✅ `cluster_sequence.csv` - 集群时间序列数据
- ✅ `cluster_throughput.csv` - 集群吞吐量数据
- ✅ `cluster_user.pkl` - 用户聚合统计数据

### **严格按照Helios分析方法**
- ✅ **集群特征分析**: 24小时利用率模式 + 集群吞吐量（对应Helios Figure 3）
- ✅ **作业特征分析**: GPU数量分布 + GPU时间分布（对应Helios Figure 4）
- ✅ **用户特征分析**: 用户资源消耗CDF + 用户行为模式（对应Helios Figure 5-6）
- ✅ **Philly比较分析**: 保持原有4张图结构，使用Helios兼容数据

### **Helios风格可视化**
- ✅ 双子图布局：`figsize=(12, 4), constrained_layout=True`
- ✅ Helios论文风格：`font.family='serif', font.size=11`
- ✅ 标准化图例和网格：`grid(linestyle=':', alpha=0.3)`
- ✅ 子图标注：`(a)`, `(b)`格式

## 数据集信息

- **数据规模**: 795万条作业记录
- **时间跨度**: 2024年作业数据
- **集群数量**: 2个主要集群 (hkuhpc-ai, HKUSTHPC-AI)
- **子集群类型**: CPU1-3, GPU1-3, BIGMEM
- **数据源**: `/mnt/raid/liuhongbin/backup/job_prediction/2_Generation_analysis/second_generation_jobs.csv`

## 架构对应关系

基于对Helios项目的深入分析，我们的架构对应关系如下：

**Helios架构:**
```
4个物理集群: Venus, Earth, Saturn, Uranus
├── 每个集群内有多个Virtual Clusters (VCs)
└── 每个VC内有多个物理节点
```

**我们的HPC架构:**
```
1个物理集群: HKUHPC-AI
├── 7个子集群: CPU1-3, GPU1-3, BIGMEM (对应Helios的物理集群)
└── 每个子集群内有多个物理节点 (cpu1-*, gpu1-*, etc.)
```

**重要**: 我们的**子集群**对应Helios的**物理集群**概念，而不是VC概念。

## 集群配置

### HPC二期硬件资源

| 节点类别 | 节点数量 | CPU规格 | GPU规格 | 内存规格 |
|---------|---------|---------|---------|---------|
| CPU1 | 110台 | 2×Intel Xeon Platinum 8358P (32C@2.60GHz) | / | 512GB |
| CPU2 | 30台 | 2×Intel Xeon Platinum 8358P (32C@2.60GHz) | / | 512GB |
| CPU3 | 20台 | 2×AMD EPYC 7763 (64C@2.45GHz) | / | 512GB |
| GPU1 | 50台 | 2×Intel Xeon Platinum 8358P (32C@2.60GHz) | 8×NVIDIA A800-SXM4-80GB | 1024GB |
| GPU2 | 10台 | 2×Intel Xeon Platinum 8358P (32C@2.60GHz) | 8×NVIDIA A800-SXM4-80GB | 2048GB |
| GPU3 | 15台 | 2×Intel Xeon Platinum 8358P (32C@2.60GHz) | 8×NVIDIA A40-48GB | 1024GB |
| BIGMEM | 6台 | 4×Intel Xeon Gold 6348H (24C@2.30GHz) | / | 大内存 |

## 项目结构

```
Trace_Analysis_v3/
├── README.md                          # 项目总体说明
├── requirements.txt                    # 依赖包列表
├── config/                            # 配置文件
│   ├── cluster_config.yaml           # 集群配置
│   └── analysis_config.yaml          # 分析配置
├── utils/                             # 工具模块
│   ├── __init__.py
│   ├── data_loader.py                # 数据加载器
│   ├── data_preprocessor.py          # 数据预处理器
│   ├── cluster_identifier.py         # 子集群识别器
│   └── visualization_utils.py        # 可视化工具
├── modules/                           # 分析模块
│   ├── 00_data_preprocessing/        # 数据预处理模块（核心）
│   ├── 01_philly_comparison/         # Philly对比分析
│   ├── 02_cluster_characterization/  # 集群特征分析
│   ├── 03_temporal_analysis/         # 时间模式分析
│   ├── 04_job_characterization/      # 作业特征分析
│   ├── 05_user_characterization/     # 用户行为分析
│   └── 06_resource_utilization/      # 资源利用率分析
├── data/                             # 数据目录
│   ├── raw/                          # 原始数据
│   ├── processed/                    # 预处理后数据
│   └── external/                     # 外部数据（如Philly数据集）
├── outputs/                          # 输出结果
│   ├── figures/                      # 图表输出
│   ├── reports/                      # 分析报告
│   └── metrics/                      # 计算指标
└── main.py                           # 主执行脚本
```

## 核心功能模块

### 0. 核心数据预处理 (00_data_preprocessing) 🔧
- 原始数据加载和清洗
- 数据标准化和增强处理
- 基础指标预计算
- 分类数据集准备
- 智能缓存机制

### 1. Philly对比分析 (01_philly_comparison) 📊
- 与Microsoft Philly数据集对比 (严格复现Helios方法)
- GPU作业持续时间CDF对比
- GPU时间按状态分布对比
- 数据集规模和效率对比

### 2. 数据概览模块 (01_data_overview) 📈
- 数据集基本统计信息
- 数据质量评估
- 缺失值分析
- 数据分布概览

### 3. 集群特征分析 (02_cluster_characterization) 🏗️
- 子集群利用率对比 (对应Helios的cluster characterization)
- 子集群24小时模式分析
- 资源效率和成功率对比
- 负载分布差异分析

### 4. 时间模式分析 (03_temporal_analysis) ⏰
- 24小时作业提交模式 (对应Helios的diurnal patterns)
- 周期性负载变化分析
- 月度趋势分析
- CPU vs GPU作业的时间分布对比

### 5. 作业特征分析 (04_job_characterization) 💼
- 作业持续时间分布 (对应Helios的job characterization)
- 排队时间分析
- 资源需求分析
- 作业状态和规模分析
- 按GPU需求的分层分析

### 6. 用户特征分析 (05_user_characterization) 👥
- 用户资源消耗分布 (对应Helios的user characterization)
- 帕累托分析和重度用户识别
- 用户排队体验分析
- 用户公平性评估 (基尼系数)
- 用户排队延迟CDF分析

### 7. 资源利用率分析 (06_resource_utilization) 🔋
- 整体和子集群利用率分析
- 时间维度利用率模式
- 负载均衡评估
- 容量需求和规划建议

## 关键设计特点

### 1. 作业类型识别策略
- **GPU作业**: exec_hosts字段包含gpu1-*, gpu2-*, gpu3-*节点
- **CPU作业**: exec_hosts字段包含cpu1-*, cpu2-*, cpu3-*, bigmem-*节点
- **混合作业**: 同时使用CPU和GPU节点

### 2. 子集群识别方法
- 通过exec_hosts字段中的节点前缀识别子集群类型
- 支持多节点作业的子集群归属判断
- 处理跨子集群作业的特殊情况

### 3. 数据流管理
- 统一的数据加载接口
- 标准化的数据预处理流程
- 模块间数据传递的一致性保证

### 4. 严格遵循Helios方法 📋
- 完全复现Helios论文的分析方法
- 相同的可视化风格和图表类型
- 一致的统计指标和计算方法
- 包含完整的Philly对比分析

### 5. 高效的数据处理架构 ⚡
- **数据预处理前置化**: 一次处理，多模块复用
- **智能缓存机制**: 自动缓存预处理结果，快速加载
- **统一数据接口**: 标准化的数据传递和访问
- **避免重复计算**: 基础指标预计算，提高效率

### 6. 大数据支持 📊
- 支持795万条作业记录
- 内存优化的数据处理
- 分块处理大数据集
- 高效的数据结构设计

## 快速开始

1. **环境准备**
```bash
cd /mnt/raid/liuhongbin/backup/Trace_Analysis_v3
pip install -r requirements.txt
```

2. **配置设置**
```bash
# 编辑集群配置
vim config/cluster_config.yaml
# 编辑分析配置
vim config/analysis_config.yaml
```

3. **执行Helios兼容分析**
```bash
# 🆕 运行完整的Helios兼容分析 (推荐)
python main_helios.py --module all

# 运行特定的Helios兼容模块
python main_helios.py --module philly    # Philly对比分析（保持4张图）
python main_helios.py --module cluster   # 集群特征分析（Helios风格）
python main_helios.py --module job       # 作业特征分析（Helios风格）
python main_helios.py --module user      # 用户特征分析（Helios风格）

# 强制重新处理数据
python main_helios.py --module all --force-reload

# 传统分析方法（兼容性保留）
python main.py --module all
```

### **推荐使用Helios兼容版本**
- ✅ **数据格式**: 完全兼容Helios标准
- ✅ **分析方法**: 严格按照Helios论文实现
- ✅ **可视化风格**: 复现Helios图表样式
- ✅ **结果可比性**: 与Helios分析结果直接可比

## 输出结果

### 🆕 Helios兼容格式数据
```
data/processed/helios_format/
├── cluster_log.csv      (725.9 MB) - 作业级别数据
├── cluster_sequence.csv (0.5 MB)   - 时间序列数据
├── cluster_throughput.csv (0.7 MB) - 吞吐量数据
└── cluster_user.pkl     (0.0 MB)   - 用户聚合数据
```

### 🆕 Helios风格可视化图表
```
output/
├── philly_comparison/           # Philly对比分析（保持4张图）
│   ├── job_type_distribution.png
│   ├── gpu_job_count_status.png
│   ├── gpu_duration_cdf.png
│   └── gpu_time_status.png
├── cluster_characterization/    # 集群特征分析（Helios Figure 3）
│   └── cluster_characterization_helios.png
├── job_characterization/        # 作业特征分析（Helios Figure 4）
│   ├── job_characterization_helios.png
│   └── job_status_distribution_helios.png
└── user_characterization/       # 用户特征分析（Helios Figure 5-6）
    ├── user_resource_cdf_helios.png
    └── user_behavior_patterns_helios.png
```

### 分析报告
- **Helios兼容分析报告**: `output/helios_analysis_report.txt`
- **数据统计摘要**: 包含Helios格式数据的完整统计信息
- **模块执行状态**: 各分析模块的执行结果和生成的图表文件

## 参考文献

本项目基于以下研究成果：
- Helios: Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters (SC'21)
- 相关GPU集群负载分析研究

## 联系信息

如有问题或建议，请联系项目维护者。
