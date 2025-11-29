# 集群特征分析模块 - Helios兼容版本

## 功能描述

本模块**严格按照Helios项目的cluster characterization方法**，分析集群的整体特征和性能表现。完全复现Helios论文Figure 3的分析方法和可视化风格，确保分析结果与Helios标准一致。

## 🆕 Helios兼容特性

### **对应Helios Figure 3**
- ✅ **集群利用率的24小时模式分析**：分析作业提交率和GPU使用率的日内变化
- ✅ **集群吞吐量的24小时模式分析**：分析作业开始率和资源消耗的日内变化
- ✅ **双子图布局**：`(a) Cluster Utilization Pattern` + `(b) Cluster Throughput Pattern`
- ✅ **Helios风格可视化**：serif字体、标准化网格、双y轴显示

## 主要功能

### 1. 子集群利用率分析
- 各子集群作业数量分布
- 总节点小时消耗统计
- 平均作业持续时间对比
- 平均节点使用量分析
- 作业成功率对比

### 2. 子集群24小时模式分析
- 各子集群的diurnal patterns
- 峰值和低谷时段识别
- 峰值/平均比计算
- 子集群间负载模式差异

### 3. 子集群作业特征分析
- 作业类型分布对比
- 持续时间统计对比
- 排队时间分析对比
- 资源需求模式差异

### 4. 资源效率分析
- 失败作业资源浪费统计
- 子集群成功率对比
- 资源利用效率评估

## 输出结果

### 可视化图表 (Helios风格)
1. `subcluster_job_distribution.pdf`: 子集群作业分布条形图
2. `subcluster_diurnal_comparison.pdf`: 子集群24小时模式对比图
3. `subcluster_success_rates.pdf`: 子集群成功率对比图
4. `subcluster_resource_waste.pdf`: 子集群资源浪费对比图

### 分析结果
- `subcluster_utilization`: 子集群利用率统计
- `subcluster_diurnal_patterns`: 24小时模式分析
- `subcluster_job_characteristics`: 作业特征对比
- `resource_efficiency`: 资源效率分析

## 关键发现 (基于Helios方法)

### Implication #2: 子集群负载不均
- 不同子集群之间资源分配不均衡
- 导致排队延迟和资源利用率低下并存
- 需要优化作业调度策略

### Implication #3: 专用vs通用资源
- GPU子集群主要服务深度学习工作负载
- CPU子集群服务传统HPC应用
- BIGMEM子集群服务内存密集型应用

## 架构对应关系

```
Helios架构:
Physical Clusters: Venus, Earth, Saturn, Uranus
├── Virtual Clusters (VCs) per group
└── Physical nodes

我们的架构:
Physical Cluster: HKUHPC-AI
├── Subclusters: CPU1-3, GPU1-3, BIGMEM
└── Physical nodes: cpu1-*, gpu1-*, etc.
```

## 使用方法

```python
from modules.02_cluster_comparison.cluster_analyzer import ClusterComparisonAnalyzer

# 初始化分析器
analyzer = ClusterComparisonAnalyzer(config, output_paths, visualizer)

# 执行分析
results = analyzer.analyze(dataframe)
```

## 配置要求

需要在 `analysis_config.yaml` 中启用：
```yaml
modules:
  cluster_comparison:
    enabled: true
```

## 数据要求

- `primary_subcluster`: 主要子集群标识（必需）
- `submit_time`: 作业提交时间（用于时间分析）
- `duration`: 作业持续时间（用于利用率分析）
- `actual_node_count`: 实际使用节点数（用于资源分析）
- `job_status_str`: 作业状态（用于成功率分析）

## 应用价值

1. **资源配置优化**: 识别过载和闲置的子集群
2. **负载均衡**: 优化作业在子集群间的分配
3. **容量规划**: 基于使用模式规划资源扩展
4. **调度策略**: 改进跨子集群的作业调度算法
