# 资源利用率分析模块 (Resource Utilization Module)

## 功能描述

本模块基于Helios项目的资源分析方法，深入分析CPU/GPU利用率、负载均衡、容量规划等关键指标，为集群优化提供数据支撑。

## 主要功能

### 1. 整体资源利用率分析
- 总资源消耗统计（节点小时）
- 按作业类型的资源分配
- 理论容量vs实际使用对比
- 平均作业持续时间和节点使用量

### 2. 子集群资源利用率分析
- 各子集群的利用率计算
- 理论容量vs实际消耗对比
- 子集群间利用率差异
- 节点类型效率分析

### 3. 时间维度利用率分析
- 24小时资源需求模式
- 每日利用率趋势
- 峰值和低谷时段识别
- 时间相关的容量规划

### 4. 负载均衡分析
- 子集群间负载分布
- 负载不平衡程度量化
- 负载方差和标准差计算
- 均衡性评估指标

### 5. 容量需求分析
- 峰值需求vs当前容量
- 容量利用率评估
- 扩容或缩容建议
- 子集群级别的容量规划

## 输出结果

### 可视化图表 (Helios风格)
1. `subcluster_utilization_rates.pdf`: 子集群利用率对比图
2. `hourly_resource_demand.pdf`: 24小时资源需求图
3. `load_distribution.pdf`: 负载分布图
4. `capacity_vs_demand.pdf`: 容量vs需求对比图

### 分析结果
- `overall_utilization`: 整体利用率统计
- `subcluster_utilization`: 子集群利用率分析
- `temporal_utilization`: 时间维度利用率
- `load_balancing`: 负载均衡分析
- `capacity_analysis`: 容量需求分析

## 关键指标

### 利用率计算公式
```
利用率 = 实际消耗节点小时 / 理论容量节点小时 × 100%
理论容量 = 节点数 × 24小时 × 分析天数
```

### 负载均衡指标
- **负载方差**: 衡量子集群间负载分布的离散程度
- **不平衡分数**: 标准差/理想平均值，越低越均衡
- **基尼系数**: 负载分配的不平等程度

### 容量评估标准
- **>90%利用率**: 建议扩容
- **<30%利用率**: 可能过度配置
- **30-90%**: 容量适当

## 容量规划建议

### 扩容建议条件
1. 峰值利用率持续超过90%
2. 排队时间显著增加
3. 作业失败率上升

### 优化建议条件
1. 平均利用率低于30%
2. 负载严重不均衡
3. 某些子集群长期闲置

## 使用方法

```python
from modules.06_resource_utilization.resource_analyzer import ResourceUtilizationAnalyzer

# 初始化分析器（需要集群配置）
analyzer = ResourceUtilizationAnalyzer(config, output_paths, visualizer, cluster_config)

# 执行分析
results = analyzer.analyze(dataframe)
```

## 配置要求

需要在 `analysis_config.yaml` 中启用：
```yaml
modules:
  resource_utilization:
    enabled: true
```

需要在 `cluster_config.yaml` 中提供子集群配置：
```yaml
subclusters:
  CPU1:
    node_count: 110
    node_type: "cpu"
  GPU1:
    node_count: 50
    node_type: "gpu"
```

## 数据要求

- `duration`: 作业持续时间（必需）
- `actual_node_count`: 实际使用节点数（必需）
- `primary_subcluster`: 主要子集群标识（必需）
- `submit_time`: 提交时间（用于时间分析）
- `job_type`: 作业类型（用于分类分析）

## 理论容量计算

### CPU子集群
```
总CPU节点 = CPU1 + CPU2 + CPU3 + BIGMEM
理论CPU容量 = 总CPU节点 × 24 × 分析天数
```

### GPU子集群
```
总GPU节点 = GPU1 + GPU2 + GPU3
理论GPU容量 = 总GPU节点 × 24 × 分析天数
```

## 性能指标

### 效率指标
- **资源利用率**: 实际使用/理论容量
- **作业吞吐量**: 单位时间完成作业数
- **平均周转时间**: 提交到完成的平均时间

### 均衡指标
- **负载标准差**: 子集群间负载分布的标准差
- **变异系数**: 标准差/平均值
- **最大最小比**: 最高负载/最低负载

## 应用价值

1. **容量规划**: 基于历史数据预测未来容量需求
2. **资源优化**: 识别低效使用的资源并重新分配
3. **负载均衡**: 优化作业调度以改善负载分布
4. **成本控制**: 避免过度配置，提高投资回报率
5. **性能调优**: 识别性能瓶颈并制定改进策略

## 监控建议

### 关键监控指标
- 实时利用率
- 排队长度
- 作业完成率
- 资源浪费率

### 告警阈值
- 利用率 > 95%: 容量告警
- 排队时间 > 24小时: 调度告警
- 失败率 > 10%: 质量告警
