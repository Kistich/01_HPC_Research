# 时间模式分析模块 (Temporal Analysis Module)

## 功能描述

本模块基于Helios项目的时间分析方法，分析HPC集群作业的时间模式，包括24小时周期性、周模式和月度趋势。

## 主要功能

### 1. 24小时模式分析 (Diurnal Patterns)
- 每小时作业提交统计
- 峰值和低谷时段识别
- 峰值/平均比计算
- 按作业类型的24小时分布

### 2. 周模式分析 (Weekly Patterns)
- 星期几作业分布
- 工作日vs周末对比
- 周内峰值和低谷识别

### 3. 月度趋势分析 (Monthly Trends)
- 月度作业提交趋势
- 增长率计算
- 季节性模式识别

### 4. 作业类型时间分析
- CPU vs GPU作业的时间分布差异
- 不同作业类型的峰值时段
- 资源竞争时段分析

## 输出结果

### 可视化图表 (Helios风格)
1. `diurnal_pattern.pdf`: 24小时作业提交模式图
2. `cpu_gpu_diurnal_comparison.pdf`: CPU vs GPU作业24小时对比图
3. `weekly_pattern.pdf`: 周模式条形图
4. `monthly_trend.pdf`: 月度趋势时间序列图
5. `submission_heatmap.pdf`: 作业提交热力图（小时×星期几）

### 分析结果
返回包含以下内容的字典：
- `diurnal_patterns`: 24小时模式分析结果
- `weekly_patterns`: 周模式分析结果
- `monthly_trends`: 月度趋势分析结果
- `job_type_temporal`: 作业类型时间分析结果

## 关键发现 (基于Helios方法)

### Implication #1: 可预测的负载模式
- 集群利用率和作业提交率表现出明显的日周期模式
- 为预测性资源管理提供了机会
- 可以基于历史模式进行容量规划

### 时间模式特征
- **峰值时段**: 通常在工作时间（9-17点）
- **低谷时段**: 通常在夜间和周末
- **周期性**: 明显的24小时和7天周期

## 使用方法

```python
from modules.03_temporal_analysis.temporal_analyzer import TemporalAnalyzer

# 初始化分析器
analyzer = TemporalAnalyzer(config, output_paths, visualizer)

# 执行分析
results = analyzer.analyze(dataframe)
```

## 配置要求

需要在 `analysis_config.yaml` 中启用：
```yaml
modules:
  temporal_analysis:
    enabled: true
```

## 数据要求

- `submit_time`: 作业提交时间（必需）
- `job_type`: 作业类型（可选，用于对比分析）

## 可视化风格

严格按照Helios论文的图表风格：
- 使用相同的颜色方案
- 相同的图表类型和布局
- 相同的统计标注方式

## 应用价值

1. **资源调度优化**: 基于时间模式优化作业调度
2. **容量规划**: 预测峰值负载需求
3. **用户行为理解**: 了解用户使用习惯
4. **系统维护窗口**: 识别最佳维护时间
