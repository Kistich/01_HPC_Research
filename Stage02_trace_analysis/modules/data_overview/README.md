# 数据概览模块 (Data Overview Module)

## 功能描述

本模块提供HPC集群作业数据的基本统计和质量评估，类似于Helios项目的数据集概览分析。

## 主要功能

### 1. 基本统计信息
- 总作业数量
- 总用户数量
- 时间跨度分析
- 数据列信息统计

### 2. 数据质量评估
- **完整性检查**: 各字段缺失值统计
- **有效性检查**: 无效数据识别（如负持续时间）
- **一致性检查**: 时间逻辑一致性验证

### 3. 时间覆盖分析
- 月度数据分布
- 星期几分布模式
- 小时级分布统计

### 4. 作业状态分析
- 作业状态分布统计
- 成功率计算
- 失败原因分析

### 5. 资源分布概览
- 作业类型分布（CPU/GPU）
- 子集群使用分布
- 节点数统计
- GPU使用统计

## 输出结果

### 报告文件
- `data_overview_report.txt`: 详细的数据概览报告

### 分析结果
返回包含以下内容的字典：
- `basic_statistics`: 基本统计信息
- `data_quality`: 数据质量评估结果
- `time_coverage`: 时间覆盖分析
- `job_status_distribution`: 作业状态分布
- `resource_distribution`: 资源分布统计

## 使用方法

```python
from modules.01_data_overview.data_overview_analyzer import DataOverviewAnalyzer

# 初始化分析器
analyzer = DataOverviewAnalyzer(config, output_paths)

# 执行分析
results = analyzer.analyze(dataframe)
```

## 配置要求

需要在 `analysis_config.yaml` 中启用：
```yaml
modules:
  data_overview:
    enabled: true
```

## 依赖模块

- `utils.data_loader`: 数据加载
- `utils.cluster_identifier`: 子集群识别

## 注意事项

1. 本模块主要用于数据质量检查和基本统计
2. 不生成可视化图表，专注于数据报告
3. 为后续分析模块提供数据质量基线
