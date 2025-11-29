# GPU Utilization Analyzer

该分析工具专为分析GPU服务器的利用率数据而设计。

## 功能概述

该工具提供以下分析功能：

1. **时间模式分析**：
   - 月度利用率模式比较
   - 周度利用率模式比较
   - 日内利用率模式分析

2. **高利用率时段识别**：
   - 查找并记录持续高利用率的时段（默认阈值80%）
   - 按服务器组和具体服务器记录利用率峰值

3. **热力图生成**：
   - 按周日和小时生成详细的利用率热力图
   - 直观展示利用率模式

4. **利用率效率分析**：
   - 分析利用率分布情况
   - 研究利用率状态转换

5. **跨组比较**：
   - 不同服务器组的关键利用率指标对比
   - 生成比较图表和数据报表

6. **综合利用率报告**：
   - 生成详细的GPU利用率分析报告
   - 根据数据提供针对性优化建议

## 使用方法

1. 确保已安装所需的Python库：
   ```
   pip install pandas matplotlib seaborn numpy
   ```

2. 运行分析脚本：
   ```
   python gpu_utilization_analyzer.py
   ```

3. 分析结果将保存在以下目录中：
   - `figures/`: 包含所有生成的图表
   - `output/`: 包含详细的分析报告和高利用率时段记录

## 输出说明

- **figures/all_groups_monthly_utilization.png**: 所有组的月度利用率对比
- **figures/all_groups_weekly_utilization.png**: 所有组的周度利用率对比
- **figures/{group}_daily_utilization.png**: 各组的日内利用率模式
- **figures/{group}_utilization_heatmap.png**: 各组的利用率热力图
- **figures/{group}_utilization_distribution.png**: 各组的利用率分布
- **figures/{group}_utilization_transitions.png**: 各组的利用率状态转换分析
- **figures/group_comparison_avg_utilization.png**: 组间平均利用率对比
- **figures/group_comparison_high_utilization.png**: 组间高利用率时长对比
- **output/high_utilization_periods.txt**: 高利用率时段详细记录
- **output/gpu_utilization_report.txt**: 综合利用率分析报告

## 自定义分析

如需修改分析参数（如高利用率阈值），可编辑脚本中的相关参数：

```python
# 修改高利用率阈值（默认80%）
analyzer.find_high_utilization_periods(threshold=70)
```

## 利用率效率优化

该分析工具可帮助识别：
- 利用率异常高的时段和服务器
- 利用率不稳定的服务器组
- 利用率效率低的使用模式

通过这些分析，可以制定更高效的GPU使用策略，提高资源利用效率。

## 数据结构说明

该分析工具假设输入的Excel数据表格具有以下格式：
- 每个工作表对应一个服务器
- 每个工作表包含至少三列：`timestamp`（时间戳）、`value`（利用率值，0-100%）和`metric`（指标名称）
