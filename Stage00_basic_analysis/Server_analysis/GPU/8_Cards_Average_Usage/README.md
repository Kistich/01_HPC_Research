# GPU 8 Cards Average Usage Analyzer

该分析工具专为分析服务器8张GPU卡的平均使用率数据而设计。

## 功能概述

该工具提供以下分析功能：

1. **时间模式分析**：
   - 月度使用率模式比较
   - 周度使用率模式比较
   - 日内使用率模式分析

2. **高使用率时段识别**：
   - 查找并记录持续高GPU使用率的时段（默认阈值80%）
   - 按服务器组和具体服务器记录使用率峰值

3. **热力图生成**：
   - 按周日和小时生成详细的使用率热力图
   - 直观展示使用率模式

4. **跨组比较**：
   - 不同服务器组的关键指标对比
   - 生成比较图表和数据报表

5. **综合利用率报告**：
   - 生成详细的GPU利用率分析报告
   - 根据数据提供针对性优化建议

## 使用方法

1. 确保已安装所需的Python库：
   ```
   pip install pandas matplotlib seaborn numpy
   ```

2. 运行分析脚本：
   ```
   python gpu_8cards_analyzer.py
   ```

3. 分析结果将保存在以下目录中：
   - `figures/`: 包含所有生成的图表
   - `output/`: 包含详细的分析报告和高使用率时段记录

## 输出说明

- **figures/all_groups_monthly_gpu_usage.png**: 所有组的月度使用率对比
- **figures/all_groups_weekly_gpu_usage.png**: 所有组的周度使用率对比
- **figures/{group}_daily_gpu_usage.png**: 各组的日内使用率模式
- **figures/{group}_heatmap.png**: 各组的使用率热力图
- **figures/group_comparison_avg.png**: 组间平均使用率对比
- **figures/group_comparison_high_usage.png**: 组间高使用率时长对比
- **output/high_usage_periods.txt**: 高使用率时段详细记录
- **output/gpu_utilization_report.txt**: 综合利用率分析报告

## 自定义分析

如需修改分析参数（如高使用率阈值），可编辑脚本中的相关参数：

```python
# 修改高使用率阈值（默认80%）
analyzer.find_high_usage_periods(threshold=70)
```

## 数据结构说明

该分析工具假设输入的Excel数据表格具有以下格式：
- 每个工作表对应一个服务器
- 每个工作表包含至少三列：`timestamp`（时间戳）、`value`（使用率值）和`metric`（指标名称）
