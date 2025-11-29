# GPU Power Draw Analyzer

该分析工具专为分析GPU服务器的功耗数据而设计。

## 功能概述

该工具提供以下分析功能：

1. **时间模式分析**：
   - 月度功耗模式比较
   - 周度功耗模式比较
   - 日内功耗模式分析

2. **高功耗时段识别**：
   - 查找并记录持续高功耗的时段（默认阈值220W）
   - 按服务器组和具体服务器记录功耗峰值

3. **热力图生成**：
   - 按周日和小时生成详细的功耗热力图
   - 直观展示功耗模式

4. **功耗效率分析**：
   - 分析功耗分布情况
   - 研究功耗状态转换

5. **跨组比较**：
   - 不同服务器组的关键功耗指标对比
   - 生成比较图表和数据报表

6. **综合功耗报告**：
   - 生成详细的GPU功耗分析报告
   - 根据数据提供针对性优化建议

## 使用方法

1. 确保已安装所需的Python库：
   ```
   pip install pandas matplotlib seaborn numpy
   ```

2. 运行分析脚本：
   ```
   python gpu_powerdraw_analyzer.py
   ```

3. 分析结果将保存在以下目录中：
   - `figures/`: 包含所有生成的图表
   - `output/`: 包含详细的分析报告和高功耗时段记录

## 输出说明

- **figures/all_groups_monthly_powerdraw.png**: 所有组的月度功耗对比
- **figures/all_groups_weekly_powerdraw.png**: 所有组的周度功耗对比
- **figures/{group}_daily_powerdraw.png**: 各组的日内功耗模式
- **figures/{group}_powerdraw_heatmap.png**: 各组的功耗热力图
- **figures/{group}_power_distribution.png**: 各组的功耗分布
- **figures/{group}_power_transitions.png**: 各组的功耗状态转换分析
- **figures/group_comparison_avg_power.png**: 组间平均功耗对比
- **figures/group_comparison_high_power.png**: 组间高功耗时长对比
- **output/high_power_periods.txt**: 高功耗时段详细记录
- **output/gpu_power_report.txt**: 综合功耗分析报告

## 自定义分析

如需修改分析参数（如高功耗阈值），可编辑脚本中的相关参数：

```python
# 修改高功耗阈值（默认220W）
analyzer.find_high_power_periods(threshold=200)
```

## 功耗效率优化

该分析工具可帮助识别：
- 功耗异常高的时段和服务器
- 功耗不稳定的服务器组
- 功耗效率低的使用模式

通过这些分析，可以制定更高效的GPU使用策略，降低能耗成本。

## 数据结构说明

该分析工具假设输入的Excel数据表格具有以下格式：
- 每个工作表对应一个服务器
- 每个工作表包含至少三列：`timestamp`（时间戳）、`value`（功耗值，单位为瓦特）和`metric`（指标名称）
