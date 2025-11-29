# GPU Graphics Clock Speed Analyzer

该分析工具专为分析GPU服务器的图形时钟速度数据而设计。

## 功能概述

该工具提供以下分析功能：

1. **时间模式分析**：
   - 月度时钟速度模式比较
   - 周度时钟速度模式比较
   - 日内时钟速度模式分析

2. **高时钟速度时段识别**：
   - 查找并记录持续高时钟速度的时段（默认阈值1500 MHz）
   - 按服务器组和具体服务器记录时钟速度峰值

3. **热力图生成**：
   - 按周日和小时生成详细的时钟速度热力图
   - 直观展示时钟速度变化模式

4. **时钟速度效率分析**：
   - 分析时钟速度分布情况
   - 研究时钟速度状态转换（大幅增加、小幅增加、稳定、小幅减少、大幅减少）

5. **跨组比较**：
   - 不同服务器组的关键时钟速度指标对比
   - 生成比较图表和数据报表

6. **综合时钟速度报告**：
   - 生成详细的GPU时钟速度分析报告
   - 根据数据提供针对性优化建议

## 使用方法

1. 确保已安装所需的Python库：
   ```
   pip install pandas matplotlib seaborn numpy
   ```

2. 运行分析脚本：
   ```
   python gpu_clockspeed_analyzer.py
   ```

3. 分析结果将保存在以下目录中：
   - `figures/`: 包含所有生成的图表
   - `output/`: 包含详细的分析报告和高时钟速度时段记录

## 输出说明

- **figures/all_groups_monthly_clockspeed.png**: 所有组的月度时钟速度对比
- **figures/all_groups_weekly_clockspeed.png**: 所有组的周度时钟速度对比
- **figures/{group}_daily_clockspeed.png**: 各组的日内时钟速度模式
- **figures/{group}_clockspeed_heatmap.png**: 各组的时钟速度热力图
- **figures/{group}_clockspeed_distribution.png**: 各组的时钟速度分布
- **figures/{group}_clockspeed_transitions.png**: 各组的时钟速度状态转换分析
- **figures/group_comparison_avg_clockspeed.png**: 组间平均时钟速度对比
- **figures/group_comparison_high_clockspeed.png**: 组间高时钟速度时长对比
- **output/high_clockspeed_periods.txt**: 高时钟速度时段详细记录
- **output/gpu_clockspeed_report.txt**: 综合时钟速度分析报告
- **output/clockspeed_group_comparison.txt**: 组间时钟速度指标详细对比

## 自定义分析

如需修改分析参数（如高时钟速度阈值），可编辑脚本中的相关参数：

```python
# 修改高时钟速度阈值（默认1500 MHz）
analyzer.find_high_clockspeed_periods(threshold=1800)
```

## 时钟速度效率优化

该分析工具可帮助识别：
- 时钟速度异常高的时段和服务器（可能表明GPU负载重）
- 时钟速度不稳定的服务器组（可能存在温度或功耗限制）
- 时钟速度频繁变化的使用模式（可能影响性能稳定性）

通过这些分析，可以制定更高效的GPU工作负载策略，提高资源利用效率和稳定性。

## 数据结构说明

该分析工具假设输入的Excel数据表格具有以下格式：
- 每个工作表对应一个服务器
- 每个工作表包含至少三列：`timestamp`（时间戳）、`value`（时钟速度值，单位MHz）和`metric`（指标名称）
