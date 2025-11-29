# GPU Metrics Comprehensive Analyzer

## 概述 (Overview)
这是一个全面的GPU指标分析工具，用于分析多个GPU相关指标的Excel文件数据，包括GPU利用率、内存利用率、时钟速度、功耗等。该工具可自动加载和处理数据，生成详细的分析报告和可视化图表，帮助理解GPU使用模式和性能特征。

This is a comprehensive GPU metrics analysis tool designed to analyze Excel file data for multiple GPU-related metrics, including GPU utilization, memory utilization, clock speeds, power draw, and more. The tool automatically loads and processes data, generates detailed analysis reports and visualizations to help understand GPU usage patterns and performance characteristics.

## 功能特点 (Features)

- **统一分析**：在单次执行中处理所有九种GPU指标
- **全面指标覆盖**：
  - GPU利用率 (GPU Utilization)
  - 内存利用率 (Memory Utilization)
  - 图形时钟速度 (Graphics Clock Speed)
  - 内存时钟速度 (Memory Clock Speed)
  - SM时钟速度 (SM Clock Speed)
  - 视频时钟速度 (Video Clock Speed)
  - 功耗 (Power Draw)
  - 内存分配 (Memory Allocation)
  - 平均利用率 (Average Utilization，跨8张GPU卡)
- **详细分析方法**：
  - 月度和周度模式比较
  - 每个服务器组的日常模式分析
  - 高阈值期间检测
  - 按日/小时生成热图
  - 值分布和状态转换分析
  - 服务器组比较和可视化

---

- **Unified Analysis**: Process all nine GPU metrics in a single execution
- **Comprehensive Metrics Coverage**:
  - GPU Utilization
  - Memory Utilization
  - Graphics Clock Speed
  - Memory Clock Speed
  - SM Clock Speed
  - Video Clock Speed
  - Power Draw
  - Memory Allocation
  - Average Utilization (across 8 GPU cards)
- **Detailed Analysis Methods**:
  - Monthly and weekly pattern comparison
  - Daily pattern analysis for each server group
  - High threshold period detection
  - Heatmap generation showing usage patterns by day/hour
  - Value distribution and state transition analysis
  - Server group comparison with visualizations

## 性能优化 (Performance Optimization)

该分析脚本利用多进程并行处理技术来提高执行效率：

- **并行指标处理**：同时分析多个GPU指标，而不是顺序处理
- **自动资源分配**：根据系统可用CPU核心数动态调整并行进程数
- **默认配置**：使用系统可用CPU核心的一半进行数据处理，确保系统其他任务不受影响
- **性能提升**：与顺序处理相比，在多核系统上可以实现3-5倍的速度提升

在64核CPU服务器上，脚本将自动使用约18-32个并行进程，使处理时间从15-25分钟缩短至约3-7分钟。

## 安装指南 (Installation)

1. 克隆或下载此代码库
2. 安装所需依赖：

```bash
pip install -r requirements.txt
```

## 使用方法 (Usage)

运行主分析脚本：

```bash
python comprehensive_gpu_analyzer.py
```

该脚本将自动：
1. 加载所有九个GPU指标的Excel数据文件
2. 为每个指标执行全面分析
3. 生成详细报告和可视化图表
4. 创建汇总报告

## 输出说明 (Output)

分析结果按以下结构组织：

```
Server_analysis/GPU/Analysis_Results/
├── prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520/
│   ├── figures/           # 包含所有图表和可视化
│   └── reports/           # 包含详细分析报告
├── prometheus_metrics_data_GPU使用率_MemoryUtilization_20250221_105207/
│   ├── figures/
│   └── reports/
├── prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235/
...
└── Combined_Analysis/     # 包含汇总报告
```

每个Excel文件对应一个独立的分析目录，包含：
- `figures`文件夹：包含月度、周度、日常模式图表，热图和分布图
- `reports`文件夹：包含高阈值期间报告，组比较和详细分析

## 数据要求 (Data Requirements)

该工具需要以下Excel文件作为输入：
- `prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx`
- `prometheus_metrics_data_GPU使用率_MemoryUtilization_20250221_105207.xlsx`
- `prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235.xlsx`
- `prometheus_metrics_data_GPU使用率_MemoryClockSpeed_20250221_113253.xlsx`
- `prometheus_metrics_data_GPU使用率_SMClockSpeed_20250221_112610.xlsx`
- `prometheus_metrics_data_GPU使用率_VideoClockSpeed_20250221_111923.xlsx`
- `prometheus_metrics_data_GPU使用率_当前GPU卡的PowerDraw_20250221_110550.xlsx`
- `prometheus_metrics_data_GPU使用率_MemoryAllocation_20250221_105847.xlsx`
- `prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx`

每个文件应包含时间戳、服务器名称和对应的指标值。

## 系统要求 (System Requirements)

- Python 3.6+
- 足够的内存处理大型Excel文件
- 足够的磁盘空间存储生成的报告和图表

## 自定义阈值 (Customizing Thresholds)

可以通过修改`GPUMetricsAnalyzer`类中的`thresholds`字典来自定义各个指标的阈值。

## 技术实现 (Technical Implementation)

- **数据加载与处理**：使用Pandas加载Excel数据并进行转换和整理
- **数据分析**：应用统计方法分析各种模式和特征
- **可视化**：使用Matplotlib和Seaborn创建图表和可视化
- **报告生成**：生成详细的文本报告和汇总分析

## 许可证 (License)

© 2025 liuhongbin
