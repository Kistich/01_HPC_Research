# 作业特征分析模块 - Helios兼容版本

## 功能描述

本模块**严格按照Helios项目的作业分析方法**，深入分析HPC集群作业的特征。完全复现Helios论文Figure 4的分析方法和可视化风格，包括GPU数量分布、GPU时间分布和作业状态分析。

## 🆕 Helios兼容特性

### **对应Helios Figure 4**
- ✅ **GPU数量分布分析**：分析作业请求的GPU数量分布（对应Figure 4a）
- ✅ **GPU时间分布分析**：分析GPU小时消耗的分布（对应Figure 4b）
- ✅ **作业状态分布分析**：分析Pass/Failed/Killed状态的分布
- ✅ **双子图布局**：`(a) GPU Demand Distribution` + `(b) GPU Time Distribution`
- ✅ **Helios风格可视化**：对数坐标、CDF曲线、标准化样式

## 主要功能

### 1. 作业持续时间分析
- 持续时间分布统计（均值、中位数、百分位数）
- 按作业类型的持续时间对比
- 长尾分布特征分析
- CDF可视化

### 2. 排队时间分析
- 排队延迟统计分析
- 按作业类型的排队时间差异
- 排队时间分布可视化
- 调度效率评估

### 3. 资源需求分析
- 节点数需求分布
- GPU使用统计
- 单节点vs多节点作业对比
- 资源消耗模式

### 4. 作业状态分析
- 作业完成率统计
- 失败原因分析
- 按作业类型的成功率对比
- 状态分布可视化

### 5. 作业规模分析
- 作业规模分类（小型、中型、大型、超大型）
- 资源消耗分析（节点小时）
- 规模与成功率关系

## 输出结果

### 可视化图表 (Helios风格)
1. `job_duration_cdf.pdf`: 作业持续时间CDF图（对数刻度）
2. `queue_time_cdf.pdf`: 排队时间CDF图（对数刻度）
3. `job_status_distribution.pdf`: 作业状态分布条形图
4. `node_count_distribution.pdf`: 节点数分布条形图
5. `job_size_categories.pdf`: 作业规模分类图

### 分析结果
返回包含以下内容的字典：
- `duration_analysis`: 持续时间分析结果
- `queue_time_analysis`: 排队时间分析结果
- `resource_demand_analysis`: 资源需求分析结果
- `job_status_analysis`: 作业状态分析结果
- `job_size_analysis`: 作业规模分析结果

## 关键发现 (基于Helios方法)

### Implication #4: 资源使用模式
- 尽管单GPU作业数量占主导，但GPU资源主要被多GPU作业消耗
- 大型作业对集群利用率影响更大

### Implication #5: 作业收敛特征
- 许多深度学习训练作业可以提前收敛
- 调度器可以自动检测并停止作业以提高资源效率

### Implication #6: 调试作业特征
- 大量失败作业是用于调试目的且持续时间很短
- 但可能遭受长时间排队，影响开发效率

## 统计指标

### 持续时间统计
- 平均持续时间
- 中位数持续时间
- 50th, 90th, 95th, 99th百分位数
- 按作业类型分层统计

### 资源使用统计
- 平均节点数
- 单节点作业比例
- 多节点作业比例
- GPU小时消耗

### 成功率指标
- 整体作业成功率
- 按作业类型成功率
- 按作业规模成功率

## 使用方法

```python
from modules.04_job_characterization.job_analyzer import JobCharacterizationAnalyzer

# 初始化分析器
analyzer = JobCharacterizationAnalyzer(config, output_paths, visualizer)

# 执行分析
results = analyzer.analyze(dataframe)
```

## 配置要求

需要在 `analysis_config.yaml` 中启用：
```yaml
modules:
  job_characterization:
    enabled: true
```

## 数据要求

- `duration`: 作业持续时间（必需）
- `queue_time`: 排队时间（可选）
- `job_status_str`: 作业状态（必需）
- `actual_node_count`: 实际使用节点数（可选）
- `gpu_num`: GPU数量（可选）

## 可视化特点

### CDF图表特征
- 使用对数刻度显示长尾分布
- 标注关键百分位数（中位数、95th）
- Helios风格的颜色和布局

### 分布图特征
- 清晰的数值标注
- 合理的分类和排序
- 统一的视觉风格

## 应用价值

1. **调度策略优化**: 基于作业特征优化调度算法
2. **资源配置**: 合理配置不同类型节点比例
3. **用户指导**: 帮助用户优化作业提交策略
4. **系统改进**: 识别系统瓶颈和改进机会
