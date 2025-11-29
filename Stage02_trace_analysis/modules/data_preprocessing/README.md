# 核心数据预处理模块 - Helios兼容版本 (Core Data Preprocessing Module - Helios Compatible)

## 功能描述

本模块是整个分析框架的核心，负责原始数据的加载、清洗、标准化和基础指标计算。**专门为Helios风格分析生成兼容的数据格式**，确保后续所有分析模块基于统一的预处理数据，避免重复计算，提高分析效率。

## 🆕 Helios兼容特性

### 1. 完整的Helios数据格式支持
- `cluster_log.csv`: 作业级别数据（完全兼容Helios格式）
- `cluster_sequence.csv`: 集群时间序列数据
- `cluster_throughput.csv`: 集群吞吐量数据
- `cluster_user.pkl`: 用户聚合数据

### 2. 智能exec_hosts解析
- 基于exec_hosts字段准确识别作业类型
- GPU作业：gpu1-, gpu2-, gpu3-主机
- CPU作业：cpu1-, cpu2-, cpu3-, bigmem-主机
- 自动过滤无效作业（training, reasoning等）

### 3. 准确的CPU数量计算
- 根据集群配置计算准确的CPU核心数
- CPU1/CPU2: 64核/节点，CPU3: 128核/节点，BIGMEM: 96核/节点
- 直接使用num_processors作为CPU核心数（符合Helios定义）

### 4. Helios状态映射
- DONE → Pass（正常完成）
- EXIT → Failed（异常退出）
- TIMEOUT → Killed（超时终止）

## 核心设计理念

### 1. 数据预处理前置化
- 在模块00中完成所有基础数据处理
- 一次处理，多模块复用
- 避免各模块重复加载和处理数据

### 2. 统一数据接口
- 标准化的数据格式和字段名
- 统一的数据访问接口
- 一致的数据质量保证

### 3. 智能缓存机制
- 预处理结果自动缓存
- 支持增量更新
- 快速数据加载

## 主要功能

### 1. 原始数据加载和清洗
- 加载HPC作业提交数据
- 时间字段标准化处理
- 数据质量检查和清洗
- 无效记录过滤

### 2. 数据标准化
- 集群名称标准化
- 作业状态标准化
- GPU/CPU作业分类
- 子集群识别和分类

### 3. 基础指标计算
- 作业持续时间计算
- 排队时间计算
- 资源消耗指标
- 成功率统计

### 4. 分类数据集准备
- GPU作业数据集
- CPU作业数据集
- 按子集群分类数据集
- 按作业类型分类数据集

### 5. 用户级别指标计算
- 用户资源消耗统计
- 用户作业模式分析
- 用户排队体验指标
- 用户成功率计算

### 6. 时间相关指标计算
- 24小时分布统计
- 周/月度分布分析
- 时间模式预计算
- 按作业类型的时间分布

## 数据处理流程

```
原始数据 → 数据清洗 → 标准化 → 增强处理 → 指标计算 → 分类准备 → 缓存保存
    ↓         ↓        ↓       ↓        ↓        ↓        ↓
  CSV文件   时间处理   字段映射  集群识别  基础统计  数据分割   PKL文件
```

## 输出数据结构

### 1. 预处理数据 (processed_data)
```python
{
    'enhanced_data': pd.DataFrame,  # 完整的增强数据集
    'job_type_datasets': {          # 按类型分类的数据集
        'gpu_jobs': pd.DataFrame,
        'cpu_jobs': pd.DataFrame,
        'gpu': pd.DataFrame,        # 按job_type分类
        'cpu': pd.DataFrame,
        # ... 其他作业类型
    }
}
```

### 2. 计算指标 (computed_metrics)
```python
{
    'basic_metrics': {              # 基础统计指标
        'total_jobs': int,
        'total_users': int,
        'date_range': dict,
        'job_type_distribution': dict,
        'subcluster_distribution': dict,
        'resource_consumption': dict,
        'gpu_statistics': dict
    },
    'user_metrics': {               # 用户级别指标
        'user_id': {
            'total_jobs': int,
            'node_hours': float,
            'success_rate': float,
            # ... 其他用户指标
        }
    },
    'temporal_metrics': {           # 时间相关指标
        'hourly_distribution': dict,
        'weekday_distribution': dict,
        'by_job_type': dict
    }
}
```

## 关键特性

### 1. 智能缓存
- 自动检测数据变化
- 增量更新机制
- 快速加载预处理结果

### 2. 数据质量保证
- 完整性检查
- 一致性验证
- 异常值处理

### 3. 灵活的数据访问
```python
# 获取特定数据集
gpu_data = preprocessor.get_dataset('gpu_jobs')
full_data = preprocessor.get_dataset('full')

# 获取计算指标
basic_stats = preprocessor.get_metrics('basic_metrics')
user_stats = preprocessor.get_metrics('user_metrics')
```

### 4. 高效的内存管理
- 按需加载数据
- 内存使用优化
- 大数据集支持

## 使用方法

### 基本使用
```python
from modules.data_preprocessing.data_preprocessor import HeliosCompatibleDataPreprocessor

# 初始化预处理器
preprocessor = HeliosCompatibleDataPreprocessor()

# 加载和预处理所有数据（自动生成Helios兼容格式）
processed_data = preprocessor.load_and_preprocess_all_data()

# 获取特定数据集
gpu_jobs = preprocessor.get_dataset('gpu_jobs')
cpu_jobs = preprocessor.get_dataset('cpu_jobs')

# 获取计算指标
basic_metrics = preprocessor.get_metrics('basic_metrics')
user_metrics = preprocessor.get_metrics('user_metrics')

# 🆕 获取Helios兼容数据
helios_data = processed_data['helios_data']
cluster_log = helios_data['cluster_log']        # 7,956,871条作业记录
cluster_sequence = helios_data['cluster_sequence']  # 时间序列数据
cluster_throughput = helios_data['cluster_throughput']  # 吞吐量数据
cluster_user = helios_data['cluster_user']      # 347个用户统计
```

### 🆕 Helios数据生成器使用
```python
from modules.data_preprocessing.helios_data_generator import HeliosDataGenerator

# 创建生成器
generator = HeliosDataGenerator()

# 生成Helios格式数据
helios_data = generator.generate_all_helios_data(preprocessed_df)

# 保存到目录（自动保存为标准Helios格式）
generator.save_to_directory('data/processed')

# 获取数据摘要
summary = generator.get_data_summary()
print(f"cluster_log: {summary['cluster_log']['records']:,} 条记录")
print(f"cluster_user: {summary['cluster_user']['records']} 个用户")
```

### 🆕 强制重新生成Helios数据
```python
# 强制重新处理数据以更新Helios格式
processed_data = preprocessor.load_and_preprocess_all_data(force_reload=True)
```

### 强制重新处理
```python
# 强制重新处理数据（忽略缓存）
processed_data = preprocessor.load_and_preprocess_all_data(force_reload=True)
```

## 配置要求

需要在 `config/cluster_config.yaml` 中配置：
```yaml
# 集群配置
subclusters:
  CPU1:
    node_count: 110
    node_type: "cpu"
  GPU1:
    node_count: 50
    node_type: "gpu"

# 作业分类规则
job_classification:
  gpu_keywords: ["gpu", "cuda", "nvidia"]
  cpu_keywords: ["cpu", "intel", "amd"]
```

## 数据要求

### 必需字段
- `submit_time`: 作业提交时间
- `start_time`: 作业开始时间  
- `end_time`: 作业结束时间
- `job_status_str`: 作业状态
- `exec_hosts`: 执行主机信息

### 可选字段
- `user_id`: 用户标识
- `gpu_num`: GPU数量
- `actual_node_count`: 实际节点数
- `cluster_name`: 集群名称

## 性能优化

### 1. 缓存机制
- 预处理结果自动缓存到 `data/processed/`
- 支持增量更新
- 智能缓存失效检测

### 2. 内存优化
- 分块处理大数据集
- 按需加载数据
- 内存使用监控

### 3. 计算优化
- 向量化操作
- 并行处理支持
- 算法优化

## 扩展性

### 1. 新数据源支持
- 可扩展的数据加载器
- 灵活的数据格式适配
- 多数据源合并

### 2. 新指标计算
- 可插拔的指标计算模块
- 自定义指标支持
- 指标依赖管理

### 3. 新数据类型
- 支持新的作业类型
- 动态数据分类
- 自适应数据处理

## 应用价值

1. **提高分析效率**: 一次预处理，多模块复用
2. **保证数据一致性**: 统一的数据标准和质量
3. **简化模块开发**: 标准化的数据接口
4. **支持大规模数据**: 高效的数据处理和缓存机制
5. **便于维护扩展**: 模块化设计，易于扩展和维护
