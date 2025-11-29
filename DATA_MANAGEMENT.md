# 📊 数据管理说明

## 📋 概述

本项目包含大量 HPC 作业数据和处理结果（总计 52GB），这些数据**不包含在 Git 仓库中**。

本文档说明如何获取、管理和使用这些数据。

---

## 📁 数据目录结构

```
01_HPC_Research/
├── Stage00_HPC_raw_data/              # 13GB - 原始数据
│   ├── jobinfo_20250224_113534.csv    # 5.1GB - 作业信息
│   ├── prometheus_metrics_*.xlsx      # 多个文件 - Prometheus 监控数据
│   └── Stage01_Generation_analysis/   # 1.8GB - 代际分析
│
├── Stage01_data_filter_preprocess/    # 28GB - 数据处理结果
│   ├── full_processing_outputs/       # 完整处理输出
│   │   ├── stage1_generation_filter/  # 10GB - 代际过滤
│   │   ├── stage2_time_processing/    # 5GB - 时间处理
│   │   ├── stage3_user_inference/     # 5GB - 用户推断
│   │   ├── stage5_intelligent_sampling/ # 4GB - 智能采样
│   │   └── stage6_data_standardization/ # 4GB - 数据标准化
│   └── stable_processing_outputs/     # 稳定处理输出
│
├── Stage02_trace_analysis/            # 7.4GB - Trace 分析
│   ├── data/processed/                # 6.8GB - 处理后数据
│   │   ├── preprocessed_data.pkl      # 6.8GB - 预处理数据
│   │   └── helios_format/             # 634MB - Helios 格式
│   └── results/                       # 分析结果
│
└── Stage03_simulator_CES_DRS/         # 3.8GB - 仿真和预测
    ├── 4_Simulator/
    │   ├── simulation_results/        # 仿真结果
    │   └── ces_experiment_results/    # CES 实验结果
    └── 5_Prediction_Model/
        └── 1_data_preparation/        # 3.5GB - 训练数据
            └── cpu1_jobs_2024.csv     # 3.5GB - CPU1 作业数据
```

---

## 🎯 数据获取方式

### 方案1：从原始位置获取（推荐）

如果你有访问原始数据源的权限：

```bash
# 数据保存在本地外部硬盘
SOURCE_DIR="/Volumes/EXTERNAL_US/backup2/01_HPC_Research"

# 克隆代码仓库
git clone https://github.com/YOUR_USERNAME/01_HPC_Research.git

# 复制数据文件（根据需要选择）
cp -r "$SOURCE_DIR/Stage00_HPC_raw_data" 01_HPC_Research/
cp -r "$SOURCE_DIR/Stage01_data_filter_preprocess/full_processing_outputs" 01_HPC_Research/Stage01_data_filter_preprocess/
# ... 其他数据目录
```

### 方案2：从云存储下载

**（待实施）** 如果数据已上传到云存储：

```bash
# 示例：从 Google Drive 下载
# 1. 访问共享链接
# 2. 下载数据压缩包
# 3. 解压到对应目录

# 或使用命令行工具
# gdown <GOOGLE_DRIVE_FILE_ID>
# unzip data.zip -d 01_HPC_Research/
```

### 方案3：重新生成数据

如果你有原始 HPC 系统访问权限，可以重新收集数据：

```bash
# 1. 收集作业信息
# 从 HPC 系统导出作业数据

# 2. 收集 Prometheus 监控数据
# 从 Prometheus 导出指标数据

# 3. 运行数据处理流程
cd 01_HPC_Research/Stage01_data_filter_preprocess
bash run_stable_processing.sh
```

---

## 📦 最小数据集（快速开始）

如果你只想运行仿真器而不需要完整数据：

### 需要的文件（~100MB）

```
01_HPC_Research/
└── Stage03_simulator_CES_DRS/
    └── 4_Simulator/
        └── trace_files/
            ├── helios_500jobs.csv      # 500 作业 trace
            └── helios_2500jobs.csv     # 2500 作业 trace
```

### 获取方式

```bash
# 这些文件已包含在 Git 仓库中（如果 < 100MB）
git clone https://github.com/YOUR_USERNAME/01_HPC_Research.git
cd 01_HPC_Research/Stage03_simulator_CES_DRS/4_Simulator

# 直接运行仿真
python3 run_all_simulations.py
```

---

## 🔧 数据处理流程

### 完整流程（需要所有数据）

```bash
# 1. 数据过滤和预处理（Stage01）
cd Stage01_data_filter_preprocess
bash run_stable_processing.sh

# 2. Trace 分析（Stage02）
cd ../Stage02_trace_analysis
python3 helios_trace_converter.py

# 3. 预测模型训练（Stage03/5_Prediction_Model）
cd ../Stage03_simulator_CES_DRS/5_Prediction_Model
python3 CES_prediction.py --cluster CPU1

# 4. 仿真实验（Stage03/4_Simulator）
cd ../4_Simulator
python3 run_ces_experiments.py
```

### 快速流程（仅需 trace 文件）

```bash
# 直接运行仿真
cd Stage03_simulator_CES_DRS/4_Simulator
python3 run_all_simulations.py
python3 run_ces_experiments.py
```

---

## 💾 数据存储建议

### 本地存储

**推荐配置：**
- **SSD：** 存放代码和小型数据（< 10GB）
- **HDD：** 存放大型数据文件（> 10GB）
- **外部硬盘：** 备份和归档

**目录结构：**
```
/Users/YOUR_NAME/Projects/
└── 01_HPC_Research/          # Git 仓库（代码）

/Volumes/EXTERNAL_US/
└── HPC_Data/                 # 数据文件
    ├── Stage00_HPC_raw_data/
    ├── Stage01_outputs/
    └── Stage02_results/
```

### 云存储

**推荐服务：**

| 服务 | 免费额度 | 适用场景 |
|------|----------|----------|
| Google Drive | 15GB | 小型数据集 |
| OneDrive | 5GB | 文档和配置 |
| 百度网盘 | 免费 | 大文件（速度慢）|
| 阿里云 OSS | 按量付费 | 团队协作 |
| AWS S3 | 按量付费 | 生产环境 |

---

## 📊 数据文件说明

### Stage00_HPC_raw_data（13GB）

**主要文件：**
- `jobinfo_20250224_113534.csv` (5.1GB)
  - HPC 作业信息
  - 包含：作业ID、用户、队列、资源、时间等

- `prometheus_metrics_*.xlsx` (多个文件)
  - Prometheus 监控数据
  - 包含：CPU、内存、温度、能耗等指标

**用途：**
- 数据分析和统计
- 模型训练
- Trace 生成

### Stage01_data_filter_preprocess（28GB）

**主要输出：**
- `stage1_generation_filter/` - 代际过滤结果
- `stage2_time_processing/` - 时间处理结果
- `stage3_user_inference/` - 用户推断结果
- `stage5_intelligent_sampling/` - 智能采样结果
- `stage6_data_standardization/` - 标准化数据

**用途：**
- 数据清洗和预处理
- 特征工程
- 数据质量提升

### Stage02_trace_analysis（7.4GB）

**主要文件：**
- `preprocessed_data.pkl` (6.8GB) - 预处理数据
- `helios_format/cluster_log.csv` (634MB) - Helios 格式 trace

**用途：**
- Trace 分析
- 仿真器输入

### Stage03_simulator_CES_DRS（3.8GB）

**主要文件：**
- `cpu1_jobs_2024.csv` (3.5GB) - CPU1 作业数据
- `simulation_results/` - 仿真结果
- `ces_experiment_results/` - CES 实验结果

**用途：**
- 预测模型训练
- 仿真实验

---

## 🔒 数据安全

### 敏感信息处理

**注意：** 原始数据可能包含敏感信息：
- ✅ 用户名
- ✅ 作业内容
- ✅ 系统配置

**建议：**
1. **脱敏处理：** 移除或匿名化用户信息
2. **访问控制：** 限制数据访问权限
3. **加密存储：** 对敏感数据加密

### 数据备份

**推荐策略：**
- **3-2-1 原则：**
  - 3 份副本
  - 2 种存储介质
  - 1 份异地备份

**实施：**
```bash
# 本地备份
rsync -av /Volumes/EXTERNAL_US/backup2/01_HPC_Research/ /Volumes/BACKUP/01_HPC_Research/

# 云备份
# 上传到云存储服务
```

---

## ❓ 常见问题

### Q1: 我需要下载所有数据吗？

**A:** 不需要。根据你的需求：
- **只运行仿真：** 只需 trace 文件（~100MB）
- **训练模型：** 需要 Stage03/5_Prediction_Model 数据（~4GB）
- **完整研究：** 需要所有数据（52GB）

### Q2: 数据文件太大，如何处理？

**A:** 可以：
1. 使用采样数据（Stage01 的智能采样结果）
2. 只使用部分时间段的数据
3. 压缩数据文件

### Q3: 如何验证数据完整性？

**A:** 使用校验和：
```bash
# 生成校验和
md5 jobinfo_20250224_113534.csv > jobinfo.md5

# 验证
md5 -c jobinfo.md5
```

---

## 📞 联系方式

如果你需要访问完整数据集，请联系：

- **GitHub Issues:** https://github.com/YOUR_USERNAME/01_HPC_Research/issues
- **Email:** your.email@example.com

---

**最后更新：** 2025-11-29

