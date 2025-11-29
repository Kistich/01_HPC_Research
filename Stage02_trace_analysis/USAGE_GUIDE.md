# Stage02 Trace Analysis 使用指南

## 📋 概述

`main.py` 是 Stage02 Trace Analysis 的统一入口，整合了所有分析功能，包括：

### 核心分析模块（Helios风格）
1. **Philly Comparison** - 与Philly数据集对比分析
2. **Cluster Characterization** - 集群特征分析
3. **Job Characterization** - 作业特征分析
4. **User Characterization** - 用户行为分析

### 扩展分析模块
5. **Peak Day Analysis** - 峰值日分析和异常检测
6. **Data Verification** - 数据验证和一致性检查
7. **Detailed User Analysis** - 详细用户行为分析
8. **Null User Analysis** - 空用户记录分析

---

## 🚀 快速开始

### 1. 运行所有分析（推荐）

```bash
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research/Stage02_trace_analysis
python3 main.py
```

这将运行：
- ✅ 所有核心分析模块（4个）
- ✅ 所有扩展分析模块（4个）
- ✅ 生成综合报告

**预计运行时间:** 10-15分钟（首次运行）

---

### 2. 只运行核心分析

```bash
python3 main.py --no-extended
```

这将只运行核心的4个Helios风格分析模块，跳过扩展分析。

**预计运行时间:** 5-8分钟

---

### 3. 运行特定模块

#### 核心模块

```bash
# Philly对比分析
python3 main.py --module philly

# 集群特征分析
python3 main.py --module cluster

# 作业特征分析
python3 main.py --module job

# 用户特征分析
python3 main.py --module user
```

#### 扩展模块

```bash
# 峰值日分析
python3 main.py --module peak_day

# 数据验证
python3 main.py --module data_verification

# 详细用户分析
python3 main.py --module detailed_user

# 空用户分析
python3 main.py --module null_user
```

---

### 4. 强制重新处理数据

```bash
python3 main.py --force-reload
```

默认情况下，程序会使用缓存的预处理数据（如果存在）。使用 `--force-reload` 强制重新处理原始数据。

**注意:** 首次运行会自动处理数据，无需此选项。

---

## 📊 输出结果

所有分析结果保存在 `output/` 目录下：

```
output/
├── helios_analysis_report.txt          # 综合分析报告
│
├── philly_comparison/                  # Philly对比分析
│   ├── job_type_distribution.png
│   ├── gpu_job_count_status.png
│   ├── gpu_duration_cdf.png
│   └── gpu_time_status.png
│
├── cluster_characterization/           # 集群特征分析
│   ├── cluster_characterization_helios.png
│   └── info.md
│
├── job_characterization/               # 作业特征分析
│   ├── job_characterization_cpu_helios.png
│   ├── job_characterization_gpu_helios.png
│   ├── job_status_distribution_helios.png
│   └── info.md
│
├── user_characterization/              # 用户特征分析
│   ├── user_resource_cdf_helios.png
│   ├── user_behavior_patterns_helios.png
│   ├── user_cpu_behavior_helios.png
│   ├── user_gpu_behavior_helios.png
│   └── info.md
│
├── peak_day_detailed/                  # 峰值日分析
│   ├── peak_day_summary_report.md
│   ├── efficiency_analysis.png
│   ├── temporal_patterns_analysis.png
│   └── user_behavior_analysis.png
│
├── data_verification/                  # 数据验证
│   ├── null_user_id_records.csv
│   └── verified_user_job_counts.csv
│
├── detailed_user_analysis/             # 详细用户分析
│   ├── comprehensive_user_analysis_report.md
│   ├── user_duration_distributions.csv
│   └── user_job_counts_detailed.csv
│
└── null_user_analysis/                 # 空用户分析
    ├── null_user_analysis_summary.json
    └── null_user_sample.csv
```

---

## 🔧 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--module` | 要运行的分析模块 | `all` |
| `--cluster-config` | 集群配置文件路径 | `config/cluster_config.yaml` |
| `--force-reload` | 强制重新处理数据 | `False` |
| `--no-extended` | 运行all时不包括扩展分析 | `False` |

---

## 📝 使用示例

### 示例1: 完整分析流程（首次运行）

```bash
# 1. 进入目录
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research/Stage02_trace_analysis

# 2. 运行所有分析
python3 main.py

# 3. 查看综合报告
cat output/helios_analysis_report.txt

# 4. 查看峰值日分析
cat output/peak_day_detailed/peak_day_summary_report.md
```

### 示例2: 只更新峰值日分析

```bash
# 只运行峰值日分析（快速）
python3 main.py --module peak_day
```

### 示例3: 重新处理数据并运行核心分析

```bash
# 强制重新处理数据，只运行核心分析
python3 main.py --force-reload --no-extended
```

---

## ⚠️ 注意事项

### 1. 数据路径

确保以下数据文件存在：
- 原始数据: `../Stage01_data_filter_preprocess/full_processing_outputs/stage6_data_standardization/standardized_data.csv`
- 集群配置: `config/cluster_config.yaml`

### 2. 内存要求

- **最小内存:** 8GB
- **推荐内存:** 16GB+
- 数据集包含 6.52M 作业记录，需要足够内存

### 3. 运行时间

| 模块 | 首次运行 | 后续运行 |
|------|---------|---------|
| 数据预处理 | 3-5分钟 | <1分钟（缓存） |
| 核心分析 | 5-8分钟 | 5-8分钟 |
| 扩展分析 | 5-7分钟 | 5-7分钟 |
| **总计** | **10-15分钟** | **10-15分钟** |

### 4. 日志文件

运行日志保存在 `helios_analysis.log`，可用于调试：

```bash
tail -f helios_analysis.log
```

---

## 🐛 故障排除

### 问题1: 找不到数据文件

**错误信息:**
```
FileNotFoundError: [Errno 2] No such file or directory: '...'
```

**解决方案:**
检查数据路径是否正确，确保 Stage01 的数据已生成。

### 问题2: 内存不足

**错误信息:**
```
MemoryError: Unable to allocate array
```

**解决方案:**
- 关闭其他程序释放内存
- 使用 `--no-extended` 只运行核心分析
- 分批运行各个模块

### 问题3: 脚本执行失败

**错误信息:**
```
脚本 xxx.py 执行失败
```

**解决方案:**
查看日志文件 `helios_analysis.log` 获取详细错误信息。

---

## 📚 相关文档

- **项目总结:** `PROJECT_SUMMARY.md`
- **配置说明:** `config/cluster_config.yaml`
- **分析结果:** `output/helios_analysis_report.txt`
- **峰值日报告:** `output/peak_day_detailed/peak_day_summary_report.md`

---

## 🎯 下一步

分析完成后，你可以：

1. **查看综合报告** - `output/helios_analysis_report.txt`
2. **检查图表** - `output/*/*.png`
3. **阅读峰值日分析** - 了解数据质量问题
4. **准备论文** - 使用生成的图表和数据

---

**最后更新:** 2025-11-29

