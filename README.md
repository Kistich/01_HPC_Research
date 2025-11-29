# 🖥️ HPC Job Scheduling Research: CES+DRS Simulator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

**Carbon-aware Energy-efficient Scheduling (CES) + Dynamic Resource Scheduling (DRS) 仿真器和预测模型**

---

## 📋 项目简介

本项目是一个完整的 HPC 作业调度研究框架，包含：

1. **数据处理流程** - 从原始 HPC 日志到标准化 trace
2. **Trace 分析** - Helios 兼容的 trace 分析工具
3. **仿真器** - 支持多种调度算法的事件驱动仿真器
4. **预测模型** - 基于 LightGBM 的工作负载预测
5. **CES+DRS 系统** - 碳感知节能调度系统

---

## 🎯 主要功能

### 1. 调度算法

- **FIFO** - 先进先出
- **SJF** - 最短作业优先
- **Energy-Tiresias** - 能量感知优先级调度

### 2. CES+DRS 系统

- **SimpleCES** - 启发式工作负载预测
- **LightGBM-CES** - 机器学习工作负载预测
- **动态资源管理** - Wake Up / Sleep 操作

### 3. 性能指标

- **作业指标：** JCT, 等待时间, 队列时间
- **系统指标：** 利用率, 吞吐量, 公平性
- **能耗指标：** 总能耗, 空闲能耗, PUE
- **碳排放：** 总碳排放, 碳强度

---

## 📁 项目结构

```
01_HPC_Research/
├── Stage00_HPC_raw_data/              # 原始数据（不包含在 Git 中）
├── Stage01_data_filter_preprocess/    # 数据处理流程
│   ├── run_stable_processing.sh       # 主处理脚本
│   └── *.py                           # 处理模块
├── Stage02_trace_analysis/            # Trace 分析
│   ├── helios_trace_converter.py      # Helios 格式转换
│   └── trace_analyzer.py              # Trace 分析工具
├── Stage03_simulator_CES_DRS/         # 仿真器和预测模型
│   ├── 4_Simulator/                   # 仿真器
│   │   ├── run_all_simulations.py     # 运行所有仿真
│   │   ├── run_ces_experiments.py     # CES 实验
│   │   ├── core/                      # 核心模块
│   │   ├── schedulers/                # 调度器
│   │   ├── power_management/          # 电源管理
│   │   └── utils/                     # 工具模块
│   └── 5_Prediction_Model/            # 预测模型
│       ├── CES_prediction.py          # 模型训练
│       └── run_all_clusters.py        # 批量训练
├── .gitignore                         # Git 忽略规则
├── README.md                          # 本文档
├── GITHUB_UPLOAD_GUIDE.md            # GitHub 上传指南
└── DATA_MANAGEMENT.md                # 数据管理说明
```

---

## 🚀 快速开始

### 环境要求

- **Python:** 3.9+
- **操作系统:** macOS / Linux
- **依赖库:** pandas, numpy, lightgbm, matplotlib, seaborn

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/01_HPC_Research.git
cd 01_HPC_Research

# 安装 Python 依赖
pip3 install pandas numpy lightgbm matplotlib seaborn openpyxl
```

### 运行仿真

```bash
# 进入仿真器目录
cd Stage03_simulator_CES_DRS/4_Simulator

# 运行基础调度算法对比
python3 run_all_simulations.py

# 运行 CES+DRS 实验
python3 run_ces_experiments.py
```

---

## 📊 实验结果

### 实验1：调度算法对比

**对比算法：** FIFO vs SJF vs Energy-Tiresias

**结果示例：**
```
Algorithm         Avg JCT    Avg Wait    Utilization    Energy
FIFO              1234.5s    567.8s      45.2%          12.3 kWh
SJF               987.6s     345.2s      52.1%          11.8 kWh
Energy-Tiresias   1056.3s    412.5s      48.9%          10.5 kWh
```

### 实验2：CES+DRS 对比

**对比方案：** Baseline vs SimpleCES vs LightGBM-CES

**结果示例：**
```
Method           Avg JCT    Utilization    Energy    Carbon
Baseline         1234.5s    45.2%          12.3 kWh  5.2 kg
SimpleCES        1189.2s    58.7%          10.8 kWh  4.5 kg
LightGBM-CES     1156.8s    62.7%          10.2 kWh  4.2 kg
```

---

## 🔧 配置说明

### 仿真器配置

编辑 `Stage03_simulator_CES_DRS/4_Simulator/utils/config_manager.py`：

```python
config = SimulationConfig(
    cpu1_nodes=10,           # CPU1 节点数
    gpu1_nodes=5,            # GPU1 节点数
    scheduler_type='energy_tiresias',  # 调度算法
    enable_ces=True,         # 启用 CES
    use_lightgbm=True,       # 使用 LightGBM
    # ... 其他配置
)
```

### CES 参数

```python
ces_params = {
    'history': 1,            # 历史窗口（小时）
    'future': 3,             # 未来窗口（小时）
    'his_threshold': 1.0,    # 历史阈值
    'fut_threshold': 1.0,    # 未来阈值
    'de_threshold': 1,       # 决策阈值
    'buffer_ratio': 0.07,    # 缓冲比例
    'check_interval': 600    # 检查间隔（秒）
}
```

---

## 📚 文档

- **[GitHub 上传指南](GITHUB_UPLOAD_GUIDE.md)** - 如何上传项目到 GitHub
- **[数据管理说明](DATA_MANAGEMENT.md)** - 数据获取和管理
- **[安装成功报告](INSTALLATION_SUCCESS_REPORT.md)** - LightGBM 安装记录
- **[清理脚本说明](CLEANUP_SCRIPT_README.md)** - 隐藏文件清理

---

## 🧪 测试

```bash
# 测试 CES 配置
cd Stage03_simulator_CES_DRS/4_Simulator
python3 test_ces_config.py

# 测试仿真器
python3 -c "from core.simulator import Simulator; print('✅ Simulator OK')"

# 测试 LightGBM
python3 -c "import lightgbm; print('✅ LightGBM version:', lightgbm.__version__)"
```

---

## 📈 性能优化

### 仿真加速

- 使用采样数据（500 jobs 而不是 2500 jobs）
- 减少指标收集频率
- 禁用详细日志

### 内存优化

- 使用 Pandas 的 `chunksize` 参数
- 及时释放大型 DataFrame
- 使用 `del` 删除不需要的变量

---

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📞 联系方式

- **GitHub Issues:** https://github.com/YOUR_USERNAME/01_HPC_Research/issues
- **Email:** your.email@example.com

---

## 🙏 致谢

- **Helios Scheduler** - Trace 格式参考
- **Energy-Tiresias** - 调度算法参考
- **LightGBM** - 机器学习框架

---

## 📊 引用

如果你在研究中使用了本项目，请引用：

```bibtex
@misc{hpc_ces_drs_2025,
  title={HPC Job Scheduling Research: CES+DRS Simulator},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/YOUR_USERNAME/01_HPC_Research}
}
```

---

**⭐ 如果这个项目对你有帮助，请给个 Star！**

