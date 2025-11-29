# 📤 GitHub 上传指南

## 📊 项目概况

**项目名称：** 01_HPC_Research  
**项目大小：** 52GB（原始数据 + 处理结果）  
**代码大小：** ~50MB（仅代码和配置）  
**GitHub 限制：** 单文件 < 100MB，仓库推荐 < 1GB

---

## 🎯 推荐方案：只上传代码和配置

### ✅ 优点

1. **符合 GitHub 最佳实践**
   - 仓库小巧（< 50MB）
   - 克隆速度快
   - 易于维护和协作

2. **数据管理更灵活**
   - 数据存储在本地或云存储
   - 避免 Git 历史膨胀
   - 可以使用专业的数据管理工具

3. **成本低**
   - 无需 Git LFS 付费
   - 无需额外存储费用

---

## 📋 实施步骤

### 步骤1：运行初始化脚本

```bash
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research
chmod +x setup_github_repo.sh
bash setup_github_repo.sh
```

**脚本功能：**
- ✅ 检查 Git 安装
- ✅ 创建 `.gitignore` 文件（排除所有数据文件）
- ✅ 初始化 Git 仓库
- ✅ 统计将要提交的文件
- ✅ 估算仓库大小
- ✅ 添加文件到 Git

---

### 步骤2：提交更改

```bash
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research
git commit -m "Initial commit: 01_HPC_Research project

- 添加数据处理流程代码
- 添加仿真器代码
- 添加预测模型代码
- 添加文档和配置文件
- 排除所有大数据文件（52GB）
"
```

---

### 步骤3：在 GitHub 创建新仓库

1. **访问：** https://github.com/new
2. **仓库名称：** `01_HPC_Research`
3. **描述：** `HPC Job Scheduling Research: CES+DRS Simulator and Prediction Models`
4. **可见性：** Public 或 Private（根据需要选择）
5. **⚠️ 重要：** 
   - ❌ **不要**勾选 "Add a README file"
   - ❌ **不要**勾选 "Add .gitignore"
   - ❌ **不要**勾选 "Choose a license"

---

### 步骤4：连接远程仓库并推送

```bash
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research

# 添加远程仓库（替换 YOUR_USERNAME 为你的 GitHub 用户名）
git remote add origin https://github.com/YOUR_USERNAME/01_HPC_Research.git

# 重命名分支为 main（GitHub 默认）
git branch -M main

# 推送到 GitHub
git push -u origin main
```

---

### 步骤5：验证上传

1. 访问你的 GitHub 仓库
2. 检查文件是否正确上传
3. 确认没有大数据文件被上传

---

## 📁 将要上传的内容

### ✅ 包含的文件

```
01_HPC_Research/
├── Stage00_HPC_raw_data/
│   └── README.md                    # 数据说明（不包含实际数据）
├── Stage01_data_filter_preprocess/
│   ├── *.py                         # Python 脚本
│   ├── *.sh                         # Shell 脚本
│   └── README.md                    # 流程说明
├── Stage02_trace_analysis/
│   ├── *.py                         # 分析脚本
│   └── README.md                    # 分析说明
├── Stage03_simulator_CES_DRS/
│   ├── 4_Simulator/
│   │   ├── *.py                     # 仿真器代码
│   │   ├── utils/                   # 工具模块
│   │   └── README.md                # 仿真器说明
│   └── 5_Prediction_Model/
│       ├── *.py                     # 预测模型代码
│       └── README.md                # 模型说明
├── .gitignore                       # Git 忽略规则
├── README.md                        # 项目主文档
├── GITHUB_UPLOAD_GUIDE.md          # 本文档
└── DATA_MANAGEMENT.md              # 数据管理说明
```

### ❌ 排除的文件

```
# 原始数据（13GB）
Stage00_HPC_raw_data/*.csv
Stage00_HPC_raw_data/*.xlsx

# 处理结果（28GB）
Stage01_data_filter_preprocess/full_processing_outputs/
Stage01_data_filter_preprocess/stable_processing_outputs/

# 分析数据（7.4GB）
Stage02_trace_analysis/data/
Stage02_trace_analysis/results/

# 仿真结果（3.8GB）
Stage03_simulator_CES_DRS/4_Simulator/simulation_results/
Stage03_simulator_CES_DRS/4_Simulator/ces_experiment_results/

# 所有大型数据文件
*.csv（除了示例文件）
*.pkl
*.xlsx（除了配置文件）
```

---

## 📊 预计上传大小

| 类别 | 大小 |
|------|------|
| Python 代码 | ~5MB |
| Shell 脚本 | ~1MB |
| 配置文件 | ~1MB |
| 文档（Markdown） | ~2MB |
| 小型示例数据 | ~10MB |
| **总计** | **~20-50MB** |

---

## 🔧 数据管理建议

### 方案A：本地存储（推荐）

**适用场景：** 数据仅供个人使用

```bash
# 数据保持在本地
/Volumes/EXTERNAL_US/backup2/01_HPC_Research/

# 在 README 中说明数据获取方式
```

### 方案B：云存储

**适用场景：** 需要团队协作或数据共享

**推荐服务：**
- **Google Drive** - 15GB 免费
- **OneDrive** - 5GB 免费
- **Dropbox** - 2GB 免费
- **百度网盘** - 免费但速度慢
- **阿里云 OSS** - 按量付费
- **AWS S3** - 按量付费

**实施步骤：**
1. 上传数据到云存储
2. 生成共享链接
3. 在 `DATA_MANAGEMENT.md` 中添加下载说明

### 方案C：Git LFS（不推荐）

**原因：**
- ⚠️ GitHub LFS 免费额度有限（1GB 存储 + 1GB/月带宽）
- ⚠️ 你的数据 52GB 远超免费额度
- ⚠️ 超出需要付费（$5/月 for 50GB）

---

## 📝 后续维护

### 添加新代码

```bash
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research
git add <new_file>.py
git commit -m "Add: <description>"
git push
```

### 更新文档

```bash
git add README.md
git commit -m "Update: documentation"
git push
```

### 同步到其他机器

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/01_HPC_Research.git

# 注意：数据文件需要单独下载（参考 DATA_MANAGEMENT.md）
```

---

## ⚠️ 注意事项

1. **不要提交大文件**
   - GitHub 单文件限制 100MB
   - 超过会导致推送失败

2. **检查 .gitignore**
   - 确保所有数据文件都被排除
   - 使用 `git status` 检查暂存区

3. **敏感信息**
   - 不要提交密码、API 密钥
   - 不要提交个人身份信息

4. **分支管理**
   - 使用 `main` 作为主分支
   - 开发新功能时创建新分支

---

## 🎯 快速开始

```bash
# 1. 运行初始化脚本
cd /Volumes/EXTERNAL_US/backup2/01_HPC_Research
bash setup_github_repo.sh

# 2. 提交更改
git commit -m "Initial commit: 01_HPC_Research project"

# 3. 在 GitHub 创建仓库
# 访问 https://github.com/new

# 4. 推送到 GitHub
git remote add origin https://github.com/YOUR_USERNAME/01_HPC_Research.git
git branch -M main
git push -u origin main
```

---

## ✅ 完成检查清单

- [ ] 运行 `setup_github_repo.sh` 脚本
- [ ] 检查 `.gitignore` 文件
- [ ] 提交更改到本地仓库
- [ ] 在 GitHub 创建新仓库
- [ ] 添加远程仓库地址
- [ ] 推送到 GitHub
- [ ] 验证上传内容
- [ ] 更新 README.md
- [ ] 添加数据管理说明

---

**🎉 准备好了吗？运行 `bash setup_github_repo.sh` 开始吧！**

