#!/bin/bash
# ============================================================================
# GitHub 仓库初始化脚本
# ============================================================================
# 功能：为 01_HPC_Research 项目创建 Git 仓库并准备上传到 GitHub
# 策略：只上传代码、配置、文档，排除所有大数据文件
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}GitHub 仓库初始化脚本${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${YELLOW}项目目录：${NC}$PROJECT_ROOT"
echo ""

# 步骤1：检查 Git 是否已安装
echo -e "${BLUE}[1/6] 检查 Git 安装...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git 未安装！请先安装 Git。${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Git 已安装：$(git --version)${NC}"
echo ""

# 步骤2：创建 .gitignore 文件
echo -e "${BLUE}[2/6] 创建 .gitignore 文件...${NC}"
cat > "$PROJECT_ROOT/.gitignore" << 'EOF'
# ============================================================================
# 01_HPC_Research 项目 .gitignore
# ============================================================================
# 策略：只上传代码、配置、文档，排除所有数据文件
# ============================================================================

# ============================================================================
# 数据文件（主要排除项）
# ============================================================================

# 原始数据目录
Stage00_HPC_raw_data/

# 数据处理输出
Stage01_data_filter_preprocess/full_processing_outputs/
Stage01_data_filter_preprocess/stable_processing_outputs/
Stage01_data_filter_preprocess/*.log

# 大型 CSV 数据文件
*.csv
!**/example*.csv
!**/sample*.csv
!**/test*.csv

# 大型数据文件
*.pkl
*.pickle
*.h5
*.hdf5
*.parquet

# Excel 数据文件（通常很大）
*.xlsx
*.xls
!**/template*.xlsx
!**/config*.xlsx

# 模型文件和结果
**/results/
**/outputs/
**/checkpoints/
**/models/
!**/results/.gitkeep
!**/outputs/.gitkeep

# 预测模型数据
Stage03_simulator_CES_DRS/5_Prediction_Model/1_data_preparation/cpu1_jobs_*.csv
Stage03_simulator_CES_DRS/5_Prediction_Model/1_data_preparation/CPU1_*.csv
Stage03_simulator_CES_DRS/5_Prediction_Model/1_data_preparation/node_sequences/

# 仿真结果
Stage03_simulator_CES_DRS/4_Simulator/simulation_results/
Stage03_simulator_CES_DRS/4_Simulator/ces_experiment_results/
Stage03_simulator_CES_DRS/4_Simulator/Results_Compare/

# Trace 分析数据
Stage02_trace_analysis/data/
Stage02_trace_analysis/results/

# ============================================================================
# macOS 系统文件
# ============================================================================
.DS_Store
**/.DS_Store
._*
**/._*

# ============================================================================
# Python
# ============================================================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 虚拟环境
venv/
env/
ENV/
.venv/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# ============================================================================
# IDE 和编辑器
# ============================================================================
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject

# ============================================================================
# 日志和临时文件
# ============================================================================
*.log
*.tmp
*.temp
.cache/
*.bak
*.backup

# ============================================================================
# Git
# ============================================================================
.git/
*.orig

# ============================================================================
# 其他
# ============================================================================
# 压缩文件
*.zip
*.tar.gz
*.rar
*.7z

# 图片（如果很大）
# *.png
# *.jpg
# *.jpeg
# 注意：如果需要保留小的图片（如图表），可以注释掉上面的规则
EOF

echo -e "${GREEN}✅ .gitignore 文件已创建${NC}"
echo ""

# 步骤3：初始化 Git 仓库
echo -e "${BLUE}[3/6] 初始化 Git 仓库...${NC}"
cd "$PROJECT_ROOT"
if [ -d ".git" ]; then
    echo -e "${YELLOW}⚠️  Git 仓库已存在，跳过初始化${NC}"
else
    git init
    echo -e "${GREEN}✅ Git 仓库已初始化${NC}"
fi
echo ""

# 步骤4：统计将要提交的文件
echo -e "${BLUE}[4/6] 统计将要提交的文件...${NC}"
git add -n . > /tmp/git_add_preview.txt 2>&1 || true
TOTAL_FILES=$(cat /tmp/git_add_preview.txt | wc -l | tr -d ' ')
echo -e "${YELLOW}将要添加的文件数量：${NC}$TOTAL_FILES"
echo ""
echo -e "${YELLOW}文件示例（前20个）：${NC}"
head -20 /tmp/git_add_preview.txt
echo ""

# 步骤5：显示仓库大小估算
echo -e "${BLUE}[5/6] 估算仓库大小...${NC}"
REPO_SIZE=$(du -sh "$PROJECT_ROOT" --exclude='.git' --exclude='Stage00_HPC_raw_data' --exclude='Stage01_data_filter_preprocess/full_processing_outputs' --exclude='Stage01_data_filter_preprocess/stable_processing_outputs' --exclude='Stage02_trace_analysis/data' --exclude='Stage03_simulator_CES_DRS/4_Simulator/simulation_results' --exclude='Stage03_simulator_CES_DRS/4_Simulator/ces_experiment_results' --exclude='*.csv' --exclude='*.pkl' --exclude='*.xlsx' 2>/dev/null | cut -f1 || echo "未知")
echo -e "${YELLOW}预计仓库大小：${NC}$REPO_SIZE"
echo ""

# 步骤6：询问是否继续
echo -e "${BLUE}[6/6] 准备提交...${NC}"
echo -e "${YELLOW}是否继续添加文件到 Git？(y/n)${NC}"
read -r CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo -e "${YELLOW}❌ 操作已取消${NC}"
    exit 0
fi

# 添加文件
echo -e "${YELLOW}正在添加文件到 Git...${NC}"
git add .
echo -e "${GREEN}✅ 文件已添加${NC}"
echo ""

# 显示状态
echo -e "${YELLOW}Git 状态：${NC}"
git status --short | head -20
echo ""

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}下一步操作${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${YELLOW}1. 提交更改：${NC}"
echo -e "   ${GREEN}git commit -m \"Initial commit: 01_HPC_Research project\"${NC}"
echo ""
echo -e "${YELLOW}2. 在 GitHub 上创建新仓库（不要初始化 README）${NC}"
echo -e "   访问：${GREEN}https://github.com/new${NC}"
echo ""
echo -e "${YELLOW}3. 添加远程仓库并推送：${NC}"
echo -e "   ${GREEN}git remote add origin https://github.com/YOUR_USERNAME/01_HPC_Research.git${NC}"
echo -e "   ${GREEN}git branch -M main${NC}"
echo -e "   ${GREEN}git push -u origin main${NC}"
echo ""
echo -e "${GREEN}✅ Git 仓库准备完成！${NC}"

