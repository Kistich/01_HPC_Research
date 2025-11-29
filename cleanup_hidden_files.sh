#!/bin/bash
# ============================================================================
# 清理隐藏文件脚本
# ============================================================================
# 功能：移除项目目录中所有以 ._ 开头的 macOS 元数据文件
# 作者：自动生成
# 日期：2025-11-29
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录的父目录（01_HPC_Research）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}清理隐藏文件脚本${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${YELLOW}项目根目录：${NC}$PROJECT_ROOT"
echo ""

# 统计要删除的文件
echo -e "${YELLOW}正在扫描 ._ 开头的文件...${NC}"
DOTUNDERSCORE_FILES=$(find "$PROJECT_ROOT" -name "._*" -type f 2>/dev/null)
DOTUNDERSCORE_COUNT=$(echo "$DOTUNDERSCORE_FILES" | grep -c "^" || echo "0")

echo -e "${YELLOW}正在扫描 .DS_Store 文件...${NC}"
DSSTORE_FILES=$(find "$PROJECT_ROOT" -name ".DS_Store" -type f 2>/dev/null)
DSSTORE_COUNT=$(echo "$DSSTORE_FILES" | grep -c "^" || echo "0")

TOTAL_COUNT=$((DOTUNDERSCORE_COUNT + DSSTORE_COUNT))

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}扫描结果${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "${YELLOW}找到 ._* 文件：${NC}$DOTUNDERSCORE_COUNT 个"
echo -e "${YELLOW}找到 .DS_Store 文件：${NC}$DSSTORE_COUNT 个"
echo -e "${YELLOW}总计：${NC}$TOTAL_COUNT 个文件"
echo ""

if [ "$TOTAL_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✅ 没有找到需要清理的文件！${NC}"
    exit 0
fi

# 显示前10个文件示例
echo -e "${YELLOW}文件示例（前100个）：${NC}"
echo "$DOTUNDERSCORE_FILES" | head -100
if [ "$DSSTORE_COUNT" -gt 0 ]; then
    echo "$DSSTORE_FILES" | head -3
fi
echo ""

# 询问用户确认
echo -e "${RED}⚠️  警告：即将删除 $TOTAL_COUNT 个文件！${NC}"
echo -e "${YELLOW}是否继续？(y/n)${NC}"
read -r CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo -e "${YELLOW}❌ 操作已取消${NC}"
    exit 0
fi

# 执行删除
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}开始清理${NC}"
echo -e "${BLUE}============================================================================${NC}"

DELETED_COUNT=0

# 删除 ._* 文件
if [ "$DOTUNDERSCORE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}正在删除 ._* 文件...${NC}"
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            rm -f "$file"
            DELETED_COUNT=$((DELETED_COUNT + 1))
            if [ $((DELETED_COUNT % 10)) -eq 0 ]; then
                echo -e "${GREEN}  已删除 $DELETED_COUNT 个文件...${NC}"
            fi
        fi
    done <<< "$DOTUNDERSCORE_FILES"
fi

# 删除 .DS_Store 文件
if [ "$DSSTORE_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}正在删除 .DS_Store 文件...${NC}"
    while IFS= read -r file; do
        if [ -f "$file" ]; then
            rm -f "$file"
            DELETED_COUNT=$((DELETED_COUNT + 1))
        fi
    done <<< "$DSSTORE_FILES"
fi

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}清理完成${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "${GREEN}✅ 成功删除 $DELETED_COUNT 个文件${NC}"
echo ""

# 验证清理结果
REMAINING_DOTUNDERSCORE=$(find "$PROJECT_ROOT" -name "._*" -type f 2>/dev/null | wc -l | tr -d ' ')
REMAINING_DSSTORE=$(find "$PROJECT_ROOT" -name ".DS_Store" -type f 2>/dev/null | wc -l | tr -d ' ')
REMAINING_TOTAL=$((REMAINING_DOTUNDERSCORE + REMAINING_DSSTORE))

if [ "$REMAINING_TOTAL" -eq 0 ]; then
    echo -e "${GREEN}✅ 验证通过：所有隐藏文件已清理干净！${NC}"
else
    echo -e "${YELLOW}⚠️  仍有 $REMAINING_TOTAL 个文件未删除（可能是权限问题）${NC}"
fi

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}建议：将以下内容添加到 .gitignore 文件中以防止这些文件被提交：${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo -e "${YELLOW}._*${NC}"
echo -e "${YELLOW}.DS_Store${NC}"
echo -e "${YELLOW}**/.DS_Store${NC}"
echo -e "${YELLOW}**/._*${NC}"
echo ""
echo -e "${GREEN}脚本执行完成！${NC}"

