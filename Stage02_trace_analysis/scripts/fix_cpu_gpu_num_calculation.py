#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复CPU和GPU数量计算错误

问题描述:
- cpu_num 当前使用的是 num_processors (CPU核心数)，应该是从 exec_hosts 解析出的不重复CPU节点数
- gpu_num 当前没有正确计算，应该是从 exec_hosts 解析出的不重复GPU节点数

修复方案:
1. 解析 exec_hosts 字段，提取所有主机名
2. 根据主机名前缀识别CPU节点和GPU节点
3. 计算不重复的CPU节点数和GPU节点数
4. 更新 cpu_num 和 gpu_num 字段
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_exec_hosts(exec_hosts_str):
    """
    解析exec_hosts字段，返回主机列表
    
    支持的格式:
    - 逗号分隔: "cpu1-1,cpu1-2,cpu1-3"
    - 空格分隔: "cpu1-1 cpu1-2 cpu1-3"
    - 范围格式: "cpu1-[1-10]"
    - 加号分隔: "cpu1-1+cpu1-2+cpu1-3"
    """
    if pd.isna(exec_hosts_str) or not exec_hosts_str:
        return []
    
    exec_hosts_str = str(exec_hosts_str).strip()
    if not exec_hosts_str:
        return []
    
    hosts = []
    
    # 1. 处理逗号分隔
    if ',' in exec_hosts_str:
        hosts = [h.strip() for h in exec_hosts_str.split(',') if h.strip()]
    
    # 2. 处理空格分隔（重复主机名）
    elif ' ' in exec_hosts_str:
        parts = exec_hosts_str.split()
        # 去重但保持顺序
        seen = set()
        hosts = []
        for part in parts:
            if part not in seen:
                hosts.append(part)
                seen.add(part)
    
    # 3. 处理范围格式 cpu1-[1-10]
    elif '[' in exec_hosts_str and ']' in exec_hosts_str:
        match = re.match(r'([a-zA-Z0-9_-]+)\[(\d+)-(\d+)\]', exec_hosts_str)
        if match:
            prefix, start, end = match.groups()
            hosts = [f"{prefix}{i}" for i in range(int(start), int(end) + 1)]
        else:
            hosts = [exec_hosts_str]
    
    # 4. 处理加号分隔
    elif '+' in exec_hosts_str:
        hosts = [h.strip() for h in exec_hosts_str.split('+') if h.strip()]
    
    # 5. 单个主机
    else:
        hosts = [exec_hosts_str]
    
    return hosts


def classify_host(hostname):
    """
    根据主机名分类为CPU节点或GPU节点
    
    GPU节点特征:
    - 包含 'gpu' 关键字
    - 包含 'dgx' 关键字
    - 包含 'a100', 'a800', 'v100' 等GPU型号
    
    Returns:
        'gpu' 或 'cpu'
    """
    if pd.isna(hostname):
        return 'cpu'
    
    hostname_lower = str(hostname).lower()
    
    # GPU节点特征
    gpu_keywords = ['gpu', 'dgx', 'a100', 'a800', 'v100', 'h100', 'tesla']
    
    for keyword in gpu_keywords:
        if keyword in hostname_lower:
            return 'gpu'
    
    return 'cpu'


def calculate_node_counts(exec_hosts_str):
    """
    从exec_hosts计算不重复的CPU节点数和GPU节点数
    
    Returns:
        (cpu_node_count, gpu_node_count)
    """
    hosts = parse_exec_hosts(exec_hosts_str)
    
    if not hosts:
        return 0, 0
    
    # 去重
    unique_hosts = list(set(hosts))
    
    # 分类统计
    cpu_nodes = set()
    gpu_nodes = set()
    
    for host in unique_hosts:
        if classify_host(host) == 'gpu':
            gpu_nodes.add(host)
        else:
            cpu_nodes.add(host)
    
    return len(cpu_nodes), len(gpu_nodes)


def fix_cpu_gpu_num(input_file, output_file=None):
    """
    修复CPU和GPU数量计算
    
    Args:
        input_file: 输入文件路径（CSV或pickle）
        output_file: 输出文件路径（可选，默认覆盖输入文件）
    """
    logger.info(f"读取数据文件: {input_file}")
    
    # 读取数据
    if str(input_file).endswith('.pkl') or str(input_file).endswith('.pickle'):
        df = pd.read_pickle(input_file)
    else:
        df = pd.read_csv(input_file)
    
    logger.info(f"数据行数: {len(df)}")
    
    # 检查必要字段
    if 'exec_hosts' not in df.columns:
        logger.error("数据中没有exec_hosts字段，无法修复")
        return
    
    # 计算CPU和GPU节点数
    logger.info("开始计算CPU和GPU节点数...")
    
    node_counts = df['exec_hosts'].apply(calculate_node_counts)
    df['cpu_node_count'] = [count[0] for count in node_counts]
    df['gpu_node_count'] = [count[1] for count in node_counts]
    
    # 更新cpu_num和gpu_num
    # 保存原始值用于对比
    if 'cpu_num' in df.columns:
        df['cpu_num_old'] = df['cpu_num']
    if 'gpu_num' in df.columns:
        df['gpu_num_old'] = df['gpu_num']
    
    df['cpu_num'] = df['cpu_node_count']
    df['gpu_num'] = df['gpu_node_count']
    
    # 统计信息
    logger.info("=" * 80)
    logger.info("修复结果统计:")
    logger.info(f"  总作业数: {len(df)}")
    logger.info(f"  CPU作业数 (gpu_num=0): {len(df[df['gpu_num'] == 0])}")
    logger.info(f"  GPU作业数 (gpu_num>0): {len(df[df['gpu_num'] > 0])}")
    logger.info("")
    logger.info("CPU节点数分布:")
    cpu_dist = df[df['gpu_num'] == 0]['cpu_num'].value_counts().sort_index()
    for cpu_count, job_count in cpu_dist.head(10).items():
        logger.info(f"  {cpu_count} CPU节点: {job_count} 作业 ({job_count/len(df)*100:.2f}%)")
    logger.info("")
    logger.info("GPU节点数分布:")
    gpu_dist = df[df['gpu_num'] > 0]['gpu_num'].value_counts().sort_index()
    for gpu_count, job_count in gpu_dist.items():
        logger.info(f"  {gpu_count} GPU节点: {job_count} 作业 ({job_count/len(df)*100:.2f}%)")
    logger.info("=" * 80)
    
    # 保存修复后的数据
    if output_file is None:
        output_file = input_file
    
    logger.info(f"保存修复后的数据到: {output_file}")
    
    if str(output_file).endswith('.pkl') or str(output_file).endswith('.pickle'):
        df.to_pickle(output_file)
    else:
        df.to_csv(output_file, index=False)
    
    logger.info("修复完成！")
    
    return df


if __name__ == '__main__':
    # 修复processed数据
    data_dir = Path(__file__).parent.parent / 'data' / 'processed'
    
    # 查找所有需要修复的文件
    files_to_fix = list(data_dir.glob('*.pkl')) + list(data_dir.glob('*.csv'))
    
    if not files_to_fix:
        logger.warning(f"在 {data_dir} 中没有找到需要修复的文件")
    else:
        logger.info(f"找到 {len(files_to_fix)} 个文件需要修复")
        
        for file_path in files_to_fix:
            logger.info(f"\n处理文件: {file_path.name}")
            try:
                fix_cpu_gpu_num(file_path)
            except Exception as e:
                logger.error(f"处理文件 {file_path.name} 时出错: {e}")

