#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
核心数据预处理模块 - Helios兼容版本
负责原始数据的加载、清洗、标准化和基础指标计算
专门为Helios风格分析生成兼容的数据格式
确保后续所有分析模块基于统一的预处理数据
"""

import pandas as pd
import numpy as np
import logging
import pickle
import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
import os

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.cluster_identifier import ClusterIdentifier
from utils.data_loader import DataLoader, add_gpu_job_flag

logger = logging.getLogger(__name__)


def compute_user_metrics_batch(user_batch_data):
    """
    批量计算用户指标的工作函数（用于多进程）

    Args:
        user_batch_data: (user_list, df_subset) 元组

    Returns:
        用户指标字典
    """
    user_list, df_subset = user_batch_data
    batch_metrics = {}

    for user in user_list:
        if pd.notna(user):
            user_data = df_subset[df_subset['user_id'] == user]

            if len(user_data) == 0:
                continue

            metrics = {
                'total_jobs': len(user_data),
                'job_types': user_data['job_type'].value_counts().to_dict() if 'job_type' in df_subset.columns else {},
                'subclusters_used': user_data['primary_subcluster'].value_counts().to_dict() if 'primary_subcluster' in df_subset.columns else {},
            }

            # 资源消耗
            if 'duration' in df_subset.columns and 'actual_node_count' in df_subset.columns:
                valid_jobs = user_data[(user_data['duration'] > 0) & (user_data['actual_node_count'] > 0)]
                if len(valid_jobs) > 0:
                    metrics['node_hours'] = float((valid_jobs['duration'] * valid_jobs['actual_node_count'] / 3600).sum())
                    metrics['avg_duration_hours'] = float(valid_jobs['duration'].mean() / 3600)

            # 排队时间
            if 'queue_time' in df_subset.columns:
                valid_queue = user_data[user_data['queue_time'] >= 0]
                if len(valid_queue) > 0:
                    metrics['avg_queue_time_hours'] = float(valid_queue['queue_time'].mean() / 3600)
                    metrics['max_queue_time_hours'] = float(valid_queue['queue_time'].max() / 3600)

            # 成功率
            if 'job_status_standardized' in df_subset.columns:
                completed_jobs = user_data[user_data['job_status_standardized'] == 'Pass']
                metrics['success_rate'] = len(completed_jobs) / len(user_data) * 100

            batch_metrics[user] = metrics

    return batch_metrics


def compute_cluster_user_batch(batch_data):
    """
    批量计算cluster_user数据的工作函数（用于多进程）

    Args:
        batch_data: (user_list, user_data_dict) 元组
        user_data_dict: {user_id: user_jobs_dict} 格式的字典

    Returns:
        用户聚合数据列表
    """
    user_list, user_data_dict = batch_data
    batch_user_data = []

    for user in user_list:
        if pd.isna(user) or user not in user_data_dict:
            continue

        user_jobs_data = user_data_dict[user]

        # 重建用户作业数据
        total_jobs = user_jobs_data['total_jobs']
        gpu_jobs_data = user_jobs_data['gpu_jobs']
        cpu_jobs_data = user_jobs_data['cpu_jobs']

        # 计算GPU相关指标
        if gpu_jobs_data['count'] > 0:
            total_gpu_time = gpu_jobs_data['total_gpu_time']
            total_gpu_pend_time = gpu_jobs_data['total_queue_time']
            completed_gpu_percent = gpu_jobs_data['completed_percent']
        else:
            total_gpu_time = 0
            total_gpu_pend_time = 0
            completed_gpu_percent = 0

        # 计算CPU相关指标
        if cpu_jobs_data['count'] > 0:
            total_cpu_only_time = cpu_jobs_data['total_cpu_time']
            total_cpu_pend_time = cpu_jobs_data.get('total_queue_time', 0)
            completed_cpu_percent = cpu_jobs_data.get('completed_percent', 0)
        else:
            total_cpu_only_time = 0
            total_cpu_pend_time = 0
            completed_cpu_percent = 0

        user_record = {
            'user': user,
            'total_gpu_time': total_gpu_time,
            'total_cpu_only_time': total_cpu_only_time,
            'total_gpu_pend_time': total_gpu_pend_time,
            'total_cpu_pend_time': total_cpu_pend_time,
            'completed_gpu_percent': completed_gpu_percent,
            'completed_cpu_percent': completed_cpu_percent,
            'total_jobs': total_jobs,
            'gpu_jobs': gpu_jobs_data['count'],
            'cpu_jobs': cpu_jobs_data['count']
        }

        batch_user_data.append(user_record)

    return batch_user_data


class HeliosCompatibleDataPreprocessor:
    """Helios兼容的数据预处理器"""

    def __init__(self, config_path: str = "config/cluster_config.yaml"):
        """
        初始化数据预处理器

        Args:
            config_path: 集群配置文件路径
        """
        self.config_path = config_path
        self.data_loader = DataLoader(config_path)
        self.cluster_identifier = ClusterIdentifier(
            self.data_loader.get_subcluster_info(),
            self.data_loader.get_job_classification_rules()
        )

        # 加载集群配置
        self._load_cluster_config()

        # 预处理后的数据存储
        self.processed_data = {}
        self.computed_metrics = {}
        self.helios_data = {}  # Helios兼容格式数据

    def _load_cluster_config(self):
        """加载集群配置信息"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 提取CPU集群配置
            self.cpu_configs = {}
            subclusters = config.get('subclusters', {})
            for name, info in subclusters.items():
                if info.get('node_type') == 'cpu':
                    # 解析CPU规格字符串
                    cpu_spec = info.get('cpu_spec', '')
                    if 'Intel(R) Xeon(R) Platinum 8358P CPU 32C' in cpu_spec:
                        # 2 * 32C = 64核
                        self.cpu_configs[name] = 64
                    elif 'AMD EPYC 7763 64-Core Processor' in cpu_spec:
                        # 2 * 64C = 128核
                        self.cpu_configs[name] = 128
                    else:
                        # 默认值
                        self.cpu_configs[name] = 64

            # 添加BIGMEM配置（如果存在）
            if 'BIGMEM' in subclusters:
                bigmem_info = subclusters['BIGMEM']
                if 'Intel(R) Xeon(R) Gold 6348H CPU 24C' in bigmem_info.get('cpu_spec', ''):
                    self.cpu_configs['BIGMEM'] = 96  # 4 * 24C = 96核

            logger.info(f"加载CPU集群配置: {self.cpu_configs}")

        except Exception as e:
            logger.warning(f"加载集群配置失败: {e}")
            # 使用默认配置
            self.cpu_configs = {
                'CPU1': 64,   # 2 * Intel Xeon Platinum 8358P (32C)
                'CPU2': 64,   # 2 * Intel Xeon Platinum 8358P (32C)
                'CPU3': 128,  # 2 * AMD EPYC 7763 (64C)
                'BIGMEM': 96  # 4 * Intel Xeon Gold 6348H (24C)
            }

    def load_and_preprocess_all_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        加载并预处理所有数据
        
        Args:
            force_reload: 是否强制重新处理
            
        Returns:
            预处理后的完整数据集
        """
        logger.info("开始核心数据预处理...")
        
        # 检查是否已有缓存的预处理数据
        cache_path = Path("data/processed/preprocessed_data.pkl")
        if not force_reload and cache_path.exists():
            logger.info("发现缓存文件，但由于数据格式更新，强制重新处理...")
            # 由于数据格式已更新，暂时跳过缓存加载
            # return self._load_cached_data(cache_path)
        
        # 1. 加载已清洗的数据（来自data_filter_preprocess）
        raw_data = self._load_raw_data()

        # 2. 作业类型识别和子集群分析（保留分析功能）
        enhanced_data = self._enhance_with_cluster_info(raw_data)
        
        # 4. 计算基础指标
        self._compute_basic_metrics(enhanced_data)
        
        # 5. 准备GPU/CPU分类数据
        self._prepare_job_type_datasets(enhanced_data)
        
        # 6. 计算用户级别指标
        self._compute_user_metrics(enhanced_data)
        
        # 7. 计算时间相关指标
        self._compute_temporal_metrics(enhanced_data)
        
        # 8. 生成Helios兼容格式数据
        self._generate_helios_compatible_data(enhanced_data)

        # 9. 保存预处理结果
        self._save_processed_data(cache_path)

        logger.info("核心数据预处理完成")

        # 返回完整的数据结构（与缓存加载保持一致）
        return {
            **self.processed_data,
            'helios_data': self.helios_data,
            'computed_metrics': self.computed_metrics
        }
    
    def _load_raw_data(self) -> pd.DataFrame:
        """加载原始数据"""
        logger.info("加载原始数据...")
        
        # 使用现有的数据加载器
        df = self.data_loader.load_job_data()
        
        logger.info(f"原始数据加载完成: {len(df):,} 条记录")
        return df
    


    def _map_job_status_enhanced(self, row) -> str:
        """增强的作业状态映射"""
        # 优先使用job_status_str
        if 'job_status_str' in row and pd.notna(row['job_status_str']):
            status = str(row['job_status_str']).upper()
            if status == 'DONE':
                return 'Pass'
            elif status in ['EXIT', 'FAILED']:
                return 'Failed'
            elif status in ['TIMEOUT', 'KILLED']:
                return 'Killed'

        # 使用jstatus
        if 'jstatus' in row and pd.notna(row['jstatus']):
            jstatus = int(row['jstatus'])
            if jstatus == 32:  # DONE
                return 'Pass'
            elif jstatus in [1, 2, 3]:  # EXIT, FAILED等
                return 'Failed'
            else:
                return 'Killed'

        # 使用exit_status
        if 'exit_status' in row and pd.notna(row['exit_status']):
            exit_status = int(row['exit_status'])
            if exit_status == 0:
                return 'Pass'
            else:
                return 'Failed'

        # 默认值
        return 'Unknown'

    def _parse_exec_hosts_and_job_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析exec_hosts字段并确定作业类型"""
        logger.info("解析exec_hosts字段并确定作业类型...")

        df = df.copy()

        # 初始化新列
        df['is_gpu_job'] = False
        df['primary_subcluster'] = 'UNKNOWN'
        df['parsed_hosts'] = None

        def parse_and_classify_exec_hosts(exec_hosts_str):
            """解析exec_hosts并分类作业类型"""
            if pd.isna(exec_hosts_str):
                return False, 'UNKNOWN', []

            exec_hosts_str = str(exec_hosts_str).strip()
            if not exec_hosts_str:
                return False, 'UNKNOWN', []

            # 解析主机列表
            hosts = self._parse_exec_hosts_value(exec_hosts_str)

            # 分析主机名确定作业类型和主集群
            is_gpu = False
            subcluster_counts = {}

            for host in hosts:
                host_lower = host.lower()

                # 判断是否为GPU作业
                if any(gpu_pattern in host_lower for gpu_pattern in ['gpu1-', 'gpu2-', 'gpu3-']):
                    is_gpu = True
                    # GPU集群分类
                    if 'gpu1-' in host_lower:
                        subcluster_counts['GPU1'] = subcluster_counts.get('GPU1', 0) + 1
                    elif 'gpu2-' in host_lower:
                        subcluster_counts['GPU2'] = subcluster_counts.get('GPU2', 0) + 1
                    elif 'gpu3-' in host_lower:
                        subcluster_counts['GPU3'] = subcluster_counts.get('GPU3', 0) + 1

                # CPU集群分类
                elif any(cpu_pattern in host_lower for cpu_pattern in ['cpu1-', 'cpu2-', 'cpu3-']):
                    if 'cpu1-' in host_lower:
                        subcluster_counts['CPU1'] = subcluster_counts.get('CPU1', 0) + 1
                    elif 'cpu2-' in host_lower:
                        subcluster_counts['CPU2'] = subcluster_counts.get('CPU2', 0) + 1
                    elif 'cpu3-' in host_lower:
                        subcluster_counts['CPU3'] = subcluster_counts.get('CPU3', 0) + 1

                # BIGMEM集群
                elif any(bigmem_pattern in host_lower for bigmem_pattern in ['bigmem-', 'bigman-', 'bigmen-']):
                    subcluster_counts['BIGMEM'] = subcluster_counts.get('BIGMEM', 0) + 1

            # 确定主集群（使用节点数最多的集群）
            if subcluster_counts:
                primary_subcluster = max(subcluster_counts.keys(), key=lambda k: subcluster_counts[k])
            else:
                primary_subcluster = 'UNKNOWN'

            return is_gpu, primary_subcluster, hosts

        # 应用解析函数
        parsed_results = df['exec_hosts'].apply(parse_and_classify_exec_hosts)

        # 分离结果
        df['is_gpu_job'] = [result[0] for result in parsed_results]
        df['primary_subcluster'] = [result[1] for result in parsed_results]
        df['parsed_hosts'] = [result[2] for result in parsed_results]

        # 统计结果
        gpu_jobs = df['is_gpu_job'].sum()
        cpu_jobs = len(df) - gpu_jobs
        subcluster_dist = df['primary_subcluster'].value_counts()

        logger.info(f"作业类型解析完成: GPU作业={gpu_jobs}, CPU作业={cpu_jobs}")
        logger.info(f"子集群分布: {dict(subcluster_dist)}")

        return df

    def _parse_exec_hosts_value(self, exec_hosts_str: str) -> List[str]:
        """解析单个exec_hosts值，返回主机列表"""
        if not exec_hosts_str or pd.isna(exec_hosts_str):
            return []

        exec_hosts_str = str(exec_hosts_str).strip()
        if not exec_hosts_str:
            return []

        # 处理各种格式
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

    def _calculate_cpu_num(self, df: pd.DataFrame) -> pd.DataFrame:
        """根据子集群配置和num_processors计算准确的CPU数量"""
        logger.info("计算准确的CPU数量...")

        df_calc = df.copy()

        # 对于所有作业，初始使用num_processors作为CPU数量
        df_calc['cpu_num'] = df_calc['num_processors']

        # 对于CPU作业，根据子集群配置进行调整（如果需要）
        cpu_jobs_mask = ~df_calc['is_gpu_job']

        # 统计各子集群的CPU使用情况
        cpu_stats = {}
        for subcluster in df_calc['primary_subcluster'].unique():
            if pd.notna(subcluster) and subcluster in self.cpu_configs:
                subcluster_mask = (df_calc['primary_subcluster'] == subcluster) & cpu_jobs_mask
                subcluster_data = df_calc[subcluster_mask]

                if len(subcluster_data) > 0:
                    avg_cpu = subcluster_data['cpu_num'].mean()
                    max_cpu = subcluster_data['cpu_num'].max()
                    cores_per_node = self.cpu_configs[subcluster]

                    cpu_stats[subcluster] = {
                        'avg_cpu': avg_cpu,
                        'max_cpu': max_cpu,
                        'cores_per_node': cores_per_node,
                        'job_count': len(subcluster_data)
                    }

        # 输出统计信息
        for subcluster, stats in cpu_stats.items():
            logger.info(f"{subcluster}: 平均CPU={stats['avg_cpu']:.1f}, 最大CPU={stats['max_cpu']}, "
                       f"每节点核心数={stats['cores_per_node']}, 作业数={stats['job_count']}")

        total_cpu_avg = df_calc[cpu_jobs_mask]['cpu_num'].mean() if cpu_jobs_mask.sum() > 0 else 0
        total_gpu_avg = df_calc[~cpu_jobs_mask]['cpu_num'].mean() if (~cpu_jobs_mask).sum() > 0 else 0

        logger.info(f"CPU数量计算完成 - CPU作业平均CPU数: {total_cpu_avg:.1f}, GPU作业平均CPU数: {total_gpu_avg:.1f}")
        return df_calc
    
    def _enhance_with_cluster_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """增强数据：添加集群和作业类型信息"""
        logger.info("增强数据：添加集群和作业类型信息...")
        
        # 使用集群识别器增强数据
        enhanced_df = self.cluster_identifier.enhance_dataframe(df)

        # 修复作业状态映射
        logger.info("修复作业状态映射...")
        enhanced_df['job_status_standardized'] = enhanced_df.apply(self._map_job_status_enhanced, axis=1)
        logger.info("作业状态映射完成")

        # 保存增强后的数据
        self.processed_data['enhanced_data'] = enhanced_df

        logger.info("集群信息增强完成")
        return enhanced_df
    
    def _compute_basic_metrics(self, df: pd.DataFrame):
        """计算基础指标"""
        logger.info("计算基础指标...")
        
        metrics = {}
        
        # 1. 总体统计
        metrics['total_jobs'] = len(df)
        metrics['total_users'] = df['user_id'].nunique() if 'user_id' in df.columns else 0
        metrics['date_range'] = {
            'start': df['submit_time'].min(),
            'end': df['submit_time'].max(),
            'days': (df['submit_time'].max() - df['submit_time'].min()).days
        }
        
        # 2. 作业类型分布
        if 'job_type' in df.columns:
            metrics['job_type_distribution'] = df['job_type'].value_counts().to_dict()
        
        # 3. 子集群分布
        if 'primary_subcluster' in df.columns:
            metrics['subcluster_distribution'] = df['primary_subcluster'].value_counts().to_dict()
        
        # 4. 作业状态分布
        metrics['job_status_distribution'] = df['state'].value_counts().to_dict()
        
        # 5. 资源使用统计
        if 'duration' in df.columns and 'actual_node_count' in df.columns:
            valid_jobs = df[(df['duration'] > 0) & (df['actual_node_count'] > 0)].copy()  # 使用.copy()避免警告
            valid_jobs['node_hours'] = valid_jobs['duration'] * valid_jobs['actual_node_count'] / 3600

            metrics['resource_consumption'] = {
                'total_node_hours': float(valid_jobs['node_hours'].sum()),
                'avg_job_duration_hours': float(valid_jobs['duration'].mean() / 3600),
                'avg_nodes_per_job': float(valid_jobs['actual_node_count'].mean())
            }
        
        # 6. GPU相关统计 - 基于exec_hosts识别
        if 'is_gpu_job' in df.columns:
            gpu_jobs = df[df['is_gpu_job'] == True]
            if len(gpu_jobs) > 0:
                # 计算GPU数量统计（如果gpu_num可用的话）
                if 'gpu_num' in df.columns and gpu_jobs['gpu_num'].notna().sum() > 0:
                    avg_gpu = float(gpu_jobs[gpu_jobs['gpu_num'] > 0]['gpu_num'].mean())
                    max_gpu = int(gpu_jobs['gpu_num'].max())
                    total_gpu_hours = float((gpu_jobs['gpu_num'] * gpu_jobs['duration'] / 3600).sum())
                else:
                    # 如果gpu_num不可用，使用默认值
                    avg_gpu = 1.0  # 假设每个GPU作业平均使用1个GPU
                    max_gpu = 1
                    total_gpu_hours = float((gpu_jobs['duration'] / 3600).sum())  # 假设每作业1GPU

                metrics['gpu_statistics'] = {
                    'total_gpu_jobs': len(gpu_jobs),
                    'gpu_job_percentage': len(gpu_jobs) / len(df) * 100,
                    'avg_gpu_per_job': avg_gpu,
                    'max_gpu_per_job': max_gpu,
                    'total_gpu_hours': total_gpu_hours,
                    'identification_method': 'exec_hosts_based'
                }
        
        self.computed_metrics['basic_metrics'] = metrics
        logger.info("基础指标计算完成")
    
    def _prepare_job_type_datasets(self, df: pd.DataFrame):
        """准备按作业类型分类的数据集"""
        logger.info("准备作业类型分类数据集...")
        
        job_type_data = {}
        
        if 'job_type' in df.columns:
            for job_type in df['job_type'].unique():
                if pd.notna(job_type):
                    job_type_data[job_type] = df[df['job_type'] == job_type].copy()
        
        # 特别准备GPU和CPU作业数据 - 基于exec_hosts识别
        if 'is_gpu_job' in df.columns:
            job_type_data['gpu_jobs'] = df[df['is_gpu_job'] == True].copy()
            job_type_data['cpu_jobs'] = df[df['is_gpu_job'] == False].copy()
        
        self.processed_data['job_type_datasets'] = job_type_data
        logger.info(f"作业类型数据集准备完成: {list(job_type_data.keys())}")
    
    def _compute_user_metrics(self, df: pd.DataFrame):
        """计算用户级别指标（高性能版本：使用多进程和进度显示）"""
        logger.info("计算用户级别指标...")

        if 'user_id' not in df.columns:
            logger.warning("缺少user_id列，跳过用户指标计算")
            return

        # 获取所有唯一用户
        unique_users = df['user_id'].dropna().unique()
        total_users = len(unique_users)
        logger.info(f"开始计算 {total_users:,} 个用户的指标...")

        # 如果用户数量较少，使用单进程
        if total_users < 1000:
            logger.info("用户数量较少，使用单进程计算...")
            self._compute_user_metrics_single_process(df, unique_users)
            return

        # 使用多进程计算
        num_processes = min(cpu_count(), 8)  # 限制最大进程数
        logger.info(f"使用 {num_processes} 个进程并行计算用户指标...")

        # 将用户分批
        batch_size = max(100, total_users // (num_processes * 4))  # 每个进程处理多个批次
        user_batches = []

        for i in range(0, total_users, batch_size):
            batch_users = unique_users[i:i + batch_size]
            user_batches.append((batch_users, df))

        logger.info(f"分为 {len(user_batches)} 个批次，每批约 {batch_size} 个用户")

        # 多进程计算
        user_metrics = {}
        processed_batches = 0

        try:
            with Pool(processes=num_processes) as pool:
                # 使用imap_unordered获得进度反馈
                for batch_result in pool.imap_unordered(compute_user_metrics_batch, user_batches):
                    user_metrics.update(batch_result)
                    processed_batches += 1
                    progress = (processed_batches / len(user_batches)) * 100
                    processed_users = min(processed_batches * batch_size, total_users)
                    logger.info(f"用户指标计算进度: {processed_users:,}/{total_users:,} ({progress:.1f}%)")

        except Exception as e:
            logger.warning(f"多进程计算失败，回退到单进程: {e}")
            self._compute_user_metrics_single_process(df, unique_users)
            return

        self.computed_metrics['user_metrics'] = user_metrics
        logger.info(f"用户指标计算完成: {len(user_metrics):,} 个用户")

    def _compute_user_metrics_single_process(self, df: pd.DataFrame, unique_users):
        """单进程计算用户指标（备用方法）"""
        user_metrics = {}
        total_users = len(unique_users)

        # 预计算分组数据
        user_groups = df.groupby('user_id')

        # 批量处理用户，每1000个用户显示一次进度
        batch_size = 1000
        processed = 0

        for i in range(0, total_users, batch_size):
            batch_users = unique_users[i:i + batch_size]

            for user in batch_users:
                if pd.notna(user):
                    try:
                        user_data = user_groups.get_group(user)

                        metrics = {
                            'total_jobs': len(user_data),
                            'job_types': user_data['job_type'].value_counts().to_dict() if 'job_type' in df.columns else {},
                            'subclusters_used': user_data['primary_subcluster'].value_counts().to_dict() if 'primary_subcluster' in df.columns else {},
                        }

                        # 资源消耗
                        if 'duration' in df.columns and 'actual_node_count' in df.columns:
                            valid_jobs = user_data[(user_data['duration'] > 0) & (user_data['actual_node_count'] > 0)]
                            if len(valid_jobs) > 0:
                                metrics['node_hours'] = float((valid_jobs['duration'] * valid_jobs['actual_node_count'] / 3600).sum())
                                metrics['avg_duration_hours'] = float(valid_jobs['duration'].mean() / 3600)

                        # 排队时间
                        if 'queue_time' in df.columns:
                            valid_queue = user_data[user_data['queue_time'] >= 0]
                            if len(valid_queue) > 0:
                                metrics['avg_queue_time_hours'] = float(valid_queue['queue_time'].mean() / 3600)
                                metrics['max_queue_time_hours'] = float(valid_queue['queue_time'].max() / 3600)

                        # 成功率
                        if 'job_status_standardized' in df.columns:
                            completed_jobs = user_data[user_data['job_status_standardized'] == 'Pass']
                            metrics['success_rate'] = len(completed_jobs) / len(user_data) * 100

                        user_metrics[user] = metrics
                    except KeyError:
                        # 用户不存在，跳过
                        continue

            processed += len(batch_users)
            progress = (processed / total_users) * 100
            logger.info(f"用户指标计算进度: {processed:,}/{total_users:,} ({progress:.1f}%)")

        self.computed_metrics['user_metrics'] = user_metrics
    
    def _compute_temporal_metrics(self, df: pd.DataFrame):
        """计算时间相关指标"""
        logger.info("计算时间相关指标...")
        
        if 'submit_time' not in df.columns:
            logger.warning("缺少submit_time列，跳过时间指标计算")
            return
        
        temporal_metrics = {}
        
        # 提取时间特征
        df_time = df.copy()
        df_time['hour'] = df_time['submit_time'].dt.hour
        df_time['weekday'] = df_time['submit_time'].dt.dayofweek
        df_time['month'] = df_time['submit_time'].dt.month
        df_time['date'] = df_time['submit_time'].dt.date
        
        # 小时分布
        temporal_metrics['hourly_distribution'] = df_time['hour'].value_counts().sort_index().to_dict()
        
        # 星期分布
        temporal_metrics['weekday_distribution'] = df_time['weekday'].value_counts().sort_index().to_dict()
        
        # 月度分布
        temporal_metrics['monthly_distribution'] = df_time['month'].value_counts().sort_index().to_dict()
        
        # 每日作业数
        daily_jobs = df_time.groupby('date').size()
        temporal_metrics['daily_statistics'] = {
            'avg_jobs_per_day': float(daily_jobs.mean()),
            'max_jobs_per_day': int(daily_jobs.max()),
            'min_jobs_per_day': int(daily_jobs.min())
        }
        
        # 按作业类型的时间分布
        if 'job_type' in df.columns:
            type_temporal = {}
            for job_type in df_time['job_type'].unique():
                if pd.notna(job_type):
                    type_data = df_time[df_time['job_type'] == job_type]
                    type_temporal[job_type] = {
                        'hourly_distribution': type_data['hour'].value_counts().sort_index().to_dict(),
                        'weekday_distribution': type_data['weekday'].value_counts().sort_index().to_dict()
                    }
            temporal_metrics['by_job_type'] = type_temporal
        
        self.computed_metrics['temporal_metrics'] = temporal_metrics
        logger.info("时间指标计算完成")

    def _generate_helios_compatible_data(self, df: pd.DataFrame):
        """生成Helios兼容格式的数据文件"""
        logger.info("生成Helios兼容格式数据...")

        # 1. 生成cluster_log.csv (作业级别数据)
        cluster_log = self._create_cluster_log(df)

        # 2. 生成cluster_sequence.csv (时间序列数据)
        cluster_sequence = self._create_cluster_sequence(df)

        # 3. 生成cluster_throughput.csv (吞吐量数据)
        cluster_throughput = self._create_cluster_throughput(df)

        # 4. 生成cluster_user.pkl (用户聚合数据)
        cluster_user = self._create_cluster_user(df)

        # 保存到helios_data
        self.helios_data = {
            'cluster_log': cluster_log,
            'cluster_sequence': cluster_sequence,
            'cluster_throughput': cluster_throughput,
            'cluster_user': cluster_user
        }

        logger.info("Helios兼容格式数据生成完成")

    def _create_cluster_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建cluster_log.csv格式数据"""
        logger.info("创建cluster_log.csv格式数据...")

        # 选择和重命名字段以匹配Helios格式
        helios_df = df.copy()

        # 基础字段映射
        field_mapping = {
            'job_id': 'job_id',
            'user_id': 'user',
            'gpu_num': 'gpu_num',
            'cpu_num': 'cpu_num',
            'num_exec_hosts': 'node_num',
            'job_status_standardized': 'state',
            'submit_time': 'submit_time',
            'start_time': 'start_time',
            'end_time': 'end_time',
            'duration': 'duration',
            'queue_time': 'queue'
        }

        # 创建Helios格式的DataFrame
        cluster_log = pd.DataFrame()

        for our_field, helios_field in field_mapping.items():
            if our_field in helios_df.columns:
                cluster_log[helios_field] = helios_df[our_field]
            else:
                # 处理缺失字段
                if helios_field == 'node_num':
                    cluster_log[helios_field] = helios_df.get('actual_node_count', 1)
                elif helios_field == 'queue':
                    cluster_log[helios_field] = helios_df.get('queue_time', 0)
                else:
                    cluster_log[helios_field] = 0

        # 确保数据类型正确
        cluster_log['gpu_num'] = pd.to_numeric(cluster_log['gpu_num'], errors='coerce').fillna(0)
        cluster_log['cpu_num'] = pd.to_numeric(cluster_log['cpu_num'], errors='coerce').fillna(0)
        cluster_log['node_num'] = pd.to_numeric(cluster_log['node_num'], errors='coerce').fillna(1)
        cluster_log['duration'] = pd.to_numeric(cluster_log['duration'], errors='coerce').fillna(0)
        cluster_log['queue'] = pd.to_numeric(cluster_log['queue'], errors='coerce').fillna(0)

        # 移除无效记录
        cluster_log = cluster_log[cluster_log['duration'] > 0]

        logger.info(f"cluster_log创建完成: {len(cluster_log)} 条记录")
        return cluster_log

    def _create_cluster_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建cluster_sequence.csv格式数据（时间序列）"""
        logger.info("创建cluster_sequence.csv格式数据...")

        if 'submit_time' not in df.columns:
            logger.warning("缺少submit_time字段，无法创建时间序列数据")
            return pd.DataFrame()

        # 按小时聚合数据
        df_time = df.copy()
        df_time['time'] = df_time['submit_time'].dt.floor('H')

        # 聚合统计
        sequence_data = df_time.groupby('time').agg({
            'job_id': 'count',           # 每小时作业提交数
            'gpu_num': 'sum',            # 每小时GPU使用总数
            'cpu_num': 'sum',            # 每小时CPU使用总数
            'duration': 'mean'           # 平均持续时间
        }).reset_index()

        # 重命名列
        sequence_data.columns = ['time', 'job_count', 'gpu_usage', 'cpu_usage', 'avg_duration']

        # 填充缺失的小时
        if len(sequence_data) > 0:
            time_range = pd.date_range(
                start=sequence_data['time'].min(),
                end=sequence_data['time'].max(),
                freq='H'
            )
            full_sequence = pd.DataFrame({'time': time_range})
            sequence_data = full_sequence.merge(sequence_data, on='time', how='left').fillna(0)

        logger.info(f"cluster_sequence创建完成: {len(sequence_data)} 条时间点记录")
        return sequence_data

    def _create_cluster_throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建cluster_throughput.csv格式数据（吞吐量）"""
        logger.info("创建cluster_throughput.csv格式数据...")

        if 'start_time' not in df.columns:
            logger.warning("缺少start_time字段，无法创建吞吐量数据")
            return pd.DataFrame()

        # 按小时聚合开始时间
        df_time = df.copy()
        df_time['time'] = df_time['start_time'].dt.floor('H')

        # 计算吞吐量指标
        throughput_data = df_time.groupby('time').agg({
            'job_id': 'count',                    # 每小时开始作业数
            'duration': ['mean', 'sum'],          # 平均和总执行时间
            'gpu_num': 'sum',                     # GPU小时数
            'cpu_num': 'sum'                      # CPU小时数
        }).reset_index()

        # 展平多级列名
        throughput_data.columns = ['time', 'jobs_started', 'avg_duration', 'total_duration', 'gpu_hours', 'cpu_hours']

        # 计算GPU和CPU小时数（考虑持续时间）
        df_time['gpu_hours_calc'] = df_time['gpu_num'] * df_time['duration'] / 3600
        df_time['cpu_hours_calc'] = df_time['cpu_num'] * df_time['duration'] / 3600

        resource_hours = df_time.groupby('time').agg({
            'gpu_hours_calc': 'sum',
            'cpu_hours_calc': 'sum'
        }).reset_index()

        # 合并数据
        throughput_data = throughput_data.merge(resource_hours, on='time', how='left')
        throughput_data['gpu_hours'] = throughput_data['gpu_hours_calc']
        throughput_data['cpu_hours'] = throughput_data['cpu_hours_calc']
        throughput_data = throughput_data.drop(['gpu_hours_calc', 'cpu_hours_calc'], axis=1)

        logger.info(f"cluster_throughput创建完成: {len(throughput_data)} 条时间点记录")
        return throughput_data

    def _create_cluster_user(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建cluster_user.pkl格式数据（用户聚合）- 多进程优化版本"""
        logger.info("创建cluster_user.pkl格式数据...")

        if 'user_id' not in df.columns:
            logger.warning("缺少user_id字段，无法创建用户聚合数据")
            return pd.DataFrame()

        # 获取所有唯一用户
        unique_users = df['user_id'].dropna().unique()
        total_users = len(unique_users)
        logger.info(f"开始处理 {total_users:,} 个用户的聚合数据...")

        # 如果用户数量较少，使用单进程
        if total_users < 1000:
            logger.info("用户数量较少，使用单进程处理...")
            return self._create_cluster_user_single_process(df, unique_users)

        # 使用多进程处理
        num_processes = min(cpu_count(), 8)  # 限制最大进程数
        logger.info(f"使用 {num_processes} 个进程并行处理cluster_user数据...")

        # 预处理用户数据，避免传递完整DataFrame
        logger.info("预处理用户数据以支持多进程...")
        user_data_dict = self._prepare_user_data_for_multiprocessing(df, unique_users)

        # 将用户分批
        batch_size = max(100, total_users // (num_processes * 4))
        user_batches = []

        for i in range(0, total_users, batch_size):
            batch_users = unique_users[i:i + batch_size]
            # 只传递当前批次用户的数据
            batch_user_data = {user: user_data_dict[user] for user in batch_users if user in user_data_dict}
            user_batches.append((batch_users, batch_user_data))

        logger.info(f"分为 {len(user_batches)} 个批次，每批约 {batch_size} 个用户")

        # 多进程处理
        all_user_data = []
        processed_batches = 0

        try:
            with Pool(processes=num_processes) as pool:
                # 使用imap_unordered获得进度反馈
                for batch_result in pool.imap_unordered(compute_cluster_user_batch, user_batches):
                    all_user_data.extend(batch_result)
                    processed_batches += 1
                    progress = (processed_batches / len(user_batches)) * 100
                    processed_users = min(processed_batches * batch_size, total_users)
                    logger.info(f"cluster_user处理进度: {processed_users:,}/{total_users:,} ({progress:.1f}%)")

        except Exception as e:
            logger.warning(f"多进程处理失败，回退到单进程: {e}")
            return self._create_cluster_user_single_process(df, unique_users)

        cluster_user = pd.DataFrame(all_user_data)
        logger.info(f"cluster_user创建完成: {len(cluster_user):,} 个用户记录")
        return cluster_user

    def _create_cluster_user_single_process(self, df: pd.DataFrame, unique_users) -> pd.DataFrame:
        """单进程创建cluster_user数据（备用方法）"""
        user_data = []
        total_users = len(unique_users)

        # 批量处理用户，每1000个用户显示一次进度
        batch_size = 1000
        processed = 0

        for i in range(0, total_users, batch_size):
            batch_users = unique_users[i:i + batch_size]

            for user in batch_users:
                if pd.isna(user):
                    continue

                user_jobs = df[df['user_id'] == user]
                if len(user_jobs) == 0:
                    continue

                gpu_jobs = user_jobs[user_jobs['is_gpu_job'] == True]
                cpu_jobs = user_jobs[user_jobs['is_gpu_job'] == False]

                # 计算GPU相关指标
                if len(gpu_jobs) > 0:
                    total_gpu_time = (gpu_jobs['gpu_num'] * gpu_jobs['duration']).sum() / 3600  # GPU小时
                    total_gpu_pend_time = gpu_jobs['queue_time'].sum() / 3600 if 'queue_time' in gpu_jobs.columns else 0  # 排队小时
                    completed_gpu = gpu_jobs[gpu_jobs['job_status_standardized'] == 'Pass']
                    completed_gpu_percent = len(completed_gpu) / len(gpu_jobs) * 100 if len(gpu_jobs) > 0 else 0
                else:
                    total_gpu_time = 0
                    total_gpu_pend_time = 0
                    completed_gpu_percent = 0

                # 计算CPU相关指标
                if len(cpu_jobs) > 0:
                    total_cpu_only_time = (cpu_jobs['cpu_num'] * cpu_jobs['duration']).sum() / 3600  # CPU小时
                    total_cpu_pend_time = cpu_jobs['queue_time'].sum() / 3600 if 'queue_time' in cpu_jobs.columns else 0  # CPU排队小时
                    completed_cpu = cpu_jobs[cpu_jobs['job_status_standardized'] == 'Pass']
                    completed_cpu_percent = len(completed_cpu) / len(cpu_jobs) * 100 if len(cpu_jobs) > 0 else 0
                else:
                    total_cpu_only_time = 0
                    total_cpu_pend_time = 0
                    completed_cpu_percent = 0

                user_record = {
                    'user': user,
                    'total_gpu_time': total_gpu_time,
                    'total_cpu_only_time': total_cpu_only_time,
                    'total_gpu_pend_time': total_gpu_pend_time,
                    'total_cpu_pend_time': total_cpu_pend_time,
                    'completed_gpu_percent': completed_gpu_percent,
                    'completed_cpu_percent': completed_cpu_percent,
                    'total_jobs': len(user_jobs),
                    'gpu_jobs': len(gpu_jobs),
                    'cpu_jobs': len(cpu_jobs)
                }

                user_data.append(user_record)

            processed += len(batch_users)
            progress = (processed / total_users) * 100
            logger.info(f"cluster_user处理进度: {processed:,}/{total_users:,} ({progress:.1f}%)")

        return pd.DataFrame(user_data)

    def _prepare_user_data_for_multiprocessing(self, df: pd.DataFrame, unique_users) -> dict:
        """
        预处理用户数据，准备用于多进程处理的数据结构
        避免在多进程中传递完整的DataFrame
        """
        logger.info("准备用户数据字典...")
        user_data_dict = {}

        # 按用户分组，但只提取必要的统计信息
        for user in unique_users:
            if pd.isna(user):
                continue

            user_jobs = df[df['user_id'] == user]
            if len(user_jobs) == 0:
                continue

            gpu_jobs = user_jobs[user_jobs['is_gpu_job'] == True]
            cpu_jobs = user_jobs[user_jobs['is_gpu_job'] == False]

            # 提取GPU作业统计信息
            gpu_jobs_data = {
                'count': len(gpu_jobs),
                'total_gpu_time': 0,
                'total_queue_time': 0,
                'completed_percent': 0
            }

            if len(gpu_jobs) > 0:
                gpu_jobs_data['total_gpu_time'] = float((gpu_jobs['gpu_num'] * gpu_jobs['duration']).sum() / 3600)
                if 'queue_time' in gpu_jobs.columns:
                    gpu_jobs_data['total_queue_time'] = float(gpu_jobs['queue_time'].sum() / 3600)
                completed_gpu = gpu_jobs[gpu_jobs['job_status_standardized'] == 'Pass']
                gpu_jobs_data['completed_percent'] = float(len(completed_gpu) / len(gpu_jobs) * 100)

            # 提取CPU作业统计信息
            cpu_jobs_data = {
                'count': len(cpu_jobs),
                'total_cpu_time': 0,
                'total_queue_time': 0,
                'completed_percent': 0
            }

            if len(cpu_jobs) > 0:
                cpu_jobs_data['total_cpu_time'] = float((cpu_jobs['cpu_num'] * cpu_jobs['duration']).sum() / 3600)
                if 'queue_time' in cpu_jobs.columns:
                    cpu_jobs_data['total_queue_time'] = float(cpu_jobs['queue_time'].sum() / 3600)
                completed_cpu = cpu_jobs[cpu_jobs['job_status_standardized'] == 'Pass']
                cpu_jobs_data['completed_percent'] = float(len(completed_cpu) / len(cpu_jobs) * 100)

            user_data_dict[user] = {
                'total_jobs': len(user_jobs),
                'gpu_jobs': gpu_jobs_data,
                'cpu_jobs': cpu_jobs_data
            }

        logger.info(f"用户数据字典准备完成: {len(user_data_dict)} 个用户")
        return user_data_dict
    
    def _save_processed_data(self, cache_path: Path):
        """保存预处理数据"""
        logger.info("保存预处理数据...")
        
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存完整的预处理结果
        save_data = {
            'processed_data': self.processed_data,
            'computed_metrics': self.computed_metrics,
            'helios_data': self.helios_data,
            'timestamp': datetime.now(),
            'config_path': self.config_path
        }

        # 同时保存Helios兼容格式的单独文件
        self._save_helios_data_files(cache_path.parent)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"预处理数据已保存: {cache_path}")

    def _save_helios_data_files(self, output_dir: Path):
        """保存Helios兼容格式的单独数据文件"""
        logger.info("保存Helios兼容格式数据文件...")

        helios_dir = output_dir / "helios_format"
        helios_dir.mkdir(exist_ok=True)

        # 保存各个数据文件
        if 'cluster_log' in self.helios_data:
            cluster_log_path = helios_dir / "cluster_log.csv"
            self.helios_data['cluster_log'].to_csv(cluster_log_path, index=False)
            logger.info(f"cluster_log.csv已保存: {cluster_log_path}")

        if 'cluster_sequence' in self.helios_data:
            sequence_path = helios_dir / "cluster_sequence.csv"
            self.helios_data['cluster_sequence'].to_csv(sequence_path, index=False)
            logger.info(f"cluster_sequence.csv已保存: {sequence_path}")

        if 'cluster_throughput' in self.helios_data:
            throughput_path = helios_dir / "cluster_throughput.csv"
            self.helios_data['cluster_throughput'].to_csv(throughput_path, index=False)
            logger.info(f"cluster_throughput.csv已保存: {throughput_path}")

        if 'cluster_user' in self.helios_data:
            user_path = helios_dir / "cluster_user.pkl"
            with open(user_path, 'wb') as f:
                pickle.dump(self.helios_data['cluster_user'], f)
            logger.info(f"cluster_user.pkl已保存: {user_path}")

        logger.info(f"Helios兼容格式数据文件保存完成: {helios_dir}")
    
    def _load_cached_data(self, cache_path: Path) -> Dict[str, Any]:
        """加载缓存的预处理数据"""
        with open(cache_path, 'rb') as f:
            saved_data = pickle.load(f)

        self.processed_data = saved_data['processed_data']
        self.computed_metrics = saved_data['computed_metrics']
        self.helios_data = saved_data.get('helios_data', {})

        # 如果缓存中没有Helios数据，重新生成
        if not self.helios_data and 'enhanced_data' in self.processed_data:
            logger.info("缓存中缺少Helios数据，重新生成...")
            enhanced_data = self.processed_data['enhanced_data']
            self._generate_helios_compatible_data(enhanced_data)

            # 更新缓存
            saved_data['helios_data'] = self.helios_data
            with open(cache_path, 'wb') as f:
                pickle.dump(saved_data, f)
            logger.info("Helios数据已添加到缓存")

        logger.info(f"缓存数据加载完成: {saved_data['timestamp']}")

        # 返回完整的数据结构
        return {
            **self.processed_data,
            'helios_data': self.helios_data,
            'computed_metrics': self.computed_metrics
        }
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """获取特定数据集"""
        if dataset_name == 'full':
            return self.processed_data.get('enhanced_data', pd.DataFrame())
        elif dataset_name in self.processed_data.get('job_type_datasets', {}):
            return self.processed_data['job_type_datasets'][dataset_name]
        else:
            raise ValueError(f"未知的数据集: {dataset_name}")
    
    def get_metrics(self, metric_category: str = None) -> Dict[str, Any]:
        """获取计算的指标"""
        if metric_category is None:
            return self.computed_metrics
        else:
            return self.computed_metrics.get(metric_category, {})

    def get_helios_data(self, data_type: str = None) -> Any:
        """获取Helios兼容格式数据"""
        if data_type is None:
            return self.helios_data
        else:
            return self.helios_data.get(data_type, pd.DataFrame())

    def save_helios_data_to_directory(self, output_dir: str):
        """将Helios兼容数据保存到指定目录"""
        output_path = Path(output_dir)
        self._save_helios_data_files(output_path)


def main():
    """测试数据预处理模块"""
    preprocessor = HeliosCompatibleDataPreprocessor()
    processed_data = preprocessor.load_and_preprocess_all_data()

    print("预处理完成!")
    print(f"可用数据集: {list(processed_data.get('job_type_datasets', {}).keys())}")
    print(f"计算指标类别: {list(preprocessor.computed_metrics.keys())}")
    print(f"Helios兼容数据: {list(preprocessor.helios_data.keys())}")

    # 显示一些统计信息
    if 'cluster_log' in preprocessor.helios_data:
        cluster_log = preprocessor.helios_data['cluster_log']
        print(f"\nHelios cluster_log统计:")
        print(f"  总作业数: {len(cluster_log)}")
        print(f"  GPU作业数: {len(cluster_log[cluster_log['gpu_num'] > 0])}")
        print(f"  CPU作业数: {len(cluster_log[cluster_log['gpu_num'] == 0])}")
        print(f"  作业状态分布: {cluster_log['state'].value_counts().to_dict()}")

    # 保存Helios数据到输出目录
    preprocessor.save_helios_data_to_directory("data/processed")


if __name__ == "__main__":
    main()
