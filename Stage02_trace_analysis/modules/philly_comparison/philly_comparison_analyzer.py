#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Philly对比分析模块
基于Helios项目的Philly对比分析方法
严格复现Helios论文中的对比分析图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.data_loader import add_gpu_job_flag

logger = logging.getLogger(__name__)


class PhillyComparisonAnalyzer:
    """Philly对比分析器"""
    
    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path], visualizer):
        """
        初始化Philly对比分析器
        
        Args:
            config: 分析配置
            output_paths: 输出路径字典
            visualizer: 可视化器实例
        """
        self.config = config
        self.output_paths = output_paths
        self.visualizer = visualizer
        self.philly_data = None
        
    def analyze(self, hpc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Philly对比分析 - 使用Helios兼容数据格式

        Args:
            hpc_data: 预处理后的HPC数据（包含Helios兼容格式）

        Returns:
            对比分析结果
        """
        logger.info("开始Philly对比分析（使用Helios兼容数据）...")

        # 1. 加载Philly数据
        self.philly_data = self._load_philly_data()

        # 2. 使用Helios兼容数据格式
        logger.info(f"传入数据的键: {list(hpc_data.keys())}")
        hpc_helios_data = hpc_data.get('helios_data', {})
        logger.info(f"Helios数据键: {list(hpc_helios_data.keys()) if hpc_helios_data else '空'}")

        if not hpc_helios_data:
            logger.warning("未找到Helios兼容数据，使用原始数据")
            hpc_gpu_jobs = self._standardize_hpc_data(hpc_data)
        else:
            logger.info("找到Helios兼容数据，使用cluster_log格式")
            # 使用cluster_log.csv作为主要数据源
            hpc_gpu_jobs = self._standardize_helios_data(hpc_helios_data)

        philly_gpu_jobs = self._standardize_philly_data(self.philly_data)

        # 3. 执行对比分析
        results = {
            'dataset_comparison': self._compare_datasets(hpc_gpu_jobs, philly_gpu_jobs),
            'duration_comparison': self._compare_job_durations(hpc_gpu_jobs, philly_gpu_jobs),
            'gpu_time_comparison': self._compare_gpu_time_by_status(hpc_gpu_jobs, philly_gpu_jobs),
            'job_type_comparison': self._compare_job_types_helios(hpc_helios_data)
        }

        # 4. 生成4张对比图表（保持原有结构）
        # 传递cluster_log作为完整数据，而不是整个helios_data字典
        hpc_full_data = hpc_helios_data.get('cluster_log', pd.DataFrame()) if hpc_helios_data else pd.DataFrame()
        self._generate_comparison_visualizations(hpc_full_data, self.philly_data, hpc_gpu_jobs, philly_gpu_jobs, results)

        logger.info("Philly对比分析完成")
        return results
    
    def _load_philly_data(self) -> pd.DataFrame:
        """加载Philly数据集"""
        logger.info("加载Philly数据集...")

        # 获取当前模块所在目录，向上两级到Stage02_trace_analysis目录
        module_dir = Path(__file__).parent.parent.parent  # Stage02_trace_analysis/
        philly_path = module_dir / "data" / "external" / "philly_trace.csv"

        if not philly_path.exists():
            raise FileNotFoundError(f"Philly数据集未找到: {philly_path}")

        philly_df = pd.read_csv(philly_path)
        logger.info(f"Philly数据加载完成: {len(philly_df):,} 条记录")

        return philly_df

    def _standardize_helios_data(self, helios_data: Dict[str, Any]) -> pd.DataFrame:
        """标准化Helios兼容数据格式"""
        logger.info("使用Helios兼容数据格式...")

        if 'cluster_log' not in helios_data:
            raise ValueError("Helios数据中缺少cluster_log")

        cluster_log = helios_data['cluster_log'].copy()

        # 筛选GPU作业
        gpu_jobs = cluster_log[cluster_log['gpu_num'] > 0].copy()

        # 标准化字段名
        gpu_jobs = gpu_jobs.rename(columns={
            'job_id': 'job_id',
            'user': 'user_id',
            'gpu_num': 'gpu_demand',
            'duration': 'duration',
            'state': 'status',
            'submit_time': 'submit_time',
            'start_time': 'start_time',
            'end_time': 'end_time'
        })

        # 确保时间字段为datetime类型
        time_columns = ['submit_time', 'start_time', 'end_time']
        for col in time_columns:
            if col in gpu_jobs.columns:
                gpu_jobs[col] = pd.to_datetime(gpu_jobs[col])

        # 计算GPU时间
        gpu_jobs['gpu_time'] = gpu_jobs['duration'] * gpu_jobs['gpu_demand']

        logger.info(f"Helios GPU作业数据标准化完成: {len(gpu_jobs):,} 条记录")
        return gpu_jobs

    def _standardize_hpc_data(self, hpc_data: Dict[str, Any]) -> pd.DataFrame:
        """标准化HPC数据格式"""
        logger.info("标准化HPC数据格式...")
        
        # 获取GPU作业数据 - 基于exec_hosts识别
        if 'job_type_datasets' in hpc_data and 'gpu_jobs' in hpc_data['job_type_datasets']:
            gpu_jobs = hpc_data['job_type_datasets']['gpu_jobs'].copy()
        else:
            # 从完整数据中筛选GPU作业
            full_data = hpc_data.get('enhanced_data', pd.DataFrame())
            # 确保有GPU作业标识
            if 'is_gpu_job' not in full_data.columns:
                full_data = add_gpu_job_flag(full_data)
            gpu_jobs = full_data[full_data['is_gpu_job'] == True].copy()

            # 标准化字段名以匹配Helios格式
            gpu_jobs = gpu_jobs.rename(columns={
                'gpu_num': 'gpu_demand',
                'job_status_standardized': 'status'
            })
        
        if len(gpu_jobs) == 0:
            logger.warning("HPC数据中未找到GPU作业")
            return pd.DataFrame()
        
        # 标准化字段名和格式
        standardized = gpu_jobs.copy()
        
        # 状态映射 (映射到Philly格式，基于我们的实际数据)
        status_mapping = {
            'COMPLETED': 'Pass',  # DONE 和 EXIT(exit_status=0)
            'FAILED': 'Failed'    # EXIT(exit_status!=0)
        }

        standardized['state'] = standardized['job_status_standardized'].map(status_mapping)
        standardized = standardized.dropna(subset=['state'])
        
        # 确保必要字段存在
        required_fields = ['duration', 'gpu_demand', 'status']
        for field in required_fields:
            if field not in standardized.columns:
                logger.error(f"HPC数据缺少必要字段: {field}")
                return pd.DataFrame()
        
        # 过滤有效数据 - 确保是GPU作业且有有效持续时间
        if 'gpu_demand' in standardized.columns:
            # 使用Helios格式的gpu_demand字段
            standardized = standardized[
                (standardized['duration'] > 0) &
                (standardized['gpu_demand'] > 0)
            ]
        else:
            # 使用原始格式的is_gpu_job字段
            if 'is_gpu_job' not in standardized.columns:
                standardized = add_gpu_job_flag(standardized)
            standardized = standardized[
                (standardized['duration'] > 0) &
                (standardized['is_gpu_job'] == True)
            ]
        
        logger.info(f"HPC GPU作业标准化完成: {len(standardized):,} 条记录")
        return standardized
    
    def _standardize_philly_data(self, philly_df: pd.DataFrame) -> pd.DataFrame:
        """标准化Philly数据格式"""
        logger.info("标准化Philly数据格式...")
        
        # Philly数据已经是标准格式，只需要筛选GPU作业
        # Philly是专用GPU集群，所有作业都是GPU作业，但仍然基于gpu_num筛选以保持一致性
        gpu_jobs = philly_df[philly_df['gpu_num'] > 0].copy()
        
        # 确保duration字段存在且有效
        gpu_jobs = gpu_jobs[gpu_jobs['duration'] > 0]
        
        logger.info(f"Philly GPU作业标准化完成: {len(gpu_jobs):,} 条记录")
        return gpu_jobs
    
    def _compare_datasets(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame) -> Dict[str, Any]:
        """对比数据集基本信息"""
        logger.info("对比数据集基本信息...")
        
        comparison = {
            'HPC': {
                'total_gpu_jobs': len(hpc_data),
                'unique_users': hpc_data['user_id'].nunique() if 'user_id' in hpc_data.columns else 0,
                'avg_duration_seconds': float(hpc_data['duration'].mean()),
                'max_gpu_demand': int(hpc_data['gpu_demand'].max()),
                'total_gpu_hours': float((hpc_data['gpu_demand'] * hpc_data['duration'] / 3600).sum())
            },
            'Philly': {
                'total_gpu_jobs': len(philly_data),
                'unique_users': philly_data['user'].nunique() if 'user' in philly_data.columns else 0,
                'avg_duration_seconds': float(philly_data['duration'].mean()),
                'max_gpu_demand': int(philly_data['gpu_num'].max()),
                'total_gpu_hours': float((philly_data['gpu_num'] * philly_data['duration'] / 3600).sum())
            }
        }
        
        # 计算比例关系
        comparison['scale_ratio'] = comparison['HPC']['total_gpu_jobs'] / comparison['Philly']['total_gpu_jobs']
        comparison['duration_ratio'] = comparison['HPC']['avg_duration_seconds'] / comparison['Philly']['avg_duration_seconds']
        
        return comparison
    
    def _compare_job_durations(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame) -> Dict[str, Any]:
        """对比作业持续时间分布"""
        logger.info("对比作业持续时间分布...")
        
        # 按照Helios的方法计算CDF
        time_points = [2**i for i in range(0, 22)]  # 1秒到2^21秒
        
        duration_comparison = {}
        
        for dataset_name, data in [('HPC', hpc_data), ('Philly', philly_data)]:
            job_ratios = []
            for t in time_points:
                ratio = len(data[data['duration'] <= t]) / len(data) * 100
                job_ratios.append(ratio)
            
            duration_comparison[dataset_name] = {
                'time_points': time_points,
                'job_ratios': job_ratios,
                'median_duration': float(data['duration'].median()),
                'p95_duration': float(data['duration'].quantile(0.95))
            }
        
        return duration_comparison
    
    def _compare_gpu_time_by_status(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame) -> Dict[str, Any]:
        """对比GPU时间按状态分布"""
        logger.info("对比GPU时间按状态分布...")

        gpu_time_comparison = {}

        # 处理HPC数据
        hpc_total_gpu_time = (hpc_data['gpu_demand'] * hpc_data['duration']).sum()
        hpc_status_gpu_time = {}

        # HPC状态映射：Pass -> Completed, Failed -> Failed
        hpc_status_mapping = {
            'Pass': 'Completed',
            'Failed': 'Failed'
        }

        for hpc_status, standard_status in hpc_status_mapping.items():
            if 'status' in hpc_data.columns:
                status_data = hpc_data[hpc_data['status'] == hpc_status]
            elif 'job_status_standardized' in hpc_data.columns:
                # 使用标准化的作业状态字段
                status_data = hpc_data[hpc_data['job_status_standardized'] == hpc_status]
            else:
                # 使用Helios格式的state字段
                status_data = hpc_data[hpc_data['state'] == hpc_status]

            if len(status_data) > 0:
                gpu_time = (status_data['gpu_demand'] * status_data['duration']).sum()
                hpc_status_gpu_time[standard_status] = gpu_time / hpc_total_gpu_time * 100
            else:
                hpc_status_gpu_time[standard_status] = 0.0

        # 确保所有标准状态都有值
        for standard_status in ['Completed', 'Failed', 'Killed']:
            if standard_status not in hpc_status_gpu_time:
                hpc_status_gpu_time[standard_status] = 0.0

        gpu_time_comparison['HPC'] = hpc_status_gpu_time

        # 处理Philly数据
        philly_total_gpu_time = (philly_data['gpu_num'] * philly_data['duration']).sum()
        philly_status_gpu_time = {}

        # Philly状态映射：Pass -> Completed, Failed -> Failed, Killed -> Killed
        philly_status_mapping = {
            'Pass': 'Completed',
            'Failed': 'Failed',
            'Killed': 'Killed'
        }

        for philly_status, standard_status in philly_status_mapping.items():
            status_data = philly_data[philly_data['state'] == philly_status]
            if len(status_data) > 0:
                gpu_time = (status_data['gpu_num'] * status_data['duration']).sum()
                philly_status_gpu_time[standard_status] = gpu_time / philly_total_gpu_time * 100
            else:
                philly_status_gpu_time[standard_status] = 0.0

        gpu_time_comparison['Philly'] = philly_status_gpu_time

        return gpu_time_comparison

    def _compare_job_types(self, hpc_data: Dict[str, Any]) -> Dict[str, Any]:
        """对比作业类型分布 (CPU vs GPU)"""
        logger.info("对比作业类型分布...")

        job_type_comparison = {}

        # HPC数据的作业类型分布
        if 'job_type_datasets' in hpc_data:
            datasets = hpc_data['job_type_datasets']

            # 计算CPU和GPU作业数量
            cpu_count = len(datasets.get('cpu_jobs', []))
            gpu_count = len(datasets.get('gpu_jobs', []))
            total_count = cpu_count + gpu_count

            job_type_comparison['HPC'] = {
                'cpu_jobs': cpu_count,
                'gpu_jobs': gpu_count,
                'cpu_percentage': cpu_count / total_count * 100 if total_count > 0 else 0,
                'gpu_percentage': gpu_count / total_count * 100 if total_count > 0 else 0
            }

        # Philly数据 (假设全部是GPU作业)
        philly_gpu_count = len(self.philly_data)
        job_type_comparison['Philly'] = {
            'cpu_jobs': 0,  # Philly是专用GPU集群
            'gpu_jobs': philly_gpu_count,
            'cpu_percentage': 0.0,
            'gpu_percentage': 100.0
        }

        return job_type_comparison

    def _compare_job_types_helios(self, helios_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用Helios数据比较作业类型分布"""
        logger.info("使用Helios数据分析作业类型分布...")

        job_type_comparison = {}

        if helios_data and 'cluster_log' in helios_data:
            cluster_log = helios_data['cluster_log']

            # 计算GPU和CPU作业比例
            total_jobs = len(cluster_log)
            gpu_jobs = len(cluster_log[cluster_log['gpu_num'] > 0])
            cpu_jobs = total_jobs - gpu_jobs

            job_type_comparison['HPC'] = {
                'cpu_jobs': cpu_jobs,
                'gpu_jobs': gpu_jobs,
                'cpu_percentage': (cpu_jobs / total_jobs) * 100 if total_jobs > 0 else 0,
                'gpu_percentage': (gpu_jobs / total_jobs) * 100 if total_jobs > 0 else 0
            }

        # Philly数据 (假设全部是GPU作业)
        if self.philly_data is not None:
            philly_gpu_count = len(self.philly_data)
            job_type_comparison['Philly'] = {
                'cpu_jobs': 0,  # Philly是专用GPU集群
                'gpu_jobs': philly_gpu_count,
                'cpu_percentage': 0.0,
                'gpu_percentage': 100.0
            }

        return job_type_comparison

    def _generate_comparison_visualizations(self, hpc_full_data, philly_full_data,
                                          hpc_gpu_data: pd.DataFrame, philly_gpu_data: pd.DataFrame, results: Dict[str, Any]):
        """生成严格按照Helios风格的对比可视化图表 - Figure 2"""
        logger.info("生成Helios风格的Philly对比可视化图表...")

        # 设置Helios论文风格
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.linewidth': 1.2,
            'lines.linewidth': 2.5,
            'grid.linewidth': 0.8,
            'grid.alpha': 0.3,
            'legend.frameon': True,
            'legend.fancybox': False,
            'legend.shadow': False,
            'legend.numpoints': 1,
            'legend.scatterpoints': 1,
            'xtick.major.size': 4,
            'xtick.minor.size': 2,
            'ytick.major.size': 4,
            'ytick.minor.size': 2,
        })

        # 生成4个独立的Helios风格图表
        self._generate_job_type_distribution(hpc_full_data, philly_full_data)
        self._generate_gpu_job_status_count(hpc_gpu_data, philly_gpu_data)
        self._generate_gpu_duration_cdf(hpc_gpu_data, philly_gpu_data)
        self._generate_gpu_time_distribution(results)

    def _generate_job_type_distribution(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame):
        """图1: 作业类型占比图 (CPU vs GPU jobs)"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # 计算HPC数据的作业类型分布
        hpc_total = len(hpc_data)
        if 'gpu_num' in hpc_data.columns:
            # 使用Helios格式的gpu_num字段
            hpc_gpu_jobs = len(hpc_data[hpc_data['gpu_num'] > 0])
        elif 'gpu_demand' in hpc_data.columns:
            # 使用标准化后的gpu_demand字段
            hpc_gpu_jobs = len(hpc_data[hpc_data['gpu_demand'] > 0])
        elif 'is_gpu_job' in hpc_data.columns:
            # 使用原始格式的is_gpu_job字段
            hpc_gpu_jobs = len(hpc_data[hpc_data['is_gpu_job'] == True])
        else:
            # 默认假设没有GPU作业
            hpc_gpu_jobs = 0
        hpc_cpu_jobs = hpc_total - hpc_gpu_jobs
        hpc_gpu_pct = (hpc_gpu_jobs / hpc_total) * 100
        hpc_cpu_pct = (hpc_cpu_jobs / hpc_total) * 100

        # 计算Philly数据的作业类型分布
        philly_total = len(philly_data)
        philly_gpu_jobs = len(philly_data[philly_data['gpu_num'] > 0])
        philly_cpu_jobs = philly_total - philly_gpu_jobs
        philly_gpu_pct = (philly_gpu_jobs / philly_total) * 100
        philly_cpu_pct = (philly_cpu_jobs / philly_total) * 100

        # 绘制堆叠柱状图
        datasets = ['HPC', 'Philly']
        cpu_percentages = [hpc_cpu_pct, philly_cpu_pct]
        gpu_percentages = [hpc_gpu_pct, philly_gpu_pct]

        x = np.arange(len(datasets))
        width = 0.6

        bars1 = ax.bar(x, cpu_percentages, width, label='CPU Jobs',
                      color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x, gpu_percentages, width, bottom=cpu_percentages,
                      label='GPU Jobs', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)

        # 添加百分比标签
        for i, (cpu_pct, gpu_pct) in enumerate(zip(cpu_percentages, gpu_percentages)):
            if cpu_pct > 5:
                ax.text(i, cpu_pct/2, f'{cpu_pct:.1f}%', ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')
            if gpu_pct > 5:
                ax.text(i, cpu_pct + gpu_pct/2, f'{gpu_pct:.1f}%', ha='center', va='center',
                       fontsize=11, fontweight='bold', color='white')

        ax.set_xlabel('Datacenter', fontsize=12)
        ax.set_ylabel('Fraction of Jobs (%)', fontsize=12)
        ax.set_title('Job Type Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = self.output_paths.get('figures', self.output_paths.get('philly_comparison', Path('output'))) / 'job_type_distribution'
        self.visualizer._save_figure(fig, output_path)
        plt.close()
        logger.info(f"作业类型分布图已保存: {output_path}")

    def _generate_gpu_job_status_count(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame):
        """图2: GPU作业状态数量分布图 (Complete/Cancel/Failed 作业数量)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # hpc_data和philly_data已经是筛选过的GPU作业数据
        hpc_gpu_jobs = hpc_data
        philly_gpu_jobs = philly_data[philly_data['gpu_num'] > 0]

        # 计算HPC GPU作业状态分布
        if 'status' in hpc_gpu_jobs.columns:
            hpc_status_counts = hpc_gpu_jobs['status'].value_counts()
            # 使用标准化后的状态映射
            hpc_completed = hpc_status_counts.get('Pass', 0)  # Pass -> Completed
            hpc_failed = hpc_status_counts.get('Failed', 0)   # Failed -> Failed
            hpc_killed = 0  # HPC数据中没有Killed状态
        else:
            # 使用Helios格式的state字段
            hpc_status_counts = hpc_gpu_jobs['state'].value_counts()
            hpc_completed = hpc_status_counts.get('Pass', 0)  # Pass -> Completed
            hpc_failed = hpc_status_counts.get('Failed', 0)   # Failed -> Failed
            hpc_killed = 0  # HPC数据中没有Killed状态

        hpc_total = len(hpc_gpu_jobs)
        hpc_completed_pct = (hpc_completed / hpc_total) * 100 if hpc_total > 0 else 0
        hpc_failed_pct = (hpc_failed / hpc_total) * 100 if hpc_total > 0 else 0
        hpc_killed_pct = (hpc_killed / hpc_total) * 100 if hpc_total > 0 else 0

        # 计算Philly GPU作业状态分布
        philly_status_counts = philly_gpu_jobs['state'].value_counts()
        philly_total = len(philly_gpu_jobs)
        philly_completed = philly_status_counts.get('Pass', 0)
        philly_failed = philly_status_counts.get('Failed', 0)
        philly_killed = philly_status_counts.get('Killed', 0)

        philly_completed_pct = (philly_completed / philly_total) * 100 if philly_total > 0 else 0
        philly_failed_pct = (philly_failed / philly_total) * 100 if philly_total > 0 else 0
        philly_killed_pct = (philly_killed / philly_total) * 100 if philly_total > 0 else 0

        # 绘制分组柱状图
        datasets = ['HPC', 'Philly']
        x = np.arange(len(datasets))
        width = 0.25

        completed_bars = ax.bar(x - width, [hpc_completed_pct, philly_completed_pct],
                               width, label='Completed', color='#2ca02c', alpha=0.8,
                               edgecolor='black', linewidth=0.5)
        failed_bars = ax.bar(x, [hpc_failed_pct, philly_failed_pct],
                            width, label='Failed', color='#d62728', alpha=0.8,
                            edgecolor='black', linewidth=0.5)
        killed_bars = ax.bar(x + width, [hpc_killed_pct, philly_killed_pct],
                            width, label='Canceled', color='#ff7f0e', alpha=0.8,
                            edgecolor='black', linewidth=0.5)

        # 添加数值标签
        def add_labels(bars, values):
            for bar, value in zip(bars, values):
                if value > 1:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

        add_labels(completed_bars, [hpc_completed_pct, philly_completed_pct])
        add_labels(failed_bars, [hpc_failed_pct, philly_failed_pct])
        add_labels(killed_bars, [hpc_killed_pct, philly_killed_pct])

        ax.set_xlabel('Datacenter', fontsize=12)
        ax.set_ylabel('Fraction of GPU Jobs (%)', fontsize=12)
        ax.set_title('GPU Job Status Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, max(max(hpc_completed_pct, hpc_failed_pct, hpc_killed_pct),
                          max(philly_completed_pct, philly_failed_pct, philly_killed_pct)) + 10)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = self.output_paths.get('figures', self.output_paths.get('philly_comparison', Path('output'))) / 'gpu_job_count_status'
        self.visualizer._save_figure(fig, output_path)
        plt.close()
        logger.info(f"GPU作业状态数量分布图已保存: {output_path}")

    def _generate_gpu_duration_cdf(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame):
        """图3: GPU作业持续时间分布图 (Duration CDF)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # 获取GPU作业持续时间
        hpc_gpu_jobs = hpc_data[hpc_data['gpu_demand'] > 0]
        philly_gpu_jobs = philly_data[philly_data['gpu_num'] > 0]

        if len(hpc_gpu_jobs) > 0 and len(philly_gpu_jobs) > 0:
            hpc_durations = hpc_gpu_jobs['duration'].values
            philly_durations = philly_gpu_jobs['duration'].values

            # 使用Helios的时间点 (2^i for i in range(0, 22))
            time_points = [2 ** i for i in range(0, 22)]

            # 计算CDF
            hpc_cdf = []
            philly_cdf = []

            for t in time_points:
                hpc_ratio = len(hpc_durations[hpc_durations <= t]) / len(hpc_durations) * 100
                philly_ratio = len(philly_durations[philly_durations <= t]) / len(philly_durations) * 100
                hpc_cdf.append(hpc_ratio)
                philly_cdf.append(philly_ratio)

            # 绘制CDF - 使用Helios的线型和颜色
            ax.plot(time_points, hpc_cdf, '-', linewidth=3, label='HPC', color='#1f77b4')
            ax.plot(time_points, philly_cdf, '--', linewidth=3, label='Philly', color='#ff7f0e')

            ax.set_xlabel('GPU Job Duration (s)', fontsize=12)
            ax.set_ylabel('Fraction of Jobs (%)', fontsize=12)
            ax.set_title('CDF of GPU Job Duration', fontsize=14, fontweight='bold')
            ax.set_xscale('log')
            ax.set_xlim(1, 2**21)
            ax.set_ylim(0, 100)
            ax.grid(True, linestyle=':', alpha=0.7)
            ax.legend(loc='lower right', fontsize=11)

        plt.tight_layout()
        output_path = self.output_paths.get('figures', self.output_paths.get('philly_comparison', Path('output'))) / 'gpu_duration_cdf'
        self.visualizer._save_figure(fig, output_path)
        plt.close()
        logger.info(f"GPU作业持续时间CDF图已保存: {output_path}")

    def _generate_gpu_time_distribution(self, results: Dict[str, Any]):
        """图4: GPU时间分布图 (GPU时间按状态分布)"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if 'gpu_time_comparison' in results:
            gpu_time_data = results['gpu_time_comparison']

            # 准备数据
            datasets = ['HPC', 'Philly']
            x = np.arange(len(datasets))
            width = 0.25

            # 获取百分比数据
            hpc_completed = gpu_time_data['HPC'].get('Completed', 0)
            hpc_failed = gpu_time_data['HPC'].get('Failed', 0)
            hpc_killed = gpu_time_data['HPC'].get('Killed', 0)

            philly_completed = gpu_time_data['Philly'].get('Completed', 0)
            philly_failed = gpu_time_data['Philly'].get('Failed', 0)
            philly_killed = gpu_time_data['Philly'].get('Killed', 0)

            # 绘制分组柱状图
            completed_bars = ax.bar(x - width, [hpc_completed, philly_completed],
                                   width, label='Completed', color='#2ca02c', alpha=0.8,
                                   edgecolor='black', linewidth=0.5)
            failed_bars = ax.bar(x, [hpc_failed, philly_failed],
                                width, label='Failed', color='#d62728', alpha=0.8,
                                edgecolor='black', linewidth=0.5)
            killed_bars = ax.bar(x + width, [hpc_killed, philly_killed],
                                width, label='Canceled', color='#ff7f0e', alpha=0.8,
                                edgecolor='black', linewidth=0.5)

            # 添加数值标签
            def add_value_labels(bars, values):
                for bar, value in zip(bars, values):
                    if value > 1:
                        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                               f'{value:.1f}%', ha='center', va='bottom', fontsize=10)

            add_value_labels(completed_bars, [hpc_completed, philly_completed])
            add_value_labels(failed_bars, [hpc_failed, philly_failed])
            add_value_labels(killed_bars, [hpc_killed, philly_killed])

            ax.set_xlabel('Datacenter', fontsize=12)
            ax.set_ylabel('Fraction of GPU Time (%)', fontsize=12)
            ax.set_title('GPU Time Distribution by Status', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.set_ylim(0, 100)
            ax.grid(axis='y', linestyle=':', alpha=0.7)
            ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = self.output_paths.get('figures', self.output_paths.get('philly_comparison', Path('output'))) / 'gpu_time_status'
        self.visualizer._save_figure(fig, output_path)
        plt.close()
        logger.info(f"GPU时间分布图已保存: {output_path}")

    def _generate_helios_figure_2(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame, results: Dict[str, Any]):
        """生成严格按照Helios Figure 2的对比图表"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # (a) GPU作业持续时间CDF对比 - 严格按照Helios实现
        hpc_gpu_jobs = hpc_data[hpc_data['gpu_demand'] > 0]
        philly_gpu_jobs = philly_data[philly_data['gpu_num'] > 0]

        if len(hpc_gpu_jobs) > 0 and len(philly_gpu_jobs) > 0:
            hpc_durations = hpc_gpu_jobs['duration'].values
            philly_durations = philly_gpu_jobs['duration'].values

            # 使用Helios的时间点 (2^i for i in range(0, 22))
            time_points = [2 ** i for i in range(0, 22)]

            # 计算CDF
            hpc_cdf = []
            philly_cdf = []

            for t in time_points:
                hpc_ratio = len(hpc_durations[hpc_durations <= t]) / len(hpc_durations) * 100
                philly_ratio = len(philly_durations[philly_durations <= t]) / len(philly_durations) * 100
                hpc_cdf.append(hpc_ratio)
                philly_cdf.append(philly_ratio)

            # 绘制CDF - 使用Helios的线型和颜色
            ax1.plot(time_points, hpc_cdf, '-', linewidth=2.5, label='HPC', color='#1f77b4')
            ax1.plot(time_points, philly_cdf, '--', linewidth=2.5, label='Philly', color='#ff7f0e')

            ax1.set_xlabel('GPU Job Duration (s)', fontsize=11)
            ax1.set_ylabel('Fraction of Jobs (%)', fontsize=11)
            ax1.set_xscale('log')
            ax1.set_xlim(1, 2**21)
            ax1.set_ylim(0, 100)
            ax1.grid(True, linestyle=':', alpha=0.7)
            ax1.legend(loc='lower right', fontsize=10)
            ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

        # (b) GPU时间按状态分布 - 严格按照Helios实现
        if 'gpu_time_comparison' in results:
            gpu_time_data = results['gpu_time_comparison']

            # 准备数据
            datasets = ['HPC', 'Philly']
            x_pos = [1, 2]  # Helios使用1,2作为x坐标
            width = 0.25

            # 获取百分比数据
            hpc_completed = gpu_time_data['HPC'].get('Completed', 0)
            hpc_canceled = gpu_time_data['HPC'].get('Killed', 0)  # Helios中Killed对应Canceled
            hpc_failed = gpu_time_data['HPC'].get('Failed', 0)

            philly_completed = gpu_time_data['Philly'].get('Completed', 0)
            philly_canceled = gpu_time_data['Philly'].get('Killed', 0)
            philly_failed = gpu_time_data['Philly'].get('Failed', 0)

            # 绘制柱状图 - 按照Helios的样式
            completed_bars = ax2.bar([x - width for x in x_pos], [hpc_completed, philly_completed],
                                   width, label='Completed', color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)
            canceled_bars = ax2.bar(x_pos, [hpc_canceled, philly_canceled],
                                  width, label='Canceled', color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
            failed_bars = ax2.bar([x + width for x in x_pos], [hpc_failed, philly_failed],
                                width, label='Failed', color='#d62728', alpha=0.8, edgecolor='black', linewidth=0.5)

            # 添加数值标签 - Helios风格
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    if height > 1:  # 只显示大于1%的标签
                        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=9)

            add_value_labels(completed_bars)
            add_value_labels(canceled_bars)
            add_value_labels(failed_bars)

            ax2.set_xlabel('Datacenter', fontsize=11)
            ax2.set_ylabel('Fraction of GPU Time (%)', fontsize=11)
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(['HPC', 'Philly'])
            ax2.set_ylim(0, 100)
            ax2.grid(axis='y', linestyle=':', alpha=0.7)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

        plt.tight_layout()
        output_path = self.output_paths['figures'] / 'helios_figure_2_comparison'
        self.visualizer._save_figure(fig, output_path)
        plt.close()

        logger.info(f"Helios风格对比图表已保存: {output_path}")

    def _generate_job_type_chart(self, job_type_data: Dict[str, Any]):
        """生成图1: 作业类型分布对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        datasets = list(job_type_data.keys())
        cpu_percentages = [job_type_data[dataset]['cpu_percentage'] for dataset in datasets]
        gpu_percentages = [job_type_data[dataset]['gpu_percentage'] for dataset in datasets]

        x = np.arange(len(datasets))
        width = 0.35

        # 绘制堆叠条形图
        bars1 = ax.bar(x, cpu_percentages, width, label='CPU Jobs',
                      alpha=0.9, color='#1f77b4', linewidth=0.8, edgecolor='k')
        bars2 = ax.bar(x, gpu_percentages, width, bottom=cpu_percentages,
                      label='GPU Jobs', alpha=0.9, color='#ff7f0e',
                      linewidth=0.8, edgecolor='k')

        # 添加数值标签
        for i, dataset in enumerate(datasets):
            cpu_pct = cpu_percentages[i]
            gpu_pct = gpu_percentages[i]

            if cpu_pct > 5:
                ax.text(i, cpu_pct/2, f'{cpu_pct:.1f}%',
                       ha='center', va='center', fontsize=12, fontweight='bold')

            if gpu_pct > 5:
                ax.text(i, cpu_pct + gpu_pct/2, f'{gpu_pct:.1f}%',
                       ha='center', va='center', fontsize=12, fontweight='bold')

        ax.set_xlabel('Datacenter', fontsize=12)
        ax.set_ylabel('Fraction of Jobs (%)', fontsize=12)
        ax.set_title('Job Type Distribution Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = self.output_paths['figures'] / 'job_type_distribution'
        self.visualizer._save_figure(fig, output_path)
        plt.close()

    def _generate_job_count_status_chart(self, hpc_data: pd.DataFrame, philly_data: pd.DataFrame):
        """生成图2: GPU作业状态数量分布图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        datasets = ['HPC', 'Philly']
        data_sources = [hpc_data, philly_data]

        # 计算每个数据集的状态分布
        status_data = {}
        for dataset, data in zip(datasets, data_sources):
            total_jobs = len(data)
            status_counts = {}

            for status in ['Pass', 'Killed', 'Failed']:
                if 'job_status_standardized' in data.columns:
                    count = len(data[data['job_status_standardized'] == status])
                else:
                    count = len(data[data['state'] == status])
                status_counts[status] = count / total_jobs * 100

            status_data[dataset] = status_counts

        # 绘制分组条形图
        x = np.arange(len(datasets))
        width = 0.25

        colors = {'Pass': '#2ca02c', 'Killed': '#ff7f0e', 'Failed': '#d62728'}
        labels = {'Pass': 'Completed', 'Killed': 'Canceled', 'Failed': 'Failed'}

        for i, status in enumerate(['Pass', 'Killed', 'Failed']):
            values = [status_data[dataset][status] for dataset in datasets]
            bars = ax.bar(x + i*width, values, width, label=labels[status],
                         color=colors[status], alpha=0.9, linewidth=0.8, edgecolor='k')

            # 添加数值标签
            for j, (bar, value) in enumerate(zip(bars, values)):
                if value > 1:  # 只在值大于1%时显示标签
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xlabel('Datacenter', fontsize=12)
        ax.set_ylabel('Fraction of Jobs (%)', fontsize=12)
        ax.set_title('GPU Job Status Distribution (by Job Count)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)

        plt.tight_layout()
        output_path = self.output_paths['figures'] / 'gpu_job_count_status'
        self.visualizer._save_figure(fig, output_path)
        plt.close()

    def _generate_duration_cdf_chart(self, duration_data: Dict[str, Any]):
        """生成图3: GPU作业持续时间CDF对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        self._plot_duration_cdf_comparison(ax, duration_data)
        ax.set_title('GPU Job Duration CDF Comparison', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_paths['figures'] / 'gpu_duration_cdf'
        self.visualizer._save_figure(fig, output_path)
        plt.close()

    def _generate_gpu_time_status_chart(self, gpu_time_data: Dict[str, Any]):
        """生成图4: GPU时间按状态分布图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        self._plot_gpu_time_status_comparison(ax, gpu_time_data)
        ax.set_title('GPU Time Distribution by Job Status', fontsize=14, fontweight='bold')

        plt.tight_layout()
        output_path = self.output_paths['figures'] / 'gpu_time_status'
        self.visualizer._save_figure(fig, output_path)
        plt.close()

    def _plot_duration_cdf_comparison(self, ax, duration_data: Dict[str, Any]):
        """绘制持续时间CDF对比图 (图a)"""
        
        linestyles = ['-', '--']  # HPC用实线，Philly用虚线
        colors = ['#1f77b4', '#ff7f0e']  # 蓝色和橙色
        
        for i, (dataset, data) in enumerate(duration_data.items()):
            ax.plot(data['time_points'], data['job_ratios'], 
                   linestyle=linestyles[i], color=colors[i], 
                   alpha=0.9, label=dataset, linewidth=2)
        
        ax.set_xlabel('GPU Job Duration (s)', fontsize=12)
        ax.set_ylabel('Fraction of Jobs (%)', fontsize=12)
        ax.set_xscale('log')
        ax.set_xlim(1, 2**21)
        ax.set_ylim(0, 100)
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(linestyle=':', alpha=0.7)
    
    def _plot_gpu_time_status_comparison(self, ax, gpu_time_data: Dict[str, Any]):
        """绘制GPU时间状态分布对比图 (图b)"""
        
        datasets = list(gpu_time_data.keys())
        statuses = ['Pass', 'Killed', 'Failed']
        status_labels = ['Completed', 'Canceled', 'Failed']
        colors = ['#2ca02c', '#ff7f0e', '#d62728']  # 绿色、橙色、红色
        
        x = np.arange(len(datasets))
        width = 0.25
        
        for i, (status, label, color) in enumerate(zip(statuses, status_labels, colors)):
            values = [gpu_time_data[dataset].get(status, 0) for dataset in datasets]
            bars = ax.bar(x + i * width - width, values, width, 
                         label=label, alpha=0.9, color=color,
                         linewidth=0.8, edgecolor='k')
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Datacenter', fontsize=12)
        ax.set_ylabel('Fraction of GPU Time (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle=':', alpha=0.7)
        ax.legend(fontsize=11)

    def generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """生成对比分析报告"""
        logger.info("生成Philly对比分析报告...")
        
        report_lines = [
            "HPC vs Philly数据集对比分析报告",
            "=" * 50,
            "",
            "数据集基本信息对比:",
            "-" * 30
        ]
        
        dataset_comp = results['dataset_comparison']
        
        report_lines.extend([
            f"HPC数据集:",
            f"  总GPU作业数: {dataset_comp['HPC']['total_gpu_jobs']:,}",
            f"  平均持续时间: {dataset_comp['HPC']['avg_duration_seconds']:.0f} 秒",
            f"  最大GPU需求: {dataset_comp['HPC']['max_gpu_demand']} GPU",
            f"  总GPU时间: {dataset_comp['HPC']['total_gpu_hours']:.0f} 小时",
            "",
            f"Philly数据集:",
            f"  总GPU作业数: {dataset_comp['Philly']['total_gpu_jobs']:,}",
            f"  平均持续时间: {dataset_comp['Philly']['avg_duration_seconds']:.0f} 秒", 
            f"  最大GPU需求: {dataset_comp['Philly']['max_gpu_demand']} GPU",
            f"  总GPU时间: {dataset_comp['Philly']['total_gpu_hours']:.0f} 小时",
            "",
            "对比结果:",
            f"  规模比例 (HPC/Philly): {dataset_comp['scale_ratio']:.1f}x",
            f"  持续时间比例 (HPC/Philly): {dataset_comp['duration_ratio']:.2f}x",
            ""
        ])
        
        # GPU时间浪费对比
        if 'gpu_time_comparison' in results:
            gpu_time = results['gpu_time_comparison']
            hpc_waste = gpu_time['HPC'].get('Killed', 0) + gpu_time['HPC'].get('Failed', 0)
            philly_waste = gpu_time['Philly'].get('Killed', 0) + gpu_time['Philly'].get('Failed', 0)
            
            report_lines.extend([
                "GPU时间浪费对比:",
                f"  HPC失败作业GPU时间浪费: {hpc_waste:.1f}%",
                f"  Philly失败作业GPU时间浪费: {philly_waste:.1f}%",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_path = self.output_paths['reports'] / 'philly_comparison_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"对比分析报告已保存: {report_path}")
        return report_content
