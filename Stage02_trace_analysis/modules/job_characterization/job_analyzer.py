#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
作业特征分析模块
基于Helios项目的作业分析方法，分析作业持续时间、资源需求等特征
严格按照Helios的分析方法和可视化风格
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.data_loader import add_gpu_job_flag

logger = logging.getLogger(__name__)


class JobCharacterizationAnalyzer:
    """作业特征分析器"""
    
    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path], visualizer):
        """
        初始化作业特征分析器
        
        Args:
            config: 分析配置
            output_paths: 输出路径字典
            visualizer: 可视化器实例
        """
        self.config = config
        self.output_paths = output_paths
        self.visualizer = visualizer
        
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行作业特征分析 - 严格按照Helios方法

        Args:
            data: 预处理后的数据（包含Helios兼容格式）

        Returns:
            分析结果字典
        """
        logger.info("开始作业特征分析（Helios风格）...")

        # 获取数据
        helios_data = data.get('helios_data', {})
        if not helios_data or 'cluster_log' not in helios_data:
            raise ValueError("缺少Helios兼容数据")

        cluster_log = helios_data['cluster_log']

        # 获取原始数据用于趋势分析
        original_data = data.get('processed_data')

        # 按照Helios的分析方法执行分析
        results = {
            # 0. 作业提交趋势分析（全局概览）
            'submission_trend': self._analyze_submission_trend(original_data) if original_data is not None else {},

            # 1. GPU数量分布分析（对应Helios Figure 4a）
            'gpu_distribution': self._analyze_gpu_distribution_helios(cluster_log),

            # 2. GPU时间分布分析（对应Helios Figure 4b）
            'gpu_time_distribution': self._analyze_gpu_time_distribution_helios(cluster_log),

            # 3. CPU数量分布分析（新增 - 对应GPU分析）
            'cpu_distribution': self._analyze_cpu_distribution_helios(cluster_log),

            # 4. CPU时间分布分析（新增 - 对应GPU分析）
            'cpu_time_distribution': self._analyze_cpu_time_distribution_helios(cluster_log),

            # 5. 作业状态分析（对应Helios的状态分布）
            'job_status': self._analyze_job_status_helios(cluster_log)
        }

        # 生成趋势可视化（如果有原始数据）
        if original_data is not None and 'submission_trend' in results:
            self._generate_submission_trend_visualization(original_data, results['submission_trend'])

        # 生成Helios风格可视化（Figure 4复现）
        self._generate_helios_visualizations(cluster_log, results)

        logger.info("作业特征分析完成")
        return results

    def _analyze_submission_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析作业提交趋势 - 全局概览"""
        logger.info("分析作业提交趋势...")

        if df is None or 'submit_time' not in df.columns:
            return {}

        # 确保submit_time是datetime类型
        df = df.copy()
        df['submit_time'] = pd.to_datetime(df['submit_time'])

        # 按日期聚合作业提交数量
        df['submit_date'] = df['submit_time'].dt.date

        # 创建GPU和CPU作业标识
        df['is_gpu_job'] = df['gpu_num'] > 0
        df['is_cpu_job'] = df['gpu_num'] == 0

        daily_submissions = df.groupby('submit_date').agg({
            'job_id': 'count',
            'is_gpu_job': 'sum',  # GPU作业数量
            'is_cpu_job': 'sum'   # CPU作业数量
        }).reset_index()

        daily_submissions.columns = ['date', 'total_jobs', 'gpu_jobs', 'cpu_jobs']

        # 计算统计信息
        stats = {
            'daily_submissions': daily_submissions,
            'total_days': len(daily_submissions),
            'avg_daily_jobs': daily_submissions['total_jobs'].mean(),
            'max_daily_jobs': daily_submissions['total_jobs'].max(),
            'min_daily_jobs': daily_submissions['total_jobs'].min(),
            'std_daily_jobs': daily_submissions['total_jobs'].std(),
            'date_range': {
                'start': daily_submissions['date'].min(),
                'end': daily_submissions['date'].max()
            },
            'gpu_job_ratio': daily_submissions['gpu_jobs'].sum() / daily_submissions['total_jobs'].sum() * 100
        }

        logger.info(f"分析了 {stats['total_days']} 天的数据")
        logger.info(f"平均每日作业数: {stats['avg_daily_jobs']:.1f}")
        logger.info(f"GPU作业占比: {stats['gpu_job_ratio']:.2f}%")

        return stats

    def _analyze_gpu_distribution_helios(self, cluster_log: pd.DataFrame) -> Dict[str, Any]:
        """分析GPU数量分布 - 严格按照Helios方法（Figure 4a）"""
        logger.info("分析GPU数量分布（Helios风格）...")

        # 筛选GPU作业
        gpu_jobs = cluster_log[cluster_log['gpu_num'] > 0].copy()

        if len(gpu_jobs) == 0:
            return {}

        # 按照Helios的GPU数量范围进行分析
        gpu_ranges = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80]
        job_ratios = []

        total_jobs = len(gpu_jobs)

        for gpu_count in gpu_ranges:
            jobs_with_gpu_le = len(gpu_jobs[gpu_jobs['gpu_num'] <= gpu_count])
            job_ratio = (jobs_with_gpu_le / total_jobs) * 100
            job_ratios.append(job_ratio)

        return {
            'gpu_ranges': gpu_ranges,
            'job_ratios': job_ratios,
            'total_gpu_jobs': total_jobs
        }

    def _analyze_gpu_time_distribution_helios(self, cluster_log: pd.DataFrame) -> Dict[str, Any]:
        """分析GPU时间分布 - 严格按照Helios方法（Figure 4b）"""
        logger.info("分析GPU时间分布（Helios风格）...")

        # 筛选GPU作业
        gpu_jobs = cluster_log[cluster_log['gpu_num'] > 0].copy()

        if len(gpu_jobs) == 0:
            return {}

        # 计算GPU时间
        gpu_jobs['gtime'] = gpu_jobs['duration'] * gpu_jobs['gpu_num']

        # 按照Helios的GPU数量范围进行分析
        gpu_ranges = [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 80]
        gtime_ratios = []

        total_gtime = gpu_jobs['gtime'].sum()

        for gpu_count in gpu_ranges:
            gtime_with_gpu_le = gpu_jobs[gpu_jobs['gpu_num'] <= gpu_count]['gtime'].sum()
            gtime_ratio = (gtime_with_gpu_le / total_gtime) * 100
            gtime_ratios.append(gtime_ratio)

        return {
            'gpu_ranges': gpu_ranges,
            'gtime_ratios': gtime_ratios,
            'total_gtime': total_gtime
        }

    def _analyze_cpu_distribution_helios(self, cluster_log: pd.DataFrame) -> Dict[str, Any]:
        """分析CPU数量分布 - 对应GPU分析方法"""
        logger.info("分析CPU数量分布（Helios风格）...")

        # 筛选CPU作业
        cpu_jobs = cluster_log[cluster_log['gpu_num'] == 0].copy()

        if len(cpu_jobs) == 0:
            return {}

        # 按照CPU核心数范围进行分析（类似GPU分析）
        cpu_ranges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        job_ratios = []

        total_jobs = len(cpu_jobs)

        for cpu_count in cpu_ranges:
            jobs_with_cpu_le = len(cpu_jobs[cpu_jobs['cpu_num'] <= cpu_count])
            job_ratio = (jobs_with_cpu_le / total_jobs) * 100
            job_ratios.append(job_ratio)

        return {
            'cpu_ranges': cpu_ranges,
            'job_ratios': job_ratios,
            'total_cpu_jobs': total_jobs
        }

    def _analyze_cpu_time_distribution_helios(self, cluster_log: pd.DataFrame) -> Dict[str, Any]:
        """分析CPU时间分布 - 对应GPU分析方法"""
        logger.info("分析CPU时间分布（Helios风格）...")

        # 筛选CPU作业
        cpu_jobs = cluster_log[cluster_log['gpu_num'] == 0].copy()

        if len(cpu_jobs) == 0:
            return {}

        # 计算CPU时间
        cpu_jobs['ctime'] = cpu_jobs['duration'] * cpu_jobs['cpu_num']

        # 按照CPU核心数范围进行分析
        cpu_ranges = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        ctime_ratios = []

        total_ctime = cpu_jobs['ctime'].sum()

        for cpu_count in cpu_ranges:
            ctime_with_cpu_le = cpu_jobs[cpu_jobs['cpu_num'] <= cpu_count]['ctime'].sum()
            ctime_ratio = (ctime_with_cpu_le / total_ctime) * 100
            ctime_ratios.append(ctime_ratio)

        return {
            'cpu_ranges': cpu_ranges,
            'ctime_ratios': ctime_ratios,
            'total_ctime': total_ctime
        }

    def _analyze_job_status_helios(self, cluster_log: pd.DataFrame) -> Dict[str, Any]:
        """分析作业状态分布 - 严格按照Helios方法"""
        logger.info("分析作业状态分布（Helios风格）...")

        # 分别分析CPU和GPU作业的状态分布
        cpu_jobs = cluster_log[cluster_log['gpu_num'] == 0]
        gpu_jobs = cluster_log[cluster_log['gpu_num'] > 0]

        cpu_status = cpu_jobs['state'].value_counts(normalize=True) * 100
        gpu_status = gpu_jobs['state'].value_counts(normalize=True) * 100

        return {
            'cpu_status_distribution': cpu_status.to_dict(),
            'gpu_status_distribution': gpu_status.to_dict(),
            'cpu_job_count': len(cpu_jobs),
            'gpu_job_count': len(gpu_jobs)
        }

    def _analyze_job_duration(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析作业持续时间 (类似Helios的duration analysis)"""
        logger.info("分析作业持续时间...")
        
        if 'duration' not in df.columns:
            return {}
        
        # 移除无效持续时间
        valid_duration = df[df['duration'] > 0]['duration']
        
        if len(valid_duration) == 0:
            return {}
        
        # 转换为小时
        duration_hours = valid_duration / 3600
        
        # 基本统计
        stats = {
            'total_jobs': len(valid_duration),
            'mean_hours': float(duration_hours.mean()),
            'median_hours': float(duration_hours.median()),
            'std_hours': float(duration_hours.std()),
            'min_hours': float(duration_hours.min()),
            'max_hours': float(duration_hours.max())
        }
        
        # 百分位数
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}_hours'] = float(np.percentile(duration_hours, p))
        
        # 按作业类型分析
        duration_by_type = {}
        if 'job_type' in df.columns:
            for job_type in df['job_type'].unique():
                if pd.notna(job_type):
                    type_data = df[df['job_type'] == job_type]
                    type_duration = type_data[type_data['duration'] > 0]['duration'] / 3600
                    
                    if len(type_duration) > 0:
                        duration_by_type[job_type] = {
                            'count': len(type_duration),
                            'mean_hours': float(type_duration.mean()),
                            'median_hours': float(type_duration.median())
                        }
        
        stats['duration_by_job_type'] = duration_by_type
        
        return stats
    
    def _analyze_queue_time(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析排队时间"""
        logger.info("分析排队时间...")
        
        if 'queue_time' not in df.columns:
            return {}
        
        # 移除无效排队时间
        valid_queue_time = df[df['queue_time'] >= 0]['queue_time']
        
        if len(valid_queue_time) == 0:
            return {}
        
        # 转换为小时
        queue_hours = valid_queue_time / 3600
        
        # 基本统计
        stats = {
            'total_jobs': len(valid_queue_time),
            'mean_hours': float(queue_hours.mean()),
            'median_hours': float(queue_hours.median()),
            'std_hours': float(queue_hours.std()),
            'min_hours': float(queue_hours.min()),
            'max_hours': float(queue_hours.max())
        }
        
        # 百分位数
        percentiles = [50, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}_hours'] = float(np.percentile(queue_hours, p))
        
        # 按作业类型分析排队时间
        queue_by_type = {}
        if 'job_type' in df.columns:
            for job_type in df['job_type'].unique():
                if pd.notna(job_type):
                    type_data = df[df['job_type'] == job_type]
                    type_queue = type_data[type_data['queue_time'] >= 0]['queue_time'] / 3600
                    
                    if len(type_queue) > 0:
                        queue_by_type[job_type] = {
                            'count': len(type_queue),
                            'mean_hours': float(type_queue.mean()),
                            'median_hours': float(type_queue.median())
                        }
        
        stats['queue_time_by_job_type'] = queue_by_type
        
        return stats
    
    def _analyze_resource_demand(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析资源需求"""
        logger.info("分析资源需求...")
        
        resource_stats = {}
        
        # 节点数分析
        if 'actual_node_count' in df.columns:
            node_counts = df[df['actual_node_count'] > 0]['actual_node_count']
            
            if len(node_counts) > 0:
                resource_stats['node_demand'] = {
                    'total_jobs': len(node_counts),
                    'mean_nodes': float(node_counts.mean()),
                    'median_nodes': float(node_counts.median()),
                    'max_nodes': int(node_counts.max()),
                    'single_node_jobs': int((node_counts == 1).sum()),
                    'multi_node_jobs': int((node_counts > 1).sum())
                }
                
                # 节点数分布
                node_dist = node_counts.value_counts().sort_index()
                resource_stats['node_distribution'] = node_dist.to_dict()
        
        # GPU数量分析 - 基于exec_hosts识别
        if 'is_gpu_job' not in df.columns:
            df = add_gpu_job_flag(df)

        gpu_data = df[df['is_gpu_job'] == True]
        if len(gpu_data) > 0:
            # 如果有gpu_num字段，使用它进行详细分析
            if 'gpu_num' in df.columns and gpu_data['gpu_num'].notna().sum() > 0:
                gpu_counts = gpu_data[gpu_data['gpu_num'].notna()]['gpu_num']
                resource_stats['gpu_demand'] = {
                    'total_gpu_jobs': len(gpu_data),
                    'jobs_with_gpu_count': len(gpu_counts),
                    'mean_gpus': float(gpu_counts.mean()),
                    'median_gpus': float(gpu_counts.median()),
                    'max_gpus': int(gpu_counts.max()),
                    'total_gpu_hours': float((gpu_counts * gpu_data['duration'] / 3600).sum())
                }
        
        # 按子集群分析资源使用
        if 'primary_subcluster' in df.columns:
            subcluster_usage = df['primary_subcluster'].value_counts()
            resource_stats['subcluster_usage'] = subcluster_usage.to_dict()
        
        return resource_stats
    
    def _analyze_job_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析作业状态 (类似Helios的job outcome analysis)"""
        logger.info("分析作业状态...")
        
        if 'job_status_str' not in df.columns:
            return {}
        
        # 状态分布
        status_counts = df['job_status_str'].value_counts()
        
        # 定义状态类别
        completed_statuses = ['COMPLETED', 'C']
        failed_statuses = ['FAILED', 'F', 'CANCELLED', 'CA', 'TIMEOUT', 'TO']
        
        completed_count = sum(status_counts.get(status, 0) for status in completed_statuses)
        failed_count = sum(status_counts.get(status, 0) for status in failed_statuses)
        total_count = len(df)
        
        stats = {
            'status_distribution': status_counts.to_dict(),
            'completion_stats': {
                'completed_jobs': int(completed_count),
                'failed_jobs': int(failed_count),
                'total_jobs': int(total_count),
                'success_rate': float(completed_count / total_count * 100) if total_count > 0 else 0,
                'failure_rate': float(failed_count / total_count * 100) if total_count > 0 else 0
            }
        }
        
        # 按作业类型分析成功率
        success_by_type = {}
        if 'job_type' in df.columns:
            for job_type in df['job_type'].unique():
                if pd.notna(job_type):
                    type_data = df[df['job_type'] == job_type]
                    type_completed = sum(
                        (type_data['job_status_str'] == status).sum() 
                        for status in completed_statuses
                    )
                    type_total = len(type_data)
                    
                    success_by_type[job_type] = {
                        'completed': int(type_completed),
                        'total': int(type_total),
                        'success_rate': float(type_completed / type_total * 100) if type_total > 0 else 0
                    }
        
        stats['success_rate_by_job_type'] = success_by_type
        
        return stats
    
    def _analyze_job_size(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析作业规模分布"""
        logger.info("分析作业规模分布...")
        
        size_stats = {}
        
        # 基于节点数的作业规模分类
        if 'actual_node_count' in df.columns:
            node_counts = df[df['actual_node_count'] > 0]['actual_node_count']
            
            if len(node_counts) > 0:
                # 定义规模类别
                small_jobs = (node_counts == 1).sum()
                medium_jobs = ((node_counts > 1) & (node_counts <= 8)).sum()
                large_jobs = ((node_counts > 8) & (node_counts <= 64)).sum()
                xlarge_jobs = (node_counts > 64).sum()
                
                size_stats['job_size_categories'] = {
                    'small_jobs_1_node': int(small_jobs),
                    'medium_jobs_2_8_nodes': int(medium_jobs),
                    'large_jobs_9_64_nodes': int(large_jobs),
                    'xlarge_jobs_65plus_nodes': int(xlarge_jobs)
                }
                
                # 计算资源消耗 (节点小时)
                if 'duration' in df.columns:
                    df_with_duration = df[(df['actual_node_count'] > 0) & (df['duration'] > 0)]
                    node_hours = df_with_duration['actual_node_count'] * df_with_duration['duration'] / 3600
                    
                    size_stats['resource_consumption'] = {
                        'total_node_hours': float(node_hours.sum()),
                        'mean_node_hours_per_job': float(node_hours.mean()),
                        'median_node_hours_per_job': float(node_hours.median())
                    }
        
        return size_stats

    def _generate_submission_trend_visualization(self, df: pd.DataFrame, trend_stats: Dict[str, Any]):
        """生成作业提交趋势可视化图表"""
        logger.info("生成作业提交趋势图...")

        if not trend_stats or 'daily_submissions' not in trend_stats:
            return

        daily_data = trend_stats['daily_submissions']

        # 设置图表风格
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # 转换日期为datetime用于绘图
        dates = pd.to_datetime(daily_data['date'])

        # 上图：总体作业提交趋势
        ax1.plot(dates, daily_data['total_jobs'],
                linewidth=1.5, color='#1f77b4', alpha=0.8, label='Total Jobs')

        # 添加7天移动平均线
        if len(daily_data) >= 7:
            rolling_avg = daily_data['total_jobs'].rolling(window=7, center=True).mean()
            ax1.plot(dates, rolling_avg,
                    linewidth=2.5, color='#ff7f0e', label='7-day Moving Average')

        ax1.set_title('Daily Job Submission Trend', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Jobs', fontsize=12)
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend()

        # 格式化x轴日期
        ax1.tick_params(axis='x', rotation=45)

        # 下图：CPU vs GPU作业对比
        ax2.plot(dates, daily_data['cpu_jobs'],
                linewidth=1.5, color='#2ca02c', alpha=0.8, label='CPU Jobs')
        ax2.plot(dates, daily_data['gpu_jobs'],
                linewidth=1.5, color='#d62728', alpha=0.8, label='GPU Jobs')

        ax2.set_title('CPU vs GPU Job Submission Trend', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Number of Jobs', fontsize=12)
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend()

        # 格式化x轴日期
        ax2.tick_params(axis='x', rotation=45)

        # 添加统计信息文本框
        stats_text = f"""Statistics:
Total Days: {trend_stats['total_days']}
Avg Daily Jobs: {trend_stats['avg_daily_jobs']:.1f}
Max Daily Jobs: {trend_stats['max_daily_jobs']:,}
GPU Job Ratio: {trend_stats['gpu_job_ratio']:.2f}%"""

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)

        plt.tight_layout()

        # 保存图表
        output_path = self.output_paths['job_characterization'] / "job_submission_trend.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"作业提交趋势图已保存到: {output_path}")

    def _generate_helios_visualizations(self, cluster_log: pd.DataFrame, results: Dict[str, Any]):
        """生成Helios风格的作业特征可视化图表 - 严格按照Helios Figure 4"""
        logger.info("生成Helios风格的作业特征可视化图表...")

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
            'legend.shadow': False
        })

        # 1. GPU作业特征图（对应Helios Figure 4）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # GPU数量分布（左图 - Figure 4a）
        if 'gpu_distribution' in results:
            self._plot_gpu_distribution_helios(ax1, results['gpu_distribution'])

        # GPU时间分布（右图 - Figure 4b）
        if 'gpu_time_distribution' in results:
            self._plot_gpu_time_distribution_helios(ax2, results['gpu_time_distribution'])

        # 保存GPU图表
        gpu_output_path = self.output_paths['job_characterization'] / "job_characterization_gpu_helios.png"
        plt.savefig(gpu_output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 2. CPU作业特征图（新增 - 对应GPU分析）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # CPU数量分布（左图）
        if 'cpu_distribution' in results:
            self._plot_cpu_distribution_helios(ax1, results['cpu_distribution'])

        # CPU时间分布（右图）
        if 'cpu_time_distribution' in results:
            self._plot_cpu_time_distribution_helios(ax2, results['cpu_time_distribution'])

        # 保存CPU图表
        cpu_output_path = self.output_paths['job_characterization'] / "job_characterization_cpu_helios.png"
        plt.savefig(cpu_output_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 作业状态分布图（单独的图）
        if 'job_status' in results:
            self._plot_job_status_helios(results['job_status'])

        logger.info(f"Helios风格GPU作业特征图表已保存: {gpu_output_path}")
        logger.info(f"Helios风格CPU作业特征图表已保存: {cpu_output_path}")

    def _plot_gpu_distribution_helios(self, ax, gpu_data: Dict[str, Any]):
        """绘制GPU数量分布 - Helios风格（Figure 4a）"""
        if 'gpu_ranges' not in gpu_data or 'job_ratios' not in gpu_data:
            return

        gpu_ranges = gpu_data['gpu_ranges']
        job_ratios = gpu_data['job_ratios']

        # 绘制线图
        ax.plot(gpu_ranges, job_ratios, linestyle='-', linewidth=2.5,
               marker='o', markersize=6, color='#1f77b4')

        # 设置标签和格式
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Percentage of Jobs (%)')
        ax.set_title('(a) GPU Job Distribution')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴为对数刻度（如果需要）
        ax.set_xscale('log')
        ax.set_xlim(1, 80)
        ax.set_ylim(0, 100)

    def _plot_gpu_time_distribution_helios(self, ax, gtime_data: Dict[str, Any]):
        """绘制GPU时间分布 - Helios风格（Figure 4b）"""
        if 'gpu_ranges' not in gtime_data or 'gtime_ratios' not in gtime_data:
            return

        gpu_ranges = gtime_data['gpu_ranges']
        gtime_ratios = gtime_data['gtime_ratios']

        # 绘制线图
        ax.plot(gpu_ranges, gtime_ratios, linestyle='-', linewidth=2.5,
               marker='s', markersize=6, color='#ff7f0e')

        # 设置标签和格式
        ax.set_xlabel('Number of GPUs')
        ax.set_ylabel('Percentage of GPU Time (%)')
        ax.set_title('(b) GPU Time Distribution')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴为对数刻度（如果需要）
        ax.set_xscale('log')
        ax.set_xlim(1, 80)
        ax.set_ylim(0, 100)

    def _plot_cpu_distribution_helios(self, ax, cpu_data: Dict[str, Any]):
        """绘制CPU数量分布 - 对应GPU分析风格"""
        if 'cpu_ranges' not in cpu_data or 'job_ratios' not in cpu_data:
            return

        cpu_ranges = cpu_data['cpu_ranges']
        job_ratios = cpu_data['job_ratios']

        # 绘制线图
        ax.plot(cpu_ranges, job_ratios, linestyle='-', linewidth=2.5,
               marker='o', markersize=6, color='#1f77b4')

        # 设置标签和格式
        ax.set_xlabel('Number of CPUs')
        ax.set_ylabel('Percentage of Jobs (%)')
        ax.set_title('(a) CPU Job Distribution')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴为对数刻度（如果需要）
        ax.set_xscale('log')
        ax.set_xlim(1, 4096)
        ax.set_ylim(0, 100)

    def _plot_cpu_time_distribution_helios(self, ax, ctime_data: Dict[str, Any]):
        """绘制CPU时间分布 - 对应GPU分析风格"""
        if 'cpu_ranges' not in ctime_data or 'ctime_ratios' not in ctime_data:
            return

        cpu_ranges = ctime_data['cpu_ranges']
        ctime_ratios = ctime_data['ctime_ratios']

        # 绘制线图
        ax.plot(cpu_ranges, ctime_ratios, linestyle='-', linewidth=2.5,
               marker='s', markersize=6, color='#ff7f0e')

        # 设置标签和格式
        ax.set_xlabel('Number of CPUs')
        ax.set_ylabel('Percentage of CPU Time (%)')
        ax.set_title('(b) CPU Time Distribution')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴为对数刻度（如果需要）
        ax.set_xscale('log')
        ax.set_xlim(1, 4096)
        ax.set_ylim(0, 100)

    def _plot_job_status_helios(self, status_data: Dict[str, Any]):
        """绘制作业状态分布 - Helios风格"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)

        # 准备数据
        job_types = ['CPU Jobs', 'GPU Jobs']
        pass_rates = []
        fail_rates = []

        # CPU作业状态
        cpu_status = status_data.get('cpu_status_distribution', {})
        cpu_pass = cpu_status.get('Pass', 0)
        cpu_fail = cpu_status.get('Failed', 0) + cpu_status.get('Killed', 0)
        pass_rates.append(cpu_pass)
        fail_rates.append(cpu_fail)

        # GPU作业状态
        gpu_status = status_data.get('gpu_status_distribution', {})
        gpu_pass = gpu_status.get('Pass', 0)
        gpu_fail = gpu_status.get('Failed', 0) + gpu_status.get('Killed', 0)
        pass_rates.append(gpu_pass)
        fail_rates.append(gpu_fail)

        # 绘制堆叠条形图
        x = np.arange(len(job_types))
        width = 0.6

        ax.bar(x, pass_rates, width, label='Completed', color='#2ca02c', alpha=0.8)
        ax.bar(x, fail_rates, width, bottom=pass_rates, label='Failed', color='#d62728', alpha=0.8)

        # 设置标签和格式
        ax.set_xlabel('Job Type')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Job Status Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(job_types)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.3)

        # 保存图表
        output_path = self.output_paths['job_characterization'] / "job_status_distribution_helios.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]):
        """生成作业特征可视化图表 (Helios风格)"""
        logger.info("生成作业特征可视化图表...")

        # 1. 作业持续时间CDF图
        if 'duration_analysis' in results and 'duration' in df.columns:
            valid_duration = df[df['duration'] > 0]['duration'] / 3600  # 转换为小时
            
            if len(valid_duration) > 0:
                self.visualizer.plot_cdf(
                    valid_duration,
                    'Job Duration Distribution (CDF)',
                    'Duration (Hours)',
                    self.output_paths['figures'] / 'job_duration_cdf',
                    log_scale=True
                )
        
        # 2. 排队时间CDF图
        if 'queue_time_analysis' in results and 'queue_time' in df.columns:
            valid_queue_time = df[df['queue_time'] >= 0]['queue_time'] / 3600  # 转换为小时
            
            if len(valid_queue_time) > 0:
                self.visualizer.plot_cdf(
                    valid_queue_time,
                    'Queue Time Distribution (CDF)',
                    'Queue Time (Hours)',
                    self.output_paths['figures'] / 'queue_time_cdf',
                    log_scale=True
                )
        
        # 3. 作业状态分布条形图
        if 'job_status_analysis' in results:
            status_analysis = results['job_status_analysis']
            if 'status_distribution' in status_analysis:
                self.visualizer.plot_bar_chart(
                    status_analysis['status_distribution'],
                    'Job Status Distribution',
                    'Job Status', 'Number of Jobs',
                    self.output_paths['figures'] / 'job_status_distribution'
                )
        
        # 4. 节点数分布条形图
        if 'resource_demand_analysis' in results:
            resource_analysis = results['resource_demand_analysis']
            if 'node_distribution' in resource_analysis:
                # 只显示前20个最常见的节点数
                node_dist = resource_analysis['node_distribution']
                top_nodes = dict(sorted(node_dist.items(), key=lambda x: x[1], reverse=True)[:20])
                
                self.visualizer.plot_bar_chart(
                    {str(k): v for k, v in top_nodes.items()},
                    'Node Count Distribution (Top 20)',
                    'Number of Nodes', 'Number of Jobs',
                    self.output_paths['figures'] / 'node_count_distribution'
                )
        
        # 5. 作业规模分类饼图 (如果有相关数据)
        if 'job_size_analysis' in results:
            size_analysis = results['job_size_analysis']
            if 'job_size_categories' in size_analysis:
                categories = size_analysis['job_size_categories']
                self.visualizer.plot_bar_chart(
                    categories,
                    'Job Size Categories',
                    'Job Size Category', 'Number of Jobs',
                    self.output_paths['figures'] / 'job_size_categories'
                )
