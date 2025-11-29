#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
集群特征分析模块
严格按照Helios项目的cluster characterization方法
分析集群利用率、资源分布和负载模式
对应Helios的2_cluster characterization模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ClusterCharacterizationAnalyzer:
    """集群特征分析器 - 严格按照Helios方法"""

    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path], visualizer):
        """
        初始化集群特征分析器

        Args:
            config: 分析配置
            output_paths: 输出路径字典
            visualizer: 可视化器实例
        """
        self.config = config
        self.output_paths = output_paths
        self.visualizer = visualizer

        # 设置Helios风格的绘图参数
        self._setup_helios_style()

        # CPU配置信息（从cluster_config.yaml获取）
        self.cpu_configs = {
            'CPU1': 64,   # 2 * Intel Xeon Platinum 8358P (32C) = 64 cores/node
            'CPU2': 64,   # 2 * Intel Xeon Platinum 8358P (32C) = 64 cores/node
            'CPU3': 128,  # 2 * AMD EPYC 7763 (64C) = 128 cores/node
            'BIGMEM': 96  # 4 * Intel Xeon Gold 6348H (24C) = 96 cores/node
        }

    def _setup_helios_style(self):
        """设置Helios风格的绘图参数"""
        # 设置Helios风格
        sns.set_style("ticks")
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.size': 12,
            'lines.linewidth': 3,
            'lines.markersize': 10,
        })
        sns.set_context("paper", font_scale=1.6)
        self.current_palette = sns.color_palette()

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

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行集群特征分析 - 严格按照Helios方法

        Args:
            data: 预处理后的数据（包含Helios兼容格式）

        Returns:
            分析结果字典
        """
        logger.info("开始集群特征分析（使用Helios兼容数据）...")

        # 获取Helios兼容数据
        helios_data = data.get('helios_data', {})
        if not helios_data:
            raise ValueError("缺少Helios兼容数据")

        # 使用cluster_sequence和cluster_throughput进行分析
        results = {
            # 1. 集群利用率的24小时模式分析（对应Helios Figure 3）
            'hourly_utilization': self._analyze_hourly_utilization_helios(helios_data),

            # 2. 集群吞吐量分析
            'cluster_throughput': self._analyze_cluster_throughput_helios(helios_data)
        }

        # 生成Helios风格的可视化图表（Figure 3复现）
        self._generate_helios_visualizations(helios_data, results)

        logger.info("集群特征分析完成")
        return results

    def _analyze_hourly_utilization_helios(self, helios_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析集群利用率的24小时模式 - 严格按照Helios方法"""
        logger.info("分析集群利用率的24小时模式（Helios风格）...")

        if 'cluster_sequence' not in helios_data:
            logger.warning("缺少cluster_sequence数据")
            return {}

        cluster_sequence = helios_data['cluster_sequence'].copy()

        # 确保时间字段为datetime类型
        cluster_sequence['time'] = pd.to_datetime(cluster_sequence['time'])
        cluster_sequence['hour'] = cluster_sequence['time'].dt.hour

        # 按小时聚合数据（对应Helios的24小时模式分析）
        hourly_stats = cluster_sequence.groupby('hour').agg({
            'job_count': 'mean',
            'gpu_usage': 'mean',
            'cpu_usage': 'mean'
        }).reset_index()

        # 重命名列以匹配期望的字段名
        hourly_stats = hourly_stats.rename(columns={
            'job_count': 'submitted_jobs',
            'gpu_usage': 'total_gpu_requested',
            'cpu_usage': 'total_cpu_requested'
        })

        return {
            'hourly_stats': hourly_stats,
            'peak_hour': hourly_stats.loc[hourly_stats['submitted_jobs'].idxmax(), 'hour'],
            'avg_jobs_per_hour': hourly_stats['submitted_jobs'].mean()
        }

    def _analyze_cluster_throughput_helios(self, helios_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析集群吞吐量 - 严格按照Helios方法"""
        logger.info("分析集群吞吐量（Helios风格）...")

        if 'cluster_throughput' not in helios_data:
            logger.warning("缺少cluster_throughput数据")
            return {}

        cluster_throughput = helios_data['cluster_throughput'].copy()

        # 确保时间字段为datetime类型
        cluster_throughput['time'] = pd.to_datetime(cluster_throughput['time'])
        cluster_throughput['hour'] = cluster_throughput['time'].dt.hour

        # 按小时聚合吞吐量数据
        hourly_throughput = cluster_throughput.groupby('hour').agg({
            'jobs_started': 'mean',
            'gpu_hours': 'mean',
            'cpu_hours': 'mean'
        }).reset_index()

        # 重命名列以匹配期望的字段名
        hourly_throughput = hourly_throughput.rename(columns={
            'gpu_hours': 'gpu_hours_consumed',
            'cpu_hours': 'cpu_hours_consumed'
        })

        return {
            'hourly_throughput': hourly_throughput,
            'peak_throughput_hour': hourly_throughput.loc[hourly_throughput['jobs_started'].idxmax(), 'hour'],
            'avg_throughput_per_hour': hourly_throughput['jobs_started'].mean()
        }

    def _analyze_cluster_utilization_timeseries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析集群利用率时间序列 - 按照Helios方法"""
        logger.info("分析集群利用率时间序列...")

        # 创建时间序列数据
        if 'submit_time' not in df.columns:
            return {}

        # 按小时聚合作业提交数据
        df_ts = df.copy()
        df_ts['submit_time'] = pd.to_datetime(df_ts['submit_time'])
        df_ts['hour'] = df_ts['submit_time'].dt.floor('H')

        # 计算每小时的作业数量和资源使用
        hourly_stats = df_ts.groupby('hour').agg({
            'job_id': 'count',  # 作业数量
            'cpu_num': 'sum',   # CPU使用量
            'gpu_num': 'sum',   # GPU使用量
            'duration': 'mean'  # 平均持续时间
        }).rename(columns={'job_id': 'job_count'})

        # 计算利用率统计
        utilization_stats = {
            'total_jobs': len(df),
            'avg_jobs_per_hour': float(hourly_stats['job_count'].mean()),
            'peak_jobs_per_hour': int(hourly_stats['job_count'].max()),
            'avg_cpu_usage': float(hourly_stats['cpu_num'].mean()),
            'avg_gpu_usage': float(hourly_stats['gpu_num'].mean()),
            'avg_duration_hours': float(hourly_stats['duration'].mean() / 3600) if 'duration' in df.columns else 0
        }

        return {
            'hourly_stats': hourly_stats,
            'utilization_stats': utilization_stats,
            'time_range': {
                'start': df_ts['submit_time'].min(),
                'end': df_ts['submit_time'].max()
            }
        }

    def _analyze_resource_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析资源分布 - 按照Helios方法"""
        logger.info("分析资源分布...")

        resource_stats = {}

        # CPU资源分布
        if 'cpu_num' in df.columns:
            cpu_data = df[df['cpu_num'] > 0]['cpu_num']
            resource_stats['cpu'] = {
                'mean': float(cpu_data.mean()),
                'median': float(cpu_data.median()),
                'std': float(cpu_data.std()),
                'min': int(cpu_data.min()),
                'max': int(cpu_data.max()),
                'percentiles': {
                    '25': float(cpu_data.quantile(0.25)),
                    '75': float(cpu_data.quantile(0.75)),
                    '90': float(cpu_data.quantile(0.90)),
                    '95': float(cpu_data.quantile(0.95)),
                    '99': float(cpu_data.quantile(0.99))
                }
            }

        # GPU资源分布
        if 'gpu_num' in df.columns:
            gpu_data = df[df['gpu_num'] > 0]['gpu_num']
            if len(gpu_data) > 0:
                resource_stats['gpu'] = {
                    'mean': float(gpu_data.mean()),
                    'median': float(gpu_data.median()),
                    'std': float(gpu_data.std()),
                    'min': int(gpu_data.min()),
                    'max': int(gpu_data.max()),
                    'percentiles': {
                        '25': float(gpu_data.quantile(0.25)),
                        '75': float(gpu_data.quantile(0.75)),
                        '90': float(gpu_data.quantile(0.90)),
                        '95': float(gpu_data.quantile(0.95)),
                        '99': float(gpu_data.quantile(0.99))
                    }
                }

        # 作业持续时间分布
        if 'duration' in df.columns:
            duration_hours = df[df['duration'] > 0]['duration'] / 3600
            resource_stats['duration'] = {
                'mean_hours': float(duration_hours.mean()),
                'median_hours': float(duration_hours.median()),
                'std_hours': float(duration_hours.std()),
                'percentiles': {
                    '25': float(duration_hours.quantile(0.25)),
                    '75': float(duration_hours.quantile(0.75)),
                    '90': float(duration_hours.quantile(0.90)),
                    '95': float(duration_hours.quantile(0.95)),
                    '99': float(duration_hours.quantile(0.99))
                }
            }

        return resource_stats

    def _analyze_load_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析负载模式 - 按照Helios方法"""
        logger.info("分析负载模式...")

        if 'submit_time' not in df.columns:
            return {}

        df_pattern = df.copy()
        df_pattern['submit_time'] = pd.to_datetime(df_pattern['submit_time'])
        df_pattern['hour'] = df_pattern['submit_time'].dt.hour
        df_pattern['day_of_week'] = df_pattern['submit_time'].dt.dayofweek

        # 24小时模式
        hourly_pattern = df_pattern.groupby('hour').size()

        # 一周模式
        weekly_pattern = df_pattern.groupby('day_of_week').size()

        # 计算峰值和低谷
        peak_hour = hourly_pattern.idxmax()
        valley_hour = hourly_pattern.idxmin()
        peak_ratio = hourly_pattern.max() / hourly_pattern.mean() if hourly_pattern.mean() > 0 else 0

        return {
            'hourly_pattern': hourly_pattern.to_dict(),
            'weekly_pattern': weekly_pattern.to_dict(),
            'peak_hour': int(peak_hour),
            'valley_hour': int(valley_hour),
            'peak_ratio': float(peak_ratio),
            'pattern_stats': {
                'hourly_std': float(hourly_pattern.std()),
                'weekly_std': float(weekly_pattern.std())
            }
        }


    
    def _generate_helios_visualizations(self, helios_data: Dict[str, Any], results: Dict[str, Any]):
        """生成Helios风格的集群特征可视化图表 - 严格按照Helios Figure 3"""
        logger.info("生成Helios风格的集群特征可视化图表...")

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

        # 创建双子图布局（对应Helios Figure 3）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # 1. 集群利用率的24小时模式（左图）
        if 'hourly_utilization' in results:
            self._plot_hourly_utilization_helios(ax1, results['hourly_utilization'])

        # 2. 集群吞吐量的24小时模式（右图）
        if 'cluster_throughput' in results:
            self._plot_cluster_throughput_helios(ax2, results['cluster_throughput'])

        # 保存图表
        output_path = self.output_paths['cluster_characterization'] / "cluster_characterization_helios.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Helios风格集群特征图表已保存: {output_path}")

    def _plot_hourly_utilization_helios(self, ax, hourly_data: Dict[str, Any]):
        """绘制集群利用率的24小时模式 - Helios风格"""
        if 'hourly_stats' not in hourly_data:
            return

        hourly_stats = hourly_data['hourly_stats']

        # 绘制作业提交率和GPU使用率
        ax.plot(hourly_stats['hour'], hourly_stats['submitted_jobs'],
               linestyle='-', linewidth=2.5, label='Job Submission Rate', color='#1f77b4')

        # 添加第二个y轴用于GPU请求
        ax2 = ax.twinx()
        ax2.plot(hourly_stats['hour'], hourly_stats['total_gpu_requested'],
                linestyle='--', linewidth=2.5, label='GPU Request Rate', color='#ff7f0e')

        # 设置标签和格式
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Job Submission Rate')
        ax2.set_ylabel('GPU Request Rate')
        ax.set_title('(a) Cluster Utilization Pattern')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴范围
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))

        # 合并图例 - 移到右下角，缩小字体
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    def _plot_cluster_throughput_helios(self, ax, throughput_data: Dict[str, Any]):
        """绘制集群吞吐量的24小时模式 - Helios风格"""
        if 'hourly_throughput' not in throughput_data:
            return

        hourly_throughput = throughput_data['hourly_throughput']

        # 绘制作业开始率和资源消耗
        ax.plot(hourly_throughput['hour'], hourly_throughput['jobs_started'],
               linestyle='-', linewidth=2.5, label='Jobs Started', color='#2ca02c')

        # 添加第二个y轴用于资源消耗
        ax2 = ax.twinx()
        ax2.plot(hourly_throughput['hour'], hourly_throughput['gpu_hours_consumed'],
                linestyle='--', linewidth=2.5, label='GPU Hours', color='#d62728')

        # 设置标签和格式
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Jobs Started per Hour')
        ax2.set_ylabel('GPU Hours Consumed')
        ax.set_title('(b) Cluster Throughput Pattern')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴范围
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 4))

        # 合并图例 - 移到右下角，缩小字体
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=9)

    def _plot_cluster_utilization_timeseries(self, utilization_data: Dict[str, Any]):
        """绘制集群利用率时间序列图 - Helios风格"""
        if 'hourly_stats' not in utilization_data:
            return

        hourly_stats = utilization_data['hourly_stats']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 上图：作业提交数量时间序列
        ax1.plot(hourly_stats.index, hourly_stats['job_count'],
                linewidth=3, color=self.current_palette[0])
        ax1.set_ylabel('Jobs Submitted per Hour', fontsize=14)
        ax1.set_title('Cluster Utilization Over Time', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # 下图：资源使用时间序列
        ax2.plot(hourly_stats.index, hourly_stats['cpu_num'],
                linewidth=3, color=self.current_palette[1], label='CPU')
        if 'gpu_num' in hourly_stats.columns and hourly_stats['gpu_num'].sum() > 0:
            ax2.plot(hourly_stats.index, hourly_stats['gpu_num'],
                    linewidth=3, color=self.current_palette[2], label='GPU')
            ax2.legend()

        ax2.set_ylabel('Resource Usage', fontsize=14)
        ax2.set_xlabel('Time', fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_helios_figure(fig, 'cluster_utilization_timeseries')
        plt.close()

    def _plot_resource_distribution_cdf(self, df: pd.DataFrame, resource_data: Dict[str, Any]):
        """绘制资源分布CDF图 - Helios风格"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # CPU分布CDF
        if 'cpu' in resource_data and 'cpu_num' in df.columns:
            cpu_data = df[df['cpu_num'] > 0]['cpu_num']
            if len(cpu_data) > 0:
                sorted_data = np.sort(cpu_data)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                axes[0].plot(sorted_data, y, linewidth=3, color=self.current_palette[0])
                axes[0].set_xlabel('CPU Count', fontsize=14)
                axes[0].set_ylabel('CDF', fontsize=14)
                axes[0].set_title('CPU Resource Distribution', fontsize=14, fontweight='bold')
                axes[0].grid(True, alpha=0.3)
                axes[0].set_xscale('log')

        # GPU分布CDF
        if 'gpu' in resource_data and 'gpu_num' in df.columns:
            gpu_data = df[df['gpu_num'] > 0]['gpu_num']
            if len(gpu_data) > 0:
                sorted_data = np.sort(gpu_data)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                axes[1].plot(sorted_data, y, linewidth=3, color=self.current_palette[1])
                axes[1].set_xlabel('GPU Count', fontsize=14)
                axes[1].set_ylabel('CDF', fontsize=14)
                axes[1].set_title('GPU Resource Distribution', fontsize=14, fontweight='bold')
                axes[1].grid(True, alpha=0.3)
                axes[1].set_xscale('log')

        # 持续时间分布CDF
        if 'duration' in resource_data and 'duration' in df.columns:
            duration_hours = df[df['duration'] > 0]['duration'] / 3600
            if len(duration_hours) > 0:
                sorted_data = np.sort(duration_hours)
                y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                axes[2].plot(sorted_data, y, linewidth=3, color=self.current_palette[2])
                axes[2].set_xlabel('Duration (Hours)', fontsize=14)
                axes[2].set_ylabel('CDF', fontsize=14)
                axes[2].set_title('Job Duration Distribution', fontsize=14, fontweight='bold')
                axes[2].grid(True, alpha=0.3)
                axes[2].set_xscale('log')

        plt.tight_layout()
        self._save_helios_figure(fig, 'resource_distribution_cdf')
        plt.close()

    def _plot_load_patterns(self, load_data: Dict[str, Any]):
        """绘制负载模式图 - Helios风格"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 24小时模式
        if 'hourly_pattern' in load_data:
            hours = list(range(24))
            counts = [load_data['hourly_pattern'].get(h, 0) for h in hours]
            ax1.plot(hours, counts, linewidth=3, color=self.current_palette[0], marker='o', markersize=6)
            ax1.set_xlabel('Hour of Day', fontsize=14)
            ax1.set_ylabel('Number of Jobs', fontsize=14)
            ax1.set_title('Diurnal Pattern', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(0, 24, 4))

        # 一周模式
        if 'weekly_pattern' in load_data:
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            counts = [load_data['weekly_pattern'].get(i, 0) for i in range(7)]
            ax2.bar(days, counts, color=self.current_palette[1], alpha=0.8)
            ax2.set_xlabel('Day of Week', fontsize=14)
            ax2.set_ylabel('Number of Jobs', fontsize=14)
            ax2.set_title('Weekly Pattern', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        self._save_helios_figure(fig, 'load_patterns')
        plt.close()

    def _save_helios_figure(self, fig, filename: str):
        """保存Helios风格的图表"""
        output_path = self.output_paths['cluster_characterization'] / f'{filename}.pdf'
        fig.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        logger.info(f"保存图表: {output_path}")
