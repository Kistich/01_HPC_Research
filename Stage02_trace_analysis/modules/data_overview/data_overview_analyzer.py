#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据概览分析模块
基于Helios项目的数据分析方法，提供数据集基本统计和质量评估
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.data_loader import add_gpu_job_flag

logger = logging.getLogger(__name__)


class DataOverviewAnalyzer:
    """数据概览分析器"""
    
    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path]):
        """
        初始化数据概览分析器
        
        Args:
            config: 分析配置
            output_paths: 输出路径字典
        """
        self.config = config
        self.output_paths = output_paths
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行数据概览分析
        
        Args:
            df: 作业数据DataFrame
            
        Returns:
            分析结果字典
        """
        logger.info("开始数据概览分析...")
        
        results = {
            'basic_statistics': self._get_basic_statistics(df),
            'data_quality': self._assess_data_quality(df),
            'time_coverage': self._analyze_time_coverage(df),
            'job_status_distribution': self._analyze_job_status(df),
            'resource_distribution': self._analyze_resource_distribution(df)
        }
        
        # 生成报告
        self._generate_report(results)
        
        logger.info("数据概览分析完成")
        return results
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """获取基本统计信息"""
        logger.info("计算基本统计信息...")
        
        stats = {
            'total_jobs': len(df),
            'total_users': df['user_id'].nunique() if 'user_id' in df.columns else 0,
            'total_clusters': df['cluster_name'].nunique() if 'cluster_name' in df.columns else 0,
            'date_range': {},
            'column_info': {}
        }
        
        # 时间范围
        if 'submit_time' in df.columns:
            submit_times = pd.to_datetime(df['submit_time']).dropna()
            if len(submit_times) > 0:
                stats['date_range'] = {
                    'start_date': submit_times.min().strftime('%Y-%m-%d'),
                    'end_date': submit_times.max().strftime('%Y-%m-%d'),
                    'duration_days': (submit_times.max() - submit_times.min()).days
                }
        
        # 列信息
        stats['column_info'] = {
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_data': df.isnull().sum().to_dict()
        }
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        logger.info("评估数据质量...")
        
        quality = {
            'completeness': {},
            'validity': {},
            'consistency': {}
        }
        
        # 完整性检查
        total_rows = len(df)
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            quality['completeness'][col] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / total_rows * 100)
            }
        
        # 有效性检查
        if 'duration' in df.columns:
            invalid_duration = (df['duration'] <= 0).sum()
            quality['validity']['invalid_duration'] = {
                'count': int(invalid_duration),
                'percentage': float(invalid_duration / total_rows * 100)
            }
        
        if 'queue_time' in df.columns:
            negative_queue_time = (df['queue_time'] < 0).sum()
            quality['validity']['negative_queue_time'] = {
                'count': int(negative_queue_time),
                'percentage': float(negative_queue_time / total_rows * 100)
            }
        
        # 一致性检查
        if 'start_time' in df.columns and 'end_time' in df.columns:
            inconsistent_times = (df['start_time'] > df['end_time']).sum()
            quality['consistency']['inconsistent_times'] = {
                'count': int(inconsistent_times),
                'percentage': float(inconsistent_times / total_rows * 100)
            }
        
        return quality
    
    def _analyze_time_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析时间覆盖情况"""
        logger.info("分析时间覆盖情况...")
        
        coverage = {}
        
        if 'submit_time' in df.columns:
            submit_times = pd.to_datetime(df['submit_time']).dropna()
            
            if len(submit_times) > 0:
                # 按月统计
                monthly_counts = submit_times.dt.to_period('M').value_counts().sort_index()
                coverage['monthly_distribution'] = {
                    str(period): int(count) for period, count in monthly_counts.items()
                }
                
                # 按星期几统计
                weekday_counts = submit_times.dt.dayofweek.value_counts().sort_index()
                weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                               'Friday', 'Saturday', 'Sunday']
                coverage['weekday_distribution'] = {
                    weekday_names[day]: int(count) for day, count in weekday_counts.items()
                }
                
                # 按小时统计
                hourly_counts = submit_times.dt.hour.value_counts().sort_index()
                coverage['hourly_distribution'] = {
                    int(hour): int(count) for hour, count in hourly_counts.items()
                }
        
        return coverage
    
    def _analyze_job_status(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析作业状态分布"""
        logger.info("分析作业状态分布...")
        
        status_analysis = {}
        
        if 'job_status_str' in df.columns:
            status_counts = df['job_status_str'].value_counts()
            status_analysis['status_distribution'] = status_counts.to_dict()
            
            # 计算成功率
            completed_statuses = ['COMPLETED', 'C']
            completed_count = sum(status_counts.get(status, 0) for status in completed_statuses)
            total_count = status_counts.sum()
            
            status_analysis['success_rate'] = {
                'completed_jobs': int(completed_count),
                'total_jobs': int(total_count),
                'success_percentage': float(completed_count / total_count * 100) if total_count > 0 else 0
            }
        
        return status_analysis
    
    def _analyze_resource_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析资源分布"""
        logger.info("分析资源分布...")
        
        resource_analysis = {}
        
        # 作业类型分布
        if 'job_type' in df.columns:
            job_type_counts = df['job_type'].value_counts()
            resource_analysis['job_type_distribution'] = job_type_counts.to_dict()
        
        # 子集群分布
        if 'primary_subcluster' in df.columns:
            subcluster_counts = df['primary_subcluster'].value_counts()
            resource_analysis['subcluster_distribution'] = subcluster_counts.to_dict()
        
        # 节点数分布
        if 'actual_node_count' in df.columns:
            node_count_stats = df['actual_node_count'].describe()
            resource_analysis['node_count_statistics'] = {
                'mean': float(node_count_stats['mean']),
                'median': float(node_count_stats['50%']),
                'max': int(node_count_stats['max']),
                'min': int(node_count_stats['min'])
            }
        
        # GPU作业分析 - 基于exec_hosts识别
        if 'is_gpu_job' not in df.columns:
            df = add_gpu_job_flag(df)

        gpu_jobs = df[df['is_gpu_job'] == True]
        resource_analysis['gpu_job_statistics'] = {
            'total_gpu_jobs': len(gpu_jobs),
            'gpu_job_percentage': len(gpu_jobs) / len(df) * 100 if len(df) > 0 else 0,
            'identification_method': 'exec_hosts_based'
        }

        # GPU数量分布 (如果gpu_num字段可用)
        if 'gpu_num' in df.columns and len(gpu_jobs) > 0:
            gpu_counts = gpu_jobs[gpu_jobs['gpu_num'].notna()]['gpu_num']
            if len(gpu_counts) > 0:
                gpu_stats = gpu_counts.describe()
                resource_analysis['gpu_count_statistics'] = {
                    'mean': float(gpu_stats['mean']),
                    'median': float(gpu_stats['50%']),
                    'max': int(gpu_stats['max']),
                    'min': int(gpu_stats['min']),
                    'jobs_with_gpu_count': len(gpu_counts)
            }
        
        return resource_analysis
    
    def _generate_report(self, results: Dict[str, Any]):
        """生成数据概览报告"""
        logger.info("生成数据概览报告...")
        
        report_path = self.output_paths['reports'] / 'data_overview_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HPC集群作业数据概览报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计
            basic = results['basic_statistics']
            f.write("基本统计信息:\n")
            f.write(f"  总作业数: {basic['total_jobs']:,}\n")
            f.write(f"  总用户数: {basic['total_users']:,}\n")
            f.write(f"  总集群数: {basic['total_clusters']}\n")
            
            if basic['date_range']:
                f.write(f"  时间范围: {basic['date_range']['start_date']} 到 {basic['date_range']['end_date']}\n")
                f.write(f"  持续天数: {basic['date_range']['duration_days']} 天\n")
            
            f.write(f"  数据列数: {basic['column_info']['total_columns']}\n\n")
            
            # 数据质量
            quality = results['data_quality']
            f.write("数据质量评估:\n")
            
            # 完整性
            f.write("  完整性 (缺失数据比例 > 5%):\n")
            for col, info in quality['completeness'].items():
                if info['missing_percentage'] > 5:
                    f.write(f"    {col}: {info['missing_percentage']:.1f}%\n")
            
            # 有效性
            if quality['validity']:
                f.write("  有效性问题:\n")
                for issue, info in quality['validity'].items():
                    f.write(f"    {issue}: {info['count']} ({info['percentage']:.1f}%)\n")
            
            f.write("\n")
            
            # 作业状态分布
            if 'job_status_distribution' in results:
                status = results['job_status_distribution']
                if 'success_rate' in status:
                    f.write("作业成功率:\n")
                    sr = status['success_rate']
                    f.write(f"  成功作业: {sr['completed_jobs']:,}\n")
                    f.write(f"  总作业数: {sr['total_jobs']:,}\n")
                    f.write(f"  成功率: {sr['success_percentage']:.1f}%\n\n")
            
            # 资源分布
            if 'resource_distribution' in results:
                resource = results['resource_distribution']
                
                if 'job_type_distribution' in resource:
                    f.write("作业类型分布:\n")
                    for job_type, count in resource['job_type_distribution'].items():
                        f.write(f"  {job_type}: {count:,}\n")
                    f.write("\n")
                
                if 'subcluster_distribution' in resource:
                    f.write("子集群使用分布:\n")
                    for subcluster, count in resource['subcluster_distribution'].items():
                        f.write(f"  {subcluster}: {count:,}\n")
        
        logger.info(f"数据概览报告已保存: {report_path}")
