#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
资源利用率分析模块
基于Helios项目的资源分析方法
分析CPU/GPU利用率、负载均衡、容量规划等
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ResourceUtilizationAnalyzer:
    """资源利用率分析器"""
    
    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path], visualizer, cluster_config: Dict[str, Any]):
        """
        初始化资源利用率分析器
        
        Args:
            config: 分析配置
            output_paths: 输出路径字典
            visualizer: 可视化器实例
            cluster_config: 集群配置信息
        """
        self.config = config
        self.output_paths = output_paths
        self.visualizer = visualizer
        self.cluster_config = cluster_config
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行资源利用率分析
        
        Args:
            df: 作业数据DataFrame
            
        Returns:
            分析结果字典
        """
        logger.info("开始资源利用率分析...")
        
        results = {
            'overall_utilization': self._analyze_overall_utilization(df),
            'subcluster_utilization': self._analyze_subcluster_utilization(df),
            'temporal_utilization': self._analyze_temporal_utilization(df),
            'load_balancing': self._analyze_load_balancing(df),
            'capacity_analysis': self._analyze_capacity_requirements(df)
        }
        
        # 生成可视化图表
        self._generate_visualizations(df, results)
        
        logger.info("资源利用率分析完成")
        return results
    
    def _analyze_overall_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析整体资源利用率"""
        logger.info("分析整体资源利用率...")
        
        if 'duration' not in df.columns or 'actual_node_count' not in df.columns:
            return {}
        
        # 计算总资源消耗
        df_valid = df[(df['duration'] > 0) & (df['actual_node_count'] > 0)]
        df_valid['node_hours'] = df_valid['duration'] * df_valid['actual_node_count'] / 3600
        
        total_node_hours = df_valid['node_hours'].sum()
        
        # 按作业类型分析
        utilization_by_type = {}
        if 'job_type' in df.columns:
            for job_type in df_valid['job_type'].unique():
                if pd.notna(job_type):
                    type_data = df_valid[df_valid['job_type'] == job_type]
                    type_node_hours = type_data['node_hours'].sum()
                    
                    utilization_by_type[job_type] = {
                        'node_hours': float(type_node_hours),
                        'percentage': float(type_node_hours / total_node_hours * 100) if total_node_hours > 0 else 0,
                        'job_count': len(type_data),
                        'avg_duration_hours': float(type_data['duration'].mean() / 3600)
                    }
        
        # 计算理论容量利用率 (基于集群配置)
        theoretical_capacity = self._calculate_theoretical_capacity(df)
        
        return {
            'total_node_hours_consumed': float(total_node_hours),
            'total_jobs_analyzed': len(df_valid),
            'utilization_by_job_type': utilization_by_type,
            'theoretical_capacity': theoretical_capacity,
            'average_job_duration_hours': float(df_valid['duration'].mean() / 3600),
            'average_nodes_per_job': float(df_valid['actual_node_count'].mean())
        }
    
    def _analyze_subcluster_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析子集群资源利用率"""
        logger.info("分析子集群资源利用率...")
        
        if 'primary_subcluster' not in df.columns:
            return {}
        
        subcluster_stats = {}
        subclusters_config = self.cluster_config.get('subclusters', {})
        
        for subcluster in df['primary_subcluster'].unique():
            if pd.notna(subcluster) and subcluster in subclusters_config:
                subcluster_data = df[df['primary_subcluster'] == subcluster]
                subcluster_config = subclusters_config[subcluster]
                
                # 计算该子集群的资源消耗
                if 'duration' in df.columns and 'actual_node_count' in df.columns:
                    valid_data = subcluster_data[(subcluster_data['duration'] > 0) & 
                                               (subcluster_data['actual_node_count'] > 0)]
                    
                    if len(valid_data) > 0:
                        node_hours = (valid_data['duration'] * valid_data['actual_node_count'] / 3600).sum()
                        
                        # 理论容量 (假设分析期间内所有节点100%利用)
                        node_count = subcluster_config.get('node_count', 0)
                        analysis_days = self._get_analysis_period_days(df)
                        theoretical_node_hours = node_count * 24 * analysis_days
                        
                        utilization_rate = (node_hours / theoretical_node_hours * 100) if theoretical_node_hours > 0 else 0
                        
                        subcluster_stats[subcluster] = {
                            'total_jobs': len(subcluster_data),
                            'node_hours_consumed': float(node_hours),
                            'theoretical_capacity_hours': float(theoretical_node_hours),
                            'utilization_rate_percentage': float(utilization_rate),
                            'average_job_duration_hours': float(valid_data['duration'].mean() / 3600),
                            'node_count': node_count,
                            'node_type': subcluster_config.get('node_type', 'unknown')
                        }
        
        return subcluster_stats
    
    def _analyze_temporal_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析时间维度的资源利用率"""
        logger.info("分析时间维度资源利用率...")
        
        if 'submit_time' not in df.columns:
            return {}
        
        # 按小时分析利用率
        df_time = df.copy()
        df_time['submit_hour'] = pd.to_datetime(df_time['submit_time']).dt.hour
        df_time['submit_date'] = pd.to_datetime(df_time['submit_time']).dt.date
        
        # 每小时作业提交数
        hourly_jobs = df_time.groupby('submit_hour').size()
        
        # 每小时资源需求
        hourly_resources = {}
        if 'actual_node_count' in df.columns:
            hourly_node_demand = df_time.groupby('submit_hour')['actual_node_count'].sum()
            hourly_resources['node_demand'] = hourly_node_demand.to_dict()
        
        # 每日利用率趋势
        daily_stats = {}
        if 'duration' in df.columns and 'actual_node_count' in df.columns:
            df_valid = df_time[(df_time['duration'] > 0) & (df_time['actual_node_count'] > 0)]
            df_valid['node_hours'] = df_valid['duration'] * df_valid['actual_node_count'] / 3600
            
            daily_consumption = df_valid.groupby('submit_date')['node_hours'].sum()
            daily_stats = {
                'daily_consumption': {str(date): float(hours) for date, hours in daily_consumption.items()},
                'peak_day': str(daily_consumption.idxmax()) if len(daily_consumption) > 0 else None,
                'peak_consumption': float(daily_consumption.max()) if len(daily_consumption) > 0 else 0
            }
        
        return {
            'hourly_job_submission': hourly_jobs.to_dict(),
            'hourly_resource_demand': hourly_resources,
            'daily_utilization_trends': daily_stats,
            'peak_hour': int(hourly_jobs.idxmax()) if len(hourly_jobs) > 0 else 0,
            'valley_hour': int(hourly_jobs.idxmin()) if len(hourly_jobs) > 0 else 0
        }
    
    def _analyze_load_balancing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析负载均衡情况"""
        logger.info("分析负载均衡...")
        
        if 'primary_subcluster' not in df.columns:
            return {}
        
        # 计算各子集群的负载分布
        subcluster_loads = df['primary_subcluster'].value_counts()
        total_jobs = len(df)
        
        # 计算负载均衡指标
        load_percentages = subcluster_loads / total_jobs * 100
        load_variance = load_percentages.var()
        load_std = load_percentages.std()
        
        # 理想情况下的均匀分布
        num_subclusters = len(subcluster_loads)
        ideal_percentage = 100 / num_subclusters if num_subclusters > 0 else 0
        
        # 计算负载不平衡程度
        imbalance_score = load_std / ideal_percentage if ideal_percentage > 0 else 0
        
        return {
            'subcluster_job_distribution': subcluster_loads.to_dict(),
            'load_percentages': load_percentages.to_dict(),
            'load_balance_metrics': {
                'load_variance': float(load_variance),
                'load_standard_deviation': float(load_std),
                'ideal_percentage_per_subcluster': float(ideal_percentage),
                'imbalance_score': float(imbalance_score),
                'interpretation': 'Lower imbalance score indicates better load balancing'
            },
            'most_loaded_subcluster': subcluster_loads.idxmax() if len(subcluster_loads) > 0 else None,
            'least_loaded_subcluster': subcluster_loads.idxmin() if len(subcluster_loads) > 0 else None
        }
    
    def _analyze_capacity_requirements(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析容量需求和规划建议"""
        logger.info("分析容量需求...")
        
        capacity_analysis = {}
        
        # 峰值需求分析
        if 'submit_time' in df.columns and 'actual_node_count' in df.columns:
            df_time = df.copy()
            df_time['submit_hour'] = pd.to_datetime(df_time['submit_time']).dt.hour
            
            # 每小时峰值节点需求
            hourly_node_demand = df_time.groupby('submit_hour')['actual_node_count'].sum()
            peak_hour_demand = hourly_node_demand.max()
            peak_hour = hourly_node_demand.idxmax()
            
            # 平均需求
            avg_hourly_demand = hourly_node_demand.mean()
            
            capacity_analysis['demand_analysis'] = {
                'peak_hourly_node_demand': int(peak_hour_demand),
                'peak_demand_hour': int(peak_hour),
                'average_hourly_demand': float(avg_hourly_demand),
                'peak_to_average_ratio': float(peak_hour_demand / avg_hourly_demand) if avg_hourly_demand > 0 else 0
            }
        
        # 按子集群的容量建议
        subcluster_recommendations = {}
        if 'primary_subcluster' in df.columns:
            subclusters_config = self.cluster_config.get('subclusters', {})
            
            for subcluster in df['primary_subcluster'].unique():
                if pd.notna(subcluster) and subcluster in subclusters_config:
                    subcluster_data = df[df['primary_subcluster'] == subcluster]
                    current_capacity = subclusters_config[subcluster].get('node_count', 0)
                    
                    if 'actual_node_count' in df.columns:
                        max_concurrent_demand = subcluster_data['actual_node_count'].max()
                        avg_demand = subcluster_data['actual_node_count'].mean()
                        
                        # 容量利用率
                        utilization_rate = (max_concurrent_demand / current_capacity * 100) if current_capacity > 0 else 0
                        
                        # 建议
                        if utilization_rate > 90:
                            recommendation = "Consider expanding capacity"
                        elif utilization_rate < 30:
                            recommendation = "Capacity may be over-provisioned"
                        else:
                            recommendation = "Capacity appears adequate"
                        
                        subcluster_recommendations[subcluster] = {
                            'current_capacity': current_capacity,
                            'max_demand_observed': int(max_concurrent_demand),
                            'average_demand': float(avg_demand),
                            'peak_utilization_percentage': float(utilization_rate),
                            'recommendation': recommendation
                        }
        
        capacity_analysis['subcluster_recommendations'] = subcluster_recommendations
        
        return capacity_analysis
    
    def _calculate_theoretical_capacity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算理论容量"""
        subclusters_config = self.cluster_config.get('subclusters', {})
        
        total_cpu_nodes = sum(
            config['node_count'] for config in subclusters_config.values()
            if config.get('node_type') in ['cpu', 'bigmem']
        )
        
        total_gpu_nodes = sum(
            config['node_count'] for config in subclusters_config.values()
            if config.get('node_type') == 'gpu'
        )
        
        analysis_days = self._get_analysis_period_days(df)
        
        return {
            'total_cpu_nodes': total_cpu_nodes,
            'total_gpu_nodes': total_gpu_nodes,
            'analysis_period_days': analysis_days,
            'theoretical_cpu_node_hours': total_cpu_nodes * 24 * analysis_days,
            'theoretical_gpu_node_hours': total_gpu_nodes * 24 * analysis_days
        }
    
    def _get_analysis_period_days(self, df: pd.DataFrame) -> int:
        """获取分析期间的天数"""
        if 'submit_time' not in df.columns:
            return 30  # 默认30天
        
        submit_times = pd.to_datetime(df['submit_time']).dropna()
        if len(submit_times) == 0:
            return 30
        
        period = (submit_times.max() - submit_times.min()).days
        return max(1, period)
    
    def _generate_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]):
        """生成资源利用率可视化图表"""
        logger.info("生成资源利用率可视化图表...")
        
        # 1. 子集群利用率对比图
        if 'subcluster_utilization' in results:
            subcluster_util = results['subcluster_utilization']
            utilization_rates = {subcluster: stats['utilization_rate_percentage'] 
                               for subcluster, stats in subcluster_util.items()}
            
            if utilization_rates:
                self.visualizer.plot_bar_chart(
                    utilization_rates,
                    'Subcluster Utilization Rates',
                    'Subcluster', 'Utilization Rate (%)',
                    self.output_paths['figures'] / 'subcluster_utilization_rates'
                )
        
        # 2. 24小时资源需求图
        if 'temporal_utilization' in results:
            temporal = results['temporal_utilization']
            if 'hourly_job_submission' in temporal:
                hours = list(range(24))
                job_counts = [temporal['hourly_job_submission'].get(h, 0) for h in hours]
                
                hourly_df = pd.DataFrame({
                    'hour': hours,
                    'job_submissions': job_counts
                })
                
                self.visualizer.plot_time_series(
                    hourly_df, 'hour', 'job_submissions',
                    'Hourly Resource Demand Pattern',
                    'Number of Job Submissions',
                    self.output_paths['figures'] / 'hourly_resource_demand'
                )
        
        # 3. 负载均衡分析图
        if 'load_balancing' in results:
            load_balance = results['load_balancing']
            if 'subcluster_job_distribution' in load_balance:
                self.visualizer.plot_bar_chart(
                    load_balance['subcluster_job_distribution'],
                    'Load Distribution Across Subclusters',
                    'Subcluster', 'Number of Jobs',
                    self.output_paths['figures'] / 'load_distribution'
                )
        
        # 4. 容量利用率vs需求对比图
        if 'capacity_analysis' in results and 'subcluster_recommendations' in results['capacity_analysis']:
            recommendations = results['capacity_analysis']['subcluster_recommendations']
            
            subclusters = list(recommendations.keys())
            capacities = [recommendations[sc]['current_capacity'] for sc in subclusters]
            max_demands = [recommendations[sc]['max_demand_observed'] for sc in subclusters]
            
            if subclusters:
                comparison_df = pd.DataFrame({
                    'subcluster': subclusters,
                    'current_capacity': capacities,
                    'max_demand': max_demands
                })
                
                # 使用双Y轴图表显示容量vs需求
                self.visualizer.plot_dual_axis(
                    comparison_df, 'subcluster', 'current_capacity', 'max_demand',
                    'Capacity vs Demand Comparison',
                    'Current Capacity', 'Max Observed Demand',
                    self.output_paths['figures'] / 'capacity_vs_demand'
                )
