#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
时间模式分析模块
基于Helios项目的时间分析方法，分析24小时和周期性负载模式
严格按照Helios的可视化风格
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """时间模式分析器"""
    
    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path], visualizer):
        """
        初始化时间模式分析器
        
        Args:
            config: 分析配置
            output_paths: 输出路径字典
            visualizer: 可视化器实例
        """
        self.config = config
        self.output_paths = output_paths
        self.visualizer = visualizer
        
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行时间模式分析
        
        Args:
            df: 作业数据DataFrame
            
        Returns:
            分析结果字典
        """
        logger.info("开始时间模式分析...")
        
        # 确保时间列存在且为datetime类型
        df = self._prepare_time_data(df)
        
        results = {
            'diurnal_patterns': self._analyze_diurnal_patterns(df),
            'weekly_patterns': self._analyze_weekly_patterns(df),
            'monthly_trends': self._analyze_monthly_trends(df),
            'job_type_temporal': self._analyze_job_type_temporal(df)
        }
        
        # 生成可视化图表
        self._generate_visualizations(df, results)
        
        logger.info("时间模式分析完成")
        return results
    
    def _prepare_time_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备时间数据"""
        df = df.copy()
        
        # 确保时间列为datetime类型
        time_columns = ['submit_time', 'start_time', 'end_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # 移除无效时间的记录
        if 'submit_time' in df.columns:
            df = df.dropna(subset=['submit_time'])
        
        return df
    
    def _analyze_diurnal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析24小时模式 (类似Helios的diurnal analysis)"""
        logger.info("分析24小时模式...")
        
        if 'submit_time' not in df.columns:
            return {}
        
        # 提取小时信息
        df_hourly = df.copy()
        df_hourly['hour'] = df_hourly['submit_time'].dt.hour
        
        # 按小时统计作业提交数
        hourly_submission = df_hourly.groupby('hour').size()
        
        # 按作业类型和小时统计
        job_type_hourly = {}
        if 'job_type' in df.columns:
            for job_type in df['job_type'].unique():
                if pd.notna(job_type):
                    type_data = df_hourly[df_hourly['job_type'] == job_type]
                    job_type_hourly[job_type] = type_data.groupby('hour').size()
        
        # 计算峰值和低谷
        peak_hour = hourly_submission.idxmax()
        valley_hour = hourly_submission.idxmin()
        peak_ratio = hourly_submission.max() / hourly_submission.mean()
        
        return {
            'hourly_submission_counts': hourly_submission.to_dict(),
            'job_type_hourly': {k: v.to_dict() for k, v in job_type_hourly.items()},
            'peak_hour': int(peak_hour),
            'valley_hour': int(valley_hour),
            'peak_to_average_ratio': float(peak_ratio),
            'total_daily_jobs': int(hourly_submission.sum())
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析周模式"""
        logger.info("分析周模式...")
        
        if 'submit_time' not in df.columns:
            return {}
        
        # 提取星期几信息
        df_weekly = df.copy()
        df_weekly['weekday'] = df_weekly['submit_time'].dt.dayofweek
        df_weekly['weekday_name'] = df_weekly['submit_time'].dt.day_name()
        
        # 按星期几统计
        weekly_submission = df_weekly.groupby('weekday_name').size()
        
        # 工作日vs周末
        df_weekly['is_weekend'] = df_weekly['weekday'].isin([5, 6])  # Saturday, Sunday
        weekend_vs_weekday = df_weekly.groupby('is_weekend').size()
        
        return {
            'weekly_submission_counts': weekly_submission.to_dict(),
            'weekend_vs_weekday': {
                'weekday': int(weekend_vs_weekday.get(False, 0)),
                'weekend': int(weekend_vs_weekday.get(True, 0))
            },
            'weekday_peak': weekly_submission.idxmax(),
            'weekday_valley': weekly_submission.idxmin()
        }
    
    def _analyze_monthly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析月度趋势"""
        logger.info("分析月度趋势...")
        
        if 'submit_time' not in df.columns:
            return {}
        
        # 按月统计
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly['submit_time'].dt.to_period('M')
        
        monthly_submission = df_monthly.groupby('month').size()
        
        # 计算增长趋势
        monthly_values = monthly_submission.values
        if len(monthly_values) > 1:
            growth_rates = []
            for i in range(1, len(monthly_values)):
                if monthly_values[i-1] > 0:
                    growth_rate = (monthly_values[i] - monthly_values[i-1]) / monthly_values[i-1] * 100
                    growth_rates.append(growth_rate)
            
            avg_growth_rate = np.mean(growth_rates) if growth_rates else 0
        else:
            avg_growth_rate = 0
        
        return {
            'monthly_submission_counts': {str(k): int(v) for k, v in monthly_submission.items()},
            'average_growth_rate': float(avg_growth_rate),
            'peak_month': str(monthly_submission.idxmax()) if len(monthly_submission) > 0 else None,
            'total_months': len(monthly_submission)
        }
    
    def _analyze_job_type_temporal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析不同作业类型的时间模式"""
        logger.info("分析作业类型时间模式...")
        
        if 'submit_time' not in df.columns or 'job_type' not in df.columns:
            return {}
        
        job_type_patterns = {}
        
        for job_type in df['job_type'].unique():
            if pd.notna(job_type):
                type_data = df[df['job_type'] == job_type].copy()
                type_data['hour'] = type_data['submit_time'].dt.hour
                
                hourly_pattern = type_data.groupby('hour').size()
                
                job_type_patterns[job_type] = {
                    'hourly_distribution': hourly_pattern.to_dict(),
                    'peak_hour': int(hourly_pattern.idxmax()) if len(hourly_pattern) > 0 else 0,
                    'total_jobs': len(type_data)
                }
        
        return job_type_patterns
    
    def _generate_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]):
        """生成时间模式可视化图表 (Helios风格)"""
        logger.info("生成时间模式可视化图表...")
        
        # 1. 24小时作业提交模式图
        if 'diurnal_patterns' in results and results['diurnal_patterns']:
            diurnal = results['diurnal_patterns']
            if 'hourly_submission_counts' in diurnal:
                hours = list(range(24))
                counts = [diurnal['hourly_submission_counts'].get(h, 0) for h in hours]
                
                hourly_df = pd.DataFrame({
                    'hour': hours,
                    'job_count': counts
                })
                
                # 绘制24小时模式图
                self.visualizer.plot_time_series(
                    hourly_df, 'hour', 'job_count',
                    'Diurnal Job Submission Pattern',
                    'Number of Jobs Submitted',
                    self.output_paths['figures'] / 'diurnal_pattern'
                )
        
        # 2. CPU vs GPU作业的24小时对比图 (双Y轴)
        if 'diurnal_patterns' in results and 'job_type_hourly' in results['diurnal_patterns']:
            job_type_hourly = results['diurnal_patterns']['job_type_hourly']
            
            if 'cpu' in job_type_hourly and 'gpu' in job_type_hourly:
                hours = list(range(24))
                cpu_counts = [job_type_hourly['cpu'].get(h, 0) for h in hours]
                gpu_counts = [job_type_hourly['gpu'].get(h, 0) for h in hours]
                
                comparison_df = pd.DataFrame({
                    'hour': hours,
                    'cpu_jobs': cpu_counts,
                    'gpu_jobs': gpu_counts
                })
                
                self.visualizer.plot_dual_axis(
                    comparison_df, 'hour', 'cpu_jobs', 'gpu_jobs',
                    'CPU vs GPU Job Submission Pattern',
                    'CPU Jobs', 'GPU Jobs',
                    self.output_paths['figures'] / 'cpu_gpu_diurnal_comparison'
                )
        
        # 3. 周模式条形图
        if 'weekly_patterns' in results and results['weekly_patterns']:
            weekly = results['weekly_patterns']
            if 'weekly_submission_counts' in weekly:
                self.visualizer.plot_bar_chart(
                    weekly['weekly_submission_counts'],
                    'Weekly Job Submission Pattern',
                    'Day of Week', 'Number of Jobs',
                    self.output_paths['figures'] / 'weekly_pattern',
                    sort_values=False
                )
        
        # 4. 月度趋势图
        if 'monthly_trends' in results and results['monthly_trends']:
            monthly = results['monthly_trends']
            if 'monthly_submission_counts' in monthly:
                months = list(monthly['monthly_submission_counts'].keys())
                counts = list(monthly['monthly_submission_counts'].values())
                
                monthly_df = pd.DataFrame({
                    'month': pd.to_datetime(months),
                    'job_count': counts
                })
                
                self.visualizer.plot_time_series(
                    monthly_df, 'month', 'job_count',
                    'Monthly Job Submission Trend',
                    'Number of Jobs',
                    self.output_paths['figures'] / 'monthly_trend'
                )
        
        # 5. 作业提交热力图 (小时 x 星期几)
        if 'submit_time' in df.columns:
            self.visualizer.plot_hourly_heatmap(
                df, 'submit_time',
                'Job Submission Heatmap (Hour vs Day of Week)',
                self.output_paths['figures'] / 'submission_heatmap'
            )
