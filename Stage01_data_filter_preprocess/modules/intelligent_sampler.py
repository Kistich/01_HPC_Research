#!/usr/bin/env python3
"""
智能采样器
基于异常检测和分层策略进行智能采样
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from parallel_processor import ParallelProcessor
from progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class IntelligentSampler:
    """智能采样器"""
    
    def __init__(self, config_path: str):
        """
        初始化智能采样器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.processor = ParallelProcessor(
            max_cores=self.config['parallel_processing']['max_cores']
        )
        
        # 统计信息
        self.stats = {
            'total_jobs': 0,
            'extreme_anomaly_days': 0,
            'severe_anomaly_days': 0,
            'moderate_anomaly_days': 0,
            'normal_days': 0,
            'jobs_before_sampling': 0,
            'jobs_after_sampling': 0,
            'sampling_ratio': 0.0
        }
        
        logger.info("智能采样器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def perform_intelligent_sampling(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        执行智能采样
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径字典
        """
        logger.info(f"开始智能采样: {input_file}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info("加载数据...")
        original_df = pd.read_csv(input_file, low_memory=False)
        self.stats['total_jobs'] = len(original_df)
        self.stats['jobs_before_sampling'] = len(original_df)

        logger.info(f"数据加载完成: {len(original_df):,} 条记录")

        # 步骤1: 异常检测
        logger.info("步骤1: 异常检测...")
        df_with_anomalies = self._detect_anomalies(original_df)

        # 步骤2: 分层采样
        logger.info("步骤2: 分层采样...")
        sampled_df = self._apply_stratified_sampling(df_with_anomalies)

        # 步骤3: 采样质量保证
        logger.info("步骤3: 采样质量保证...")
        final_df = self._ensure_sampling_quality(sampled_df)

        # 步骤4: 生成采样前后对比可视化
        logger.info("步骤4: 生成采样前后对比可视化...")
        self._generate_comparison_visualizations(original_df, final_df, output_dir)

        # 保存结果
        output_files = self._save_results(final_df, output_dir)

        # 生成采样报告
        self._generate_sampling_report(output_dir)

        logger.info("智能采样完成")
        return output_files
    
    def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测异常"""
        df = df.copy()
        
        # 确保有submit_time字段
        if 'submit_time' not in df.columns:
            logger.warning("缺少submit_time字段，无法进行时间异常检测")
            df['anomaly_level'] = 'normal'
            return df
        
        # 转换时间格式
        df['submit_time'] = pd.to_datetime(df['submit_time'], errors='coerce')
        df = df[df['submit_time'].notna()]  # 移除无效时间
        
        # 按日统计作业数量
        df['submit_date'] = df['submit_time'].dt.date
        daily_counts = df.groupby('submit_date').size()
        
        logger.info(f"分析 {len(daily_counts)} 天的提交数据...")
        
        # 统计学异常检测
        anomaly_config = self.config['intelligent_sampling']['anomaly_detection']['statistical_methods']['daily_submission']
        
        # 计算阈值
        median_count = daily_counts.median()
        mean_count = daily_counts.mean()
        std_count = daily_counts.std()
        
        # 自定义阈值
        extreme_threshold = anomaly_config['custom_thresholds']['extreme_anomaly']
        severe_threshold = anomaly_config['custom_thresholds']['severe_anomaly']
        moderate_threshold = anomaly_config['custom_thresholds']['moderate_anomaly']
        
        # 统计学阈值
        iqr_threshold = median_count + (daily_counts.quantile(0.75) - daily_counts.quantile(0.25)) * anomaly_config['thresholds']['iqr_multiplier']
        zscore_threshold = mean_count + std_count * anomaly_config['thresholds']['zscore_threshold']
        
        # 综合阈值
        final_extreme_threshold = max(extreme_threshold, median_count * 10, mean_count + 5 * std_count)
        final_severe_threshold = max(severe_threshold, iqr_threshold, zscore_threshold)
        final_moderate_threshold = moderate_threshold
        
        logger.info(f"异常检测阈值:")
        logger.info(f"  极端异常: {final_extreme_threshold:,.0f} 作业/天")
        logger.info(f"  严重异常: {final_severe_threshold:,.0f} 作业/天")
        logger.info(f"  中等异常: {final_moderate_threshold:,.0f} 作业/天")
        
        # 分类异常日期
        extreme_dates = daily_counts[daily_counts >= final_extreme_threshold].index
        severe_dates = daily_counts[(daily_counts >= final_severe_threshold) & (daily_counts < final_extreme_threshold)].index
        moderate_dates = daily_counts[(daily_counts >= final_moderate_threshold) & (daily_counts < final_severe_threshold)].index
        
        # 统计异常天数
        self.stats['extreme_anomaly_days'] = len(extreme_dates)
        self.stats['severe_anomaly_days'] = len(severe_dates)
        self.stats['moderate_anomaly_days'] = len(moderate_dates)
        self.stats['normal_days'] = len(daily_counts) - len(extreme_dates) - len(severe_dates) - len(moderate_dates)
        
        logger.info(f"异常检测结果:")
        logger.info(f"  极端异常天数: {self.stats['extreme_anomaly_days']} 天")
        logger.info(f"  严重异常天数: {self.stats['severe_anomaly_days']} 天")
        logger.info(f"  中等异常天数: {self.stats['moderate_anomaly_days']} 天")
        logger.info(f"  正常天数: {self.stats['normal_days']} 天")
        
        # 标记异常级别
        df['anomaly_level'] = 'normal'
        df.loc[df['submit_date'].isin(extreme_dates), 'anomaly_level'] = 'extreme'
        df.loc[df['submit_date'].isin(severe_dates), 'anomaly_level'] = 'severe'
        df.loc[df['submit_date'].isin(moderate_dates), 'anomaly_level'] = 'moderate'
        
        return df
    
    def _apply_stratified_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用分层采样策略"""
        df = df.copy()
        
        sampling_strategies = self.config['intelligent_sampling']['sampling_strategies']
        sampled_dfs = []
        
        # 按异常级别分层采样
        for anomaly_level in ['extreme', 'severe', 'moderate', 'normal']:
            level_df = df[df['anomaly_level'] == anomaly_level].copy()
            
            if len(level_df) == 0:
                continue
            
            logger.info(f"处理 {anomaly_level} 异常级别: {len(level_df):,} 条记录")
            
            if anomaly_level == 'extreme':
                sampled_df = self._sample_extreme_anomaly(level_df, sampling_strategies['extreme_anomaly'])
            elif anomaly_level == 'severe':
                sampled_df = self._sample_severe_anomaly(level_df, sampling_strategies['severe_anomaly'])
            elif anomaly_level == 'moderate':
                sampled_df = self._sample_moderate_anomaly(level_df, sampling_strategies['moderate_anomaly'])
            else:  # normal
                sampled_df = self._sample_normal_data(level_df, sampling_strategies['normal_data'])
            
            sampled_dfs.append(sampled_df)
            
            logger.info(f"  采样后: {len(sampled_df):,} 条记录 (保留率: {len(sampled_df)/len(level_df)*100:.1f}%)")
        
        # 合并采样结果
        if sampled_dfs:
            result_df = pd.concat(sampled_dfs, ignore_index=True)
        else:
            result_df = df.copy()
        
        self.stats['jobs_after_sampling'] = len(result_df)
        self.stats['sampling_ratio'] = len(result_df) / len(df) if len(df) > 0 else 0
        
        logger.info(f"总体采样结果: {len(result_df):,} / {len(df):,} (保留率: {self.stats['sampling_ratio']*100:.1f}%)")
        
        return result_df
    
    def _sample_extreme_anomaly(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """极端异常采样"""
        if len(df) == 0:
            return df
        
        target_ratio = strategy['sampling_ratio']
        min_samples = strategy['min_samples']
        max_samples = strategy['max_samples']
        
        # 计算目标样本数
        target_count = max(min_samples, min(max_samples, int(len(df) * target_ratio)))
        
        if target_count >= len(df):
            return df
        
        # 分层采样
        if strategy['sampling_method'] == 'stratified':
            return self._stratified_sampling(df, target_count, strategy)
        else:
            return df.sample(n=target_count, random_state=42)
    
    def _sample_severe_anomaly(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """严重异常采样 - 支持配置化采样策略"""
        if len(df) == 0:
            return df

        # 检查是否启用采样
        if not strategy.get('enable_sampling', True):
            logger.info(f"严重异常数据不采样，保留所有 {len(df):,} 条记录")
            return df

        # 如果采样方法是no_sampling，直接返回所有数据
        if strategy.get('sampling_method') == 'no_sampling':
            logger.info(f"严重异常数据不采样，保留所有 {len(df):,} 条记录")
            return df

        # 执行采样
        target_ratio = strategy.get('sampling_ratio', 1.0)
        min_samples = strategy.get('min_samples', 1000)
        max_samples = strategy.get('max_samples', 50000)

        # 计算目标样本数
        target_count = max(min_samples, min(max_samples, int(len(df) * target_ratio)))

        if target_count >= len(df):
            logger.info(f"严重异常数据无需采样，保留所有 {len(df):,} 条记录")
            return df

        logger.info(f"严重异常数据采样: {len(df):,} → {target_count:,} (采样率: {target_ratio:.1%})")

        # 分层采样
        if strategy.get('sampling_method') == 'stratified':
            return self._stratified_sampling(df, target_count, strategy)
        # 系统采样
        elif strategy.get('sampling_method') == 'systematic':
            step = len(df) // target_count
            indices = list(range(0, len(df), step))[:target_count]
            return df.iloc[indices]
        else:
            return df.sample(n=target_count, random_state=42)
    
    def _sample_moderate_anomaly(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """中等异常采样 - 支持配置化采样策略"""
        if len(df) == 0:
            return df

        # 检查是否启用采样
        if not strategy.get('enable_sampling', True):
            logger.info(f"中等异常数据不采样，保留所有 {len(df):,} 条记录")
            return df

        # 如果采样方法是no_sampling，直接返回所有数据
        if strategy.get('sampling_method') == 'no_sampling':
            logger.info(f"中等异常数据不采样，保留所有 {len(df):,} 条记录")
            return df

        # 执行采样
        target_ratio = strategy.get('sampling_ratio', 1.0)
        min_samples = strategy.get('min_samples', 1000)
        max_samples = strategy.get('max_samples', 15000)

        # 计算目标样本数
        target_count = max(min_samples, min(max_samples, int(len(df) * target_ratio)))

        if target_count >= len(df):
            logger.info(f"中等异常数据无需采样，保留所有 {len(df):,} 条记录")
            return df

        logger.info(f"中等异常数据采样: {len(df):,} → {target_count:,} (采样率: {target_ratio:.1%})")

        # 分层采样
        if strategy.get('sampling_method') == 'stratified':
            return self._stratified_sampling(df, target_count, strategy)
        # 聚类采样
        elif strategy.get('sampling_method') == 'cluster':
            return self._cluster_sampling(df, target_count, strategy)
        else:
            return df.sample(n=target_count, random_state=42)
    
    def _sample_normal_data(self, df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
        """正常数据采样 - 修改为不采样，保留所有数据"""
        if len(df) == 0:
            return df

        # 检查是否启用采样
        if not strategy.get('enable_sampling', True):
            logger.info(f"正常数据不采样，保留所有 {len(df):,} 条记录")
            return df

        # 如果采样方法是no_sampling，直接返回所有数据
        if strategy.get('sampling_method') == 'no_sampling':
            logger.info(f"正常数据不采样，保留所有 {len(df):,} 条记录")
            return df

        # 检测脚本错误（如果启用）
        if strategy.get('script_error_detection', {}).get('enable', False):
            df = self._detect_and_sample_script_errors(df, strategy['script_error_detection'])

        # 原有采样逻辑（保留以防需要）
        target_ratio = strategy['sampling_ratio']
        target_count = int(len(df) * target_ratio)

        if target_count >= len(df):
            return df

        return df.sample(n=target_count, random_state=42)
    
    def _stratified_sampling(self, df: pd.DataFrame, target_count: int, strategy: Dict[str, Any]) -> pd.DataFrame:
        """分层采样"""
        stratification = strategy.get('stratification', {})
        sampled_dfs = []
        
        # 按用户分层
        if stratification.get('by_user', False) and 'final_user_id' in df.columns:
            for user_id in df['final_user_id'].unique():
                user_df = df[df['final_user_id'] == user_id]
                user_target = max(1, int(target_count * len(user_df) / len(df)))
                user_target = min(user_target, len(user_df))
                
                if user_target > 0:
                    sampled_user_df = user_df.sample(n=user_target, random_state=42)
                    sampled_dfs.append(sampled_user_df)
        else:
            # 简单随机采样
            return df.sample(n=target_count, random_state=42)
        
        if sampled_dfs:
            result = pd.concat(sampled_dfs, ignore_index=True)
            # 如果超出目标数量，再次采样
            if len(result) > target_count:
                result = result.sample(n=target_count, random_state=42)
            return result
        else:
            return df.sample(n=target_count, random_state=42)

    def _cluster_sampling(self, df: pd.DataFrame, target_count: int, strategy: Dict[str, Any]) -> pd.DataFrame:
        """聚类采样"""
        # 简化实现：按作业名聚类采样
        if 'job_name' in df.columns:
            sampled_dfs = []
            for job_name in df['job_name'].unique():
                job_df = df[df['job_name'] == job_name]
                job_target = max(1, int(target_count * len(job_df) / len(df)))
                job_target = min(job_target, len(job_df))

                if job_target > 0:
                    sampled_job_df = job_df.sample(n=job_target, random_state=42)
                    sampled_dfs.append(sampled_job_df)

            if sampled_dfs:
                result = pd.concat(sampled_dfs, ignore_index=True)
                if len(result) > target_count:
                    result = result.sample(n=target_count, random_state=42)
                return result

        return df.sample(n=target_count, random_state=42)

    def _detect_and_sample_script_errors(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """检测和采样脚本错误"""
        if 'exit_status' not in df.columns:
            return df

        # 分离成功和失败的作业
        success_df = df[df['exit_status'] == 0]
        error_df = df[df['exit_status'] != 0]

        # 对错误作业进行特殊处理
        if len(error_df) > 0:
            error_ratio = config.get('error_sampling_ratio', 0.5)
            error_target = max(1, int(len(error_df) * error_ratio))
            error_target = min(error_target, len(error_df))

            sampled_error_df = error_df.sample(n=error_target, random_state=42)

            # 合并成功和采样后的错误作业
            return pd.concat([success_df, sampled_error_df], ignore_index=True)

        return df

    def _ensure_sampling_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保采样质量"""
        df = df.copy()

        quality_config = self.config['intelligent_sampling']['quality_assurance']

        # 检查代表性
        representativeness_config = quality_config.get('representativeness', {})
        if representativeness_config.get('time_distribution_check', False):
            df = self._check_representativeness(df, representativeness_config)

        # 检查偏差 - 简化实现
        if quality_config.get('bias_detection', {}).get('enable', False):
            df = self._detect_bias(df, quality_config['bias_detection'])

        return df

    def _check_representativeness(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """检查代表性"""
        # 简单检查：确保各个时间段都有代表
        if 'submit_time' in df.columns:
            df['submit_hour'] = pd.to_datetime(df['submit_time']).dt.hour
            hourly_counts = df['submit_hour'].value_counts()

            # 如果某些小时没有数据，记录警告
            missing_hours = set(range(24)) - set(hourly_counts.index)
            if missing_hours:
                logger.warning(f"采样后缺少以下小时的数据: {sorted(missing_hours)}")

        return df

    def _detect_bias(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """检测偏差"""
        # 简单偏差检测：检查用户分布
        if 'final_user_id' in df.columns:
            user_counts = df['final_user_id'].value_counts()

            # 检查是否有用户占比过高
            max_user_ratio = user_counts.iloc[0] / len(df) if len(user_counts) > 0 else 0
            threshold = config.get('max_user_ratio', 0.5)

            if max_user_ratio > threshold:
                logger.warning(f"检测到用户偏差: 最大用户占比 {max_user_ratio:.2%} > 阈值 {threshold:.2%}")

        return df

    def _save_results(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """保存采样结果"""
        output_files = {}

        # 清理临时列
        temp_columns = ['submit_date', 'anomaly_level', 'submit_hour']
        result_df = df.drop(columns=temp_columns, errors='ignore')

        # 保存主要结果
        main_output_file = os.path.join(output_dir, "intelligent_sampling_result.csv")
        result_df.to_csv(main_output_file, index=False)
        output_files['sampled_data'] = main_output_file

        logger.info(f"保存采样结果: {len(result_df):,} 条 -> {main_output_file}")

        # 保存采样统计
        if 'anomaly_level' in df.columns:
            sampling_stats = df.groupby('anomaly_level').size().reset_index(name='count')
            stats_file = os.path.join(output_dir, "sampling_statistics.csv")
            sampling_stats.to_csv(stats_file, index=False)
            output_files['statistics'] = stats_file

            logger.info(f"保存采样统计: {stats_file}")

        return output_files

    def _generate_sampling_report(self, output_dir: str):
        """生成采样报告"""
        report_file = os.path.join(output_dir, "intelligent_sampling_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 智能采样报告 ===\n\n")
            f.write(f"原始作业数: {self.stats['jobs_before_sampling']:,}\n")
            f.write(f"采样后作业数: {self.stats['jobs_after_sampling']:,}\n")
            f.write(f"总体保留率: {self.stats['sampling_ratio']*100:.2f}%\n\n")

            f.write("异常检测结果:\n")
            f.write(f"  极端异常天数: {self.stats['extreme_anomaly_days']} 天\n")
            f.write(f"  严重异常天数: {self.stats['severe_anomaly_days']} 天\n")
            f.write(f"  中等异常天数: {self.stats['moderate_anomaly_days']} 天\n")
            f.write(f"  正常天数: {self.stats['normal_days']} 天\n\n")

            f.write("采样策略:\n")
            f.write("  - 极端异常: 大幅采样保留代表性\n")
            f.write("  - 严重异常: 系统采样保持分布\n")
            f.write("  - 中等异常: 聚类采样保留多样性\n")
            f.write("  - 正常数据: 高比例保留\n")

        logger.info(f"采样报告已保存: {report_file}")

    def _generate_comparison_visualizations(self, original_df: pd.DataFrame, sampled_df: pd.DataFrame, output_dir: str):
        """生成采样前后对比可视化图表"""
        logger.info("生成采样前后对比可视化...")

        try:
            # 设置中文字体和样式
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False
            sns.set_style("whitegrid")

            # 1. Daily Submission Trend 对比 - 分离图表
            self._plot_daily_submission_trend(original_df, sampled_df, output_dir)

            # 2. Daily Submission Trend 对比 - 同一坐标轴
            self._plot_daily_submission_overlay(original_df, sampled_df, output_dir)

            # 3. CPU vs GPU Submission Trend 对比
            self._plot_cpu_gpu_trend(original_df, sampled_df, output_dir)

            logger.info("可视化图表生成完成")

        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")

    def _plot_daily_submission_trend(self, original_df: pd.DataFrame, sampled_df: pd.DataFrame, output_dir: str):
        """绘制每日提交趋势对比图（已修复）"""
        # 转换时间格式并统计每日提交数量
        original_daily = original_df.groupby(pd.to_datetime(original_df['submit_time']).dt.date).size().reset_index(name='original_count')
        original_daily.rename(columns={'submit_time': 'date'}, inplace=True)
        
        sampled_daily = sampled_df.groupby(pd.to_datetime(sampled_df['submit_time']).dt.date).size().reset_index(name='sampled_count')
        sampled_daily.rename(columns={'submit_time': 'date'}, inplace=True)

        # 使用merge安全地合并数据，避免数据污染
        comparison_df = pd.merge(original_daily, sampled_daily, on='date', how='left').fillna(0)
        comparison_df['sampling_rate'] = comparison_df['sampled_count'] / comparison_df['original_count']

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # 1. 原始数据趋势
        ax1.plot(comparison_df['date'], comparison_df['original_count'], color='blue', linewidth=1.5, alpha=0.8, label='Original Data')
        ax1.set_title('Daily Submission Trend - Before Sampling', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Jobs', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, comparison_df['original_count'].max() * 1.1)

        # 2. 采样后数据趋势
        ax2.plot(comparison_df['date'], comparison_df['sampled_count'], color='red', linewidth=1.5, alpha=0.8, label='Sampled Data')
        ax2.set_title('Daily Submission Trend - After Sampling', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Number of Jobs', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # Y轴范围根据采样后的非极端峰值来设置，以获得更好的可视化效果
        normal_peak = comparison_df[comparison_df['sampling_rate'] > 0.9]['sampled_count'].max()
        ax2.set_ylim(0, normal_peak * 1.5)

        # 3. 添加采样率标注（只标注采样率不为100%的高峰）
        high_peak_threshold = comparison_df['original_count'].quantile(0.95)  # 95分位数
        peaks_to_annotate = comparison_df[(comparison_df['original_count'] > high_peak_threshold) & (comparison_df['sampling_rate'] < 0.99)]

        for _, row in peaks_to_annotate.iterrows():
            ax2.annotate(f'{row.sampling_rate:.1%}',
                       xy=(row.date, row.sampled_count),
                       xytext=(0, 15), # 向上偏移15个点
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black'))

        # 统一格式化x轴
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.1)

        # 保存图表
        trend_file = os.path.join(output_dir, "daily_submission_trend_comparison.png")
        plt.savefig(trend_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"每日提交趋势对比图已修复并保存: {trend_file}")

    def _plot_daily_submission_overlay(self, original_df: pd.DataFrame, sampled_df: pd.DataFrame, output_dir: str):
        """绘制每日提交趋势对比图（同一坐标轴）"""
        # 转换时间格式并统计每日提交数量
        original_daily = original_df.groupby(pd.to_datetime(original_df['submit_time']).dt.date).size().reset_index(name='original_count')
        original_daily.rename(columns={'submit_time': 'date'}, inplace=True)

        sampled_daily = sampled_df.groupby(pd.to_datetime(sampled_df['submit_time']).dt.date).size().reset_index(name='sampled_count')
        sampled_daily.rename(columns={'submit_time': 'date'}, inplace=True)

        # 使用merge安全地合并数据，避免数据污染
        comparison_df = pd.merge(original_daily, sampled_daily, on='date', how='left').fillna(0)
        comparison_df['sampling_rate'] = comparison_df['sampled_count'] / comparison_df['original_count']

        # 创建图表
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

        # 绘制原始数据和采样数据在同一坐标轴上
        ax.plot(comparison_df['date'], comparison_df['original_count'],
                color='blue', linewidth=2, alpha=0.8, label='Original Data')
        ax.plot(comparison_df['date'], comparison_df['sampled_count'],
                color='red', linewidth=2, alpha=0.8, label='Sampled Data')

        # 设置图表属性
        ax.set_title('Daily Submission Trend - Before vs After Sampling (Overlay)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Jobs', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # 添加采样率标注（只标注采样率不为100%的高峰）
        high_peak_threshold = comparison_df['original_count'].quantile(0.95)  # 95分位数
        peaks_to_annotate = comparison_df[(comparison_df['original_count'] > high_peak_threshold) &
                                        (comparison_df['sampling_rate'] < 0.99)]

        for _, row in peaks_to_annotate.iterrows():
            ax.annotate(f'Sampling: {row.sampling_rate:.1%}',
                       xy=(row.date, row.sampled_count),
                       xytext=(0, 20), # 向上偏移20个点
                       textcoords='offset points',
                       ha='center',
                       fontsize=10,
                       fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='black'))

        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        # 保存图表
        overlay_file = os.path.join(output_dir, "daily_submission_trend_overlay.png")
        plt.savefig(overlay_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"每日提交趋势叠加对比图已保存: {overlay_file}")

    def _plot_cpu_gpu_trend(self, original_df: pd.DataFrame, sampled_df: pd.DataFrame, output_dir: str):
        """绘制CPU vs GPU提交趋势对比图 - 多种可视化方案"""
        # 判断作业类型 (基于exec_hosts或其他字段)
        def classify_job_type(df):
            df = df.copy()
            df['job_type'] = 'CPU'  # 默认为CPU作业

            # 基于exec_hosts字段判断GPU作业
            if 'exec_hosts' in df.columns:
                gpu_mask = df['exec_hosts'].str.contains('gpu', case=False, na=False)
                df.loc[gpu_mask, 'job_type'] = 'GPU'

            return df

        # 分类作业类型
        original_classified = classify_job_type(original_df)
        sampled_classified = classify_job_type(sampled_df)

        # 添加日期列
        original_classified['submit_date'] = pd.to_datetime(original_classified['submit_time']).dt.date
        sampled_classified['submit_date'] = pd.to_datetime(sampled_classified['submit_time']).dt.date

        # 统计每日CPU/GPU提交数量
        original_type_daily = original_classified.groupby(['submit_date', 'job_type']).size().unstack(fill_value=0)
        sampled_type_daily = sampled_classified.groupby(['submit_date', 'job_type']).size().unstack(fill_value=0)

        # 生成多种可视化方案
        self._plot_dual_y_axis_trend(original_type_daily, sampled_type_daily, output_dir)
        self._plot_log_scale_trend(original_type_daily, sampled_type_daily, output_dir)
        self._plot_separated_trend(original_type_daily, sampled_type_daily, output_dir)
        self._plot_ratio_trend(original_type_daily, sampled_type_daily, output_dir)

    def _plot_dual_y_axis_trend(self, original_type_daily, sampled_type_daily, output_dir):
        """双Y轴图表 - 解决尺度不匹配问题"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # 原始数据 - 双Y轴
        if 'CPU' in original_type_daily.columns and 'GPU' in original_type_daily.columns:
            color1 = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('CPU Jobs', color=color1, fontsize=12)
            line1 = ax1.plot(original_type_daily.index, original_type_daily['CPU'],
                           color=color1, linewidth=2, label='CPU Jobs')
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(True, alpha=0.3)

            ax1_twin = ax1.twinx()
            color2 = 'tab:red'
            ax1_twin.set_ylabel('GPU Jobs', color=color2, fontsize=12)
            line2 = ax1_twin.plot(original_type_daily.index, original_type_daily['GPU'],
                                color=color2, linewidth=2, label='GPU Jobs')
            ax1_twin.tick_params(axis='y', labelcolor=color2)

            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper left')

        ax1.set_title('CPU vs GPU Submission Trend - Before Sampling (Dual Y-Axis)',
                     fontsize=14, fontweight='bold')

        # 采样后数据 - 双Y轴
        if 'CPU' in sampled_type_daily.columns and 'GPU' in sampled_type_daily.columns:
            color1 = 'tab:blue'
            ax2.set_xlabel('Date')
            ax2.set_ylabel('CPU Jobs', color=color1, fontsize=12)
            line1 = ax2.plot(sampled_type_daily.index, sampled_type_daily['CPU'],
                           color=color1, linewidth=2, label='CPU Jobs')
            ax2.tick_params(axis='y', labelcolor=color1)
            ax2.grid(True, alpha=0.3)

            ax2_twin = ax2.twinx()
            color2 = 'tab:red'
            ax2_twin.set_ylabel('GPU Jobs', color=color2, fontsize=12)
            line2 = ax2_twin.plot(sampled_type_daily.index, sampled_type_daily['GPU'],
                                color=color2, linewidth=2, label='GPU Jobs')
            ax2_twin.tick_params(axis='y', labelcolor=color2)

            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2.legend(lines, labels, loc='upper left')

        ax2.set_title('CPU vs GPU Submission Trend - After Sampling (Dual Y-Axis)',
                     fontsize=14, fontweight='bold')

        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        dual_y_file = os.path.join(output_dir, "cpu_gpu_trend_dual_y_axis.png")
        plt.savefig(dual_y_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"双Y轴CPU vs GPU趋势图已保存: {dual_y_file}")

    def _plot_log_scale_trend(self, original_type_daily, sampled_type_daily, output_dir):
        """对数尺度图表 - 显示相对变化趋势"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 原始数据 - 对数尺度
        if 'CPU' in original_type_daily.columns:
            ax1.plot(original_type_daily.index, original_type_daily['CPU'],
                    color='blue', linewidth=2, alpha=0.8, label='CPU Jobs')
        if 'GPU' in original_type_daily.columns:
            ax1.plot(original_type_daily.index, original_type_daily['GPU'],
                    color='red', linewidth=2, alpha=0.8, label='GPU Jobs')

        ax1.set_yscale('log')
        ax1.set_title('CPU vs GPU Submission Trend - Before Sampling (Log Scale)',
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Jobs (Log Scale)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 采样后数据 - 对数尺度
        if 'CPU' in sampled_type_daily.columns:
            ax2.plot(sampled_type_daily.index, sampled_type_daily['CPU'],
                    color='blue', linewidth=2, alpha=0.8, label='CPU Jobs')
        if 'GPU' in sampled_type_daily.columns:
            ax2.plot(sampled_type_daily.index, sampled_type_daily['GPU'],
                    color='red', linewidth=2, alpha=0.8, label='GPU Jobs')

        ax2.set_yscale('log')
        ax2.set_title('CPU vs GPU Submission Trend - After Sampling (Log Scale)',
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Number of Jobs (Log Scale)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        log_scale_file = os.path.join(output_dir, "cpu_gpu_trend_log_scale.png")
        plt.savefig(log_scale_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"对数尺度CPU vs GPU趋势图已保存: {log_scale_file}")

    def _plot_separated_trend(self, original_type_daily, sampled_type_daily, output_dir):
        """分离子图 - CPU和GPU各自独立显示"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # 原始数据 - CPU子图
        if 'CPU' in original_type_daily.columns:
            ax1.plot(original_type_daily.index, original_type_daily['CPU'],
                    color='blue', linewidth=2)
            ax1.set_title('CPU Jobs - Before Sampling', fontsize=12, fontweight='bold')
            ax1.set_ylabel('CPU Jobs', fontsize=10)
            ax1.grid(True, alpha=0.3)

        # 原始数据 - GPU子图
        if 'GPU' in original_type_daily.columns:
            ax2.plot(original_type_daily.index, original_type_daily['GPU'],
                    color='red', linewidth=2)
            ax2.set_title('GPU Jobs - Before Sampling', fontsize=12, fontweight='bold')
            ax2.set_ylabel('GPU Jobs', fontsize=10)
            ax2.grid(True, alpha=0.3)

        # 采样后数据 - CPU子图
        if 'CPU' in sampled_type_daily.columns:
            ax3.plot(sampled_type_daily.index, sampled_type_daily['CPU'],
                    color='blue', linewidth=2)
            ax3.set_title('CPU Jobs - After Sampling', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_ylabel('CPU Jobs', fontsize=10)
            ax3.grid(True, alpha=0.3)

        # 采样后数据 - GPU子图
        if 'GPU' in sampled_type_daily.columns:
            ax4.plot(sampled_type_daily.index, sampled_type_daily['GPU'],
                    color='red', linewidth=2)
            ax4.set_title('GPU Jobs - After Sampling', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Date', fontsize=10)
            ax4.set_ylabel('GPU Jobs', fontsize=10)
            ax4.grid(True, alpha=0.3)

        # 格式化x轴日期
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, fontsize=8)

        plt.tight_layout()
        separated_file = os.path.join(output_dir, "cpu_gpu_trend_separated.png")
        plt.savefig(separated_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"分离子图CPU vs GPU趋势图已保存: {separated_file}")

    def _plot_ratio_trend(self, original_type_daily, sampled_type_daily, output_dir):
        """比例图表 - 显示GPU作业占比趋势"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # 计算GPU作业占比
        def calculate_gpu_ratio(type_daily):
            if 'CPU' in type_daily.columns and 'GPU' in type_daily.columns:
                total_jobs = type_daily['CPU'] + type_daily['GPU']
                gpu_ratio = (type_daily['GPU'] / total_jobs * 100).fillna(0)
                return gpu_ratio
            return pd.Series(dtype=float)

        # 原始数据GPU占比
        original_gpu_ratio = calculate_gpu_ratio(original_type_daily)
        if not original_gpu_ratio.empty:
            ax1.plot(original_gpu_ratio.index, original_gpu_ratio,
                    color='green', linewidth=2, marker='o', markersize=3)
            ax1.set_title('GPU Jobs Percentage - Before Sampling', fontsize=14, fontweight='bold')
            ax1.set_ylabel('GPU Jobs Percentage (%)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(100, original_gpu_ratio.max() * 1.1))

        # 采样后数据GPU占比
        sampled_gpu_ratio = calculate_gpu_ratio(sampled_type_daily)
        if not sampled_gpu_ratio.empty:
            ax2.plot(sampled_gpu_ratio.index, sampled_gpu_ratio,
                    color='green', linewidth=2, marker='o', markersize=3)
            ax2.set_title('GPU Jobs Percentage - After Sampling', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('GPU Jobs Percentage (%)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, max(100, sampled_gpu_ratio.max() * 1.1))

        # 格式化x轴日期
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        ratio_file = os.path.join(output_dir, "gpu_jobs_percentage_trend.png")
        plt.savefig(ratio_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"GPU作业占比趋势图已保存: {ratio_file}")
