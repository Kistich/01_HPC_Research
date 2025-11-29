#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
用户行为分析模块
基于Helios项目的user characterization方法
分析用户的资源使用模式、作业提交行为等
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class UserBehaviorAnalyzer:
    """用户行为分析器"""
    
    def __init__(self, config: Dict[str, Any], output_paths: Dict[str, Path], visualizer):
        """
        初始化用户行为分析器
        
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
        执行用户特征分析 - 严格按照Helios方法

        Args:
            data: 预处理后的数据（包含Helios兼容格式）

        Returns:
            分析结果字典
        """
        logger.info("开始用户特征分析（Helios风格）...")

        # 获取Helios兼容数据
        helios_data = data.get('helios_data', {})
        if not helios_data or 'cluster_user' not in helios_data:
            raise ValueError("缺少Helios兼容数据")

        cluster_user = helios_data['cluster_user']

        # 按照Helios的分析方法执行分析
        results = {
            # 1. 用户GPU时间CDF分析（对应Helios Figure 5a）
            'user_gpu_time_cdf': self._analyze_user_gpu_time_cdf_helios(cluster_user),

            # 2. 用户CPU时间CDF分析（对应Helios Figure 5b）
            'user_cpu_time_cdf': self._analyze_user_cpu_time_cdf_helios(cluster_user),

            # 3. 用户GPU排队延迟CDF分析（对应Helios Figure 6a）
            'user_gpu_pend_cdf': self._analyze_user_gpu_pend_cdf_helios(cluster_user),

            # 4. 用户CPU排队延迟CDF分析（新增 - 对应GPU分析）
            'user_cpu_pend_cdf': self._analyze_user_cpu_pend_cdf_helios(cluster_user),

            # 5. 用户GPU完成率分布分析（对应Helios Figure 6b）
            'user_gpu_completion_rate': self._analyze_user_completion_rate_helios(cluster_user),

            # 6. 用户CPU完成率分布分析（新增 - 对应GPU分析）
            'user_cpu_completion_rate': self._analyze_user_cpu_completion_rate_helios(cluster_user)
        }

        # 生成Helios风格可视化（Figure 5和6复现）
        self._generate_helios_visualizations(cluster_user, results)

        logger.info("用户特征分析完成")
        return results

    def _analyze_user_gpu_time_cdf_helios(self, cluster_user: pd.DataFrame) -> Dict[str, Any]:
        """分析用户GPU时间CDF - 严格按照Helios方法（Figure 5a）"""
        logger.info("分析用户GPU时间CDF（Helios风格）...")

        if 'total_gpu_time' not in cluster_user.columns:
            return {}

        # 获取用户GPU时间数据
        gpu_time = cluster_user['total_gpu_time'].copy()
        gpu_time = gpu_time[gpu_time > 0]  # 只考虑有GPU使用的用户

        if len(gpu_time) == 0:
            return {}

        # 按GPU时间降序排列
        gpu_time_sorted = gpu_time.sort_values(ascending=False)
        total_gpu_time = gpu_time_sorted.sum()

        # 计算累积分布
        cumulative_percentages = []
        for i in range(len(gpu_time_sorted)):
            cumulative_gpu_time = gpu_time_sorted.iloc[:i+1].sum()
            percentage = (cumulative_gpu_time / total_gpu_time) * 100
            cumulative_percentages.append(percentage)

        # 计算用户百分比
        user_percentages = [(i+1) / len(gpu_time_sorted) * 100 for i in range(len(gpu_time_sorted))]

        return {
            'user_percentages': user_percentages,
            'cumulative_gpu_percentages': cumulative_percentages,
            'total_users_with_gpu': len(gpu_time_sorted),
            'total_gpu_time': total_gpu_time
        }

    def _analyze_user_cpu_time_cdf_helios(self, cluster_user: pd.DataFrame) -> Dict[str, Any]:
        """分析用户CPU时间CDF - 严格按照Helios方法（Figure 5b）"""
        logger.info("分析用户CPU时间CDF（Helios风格）...")

        if 'total_cpu_only_time' not in cluster_user.columns:
            return {}

        # 获取用户CPU时间数据
        cpu_time = cluster_user['total_cpu_only_time'].copy()
        cpu_time = cpu_time[cpu_time > 0]  # 只考虑有CPU使用的用户

        if len(cpu_time) == 0:
            return {}

        # 按CPU时间降序排列
        cpu_time_sorted = cpu_time.sort_values(ascending=False)
        total_cpu_time = cpu_time_sorted.sum()

        # 计算累积分布
        cumulative_percentages = []
        for i in range(len(cpu_time_sorted)):
            cumulative_cpu_time = cpu_time_sorted.iloc[:i+1].sum()
            percentage = (cumulative_cpu_time / total_cpu_time) * 100
            cumulative_percentages.append(percentage)

        # 计算用户百分比
        user_percentages = [(i+1) / len(cpu_time_sorted) * 100 for i in range(len(cpu_time_sorted))]

        return {
            'user_percentages': user_percentages,
            'cumulative_cpu_percentages': cumulative_percentages,
            'total_users_with_cpu': len(cpu_time_sorted),
            'total_cpu_time': total_cpu_time
        }

    def _analyze_user_gpu_pend_cdf_helios(self, cluster_user: pd.DataFrame) -> Dict[str, Any]:
        """分析用户GPU排队延迟CDF - 严格按照Helios方法（Figure 6a）"""
        logger.info("分析用户GPU排队延迟CDF（Helios风格）...")

        if 'total_gpu_pend_time' not in cluster_user.columns:
            return {}

        # 获取用户GPU排队时间数据
        gpu_pend_time = cluster_user['total_gpu_pend_time'].copy()
        gpu_pend_time = gpu_pend_time[gpu_pend_time > 0]  # 只考虑有排队时间的用户

        if len(gpu_pend_time) == 0:
            return {}

        # 按排队时间排序
        gpu_pend_sorted = gpu_pend_time.sort_values()

        # 计算CDF
        cdf_values = []
        for i in range(len(gpu_pend_sorted)):
            cdf_value = (i + 1) / len(gpu_pend_sorted) * 100
            cdf_values.append(cdf_value)

        return {
            'gpu_pend_times': gpu_pend_sorted.tolist(),
            'cdf_values': cdf_values,
            'total_users_with_pend': len(gpu_pend_sorted)
        }

    def _analyze_user_completion_rate_helios(self, cluster_user: pd.DataFrame) -> Dict[str, Any]:
        """分析用户GPU完成率分布 - 严格按照Helios方法（Figure 6b）"""
        logger.info("分析用户GPU完成率分布（Helios风格）...")

        if 'completed_gpu_percent' not in cluster_user.columns:
            return {}

        # 获取用户GPU完成率数据
        completion_rates = cluster_user['completed_gpu_percent'].copy()
        completion_rates = completion_rates[completion_rates >= 0]  # 移除无效值

        if len(completion_rates) == 0:
            return {}

        # 创建直方图数据
        bins = np.arange(0, 101, 10)  # 0-100%，每10%一个区间
        hist, bin_edges = np.histogram(completion_rates, bins=bins)

        # 计算百分比
        hist_percentages = (hist / len(completion_rates)) * 100

        return {
            'bin_edges': bin_edges[:-1].tolist(),  # 去掉最后一个边界
            'hist_percentages': hist_percentages.tolist(),
            'total_users': len(completion_rates),
            'mean_completion_rate': completion_rates.mean()
        }

    def _analyze_user_cpu_pend_cdf_helios(self, cluster_user: pd.DataFrame) -> Dict[str, Any]:
        """分析用户CPU排队延迟CDF - 对应GPU分析方法"""
        logger.info("分析用户CPU排队延迟CDF（Helios风格）...")

        if 'total_cpu_pend_time' not in cluster_user.columns:
            return {}

        # 获取用户CPU排队时间数据
        cpu_pend_times = cluster_user['total_cpu_pend_time'].copy()
        cpu_pend_times = cpu_pend_times[cpu_pend_times > 0]  # 移除无效值

        if len(cpu_pend_times) == 0:
            return {}

        # 排序并计算CDF
        cpu_pend_sorted = np.sort(cpu_pend_times)
        cdf_values = np.arange(1, len(cpu_pend_sorted) + 1) / len(cpu_pend_sorted)

        return {
            'cpu_pend_times': cpu_pend_sorted.tolist(),
            'cdf_values': cdf_values.tolist(),
            'total_users_with_pend': len(cpu_pend_sorted)
        }

    def _analyze_user_cpu_completion_rate_helios(self, cluster_user: pd.DataFrame) -> Dict[str, Any]:
        """分析用户CPU完成率分布 - 对应GPU分析方法"""
        logger.info("分析用户CPU完成率分布（Helios风格）...")

        if 'completed_cpu_percent' not in cluster_user.columns:
            return {}

        # 获取用户CPU完成率数据
        completion_rates = cluster_user['completed_cpu_percent'].copy()
        completion_rates = completion_rates[completion_rates >= 0]  # 移除无效值

        if len(completion_rates) == 0:
            return {}

        # 创建直方图数据
        bins = np.arange(0, 101, 10)  # 0-100%，每10%一个区间
        hist, bin_edges = np.histogram(completion_rates, bins=bins)

        # 计算百分比
        hist_percentages = (hist / len(completion_rates)) * 100

        return {
            'bin_edges': bin_edges[:-1].tolist(),  # 去掉最后一个边界
            'hist_percentages': hist_percentages.tolist(),
            'total_users': len(completion_rates),
            'mean_completion_rate': completion_rates.mean()
        }

    def _analyze_user_resource_consumption(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析用户资源消耗 (类似Helios的user resource consumption)"""
        logger.info("分析用户资源消耗...")
        
        user_stats = {}
        
        # 按用户统计作业数量
        user_job_counts = df['user_id'].value_counts()
        
        # 按用户统计资源消耗
        if 'duration' in df.columns and 'actual_node_count' in df.columns:
            df_valid = df[(df['duration'] > 0) & (df['actual_node_count'] > 0)].copy()
            df_valid['node_hours'] = df_valid['duration'] * df_valid['actual_node_count'] / 3600
            
            user_node_hours = df_valid.groupby('user_id')['node_hours'].sum().sort_values(ascending=False)
            
            # 计算累积分布
            total_node_hours = user_node_hours.sum()
            cumulative_percentage = user_node_hours.cumsum() / total_node_hours * 100
            
            # 找出消耗80%资源的用户比例 (帕累托分析)
            heavy_users_threshold = cumulative_percentage[cumulative_percentage <= 80].index
            heavy_users_percentage = len(heavy_users_threshold) / len(user_node_hours) * 100
            
            user_stats = {
                'total_users': len(user_job_counts),
                'user_job_distribution': user_job_counts.head(20).to_dict(),  # 前20用户
                'user_resource_distribution': user_node_hours.head(20).to_dict(),  # 前20用户
                'pareto_analysis': {
                    'heavy_users_count': len(heavy_users_threshold),
                    'heavy_users_percentage': float(heavy_users_percentage),
                    'resource_concentration': 80.0  # 80%资源集中度
                },
                'resource_statistics': {
                    'total_node_hours': float(total_node_hours),
                    'mean_per_user': float(user_node_hours.mean()),
                    'median_per_user': float(user_node_hours.median()),
                    'top_user_consumption': float(user_node_hours.iloc[0]) if len(user_node_hours) > 0 else 0
                }
            }
        
        return user_stats
    
    def _analyze_user_job_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析用户作业模式"""
        logger.info("分析用户作业模式...")
        
        patterns = {}
        
        # 按用户分析作业类型偏好 (基于exec_hosts字段判断GPU节点)
        if 'exec_hosts' in df.columns:
            try:
                def determine_job_type(exec_hosts):
                    """基于执行主机列表判断作业类型"""
                    if pd.isna(exec_hosts) or exec_hosts == '':
                        return 'unknown'

                    # 将exec_hosts转换为字符串并检查是否包含GPU节点
                    hosts_str = str(exec_hosts).lower()

                    # 检查是否包含GPU节点标识
                    # 基于实际数据，主要模式是 gpu1-xx, cpu1-xx, bigmem-xx
                    if 'gpu' in hosts_str:
                        return 'gpu'
                    elif 'cpu' in hosts_str or 'bigmem' in hosts_str:
                        return 'cpu'
                    else:
                        # 其他GPU节点命名模式
                        gpu_patterns = ['v100', 'a100', 'rtx', 'titan', 'tesla', 'k80', 'p100']
                        for pattern in gpu_patterns:
                            if pattern in hosts_str:
                                return 'gpu'
                        return 'cpu'  # 默认为CPU作业

                # 基于exec_hosts判断作业类型
                df_with_type = df.copy()
                df_with_type['job_type_derived'] = df_with_type['exec_hosts'].apply(determine_job_type)

                # 过滤掉unknown类型的作业
                df_known_type = df_with_type[df_with_type['job_type_derived'] != 'unknown']

                if len(df_known_type) > 0:
                    user_job_types = df_known_type.groupby('user_id')['job_type_derived'].apply(
                        lambda x: x.value_counts().to_dict()
                    )

                    # 统计专门使用GPU/CPU的用户
                    gpu_only_users = 0
                    cpu_only_users = 0
                    mixed_users = 0

                    for user_id, job_type_counts in user_job_types.items():
                        if isinstance(job_type_counts, dict):
                            has_gpu = 'gpu' in job_type_counts
                            has_cpu = 'cpu' in job_type_counts

                            if has_gpu and not has_cpu:
                                gpu_only_users += 1
                            elif has_cpu and not has_gpu:
                                cpu_only_users += 1
                            else:
                                mixed_users += 1

                    patterns['job_type_preferences'] = {
                        'gpu_only_users': gpu_only_users,
                        'cpu_only_users': cpu_only_users,
                        'mixed_users': mixed_users,
                        'total_analyzed_users': len(user_job_types),
                        'total_jobs_analyzed': len(df_known_type),
                        'unknown_jobs': len(df_with_type) - len(df_known_type)
                    }
                else:
                    logger.warning("没有找到可识别类型的作业")
                    patterns['job_type_preferences'] = {
                        'gpu_only_users': 0,
                        'cpu_only_users': 0,
                        'mixed_users': 0,
                        'total_analyzed_users': 0,
                        'total_jobs_analyzed': 0,
                        'unknown_jobs': len(df)
                    }

            except Exception as e:
                logger.warning(f"分析用户作业类型偏好时出错: {e}")
                patterns['job_type_preferences'] = {
                    'gpu_only_users': 0,
                    'cpu_only_users': 0,
                    'mixed_users': 0,
                    'total_analyzed_users': 0,
                    'total_jobs_analyzed': 0,
                    'unknown_jobs': 0,
                    'error': str(e)
                }
        
        # 分析用户提交时间模式
        if 'submit_time' in df.columns:
            df_time = df.copy()
            df_time['hour'] = pd.to_datetime(df_time['submit_time']).dt.hour
            df_time['weekday'] = pd.to_datetime(df_time['submit_time']).dt.dayofweek
            
            # 按用户统计提交时间偏好
            user_time_patterns = {}
            for user in df_time['user_id'].unique():
                user_data = df_time[df_time['user_id'] == user]
                
                # 工作时间 vs 非工作时间
                work_hours = user_data[(user_data['hour'] >= 9) & (user_data['hour'] <= 17)]
                work_days = user_data[user_data['weekday'] < 5]  # Monday-Friday
                
                # 避免除零错误
                total_jobs = len(user_data)
                if total_jobs > 0:
                    work_hours_pct = len(work_hours) / total_jobs * 100
                    work_days_pct = len(work_days) / total_jobs * 100
                else:
                    work_hours_pct = 0.0
                    work_days_pct = 0.0

                user_time_patterns[user] = {
                    'total_jobs': total_jobs,
                    'work_hours_percentage': work_hours_pct,
                    'work_days_percentage': work_days_pct
                }
            
            patterns['temporal_patterns'] = user_time_patterns
        
        return patterns
    
    def _analyze_user_queue_experience(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析用户排队体验 (类似Helios的user queue analysis)"""
        logger.info("分析用户排队体验...")
        
        if 'queue_time' not in df.columns:
            return {}
        
        queue_analysis = {}
        
        # 按用户统计排队时间
        df_valid_queue = df[df['queue_time'] >= 0]
        user_queue_stats = df_valid_queue.groupby('user_id')['queue_time'].agg([
            'count', 'mean', 'median', 'max'
        ]).reset_index()
        
        user_queue_stats['mean_hours'] = user_queue_stats['mean'] / 3600
        user_queue_stats['median_hours'] = user_queue_stats['median'] / 3600
        user_queue_stats['max_hours'] = user_queue_stats['max'] / 3600
        
        # 识别排队时间过长的用户
        long_queue_threshold = user_queue_stats['mean_hours'].quantile(0.9)  # 90th percentile
        long_queue_users = user_queue_stats[user_queue_stats['mean_hours'] > long_queue_threshold]
        
        queue_analysis = {
            'user_queue_statistics': {
                'total_users_with_queue_data': len(user_queue_stats),
                'mean_queue_time_hours': float(user_queue_stats['mean_hours'].mean()),
                'median_queue_time_hours': float(user_queue_stats['median_hours'].median()),
                'long_queue_threshold_hours': float(long_queue_threshold),
                'users_with_long_queues': len(long_queue_users)
            },
            'top_queue_sufferers': user_queue_stats.nlargest(10, 'mean_hours')[
                ['user_id', 'count', 'mean_hours', 'max_hours']
            ].to_dict('records')
        }
        
        return queue_analysis
    
    def _analyze_heavy_users(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析重度用户特征"""
        logger.info("分析重度用户特征...")
        
        heavy_user_threshold = self.config.get('statistics', {}).get('user_analysis', {}).get('heavy_user_threshold', 0.8)
        
        if 'duration' not in df.columns or 'actual_node_count' not in df.columns:
            return {}
        
        # 计算用户资源消耗
        df_valid = df[(df['duration'] > 0) & (df['actual_node_count'] > 0)].copy()
        df_valid['node_hours'] = df_valid['duration'] * df_valid['actual_node_count'] / 3600
        
        user_consumption = df_valid.groupby('user_id')['node_hours'].sum().sort_values(ascending=False)
        total_consumption = user_consumption.sum()

        # 检查总消耗是否为零
        if total_consumption == 0:
            logger.warning("总资源消耗为零，无法分析重度用户")
            return {
                'heavy_users_count': 0,
                'heavy_users_resource_share': 0.0,
                'total_consumption_zero': True
            }

        # 找出消耗80%资源的用户
        cumulative_consumption = user_consumption.cumsum()
        heavy_users = cumulative_consumption[cumulative_consumption <= total_consumption * heavy_user_threshold].index
        
        heavy_user_analysis = {}
        
        for user in heavy_users[:10]:  # 分析前10个重度用户
            user_data = df[df['user_id'] == user]
            
            # 安全计算资源百分比
            resource_pct = float(user_consumption[user] / total_consumption * 100) if total_consumption > 0 else 0.0

            # 安全计算平均作业持续时间
            avg_duration = 0.0
            if 'duration' in df.columns and len(user_data) > 0:
                duration_mean = user_data['duration'].mean()
                if pd.notna(duration_mean) and duration_mean > 0:
                    avg_duration = float(duration_mean / 3600)

            analysis = {
                'total_jobs': len(user_data),
                'total_node_hours': float(user_consumption[user]),
                'resource_percentage': resource_pct,
                'avg_job_duration_hours': avg_duration,
                'preferred_subclusters': user_data['primary_subcluster'].value_counts().head(3).to_dict() if 'primary_subcluster' in df.columns else {},
                'job_types': user_data['job_type'].value_counts().to_dict() if 'job_type' in df.columns else {}
            }
            
            heavy_user_analysis[user] = analysis
        
        return {
            'heavy_users_count': len(heavy_users),
            'heavy_users_resource_share': float(heavy_user_threshold * 100),
            'detailed_analysis': heavy_user_analysis
        }
    
    def _analyze_user_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析用户公平性 (对应Helios的Implication #7)"""
        logger.info("分析用户公平性...")
        
        fairness_metrics = {}
        
        # 基尼系数计算 (资源分配不平等程度)
        if 'duration' in df.columns and 'actual_node_count' in df.columns:
            df_valid = df[(df['duration'] > 0) & (df['actual_node_count'] > 0)].copy()
            df_valid['node_hours'] = df_valid['duration'] * df_valid['actual_node_count'] / 3600
            
            user_resources = df_valid.groupby('user_id')['node_hours'].sum().sort_values()
            gini_coefficient = self._calculate_gini_coefficient(user_resources.values)
            
            fairness_metrics['resource_fairness'] = {
                'gini_coefficient': float(gini_coefficient),
                'interpretation': 'Higher values indicate more inequality (0=perfect equality, 1=perfect inequality)'
            }
        
        # 排队时间公平性
        if 'queue_time' in df.columns:
            user_queue_times = df[df['queue_time'] >= 0].groupby('user_id')['queue_time'].mean()

            if len(user_queue_times) > 0:
                queue_gini = self._calculate_gini_coefficient(user_queue_times.values)

                # 安全计算平均值和标准差
                mean_queue = user_queue_times.mean()
                std_queue = user_queue_times.std()

                fairness_metrics['queue_fairness'] = {
                    'gini_coefficient': float(queue_gini),
                    'mean_queue_time_hours': float(mean_queue / 3600) if pd.notna(mean_queue) else 0.0,
                    'queue_time_std_hours': float(std_queue / 3600) if pd.notna(std_queue) else 0.0,
                    'users_analyzed': len(user_queue_times)
                }
            else:
                fairness_metrics['queue_fairness'] = {
                    'gini_coefficient': 0.0,
                    'mean_queue_time_hours': 0.0,
                    'queue_time_std_hours': 0.0,
                    'users_analyzed': 0
                }
        
        return fairness_metrics
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """计算基尼系数"""
        if len(values) == 0:
            return 0.0
        
        # 排序
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # 计算基尼系数
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))  # 确保在[0,1]范围内

    def _generate_helios_visualizations(self, cluster_user: pd.DataFrame, results: Dict[str, Any]):
        """生成Helios风格的用户特征可视化图表 - 严格按照Helios Figure 5和6"""
        logger.info("生成Helios风格的用户特征可视化图表...")

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

        # 创建Figure 5：用户资源消耗CDF
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # 1. 用户GPU时间CDF（左图 - Figure 5a）
        if 'user_gpu_time_cdf' in results:
            self._plot_user_gpu_time_cdf_helios(ax1, results['user_gpu_time_cdf'])

        # 2. 用户CPU时间CDF（右图 - Figure 5b）
        if 'user_cpu_time_cdf' in results:
            self._plot_user_cpu_time_cdf_helios(ax2, results['user_cpu_time_cdf'])

        # 保存Figure 5
        output_path1 = self.output_paths['user_characterization'] / "user_resource_cdf_helios.png"
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        plt.close()

        # 创建Figure 6：用户GPU排队体验
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # 3. 用户GPU排队延迟CDF（左图 - Figure 6a）
        if 'user_gpu_pend_cdf' in results:
            self._plot_user_gpu_pend_cdf_helios(ax3, results['user_gpu_pend_cdf'])

        # 4. 用户GPU完成率分布（右图 - Figure 6b）
        if 'user_gpu_completion_rate' in results:
            self._plot_user_completion_rate_helios(ax4, results['user_gpu_completion_rate'])

        # 保存Figure 6
        output_path2 = self.output_paths['user_characterization'] / "user_gpu_behavior_helios.png"
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        plt.close()

        # 创建Figure 7：用户CPU排队体验（新增）
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # 5. 用户CPU排队延迟CDF（左图）
        if 'user_cpu_pend_cdf' in results:
            self._plot_user_cpu_pend_cdf_helios(ax5, results['user_cpu_pend_cdf'])

        # 6. 用户CPU完成率分布（右图）
        if 'user_cpu_completion_rate' in results:
            self._plot_user_cpu_completion_rate_helios(ax6, results['user_cpu_completion_rate'])

        # 保存Figure 7
        output_path3 = self.output_paths['user_characterization'] / "user_cpu_behavior_helios.png"
        plt.savefig(output_path3, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Helios风格用户特征图表已保存:")
        logger.info(f"  - 资源消耗CDF: {output_path1}")
        logger.info(f"  - GPU行为模式: {output_path2}")
        logger.info(f"  - CPU行为模式: {output_path3}")

    def _plot_user_gpu_time_cdf_helios(self, ax, gpu_time_data: Dict[str, Any]):
        """绘制用户GPU时间CDF - Helios风格（Figure 5a）"""
        if 'user_percentages' not in gpu_time_data or 'cumulative_gpu_percentages' not in gpu_time_data:
            return

        user_percentages = gpu_time_data['user_percentages']
        cumulative_gpu_percentages = gpu_time_data['cumulative_gpu_percentages']

        # 绘制CDF线图
        ax.plot(user_percentages, cumulative_gpu_percentages,
               linestyle='-', linewidth=2.5, color='#1f77b4')

        # 设置标签和格式
        ax.set_xlabel('Percentage of Users (%)')
        ax.set_ylabel('Percentage of GPU Time (%)')
        ax.set_title('(a) User GPU Time CDF')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置坐标轴范围
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def _plot_user_cpu_time_cdf_helios(self, ax, cpu_time_data: Dict[str, Any]):
        """绘制用户CPU时间CDF - Helios风格（Figure 5b）"""
        if 'user_percentages' not in cpu_time_data or 'cumulative_cpu_percentages' not in cpu_time_data:
            return

        user_percentages = cpu_time_data['user_percentages']
        cumulative_cpu_percentages = cpu_time_data['cumulative_cpu_percentages']

        # 绘制CDF线图
        ax.plot(user_percentages, cumulative_cpu_percentages,
               linestyle='-', linewidth=2.5, color='#ff7f0e')

        # 设置标签和格式
        ax.set_xlabel('Percentage of Users (%)')
        ax.set_ylabel('Percentage of CPU Time (%)')
        ax.set_title('(b) User CPU Time CDF')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置坐标轴范围
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    def _plot_user_gpu_pend_cdf_helios(self, ax, pend_data: Dict[str, Any]):
        """绘制用户GPU排队延迟CDF - Helios风格（Figure 6a）"""
        if 'gpu_pend_times' not in pend_data or 'cdf_values' not in pend_data:
            return

        gpu_pend_times = pend_data['gpu_pend_times']
        cdf_values = pend_data['cdf_values']

        # 绘制CDF线图
        ax.plot(gpu_pend_times, cdf_values,
               linestyle='-', linewidth=2.5, color='#2ca02c')

        # 设置标签和格式
        ax.set_xlabel('GPU Pending Time (hours)')
        ax.set_ylabel('CDF (%)')
        ax.set_title('(a) User GPU Pending Time CDF')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴为对数刻度（如果需要）
        if max(gpu_pend_times) > 100:
            ax.set_xscale('log')

    def _plot_user_completion_rate_helios(self, ax, completion_data: Dict[str, Any]):
        """绘制用户GPU完成率分布 - Helios风格（Figure 6b）"""
        if 'bin_edges' not in completion_data or 'hist_percentages' not in completion_data:
            return

        bin_edges = completion_data['bin_edges']
        hist_percentages = completion_data['hist_percentages']

        # 绘制直方图
        ax.bar(bin_edges, hist_percentages, width=8, alpha=0.8,
              color='#d62728', edgecolor='black', linewidth=0.5)

        # 设置标签和格式
        ax.set_xlabel('GPU Job Completion Rate (%)')
        ax.set_ylabel('Percentage of Users (%)')
        ax.set_title('(b) User GPU Completion Rate Distribution')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置坐标轴范围
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))

    def _plot_user_cpu_pend_cdf_helios(self, ax, cpu_pend_data: Dict[str, Any]):
        """绘制用户CPU排队延迟CDF - 对应GPU分析风格"""
        if 'cpu_pend_times' not in cpu_pend_data or 'cdf_values' not in cpu_pend_data:
            return

        cpu_pend_times = cpu_pend_data['cpu_pend_times']
        cdf_values = cpu_pend_data['cdf_values']

        # 绘制CDF线图
        ax.plot(cpu_pend_times, cdf_values, linestyle='-', linewidth=2.5,
               color='#2ca02c', alpha=0.8)

        # 设置标签和格式
        ax.set_xlabel('CPU Pending Time (hours)')
        ax.set_ylabel('CDF')
        ax.set_title('(a) User CPU Pending Time CDF')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置x轴为对数刻度（如果需要）
        if max(cpu_pend_times) > 100:
            ax.set_xscale('log')

    def _plot_user_cpu_completion_rate_helios(self, ax, completion_data: Dict[str, Any]):
        """绘制用户CPU完成率分布 - 对应GPU分析风格"""
        if 'bin_edges' not in completion_data or 'hist_percentages' not in completion_data:
            return

        bin_edges = completion_data['bin_edges']
        hist_percentages = completion_data['hist_percentages']

        # 绘制直方图
        ax.bar(bin_edges, hist_percentages, width=8, alpha=0.8,
              color='#2ca02c', edgecolor='black', linewidth=0.5)

        # 设置标签和格式
        ax.set_xlabel('CPU Job Completion Rate (%)')
        ax.set_ylabel('Percentage of Users (%)')
        ax.set_title('(b) User CPU Completion Rate Distribution')

        # 设置网格
        ax.grid(True, linestyle=':', alpha=0.3)

        # 设置坐标轴范围
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))

    def _generate_visualizations(self, df: pd.DataFrame, results: Dict[str, Any]):
        """生成用户行为可视化图表 (Helios风格)"""
        logger.info("生成用户行为可视化图表...")

        # 1. 用户资源消耗CDF图 (类似Helios的user resource CDF)
        if 'user_resource_consumption' in results:
            resource_data = results['user_resource_consumption']
            if 'user_resource_distribution' in resource_data:
                user_resources = pd.Series(resource_data['user_resource_distribution'])
                
                self.visualizer.plot_cdf(
                    user_resources,
                    'User Resource Consumption Distribution (CDF)',
                    'Node Hours Consumed',
                    self.output_paths['figures'] / 'user_resource_consumption_cdf',
                    log_scale=True
                )
        
        # 2. 用户作业数量CDF图
        if 'user_resource_consumption' in results:
            resource_data = results['user_resource_consumption']
            if 'user_job_distribution' in resource_data:
                user_jobs = pd.Series(resource_data['user_job_distribution'])
                
                self.visualizer.plot_cdf(
                    user_jobs,
                    'User Job Count Distribution (CDF)',
                    'Number of Jobs Submitted',
                    self.output_paths['figures'] / 'user_job_count_cdf',
                    log_scale=True
                )
        
        # 3. 重度用户资源占比图
        if 'heavy_users_analysis' in results:
            heavy_analysis = results['heavy_users_analysis']
            if 'detailed_analysis' in heavy_analysis:
                top_users = heavy_analysis['detailed_analysis']
                user_percentages = {user: data['resource_percentage'] 
                                  for user, data in list(top_users.items())[:10]}
                
                if user_percentages:
                    self.visualizer.plot_bar_chart(
                        user_percentages,
                        'Top 10 Users Resource Consumption',
                        'User ID', 'Resource Percentage (%)',
                        self.output_paths['figures'] / 'top_users_resource_share'
                    )
        
        # 4. 用户排队时间分布
        if 'user_queue_experience' in results:
            queue_data = results['user_queue_experience']
            if 'top_queue_sufferers' in queue_data:
                queue_sufferers = queue_data['top_queue_sufferers']
                if queue_sufferers:
                    queue_times = {record['user_id']: record['mean_hours'] 
                                 for record in queue_sufferers[:10]}
                    
                    self.visualizer.plot_bar_chart(
                        queue_times,
                        'Top 10 Users with Longest Queue Times',
                        'User ID', 'Average Queue Time (Hours)',
                        self.output_paths['figures'] / 'top_queue_sufferers'
                    )
