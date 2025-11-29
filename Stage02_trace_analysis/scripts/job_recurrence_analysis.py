#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
作业重复性分析 (Job Recurrence Analysis)
分析HPC集群中作业的重复性特征，用于workload characterization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import Counter, defaultdict

# 设置matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class JobRecurrenceAnalyzer:
    """作业重复性分析器"""
    
    def __init__(self, data_path, output_dir):
        """
        初始化分析器
        
        Args:
            data_path: 数据文件路径（CSV格式）
            output_dir: 输出目录
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        print(f"加载数据: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"数据加载完成: {len(self.df)} 条作业记录")
        
        # 结果存储
        self.results = {}
    
    def analyze_job_name_recurrence(self):
        """分析基于作业名称的重复性"""
        print("\n=== 分析作业名称重复性 ===")
        
        # 统计每个作业名称的出现次数
        job_name_counts = self.df['job_name'].value_counts()
        
        # 计算重复作业的比例
        total_jobs = len(self.df)
        unique_job_names = len(job_name_counts)
        
        # 统计重复次数分布
        recurrence_dist = job_name_counts.value_counts().sort_index()
        
        # 计算重复至少N次的作业比例
        repeat_thresholds = [2, 5, 10, 20, 50, 100]
        repeat_stats = {}
        for threshold in repeat_thresholds:
            jobs_above_threshold = (job_name_counts >= threshold).sum()
            submissions_above_threshold = job_name_counts[job_name_counts >= threshold].sum()
            repeat_stats[f'repeat_{threshold}+'] = {
                'unique_jobs': jobs_above_threshold,
                'unique_jobs_pct': jobs_above_threshold / unique_job_names * 100,
                'total_submissions': submissions_above_threshold,
                'total_submissions_pct': submissions_above_threshold / total_jobs * 100
            }
        
        self.results['job_name_recurrence'] = {
            'total_jobs': total_jobs,
            'unique_job_names': unique_job_names,
            'avg_submissions_per_job': total_jobs / unique_job_names,
            'recurrence_distribution': recurrence_dist.to_dict(),
            'repeat_statistics': repeat_stats,
            'top_10_jobs': job_name_counts.head(10).to_dict()
        }
        
        print(f"总作业数: {total_jobs}")
        print(f"唯一作业名称数: {unique_job_names}")
        print(f"平均每个作业提交次数: {total_jobs / unique_job_names:.2f}")
        print(f"重复5次以上的作业占比: {repeat_stats['repeat_5+']['total_submissions_pct']:.2f}%")
        
        return job_name_counts, recurrence_dist
    
    def analyze_user_recurrence(self):
        """分析基于用户的重复性"""
        print("\n=== 分析用户作业重复性 ===")
        
        # 每个用户的作业提交次数
        user_job_counts = self.df.groupby('user_name').size()
        
        # 每个用户的唯一作业名称数
        user_unique_jobs = self.df.groupby('user_name')['job_name'].nunique()
        
        # 计算每个用户的平均重复次数
        user_repeat_ratio = user_job_counts / user_unique_jobs
        
        self.results['user_recurrence'] = {
            'total_users': len(user_job_counts),
            'avg_jobs_per_user': user_job_counts.mean(),
            'avg_unique_jobs_per_user': user_unique_jobs.mean(),
            'avg_repeat_ratio': user_repeat_ratio.mean(),
            'top_10_users_by_submissions': user_job_counts.nlargest(10).to_dict(),
            'top_10_users_by_repeat_ratio': user_repeat_ratio.nlargest(10).to_dict()
        }
        
        print(f"总用户数: {len(user_job_counts)}")
        print(f"平均每用户提交作业数: {user_job_counts.mean():.2f}")
        print(f"平均每用户唯一作业数: {user_unique_jobs.mean():.2f}")
        print(f"平均重复比例: {user_repeat_ratio.mean():.2f}")
        
        return user_job_counts, user_unique_jobs, user_repeat_ratio
    
    def analyze_resource_recurrence(self):
        """分析基于资源请求的重复性"""
        print("\n=== 分析资源配置重复性 ===")
        
        # 创建资源配置签名（CPU, GPU, Memory）
        self.df['resource_signature'] = (
            self.df['num_cpu'].astype(str) + '_' +
            self.df['num_gpu'].astype(str) + '_' +
            self.df['req_mem'].astype(str)
        )
        
        # 统计每种资源配置的出现次数
        resource_counts = self.df['resource_signature'].value_counts()
        
        # 计算资源配置的多样性
        total_jobs = len(self.df)
        unique_configs = len(resource_counts)
        
        self.results['resource_recurrence'] = {
            'total_jobs': total_jobs,
            'unique_resource_configs': unique_configs,
            'avg_jobs_per_config': total_jobs / unique_configs,
            'top_10_configs': resource_counts.head(10).to_dict()
        }
        
        print(f"唯一资源配置数: {unique_configs}")
        print(f"平均每种配置的作业数: {total_jobs / unique_configs:.2f}")
        
        return resource_counts
    
    def generate_visualizations(self, job_name_counts, recurrence_dist, 
                                user_job_counts, user_repeat_ratio, resource_counts):
        """生成可视化图表"""
        print("\n=== 生成可视化图表 ===")
        
        # 图表1: 作业重复次数分布（CDF）
        # 图表2: 用户重复性分析
        # 图表3: 资源配置重复性
        # 图表4: Top重复作业
        
        # TODO: 实现可视化逻辑
        pass
    
    def save_results(self):
        """保存分析结果"""
        print("\n=== 保存分析结果 ===")
        
        # 保存JSON结果
        output_file = self.output_dir / 'job_recurrence_analysis_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存: {output_file}")
    
    def run_analysis(self):
        """运行完整分析"""
        print("=" * 80)
        print("开始作业重复性分析")
        print("=" * 80)
        
        # 1. 作业名称重复性
        job_name_counts, recurrence_dist = self.analyze_job_name_recurrence()
        
        # 2. 用户重复性
        user_job_counts, user_unique_jobs, user_repeat_ratio = self.analyze_user_recurrence()
        
        # 3. 资源配置重复性
        resource_counts = self.analyze_resource_recurrence()
        
        # 4. 生成可视化
        # self.generate_visualizations(...)
        
        # 5. 保存结果
        self.save_results()
        
        print("\n" + "=" * 80)
        print("作业重复性分析完成")
        print("=" * 80)

def main():
    """主函数"""
    # 数据路径
    data_path = "../data/helios_jobs_cleaned.csv"
    output_dir = "../output/job_recurrence_analysis"
    
    # 创建分析器并运行
    analyzer = JobRecurrenceAnalyzer(data_path, output_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()

