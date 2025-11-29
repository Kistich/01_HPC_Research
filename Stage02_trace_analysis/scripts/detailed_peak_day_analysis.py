#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细分析2024-04-14峰值日的作业提交情况
重点分析用户行为、作业重复性和资源利用效率
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime, timedelta

# 设置中文字体和日志
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedPeakDayAnalyzer:
    """详细峰值日分析器"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.peak_day_data = None

        # 使用相对路径
        script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
        self.output_dir = script_dir / "output" / "peak_day_detailed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_peak_day_data(self):
        """加载2024-04-14的数据"""
        logger.info("加载峰值日数据...")
        
        # 分块读取数据以节省内存
        chunk_size = 100000
        peak_day_chunks = []
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            # 转换时间字段
            chunk['submit_time'] = pd.to_datetime(chunk['submit_time'])
            chunk['submit_date'] = chunk['submit_time'].dt.date
            
            # 筛选峰值日数据
            peak_day_chunk = chunk[chunk['submit_date'] == pd.to_datetime('2024-04-14').date()]
            
            if len(peak_day_chunk) > 0:
                peak_day_chunks.append(peak_day_chunk)
        
        self.peak_day_data = pd.concat(peak_day_chunks, ignore_index=True)
        
        # 计算持续时间
        if 'start_time' in self.peak_day_data.columns and 'end_time' in self.peak_day_data.columns:
            self.peak_day_data['start_time'] = pd.to_datetime(self.peak_day_data['start_time'])
            self.peak_day_data['end_time'] = pd.to_datetime(self.peak_day_data['end_time'])
            self.peak_day_data['duration'] = (self.peak_day_data['end_time'] - self.peak_day_data['start_time']).dt.total_seconds()
        
        logger.info(f"峰值日数据加载完成，共{len(self.peak_day_data)}条记录")
        
    def analyze_user_behavior(self):
        """详细分析用户行为"""
        logger.info("分析用户行为...")
        
        if self.peak_day_data is None:
            self.load_peak_day_data()
        
        # 用户作业统计
        user_stats = self.peak_day_data.groupby('user_id').agg({
            'job_id': 'count',
            'duration': ['mean', 'median', 'std', 'min', 'max'],
            'gpu_num': ['mean', 'sum'],
            'num_processors': ['mean', 'sum'],
            'submit_time': ['min', 'max']
        }).round(2)

        user_stats.columns = ['job_count', 'avg_duration', 'median_duration', 'std_duration',
                             'min_duration', 'max_duration', 'avg_gpu', 'total_gpu',
                             'avg_cpu', 'total_cpu', 'first_submit', 'last_submit']
        
        # 计算提交时间跨度
        user_stats['submit_span_hours'] = (user_stats['last_submit'] - user_stats['first_submit']).dt.total_seconds() / 3600
        
        # 按作业数量排序
        user_stats = user_stats.sort_values('job_count', ascending=False)
        
        # 保存用户统计
        user_stats.to_csv(self.output_dir / 'user_behavior_stats.csv')
        
        # 分析异常用户
        high_volume_users = user_stats[user_stats['job_count'] > 1000]
        
        print(f"\n=== 用户行为分析 ===")
        print(f"总用户数: {len(user_stats)}")
        print(f"高频用户(>1000作业): {len(high_volume_users)}")
        print(f"\n前10名用户作业数:")
        print(user_stats[['job_count', 'avg_duration', 'submit_span_hours']].head(10))
        
        return user_stats
    
    def analyze_job_patterns(self):
        """分析作业模式"""
        logger.info("分析作业模式...")
        
        # 1. 持续时间分布分析
        valid_duration = self.peak_day_data[self.peak_day_data['duration'] > 0]['duration']
        
        # 定义更细粒度的持续时间区间
        duration_bins = [0, 1, 2, 5, 10, 30, 60, 300, 1800, 3600, 7200, 21600, 86400, float('inf')]
        duration_labels = ['<1s', '1-2s', '2-5s', '5-10s', '10-30s', '30s-1min', 
                          '1-5min', '5-30min', '30min-1h', '1-2h', '2-6h', '6-24h', '>24h']
        
        duration_categories = pd.cut(valid_duration, bins=duration_bins, labels=duration_labels)
        duration_dist = duration_categories.value_counts().sort_index()
        
        print(f"\n=== 作业持续时间分布 ===")
        for label, count in duration_dist.items():
            percentage = count / len(valid_duration) * 100
            print(f"{label}: {count:,} ({percentage:.1f}%)")
        
        # 2. 资源请求模式
        resource_patterns = self.peak_day_data.groupby(['num_processors', 'gpu_num']).size().sort_values(ascending=False)
        
        print(f"\n=== 前10种资源请求模式 ===")
        for (cpu, gpu), count in resource_patterns.head(10).items():
            percentage = count / len(self.peak_day_data) * 100
            print(f"CPU:{cpu}, GPU:{gpu} -> {count:,} ({percentage:.1f}%)")
        
        # 3. 作业状态分析
        if 'job_status_str' in self.peak_day_data.columns:
            status_dist = self.peak_day_data['job_status_str'].value_counts()
            print(f"\n=== 作业状态分布 ===")
            for status, count in status_dist.items():
                percentage = count / len(self.peak_day_data) * 100
                print(f"{status}: {count:,} ({percentage:.1f}%)")
        
        return {
            'duration_distribution': duration_dist,
            'resource_patterns': resource_patterns,
            'status_distribution': status_dist if 'job_status_str' in self.peak_day_data.columns else None
        }
    
    def analyze_temporal_patterns(self):
        """分析时间模式"""
        logger.info("分析时间提交模式...")
        
        # 按小时分析
        self.peak_day_data['submit_hour'] = self.peak_day_data['submit_time'].dt.hour
        hourly_counts = self.peak_day_data['submit_hour'].value_counts().sort_index()
        
        # 按分钟分析（找出异常集中的时间点）
        self.peak_day_data['submit_minute'] = self.peak_day_data['submit_time'].dt.floor('min')
        minute_counts = self.peak_day_data['submit_minute'].value_counts().sort_values(ascending=False)
        
        print(f"\n=== 时间分布分析 ===")
        print(f"提交最集中的小时: {hourly_counts.idxmax()}点 ({hourly_counts.max():,}个作业)")
        print(f"提交最集中的分钟: {minute_counts.index[0]} ({minute_counts.iloc[0]:,}个作业)")
        
        # 找出异常高频的分钟
        high_freq_minutes = minute_counts[minute_counts > 1000]
        print(f"\n单分钟提交>1000个作业的时间点: {len(high_freq_minutes)}个")
        
        if len(high_freq_minutes) > 0:
            print("前5个高频时间点:")
            for time_point, count in high_freq_minutes.head().items():
                print(f"  {time_point}: {count:,}个作业")
        
        return {
            'hourly_distribution': hourly_counts,
            'minute_distribution': minute_counts,
            'high_frequency_minutes': high_freq_minutes
        }
    
    def identify_duplicate_jobs(self):
        """识别重复作业"""
        logger.info("识别重复作业...")
        
        # 基于多个维度识别可能的重复作业
        duplicate_criteria = [
            ['user_id', 'num_processors', 'gpu_num'],  # 相同用户相同资源
            ['user_id', 'num_processors', 'gpu_num', 'duration'],  # 相同用户相同资源相同时长
        ]
        
        duplicate_analysis = {}
        
        for i, criteria in enumerate(duplicate_criteria):
            # 只考虑有效字段
            valid_criteria = [col for col in criteria if col in self.peak_day_data.columns]
            
            if len(valid_criteria) > 0:
                grouped = self.peak_day_data.groupby(valid_criteria).size()
                duplicates = grouped[grouped > 1].sort_values(ascending=False)
                
                duplicate_analysis[f'criteria_{i+1}'] = {
                    'criteria': valid_criteria,
                    'duplicate_groups': len(duplicates),
                    'total_duplicate_jobs': duplicates.sum(),
                    'top_duplicates': duplicates.head(10).to_dict()
                }
                
                print(f"\n=== 重复作业分析 (标准{i+1}: {valid_criteria}) ===")
                print(f"重复组数: {len(duplicates)}")
                print(f"涉及作业总数: {duplicates.sum():,}")
                
                if len(duplicates) > 0:
                    print("前5个重复最多的组:")
                    for group, count in duplicates.head().items():
                        print(f"  {group}: {count}个作业")
        
        return duplicate_analysis
    
    def calculate_resource_efficiency(self):
        """计算资源利用效率"""
        logger.info("计算资源利用效率...")
        
        # 计算有效作业（持续时间>10秒）
        effective_jobs = self.peak_day_data[self.peak_day_data['duration'] > 10]
        ineffective_jobs = self.peak_day_data[self.peak_day_data['duration'] <= 10]
        
        # 计算资源浪费
        total_cpu_hours = (self.peak_day_data['duration'] * self.peak_day_data['num_processors']).sum() / 3600
        wasted_cpu_hours = (ineffective_jobs['duration'] * ineffective_jobs['num_processors']).sum() / 3600
        
        total_gpu_hours = (self.peak_day_data['duration'] * self.peak_day_data['gpu_num']).sum() / 3600
        wasted_gpu_hours = (ineffective_jobs['duration'] * ineffective_jobs['gpu_num']).sum() / 3600
        
        efficiency_stats = {
            'total_jobs': len(self.peak_day_data),
            'effective_jobs': len(effective_jobs),
            'ineffective_jobs': len(ineffective_jobs),
            'effectiveness_rate': len(effective_jobs) / len(self.peak_day_data) * 100,
            'total_cpu_hours': total_cpu_hours,
            'wasted_cpu_hours': wasted_cpu_hours,
            'cpu_waste_rate': wasted_cpu_hours / total_cpu_hours * 100 if total_cpu_hours > 0 else 0,
            'total_gpu_hours': total_gpu_hours,
            'wasted_gpu_hours': wasted_gpu_hours,
            'gpu_waste_rate': wasted_gpu_hours / total_gpu_hours * 100 if total_gpu_hours > 0 else 0
        }
        
        print(f"\n=== 资源利用效率分析 ===")
        print(f"总作业数: {efficiency_stats['total_jobs']:,}")
        print(f"有效作业数(>10s): {efficiency_stats['effective_jobs']:,}")
        print(f"无效作业数(≤10s): {efficiency_stats['ineffective_jobs']:,}")
        print(f"作业有效率: {efficiency_stats['effectiveness_rate']:.1f}%")
        print(f"CPU资源浪费率: {efficiency_stats['cpu_waste_rate']:.1f}%")
        print(f"GPU资源浪费率: {efficiency_stats['gpu_waste_rate']:.1f}%")
        
        return efficiency_stats
    
    def generate_cleaning_recommendations(self):
        """生成数据清洗建议"""
        logger.info("生成数据清洗建议...")
        
        recommendations = []
        
        # 基于持续时间的过滤建议
        short_jobs = len(self.peak_day_data[self.peak_day_data['duration'] <= 10])
        if short_jobs > 0:
            recommendations.append({
                'type': '超短时长过滤',
                'description': f'过滤持续时间≤10秒的作业',
                'affected_jobs': short_jobs,
                'percentage': short_jobs / len(self.peak_day_data) * 100,
                'filter_condition': 'duration > 10'
            })
        
        # 基于用户行为的过滤建议
        user_job_counts = self.peak_day_data['user_id'].value_counts()
        high_volume_users = user_job_counts[user_job_counts > 10000]
        
        if len(high_volume_users) > 0:
            affected_jobs = high_volume_users.sum()
            recommendations.append({
                'type': '异常用户过滤',
                'description': f'限制单用户单日作业数量上限',
                'affected_jobs': affected_jobs,
                'percentage': affected_jobs / len(self.peak_day_data) * 100,
                'filter_condition': '单用户单日作业数 < 1000'
            })
        
        # 基于资源请求的过滤建议
        zero_resource_jobs = len(self.peak_day_data[(self.peak_day_data['num_processors'] == 0) & (self.peak_day_data['gpu_num'] == 0)])
        if zero_resource_jobs > 0:
            recommendations.append({
                'type': '零资源作业过滤',
                'description': f'过滤CPU和GPU都为0的作业',
                'affected_jobs': zero_resource_jobs,
                'percentage': zero_resource_jobs / len(self.peak_day_data) * 100,
                'filter_condition': 'num_processors > 0 OR gpu_num > 0'
            })
        
        print(f"\n=== 数据清洗建议 ===")
        total_removable = 0
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['type']}")
            print(f"   描述: {rec['description']}")
            print(f"   影响作业数: {rec['affected_jobs']:,} ({rec['percentage']:.1f}%)")
            print(f"   过滤条件: {rec['filter_condition']}")
            total_removable += rec['affected_jobs']
            print()
        
        print(f"总计可移除作业数: {total_removable:,} ({total_removable/len(self.peak_day_data)*100:.1f}%)")
        print(f"清洗后剩余作业数: {len(self.peak_day_data) - total_removable:,}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """运行完整分析"""
        logger.info("开始完整分析...")
        
        # 加载数据
        self.load_peak_day_data()
        
        # 执行各项分析
        user_stats = self.analyze_user_behavior()
        job_patterns = self.analyze_job_patterns()
        temporal_patterns = self.analyze_temporal_patterns()
        duplicate_analysis = self.identify_duplicate_jobs()
        efficiency_stats = self.calculate_resource_efficiency()
        recommendations = self.generate_cleaning_recommendations()
        
        # 保存分析结果
        results = {
            'user_behavior': user_stats,
            'job_patterns': job_patterns,
            'temporal_patterns': temporal_patterns,
            'duplicate_analysis': duplicate_analysis,
            'efficiency_stats': efficiency_stats,
            'recommendations': recommendations
        }
        
        # 保存到文件
        import json
        with open(self.output_dir / 'complete_analysis_results.json', 'w', encoding='utf-8') as f:
            # 转换不可序列化的对象
            serializable_results = {}
            for key, value in results.items():
                if key == 'user_behavior':
                    continue  # 已经保存为CSV
                elif isinstance(value, dict):
                    serializable_results[key] = {k: str(v) if hasattr(v, 'to_dict') else v for k, v in value.items()}
                else:
                    serializable_results[key] = str(value)
            
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"完整分析完成，结果保存在: {self.output_dir}")
        
        return results


def main():
    """主函数"""
    # 使用相对路径
    script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
    project_root = script_dir.parent  # 01_HPC_Research/
    data_path = str(project_root / "Stage01_data_filter_preprocess" / "full_processing_outputs" / "stage6_data_standardization" / "standardized_data.csv")

    analyzer = DetailedPeakDayAnalyzer(data_path)
    results = analyzer.run_complete_analysis()
    
    print(f"\n=== 分析完成 ===")
    print(f"详细结果保存在: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
