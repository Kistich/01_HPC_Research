#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析2024年4月异常峰值日的作业提交情况
识别重复作业、短时长作业和异常提交模式
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PeakDayAnalyzer:
    """异常峰值日分析器"""
    
    def __init__(self, data_path: str):
        """
        初始化分析器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = Path(data_path)
        self.df = None
        self.peak_day = None
        self.peak_day_data = None
        
    def load_data(self):
        """加载数据"""
        logger.info(f"加载数据: {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"数据加载成功，共{len(self.df)}条记录")
            
            # 转换时间字段
            time_columns = ['submit_time', 'start_time', 'end_time']
            for col in time_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col])
            
            # 计算持续时间
            if 'start_time' in self.df.columns and 'end_time' in self.df.columns:
                self.df['duration'] = (self.df['end_time'] - self.df['start_time']).dt.total_seconds()
            
            # 添加日期字段
            if 'submit_time' in self.df.columns:
                self.df['submit_date'] = self.df['submit_time'].dt.date
                
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def find_peak_day(self):
        """找到作业提交数量最多的日期"""
        logger.info("查找峰值日...")
        
        if self.df is None:
            self.load_data()
        
        # 按日期统计作业数量
        daily_counts = self.df.groupby('submit_date').size().sort_values(ascending=False)
        
        self.peak_day = daily_counts.index[0]
        peak_count = daily_counts.iloc[0]
        
        logger.info(f"峰值日: {self.peak_day}, 作业数量: {peak_count:,}")
        
        # 提取峰值日数据
        self.peak_day_data = self.df[self.df['submit_date'] == self.peak_day].copy()
        
        return self.peak_day, peak_count
    
    def analyze_peak_day_details(self):
        """详细分析峰值日的作业情况"""
        logger.info(f"详细分析峰值日 {self.peak_day} 的作业情况...")
        
        if self.peak_day_data is None:
            self.find_peak_day()
        
        analysis_results = {}
        
        # 1. 基本统计
        total_jobs = len(self.peak_day_data)
        analysis_results['basic_stats'] = {
            'total_jobs': total_jobs,
            'unique_users': self.peak_day_data['user_id'].nunique() if 'user_id' in self.peak_day_data.columns else 0,
            'unique_clusters': self.peak_day_data['cluster_name'].nunique() if 'cluster_name' in self.peak_day_data.columns else 0
        }
        
        # 2. 时间分布分析
        if 'submit_time' in self.peak_day_data.columns:
            self.peak_day_data['submit_hour'] = self.peak_day_data['submit_time'].dt.hour
            hourly_distribution = self.peak_day_data['submit_hour'].value_counts().sort_index()
            
            analysis_results['temporal_distribution'] = {
                'hourly_counts': hourly_distribution.to_dict(),
                'peak_hour': hourly_distribution.idxmax(),
                'peak_hour_count': hourly_distribution.max(),
                'submission_span_hours': hourly_distribution[hourly_distribution > 0].index.max() - 
                                       hourly_distribution[hourly_distribution > 0].index.min() + 1
            }
        
        # 3. 持续时间分析
        if 'duration' in self.peak_day_data.columns:
            valid_duration = self.peak_day_data[self.peak_day_data['duration'] > 0]['duration']
            
            # 定义持续时间区间
            duration_bins = [0, 1, 5, 10, 30, 60, 300, 1800, 3600, float('inf')]
            duration_labels = ['<1s', '1-5s', '5-10s', '10-30s', '30s-1min', '1-5min', '5-30min', '30min-1h', '>1h']
            
            duration_categories = pd.cut(valid_duration, bins=duration_bins, labels=duration_labels)
            duration_distribution = duration_categories.value_counts()
            
            analysis_results['duration_analysis'] = {
                'total_with_duration': len(valid_duration),
                'mean_duration_seconds': float(valid_duration.mean()) if len(valid_duration) > 0 else 0,
                'median_duration_seconds': float(valid_duration.median()) if len(valid_duration) > 0 else 0,
                'duration_distribution': duration_distribution.to_dict(),
                'very_short_jobs_count': len(valid_duration[valid_duration < 10]),  # <10秒
                'very_short_jobs_percentage': len(valid_duration[valid_duration < 10]) / len(valid_duration) * 100 if len(valid_duration) > 0 else 0
            }
        
        # 4. 用户行为分析
        if 'user_id' in self.peak_day_data.columns:
            user_job_counts = self.peak_day_data['user_id'].value_counts()
            
            analysis_results['user_behavior'] = {
                'top_10_users': user_job_counts.head(10).to_dict(),
                'users_with_many_jobs': len(user_job_counts[user_job_counts >= 100]),
                'max_jobs_per_user': user_job_counts.max(),
                'avg_jobs_per_user': float(user_job_counts.mean()),
                'median_jobs_per_user': float(user_job_counts.median())
            }
        
        # 5. 作业类型分析（基于GPU使用）
        if 'gpu_num' in self.peak_day_data.columns:
            gpu_jobs = self.peak_day_data[self.peak_day_data['gpu_num'] > 0]
            cpu_jobs = self.peak_day_data[self.peak_day_data['gpu_num'] == 0]
            
            analysis_results['job_type_analysis'] = {
                'gpu_jobs_count': len(gpu_jobs),
                'cpu_jobs_count': len(cpu_jobs),
                'gpu_job_percentage': len(gpu_jobs) / total_jobs * 100,
                'cpu_job_percentage': len(cpu_jobs) / total_jobs * 100
            }
        
        # 6. 集群分布分析
        if 'cluster_name' in self.peak_day_data.columns:
            cluster_distribution = self.peak_day_data['cluster_name'].value_counts()
            
            analysis_results['cluster_distribution'] = {
                'cluster_counts': cluster_distribution.to_dict(),
                'dominant_cluster': cluster_distribution.index[0],
                'dominant_cluster_percentage': cluster_distribution.iloc[0] / total_jobs * 100
            }
        
        # 7. 作业状态分析
        if 'job_status_str' in self.peak_day_data.columns:
            status_distribution = self.peak_day_data['job_status_str'].value_counts()
            
            analysis_results['status_analysis'] = {
                'status_counts': status_distribution.to_dict(),
                'completion_rate': status_distribution.get('COMPLETED', 0) / total_jobs * 100,
                'failure_rate': status_distribution.get('FAILED', 0) / total_jobs * 100
            }
        
        return analysis_results
    
    def identify_suspicious_patterns(self):
        """识别可疑的提交模式"""
        logger.info("识别可疑的提交模式...")
        
        if self.peak_day_data is None:
            self.find_peak_day()
        
        suspicious_patterns = {}
        
        # 1. 识别重复作业（基于用户、资源请求、提交时间相近）
        if all(col in self.peak_day_data.columns for col in ['user_id', 'gpu_num', 'cpu_num']):
            # 按用户和资源请求分组
            grouped = self.peak_day_data.groupby(['user_id', 'gpu_num', 'cpu_num']).size()
            repeated_submissions = grouped[grouped > 10]  # 相同配置提交超过10次
            
            suspicious_patterns['repeated_submissions'] = {
                'count': len(repeated_submissions),
                'details': repeated_submissions.to_dict(),
                'total_jobs_in_repeated': repeated_submissions.sum()
            }
        
        # 2. 识别短时长作业集中
        if 'duration' in self.peak_day_data.columns:
            very_short_jobs = self.peak_day_data[self.peak_day_data['duration'] < 10]  # <10秒
            
            if len(very_short_jobs) > 0:
                short_job_users = very_short_jobs['user_id'].value_counts() if 'user_id' in very_short_jobs.columns else pd.Series()
                
                suspicious_patterns['short_duration_jobs'] = {
                    'total_short_jobs': len(very_short_jobs),
                    'percentage_of_day': len(very_short_jobs) / len(self.peak_day_data) * 100,
                    'top_short_job_users': short_job_users.head(10).to_dict() if len(short_job_users) > 0 else {}
                }
        
        # 3. 识别时间集中提交
        if 'submit_time' in self.peak_day_data.columns:
            # 按分钟统计提交数量
            self.peak_day_data['submit_minute'] = self.peak_day_data['submit_time'].dt.floor('min')
            minute_counts = self.peak_day_data['submit_minute'].value_counts()
            
            # 找出提交数量异常高的分钟
            high_submission_minutes = minute_counts[minute_counts > minute_counts.quantile(0.95)]
            
            suspicious_patterns['time_concentrated_submissions'] = {
                'high_submission_minutes_count': len(high_submission_minutes),
                'max_submissions_per_minute': minute_counts.max(),
                'avg_submissions_per_minute': float(minute_counts.mean()),
                'top_submission_minutes': high_submission_minutes.head(10).to_dict()
            }
        
        return suspicious_patterns
    
    def generate_recommendations(self, analysis_results, suspicious_patterns):
        """生成数据清洗建议"""
        logger.info("生成数据清洗建议...")
        
        recommendations = []
        
        # 基于持续时间的建议
        if 'duration_analysis' in analysis_results:
            short_job_percentage = analysis_results['duration_analysis']['very_short_jobs_percentage']
            if short_job_percentage > 50:
                recommendations.append({
                    'type': 'duration_filter',
                    'description': f'建议过滤掉持续时间<10秒的作业，占比{short_job_percentage:.1f}%',
                    'filter_condition': 'duration >= 10',
                    'estimated_removal': analysis_results['duration_analysis']['very_short_jobs_count']
                })
        
        # 基于重复提交的建议
        if 'repeated_submissions' in suspicious_patterns:
            repeated_jobs = suspicious_patterns['repeated_submissions']['total_jobs_in_repeated']
            if repeated_jobs > 1000:
                recommendations.append({
                    'type': 'duplicate_filter',
                    'description': f'发现{repeated_jobs}个可能的重复提交作业',
                    'filter_condition': '去重相同用户的相同资源配置作业',
                    'estimated_removal': repeated_jobs
                })
        
        # 基于用户行为的建议
        if 'user_behavior' in analysis_results:
            max_jobs = analysis_results['user_behavior']['max_jobs_per_user']
            if max_jobs > 10000:
                recommendations.append({
                    'type': 'user_limit_filter',
                    'description': f'单个用户最多提交{max_jobs}个作业，可能存在异常',
                    'filter_condition': f'限制单用户单日作业数量<1000',
                    'estimated_removal': 'TBD'
                })
        
        return recommendations
    
    def save_analysis_report(self, output_path: str):
        """保存分析报告"""
        logger.info(f"保存分析报告到: {output_path}")
        
        # 执行完整分析
        peak_day, peak_count = self.find_peak_day()
        analysis_results = self.analyze_peak_day_details()
        suspicious_patterns = self.identify_suspicious_patterns()
        recommendations = self.generate_recommendations(analysis_results, suspicious_patterns)
        
        # 生成报告
        report = f"""
# 峰值日作业提交分析报告

## 基本信息
- **峰值日期**: {peak_day}
- **作业总数**: {peak_count:,}
- **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 详细分析结果

### 1. 基本统计
- 总作业数: {analysis_results.get('basic_stats', {}).get('total_jobs', 0):,}
- 唯一用户数: {analysis_results.get('basic_stats', {}).get('unique_users', 0):,}
- 涉及集群数: {analysis_results.get('basic_stats', {}).get('unique_clusters', 0)}

### 2. 时间分布
- 峰值小时: {analysis_results.get('temporal_distribution', {}).get('peak_hour', 'N/A')}点
- 峰值小时作业数: {analysis_results.get('temporal_distribution', {}).get('peak_hour_count', 0):,}
- 提交时间跨度: {analysis_results.get('temporal_distribution', {}).get('submission_span_hours', 0)}小时

### 3. 持续时间分析
- 平均持续时间: {analysis_results.get('duration_analysis', {}).get('mean_duration_seconds', 0):.1f}秒
- 中位数持续时间: {analysis_results.get('duration_analysis', {}).get('median_duration_seconds', 0):.1f}秒
- 超短作业(<10秒)数量: {analysis_results.get('duration_analysis', {}).get('very_short_jobs_count', 0):,}
- 超短作业占比: {analysis_results.get('duration_analysis', {}).get('very_short_jobs_percentage', 0):.1f}%

### 4. 用户行为
- 单用户最多作业数: {analysis_results.get('user_behavior', {}).get('max_jobs_per_user', 0):,}
- 平均每用户作业数: {analysis_results.get('user_behavior', {}).get('avg_jobs_per_user', 0):.1f}
- 提交>100个作业的用户数: {analysis_results.get('user_behavior', {}).get('users_with_many_jobs', 0)}

### 5. 作业类型
- GPU作业数: {analysis_results.get('job_type_analysis', {}).get('gpu_jobs_count', 0):,}
- CPU作业数: {analysis_results.get('job_type_analysis', {}).get('cpu_jobs_count', 0):,}
- GPU作业占比: {analysis_results.get('job_type_analysis', {}).get('gpu_job_percentage', 0):.1f}%

## 可疑模式识别

### 重复提交
- 重复提交组数: {suspicious_patterns.get('repeated_submissions', {}).get('count', 0)}
- 涉及作业总数: {suspicious_patterns.get('repeated_submissions', {}).get('total_jobs_in_repeated', 0):,}

### 短时长作业
- 超短作业数: {suspicious_patterns.get('short_duration_jobs', {}).get('total_short_jobs', 0):,}
- 占当日比例: {suspicious_patterns.get('short_duration_jobs', {}).get('percentage_of_day', 0):.1f}%

### 时间集中提交
- 高频提交分钟数: {suspicious_patterns.get('time_concentrated_submissions', {}).get('high_submission_minutes_count', 0)}
- 单分钟最大提交数: {suspicious_patterns.get('time_concentrated_submissions', {}).get('max_submissions_per_minute', 0):,}

## 数据清洗建议

"""
        
        for i, rec in enumerate(recommendations, 1):
            report += f"""
### 建议 {i}: {rec['type']}
- **描述**: {rec['description']}
- **过滤条件**: {rec['filter_condition']}
- **预计移除数量**: {rec['estimated_removal']}
"""
        
        # 保存报告
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"分析报告已保存: {output_path}")
        
        return analysis_results, suspicious_patterns, recommendations


def main():
    """主函数"""
    # 使用相对路径
    script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
    project_root = script_dir.parent  # 01_HPC_Research/
    data_path = str(project_root / "Stage01_data_filter_preprocess" / "full_processing_outputs" / "stage6_data_standardization" / "standardized_data.csv")

    # 创建分析器
    analyzer = PeakDayAnalyzer(data_path)

    # 执行分析并保存报告
    output_path = str(script_dir / "output" / "peak_day_analysis_report.md")
    
    try:
        results = analyzer.save_analysis_report(output_path)
        logger.info("峰值日分析完成！")
        
        # 打印关键发现
        peak_day, peak_count = analyzer.find_peak_day()
        print(f"\n=== 关键发现 ===")
        print(f"峰值日期: {peak_day}")
        print(f"作业数量: {peak_count:,}")
        
        if analyzer.peak_day_data is not None:
            short_jobs = len(analyzer.peak_day_data[analyzer.peak_day_data['duration'] < 10]) if 'duration' in analyzer.peak_day_data.columns else 0
            print(f"超短作业(<10秒): {short_jobs:,}")
            print(f"超短作业占比: {short_jobs/peak_count*100:.1f}%")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise


if __name__ == "__main__":
    main()
