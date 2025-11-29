#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析空用户ID记录的特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_null_user_records():
    """分析空用户ID记录"""

    # 使用相对路径
    script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
    project_root = script_dir.parent  # 01_HPC_Research/
    data_path = str(project_root / "Stage01_data_filter_preprocess" / "full_processing_outputs" / "stage6_data_standardization" / "standardized_data.csv")

    logger.info("重新加载数据以分析空用户记录...")
    
    # 分块读取数据
    chunk_size = 100000
    null_user_chunks = []
    valid_user_chunks = []
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        chunk['submit_time'] = pd.to_datetime(chunk['submit_time'])
        chunk['submit_date'] = chunk['submit_time'].dt.date
        
        # 筛选峰值日数据
        peak_day_chunk = chunk[chunk['submit_date'] == pd.to_datetime('2024-04-14').date()]
        
        if len(peak_day_chunk) > 0:
            # 分离空用户和有效用户数据
            null_users = peak_day_chunk[peak_day_chunk['user_id'].isnull()]
            valid_users = peak_day_chunk[peak_day_chunk['user_id'].notna()]
            
            if len(null_users) > 0:
                null_user_chunks.append(null_users)
            if len(valid_users) > 0:
                valid_user_chunks.append(valid_users)
    
    # 合并数据
    null_user_data = pd.concat(null_user_chunks, ignore_index=True) if null_user_chunks else pd.DataFrame()
    valid_user_data = pd.concat(valid_user_chunks, ignore_index=True) if valid_user_chunks else pd.DataFrame()
    
    print(f"\n=== 数据分离结果 ===")
    print(f"空用户ID记录数: {len(null_user_data):,}")
    print(f"有效用户ID记录数: {len(valid_user_data):,}")
    print(f"总记录数: {len(null_user_data) + len(valid_user_data):,}")
    
    if len(null_user_data) > 0:
        print(f"\n=== 空用户ID记录分析 ===")
        
        # 分析作业名称模式
        job_name_counts = null_user_data['job_name'].value_counts()
        print(f"空用户记录的作业名称分布 (前10):")
        for job_name, count in job_name_counts.head(10).items():
            percentage = count / len(null_user_data) * 100
            print(f"  {job_name}: {count:,} ({percentage:.1f}%)")
        
        # 分析作业ID模式
        job_id_counts = null_user_data['job_id'].value_counts()
        print(f"\n空用户记录的作业ID分布 (前10):")
        for job_id, count in job_id_counts.head(10).items():
            percentage = count / len(null_user_data) * 100
            print(f"  {job_id}: {count:,} ({percentage:.1f}%)")
        
        # 分析时间分布
        null_user_data['submit_hour'] = null_user_data['submit_time'].dt.hour
        hourly_counts = null_user_data['submit_hour'].value_counts().sort_index()
        print(f"\n空用户记录的小时分布:")
        for hour, count in hourly_counts.items():
            percentage = count / len(null_user_data) * 100
            print(f"  {hour:02d}点: {count:,} ({percentage:.1f}%)")
        
        # 检查是否有其他标识字段
        print(f"\n=== 空用户记录的其他字段分析 ===")
        
        # 检查集群信息
        if 'cluster_name' in null_user_data.columns:
            cluster_counts = null_user_data['cluster_name'].value_counts()
            print(f"集群分布:")
            for cluster, count in cluster_counts.items():
                percentage = count / len(null_user_data) * 100
                print(f"  {cluster}: {count:,} ({percentage:.1f}%)")
        
        # 检查队列信息
        if 'queue' in null_user_data.columns:
            queue_counts = null_user_data['queue'].value_counts()
            print(f"队列分布:")
            for queue, count in queue_counts.head(5).items():
                percentage = count / len(null_user_data) * 100
                print(f"  {queue}: {count:,} ({percentage:.1f}%)")
        
        # 检查资源请求
        if 'num_processors' in null_user_data.columns and 'gpu_num' in null_user_data.columns:
            resource_patterns = null_user_data.groupby(['num_processors', 'gpu_num']).size().sort_values(ascending=False)
            print(f"\n资源请求模式 (前5):")
            for (cpu, gpu), count in resource_patterns.head(5).items():
                percentage = count / len(null_user_data) * 100
                print(f"  CPU:{cpu}, GPU:{gpu}: {count:,} ({percentage:.1f}%)")
        
        # 分析持续时间
        if 'start_time' in null_user_data.columns and 'end_time' in null_user_data.columns:
            null_user_data['start_time'] = pd.to_datetime(null_user_data['start_time'])
            null_user_data['end_time'] = pd.to_datetime(null_user_data['end_time'])
            null_user_data['duration'] = (null_user_data['end_time'] - null_user_data['start_time']).dt.total_seconds()
            
            valid_duration = null_user_data[null_user_data['duration'] > 0]['duration']
            if len(valid_duration) > 0:
                print(f"\n持续时间统计:")
                print(f"  平均持续时间: {valid_duration.mean():.1f}秒")
                print(f"  中位数持续时间: {valid_duration.median():.1f}秒")
                print(f"  超短作业(≤10秒): {len(valid_duration[valid_duration <= 10]):,} ({len(valid_duration[valid_duration <= 10])/len(valid_duration)*100:.1f}%)")
        
        # 保存样例数据
        script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
        output_dir = script_dir / "output" / "null_user_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存前1000条空用户记录样例
        sample_data = null_user_data.head(1000)
        sample_data.to_csv(output_dir / 'null_user_sample.csv', index=False)
        
        # 保存统计摘要
        summary = {
            'total_null_records': len(null_user_data),
            'job_name_distribution': job_name_counts.head(10).to_dict(),
            'job_id_distribution': job_id_counts.head(10).to_dict(),
            'hourly_distribution': hourly_counts.to_dict()
        }
        
        import json
        with open(output_dir / 'null_user_analysis_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"空用户记录分析结果已保存到: {output_dir}")
    
    # 对比有效用户和空用户的特征
    if len(valid_user_data) > 0 and len(null_user_data) > 0:
        print(f"\n=== 有效用户 vs 空用户对比 ===")
        
        # 时间分布对比
        valid_user_data['submit_hour'] = valid_user_data['submit_time'].dt.hour
        valid_hourly = valid_user_data['submit_hour'].value_counts().sort_index()
        null_hourly = null_user_data['submit_hour'].value_counts().sort_index()
        
        print(f"时间分布对比 (峰值小时):")
        print(f"  有效用户峰值: {valid_hourly.idxmax()}点 ({valid_hourly.max():,}个作业)")
        print(f"  空用户峰值: {null_hourly.idxmax()}点 ({null_hourly.max():,}个作业)")
        
        # 作业名称对比
        valid_job_names = set(valid_user_data['job_name'].unique())
        null_job_names = set(null_user_data['job_name'].unique())
        common_job_names = valid_job_names.intersection(null_job_names)
        
        print(f"\n作业名称对比:")
        print(f"  有效用户唯一作业名: {len(valid_job_names)}")
        print(f"  空用户唯一作业名: {len(null_job_names)}")
        print(f"  共同作业名: {len(common_job_names)}")
        
        if len(common_job_names) > 0:
            print(f"  共同作业名示例: {list(common_job_names)[:5]}")

def main():
    """主函数"""
    try:
        analyze_null_user_records()
    except Exception as e:
        logger.error(f"分析过程出错: {e}")
        raise

if __name__ == "__main__":
    main()
