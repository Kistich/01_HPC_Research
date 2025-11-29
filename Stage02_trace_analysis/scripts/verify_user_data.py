#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证用户数据的完整性和准确性
检查为什么用户作业数量加起来与总数不符
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_peak_day_data():
    """验证峰值日数据的完整性"""
    # 使用相对路径
    script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
    project_root = script_dir.parent  # 01_HPC_Research/
    data_path = str(project_root / "Stage01_data_filter_preprocess" / "full_processing_outputs" / "stage6_data_standardization" / "standardized_data.csv")

    logger.info("开始验证数据完整性...")
    
    # 分块读取数据
    chunk_size = 100000
    peak_day_chunks = []
    total_rows_processed = 0
    
    for chunk in pd.read_csv(data_path, chunksize=chunk_size):
        total_rows_processed += len(chunk)
        
        # 转换时间字段
        chunk['submit_time'] = pd.to_datetime(chunk['submit_time'])
        chunk['submit_date'] = chunk['submit_time'].dt.date
        
        # 筛选峰值日数据
        peak_day_chunk = chunk[chunk['submit_date'] == pd.to_datetime('2024-04-14').date()]
        
        if len(peak_day_chunk) > 0:
            peak_day_chunks.append(peak_day_chunk)
            logger.info(f"块中找到峰值日数据: {len(peak_day_chunk)}条")
    
    logger.info(f"总共处理了 {total_rows_processed} 行数据")
    
    # 合并所有峰值日数据
    peak_day_data = pd.concat(peak_day_chunks, ignore_index=True)
    logger.info(f"峰值日总数据: {len(peak_day_data)}条")
    
    # 检查用户ID的情况
    print(f"\n=== 数据验证结果 ===")
    print(f"峰值日总作业数: {len(peak_day_data):,}")
    
    # 检查用户ID字段
    print(f"\n=== 用户ID字段分析 ===")
    print(f"用户ID字段名: {peak_day_data.columns[peak_day_data.columns.str.contains('user', case=False)].tolist()}")
    
    # 检查是否有空值
    user_id_null = peak_day_data['user_id'].isnull().sum()
    print(f"用户ID为空的记录数: {user_id_null:,}")
    
    # 统计有效用户ID的作业数
    valid_user_data = peak_day_data[peak_day_data['user_id'].notna()]
    print(f"有效用户ID的作业数: {len(valid_user_data):,}")
    
    # 统计每个用户的作业数
    user_job_counts = valid_user_data['user_id'].value_counts().sort_values(ascending=False)
    print(f"唯一用户数: {len(user_job_counts)}")
    
    print(f"\n=== 详细用户作业统计 ===")
    print(f"{'用户ID':<15} {'作业数量':<10} {'占比(%)':<10} {'累计作业数':<12}")
    print("-" * 55)
    
    cumulative_jobs = 0
    for i, (user_id, count) in enumerate(user_job_counts.items(), 1):
        cumulative_jobs += count
        percentage = count / len(valid_user_data) * 100
        print(f"{str(user_id):<15} {count:<10,} {percentage:<10.2f} {cumulative_jobs:<12,}")
    
    print(f"\n=== 验证结果 ===")
    print(f"所有用户作业数总和: {user_job_counts.sum():,}")
    print(f"有效用户数据总数: {len(valid_user_data):,}")
    print(f"峰值日总数据: {len(peak_day_data):,}")
    print(f"用户ID为空的数据: {user_id_null:,}")
    
    # 验证计算
    calculated_total = user_job_counts.sum() + user_id_null
    print(f"计算的总数 (有效+空值): {calculated_total:,}")
    
    if calculated_total == len(peak_day_data):
        print("✅ 数据验证通过！")
    else:
        print("❌ 数据验证失败！存在数据不一致问题")
    
    # 检查空值用户ID的数据
    if user_id_null > 0:
        null_user_data = peak_day_data[peak_day_data['user_id'].isnull()]
        print(f"\n=== 空用户ID数据分析 ===")
        print(f"空用户ID记录数: {len(null_user_data):,}")
        
        # 检查这些记录的其他字段
        print("空用户ID记录的样例:")
        print(null_user_data[['job_id', 'submit_time', 'user_id', 'job_name']].head())
    
    # 保存详细统计
    script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
    output_dir = script_dir / "output" / "data_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存用户统计
    user_stats_df = pd.DataFrame({
        'user_id': user_job_counts.index,
        'job_count': user_job_counts.values,
        'percentage': user_job_counts.values / len(valid_user_data) * 100
    })
    user_stats_df.to_csv(output_dir / 'verified_user_job_counts.csv', index=False)
    
    # 如果有空值数据，也保存
    if user_id_null > 0:
        null_user_data.to_csv(output_dir / 'null_user_id_records.csv', index=False)
    
    logger.info(f"验证结果已保存到: {output_dir}")
    
    return peak_day_data, user_job_counts, user_id_null

def analyze_missing_jobs():
    """分析缺失的作业数据"""
    logger.info("分析可能的数据缺失原因...")
    
    # 重新验证数据
    peak_day_data, user_job_counts, user_id_null = verify_peak_day_data()
    
    # 计算差异
    expected_total = 1416444
    actual_total = len(peak_day_data)
    difference = expected_total - actual_total
    
    print(f"\n=== 数据差异分析 ===")
    print(f"期望总数: {expected_total:,}")
    print(f"实际总数: {actual_total:,}")
    print(f"差异: {difference:,}")
    
    if difference > 0:
        print(f"❌ 缺失了 {difference:,} 条记录")
        print("可能原因:")
        print("1. 数据文件不完整")
        print("2. 日期筛选条件有问题")
        print("3. 数据读取过程中出现错误")
        print("4. 原始统计数字有误")
    elif difference < 0:
        print(f"⚠️ 多出了 {abs(difference):,} 条记录")
        print("可能原因:")
        print("1. 日期筛选条件过宽")
        print("2. 重复计算")
    else:
        print("✅ 数据总数匹配")
    
    # 检查日期范围
    print(f"\n=== 日期范围检查 ===")
    date_range = peak_day_data['submit_time'].dt.date.value_counts().sort_index()
    print("数据中的日期分布:")
    for date, count in date_range.items():
        print(f"  {date}: {count:,} 条记录")

def main():
    """主函数"""
    try:
        analyze_missing_jobs()
    except Exception as e:
        logger.error(f"验证过程出错: {e}")
        raise

if __name__ == "__main__":
    main()
