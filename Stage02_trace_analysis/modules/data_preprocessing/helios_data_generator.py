#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helios数据格式生成器
专门用于将我们的HPC作业数据转换为Helios兼容的数据格式
支持生成cluster_log.csv, cluster_sequence.csv, cluster_throughput.csv, cluster_user.pkl
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class HeliosDataGenerator:
    """Helios数据格式生成器"""
    
    def __init__(self):
        """初始化生成器"""
        self.generated_data = {}
        
    def generate_all_helios_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成所有Helios兼容格式数据
        
        Args:
            df: 预处理后的作业数据
            
        Returns:
            包含所有Helios格式数据的字典
        """
        logger.info("开始生成Helios兼容格式数据...")
        
        # 验证必要字段
        required_fields = ['job_id', 'user_id', 'submit_time', 'start_time', 'end_time', 
                          'duration', 'is_gpu_job', 'cpu_num', 'state']
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            raise ValueError(f"缺少必要字段: {missing_fields}")
        
        # 生成各种格式数据
        self.generated_data['cluster_log'] = self._generate_cluster_log(df)
        self.generated_data['cluster_sequence'] = self._generate_cluster_sequence(df)
        self.generated_data['cluster_throughput'] = self._generate_cluster_throughput(df)
        self.generated_data['cluster_user'] = self._generate_cluster_user(df)
        
        logger.info("Helios兼容格式数据生成完成")
        return self.generated_data
    
    def _generate_cluster_log(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成cluster_log.csv格式数据"""
        logger.info("生成cluster_log.csv...")
        
        # 创建Helios格式的作业日志
        cluster_log = pd.DataFrame()
        
        # 基础字段映射
        cluster_log['job_id'] = df['job_id']
        cluster_log['user'] = df['user_id']
        cluster_log['gpu_num'] = df.get('gpu_num', 0)
        cluster_log['cpu_num'] = df['cpu_num']
        cluster_log['node_num'] = df.get('num_exec_hosts', 1)
        cluster_log['state'] = df['state']
        cluster_log['submit_time'] = df['submit_time']
        cluster_log['start_time'] = df['start_time']
        cluster_log['end_time'] = df['end_time']
        cluster_log['duration'] = df['duration']
        cluster_log['queue'] = df.get('queue_time', 0)
        
        # 数据清洗和类型转换
        cluster_log['gpu_num'] = pd.to_numeric(cluster_log['gpu_num'], errors='coerce').fillna(0).astype(int)
        cluster_log['cpu_num'] = pd.to_numeric(cluster_log['cpu_num'], errors='coerce').fillna(1).astype(int)
        cluster_log['node_num'] = pd.to_numeric(cluster_log['node_num'], errors='coerce').fillna(1).astype(int)
        cluster_log['duration'] = pd.to_numeric(cluster_log['duration'], errors='coerce').fillna(0)
        cluster_log['queue'] = pd.to_numeric(cluster_log['queue'], errors='coerce').fillna(0)
        
        # 移除无效记录
        cluster_log = cluster_log[cluster_log['duration'] > 0]
        cluster_log = cluster_log.dropna(subset=['submit_time', 'start_time', 'end_time'])
        
        logger.info(f"cluster_log生成完成: {len(cluster_log)} 条记录")
        return cluster_log
    
    def _generate_cluster_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成cluster_sequence.csv格式数据"""
        logger.info("生成cluster_sequence.csv...")
        
        # 按小时聚合提交时间
        df_hourly = df.copy()
        df_hourly['hour'] = df_hourly['submit_time'].dt.floor('H')
        
        # 聚合统计
        sequence_stats = df_hourly.groupby('hour').agg({
            'job_id': 'count',
            'gpu_num': 'sum',
            'cpu_num': 'sum',
            'duration': 'mean',
            'queue_time': 'mean'
        }).reset_index()
        
        # 重命名列以匹配Helios格式
        sequence_stats.columns = ['time', 'submitted_jobs', 'total_gpu_requested', 
                                 'total_cpu_requested', 'avg_duration', 'avg_queue_time']
        
        # 填充时间间隙
        if len(sequence_stats) > 0:
            full_time_range = pd.date_range(
                start=sequence_stats['time'].min(),
                end=sequence_stats['time'].max(),
                freq='H'
            )
            full_sequence = pd.DataFrame({'time': full_time_range})
            sequence_stats = full_sequence.merge(sequence_stats, on='time', how='left').fillna(0)
        
        logger.info(f"cluster_sequence生成完成: {len(sequence_stats)} 个时间点")
        return sequence_stats
    
    def _generate_cluster_throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成cluster_throughput.csv格式数据"""
        logger.info("生成cluster_throughput.csv...")
        
        # 按小时聚合开始时间
        df_hourly = df.copy()
        df_hourly['hour'] = df_hourly['start_time'].dt.floor('H')
        
        # 计算资源小时数
        df_hourly['gpu_hours'] = df_hourly['gpu_num'] * df_hourly['duration'] / 3600
        df_hourly['cpu_hours'] = df_hourly['cpu_num'] * df_hourly['duration'] / 3600
        
        # 聚合吞吐量统计
        throughput_stats = df_hourly.groupby('hour').agg({
            'job_id': 'count',
            'duration': ['mean', 'sum'],
            'gpu_hours': 'sum',
            'cpu_hours': 'sum'
        }).reset_index()
        
        # 展平多级列名
        throughput_stats.columns = ['time', 'jobs_started', 'avg_duration', 
                                   'total_duration', 'gpu_hours_consumed', 'cpu_hours_consumed']
        
        logger.info(f"cluster_throughput生成完成: {len(throughput_stats)} 个时间点")
        return throughput_stats
    
    def _generate_cluster_user(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成cluster_user.pkl格式数据"""
        logger.info("生成cluster_user.pkl...")
        
        user_stats = []
        
        for user_id in df['user_id'].unique():
            if pd.isna(user_id):
                continue
                
            user_jobs = df[df['user_id'] == user_id]
            gpu_jobs = user_jobs[user_jobs['is_gpu_job'] == True]
            cpu_jobs = user_jobs[user_jobs['is_gpu_job'] == False]
            
            # GPU相关统计
            total_gpu_time = 0
            total_gpu_pend_time = 0
            completed_gpu_percent = 0
            
            if len(gpu_jobs) > 0:
                total_gpu_time = (gpu_jobs['gpu_num'] * gpu_jobs['duration']).sum() / 3600
                if 'queue_time' in gpu_jobs.columns:
                    total_gpu_pend_time = gpu_jobs['queue_time'].sum() / 3600
                completed_gpu = gpu_jobs[gpu_jobs['state'] == 'Pass']
                completed_gpu_percent = len(completed_gpu) / len(gpu_jobs) * 100
            
            # CPU相关统计
            total_cpu_only_time = 0
            total_cpu_pend_time = 0
            completed_cpu_percent = 0

            if len(cpu_jobs) > 0:
                total_cpu_only_time = (cpu_jobs['cpu_num'] * cpu_jobs['duration']).sum() / 3600
                if 'queue_time' in cpu_jobs.columns:
                    total_cpu_pend_time = cpu_jobs['queue_time'].sum() / 3600
                completed_cpu = cpu_jobs[cpu_jobs['state'] == 'Pass']
                completed_cpu_percent = len(completed_cpu) / len(cpu_jobs) * 100

            user_record = {
                'user': user_id,
                'total_gpu_time': total_gpu_time,
                'total_cpu_only_time': total_cpu_only_time,
                'total_gpu_pend_time': total_gpu_pend_time,
                'total_cpu_pend_time': total_cpu_pend_time,
                'completed_gpu_percent': completed_gpu_percent,
                'completed_cpu_percent': completed_cpu_percent,
                'total_jobs': len(user_jobs),
                'gpu_jobs_count': len(gpu_jobs),
                'cpu_jobs_count': len(cpu_jobs)
            }
            
            user_stats.append(user_record)
        
        cluster_user = pd.DataFrame(user_stats)
        
        logger.info(f"cluster_user生成完成: {len(cluster_user)} 个用户")
        return cluster_user
    
    def save_to_directory(self, output_dir: str, format_name: str = "helios_format"):
        """
        保存生成的数据到指定目录
        
        Args:
            output_dir: 输出目录
            format_name: 格式目录名称
        """
        output_path = Path(output_dir) / format_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存Helios格式数据到: {output_path}")
        
        # 保存CSV文件
        for data_name in ['cluster_log', 'cluster_sequence', 'cluster_throughput']:
            if data_name in self.generated_data:
                file_path = output_path / f"{data_name}.csv"
                self.generated_data[data_name].to_csv(file_path, index=False)
                logger.info(f"已保存: {file_path}")
        
        # 保存pickle文件
        if 'cluster_user' in self.generated_data:
            file_path = output_path / "cluster_user.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(self.generated_data['cluster_user'], f)
            logger.info(f"已保存: {file_path}")
        
        logger.info("Helios格式数据保存完成")
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取生成数据的摘要信息"""
        summary = {}
        
        for data_name, data in self.generated_data.items():
            if isinstance(data, pd.DataFrame):
                summary[data_name] = {
                    'records': len(data),
                    'columns': list(data.columns),
                    'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024
                }
        
        return summary


def main():
    """测试Helios数据生成器"""
    # 这里需要实际的预处理数据来测试
    print("Helios数据生成器模块已创建")
    print("使用方法:")
    print("1. 创建生成器实例: generator = HeliosDataGenerator()")
    print("2. 生成数据: helios_data = generator.generate_all_helios_data(preprocessed_df)")
    print("3. 保存数据: generator.save_to_directory('output_dir')")


if __name__ == "__main__":
    main()
