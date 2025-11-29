#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据加载器模块
基于Helios项目的数据处理方法，适配HPC集群数据
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def is_gpu_job(exec_hosts):
    """
    基于exec_hosts字段判断是否为GPU作业

    Args:
        exec_hosts: 执行主机列表字符串

    Returns:
        bool: True表示GPU作业，False表示CPU作业
    """
    if pd.isna(exec_hosts) or exec_hosts == '':
        return False

    # 将exec_hosts转换为字符串并检查是否包含GPU节点
    hosts_str = str(exec_hosts).lower()

    # 检查是否包含GPU节点标识
    # 基于实际数据，主要模式是 gpu1-xx, cpu1-xx, bigmem-xx
    if 'gpu' in hosts_str:
        return True

    # 其他GPU节点命名模式
    gpu_patterns = ['v100', 'a100', 'rtx', 'titan', 'tesla', 'k80', 'p100']
    for pattern in gpu_patterns:
        if pattern in hosts_str:
            return True

    return False


def add_gpu_job_flag(df):
    """
    为DataFrame添加is_gpu_job标志列 - 优先使用gpu_num字段

    Args:
        df: 包含exec_hosts或gpu_num列的DataFrame

    Returns:
        DataFrame: 添加了is_gpu_job列的DataFrame
    """
    df = df.copy()

    # 优先使用gpu_num字段（新数据格式）
    if 'gpu_num' in df.columns:
        df['is_gpu_job'] = pd.to_numeric(df['gpu_num'], errors='coerce').fillna(0) > 0
        logger.info("使用gpu_num字段识别GPU作业")
    elif 'exec_hosts' in df.columns:
        df['is_gpu_job'] = df['exec_hosts'].apply(is_gpu_job)
        logger.info("使用exec_hosts字段识别GPU作业")
    else:
        logger.warning("DataFrame中没有gpu_num或exec_hosts列，无法判断GPU作业")
        df['is_gpu_job'] = False

    # 统计GPU作业数量
    gpu_count = df['is_gpu_job'].sum()
    total_count = len(df)
    logger.info(f"GPU作业识别完成: {gpu_count:,}/{total_count:,} ({gpu_count/total_count*100:.1f}%)")

    return df


class DataLoader:
    """数据加载器类"""
    
    def __init__(self, config_path: str = "config/cluster_config.yaml"):
        """
        初始化数据加载器
        
        Args:
            config_path: 集群配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_cache = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def load_job_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        加载作业数据

        Args:
            force_reload: 是否强制重新加载

        Returns:
            作业数据DataFrame
        """
        cache_key = "job_data"

        if not force_reload and cache_key in self.data_cache:
            logger.info("从缓存加载作业数据")
            return self.data_cache[cache_key]

        data_path_str = self.config['data_sources']['job_data_path']

        # 将相对路径转换为绝对路径（相对于config文件所在目录）
        data_path = Path(data_path_str)
        if not data_path.is_absolute():
            # 相对于config文件所在目录解析路径
            config_dir = self.config_path.parent
            data_path = (config_dir / data_path).resolve()

        logger.info(f"从文件加载作业数据: {data_path}")

        try:
            # 读取CSV文件
            df = pd.read_csv(data_path)
            logger.info(f"原始数据加载完成，共 {len(df):,} 条记录")

            # 基础数据类型转换
            df = self._convert_data_types(df)

            # 缓存数据
            self.data_cache[cache_key] = df

            logger.info(f"数据处理完成，共 {len(df):,} 条有效记录")
            return df

        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        基础数据类型转换 - 简化版，只进行必要的类型转换

        Args:
            df: 原始数据DataFrame

        Returns:
            类型转换后的DataFrame
        """
        logger.info("开始数据类型转换...")

        # 时间字段转换
        time_columns = ['submit_time', 'start_time', 'end_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # 数值字段转换
        numeric_columns = ['job_id', 'cpu_num', 'gpu_num', 'node_num', 'duration', 'queue_time']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 布尔字段转换
        if 'is_gpu_job' in df.columns:
            df['is_gpu_job'] = df['is_gpu_job'].astype(bool)

        logger.info("数据类型转换完成")
        return df


    def get_cluster_info(self) -> Dict[str, Any]:
        """获取集群配置信息"""
        return self.config.get('cluster_info', {})


    
    def get_subcluster_info(self) -> Dict[str, Any]:
        """获取子集群配置信息"""
        return self.config.get('subclusters', {})
    
    def get_job_classification_rules(self) -> Dict[str, Any]:
        """获取作业分类规则"""
        return self.config.get('job_classification', {})
    
    def clear_cache(self):
        """清空数据缓存"""
        self.data_cache.clear()
        logger.info("数据缓存已清空")


def load_analysis_config(config_path: str = "config/analysis_config.yaml") -> Dict[str, Any]:
    """
    加载分析配置文件
    
    Args:
        config_path: 分析配置文件路径
        
    Returns:
        分析配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"分析配置加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"分析配置加载失败: {e}")
        raise
