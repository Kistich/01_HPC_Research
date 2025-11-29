#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
子集群识别器模块
基于exec_hosts字段识别作业使用的子集群类型
参考现有的exec_hosts解析逻辑
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class ClusterIdentifier:
    """子集群识别器类"""
    
    def __init__(self, subcluster_config: Dict[str, Any], job_classification: Dict[str, Any]):
        """
        初始化子集群识别器
        
        Args:
            subcluster_config: 子集群配置信息
            job_classification: 作业分类规则
        """
        self.subcluster_config = subcluster_config
        self.job_classification = job_classification
        self.gpu_patterns = job_classification.get('gpu_job_patterns', [])
        self.cpu_patterns = job_classification.get('cpu_job_patterns', [])
        
    def parse_exec_hosts(self, exec_hosts_str: str) -> List[str]:
        """
        解析exec_hosts字符串，提取节点列表
        基于现有的解析逻辑进行简化实现
        
        Args:
            exec_hosts_str: exec_hosts字段值
            
        Returns:
            节点名称列表
        """
        if pd.isna(exec_hosts_str) or exec_hosts_str == '':
            return []
        
        try:
            hosts_str = str(exec_hosts_str).strip()
            
            # 处理常见的分隔符
            # 按空格、逗号、分号分割
            hosts = re.split(r'[\s,;]+', hosts_str)
            
            # 清理和去重
            unique_hosts = []
            for host in hosts:
                host = host.strip()
                if host:
                    # 简单的节点名称标准化
                    host = self._standardize_node_name(host)
                    if host and host not in unique_hosts:
                        unique_hosts.append(host)
            
            return unique_hosts
            
        except Exception as e:
            logger.debug(f"解析exec_hosts失败 '{exec_hosts_str}': {e}")
            return []
    
    def _standardize_node_name(self, node_name: str) -> str:
        """
        标准化节点名称
        
        Args:
            node_name: 原始节点名称
            
        Returns:
            标准化后的节点名称
        """
        if not node_name:
            return ""
        
        name = node_name.strip().lower()
        
        # 移除常见的后缀
        suffixes_to_remove = ['.local', '.domain', '.com']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # 移除无效字符，只保留字母、数字和连字符
        name = re.sub(r'[^a-zA-Z0-9\-]', '', name)
        
        return name
    
    def identify_job_type(self, exec_hosts: List[str]) -> str:
        """
        识别作业类型 (GPU/CPU)
        
        Args:
            exec_hosts: 节点列表
            
        Returns:
            作业类型 ('gpu', 'cpu', 'mixed', 'unknown')
        """
        if not exec_hosts:
            return 'unknown'
        
        gpu_nodes = 0
        cpu_nodes = 0
        
        for host in exec_hosts:
            if any(pattern in host for pattern in self.gpu_patterns):
                gpu_nodes += 1
            elif any(pattern in host for pattern in self.cpu_patterns):
                cpu_nodes += 1
        
        if gpu_nodes > 0 and cpu_nodes == 0:
            return 'gpu'
        elif cpu_nodes > 0 and gpu_nodes == 0:
            return 'cpu'
        elif gpu_nodes > 0 and cpu_nodes > 0:
            return 'mixed'
        else:
            return 'unknown'
    
    def identify_subclusters(self, exec_hosts: List[str]) -> List[str]:
        """
        识别作业使用的子集群
        
        Args:
            exec_hosts: 节点列表
            
        Returns:
            子集群名称列表
        """
        subclusters = set()
        
        for host in exec_hosts:
            for subcluster_name, config in self.subcluster_config.items():
                prefix = config.get('node_prefix', '')
                if prefix and host.startswith(prefix.lower().rstrip('-')):
                    subclusters.add(subcluster_name)
                    break
        
        return list(subclusters)
    
    def get_primary_subcluster(self, exec_hosts: List[str]) -> str:
        """
        获取主要子集群 (节点数最多的子集群)
        
        Args:
            exec_hosts: 节点列表
            
        Returns:
            主要子集群名称
        """
        subcluster_counts = defaultdict(int)
        
        for host in exec_hosts:
            for subcluster_name, config in self.subcluster_config.items():
                prefix = config.get('node_prefix', '')
                if prefix and host.startswith(prefix.lower().rstrip('-')):
                    subcluster_counts[subcluster_name] += 1
                    break
        
        if not subcluster_counts:
            return 'unknown'
        
        # 返回节点数最多的子集群
        return max(subcluster_counts.items(), key=lambda x: x[1])[0]
    
    def analyze_job_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        分析作业在子集群间的分布
        
        Args:
            df: 包含解析后exec_hosts信息的DataFrame
            
        Returns:
            分布统计信息
        """
        logger.info("分析作业在子集群间的分布...")
        
        # 确保有必要的列
        if 'parsed_exec_hosts' not in df.columns:
            logger.warning("DataFrame中缺少parsed_exec_hosts列")
            return {}
        
        stats = {
            'total_jobs': len(df),
            'job_type_distribution': {},
            'subcluster_distribution': {},
            'primary_subcluster_distribution': {},
            'multi_subcluster_jobs': 0
        }
        
        # 作业类型分布
        job_types = df['parsed_exec_hosts'].apply(self.identify_job_type)
        stats['job_type_distribution'] = job_types.value_counts().to_dict()
        
        # 子集群分布
        all_subclusters = []
        primary_subclusters = []
        multi_subcluster_count = 0
        
        for hosts in df['parsed_exec_hosts']:
            if isinstance(hosts, list) and hosts:
                subclusters = self.identify_subclusters(hosts)
                all_subclusters.extend(subclusters)
                
                primary = self.get_primary_subcluster(hosts)
                primary_subclusters.append(primary)
                
                if len(subclusters) > 1:
                    multi_subcluster_count += 1
        
        # 统计子集群使用情况
        subcluster_series = pd.Series(all_subclusters)
        stats['subcluster_distribution'] = subcluster_series.value_counts().to_dict()
        
        # 统计主要子集群分布
        primary_series = pd.Series(primary_subclusters)
        stats['primary_subcluster_distribution'] = primary_series.value_counts().to_dict()
        
        stats['multi_subcluster_jobs'] = multi_subcluster_count
        
        logger.info(f"分布分析完成: {stats['total_jobs']} 个作业")
        return stats
    
    def enhance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        增强DataFrame，添加子集群识别信息
        
        Args:
            df: 原始DataFrame
            
        Returns:
            增强后的DataFrame
        """
        logger.info("增强DataFrame，添加子集群识别信息...")
        
        df = df.copy()
        
        # 解析exec_hosts
        if 'exec_hosts' in df.columns:
            df['parsed_exec_hosts'] = df['exec_hosts'].apply(self.parse_exec_hosts)
        else:
            df['parsed_exec_hosts'] = [[] for _ in range(len(df))]
        
        # 识别作业类型
        df['job_type'] = df['parsed_exec_hosts'].apply(self.identify_job_type)
        
        # 识别主要子集群
        df['primary_subcluster'] = df['parsed_exec_hosts'].apply(self.get_primary_subcluster)
        
        # 计算实际使用的节点数
        df['actual_node_count'] = df['parsed_exec_hosts'].apply(len)
        
        # 识别是否为跨子集群作业
        df['is_multi_subcluster'] = df['parsed_exec_hosts'].apply(
            lambda hosts: len(self.identify_subclusters(hosts)) > 1
        )
        
        logger.info("DataFrame增强完成")
        return df
