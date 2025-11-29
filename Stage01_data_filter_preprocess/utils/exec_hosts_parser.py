#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
exec_hosts解析器
基于Trace_Analysis_Module/module_02_exec_hosts_parser/exec_hosts_parser.py的解析逻辑
结合集群配置计算实际的CPU数量和节点信息
"""

import re
import json
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class ExecHostsParser:
    """exec_hosts解析器"""
    
    def __init__(self, cluster_config_path: str):
        """
        初始化解析器
        
        Args:
            cluster_config_path: 集群配置文件路径
        """
        self.cluster_config = self._load_cluster_config(cluster_config_path)
        self.subclusters = self.cluster_config.get('subclusters', {})
        
        # 构建节点前缀到子集群的映射
        self.prefix_to_subcluster = {}
        self.subcluster_specs = {}
        
        for subcluster_name, config in self.subclusters.items():
            prefix = config.get('node_prefix', '')
            self.prefix_to_subcluster[prefix] = subcluster_name
            self.subcluster_specs[subcluster_name] = {
                'cpu_cores_per_node': self._parse_cpu_cores(config.get('cpu_spec', '')),
                'gpu_count_per_node': self._parse_gpu_count(config.get('gpu_spec', '')),
                'memory_gb_per_node': self._parse_memory_gb(config.get('memory_spec', '')),
                'node_type': config.get('node_type', 'unknown')
            }
        
        # 解析统计
        self.parsing_stats = {
            'total_processed': 0,
            'successful_parsed': 0,
            'failed_parsed': 0,
            'pattern_usage': defaultdict(int),
            'parsing_errors': []
        }
    
    def _load_cluster_config(self, config_path: str) -> Dict[str, Any]:
        """加载集群配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load cluster config: {e}")
            return {}
    
    def _parse_cpu_cores(self, cpu_spec: str) -> int:
        """解析CPU核心数"""
        if not cpu_spec:
            return 0
        
        # 匹配模式: "2 * Intel(R) Xeon(R) Platinum 8358P CPU 32C @ 2.60GHz"
        # 或 "4 * Intel(R) Xeon(R) Gold 6348H CPU 24C @ 2.30GHz"
        match = re.search(r'(\d+)\s*\*.*?(\d+)C', cpu_spec)
        if match:
            sockets = int(match.group(1))
            cores_per_socket = int(match.group(2))
            return sockets * cores_per_socket
        
        # 备用匹配模式
        match = re.search(r'(\d+)\s*cores?', cpu_spec, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _parse_gpu_count(self, gpu_spec: str) -> int:
        """解析GPU数量"""
        if not gpu_spec or gpu_spec.lower() == 'null':
            return 0
        
        # 匹配模式: "8 * NVIDIA A800-SXM4-80GB"
        match = re.search(r'(\d+)\s*\*', gpu_spec)
        if match:
            return int(match.group(1))
        
        return 0
    
    def _parse_memory_gb(self, memory_spec: str) -> int:
        """解析内存大小(GB)"""
        if not memory_spec:
            return 0
        
        # 匹配模式: "512GB", "1024GB", "2048GB"
        match = re.search(r'(\d+)GB', memory_spec, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # 特殊情况: "大内存"
        if '大内存' in memory_spec:
            return 2048  # 假设大内存节点为2TB
        
        return 0
    
    def parse_exec_hosts_value(self, exec_hosts_str: str) -> Dict[str, Any]:
        """
        解析单个exec_hosts值
        
        Returns:
            Dict包含:
            - node_list: 节点列表
            - node_count: 节点数量
            - total_cpu_cores: 总CPU核心数
            - total_gpu_count: 总GPU数量
            - subcluster_distribution: 子集群分布
            - parsing_success: 解析是否成功
        """
        if not exec_hosts_str or pd.isna(exec_hosts_str):
            return self._empty_result()
        
        self.parsing_stats['total_processed'] += 1
        
        try:
            # 识别模式并解析节点列表
            pattern = self._identify_pattern(str(exec_hosts_str))
            self.parsing_stats['pattern_usage'][pattern] += 1
            
            node_list = self._parse_by_pattern(exec_hosts_str, pattern)
            
            if not node_list:
                self.parsing_stats['failed_parsed'] += 1
                return self._empty_result()
            
            # 计算资源信息
            result = self._calculate_resources(node_list)
            result['parsing_success'] = True
            
            self.parsing_stats['successful_parsed'] += 1
            return result
            
        except Exception as e:
            self.parsing_stats['failed_parsed'] += 1
            self.parsing_stats['parsing_errors'].append({
                'input': exec_hosts_str,
                'error': str(e)
            })
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'node_list': [],
            'node_count': 0,
            'total_cpu_cores': 0,
            'total_gpu_count': 0,
            'subcluster_distribution': {},
            'parsing_success': False
        }
    
    def _identify_pattern(self, value_str: str) -> str:
        """识别exec_hosts值的模式"""
        if not value_str or value_str.lower() in ['nan', 'null', 'none']:
            return 'empty_or_null'
        
        if value_str.startswith('{') and value_str.endswith('}'):
            return 'json_format'
        
        if '=' in value_str:
            if 'host=' in value_str.lower():
                return 'key_value_host'
            else:
                return 'key_value_other'
        
        if '[' in value_str and ']' in value_str:
            bracket_content = re.findall(r'\[([^\]]+)\]', value_str)
            if bracket_content:
                content = bracket_content[0]
                if '-' in content and ',' in content:
                    return 'bracket_range_and_list'
                elif '-' in content:
                    return 'bracket_range'
                elif ',' in content:
                    return 'bracket_list'
        
        if '+' in value_str:
            return 'plus_separated'
        elif ',' in value_str:
            return 'comma_separated'
        elif ' ' in value_str.strip() and len(value_str.split()) > 1:
            return 'space_separated'
        elif len(value_str.split()) == 1:
            return 'single_value'
        
        return 'complex_or_unknown'
    
    def _parse_by_pattern(self, exec_hosts_str: str, pattern: str) -> List[str]:
        """根据模式解析节点列表"""
        if pattern == 'single_value':
            return [exec_hosts_str.strip()]
        
        elif pattern == 'comma_separated':
            return [host.strip() for host in exec_hosts_str.split(',')]
        
        elif pattern == 'space_separated':
            return exec_hosts_str.split()
        
        elif pattern == 'plus_separated':
            return [host.strip() for host in exec_hosts_str.split('+')]
        
        elif pattern == 'bracket_range':
            return self._expand_bracket_range(exec_hosts_str)
        
        elif pattern == 'bracket_list':
            return self._extract_bracket_list(exec_hosts_str)
        
        elif pattern == 'json_format':
            return self._parse_json_format(exec_hosts_str)
        
        elif pattern == 'key_value_host':
            return self._extract_key_value(exec_hosts_str, 'host')
        
        else:
            return self._fallback_parsing(exec_hosts_str)
    
    def _expand_bracket_range(self, exec_hosts_str: str) -> List[str]:
        """展开方括号范围格式，如 cpu1-[1-5] -> [cpu1-1, cpu1-2, cpu1-3, cpu1-4, cpu1-5]"""
        pattern = r'([a-zA-Z0-9_-]+)\[(\d+)-(\d+)\]'
        match = re.search(pattern, exec_hosts_str)
        
        if match:
            prefix = match.group(1)
            start = int(match.group(2))
            end = int(match.group(3))
            
            return [f"{prefix}{i}" for i in range(start, end + 1)]
        
        return [exec_hosts_str]
    
    def _extract_bracket_list(self, exec_hosts_str: str) -> List[str]:
        """提取方括号列表格式，如 cpu1-[1,3,5] -> [cpu1-1, cpu1-3, cpu1-5]"""
        pattern = r'([a-zA-Z0-9_-]+)\[([0-9,]+)\]'
        match = re.search(pattern, exec_hosts_str)
        
        if match:
            prefix = match.group(1)
            numbers = match.group(2).split(',')
            
            return [f"{prefix}{num.strip()}" for num in numbers]
        
        return [exec_hosts_str]
    
    def _parse_json_format(self, exec_hosts_str: str) -> List[str]:
        """解析JSON格式"""
        try:
            data = json.loads(exec_hosts_str)
            if isinstance(data, dict):
                for key in ['nodes', 'hosts', 'servers', 'machines']:
                    if key in data and isinstance(data[key], list):
                        return data[key]
            elif isinstance(data, list):
                return data
        except:
            pass
        
        return [exec_hosts_str]
    
    def _extract_key_value(self, exec_hosts_str: str, key: str) -> List[str]:
        """提取键值对中的值"""
        pattern = f'{key}=([a-zA-Z0-9_.-]+)'
        matches = re.findall(pattern, exec_hosts_str, re.IGNORECASE)
        
        if matches:
            return matches
        
        return [exec_hosts_str]
    
    def _fallback_parsing(self, exec_hosts_str: str) -> List[str]:
        """回退解析方法"""
        separators = [',', '+', ';', '|', ' ', '\t']
        
        for sep in separators:
            if sep in exec_hosts_str:
                parts = [part.strip() for part in exec_hosts_str.split(sep)]
                if len(parts) > 1:
                    return parts
        
        return [exec_hosts_str.strip()]
    
    def _calculate_resources(self, node_list: List[str]) -> Dict[str, Any]:
        """计算资源信息"""
        total_cpu_cores = 0
        total_gpu_count = 0
        subcluster_distribution = defaultdict(int)
        
        for node in node_list:
            # 识别节点所属的子集群
            subcluster = self._identify_node_subcluster(node)
            subcluster_distribution[subcluster] += 1
            
            # 累加资源
            if subcluster in self.subcluster_specs:
                specs = self.subcluster_specs[subcluster]
                total_cpu_cores += specs['cpu_cores_per_node']
                total_gpu_count += specs['gpu_count_per_node']
        
        return {
            'node_list': node_list,
            'node_count': len(node_list),
            'total_cpu_cores': total_cpu_cores,
            'total_gpu_count': total_gpu_count,
            'subcluster_distribution': dict(subcluster_distribution)
        }
    
    def _identify_node_subcluster(self, node_name: str) -> str:
        """识别节点所属的子集群"""
        for prefix, subcluster in self.prefix_to_subcluster.items():
            if node_name.startswith(prefix):
                return subcluster
        
        return 'unknown'
    
    def get_success_rate(self) -> float:
        """获取解析成功率"""
        if self.parsing_stats['total_processed'] == 0:
            return 0.0
        
        return self.parsing_stats['successful_parsed'] / self.parsing_stats['total_processed']
