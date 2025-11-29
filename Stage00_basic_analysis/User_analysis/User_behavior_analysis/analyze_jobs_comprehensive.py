#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
作业数据综合分析工具
针对超算/集群作业数据分析并识别LLM任务
"""

import os
import re
import time
import json
import logging
import random
import warnings
import threading
import multiprocessing as mp
from datetime import datetime
from collections import defaultdict, Counter
from functools import partial  # 添加这一行
import math

import pandas as pd
import numpy as np
from tqdm.auto import tqdm as auto_tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('JobAnalyzer')

# 默认函数
def default_dict_int():
    return defaultdict(int)

def default_dict_float():
    return defaultdict(float)

def default_dict_list():
    return defaultdict(list)

def default_llm_dict():
    return {"llm_count": 0, "non_llm_count": 0}

# GPU加速支持检测
try:
    import cudf
    import cuml
    from cuml.feature_extraction.text import TfidfVectorizer as CumlTfidfVectorizer
    HAS_GPU = True
except (ImportError, ModuleNotFoundError):
    HAS_GPU = False
    logger.info("未检测到GPU加速库，将使用CPU模式")


class ResourceManager:
    """资源管理类：处理系统资源监控和提取"""
    
    def __init__(self):
        # 资源列名映射
        self.column_mappings = {
            'gpu': ['gpu', 'ngpus', 'gpu_num', 'ngpu', 'gpus', 'requested_gpus', 'used_gpus'],
            'memory': ['mem', 'memory', 'mem_gb', 'ram', 'requested_mem', 'used_mem', 'max_mem'],
            'runtime': ['runtime', 'walltime', 'elapsed', 'time', 'cputime', 'wall_clock', 'elapsed_time'],
            'cpu': ['cpu', 'ncpus', 'cores', 'processors', 'slots', 'requested_cpus', 'used_cpus'],
            'gpu_memory': ['gpu_mem', 'gpu_memory', 'gmem', 'gpu_ram', 'max_gpu_mem']
        }
        
        # 资源使用情况
        self.resource_usage = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.start_time = None
    
    def start_monitoring(self):
        """启动资源监控"""
        self.monitoring_active = True
        self.start_time = datetime.now()
        self.resource_usage = []
        
        # 创建监控线程
        monitor_thread = threading.Thread(target=self._monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        return monitor_thread
    
    def stop_monitoring(self):
        """停止资源监控"""
        self.monitoring_active = False
        
    def _monitor_resources(self):
        """资源监控线程函数"""
        try:
            import psutil
            
            while self.monitoring_active:
                # 获取CPU使用率
                cpu_percent = psutil.cpu_percent(interval=1)
                
                # 获取内存使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used = memory.used / (1024 * 1024 * 1024)  # 转换为GB
                
                # 记录时间戳和资源使用情况
                timestamp = (datetime.now() - self.start_time).total_seconds()
                self.resource_usage.append({
                    'timestamp': timestamp,
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_used': memory_used
                })
                
                # 每5秒记录一次
                time.sleep(5)
        except ImportError:
            logger.warning("未安装psutil库，无法监控系统资源")
            return
        except Exception as e:
            logger.error(f"监控资源时出错: {str(e)}")
            return
    
    def extract_resource_value(self, row, resource_type, columns=None):
        """从数据行中提取特定资源值并标准化
        
        Args:
            row: 数据行
            resource_type: 资源类型 ('gpu', 'memory', 'cpu', 'runtime', 'gpu_memory')
            columns: 可选，指定要查找的列名列表
            
        Returns:
            标准化的资源值或None
        """
        # 使用提供的列或默认映射
        search_columns = columns if columns else self.column_mappings.get(resource_type, [])
        
        # 查找匹配的列
        for col in search_columns:
            if col in row.index and pd.notna(row[col]):
                return self._normalize_resource_value(row[col], resource_type)
        
        return None
    
    def _normalize_resource_value(self, value, resource_type):
        """标准化资源值
        
        Args:
            value: 原始值
            resource_type: 资源类型
            
        Returns:
            标准化后的值（浮点数）
        """
        # 如果已经是数字，直接返回
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # 应用特定资源的标准化逻辑
            if resource_type == 'memory' and value > 10000:
                return value / 1024  # 大值可能是MB，转换为GB
            if resource_type == 'runtime' and value > 10000:
                return value / 3600  # 大值可能是秒，转换为小时
            return float(value)
        
        # 处理字符串值
        if isinstance(value, str):
            value_str = value.lower().strip()
            
            # 内存处理
            if resource_type == 'memory' or resource_type == 'gpu_memory':
                # 提取数字部分
                match = re.search(r'(\d+\.?\d*)', value_str)
                if not match:
                    return None
                
                num_value = float(match.group(1))
                
                # 单位转换为GB
                if 'tb' in value_str or 't' in value_str:
                    return num_value * 1024
                elif 'gb' in value_str or 'g' in value_str:
                    return num_value
                elif 'mb' in value_str or 'm' in value_str:
                    return num_value / 1024
                elif 'kb' in value_str or 'k' in value_str:
                    return num_value / (1024 * 1024)
                else:
                    # 没有单位时的默认处理
                    if num_value > 10000:  # 大数可能是MB
                        return num_value / 1024
                    return num_value
            
            # 运行时间处理
            elif resource_type == 'runtime':
                # 时分秒格式 (HH:MM:SS)
                if ':' in value_str:
                    parts = value_str.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        return int(parts[0]) + int(parts[1])/60 + int(parts[2])/3600
                    elif len(parts) == 2:  # MM:SS or HH:MM
                        return int(parts[0])/60 + int(parts[1])/3600
                
                # 数字+单位格式
                match = re.search(r'(\d+\.?\d*)\s*([a-z]*)', value_str)
                if match:
                    num_value = float(match.group(1))
                    unit = match.group(2)
                    
                    if 'day' in unit or 'd' == unit:
                        return num_value * 24  # 天转小时
                    elif 'hour' in unit or 'hr' in unit or 'h' == unit:
                        return num_value  # 已是小时
                    elif 'min' in unit or 'm' == unit:
                        return num_value / 60  # 分钟转小时
                    elif 'sec' in unit or 's' == unit:
                        return num_value / 3600  # 秒转小时
                
                # 纯数字处理
                try:
                    num_value = float(re.sub(r'[^\d.]', '', value_str))
                    # 如果是大数，可能是秒
                    if num_value > 10000:
                        return num_value / 3600  # 转为小时
                    return num_value
                except:
                    return None
            
            # CPU和GPU处理
            else:
                try:
                    return float(re.sub(r'[^\d.]', '', value_str))
                except:
                    return None
        
        return None

    def get_all_resources(self, row, exclude_cols=None):
        """从一行数据中提取所有可能的资源信息
        
        Args:
            row: 数据行
            exclude_cols: 要排除的列
            
        Returns:
            资源信息字典
        """
        exclude_cols = exclude_cols or []
        result = {}
        
        # 尝试提取所有资源类型
        for resource_type in self.column_mappings:
            value = self.extract_resource_value(row, resource_type)
            if value is not None:
                result[resource_type] = value
        
        return result
    
    def calculate_runtime_from_timestamps(self, start_time, end_time):
        """从开始和结束时间计算运行时间（小时）
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            运行时间（小时）或None
        """
        if pd.isna(start_time) or pd.isna(end_time):
            return None
        
        try:
            # 确保是datetime类型
            if not isinstance(start_time, pd.Timestamp):
                start_time = pd.to_datetime(start_time)
            if not isinstance(end_time, pd.Timestamp):
                end_time = pd.to_datetime(end_time)
            
            # 计算时间差（小时）
            delta = (end_time - start_time).total_seconds() / 3600
            
            # 验证合理性
            if delta < 0 or delta > 8760:  # 不超过1年
                return None
                
            return delta
        except:
            return None
    
    def generate_resource_report(self, results_dir, timestamp):
        """生成资源使用报告"""
        try:
            # 检查是否有资源使用数据
            if not hasattr(self, 'resource_usage') or len(self.resource_usage) == 0:
                logger.warning("没有收集到资源使用数据，跳过资源报告生成")
                return
            
            # 收集数据
            timestamps = [entry['timestamp'] for entry in self.resource_usage]
            cpu_usage = [entry['cpu_percent'] for entry in self.resource_usage]
            memory_percent = [entry['memory_percent'] for entry in self.resource_usage]
            memory_used = [entry['memory_used'] for entry in self.resource_usage]
            
            # 生成资源报告TXT
            report_path = os.path.join(results_dir, f"resource_usage_{timestamp}.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=================================================\n")
                f.write("系统资源使用报告\n")
                f.write("=================================================\n\n")
                
                f.write("资源使用摘要:\n")
                f.write("-------------------------------------------------\n")
                f.write(f"数据收集开始: {timestamps[0]}\n")
                f.write(f"数据收集结束: {timestamps[-1]}\n")
                f.write(f"数据点数量: {len(timestamps)}\n\n")
                
                f.write("CPU使用情况:\n")
                f.write(f"  平均: {np.mean(cpu_usage):.1f}%\n")
                f.write(f"  最大: {np.max(cpu_usage):.1f}%\n")
                f.write(f"  最小: {np.min(cpu_usage):.1f}%\n\n")
                
                f.write("内存使用情况:\n")
                f.write(f"  平均使用率: {np.mean(memory_percent):.1f}%\n")
                f.write(f"  最大使用率: {np.max(memory_percent):.1f}%\n")
                f.write(f"  平均使用量: {np.mean(memory_used):.1f} GB\n")
                f.write(f"  最大使用量: {np.max(memory_used):.1f} GB\n\n")
                
                f.write("=================================================\n")
                f.write("详细资源使用数据 (每5分钟采样)\n")
                f.write("=================================================\n\n")
                
                f.write("时间戳,CPU使用率(%),内存使用率(%),内存使用量(GB)\n")
                
                # 每5个数据点记录一个
                for i in range(0, len(timestamps), 5):
                    ts = timestamps[i].strftime("%Y-%m-%d %H:%M:%S")
                    cpu = cpu_usage[i]
                    mem_pct = memory_percent[i]
                    mem_gb = memory_used[i]
                    f.write(f"{ts},{cpu:.1f},{mem_pct:.1f},{mem_gb:.1f}\n")
            
            logger.info(f"资源使用报告已保存到: {report_path}")
            
        except Exception as e:
            logger.error(f"生成资源报告出错: {str(e)}")


class KeywordProcessor:
    """关键词处理器，用于识别与LLM相关的关键词"""
    
    def __init__(self, min_count=5, min_length=3):
        """初始化关键词处理器
        
        Args:
            min_count: 最小词频
            min_length: 最小词长
        """
        self.keywords = {}
        self.llm_keywords = set()
        self.non_llm_keywords = set()
        self.keyword_counts = {}
        self.min_count = min_count
        self.min_length = min_length
        self.llm_affinities = {}
    
    def process_keywords(self, keywords_freq):
        """处理关键词频率字典
        
        Args:
            keywords_freq: 关键词频率字典
        """
        # 过滤低频词
        filtered_keywords = {k: v for k, v in keywords_freq.items() 
                           if v >= self.min_count and len(k) >= self.min_length}
        
        # 更新频率
        self.keyword_counts.update(filtered_keywords)
        
        # 初始化亲和度
        for keyword in filtered_keywords:
            if keyword not in self.llm_affinities:
                self.llm_affinities[keyword] = 0.0
                
        logger.info(f"处理了 {len(filtered_keywords)} 个关键词")
    
    def update_keyword_llm_affinity(self, keyword, is_llm, weight=1.0):
        """更新关键词的LLM亲和度
        
        Args:
            keyword: 关键词
            is_llm: 是否LLM相关
            weight: 权重
        """
        if keyword not in self.keyword_counts:
            return
            
        # 简单线性调整
        delta = 0.1 * weight
        if is_llm:
            self.llm_affinities[keyword] += delta
        else:
            self.llm_affinities[keyword] -= delta
        
        # 确保在[0,1]范围内
        self.llm_affinities[keyword] = max(0.0, min(1.0, self.llm_affinities[keyword]))
        
        # 更新关键词集合
        if self.llm_affinities[keyword] > 0.7:
            self.llm_keywords.add(keyword)
            if keyword in self.non_llm_keywords:
                self.non_llm_keywords.remove(keyword)
        elif self.llm_affinities[keyword] < 0.3:
            self.non_llm_keywords.add(keyword)
            if keyword in self.llm_keywords:
                self.llm_keywords.remove(keyword)
    
    def get_keywords_by_affinity(self, threshold=0.7):
        """获取高于阈值的LLM关键词
        
        Args:
            threshold: 亲和度阈值
            
        Returns:
            高亲和度关键词集合
        """
        return {k for k, v in self.llm_affinities.items() if v >= threshold}
    
    def get_top_keywords(self, top_n=20):
        """获取前N个最相关的LLM关键词
        
        Args:
            top_n: 返回的关键词数量
            
        Returns:
            (关键词, (亲和度, 频率)) 的列表
        """
        # 根据亲和度和频率排序
        keywords = [(k, (v, self.keyword_counts.get(k, 0))) 
                   for k, v in self.llm_affinities.items()]
        keywords.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
        return keywords[:top_n]
    
    def analyze_command(self, command):
        """分析命令中的LLM关键词
        
        Args:
            command: 命令字符串
            
        Returns:
            (匹配的LLM关键词集合, 匹配的非LLM关键词集合)
        """
        if not command or not isinstance(command, str):
            return set(), set()
            
        command = command.lower()
        words = re.findall(r'\b[a-z0-9_-]{3,}\b', command)
        
        llm_matched = {word for word in words if word in self.llm_keywords}
        non_llm_matched = {word for word in words if word in self.non_llm_keywords}
        
        return llm_matched, non_llm_matched
    
    def extract_keywords(self, text):
        """从文本中提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            提取的关键词集合
        """
        if not text or not isinstance(text, str):
            return set()
            
        text = text.lower()
        words = re.findall(r'\b[a-z0-9_-]{3,}\b', text)
        
        # 返回文本中与已知关键词匹配的词集合
        extracted = {word for word in words if word in self.keyword_counts}
        
        return extracted


class JobAnalyzer:
    """作业数据分析器"""
    
    def __init__(self, max_workers=None, batch_size=1000, use_gpu=True):
        """初始化作业数据分析器
        
        Args:
            max_workers: 最大工作进程数
            batch_size: 批处理大小
            use_gpu: 是否使用GPU加速
        """
        # 设置工作进程数
        if max_workers is None:
            # 自动设置为CPU核心数
            max_workers = mp.cpu_count()
        self.max_workers = max_workers
        
        # 批处理大小
        self.batch_size = batch_size
        
        # 检查GPU可用性
        self.use_gpu = use_gpu
        self.gpu_available = False
        if use_gpu:
            self.gpu_available = self._check_gpu_availability()
            
        # 初始化关键词处理器
        self.keyword_processor = KeywordProcessor(min_count=5, min_length=3)
        
        # 初始化资源管理器
        self.resource_manager = ResourceManager()
        
        # 初始化LLM资源模式
        self.llm_resource_patterns = self._init_llm_resource_patterns()
        
        # 统一的资源列名映射
        self.resource_column_mappings = {
            'gpu': ['gpu_num', 'gpu', 'ngpus', 'ngpu', 'gpus'],  # 确保gpu_num排在前面
            'memory': ['mem', 'memory', 'mem_gb', 'ram', 'requested_mem'],
            'runtime': ['runtime', 'walltime', 'elapsed', 'time', 'cputime'],
            'cpu': ['num_processors', 'cpu', 'ncpus', 'cores', 'processors', 'slots'],  # 确保num_processors排在前面
            'gpu_memory': ['gpu_mem', 'gpu_memory', 'gmem']  # 确保gpu_mem排在前面
        }
        
        # 常见LLM任务类型
        self.task_types = {
            'training': ['train', 'finetune', 'pretrain', 'optimize'],
            'inference': ['infer', 'predict', 'generate', 'eval', 'serve'],
            'data_processing': ['process', 'prepare', 'clean', 'format'],
            'evaluation': ['evaluate', 'benchmark', 'test', 'validate'],
            'unknown': []
        }
        
        # 资源监控相关
        self.monitoring_active = False
        self.monitor_thread = None
        self.resource_usage = []
        
        # 分析状态
        self.analyzed_chunks = 0
        self.total_chunks = 0
        
        # 常见模型名称
        self.llm_model_names = [
            'gpt', 'bert', 'transformer', 't5', 'llama', 'falcon', 'mistral', 
            'bloom', 'glm', 'galactica', 'roberta', 'claude', 'palm', 'gemini',
            'megatron', 'qwen', 'baichuan', 'chatglm', 'ernie', 'vicuna', 'alpaca',
            'pythia', 'llava', 'stable', 'bard', 'phi', 'mpt'
        ]
        
        logger.info(f"初始化JobAnalyzer: 最大进程数={max_workers}, 使用GPU={self.gpu_available}")
        
        # 标记是否已学习关键词，避免重复学习
        self.keywords_learned = False
    
    def _configure_logging(self):
        """配置日志系统"""
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        
        # 设置日志级别
        logger.setLevel(logging.INFO)
        logger.addHandler(console_handler)
        
        # 删除可能的重复处理器
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler != console_handler:
                logger.removeHandler(handler)
    
    def _check_gpu_availability(self):
        """检查GPU是否可用
        
        Returns:
            bool: GPU是否可用
        """
        try:
            # 尝试导入CUDA库
            import cudf
            import cuml
            logger.info("检测到RAPIDS库，启用GPU加速")
            return True
        except ImportError:
            try:
                # 尝试导入PyTorch并检查CUDA可用性
                import torch
                if torch.cuda.is_available():
                    logger.info("检测到PyTorch CUDA可用，启用GPU加速")
                    return True
                else:
                    logger.info("PyTorch CUDA不可用")
                    return False
            except ImportError:
                logger.info("未安装GPU加速库")
                return False
    
    def _init_llm_resource_patterns(self):
        """初始化LLM资源模式列表
        
        Returns:
            LLM资源模式列表
        """
        # 定义资源模式列表
        return [
            # 大规模训练
            {
                'evidence': 'LLM大规模训练',
                'min_gpu': 8, 
                'min_gpu_memory': 32,
                'min_memory': 256,
                'min_cpu': 32,  # 添加CPU要求
                'min_runtime_hours': 24,
                'confidence': 0.95
            },
            # 中等规模训练
            {
                'evidence': 'LLM中等规模训练',
                'min_gpu': 4,
                'min_gpu_memory': 16,
                'min_memory': 128,
                'min_cpu': 16,  # 添加CPU要求
                'min_runtime_hours': 12,
                'confidence': 0.85
            },
            # 小规模训练或微调
            {
                'evidence': 'LLM小规模训练/微调',
                'min_gpu': 1,
                'min_gpu_memory': 24,
                'min_memory': 64,
                'min_cpu': 8,  # 添加CPU要求
                'min_runtime_hours': 4,
                'confidence': 0.75
            },
            # 大规模推理
            {
                'evidence': 'LLM大规模推理',
                'min_gpu': 2,
                'min_gpu_memory': 24,
                'min_memory': 64,
                'min_runtime_hours': 6,
                'confidence': 0.8
            },
            # 标准推理
            {
                'evidence': 'LLM标准推理',
                'min_gpu': 1,
                'min_gpu_memory': 16,
                'min_memory': 32,
                'confidence': 0.7
            }
        ]
    
    def analyze_job_dataset(self, filepath, output_dir=None, n_workers=None, batch_size=None):
        """分析作业数据集
        
        Args:
            filepath: 输入CSV文件路径
            output_dir: 输出目录
            n_workers: 工作进程数(None表示使用self.max_workers)
            batch_size: 批处理大小(None表示使用self.batch_size)
        
        Returns:
            分析结果或None(出错)
        """
        try:
            logger.info(f"开始分析数据集: {filepath}")
            
            # 设置工作进程数
            if n_workers is None:
                n_workers = self.max_workers
            logger.info(f"使用{n_workers}个工作进程")
            
            # 设置批处理大小
            if batch_size is None:
                batch_size = self.batch_size
            
            # 读取数据并分块 - 只读取一次并保存
            logger.info(f"开始读取数据文件: {filepath}")
            chunks = self._read_and_chunk_data(filepath, batch_size)
            if not chunks:
                logger.error("数据读取失败")
                return None
            
            # 学习关键词
            self._learn_keywords(chunks)
            
            # 设置关键词已学习标志
            self.keywords_learned = True
            
            # 开始分析 - 传入已读取的数据块，避免重复读取
            results = self.analyze_data(filepath, output_dir=output_dir, chunks=chunks)
            return results
        
        except Exception as e:
            logger.error(f"分析过程出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _safe_execute(self, func, *args, critical=False, default_return=None, error_msg="操作失败", **kwargs):
        """统一的错误处理包装器
        
        Args:
            func: 要执行的函数
            args, kwargs: 传递给函数的参数
            critical: 是否为关键操作（出错时抛出异常）
            default_return: 出错时的默认返回值
            error_msg: 出错时的错误消息前缀
            
        Returns:
            函数执行结果或默认值
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if critical:
                logger.error(f"{error_msg}: {str(e)}")
                raise
            else:
                logger.warning(f"{error_msg}: {str(e)}")
                return default_return
    
    def _read_and_chunk_data(self, filepath, chunk_size):
        """读取CSV数据并分块
        
        Args:
            filepath: CSV文件路径
            chunk_size: 块大小
            
        Returns:
            数据块列表
        """
        try:
            logger.info(f"开始读取数据文件: {filepath}")
            
            # 设置固定的总行数 - 不再计算
            total_rows = 8_877_313
            logger.info(f"数据集总行数: {total_rows:,}")
            
            # 计算预期的块数
            expected_chunks = (total_rows + chunk_size - 1) // chunk_size
            logger.info(f"预期数据块数: {expected_chunks}")
            
            # 使用pandas读取CSV
            chunks = []
            reader = pd.read_csv(filepath, chunksize=chunk_size)
            
            # 读取数据块
            with auto_tqdm(total=expected_chunks, desc="读取数据") as pbar:
                for i, chunk in enumerate(reader):
                    chunks.append(chunk)
                    pbar.update(1)
                    if (i+1) % 10 == 0:
                        logger.info(f"已读取 {(i+1)*chunk_size:,} 行 ({(i+1)/expected_chunks*100:.1f}%)")
            
            # 检查实际读取的行数
            actual_rows = sum(len(chunk) for chunk in chunks)
            logger.info(f"实际读取行数: {actual_rows:,}")
            
            return chunks
        except Exception as e:
            logger.error(f"读取数据出错: {str(e)}")
            return []

    def _parallel_process(self, items, process_func, n_workers=None, desc="处理中"):
        """通用并行处理框架
        
        Args:
            items: 要处理的项目列表
            process_func: 处理函数，接收单个项目返回处理结果
            n_workers: 并行进程数(默认使用self.max_workers)
            desc: 进度条描述
            
        Returns:
            处理结果列表
        """
        if n_workers is None:
            n_workers = self.max_workers
            
        n_workers = min(n_workers, len(items))  # 不要超过项目数
        
        results = []
        
        if n_workers <= 1:
            # 单进程处理
            with auto_tqdm(items, desc=f"{desc} (单进程)") as pbar:
                for item in pbar:
                    result = process_func(item)
                    results.append(result)
        else:
            # 多进程处理
            try:
                # 创建进程池
                with mp.Pool(processes=n_workers) as pool:
                    # 使用imap执行带进度条的并行处理
                    with auto_tqdm(total=len(items), desc=f"{desc} ({n_workers}进程)") as pbar:
                        for result in pool.imap(process_func, items):
                            results.append(result)
                            pbar.update()
            except Exception as e:
                logger.error(f"并行处理出错: {str(e)}")
                logger.info("切换到单进程模式")
                
                # 如果出错，回退到单进程
                results = []
                with auto_tqdm(items, desc=f"{desc} (单进程回退)") as pbar:
                    for item in pbar:
                        result = process_func(item)
                        results.append(result)
        
        return results
    
    def _learn_from_chunk(self, chunk):
        """从单个数据块中学习关键词
        
        Args:
            chunk: 数据块
            
        Returns:
            关键词频率字典
        """
        # 找出命令字段
        command_field = self._find_command_field(chunk)
        if not command_field:
            return {}
            
        # 统计关键词频率
        keyword_freq = defaultdict(int)
        
        # 处理每个命令
        for cmd in chunk[command_field].fillna('').astype(str):
            # 提取关键词
            cmd = cmd.lower()
            words = re.findall(r'\b[a-z0-9_-]{3,}\b', cmd)
            
            # 统计词频
            for word in words:
                keyword_freq[word] += 1
        
        return keyword_freq
    
    def _learn_keywords(self, chunks):
        """从数据块学习关键词
        
        Args:
            chunks: 数据块列表
        """
        logger.info("开始学习关键词...")
        
        try:
            # 使用并行处理
            results = self._parallel_process(
                chunks, 
                self._learn_from_chunk,
                desc="学习关键词"
            )
            
            # 合并结果
            all_keywords = defaultdict(int)
            for keyword_freq in results:
                for keyword, freq in keyword_freq.items():
                    all_keywords[keyword] += freq
            
            # 更新关键词处理器
            self.keyword_processor.process_keywords(all_keywords)
            
            logger.info(f"学习了 {len(all_keywords)} 个关键词")
            
        except Exception as e:
            logger.error(f"学习关键词时出错: {str(e)}")
            # 尝试单进程处理
            all_keywords = defaultdict(int)
            
            with auto_tqdm(chunks, desc="学习关键词 (单进程)") as pbar:
                for chunk in pbar:
                    keyword_freq = self._learn_from_chunk(chunk)
                    for keyword, freq in keyword_freq.items():
                        all_keywords[keyword] += freq
            
            # 更新关键词处理器
            self.keyword_processor.process_keywords(all_keywords)
            
            logger.info(f"单进程学习了 {len(all_keywords)} 个关键词")
    
    def _process_chunks(self, chunks, n_workers=None):
        """处理所有数据块并执行分类
        
        Args:
            chunks: 数据块列表
            n_workers: 工作进程数
            
        Returns:
            包含所有作业的数据框
        """
        logger.info("开始处理和分类数据块...")
        
        # 定义块处理函数
        def process_chunk(chunk):
            return self._classify_jobs(chunk)
        
        # 并行处理所有块
        processed_chunks = self._parallel_process(
            chunks, 
            process_chunk, 
            n_workers=n_workers,
            desc="处理数据块"
        )
        
        # 合并结果
        logger.info("合并处理结果...")
        all_jobs = pd.concat([chunk for chunk in processed_chunks if chunk is not None], 
                            ignore_index=True)
        
        logger.info(f"处理完成，共{len(all_jobs)}个作业")
        return all_jobs
    
    def _find_command_field(self, df):
        """在数据框中查找命令字段
        
        Args:
            df: 数据框
            
        Returns:
            命令字段名或None
        """
        # 可能的命令字段名列表
        possible_cmd_fields = ['command', 'cmd', 'job_cmd', 'exec_cmd', 'script', 
                            'job_script', 'cmdline', 'args', 'arguments']
        
        # 检查列名
        for col in df.columns:
            if col.lower() in possible_cmd_fields or any(cmd in col.lower() for cmd in possible_cmd_fields):
                return col
        
        # 如果没有明确的命令列，尝试分析列内容寻找可能的命令列
        for col in df.columns:
            # 跳过数值列和ID列
            if df[col].dtype in [np.int64, np.float64] or 'id' in col.lower():
                continue
                
            # 抽样检查该列是否包含命令特征
            if df[col].dtype == object:
                sample = df[col].dropna().sample(min(100, len(df))).astype(str)
                
                # 计算可能的命令行特征
                cmd_patterns = [
                    r'python|bash|sbatch|qsub|srun|perl|java|gcc|mpirun',
                    r'\.py|\.sh|\.pl|\.java|\.c|\.cpp',
                    r'-\w+\s+--\w+',
                    r'/\w+/\w+/\w+'
                ]
                
                pattern_matches = 0
                for pattern in cmd_patterns:
                    if sample.str.contains(pattern, regex=True).any():
                        pattern_matches += 1
                
                # 如果匹配多个特征，可能是命令列
                if pattern_matches >= 2:
                    return col
        
        logger.warning(f"未找到可能的命令字段，分类可能仅基于资源指标")
        return None
    
    def _extract_keywords(self, text):
        """从文本中提取关键词
        
        Args:
            text: 输入文本
            
        Returns:
            关键词集合
        """
        return self.keyword_processor.extract_keywords(text)
    
    def _classify_jobs(self, df):
        """使用学习到的规则对作业进行分类
        
        Args:
            df: 输入数据框
            
        Returns:
            分类后的数据框
        """
        logger.info("开始作业分类...")
        
        # 复制数据框以避免修改原始数据
        result_df = df.copy()
        
        # 添加分类字段
        result_df['is_llm'] = False
        result_df['llm_confidence'] = 0.0
        result_df['task_type'] = 'Unknown'
        result_df['evidence'] = ''
        result_df['confidence'] = 0.0
        
        # 提取和预处理命令字段
        command_field = self._find_command_field(result_df)
        
        if command_field:
            logger.info(f"使用字段 '{command_field}' 作为命令分类依据")
            
            # 按命令分类
            classified = 0
            for idx, row in result_df.iterrows():
                cmd = str(row[command_field]) if pd.notna(row[command_field]) else ""
                
                # 提取关键词
                keywords = self._extract_keywords(cmd)
                
                # 判断是否是LLM任务
                is_llm, confidence, evidence = self._determine_if_llm_by_keywords(keywords, cmd)
                
                if is_llm:
                    result_df.loc[idx, 'is_llm'] = True
                    result_df.loc[idx, 'llm_confidence'] = confidence
                    result_df.loc[idx, 'task_type'] = 'LLM'
                    result_df.loc[idx, 'evidence'] = evidence
                    result_df.loc[idx, 'confidence'] = confidence
                    classified += 1
                elif confidence > 0.5:  # 有足够置信度判定为非LLM
                    result_df.loc[idx, 'task_type'] = 'Non_LLM'
                    result_df.loc[idx, 'evidence'] = evidence
                    result_df.loc[idx, 'confidence'] = confidence
                    classified += 1
            
            logger.info(f"基于命令分类了 {classified} 个作业")
            
            # 找出仍然未分类的记录
            unknown_jobs = result_df[result_df['task_type'] == 'Unknown'].copy()
            logger.info(f"发现 {len(unknown_jobs)} 个未能通过命令分类的作业")
            
            # 如果有未分类的记录，尝试通过资源使用情况来分类
            if len(unknown_jobs) > 0:
                classified_by_resources = self._classify_unknown_by_resources(unknown_jobs)
                
                # 更新原始数据框中的未知记录
                for idx, row in classified_by_resources.iterrows():
                    if row['is_llm']:
                        result_df.loc[idx, 'is_llm'] = True
                        result_df.loc[idx, 'llm_confidence'] = row['llm_confidence']
                        result_df.loc[idx, 'task_type'] = 'LLM'
                        result_df.loc[idx, 'evidence'] = row['evidence']
                        result_df.loc[idx, 'confidence'] = row['confidence']
        else:
            logger.warning("未找到有效的命令字段，将尝试完全基于资源使用进行分类")
            # 尝试完全通过资源使用情况来分类
            classified_by_resources = self._classify_unknown_by_resources(result_df)
            
            # 更新结果
            for idx, row in classified_by_resources.iterrows():
                if row['is_llm']:
                    result_df.loc[idx, 'is_llm'] = True
                    result_df.loc[idx, 'llm_confidence'] = row['llm_confidence']
                    result_df.loc[idx, 'task_type'] = 'LLM'
                    result_df.loc[idx, 'evidence'] = row['evidence']
                    result_df.loc[idx, 'confidence'] = row['confidence']
        
        # 汇总分类结果
        llm_count = (result_df['is_llm'] == True).sum()
        non_llm_count = len(result_df) - llm_count
        
        logger.info(f"分类完成. 总共: {len(result_df)}, LLM: {llm_count}, 非LLM: {non_llm_count}")
        
        return result_df
    
    def _determine_if_llm_by_keywords(self, keywords, full_text):
        """根据关键词判断任务是否为LLM
        
        Args:
            keywords: 提取的关键词集合
            full_text: 完整文本内容
            
        Returns:
            (is_llm, confidence, evidence)元组
        """
        # LLM相关关键词
        llm_indicators = {
            'high': [
                'llm', 'llms', 'gpt', 'gpt-3', 'gpt-4', 'gpt3', 'gpt4',
                'chatgpt', 'claude', 'llama', 'vicuna', 'mistral', 'mixtral',
                'transformer', 'transformers', 'bert', 'roberta', 'openai',
                't5', 'flan-t5', 'qlora', 'lora', 'falcon', 'phi', 'phi-2',
                'gemma', 'gemini', 'palm', 'anthropic', 'huggingface'
            ],
            'medium': [
                'tokenizer', 'tokenize', 'token', 'tokens', 'embedding', 
                'embeddings', 'prompt', 'prompts', 'context_length',
                'pretrained', 'pre-trained', 'finetune', 'fine-tune',
                'attention', 'language-model', 'language_model',
                'decoder', 'diffusion', 'stable-diffusion',
                'generative', 'generate', 'inference'
            ],
            'supporting': [
                'deepspeed', 'megatron', 'accelerate', 'peft', 'rag',
                'bitsandbytes', 'retrieval', 'retriever', 'langchain',
                'llamaindex', 'vllm', 'tgi', 'text-generation-inference',
                'flash-attention', 'flashattention'
            ]
        }
        
        # 非LLM领域相关关键词
        non_llm_indicators = {
            'hpc': [
                'mpi', 'openmpi', 'mpich', 'openmp', 'infiniband', 'scalapack',
                'petsc', 'slurm', 'flux', 'fluent', 'ansys', 'hpc', 'fortran',
                'vasp', 'lammps', 'hdf5', 'checkpoint', 'wrf', 'pbspro', 'pbs'
            ],
            'scientific': [
                'molecular', 'dynamics', 'simulation', 'physics', 'chemistry',
                'fluid', 'genomics', 'bioinformatics', 'monte-carlo',
                'montecarlo', 'cfd', 'molecular-dynamics', 'gaussian'
            ],
            'database': [
                'postgresql', 'mongodb', 'mysql', 'mariadb', 'database',
                'sql', 'nosql', 'query', 'postgres', 'redis', 'sqlite'
            ],
            'web': [
                'django', 'flask', 'fastapi', 'node', 'nodejs', 'express',
                'react', 'vue', 'angular', 'nginx', 'apache', 'http',
                'frontend', 'backend', 'fullstack'
            ]
        }
        
        # 匹配评分
        llm_score = 0
        llm_evidence = []
        non_llm_score = 0
        non_llm_evidence = []
        
        # 评估LLM关键词匹配
        for keyword in keywords:
            if keyword in llm_indicators['high']:
                llm_score += 3
                llm_evidence.append(keyword)
            elif keyword in llm_indicators['medium']:
                llm_score += 2
                llm_evidence.append(keyword)
            elif keyword in llm_indicators['supporting']:
                llm_score += 1
                llm_evidence.append(keyword)
                
            # 评估非LLM领域
            for domain, domain_keywords in non_llm_indicators.items():
                if keyword in domain_keywords:
                    non_llm_score += 1
                    non_llm_evidence.append(f"{keyword}({domain})")
        
        # 额外检查完整文本中的模式
        llm_model_patterns = [
            r'\bgpt-[234]\b', r'\bllama-?[23]\b', r'\bclaude-?[2v]\b',
            r'\bmistral-?7b\b', r'\bchatglm\b', r'\bgemma-?[27]b\b',
            r'\bgemini\b', r'\bmistral-?instruct\b', r'\bphilosopher\b',
            r'\bfalcon-?[1740]\b', r'\bstarcoder\b', r'\bmpt-[357]\b'
        ]
        
        for pattern in llm_model_patterns:
            if re.search(pattern, full_text, re.IGNORECASE):
                match = re.search(pattern, full_text, re.IGNORECASE).group(0)
                llm_score += 5
                llm_evidence.append(f"模型名: {match}")
        
        # 决策逻辑
        is_llm = False
        confidence = 0.0
        evidence = ""
        
        if llm_score > 0:
            # 有LLM特征，计算置信度
            total_score = max(1, llm_score + non_llm_score)
            raw_confidence = min(1.0, llm_score / total_score)
            
            # 调整置信度 - 更倾向于LLM分类
            confidence = 0.5 + (raw_confidence - 0.5) * 1.2
            confidence = min(1.0, max(0.0, confidence))
            
            # 如果有足够的LLM证据且置信度高，则归类为LLM
            if llm_score >= 3 and confidence >= 0.65:
                is_llm = True
                evidence = f"LLM关键词: {', '.join(llm_evidence[:5])}"
                
                if non_llm_evidence:
                    evidence += f" | 存在非LLM特征: {', '.join(non_llm_evidence[:3])}"
            elif llm_score >= 1:
                evidence = f"可能的LLM关键词: {', '.join(llm_evidence[:5])}"
                if non_llm_evidence:
                    evidence += f" | 强烈的非LLM特征: {', '.join(non_llm_evidence[:3])}"
            
        elif non_llm_score > 0:
            # 只有非LLM特征
            confidence = min(1.0, 0.5 + non_llm_score * 0.1)
            evidence = f"非LLM特征: {', '.join(non_llm_evidence[:5])}"
        
        return is_llm, confidence, evidence
    
    def _classify_unknown_by_resources(self, jobs_df):
        """根据资源使用情况分类未知作业
        
        Args:
            jobs_df: 作业数据框
            
        Returns:
            更新了分类的数据框
        """
        logger.info(f"开始基于资源使用模式分类{len(jobs_df)}个未知任务...")
        
        # 统计信息
        classified_count = 0
        llm_count = 0
        
        # 复制以避免修改原始数据
        result_df = jobs_df.copy()
        
        # 预处理：尝试找到和规范化资源列
        resource_cols = {}
        for resource, possible_names in self.resource_manager.column_mappings.items():
            for col in possible_names:
                if col in result_df.columns:
                    resource_cols[resource] = col
                    break
                    
        logger.info(f"找到的资源列: {resource_cols}")
        
        # 如果没有足够的资源信息，尝试从其他列提取
        if len(resource_cols) < 2:
            logger.info("直接资源列不足，尝试从其他列提取资源信息...")
            
        # 遍历每一行作业
        for idx, row in result_df.iterrows():
            # 检查是否符合任何LLM资源模式
            for pattern in self.llm_resource_patterns:
                matched_criteria = []
                match_count = 0
                check_count = 0
                
                # 检查GPU数量 - 这里会优先使用gpu_num列
                if 'min_gpu' in pattern:
                    check_count += 1
                    gpu_count = self._get_resource_value(row, 'gpu')  # 这会使用resource_column_mappings中的映射
                    if gpu_count is not None and gpu_count >= pattern['min_gpu']:
                        match_count += 1
                        matched_criteria.append(f"GPU数量: {gpu_count} >= {pattern['min_gpu']}")
                
                # 检查GPU显存 - 这里会优先使用gpu_mem列
                if 'min_gpu_memory' in pattern:
                    check_count += 1
                    gpu_mem = self._get_resource_value(row, 'gpu_memory')  # 这会使用resource_column_mappings中的映射
                    if gpu_mem is not None and gpu_mem >= pattern['min_gpu_memory']:
                        match_count += 1
                        matched_criteria.append(f"GPU显存: {gpu_mem} >= {pattern['min_gpu_memory']}GB")
                
                # 检查内存
                if 'min_memory' in pattern:
                    check_count += 1
                    memory = self._get_resource_value(row, 'memory')
                    if memory is not None and memory >= pattern['min_memory']:
                        match_count += 1
                        matched_criteria.append(f"系统内存 >= {pattern['min_memory']}GB")
                
                # 检查运行时间
                if 'min_runtime_hours' in pattern:
                    check_count += 1
                    runtime = self._get_resource_value(row, 'runtime')
                    if runtime is not None and runtime >= pattern['min_runtime_hours']:
                        match_count += 1
                        matched_criteria.append(f"运行时间 >= {pattern['min_runtime_hours']}小时")
                
                # 检查CPU核心数
                if 'min_cpu' in pattern:
                    check_count += 1
                    cpu = self._get_resource_value(row, 'cpu')
                    if cpu is not None and cpu >= pattern['min_cpu']:
                        match_count += 1
                        matched_criteria.append(f"CPU核心 >= {pattern['min_cpu']}")
                
                # 计算匹配比例
                if check_count == 0:
                    result_df.loc[idx, 'evidence'] = f"资源信息不足 ({', '.join(matched_criteria)})"
                    continue
                
                match_ratio = match_count / check_count
                
                # 只有匹配比例超过阈值才认为匹配成功
                if match_ratio >= 0.5:
                    # 计算置信度
                    confidence = pattern.get('confidence', 0.7) * match_ratio
                    result_df.loc[idx, 'is_llm'] = True
                    result_df.loc[idx, 'llm_confidence'] = confidence
                    result_df.loc[idx, 'evidence'] = f"{pattern['evidence']} ({', '.join(matched_criteria)})"
                    result_df.loc[idx, 'confidence'] = confidence
                    llm_count += 1
                    classified_count += 1
                    break
                else:
                    result_df.loc[idx, 'evidence'] = f"资源特征不符合LLM模式 ({', '.join(matched_criteria)})"
                    classified_count += 1
            
            # 如果未分类，尝试通过资源使用情况来分类
            if result_df.loc[idx, 'is_llm'] is None:
                result_df.loc[idx, 'task_type'] = 'Non_LLM'
                result_df.loc[idx, 'evidence'] = f"资源特征不符合LLM模式 ({', '.join(matched_criteria)})"
                classified_count += 1
        
        # 汇报分类结果
        logger.info(f"基于资源特征分类完成")
        logger.info(f"总分类: {classified_count} 个任务")
        logger.info(f"识别为LLM任务: {llm_count} ({llm_count/max(1,len(jobs_df))*100:.1f}%)")
        
        return result_df
    
    def _extract_resource_value(self, value_str, resource_type):
        """从字符串提取资源值并标准化
        
        Args:
            value_str: 原始资源值字符串
            resource_type: 资源类型('memory', 'cpu', 'gpu', 'runtime', 'gpu_memory')
            
        Returns:
            标准化的资源值（浮点数）
        """
        if pd.isna(value_str):
            return None
            
        value_str = str(value_str).strip().lower()
        
        # 空值处理
        if not value_str or value_str in ['na', 'n/a', 'none', 'unknown', 'null']:
            return None
            
        # 根据资源类型处理值
        if resource_type == 'memory':
            # 处理内存格式，标准化为GB
            if 'gb' in value_str or 'g' in value_str:
                # 提取数字部分
                match = re.search(r'(\d+(\.\d+)?)', value_str)
                if match:
                    return float(match.group(1))
            elif 'mb' in value_str or 'm' in value_str:
                match = re.search(r'(\d+(\.\d+)?)', value_str)
                if match:
                    return float(match.group(1)) / 1024
            elif 'tb' in value_str or 't' in value_str:
                match = re.search(r'(\d+(\.\d+)?)', value_str)
                if match:
                    return float(match.group(1)) * 1024
            else:
                try:
                    value = float(re.sub(r'[^\d.]', '', value_str))
                    # 大多数GPU内存以GB为单位
                    if value < 1:  # 可能是TB
                        return value * 1024
                    elif value < 1000:  # 可能是GB
                        return value
                    else:  # 可能是MB
                        return value / 1024
                except:
                    return None
                    
        elif resource_type in ['gpu', 'cpu']:
            # 处理计算资源，提取数字
            try:
                # 直接尝试转换
                return int(re.sub(r'[^\d]', '', value_str))
            except:
                return None
                
        elif resource_type == 'gpu_memory':
            # 处理GPU内存，标准化为GB
            if 'gb' in value_str or 'g' in value_str:
                match = re.search(r'(\d+(\.\d+)?)', value_str)
                if match:
                    return float(match.group(1))
            elif 'mb' in value_str or 'm' in value_str:
                match = re.search(r'(\d+(\.\d+)?)', value_str)
                if match:
                    return float(match.group(1)) / 1024
            elif 'tb' in value_str or 't' in value_str:
                match = re.search(r'(\d+(\.\d+)?)', value_str)
                if match:
                    return float(match.group(1)) * 1024
            else:
                try:
                    value = float(re.sub(r'[^\d.]', '', value_str))
                    # 大多数GPU内存以GB为单位
                    if value < 1:  # 可能是TB
                        return value * 1024
                    elif value < 1000:  # 可能是GB
                        return value
                    else:  # 可能是MB
                        return value / 1024
                except:
                    return None
                    
        elif resource_type == 'runtime':
            # 处理运行时间，标准化为小时
            try:
                # 检查常见时间格式
                if ':' in value_str:  # HH:MM:SS 或 DD-HH:MM:SS
                    parts = value_str.replace('-', ':').split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        return (int(parts[0]) + int(parts[1])/60 + int(parts[2])/3600)
                    elif len(parts) == 4:  # DD-HH:MM:SS
                        return (int(parts[0])*24 + int(parts[1]) + int(parts[2])/60 + int(parts[3])/3600)
                # 检查带单位的时间
                elif 'h' in value_str or 'hr' in value_str or 'hour' in value_str:
                    match = re.search(r'(\d+(\.\d+)?)', value_str)
                    if match:
                        return float(match.group(1))
                elif 'd' in value_str or 'day' in value_str:
                    match = re.search(r'(\d+(\.\d+)?)', value_str)
                    if match:
                        return float(match.group(1)) * 24
                elif 'm' in value_str or 'min' in value_str:
                    match = re.search(r'(\d+(\.\d+)?)', value_str)
                    if match:
                        return float(match.group(1)) / 60
                elif 's' in value_str or 'sec' in value_str:
                    match = re.search(r'(\d+(\.\d+)?)', value_str)
                    if match:
                        return float(match.group(1)) / 3600
                else:
                    # 尝试直接解析数字
                    value = float(re.sub(r'[^\d.]', '', value_str))
                    # 根据数值范围猜测单位
                    if value < 10:  # 可能是天
                        return value * 24
                    elif value < 100:  # 可能是小时
                        return value
                    elif value < 10000:  # 可能是分钟
                        return value / 60
                    else:  # 可能是秒
                        return value / 3600
            except:
                return None
        
        return None
    
    def _generate_final_report(self, result_data, results_dir, timestamp, duration):
        """生成最终分析报告
        
        Args:
            result_data: 分析结果数据
            results_dir: 结果保存目录
            timestamp: 时间戳
            duration: 分析用时（秒）
        """
        try:
            logger.info("生成最终分析报告...")
            
            # 准备报告数据
            llm_count = len(result_data['llm_jobs'])
            non_llm_count = len(result_data['non_llm_jobs'])
            total_jobs = len(result_data['all_jobs'])
            
            # 创建报告文件
            report_path = os.path.join(results_dir, f"analysis_report_{timestamp}.txt")
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("==================================================\n")
                f.write(f"  作业数据分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("==================================================\n\n")
                
                # 写入基本统计信息
                f.write("1. 基本统计\n")
                f.write("-----------------\n")
                f.write(f"总作业数: {total_jobs:,}\n")
                f.write(f"LLM作业: {llm_count:,} ({llm_count/total_jobs*100:.2f}%)\n")
                f.write(f"非LLM作业: {non_llm_count:,} ({non_llm_count/total_jobs*100:.2f}%)\n")
                f.write(f"分析用时: {duration:.2f}秒\n\n")
                
                # 写入LLM任务类型分布
                f.write("2. LLM任务类型分布\n")
                f.write("-----------------\n")
                if 'task_distribution' in result_data:
                    for _, row in result_data['task_distribution'].iterrows():
                        f.write(f"{row['task_type']}: {int(row['count']):,} ({row['percentage']:.2f}%)\n")
                else:
                    f.write("未生成任务类型分布数据\n")
                f.write("\n")
                
                # 写入资源使用情况
                f.write("3. 系统资源使用情况\n")
                f.write("-----------------\n")
                if self.resource_manager.resource_usage:
                    # 计算平均值
                    cpu_avg = sum(r['cpu_percent'] for r in self.resource_manager.resource_usage) / len(self.resource_manager.resource_usage)
                    mem_avg = sum(r['memory_percent'] for r in self.resource_manager.resource_usage) / len(self.resource_manager.resource_usage)
                    mem_used_avg = sum(r['memory_used'] for r in self.resource_manager.resource_usage) / len(self.resource_manager.resource_usage)
                    
                    f.write(f"平均CPU使用率: {cpu_avg:.2f}%\n")
                    f.write(f"平均内存使用率: {mem_avg:.2f}%\n")
                    f.write(f"平均内存用量: {mem_used_avg:.2f} GB\n")
                    
                    # 获取最大值
                    cpu_max = max(r['cpu_percent'] for r in self.resource_manager.resource_usage)
                    mem_max = max(r['memory_percent'] for r in self.resource_manager.resource_usage)
                    mem_used_max = max(r['memory_used'] for r in self.resource_manager.resource_usage)
                    
                    f.write(f"最大CPU使用率: {cpu_max:.2f}%\n")
                    f.write(f"最大内存使用率: {mem_max:.2f}%\n")
                    f.write(f"最大内存用量: {mem_used_max:.2f} GB\n")
                else:
                    f.write("未记录资源使用情况\n")
                f.write("\n")
                
                # 写入关键词分析
                f.write("4. 关键LLM识别关键词\n")
                f.write("-----------------\n")
                top_keywords = self.keyword_processor.get_top_keywords(20)
                for keyword, (affinity, count) in top_keywords:
                    f.write(f"{keyword:<30} - 出现频率: {count:>6,} - LLM倾向性: {affinity:.4f}\n")
                f.write("\n")
                
                # 写入代表性LLM任务示例
                f.write("5. 代表性LLM任务示例\n")
                f.write("-----------------\n")
                if not result_data['llm_jobs'].empty:
                    # 选择一些高置信度的LLM任务
                    high_conf_llm = result_data['llm_jobs'].sort_values('llm_confidence', ascending=False).head(5)
                    for idx, row in high_conf_llm.iterrows():
                        f.write(f"任务ID: {idx}\n")
                        f.write(f"分类证据: {row['evidence']}\n")
                        f.write(f"置信度: {row['llm_confidence']:.4f}\n")
                        if 'command' in row and not pd.isna(row['command']):
                            command = str(row['command'])
                            if len(command) > 150:
                                command = command[:147] + "..."
                            f.write(f"命令: {command}\n")
                        f.write("-" * 40 + "\n")
                else:
                    f.write("未分类出LLM任务\n")
                f.write("\n")
                
                # 写入文件信息
                f.write("6. 结果文件\n")
                f.write("-----------------\n")
                files = {
                    "所有任务分类": os.path.join(results_dir, f"all_classifications_{timestamp}.csv"),
                    "LLM任务": os.path.join(results_dir, f"llm_jobs_{timestamp}.csv"),
                    "非LLM任务": os.path.join(results_dir, f"non_llm_jobs_{timestamp}.csv"),
                    "原始LLM任务": os.path.join(results_dir, f"original_llm_{timestamp}.csv"),
                    "原始非LLM任务": os.path.join(results_dir, f"original_non_llm_{timestamp}.csv")
                }
                
                for description, filepath in files.items():
                    if os.path.exists(filepath):
                        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                        record_count = sum(1 for _ in open(filepath, encoding='utf-8')) - 1  # 减去标题行
                        f.write(f"{description}: {filepath}\n")
                        f.write(f"    - 大小: {file_size:.2f} MB\n")
                        f.write(f"    - 记录数: {record_count:,}\n")
                f.write("\n")
                
                # 写入结束信息
                f.write("==================================================\n")
                f.write(f"报告生成于: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("==================================================\n")
                
            logger.info(f"最终分析报告已保存至: {report_path}")
            
        except Exception as e:
            logger.error(f"生成最终报告时出错: {str(e)}")
    
    def sample_test_analysis(self, filepath, sample_size=10000):
        """在数据集的随机样本上进行分析测试
        
        Args:
            filepath: 输入CSV文件路径
            sample_size: 样本大小
            
        Returns:
            测试分析结果
        """
        try:
            logger.info(f"开始对{filepath}进行抽样测试分析")
            logger.info(f"抽样大小: {sample_size}")
            
            # 记录开始时间
            analysis_start = datetime.now()
            
            # 使用固定总行数
            total_rows = 8_877_313
            logger.info(f"数据集总行数: {total_rows:,}")
            
            # 计算抽样比例
            skip_ratio = max(1, total_rows // sample_size)
            
            # 读取数据
            logger.info(f"使用抽样比例 1:{skip_ratio}")
            df = pd.read_csv(filepath, skiprows=lambda i: i > 0 and i % skip_ratio != 0)
            
            # 确保不超过请求的样本大小
            if len(df) > sample_size:
                df = df.sample(sample_size, random_state=42)
            
            logger.info(f"成功读取样本数据: {len(df):,} 行")
            
            # 分析单个数据块
            chunks = [df]
            
            # 学习关键词
            self._learn_keywords(chunks)
            
            # 分类作业
            chunk_classifier = partial(self._classify_chunk, important_columns=None)
            all_jobs = self._parallel_process(chunks, chunk_classifier, n_workers=1, desc="分类样本")
            all_jobs = pd.concat(all_jobs, ignore_index=False)
            
            # 处理结果
            analysis_results = {
                'all_jobs': all_jobs,
                'llm_jobs': all_jobs[all_jobs['is_llm'] == True].copy(),
                'non_llm_jobs': all_jobs[all_jobs['is_llm'] == False].copy()
            }
            
            # 计算任务分布
            task_types = all_jobs['task_type'].value_counts().reset_index()
            task_types.columns = ['task_type', 'count']
            task_types['percentage'] = task_types['count'] / len(all_jobs) * 100
            analysis_results['task_distribution'] = task_types
            
            # 计算分析用时
            analysis_duration = (datetime.now() - analysis_start).total_seconds()
            estimated_full_time = analysis_duration * (total_rows / len(df))
            
            # 显示时间估计
            hours = int(estimated_full_time // 3600)
            minutes = int((estimated_full_time % 3600) // 60)
            seconds = int(estimated_full_time % 60)
            
            logger.info(f"抽样分析完成，用时: {analysis_duration:.2f} 秒")
            logger.info(f"全数据集分析估计时间: {hours}小时 {minutes}分钟 {seconds}秒")
            
            # 显示基本统计
            llm_count = len(analysis_results['llm_jobs'])
            non_llm_count = len(analysis_results['non_llm_jobs'])
            
            logger.info("=" * 50)
            logger.info("抽样分析结果摘要:")
            logger.info(f"总样本数: {len(all_jobs):,}")
            logger.info(f"LLM任务: {llm_count:,} ({llm_count/len(all_jobs)*100:.2f}%)")
            logger.info(f"非LLM任务: {non_llm_count:,} ({non_llm_count/len(all_jobs)*100:.2f}%)")
            logger.info("-" * 50)
            logger.info("任务类型分布:")
            for _, row in analysis_results['task_distribution'].iterrows():
                logger.info(f"{row['task_type']}: {int(row['count']):,} ({row['percentage']:.2f}%)")
            logger.info("=" * 50)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"抽样测试过程中出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _classify_chunk(self, chunk, important_columns=None):
        """对单个数据块进行分类处理
        
        Args:
            chunk: 数据块
            important_columns: 需要保留的重要列(如果为None则保留所有列)
            
        Returns:
            分类后的数据块
        """
        try:
            # 对数据块进行分类
            classified_chunk = self._classify_jobs(chunk)
            
            # 如果指定了需要保留的列，只保留这些列
            if important_columns is not None and isinstance(important_columns, list):
                # 确保必要的分类结果列被保留
                required_cols = ['is_llm', 'llm_confidence', 'task_type', 'evidence', 'confidence']
                keep_cols = list(set(important_columns + required_cols))
                
                # 只保留指定列
                classified_chunk = classified_chunk[keep_cols]
            
            return classified_chunk
        except Exception as e:
            logger.error(f"数据块分类出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return chunk  # 返回原始数据块

    def _check_resource_pattern(self, job_row, pattern):
        """检查作业资源是否符合特定模式
        
        Args:
            job_row: 作业数据行
            pattern: 资源模式
            
        Returns:
            (是否匹配, 置信度, 证据)
        """
        evidence = []
        match_count = 0
        total_checks = 0
        
        # 检查GPU数量
        if 'min_gpu' in pattern:
            total_checks += 1
            gpu_count = self._get_resource_value(job_row, 'gpu')
            if gpu_count is not None and gpu_count >= pattern['min_gpu']:
                match_count += 1
                evidence.append(f"GPU数量: {gpu_count} >= {pattern['min_gpu']}")
        
        # 检查GPU显存
        if 'min_gpu_memory' in pattern:
            total_checks += 1
            gpu_mem = self._get_resource_value(job_row, 'gpu_memory')
            if gpu_mem is not None and gpu_mem >= pattern['min_gpu_memory']:
                match_count += 1
                evidence.append(f"GPU显存: {gpu_mem} >= {pattern['min_gpu_memory']}")
        
        # 检查内存
        if 'min_memory' in pattern:
            total_checks += 1
            memory = self._get_resource_value(job_row, 'memory')
            if memory is not None and memory >= pattern['min_memory']:
                match_count += 1
                evidence.append(f"内存: {memory} >= {pattern['min_memory']}")
        
        # 检查运行时间
        if 'min_runtime_hours' in pattern:
            total_checks += 1
            runtime = self._get_resource_value(job_row, 'runtime')
            if runtime is not None and runtime >= pattern['min_runtime_hours']:
                match_count += 1
                evidence.append(f"运行时间: {runtime} >= {pattern['min_runtime_hours']}")
        
        # 计算匹配比例
        if total_checks == 0:
            return False, 0.0, ""
            
        match_ratio = match_count / total_checks
        
        # 只有匹配比例超过阈值才认为匹配成功
        if match_ratio >= 0.5:
            # 计算置信度
            confidence = pattern.get('confidence', 0.7) * match_ratio
            return True, confidence, pattern['evidence'] + ": " + ", ".join(evidence)
        else:
            return False, 0.0, ""

    def _determine_task_type(self, command):
        """基于命令确定LLM任务类型
        
        Args:
            command: 命令字符串
            
        Returns:
            任务类型
        """
        if not command:
            return "unknown"
            
        command = command.lower()
        
        # 检查每种任务类型的关键词
        for task_type, keywords in self.task_types.items():
            for keyword in keywords:
                if keyword in command:
                    return task_type
        
        # 默认返回未知类型
        return "unknown"

    def _get_resource_value(self, job_row, resource_type):
        """从作业行中提取特定类型的资源数值
        
        Args:
            job_row: 作业数据行
            resource_type: 资源类型
            
        Returns:
            资源数值或None
        """
        # 查找可能的列名
        possible_columns = self.resource_column_mappings.get(resource_type, [])
        
        # 尝试所有可能的列名 - 按照我们在映射中定义的顺序
        for col_name in possible_columns:
            if col_name in job_row.index:
                try:
                    value = job_row[col_name]
                    # 处理可能的非数值
                    if pd.isna(value):
                        continue
                    # 尝试转换为数值
                    return self._normalize_resource_value(value, resource_type)
                except:
                    continue
            
        # 尝试查找包含资源类型的列名
        for col in job_row.index:
            col_lower = str(col).lower()
            if resource_type in col_lower:
                try:
                    value = job_row[col]
                    if pd.isna(value):
                        continue
                    return self._normalize_resource_value(value, resource_type)
                except:
                    continue
        
        return None

    def analyze_data(self, filepath, output_dir=None, save_results=True, chunks=None):
        """分析完整数据集
        
        Args:
            filepath: 输入CSV文件路径
            output_dir: 输出目录
            save_results: 是否保存结果
            chunks: 预先读取的数据块 (可选，如果提供则不再读取文件)
            
        Returns:
            分析结果字典
        """
        try:
            logger.info(f"开始分析数据文件: {filepath}")
            
            # 设置输出目录
            if save_results:
                if output_dir is None:
                    output_dir = os.path.join(os.path.dirname(filepath), "output")
                
                # 创建带时间戳的输出目录
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_subdir = os.path.join(output_dir, f"analysis_{timestamp}")
                os.makedirs(output_subdir, exist_ok=True)
                
                logger.info(f"输出将保存到: {output_subdir}")
            
            # 开始资源监控
            self._start_resource_monitoring()
            
            # 记录开始时间
            start_time = time.time()
            
            # 如果没有提供数据块，才读取数据
            if chunks is None:
                logger.info("读取数据文件...")
                chunks = self._read_and_chunk_data(filepath, self.batch_size)
                if not chunks:
                    logger.error("数据读取失败")
                    return None
                
                # 只有在没有预先学习过关键词时才学习
                if not hasattr(self, 'keywords_learned') or not self.keywords_learned:
                    logger.info("分析前学习关键词...")
                    self._learn_keywords(chunks)
                    self.keywords_learned = True
            else:
                logger.info(f"使用预先读取的{len(chunks)}个数据块，跳过文件读取")
            
            # 记录总块数
            self.total_chunks = len(chunks)
            logger.info(f"数据已分为 {self.total_chunks} 块进行处理")
            
            # 分析所有块
            logger.info("开始对所有数据进行分类...")
            
            # 保存所有标记后的数据
            all_classifications = []
            
            # 分别存储LLM和非LLM任务
            llm_jobs = []
            non_llm_jobs = []
            
            # 新增: 保存原始的LLM和非LLM任务数据
            llm_jobs_raw = []
            non_llm_jobs_raw = []
            
            # 使用并行处理
            self.analyzed_chunks = 0
            
            with auto_tqdm(total=self.total_chunks, desc="分类作业") as pbar:
                for chunk_idx, chunk in enumerate(chunks):
                    # 处理单个块
                    logger.info(f"处理数据块 {chunk_idx+1}/{self.total_chunks}")
                    
                    try:
                        # 对数据块进行分类
                        classified_chunk = self._classify_jobs(chunk)
                        
                        # 更新计数和进度
                        self.analyzed_chunks += 1
                        pbar.update(1)
                        
                        # 分离LLM和非LLM任务
                        llm_chunk = classified_chunk[classified_chunk['is_llm'] == True]
                        non_llm_chunk = classified_chunk[classified_chunk['is_llm'] == False]
                        
                        # 累积结果
                        all_classifications.append(classified_chunk)
                        llm_jobs.append(llm_chunk)
                        non_llm_jobs.append(non_llm_chunk)
                        
                        # 新增: 保存原始的LLM和非LLM任务数据
                        llm_jobs_raw.append(chunk[chunk.index.isin(llm_chunk.index)])
                        non_llm_jobs_raw.append(chunk[chunk.index.isin(non_llm_chunk.index)])
                        
                        # 每10块显示一次进度
                        if (chunk_idx + 1) % 10 == 0 or chunk_idx == len(chunks) - 1:
                            logger.info(f"已处理 {chunk_idx+1}/{self.total_chunks} 块 ({(chunk_idx+1)/self.total_chunks*100:.1f}%)")
                            
                    except Exception as e:
                        logger.error(f"处理数据块 {chunk_idx+1} 时出错: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                        
            # 合并所有结果
            logger.info("合并所有分析结果...")
            
            try:
                all_results = pd.concat(all_classifications, ignore_index=True)
                llm_results = pd.concat(llm_jobs, ignore_index=True)
                non_llm_results = pd.concat(non_llm_jobs, ignore_index=True)
                
                # 新增: 合并原始的LLM和非LLM任务数据
                llm_results_raw = pd.concat(llm_jobs_raw, ignore_index=True)
                non_llm_results_raw = pd.concat(non_llm_jobs_raw, ignore_index=True)
                
                logger.info(f"总计处理了 {len(all_results):,} 个作业")
                logger.info(f"识别出 {len(llm_results):,} 个LLM相关作业 ({len(llm_results)/len(all_results)*100:.2f}%)")
                
                # 计算任务类型分布
                task_distribution = llm_results['task_type'].value_counts()
                task_distribution_percent = task_distribution / len(llm_results) * 100
                
                task_dist_df = pd.DataFrame({
                    'task_type': task_distribution.index,
                    'count': task_distribution.values,
                    'percentage': task_distribution_percent.values
                })
                
                # 停止资源监控
                self._stop_resource_monitoring()
                
                # 计算处理时间
                processing_time = time.time() - start_time
                hours = int(processing_time // 3600)
                minutes = int((processing_time % 3600) // 60)
                seconds = int(processing_time % 60)
                
                logger.info(f"分析完成，总用时: {hours}小时 {minutes}分钟 {seconds}秒")
                
                # 创建分析结果字典
                analysis_results = {
                    'timestamp': timestamp,
                    'all_jobs': all_results,
                    'llm_jobs': llm_results,
                    'non_llm_jobs': non_llm_results,
                    'task_distribution': task_dist_df,
                    'processing_time': processing_time,
                    'llm_percentage': len(llm_results) / len(all_results) * 100,
                    'total_jobs': len(all_results)
                }
                
                # 如果需要保存结果
                if save_results:
                    logger.info("保存分析结果...")
                    
                    # 保存所有作业的分类结果
                    all_output_path = os.path.join(output_subdir, f"all_classifications_{timestamp}.csv")
                    all_results.to_csv(all_output_path, index=False)
                    logger.info(f"所有分类结果已保存到: {all_output_path}")
                    
                    # 保存LLM作业
                    llm_output_path = os.path.join(output_subdir, f"llm_jobs_{timestamp}.csv")
                    llm_results.to_csv(llm_output_path, index=False)
                    logger.info(f"LLM作业已保存到: {llm_output_path}")
                    
                    # 保存非LLM作业
                    non_llm_output_path = os.path.join(output_subdir, f"non_llm_jobs_{timestamp}.csv")
                    non_llm_results.to_csv(non_llm_output_path, index=False)
                    logger.info(f"非LLM作业已保存到: {non_llm_output_path}")
                    
                    # 新增: 保存原始的LLM作业数据
                    llm_raw_output_path = os.path.join(output_subdir, f"llm_jobs_raw_{timestamp}.csv")
                    llm_results_raw.to_csv(llm_raw_output_path, index=False)
                    logger.info(f"LLM作业原始数据已保存到: {llm_raw_output_path}")
                    
                    # 新增: 保存原始的非LLM作业数据
                    non_llm_raw_output_path = os.path.join(output_subdir, f"non_llm_jobs_raw_{timestamp}.csv")
                    non_llm_results_raw.to_csv(non_llm_raw_output_path, index=False)
                    logger.info(f"非LLM作业原始数据已保存到: {non_llm_raw_output_path}")
                    
                    # 保存任务分布
                    task_dist_path = os.path.join(output_subdir, f"task_distribution_{timestamp}.csv")
                    task_dist_df.to_csv(task_dist_path, index=False)
                    logger.info(f"任务类型分布已保存到: {task_dist_path}")
                    
                    # 生成分析报告
                    self._generate_analysis_report(analysis_results, output_subdir, timestamp)
                    
                return analysis_results
                
            except Exception as e:
                logger.error(f"合并结果时出错: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"分析数据时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _start_resource_monitoring(self):
        """启动资源监控
        
        Returns:
            监控线程
        """
        # 如果已经在监控，先停止
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            self.resource_manager.stop_monitoring()
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.join(1.0)
        
        # 启动新的监控
        self.monitoring_active = True
        self.monitor_thread = self.resource_manager.start_monitoring()
        
        return self.monitor_thread

    def _stop_resource_monitoring(self):
        """停止资源监控"""
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            self.resource_manager.stop_monitoring()
            self.monitoring_active = False
            if hasattr(self, 'monitor_thread') and self.monitor_thread:
                self.monitor_thread.join(1.0)
                self.monitor_thread = None
                
        logger.info("已停止资源监控")

    def _normalize_resource_value(self, value, resource_type):
        """标准化资源值
        
        Args:
            value: 原始值
            resource_type: 资源类型
            
        Returns:
            标准化后的值
        """
        try:
            # 如果是字符串，尝试提取数字
            if isinstance(value, str):
                # 提取数字部分
                import re
                num_match = re.search(r'[\d.]+', value)
                if num_match:
                    numeric_value = float(num_match.group())
                else:
                    return None
            else:
                numeric_value = float(value)
                
            # 根据资源类型进行单位转换
            if resource_type == 'memory':
                # 如果数值太小，可能是以GB为单位，否则可能是以MB为单位
                if numeric_value < 100:
                    return numeric_value  # 假设已经是GB
                else:
                    return numeric_value / 1024  # MB转GB
            elif resource_type == 'runtime':
                # 如果数值太小，可能是以小时为单位，否则可能是以分钟或秒为单位
                if numeric_value < 24:
                    return numeric_value  # 假设已经是小时
                elif numeric_value < 1440:  # 小于1440分钟(24小时)
                    return numeric_value / 60  # 分钟转小时
                else:
                    return numeric_value / 3600  # 秒转小时
            else:
                # 对于其他资源类型，直接返回数值
                return numeric_value
                
        except Exception as e:
            logger.warning(f"标准化资源值时出错: {str(e)}")
            return None

    def _generate_analysis_report(self, analysis_results, output_dir, timestamp):
        """生成分析报告
        
        Args:
            analysis_results: 分析结果字典
            output_dir: 输出目录
            timestamp: 时间戳
        """
        try:
            report_path = os.path.join(output_dir, f"analysis_report_{timestamp}.txt")
            
            with open(report_path, 'w') as f:
                f.write(f"LLM作业分析报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                # 总体统计
                total_jobs = analysis_results['total_jobs']
                llm_jobs = len(analysis_results['llm_jobs'])
                non_llm_jobs = len(analysis_results['non_llm_jobs'])
                llm_percent = analysis_results['llm_percentage']
                
                f.write(f"总作业数: {total_jobs:,}\n")
                f.write(f"LLM相关作业: {llm_jobs:,} ({llm_percent:.2f}%)\n")
                f.write(f"非LLM作业: {non_llm_jobs:,} ({100-llm_percent:.2f}%)\n\n")
                
                # 所有任务类型分布
                f.write("所有任务类型分布:\n")
                f.write("-"*50 + "\n")
                
                # 使用安全的字典获取方法，避免KeyError
                if 'all_task_distribution' in analysis_results:
                    all_task_dist = analysis_results['all_task_distribution']
                    for _, row in all_task_dist.iterrows():
                        task = row['task_type']
                        count = row['count']
                        percent = row['percentage']
                        f.write(f"{task}: {count:,} ({percent:.2f}%)\n")
                else:
                    f.write("无法显示所有任务分布，数据不可用\n")
                
                f.write("\n")
                
                # LLM任务类型分布
                f.write("LLM任务类型分布(仅LLM作业):\n")
                f.write("-"*50 + "\n")
                
                # 保持兼容性，使用原始键名task_distribution
                task_dist = analysis_results['task_distribution']
                for _, row in task_dist.iterrows():
                    task = row['task_type']
                    count = row['count']
                    percent = row['percentage']
                    f.write(f"{task}: {count:,} ({percent:.2f}%)\n")
                
                f.write("\n")
                
                # 资源使用情况
                f.write("资源使用情况:\n")
                f.write("-"*50 + "\n")
                
                # 统计LLM作业的资源使用
                llm_df = analysis_results['llm_jobs']
                
                # GPU使用
                try:
                    gpu_col = None
                    for col in self.resource_column_mappings['gpu']:
                        if col in llm_df.columns:
                            gpu_col = col
                            break
                    
                    if gpu_col:
                        avg_gpu = llm_df[gpu_col].mean()
                        max_gpu = llm_df[gpu_col].max()
                        f.write(f"平均GPU数量: {avg_gpu:.1f}\n")
                        f.write(f"最大GPU数量: {max_gpu}\n")
                except Exception as e:
                    logger.warning(f"统计GPU使用时出错: {str(e)}")
                    
                # 内存使用
                try:
                    mem_col = None
                    for col in self.resource_column_mappings['memory']:
                        if col in llm_df.columns:
                            mem_col = col
                            break
                    
                    if mem_col:
                        avg_mem = llm_df[mem_col].mean()
                        f.write(f"平均内存使用: {avg_mem:.1f}GB\n")
                except Exception as e:
                    logger.warning(f"统计内存使用时出错: {str(e)}")
                
                # CPU使用
                try:
                    cpu_col = None
                    for col in self.resource_column_mappings['cpu']:
                        if col in llm_df.columns:
                            cpu_col = col
                            break
                    
                    if cpu_col:
                        avg_cpu = llm_df[cpu_col].mean()
                        f.write(f"平均CPU核心数: {avg_cpu:.1f}\n")
                except Exception as e:
                    logger.warning(f"统计CPU使用时出错: {str(e)}")
                
                # 运行时间
                try:
                    runtime_col = None
                    for col in self.resource_column_mappings['runtime']:
                        if col in llm_df.columns:
                            runtime_col = col
                            break
                    
                    if runtime_col:
                        avg_runtime = llm_df[runtime_col].mean()
                        f.write(f"平均运行时间: {avg_runtime:.1f}小时\n")
                except Exception as e:
                    logger.warning(f"统计运行时间时出错: {str(e)}")
                    
                f.write("\n")
                
                # 处理时间
                hours = int(analysis_results['processing_time'] // 3600)
                minutes = int((analysis_results['processing_time'] % 3600) // 60)
                seconds = int(analysis_results['processing_time'] % 60)
                
                f.write(f"分析处理时间: {hours}小时 {minutes}分钟 {seconds}秒\n")
            
            logger.info(f"分析报告已保存到: {report_path}")
            
        except Exception as e:
            logger.error(f"生成分析报告时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())


def main():
    """主函数：提供命令行接口"""
    import argparse
    parser = argparse.ArgumentParser(description='作业数据分析工具')
    parser.add_argument('-i', '--input', required=True, help='输入CSV文件路径')
    parser.add_argument('-o', '--output', default='results', help='输出目录')
    parser.add_argument('-w', '--workers', type=int, default=51, help='工作进程数')
    parser.add_argument('-g', '--gpu', action='store_true', help='启用GPU加速')
    parser.add_argument('--no-gpu', action='store_true', help='禁用GPU加速')
    parser.add_argument('-c', '--chunk-size', type=int, default=10000, help='数据块大小')
    parser.add_argument('--sample', action='store_true', help='是否仅进行抽样测试')
    parser.add_argument('--sample-size', type=int, default=10000, help='抽样大小')
    
    args = parser.parse_args()
    
    # 创建分析器实例
    use_gpu = args.gpu and not args.no_gpu
    analyzer = JobAnalyzer(max_workers=args.workers, use_gpu=use_gpu)
    
    if args.sample:
        # 进行抽样测试
        analyzer.sample_test_analysis(
            filepath=args.input,
            sample_size=args.sample_size
        )
    else:
        # 分析完整数据集
        analyzer.analyze_job_dataset(
            filepath=args.input,
            output_dir=args.output,
            n_workers=args.workers,
            batch_size=args.chunk_size
        )

if __name__ == "__main__":
    main()