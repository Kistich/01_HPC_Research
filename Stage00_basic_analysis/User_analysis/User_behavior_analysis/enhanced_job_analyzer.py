#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import defaultdict
import re
import gc
import logging
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Generator, Set, Optional
import multiprocessing
from functools import partial
import warnings
warnings.filterwarnings('ignore')

class EnhancedJobAnalyzer:
    def __init__(self, output_dir="./output"):
        # 初始化日志
        self.setup_logging()
        
        # 设置输出目录
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置分布式作业识别参数
        self.dist_job_patterns = [
            # DeepSpeed参数
            r'deepspeed\s+--num_nodes\b',
            r'--hostfile\b',
            # SLURM多节点参数
            r'--nodes=[2-9][0-9]*\b',
            r'--ntasks-per-node\b',
            # MPI参数
            r'mpirun\b',
            r'mpiexec\b',
            r'machinefile\b',
            # 其他分布式框架
            r'horovod\b',
            r'torch\.distributed\b',
            r'dist_training\b'
        ]
        
        # 作业类型定义 (从analyze_job_types.py复制)
        self.job_types = {
            'DL_Training': {
                'patterns': [
                    'pytorch', 'tensorflow', 'caffe', 'mindspore', 'paddlepaddle', 'mxnet', 'keras',
                    'cnn', 'resnet', 'vgg', 'yolo', 'densenet', 'efficientnet', 'mobilenet',
                    'detection', 'segmentation', 'classification', 'recognition', 'tracking',
                    'train', 'finetune', 'pretrain', 'backbone', 'epoch', 'batch', 'optimizer',
                    'coco', 'imagenet', 'voc', 'cifar', 'mnist'
                ],
                'resource_patterns': {
                    'gpu_num': {'min': 1, 'typical': 4},
                    'gpu_mem': {'min': 10000, 'typical': 24000},
                    'num_processors': {'min': 4, 'typical': 16},
                    'max_mem': {'min': 32000},
                    'duration': {'min': 1800, 'max': 86400},
                    'gpu_types': ['NVIDIAA100', 'NVIDIAA800', 'NVIDIAA40']
                },
                'queue_patterns': ['gpu', 'dgx', 'ml', 'ai', 'vision'],
                'status_patterns': {
                    'exit_status': [0],
                    'jstatus': [64]
                }
            },
            'LLM_Task': {
                'patterns': [
                    'llm', 'gpt', 'bert', 'transformer', 't5', 'palm', 'llama', 'bloom',
                    'roberta', 'bart', 'opt', 'chatgpt', 'claude', 'alpaca', 'vicuna',
                    'attention', 'decoder', 'encoder', 'multihead', 'embedding', 'tokenizer',
                    'language', 'token', 'text', 'prompt', 'completion', 'generation',
                    'translation', 'summarization', 'qa', 'dialogue', 'chat',
                    'pretrain', 'finetune', 'lora', 'peft', 'quantization', 'int8', 'int4',
                    'pile', 'c4', 'bookcorpus', 'wikipedia'
                ],
                'resource_patterns': {
                    'gpu_num': {'min': 2, 'typical': 8},
                    'gpu_mem': {'min': 40000, 'typical': 80000},
                    'num_processors': {'min': 8, 'typical': 32},
                    'max_mem': {'min': 100000},
                    'duration': {'min': 3600, 'max': 604800},
                    'gpu_types': ['NVIDIAA800-SXM4-80GB', 'NVIDIAA100-80GB']
                },
                'queue_patterns': ['gpu', 'dgx', 'ml', 'llm', 'nlp'],
                'status_patterns': {
                    'exit_status': [0],
                    'jstatus': [64]
                }
            },
            # 可以添加其他作业类型...
        }
        
        # 初始化分布式作业统计
        self.dist_job_stats = {
            'total_jobs': 0,
            'distributed_jobs': 0,
            'nodes_distribution': defaultdict(int),
            'tasks_distribution': defaultdict(int)
        }
        
        # 保存合并映射关系
        self.job_mapping = {}
        
        # 优化的数据类型
        self.dtypes = {
            'job_id': str,
            'job_name': str,
            'command': str,
            'application': str,
            'gpu_num': 'Int64',
            'gpu_mem': 'Int64',
            'num_processors': 'Int64',
            'max_mem': 'Int64',
            'queue': str,
            'start_time': 'Int64',
            'end_time': 'Int64',
            'jstatus': 'Int64',
            'exit_status': 'Int64',
            'num_exec_hosts': 'Int64',
            'exec_hosts': str,
            'gpu_types': str,
            'cluster_name': str,
            'cluster_gen': 'Int64',
            'submit_time': 'Int64',
            'from_host': str
        }
    
    def setup_logging(self):
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("EnhancedJobAnalyzer")
        # 添加文件处理器
        fh = logging.FileHandler('job_analysis.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
    
    def read_csv_in_chunks(self, file_path: str, chunk_size: int = 1000000, max_rows: int = None) -> Generator:
        """最可靠的CSV文件读取方法"""
        self.logger.info(f"开始读取CSV文件: {file_path}")
        
        try:
            # 首先读取标题行获取列名
            header = pd.read_csv(file_path, nrows=0)
            columns = header.columns.tolist()
            
            # 构建将所有列都视为字符串的字典
            string_dtypes = {col: str for col in columns}
            
            # 对于确定是整数的列使用Int64
            integer_columns = ['gpu_num', 'num_processors', 'jstatus', 'exit_status', 'num_exec_hosts', 'cluster_gen']
            for col in integer_columns:
                if col in string_dtypes:
                    string_dtypes[col] = 'Int64'
            
            # 使用无类型推断的读取方式
            total_read = 0
            with tqdm(total=max_rows if max_rows else 9000000, desc="读取数据") as pbar:
                for chunk in pd.read_csv(file_path,
                                        chunksize=chunk_size,
                                        dtype=string_dtypes,  # 全部作为字符串读取
                                        engine='c',
                                        na_values=['', 'NA', 'null', 'NULL', 'none', 'None'],
                                        keep_default_na=True,
                                        low_memory=False,
                                        nrows=max_rows):
                    
                    # 在读取后手动转换整数列
                    for int_col in integer_columns:
                        if int_col in chunk.columns:
                            chunk[int_col] = pd.to_numeric(chunk[int_col], errors='coerce')
                    
                    # 更新进度
                    total_read += len(chunk)
                    pbar.update(len(chunk))
                    pbar.set_description(f"读取数据 ({total_read:,}行)")
                    
                    yield chunk
                    
                    # 如果达到最大行数，停止
                    if max_rows is not None and total_read >= max_rows:
                        break
                        
        except Exception as e:
            self.logger.error(f"读取CSV文件时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def is_distributed_job(self, row) -> bool:
        """判断作业是否为分布式作业"""
        # 1. 明确标记为多节点的作业
        if pd.notna(row['num_exec_hosts']) and row['num_exec_hosts'] > 1:
            return True
        
        # 2. exec_hosts字段包含多个节点
        if pd.notna(row['exec_hosts']) and ',' in str(row['exec_hosts']):
            return True
        
        # 3. 命令行包含分布式作业特征
        if pd.notna(row['command']):
            command = str(row['command']).lower()
            for pattern in self.dist_job_patterns:
                if re.search(pattern, command):
                    return True
        
        # 4. 大规模GPU请求（通常表示分布式）
        if pd.notna(row['gpu_num']) and row['gpu_num'] > 4:
            return True
            
        return False
    
    def identify_distributed_jobs(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """识别分布式作业并创建任务组"""
        # 添加是否为分布式作业的标记
        chunk['is_distributed'] = chunk.apply(self.is_distributed_job, axis=1)
        
        # 统计分布式作业数量
        total_jobs = len(chunk)
        distributed_jobs = chunk['is_distributed'].sum()
        
        # 更新分布式作业统计
        self.dist_job_stats['total_jobs'] += total_jobs
        self.dist_job_stats['distributed_jobs'] += distributed_jobs
        
        # 创建作业组
        job_groups = defaultdict(list)
        
        # 根据不同规则对分布式作业进行分组
        for _, job in chunk[chunk['is_distributed'] == True].iterrows():
            group_key = None
            
            # 1. 使用作业名称相似性进行分组
            if pd.notna(job['job_name']):
                # 移除作业名称中可能的数字后缀
                base_name = re.sub(r'_?\d+$', '', job['job_name'])
                if base_name:
                    # 查找接近的提交时间
                    submit_time = job['submit_time'] if pd.notna(job['submit_time']) else 0
                    time_window = 300  # 5分钟
                    
                    # 尝试寻找同一用户近期提交的相似作业名
                    for existing_key in list(job_groups.keys()):
                        if base_name in existing_key:
                            # 检查提交时间是否接近
                            existing_jobs = job_groups[existing_key]
                            if existing_jobs:
                                existing_time = existing_jobs[0].get('submit_time', 0)
                                if abs(existing_time - submit_time) <= time_window:
                                    group_key = existing_key
                                    break
            
            # 2. 如果没找到组，创建新组
            if group_key is None:
                # 使用作业ID作为新组的标识
                group_key = f"group_{job['job_id']}"
            
            # 添加到组
            job_dict = job.to_dict()
            # 标记源作业ID
            job_dict['source_job_id'] = job['job_id']
            job_groups[group_key].append(job_dict)
            
            # 更新节点数分布
            num_nodes = job['num_exec_hosts'] if pd.notna(job['num_exec_hosts']) else 1
            self.dist_job_stats['nodes_distribution'][int(num_nodes)] += 1
            
            # 更新作业映射
            self.job_mapping[job['job_id']] = group_key
        
        return chunk, job_groups
    
    def merge_distributed_jobs(self, job_groups: Dict) -> pd.DataFrame:
        """合并分布式作业组为单一记录"""
        self.logger.info(f"合并 {len(job_groups)} 个分布式作业组...")
        
        merged_jobs = []
        for group_id, jobs in job_groups.items():
            # 使用首个作业作为基础
            base_job = jobs[0]
            
            if len(jobs) == 1:
                # 单作业组，直接添加
                merged_jobs.append(base_job)
                continue
            
            # 合并相关属性
            unique_hosts = set()
            all_job_ids = []
            
            # 初始化聚合值
            total_mem = 0
            max_gpu_num = 0
            max_gpu_mem = 0
            max_num_processors = 0
            
            for job in jobs:
                # 记录作业ID
                all_job_ids.append(job['job_id'])
                
                # 合并执行主机
                if 'exec_hosts' in job and pd.notna(job['exec_hosts']):
                    hosts = str(job['exec_hosts']).split(',')
                    for host in hosts:
                        if host.strip():
                            unique_hosts.add(host.strip())
                
                # 合并资源使用 - 安全地转换为数值
                try:
                    # 安全地将内存字段转换为浮点数
                    if 'max_mem' in job and pd.notna(job['max_mem']):
                        job_mem = float(job['max_mem']) if isinstance(job['max_mem'], str) else job['max_mem']
                        total_mem = max(total_mem, job_mem)
                except (ValueError, TypeError):
                    # 无法转换时记录日志并跳过该值
                    self.logger.warning(f"无法转换max_mem值: {job.get('max_mem', 'None')}")
                
                try:
                    if 'gpu_num' in job and pd.notna(job['gpu_num']):
                        job_gpu_num = int(job['gpu_num']) if isinstance(job['gpu_num'], str) else job['gpu_num']
                        max_gpu_num = max(max_gpu_num, job_gpu_num)
                except (ValueError, TypeError):
                    self.logger.warning(f"无法转换gpu_num值: {job.get('gpu_num', 'None')}")
                
                try:
                    if 'gpu_mem' in job and pd.notna(job['gpu_mem']):
                        job_gpu_mem = float(job['gpu_mem']) if isinstance(job['gpu_mem'], str) else job['gpu_mem']
                        max_gpu_mem = max(max_gpu_mem, job_gpu_mem)
                except (ValueError, TypeError):
                    self.logger.warning(f"无法转换gpu_mem值: {job.get('gpu_mem', 'None')}")
                
                try:
                    if 'num_processors' in job and pd.notna(job['num_processors']):
                        job_procs = int(job['num_processors']) if isinstance(job['num_processors'], str) else job['num_processors']
                        max_num_processors = max(max_num_processors, job_procs)
                except (ValueError, TypeError):
                    self.logger.warning(f"无法转换num_processors值: {job.get('num_processors', 'None')}")
            
            # 创建合并后的作业记录
            merged_job = base_job.copy()
            
            # 更新分布式信息
            merged_job['is_distributed'] = True
            merged_job['distributed_group_id'] = group_id
            merged_job['child_job_ids'] = ','.join(all_job_ids)
            merged_job['job_count'] = len(jobs)
            
            # 更新资源信息
            merged_job['max_mem'] = total_mem
            merged_job['gpu_num'] = max_gpu_num
            merged_job['gpu_mem'] = max_gpu_mem
            merged_job['num_processors'] = max_num_processors
            
            # 更新执行主机信息
            merged_job['num_exec_hosts'] = len(unique_hosts)
            merged_job['exec_hosts'] = ','.join(sorted(unique_hosts))
            
            # 添加到结果
            merged_jobs.append(merged_job)
        
        # 转换为DataFrame
        return pd.DataFrame(merged_jobs)
    
    def analyze_jobs(self, input_file: str, max_rows: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """分析作业集群数据，识别和合并分布式作业"""
        self.logger.info(f"开始分析作业: {input_file}")
        
        # 初始化结果
        all_jobs = []
        all_job_groups = {}
        
        # 分块读取CSV文件
        for chunk in self.read_csv_in_chunks(input_file, chunk_size=100000, max_rows=max_rows):
            # 识别分布式作业
            processed_chunk, job_groups = self.identify_distributed_jobs(chunk)
            
            # 保存处理后的块
            all_jobs.append(processed_chunk)
            
            # 合并作业组
            all_job_groups.update(job_groups)
            
            # 输出进度
            self.logger.info(f"已处理 {len(processed_chunk):,} 行，发现 {len(job_groups):,} 个分布式作业组")
        
        # 合并所有处理后的块
        if all_jobs:
            all_jobs_df = pd.concat(all_jobs, ignore_index=True)
            self.logger.info(f"总计处理 {len(all_jobs_df):,} 行，发现 {len(all_job_groups):,} 个分布式作业组")
        else:
            all_jobs_df = pd.DataFrame()
            self.logger.warning("没有找到任何作业数据")
            return all_jobs_df, None
        
        # 合并分布式作业
        merged_jobs_df = self.merge_distributed_jobs(all_job_groups)
        
        # 创建分布式视图
        non_dist_jobs = all_jobs_df[all_jobs_df['is_distributed'] == False].copy()
        dist_view = pd.concat([non_dist_jobs, merged_jobs_df], ignore_index=True)
        
        # 保存分布式作业统计
        self.save_distributed_stats()
        
        # 保存作业映射
        self.save_job_mapping()
        
        return all_jobs_df, dist_view
    
    def save_distributed_stats(self):
        """保存分布式作业统计信息"""
        stats_file = os.path.join(self.output_dir, "distributed_jobs_stats.json")
        
        # 使用辅助函数转换所有NumPy类型
        stats = self.convert_to_python_type(self.dist_job_stats)
        
        # 保存为JSON
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"分布式作业统计信息已保存至 {stats_file}")
    
    def save_job_mapping(self):
        """保存作业映射关系"""
        mapping_file = os.path.join(self.output_dir, 'job_mapping.csv')
        
        mapping_data = []
        for job_id, group_id in self.job_mapping.items():
            mapping_data.append({
                'job_id': job_id,
                'group_id': group_id
            })
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            mapping_df.to_csv(mapping_file, index=False)
            self.logger.info(f"作业映射关系已保存至 {mapping_file}")
    
    def save_analyzed_jobs(self, all_jobs_df, dist_view):
        """保存分析后的作业数据"""
        # 保存原始作业（添加了分布式标记）
        original_file = os.path.join(self.output_dir, 'original_jobs.csv')
        all_jobs_df.to_csv(original_file, index=False)
        self.logger.info(f"原始作业数据已保存至 {original_file}")
        
        # 保存分布式视图
        if dist_view is not None:
            dist_file = os.path.join(self.output_dir, 'distributed_view.csv')
            dist_view.to_csv(dist_file, index=False)
            self.logger.info(f"分布式视图已保存至 {dist_file}")
    
    def analyze_job_types(self, jobs_df: pd.DataFrame, use_dist_view: bool = True):
        """分析作业类型"""
        self.logger.info(f"开始分析作业类型 ({'分布式视图' if use_dist_view else '原始视图'})")
        
        # 提取特征
        jobs_with_features = []
        
        for _, job in tqdm(jobs_df.iterrows(), total=len(jobs_df), desc="提取特征"):
            features = self._extract_job_features(job)
            jobs_with_features.append((job['job_id'], features))
        
        # 使用多进程计算作业类型
        with multiprocessing.Pool() as pool:
            results = list(tqdm(
                pool.imap(partial(self._compute_job_type_parallel, job_types=self.job_types), 
                         jobs_with_features),
                total=len(jobs_with_features),
                desc="计算作业类型"
            ))
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 保存结果
        output_file = os.path.join(
            self.output_dir, 
            f"job_types_{'dist' if use_dist_view else 'original'}.csv"
        )
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"作业类型分析结果已保存至 {output_file}")
        
        # 分析类型分布
        self._analyze_type_distribution(results_df, use_dist_view)
        
        return results_df
    
    def _extract_job_features(self, job) -> Dict:
        """从作业记录中提取特征"""
        features = {}
        
        # 基本信息
        features['job_id'] = job['job_id'] if pd.notna(job['job_id']) else None
        features['job_name'] = job['job_name'].lower() if pd.notna(job['job_name']) else None
        features['command'] = job['command'].lower() if pd.notna(job['command']) else None
        features['application'] = job['application'].lower() if pd.notna(job['application']) else None
        features['queue'] = job['queue'].lower() if pd.notna(job['queue']) else None
        
        # 资源信息
        features['gpu_num'] = int(job['gpu_num']) if pd.notna(job['gpu_num']) else None
        features['gpu_mem'] = int(job['gpu_mem']) if pd.notna(job['gpu_mem']) else None
        features['num_processors'] = int(job['num_processors']) if pd.notna(job['num_processors']) else None
        features['max_mem'] = int(job['max_mem']) if pd.notna(job['max_mem']) else None
        
        # 状态信息
        features['jstatus'] = int(job['jstatus']) if pd.notna(job['jstatus']) else None
        features['exit_status'] = int(job['exit_status']) if pd.notna(job['exit_status']) else None
        
        # 时间信息
        if pd.notna(job['start_time']) and pd.notna(job['end_time']):
            features['duration'] = int(job['end_time']) - int(job['start_time'])
        else:
            features['duration'] = None
        
        # GPU类型
        features['gpu_types'] = job['gpu_types'] if pd.notna(job['gpu_types']) else None
        
        # 分布式信息
        features['num_exec_hosts'] = int(job['num_exec_hosts']) if pd.notna(job['num_exec_hosts']) else None
        features['exec_hosts'] = job['exec_hosts'] if pd.notna(job['exec_hosts']) else None
        
        return features
    
    def _compute_job_type_parallel(self, job_data: Tuple[str, Dict], job_types: Dict) -> Dict:
        """并行计算作业类型（为多进程设计）"""
        job_id, features = job_data
        
        try:
            # 初始化
            max_prob = 0.0
            best_type = "Unknown"
            best_evidence = {}
            
            # 对每种作业类型计算匹配概率
            for job_type, type_info in job_types.items():
                # 文本匹配分数 (权重40%)
                text_score = 0
                if 'command' in features and features['command']:
                    command = features['command'].lower()
                    matches = []
                    for pattern in type_info['patterns']:
                        if pattern in command:
                            matches.append(pattern)
                    text_score = len(matches) / max(1, len(type_info['patterns']) * 0.1)
                    text_score = min(1.0, text_score)
                
                # 资源匹配分数 (35%)
                resource_score = 0
                resource_matches = []
                resource_patterns = type_info['resource_patterns']
                resource_fields = 0
                
                for field, criteria in resource_patterns.items():
                    if field in features and features[field] is not None:
                        resource_fields += 1
                        field_match = False
                        
                        if field == 'gpu_types' and isinstance(criteria, list):
                            field_match = any(gpu_type.lower() in features[field].lower() for gpu_type in criteria)
                        elif isinstance(criteria, dict):
                            value = features[field]
                            if 'min' in criteria and value < criteria['min']:
                                field_match = False
                            elif 'max' in criteria and value > criteria['max']:
                                field_match = False
                            else:
                                field_match = True
                        
                        if field_match:
                            resource_matches.append(field)
                
                if resource_fields > 0:
                    resource_score = len(resource_matches) / resource_fields
                
                # 队列匹配分数 (15%)
                queue_score = 0
                if 'queue' in features and features['queue']:
                    queue_matches = []
                    for pattern in type_info['queue_patterns']:
                        if pattern in features['queue']:
                            queue_matches.append(pattern)
                    queue_score = len(queue_matches) / max(1, len(type_info['queue_patterns']))
                    queue_score = min(1.0, queue_score)
                
                # 状态匹配分数 (10%)
                status_score = 0
                status_fields = 0
                status_matches = []
                
                if 'exit_status' in features and features['exit_status'] is not None:
                    status_fields += 1
                    if features['exit_status'] in type_info['status_patterns']['exit_status']:
                        status_matches.append('exit_status')
                
                if 'jstatus' in features and features['jstatus'] is not None:
                    status_fields += 1
                    if features['jstatus'] in type_info['status_patterns']['jstatus']:
                        status_matches.append('jstatus')
                
                if status_fields > 0:
                    status_score = len(status_matches) / status_fields
                
                # 计算总概率
                weights = [0.40, 0.35, 0.15, 0.10]
                scores = [text_score, resource_score, queue_score, status_score]
                
                weighted_sum = sum(w * s for w, s in zip(weights, scores))
                total_weight = sum(weights)
                
                probability = weighted_sum / total_weight
                
                # 记录证据
                evidence = {
                    'text_match': {
                        'score': text_score,
                        'matched': [p for p in type_info['patterns'] if 'command' in features and features['command'] and p in features['command']]
                    },
                    'resource_match': {
                        'score': resource_score,
                        'matched': resource_matches
                    },
                    'queue_match': {
                        'score': queue_score,
                        'matched': [p for p in type_info['queue_patterns'] if 'queue' in features and features['queue'] and p in features['queue']]
                    },
                    'status_match': {
                        'score': status_score,
                        'matched': status_matches
                    }
                }
                
                if probability > max_prob:
                    max_prob = probability
                    best_type = job_type
                    best_evidence = evidence
            
            # 返回结果字典
            return {
                'job_id': job_id,
                'job_type': best_type,
                'confidence': max_prob,
                'evidence': json.dumps(best_evidence)
            }
            
        except Exception as e:
            self.logger.error(f"计算作业类型时出错: {e}")
            return {
                'job_id': job_id if 'job_id' in locals() else 'unknown',
                'job_type': '错误',
                'confidence': 0,
                'evidence': json.dumps({"error": str(e)})
            }
    
    def _analyze_type_distribution(self, results_df: pd.DataFrame, use_dist_view: bool):
        """分析作业类型分布"""
        # 计算各类型作业数量
        type_counts = results_df['job_type'].value_counts()
        total_jobs = len(results_df)
        
        # 保存类型分布
        dist_file = os.path.join(self.output_dir, f"job_type_distribution_{'dist' if use_dist_view else 'original'}.csv")
        type_df = pd.DataFrame({
            'job_type': type_counts.index,
            'count': type_counts.values,
            'percentage': type_counts.values / total_jobs * 100
        })
        type_df.to_csv(dist_file, index=False)
        
        # 创建类型分布报告
        report_file = os.path.join(self.output_dir, f"job_type_report_{'dist' if use_dist_view else 'original'}.md")
        
        with open(report_file, 'w') as f:
            f.write(f"# 作业类型分析报告 ({'分布式视图' if use_dist_view else '原始视图'})\n\n")
            
            f.write("## 类型分布\n\n")
            f.write("| 作业类型 | 数量 | 百分比 | 平均置信度 |\n")
            f.write("|----------|------|--------|------------|\n")
            
            # 计算每种类型的平均置信度
            for job_type in type_df['job_type']:
                count = type_df[type_df['job_type'] == job_type]['count'].values[0]
                percentage = type_df[type_df['job_type'] == job_type]['percentage'].values[0]
                avg_confidence = results_df[results_df['job_type'] == job_type]['confidence'].mean()
                
                f.write(f"| {job_type} | {count:,} | {percentage:.2f}% | {avg_confidence:.2f} |\n")
            
            # 添加高置信度分布
            f.write("\n## 高置信度分布 (置信度 >= 0.7)\n\n")
            high_conf = results_df[results_df['confidence'] >= 0.7]
            high_type_counts = high_conf['job_type'].value_counts()
            
            f.write("| 作业类型 | 高置信度数量 | 高置信度百分比 | 类别覆盖率 |\n")
            f.write("|----------|--------------|----------------|------------|\n")
            
            for job_type in high_type_counts.index:
                high_count = high_type_counts[job_type]
                high_percentage = high_count / len(high_conf) * 100
                type_coverage = high_count / type_counts[job_type] * 100
                
                f.write(f"| {job_type} | {high_count:,} | {high_percentage:.2f}% | {type_coverage:.2f}% |\n")

    def convert_to_python_type(self, obj):
        """递归转换NumPy类型为Python原生类型"""
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return [self.convert_to_python_type(x) for x in obj]
        elif isinstance(obj, (dict, defaultdict)):
            return {self.convert_to_python_type(k): self.convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_python_type(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_to_python_type(x) for x in obj)
        else:
            return obj

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="增强版作业分析工具")
    parser.add_argument("--input", "-i", required=True, 
                       help="输入的作业CSV文件路径")
    parser.add_argument("--output-dir", "-o", default="./job_analysis_output", 
                       help="输出目录")
    parser.add_argument("--max-rows", "-m", type=int, default=None,
                       help="处理的最大行数，用于测试（默认处理全部）")
    parser.add_argument("--analyze-types", "-t", action="store_true",
                       help="分析作业类型")
    
    args = parser.parse_args()
    
    # 初始化分析器
    analyzer = EnhancedJobAnalyzer(output_dir=args.output_dir)
    
    try:
        # 分析作业
        all_jobs, dist_view = analyzer.analyze_jobs(args.input, max_rows=args.max_rows)
        
        # 保存分析结果
        analyzer.save_analyzed_jobs(all_jobs, dist_view)
        
        # 可选：分析作业类型
        if args.analyze_types and dist_view is not None:
            # 使用分布式视图分析类型
            analyzer.analyze_job_types(dist_view, use_dist_view=True)
            # 也可以使用原始视图分析类型
            sample_size = min(len(all_jobs), 10000)  # 限制样本大小以加快处理
            analyzer.analyze_job_types(all_jobs.sample(sample_size) if len(all_jobs) > sample_size else all_jobs, 
                                     use_dist_view=False)
        
        print(f"分析完成，结果保存在 {args.output_dir}")
        
    except Exception as e:
        print(f"分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 