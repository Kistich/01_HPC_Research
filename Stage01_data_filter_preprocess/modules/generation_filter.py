#!/usr/bin/env python3
"""
一期二期数据过滤器
基于脚本特征和节点信息进行精确分类
"""

import pandas as pd
import numpy as np
import re
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from pathlib import Path

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from parallel_processor import ParallelProcessor
from progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class GenerationFilter:
    """一期二期数据过滤器"""
    
    def __init__(self, config_path: str):
        """
        初始化过滤器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.processor = ParallelProcessor(
            max_cores=self.config['parallel_processing']['max_cores'],
            memory_limit_gb=self.config['parallel_processing']['memory_limit_gb']
        )
        
        # 编译正则表达式模式
        self._compile_patterns()
        
        # 统计信息
        self.stats = {
            'total_jobs': 0,
            'second_generation_high': 0,
            'second_generation_medium': 0,
            'second_generation_low': 0,
            'first_generation': 0,
            'mixed_features': 0,
            'management_nodes': 0,
            'unknown_category': 0
        }
        
        logger.info("一期二期数据过滤器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {config_path}")
            return config
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            raise
    
    def _compile_patterns(self):
        """编译正则表达式模式"""
        self.first_gen_patterns = []
        self.second_gen_patterns = []
        self.mgmt_node_patterns = []
        self.first_gen_node_patterns = []
        self.second_gen_node_patterns = []
        
        # 编译脚本特征模式
        for pattern in self.config['generation_filter']['script_features']['first_generation_patterns']:
            self.first_gen_patterns.append(re.compile(pattern, re.IGNORECASE))
        
        for pattern in self.config['generation_filter']['script_features']['second_generation_patterns']:
            self.second_gen_patterns.append(re.compile(pattern, re.IGNORECASE))
        
        # 编译节点模式
        for pattern in self.config['generation_filter']['node_classification']['management_nodes']:
            self.mgmt_node_patterns.append(re.compile(pattern, re.IGNORECASE))

        # 明确的二期节点模式 - 最高优先级
        self.definitive_second_gen_patterns = []
        for pattern in self.config['generation_filter']['node_classification']['definitive_second_generation_nodes']:
            self.definitive_second_gen_patterns.append(re.compile(pattern, re.IGNORECASE))

        # 可能的一期节点模式
        self.possible_first_gen_patterns = []
        for pattern in self.config['generation_filter']['node_classification']['possible_first_generation_nodes']:
            self.possible_first_gen_patterns.append(re.compile(pattern, re.IGNORECASE))

        # 可能的二期节点模式
        self.possible_second_gen_patterns = []
        for pattern in self.config['generation_filter']['node_classification']['possible_second_generation_nodes']:
            self.possible_second_gen_patterns.append(re.compile(pattern, re.IGNORECASE))
        
        logger.info("正则表达式模式编译完成")
    
    def filter_data(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        执行数据过滤
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径字典
        """
        logger.info(f"开始一期二期数据过滤: {input_file}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义中间文件路径
        analyzed_file = os.path.join(output_dir, "analyzed_features.csv")

        # 检查是否存在分析结果文件
        if os.path.exists(analyzed_file):
            logger.info(f"✅ 发现中间文件: {analyzed_file}")
            file_size_mb = os.path.getsize(analyzed_file) / (1024 * 1024)
            logger.info(f"文件大小: {file_size_mb:.1f} MB")

            try:
                # 加载已分析的数据
                logger.info("加载已分析的特征数据...")
                analyzed_df = pd.read_csv(analyzed_file)
                # 设置总作业数统计
                self.stats['total_jobs'] = len(analyzed_df)
                logger.info(f"✅ 成功加载分析结果: {len(analyzed_df):,} 条记录")
                logger.info("⏭️  跳过脚本特征分析，直接进行分类决策...")

            except Exception as e:
                logger.warning(f"⚠️  加载中间文件失败: {e}")
                logger.info("🔄 重新进行完整分析...")
                analyzed_df = self._perform_full_analysis(input_file, analyzed_file)
        else:
            logger.info("❌ 未发现中间文件，进行完整分析...")
            analyzed_df = self._perform_full_analysis(input_file, analyzed_file)
        
        # 执行分类决策
        logger.info("执行分类决策...")
        classified_df = self._make_classification_decisions(analyzed_df)
        
        # 分离不同类别的数据
        output_files = self._separate_categories(classified_df, output_dir)
        
        # 生成统计报告
        self._generate_classification_report(output_dir)
        
        logger.info("一期二期数据过滤完成")
        return output_files

    def _perform_full_analysis(self, input_file: str, analyzed_file: str) -> pd.DataFrame:
        """执行完整的脚本特征分析并保存中间结果"""
        # 加载数据
        logger.info("加载数据...")
        df = pd.read_csv(input_file)
        self.stats['total_jobs'] = len(df)

        logger.info(f"数据加载完成: {len(df):,} 条记录")

        # 分块处理
        chunks = self.processor.split_dataframe(df)

        # 并行分析脚本特征和节点信息
        with ProgressTracker(len(chunks), "脚本特征分析", "块") as pbar:
            def progress_callback(completed, total):
                pbar.update(1)

            results = []
            for i, chunk in enumerate(chunks):
                result = self._analyze_chunk(chunk)
                results.append(result)
                progress_callback(i + 1, len(chunks))

        # 合并结果
        logger.info("合并分析结果...")
        analyzed_df = self.processor.merge_results(results)

        # 保存中间结果
        logger.info(f"💾 保存分析结果到: {analyzed_file}")
        analyzed_df.to_csv(analyzed_file, index=False)
        file_size_mb = os.path.getsize(analyzed_file) / (1024 * 1024)
        logger.info(f"✅ 中间文件保存成功: {file_size_mb:.1f} MB")

        return analyzed_df

    def _analyze_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        分析数据块的脚本特征和节点信息

        Args:
            chunk: 数据块

        Returns:
            分析后的数据块
        """
        chunk = chunk.copy()
        
        # 初始化分析结果列
        chunk['first_gen_script_score'] = 0.0
        chunk['second_gen_script_score'] = 0.0
        chunk['definitive_second_gen_score'] = 0.0  # 明确二期节点分数
        chunk['possible_first_gen_score'] = 0.0     # 可能一期节点分数
        chunk['possible_second_gen_score'] = 0.0    # 可能二期节点分数
        chunk['is_management_node'] = False
        chunk['node_classification_decisive'] = False  # 节点分类是否具有决定性
        
        # 分析脚本特征 - 向量化处理
        # 构建脚本内容 - 安全的字符串拼接
        command_series = chunk.get('command', pd.Series([''] * len(chunk))).fillna('').astype(str)
        job_name_series = chunk.get('job_name', pd.Series([''] * len(chunk))).fillna('').astype(str)
        script_contents = command_series + ' ' + job_name_series

        # 向量化计算脚本分数
        chunk['first_gen_script_score'] = script_contents.apply(
            lambda x: self._calculate_script_score(x, self.first_gen_patterns)
        )
        chunk['second_gen_script_score'] = script_contents.apply(
            lambda x: self._calculate_script_score(x, self.second_gen_patterns)
        )

        # 分析节点信息 - 向量化处理
        exec_hosts_series = chunk.get('exec_hosts', '').fillna('').astype(str)
        first_exec_host_series = chunk.get('first_exec_host', '').fillna('').astype(str)
        from_host_series = chunk.get('from_host', '').fillna('').astype(str)

        # 向量化检查管理节点 - 安全的字符串转换
        def safe_str(value):
            """安全的字符串转换，处理NaN和float"""
            if pd.isna(value):
                return ''
            return str(value)

        chunk['is_management_node'] = chunk.apply(
            lambda row: self._check_management_node(
                safe_str(row.get('exec_hosts', '')),
                safe_str(row.get('first_exec_host', '')),
                safe_str(row.get('from_host', ''))
            ), axis=1
        )

        # 对非管理节点进行进一步分析
        non_mgmt_mask = ~chunk['is_management_node']
        non_mgmt_chunk = chunk[non_mgmt_mask].copy()

        if len(non_mgmt_chunk) > 0:
            # 向量化计算明确的二期节点分数
            chunk.loc[non_mgmt_mask, 'definitive_second_gen_score'] = non_mgmt_chunk.apply(
                lambda row: self._calculate_node_score(
                    safe_str(row.get('exec_hosts', '')),
                    safe_str(row.get('first_exec_host', '')),
                    safe_str(row.get('from_host', '')),
                    self.definitive_second_gen_patterns
                ), axis=1
            )

            # 标记决定性分类
            decisive_mask = chunk['definitive_second_gen_score'] > 0
            chunk.loc[decisive_mask, 'node_classification_decisive'] = True

            # 对非决定性的记录计算可能的节点分数
            non_decisive_mask = non_mgmt_mask & ~decisive_mask
            non_decisive_chunk = chunk[non_decisive_mask].copy()

            if len(non_decisive_chunk) > 0:
                # 计算可能的一期节点分数
                chunk.loc[non_decisive_mask, 'possible_first_gen_score'] = non_decisive_chunk.apply(
                    lambda row: self._calculate_node_score(
                        safe_str(row.get('exec_hosts', '')),
                        safe_str(row.get('first_exec_host', '')),
                        safe_str(row.get('from_host', '')),
                        self.possible_first_gen_patterns
                    ), axis=1
                )

                # 计算可能的二期节点分数
                chunk.loc[non_decisive_mask, 'possible_second_gen_score'] = non_decisive_chunk.apply(
                    lambda row: self._calculate_node_score(
                        safe_str(row.get('exec_hosts', '')),
                        safe_str(row.get('first_exec_host', '')),
                        safe_str(row.get('from_host', '')),
                        self.possible_second_gen_patterns
                    ), axis=1
                )
        
        return chunk
    
    def _calculate_script_score(self, script_content: str, patterns: List[re.Pattern]) -> float:
        """计算脚本特征分数"""
        if not script_content or script_content == 'nan':
            return 0.0
        
        score = 0.0
        weights = self.config['generation_filter']['script_features']['feature_weights']
        
        for pattern in patterns:
            matches = pattern.findall(script_content)
            if matches:
                # 根据模式类型分配权重
                if pattern.pattern.startswith('#'):
                    score += weights['script_header']
                elif any(cmd in pattern.pattern for cmd in ['jsub', 'bsub', 'sbatch', 'srun']):
                    score += weights['command_type']
                elif any(res in pattern.pattern for res in ['-n', '-R', '--ntasks', '--mem']):
                    score += weights['resource_syntax']
                elif any(env in pattern.pattern for env in ['LSF_', 'SLURM_']):
                    score += weights['environment_var']
                else:
                    score += weights['queue_syntax']
        
        return score
    
    def _check_management_node(self, exec_hosts: str, first_exec_host: str, from_host: str = '') -> bool:
        """检查是否为管理节点"""
        # 确保所有输入都是字符串
        exec_hosts = str(exec_hosts) if exec_hosts is not None else ''
        first_exec_host = str(first_exec_host) if first_exec_host is not None else ''
        from_host = str(from_host) if from_host is not None else ''

        for pattern in self.mgmt_node_patterns:
            if (pattern.search(exec_hosts) or
                pattern.search(first_exec_host) or
                pattern.search(from_host)):
                return True
        return False
    
    def _calculate_node_score(self, exec_hosts: str, first_exec_host: str, from_host: str, patterns: List[re.Pattern]) -> float:
        """计算节点分数"""
        # 确保所有输入都是字符串
        exec_hosts = str(exec_hosts) if exec_hosts is not None else ''
        first_exec_host = str(first_exec_host) if first_exec_host is not None else ''
        from_host = str(from_host) if from_host is not None else ''

        score = 0.0
        weights = self.config['generation_filter']['node_classification']['node_weights']

        for pattern in patterns:
            if pattern.search(exec_hosts):
                score += weights.get('exec_hosts', 2.0)
            if pattern.search(first_exec_host):
                score += weights.get('first_exec_host', 1.5)
            if pattern.search(from_host):
                score += weights.get('from_host', 1.0)  # from_host权重较低

        return score

    def _apply_cluster_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用集群过滤，移除一期集群数据"""
        if 'cluster_filtering' not in self.config['generation_filter']:
            return df

        cluster_config = self.config['generation_filter']['cluster_filtering']
        excluded_clusters = cluster_config.get('excluded_clusters', [])

        if not excluded_clusters or 'cluster_name' not in df.columns:
            return df

        original_count = len(df)

        # 过滤掉排除的集群
        mask = ~df['cluster_name'].isin(excluded_clusters)
        filtered_df = df[mask].copy()

        filtered_count = len(filtered_df)
        excluded_count = original_count - filtered_count

        if excluded_count > 0:
            logger.info(f"集群过滤: 移除 {excluded_count:,} 条一期集群记录 ({excluded_clusters})")

        return filtered_df

    def _parse_exec_hosts_detailed(self, exec_hosts_str: str) -> Dict[str, Any]:
        """详细解析exec_hosts字段，支持多种格式"""
        if pd.isna(exec_hosts_str) or exec_hosts_str == '':
            return {
                'hosts': [],
                'node_count': 0,
                'primary_subcluster': 'unknown',
                'is_gpu_job': False,
                'subcluster_distribution': {}
            }

        exec_hosts_str = str(exec_hosts_str).strip()
        hosts = []

        # 1. 处理空格分隔格式: "cpu1-01 cpu1-02 cpu1-03"
        if ' ' in exec_hosts_str and '+' not in exec_hosts_str and '[' not in exec_hosts_str:
            hosts = [h.strip() for h in exec_hosts_str.split() if h.strip()]

        # 2. 处理范围格式: cpu1-[01-05] 或 gpu1-[1-8]
        elif '[' in exec_hosts_str and ']' in exec_hosts_str:
            import re
            match = re.match(r'([a-zA-Z0-9_-]+)\[(\d+)-(\d+)\]', exec_hosts_str)
            if match:
                prefix, start, end = match.groups()
                start_num, end_num = int(start), int(end)
                # 保持原始格式的零填充
                if len(start) == len(end) and len(start) > 1:
                    hosts = [f"{prefix}{i:0{len(start)}d}" for i in range(start_num, end_num + 1)]
                else:
                    hosts = [f"{prefix}{i}" for i in range(start_num, end_num + 1)]

        # 3. 处理加号分隔格式: "cpu1-01+cpu1-02+cpu1-03"
        elif '+' in exec_hosts_str:
            hosts = [h.strip() for h in exec_hosts_str.split('+') if h.strip()]

        # 4. 处理重复格式: "gpu1-31 gpu1-31 gpu1-31 gpu1-31"
        elif ' ' in exec_hosts_str:
            hosts = [h.strip() for h in exec_hosts_str.split() if h.strip()]

        # 5. 单个主机
        else:
            hosts = [exec_hosts_str]

        # 分析主机信息
        node_count = len(hosts)
        subcluster_distribution = {}
        is_gpu_job = False

        for host in hosts:
            # 识别子集群
            subcluster = self._identify_subcluster(host)
            subcluster_distribution[subcluster] = subcluster_distribution.get(subcluster, 0) + 1

            # 检查是否为GPU作业（检查主机名中是否包含gpu）
            if 'gpu' in host.lower():
                is_gpu_job = True

        # 确定主要子集群
        primary_subcluster = max(subcluster_distribution.keys(),
                               key=lambda k: subcluster_distribution[k]) if subcluster_distribution else 'unknown'

        return {
            'hosts': hosts,
            'node_count': node_count,
            'primary_subcluster': primary_subcluster,
            'is_gpu_job': is_gpu_job,
            'subcluster_distribution': subcluster_distribution
        }

    def _identify_subcluster(self, hostname: str) -> str:
        """识别主机所属的子集群"""
        hostname = hostname.lower()

        # GPU集群
        if hostname.startswith('gpu1-'):
            return 'GPU1'
        elif hostname.startswith('gpu2-'):
            return 'GPU2'
        elif hostname.startswith('gpu3-'):
            return 'GPU3'

        # CPU集群
        elif hostname.startswith('cpu1-'):
            return 'CPU1'
        elif hostname.startswith('cpu2-'):
            return 'CPU2'
        elif hostname.startswith('cpu3-'):
            return 'CPU3'

        # 大内存集群
        elif hostname.startswith('bigmem-') or hostname.startswith('bigmen-'):
            return 'BIGMEM'

        # 其他
        else:
            return 'unknown'

    def _calculate_accurate_resources(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算准确的CPU和GPU资源"""
        logger.info("计算准确的CPU和GPU资源...")

        df = df.copy()
        resource_config = self.config.get('resource_calculation', {})
        cpu_configs = resource_config.get('cpu_configs', {})
        gpu_configs = resource_config.get('gpu_configs', {})

        # 初始化资源字段
        df['accurate_cpu_cores'] = 0
        df['accurate_gpu_count'] = 0
        df['accurate_node_count'] = 0
        df['gpu_type'] = ''
        df['subcluster_type'] = ''

        for idx, row in df.iterrows():
            exec_info = self._parse_exec_hosts_detailed(row.get('exec_hosts', ''))
            primary_subcluster = exec_info['primary_subcluster']
            node_count = exec_info['node_count']

            df.at[idx, 'accurate_node_count'] = node_count
            df.at[idx, 'subcluster_type'] = primary_subcluster

            # 计算CPU资源
            if primary_subcluster in cpu_configs:
                cores_per_node = cpu_configs[primary_subcluster]['cores_per_node']
                df.at[idx, 'accurate_cpu_cores'] = cores_per_node * node_count
            elif primary_subcluster in gpu_configs:
                cores_per_node = gpu_configs[primary_subcluster]['cpu_cores_per_node']
                df.at[idx, 'accurate_cpu_cores'] = cores_per_node * node_count
            else:
                # 使用num_processors作为备选
                df.at[idx, 'accurate_cpu_cores'] = row.get('num_processors', 0)

            # 计算GPU资源
            if primary_subcluster in gpu_configs and exec_info['is_gpu_job']:
                gpu_config = gpu_configs[primary_subcluster]
                gpus_per_node = gpu_config['gpus_per_node']
                df.at[idx, 'accurate_gpu_count'] = gpus_per_node * node_count
                df.at[idx, 'gpu_type'] = gpu_config['gpu_type']

        logger.info("资源计算完成")
        return df

    def _make_classification_decisions(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行分类决策 - 使用向量化操作和并行处理"""
        logger.info(f"开始分类决策处理: {len(df):,} 条记录")

        # 使用并行处理进行分类决策
        chunk_size = 50000  # 每个块处理5万条记录
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        logger.info(f"数据分割完成: {len(chunks)}个块, 每块约{chunk_size:,}行")

        # 并行处理各个块
        with ProgressTracker(len(chunks), "分类决策处理") as tracker:
            # 创建进度回调适配器
            last_current = [0]  # 使用列表来存储可变值
            def progress_adapter(current, total):
                # 计算增量并更新
                increment = current - last_current[0]
                if increment > 0:
                    tracker.update(increment)
                    last_current[0] = current

            results = self.processor.process_chunks_with_pool(
                chunks,
                self._classify_chunk,
                progress_callback=progress_adapter
            )

        # 合并结果
        logger.info("合并分类决策结果...")
        classified_df = self.processor.merge_results(results)

        logger.info(f"分类决策完成: {len(classified_df):,} 条记录")
        return classified_df

    def _classify_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """对数据块进行分类决策 - 使用向量化操作"""
        df = chunk_df.copy()

        # 1. 首先进行集群过滤
        df = self._apply_cluster_filtering(df)

        # 如果过滤后没有数据，直接返回
        if len(df) == 0:
            return df

        # 2. 计算准确的资源信息
        df = self._calculate_accurate_resources(df)

        # 初始化分类和置信度
        df['classification'] = 'unknown_category'
        df['confidence_score'] = 0.0

        thresholds = self.config['generation_filter']['classification']['confidence_thresholds']
        weights = self.config['generation_filter']['node_classification']['node_weights']
        node_weight = weights.get('possible_pattern', 3.0)
        script_weight = 1.0

        # 1. 管理节点分类 (向量化)
        management_mask = df['is_management_node'] == True
        df.loc[management_mask, 'classification'] = 'management_nodes'
        df.loc[management_mask, 'confidence_score'] = 1.0

        # 2. 明确的二期节点分类 (向量化)
        decisive_mask = (df['node_classification_decisive'] == True) & (~management_mask)
        df.loc[decisive_mask, 'classification'] = 'second_generation_high'
        df.loc[decisive_mask, 'confidence_score'] = 0.95

        # 3. 需要综合判断的记录 (向量化)
        remaining_mask = (~management_mask) & (~decisive_mask)
        remaining_df = df[remaining_mask].copy()

        if len(remaining_df) > 0:
            # 计算加权总分 (向量化)
            first_gen_total = (remaining_df['possible_first_gen_score'] * node_weight +
                             remaining_df['first_gen_script_score'] * script_weight)
            second_gen_total = (remaining_df['possible_second_gen_score'] * node_weight +
                              remaining_df['second_gen_script_score'] * script_weight)

            # 计算置信度 (向量化)
            total_score = first_gen_total + second_gen_total

            # 二期数据判断
            second_gen_mask = second_gen_total > first_gen_total
            second_gen_indices = remaining_df[second_gen_mask].index

            if len(second_gen_indices) > 0:
                confidence = second_gen_total[second_gen_mask] / (total_score[second_gen_mask] + 1e-6)
                confidence = confidence.fillna(0.5)

                # 高置信度二期
                high_conf_mask = confidence >= thresholds['high_confidence']
                high_conf_indices = second_gen_indices[high_conf_mask]
                df.loc[high_conf_indices, 'classification'] = 'second_generation_high'
                df.loc[high_conf_indices, 'confidence_score'] = confidence[high_conf_mask]

                # 中等置信度二期
                medium_conf_mask = (confidence >= thresholds['medium_confidence']) & (confidence < thresholds['high_confidence'])
                medium_conf_indices = second_gen_indices[medium_conf_mask]
                df.loc[medium_conf_indices, 'classification'] = 'second_generation_medium'
                df.loc[medium_conf_indices, 'confidence_score'] = confidence[medium_conf_mask]

                # 低置信度二期
                low_conf_mask = confidence < thresholds['medium_confidence']
                low_conf_indices = second_gen_indices[low_conf_mask]
                df.loc[low_conf_indices, 'classification'] = 'second_generation_low'
                df.loc[low_conf_indices, 'confidence_score'] = confidence[low_conf_mask]

            # 一期数据判断
            first_gen_mask = (first_gen_total > second_gen_total) & (~second_gen_mask)
            first_gen_indices = remaining_df[first_gen_mask].index

            if len(first_gen_indices) > 0:
                confidence = first_gen_total[first_gen_mask] / (total_score[first_gen_mask] + 1e-6)
                confidence = confidence.fillna(0.5)
                df.loc[first_gen_indices, 'classification'] = 'first_generation'
                df.loc[first_gen_indices, 'confidence_score'] = confidence

            # 混合特征判断
            mixed_mask = (first_gen_total > 0) & (second_gen_total > 0) & (first_gen_total == second_gen_total)
            mixed_indices = remaining_df[mixed_mask].index
            df.loc[mixed_indices, 'classification'] = 'mixed_features'
            df.loc[mixed_indices, 'confidence_score'] = 0.5

        return df
    
    def _separate_categories(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """分离不同类别的数据"""
        output_files = {}
        
        for category in self.config['generation_filter']['classification']['output_categories']:
            category_df = df[df['classification'] == category].copy()
            
            if len(category_df) > 0:
                # 移除分析用的临时列
                columns_to_drop = [
                    'first_gen_script_score', 'second_gen_script_score',
                    'definitive_second_gen_score', 'possible_first_gen_score', 'possible_second_gen_score',
                    'is_management_node', 'node_classification_decisive', 'classification', 'confidence_score'
                ]
                category_df = category_df.drop(columns=columns_to_drop, errors='ignore')
                
                # 保存文件
                output_file = os.path.join(output_dir, f"{category}.csv")
                category_df.to_csv(output_file, index=False)
                output_files[category] = output_file
                
                # 更新统计
                self.stats[category] = len(category_df)
                
                logger.info(f"保存 {category}: {len(category_df):,} 条记录 -> {output_file}")
        
        return output_files
    
    def _generate_classification_report(self, output_dir: str):
        """生成分类报告"""
        report_file = os.path.join(output_dir, "classification_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 一期二期数据分类报告 ===\n\n")
            f.write(f"总作业数: {self.stats['total_jobs']:,}\n\n")
            
            f.write("分类结果:\n")
            for category, count in self.stats.items():
                if category != 'total_jobs' and count > 0:
                    percentage = count / self.stats['total_jobs'] * 100
                    f.write(f"  {category}: {count:,} ({percentage:.2f}%)\n")
            
            f.write(f"\n二期数据总计: {self.stats['second_generation_high'] + self.stats['second_generation_medium'] + self.stats['second_generation_low']:,}\n")
            f.write(f"数据保留率: {(self.stats['second_generation_high'] + self.stats['second_generation_medium'] + self.stats['second_generation_low']) / self.stats['total_jobs'] * 100:.2f}%\n")
        
        logger.info(f"分类报告已保存: {report_file}")
