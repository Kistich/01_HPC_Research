#!/usr/bin/env python3
"""
用户ID推断器
基于作业指纹进行智能用户推断
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from pathlib import Path
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
import warnings
warnings.filterwarnings('ignore')

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from parallel_processor import ParallelProcessor
from progress_tracker import ProgressTracker
from exec_hosts_parser import ExecHostsParser

logger = logging.getLogger(__name__)

class UserInferrer:
    """用户ID推断器"""
    
    def __init__(self, config_path: str):
        """
        初始化用户推断器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.processor = ParallelProcessor(
            max_cores=self.config['parallel_processing']['max_cores']
        )

        # 初始化exec_hosts解析器
        cluster_config_path = str(Path(config_path).parent / "cluster_config.yaml")
        self.exec_hosts_parser = ExecHostsParser(cluster_config_path)

        # 统计信息
        self.stats = {
            'total_jobs': 0,
            'user_id_missing': 0,
            'clusters_found': 0,
            'high_confidence_inferred': 0,
            'medium_confidence_inferred': 0,
            'low_confidence_inferred': 0,
            'singleton_jobs': 0
        }

        logger.info("用户ID推断器初始化完成")
    
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
    
    def infer_user_ids(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        推断用户ID
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径字典
        """
        logger.info(f"开始用户ID推断: {input_file}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info("加载数据...")
        df = pd.read_csv(input_file)
        self.stats['total_jobs'] = len(df)
        
        logger.info(f"数据加载完成: {len(df):,} 条记录")
        
        # 检查用户ID缺失情况
        missing_mask = df['user_id'].isna() if 'user_id' in df.columns else pd.Series([True] * len(df))
        self.stats['user_id_missing'] = missing_mask.sum()
        
        if self.stats['user_id_missing'] == 0:
            logger.info("所有作业都有用户ID，无需推断")
            return self._save_results_no_inference(df, output_dir)
        
        logger.info(f"需要推断用户ID的作业: {self.stats['user_id_missing']:,} 条")
        
        # 步骤1: 构建作业指纹
        logger.info("步骤1: 构建作业指纹...")
        df = self._build_job_fingerprints(df)
        
        # 步骤2: 聚类分析
        logger.info("步骤2: 聚类分析...")
        df = self._perform_clustering(df)
        
        # 步骤3: 用户推断
        logger.info("步骤3: 用户推断...")
        df = self._infer_users_from_clusters(df)
        
        # 步骤4: 推断验证
        logger.info("步骤4: 推断验证...")
        df = self._validate_inference(df)
        
        # 保存结果
        output_files = self._save_results(df, output_dir)
        
        # 生成推断报告
        self._generate_inference_report(output_dir)
        
        logger.info("用户ID推断完成")
        return output_files
    
    def _build_job_fingerprints(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建作业指纹"""
        df = df.copy()
        
        logger.info("构建基础特征...")
        
        # 基础特征 - 使用向量化操作
        def vectorized_hash(series):
            """向量化的哈希计算"""
            return series.fillna('').astype(str).apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:8])

        df['job_name_hash'] = vectorized_hash(df['job_name'])
        # 使用from_host作为工作目录的替代特征
        df['working_dir_hash'] = vectorized_hash(df.get('from_host', pd.Series([''] * len(df))))
        df['command_hash'] = vectorized_hash(df.get('command', pd.Series([''] * len(df))))
        df['queue_name_clean'] = df.get('queue', pd.Series(['default'] * len(df))).fillna('default')

        # 资源配置特征 - 使用并行处理解析exec_hosts
        logger.info("解析exec_hosts字段计算实际资源...")

        # 分块处理exec_hosts解析
        chunk_size = 10000
        exec_hosts_chunks = [df['exec_hosts'][i:i + chunk_size] for i in range(0, len(df), chunk_size)]

        # 并行解析exec_hosts
        with ProgressTracker(len(exec_hosts_chunks), "exec_hosts解析") as tracker:
            exec_results = []
            for chunk in exec_hosts_chunks:
                chunk_results = chunk.apply(self.exec_hosts_parser.parse_exec_hosts_value)
                exec_results.append(chunk_results)
                tracker.update(1)

        # 合并结果
        exec_hosts_results = pd.concat(exec_results, ignore_index=True)

        # 提取资源信息 - 向量化操作
        df['actual_cpu_cores'] = [x.get('total_cpu_cores', 0) for x in exec_hosts_results]
        df['actual_gpu_count'] = [x.get('total_gpu_count', 0) for x in exec_hosts_results]
        df['actual_node_count'] = [x.get('node_count', 0) for x in exec_hosts_results]

        # 标准化资源特征
        df['cpu_count_norm'] = df['actual_cpu_cores'].fillna(0)
        df['memory_norm'] = pd.to_numeric(df.get('max_mem', pd.Series([0] * len(df))), errors='coerce').fillna(0)
        df['gpu_count_norm'] = df['actual_gpu_count'].fillna(0)
        df['node_count_norm'] = df['actual_node_count'].fillna(0)
        
        # 时间模式特征
        if 'submit_time' in df.columns:
            df['submit_hour'] = pd.to_datetime(df['submit_time'], errors='coerce').dt.hour.fillna(12)
            df['submit_weekday'] = pd.to_datetime(df['submit_time'], errors='coerce').dt.weekday.fillna(1)
        else:
            df['submit_hour'] = 12
            df['submit_weekday'] = 1
        
        # 行为模式特征
        if 'duration' in df.columns:
            df['duration_norm'] = pd.to_numeric(df['duration'], errors='coerce').fillna(300)
        else:
            df['duration_norm'] = pd.Series([300] * len(df))

        if 'exit_status' in df.columns:
            df['success_rate'] = (pd.to_numeric(df['exit_status'], errors='coerce') == 0).astype(int)
        else:
            df['success_rate'] = pd.Series([1] * len(df))
        
        logger.info("作业指纹构建完成")
        return df
    
    def _perform_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """执行聚类分析"""
        df = df.copy()
        
        # 只对缺失用户ID的作业进行聚类
        missing_mask = df['user_id'].isna() if 'user_id' in df.columns else pd.Series([True] * len(df))
        missing_df = df[missing_mask].copy()
        
        if len(missing_df) == 0:
            return df
        
        logger.info(f"对 {len(missing_df):,} 条记录进行聚类分析...")
        
        # 准备特征矩阵
        feature_columns = [
            'job_name_hash', 'working_dir_hash', 'command_hash', 'queue_name_clean',
            'cpu_count_norm', 'memory_norm', 'gpu_count_norm',
            'submit_hour', 'submit_weekday', 'duration_norm', 'success_rate'
        ]
        
        # 处理分类特征
        categorical_features = ['job_name_hash', 'working_dir_hash', 'command_hash', 'queue_name_clean']
        numerical_features = [col for col in feature_columns if col not in categorical_features]
        
        # 编码分类特征
        feature_matrix = []
        
        # 数值特征标准化
        if numerical_features:
            scaler = StandardScaler()
            numerical_data = scaler.fit_transform(missing_df[numerical_features])
            feature_matrix.append(numerical_data)
        
        # 分类特征编码
        for cat_feature in categorical_features:
            unique_values = missing_df[cat_feature].unique()
            for value in unique_values:
                feature_matrix.append((missing_df[cat_feature] == value).astype(int).values.reshape(-1, 1))
        
        # 合并特征矩阵
        if feature_matrix:
            X = np.hstack(feature_matrix)
        else:
            logger.warning("无有效特征用于聚类")
            return df
        
        # 执行聚类
        cluster_labels = self._run_clustering_algorithms(X)
        
        # 将聚类结果映射回DataFrame
        df.loc[missing_mask, 'cluster_id'] = cluster_labels
        
        # 统计聚类结果
        unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        self.stats['clusters_found'] = unique_clusters
        
        logger.info(f"聚类完成: 发现 {unique_clusters} 个聚类")
        
        return df
    
    def _run_clustering_algorithms(self, X: np.ndarray) -> np.ndarray:
        """运行聚类算法 - 针对大数据优化"""
        logger.info(f"对 {len(X):,} 条记录进行聚类分析...")

        # 对于大数据集，使用更高效的算法
        if len(X) > 100000:  # 超过10万条记录
            logger.info("数据量较大，使用MiniBatchKMeans进行预聚类...")

            # 第一步：使用MiniBatchKMeans进行粗聚类
            from sklearn.cluster import MiniBatchKMeans
            n_init_clusters = min(1000, len(X) // 100)  # 初始聚类数

            kmeans = MiniBatchKMeans(
                n_clusters=n_init_clusters,
                batch_size=10000,
                random_state=42,
                n_init=3
            )

            coarse_labels = kmeans.fit_predict(X)
            logger.info(f"粗聚类完成，产生 {len(set(coarse_labels))} 个初始聚类")

            # 第二步：对每个粗聚类内部使用DBSCAN细化
            final_labels = np.full(len(X), -1, dtype=int)
            current_cluster_id = 0

            for cluster_id in set(coarse_labels):
                if cluster_id == -1:  # 噪声点
                    continue

                cluster_mask = coarse_labels == cluster_id
                cluster_data = X[cluster_mask]

                if len(cluster_data) < 10:  # 太小的聚类直接标记
                    final_labels[cluster_mask] = current_cluster_id
                    current_cluster_id += 1
                    continue

                # 对聚类内部使用DBSCAN
                try:
                    dbscan = DBSCAN(eps=0.3, min_samples=3)
                    sub_labels = dbscan.fit_predict(cluster_data)

                    # 重新编号
                    for sub_cluster_id in set(sub_labels):
                        if sub_cluster_id == -1:
                            continue
                        sub_mask = sub_labels == sub_cluster_id
                        global_mask = np.where(cluster_mask)[0][sub_mask]
                        final_labels[global_mask] = current_cluster_id
                        current_cluster_id += 1

                except Exception as e:
                    logger.warning(f"子聚类失败: {e}，使用粗聚类结果")
                    final_labels[cluster_mask] = current_cluster_id
                    current_cluster_id += 1

            logger.info(f"分层聚类完成，最终产生 {len(set(final_labels[final_labels >= 0]))} 个聚类")
            return final_labels

        else:
            # 小数据集使用原来的方法
            try:
                logger.info("使用DBSCAN聚类...")
                clusterer = DBSCAN(eps=0.3, min_samples=3)
                labels = clusterer.fit_predict(X)
                logger.info(f"DBSCAN聚类完成，产生 {len(set(labels[labels >= 0]))} 个聚类")
                return labels

            except Exception as e:
                logger.warning(f"DBSCAN聚类失败: {e}，使用默认标签")
                return np.zeros(len(X), dtype=int)
    
    def _infer_users_from_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """从聚类结果推断用户"""
        df = df.copy()
        
        if 'cluster_id' not in df.columns:
            return df
        
        inference_rules = self.config['user_inference']['user_inference_rules']
        confidence_levels = inference_rules['confidence_levels']
        
        # 初始化推断字段
        df['inferred_user_id'] = df.get('user_id', '')
        df['inference_confidence'] = 'existing'
        
        # 分析每个聚类
        for cluster_id in df['cluster_id'].unique():
            if pd.isna(cluster_id) or cluster_id == -1:
                continue
            
            cluster_mask = df['cluster_id'] == cluster_id
            cluster_data = df[cluster_mask]
            cluster_size = len(cluster_data)
            
            # 计算聚类质量指标
            feature_consistency = self._calculate_feature_consistency(cluster_data)
            
            # 确定置信度级别
            if (cluster_size >= confidence_levels['high']['min_cluster_size'] and 
                feature_consistency >= confidence_levels['high']['min_feature_consistency']):
                confidence = 'high'
                user_id_format = confidence_levels['high']['id_format']
                self.stats['high_confidence_inferred'] += cluster_size
            elif (cluster_size >= confidence_levels['medium']['min_cluster_size'] and 
                  feature_consistency >= confidence_levels['medium']['min_feature_consistency']):
                confidence = 'medium'
                user_id_format = confidence_levels['medium']['id_format']
                self.stats['medium_confidence_inferred'] += cluster_size
            elif cluster_size >= confidence_levels['low']['min_cluster_size']:
                confidence = 'low'
                user_id_format = confidence_levels['low']['id_format']
                self.stats['low_confidence_inferred'] += cluster_size
            else:
                # 单例作业
                confidence = 'singleton'
                user_id_format = confidence_levels['singleton']['id_format']
                self.stats['singleton_jobs'] += cluster_size
            
            # 分配推断用户ID
            inferred_user_id = user_id_format.format(int(cluster_id) + 1)
            
            # 只更新缺失用户ID的记录
            missing_in_cluster = cluster_mask & (df['user_id'].isna() if 'user_id' in df.columns else True)
            df.loc[missing_in_cluster, 'inferred_user_id'] = inferred_user_id
            df.loc[missing_in_cluster, 'inference_confidence'] = confidence
        
        return df
    
    def _calculate_feature_consistency(self, cluster_data: pd.DataFrame) -> float:
        """计算特征一致性"""
        consistency_scores = []
        
        # 检查作业名一致性
        if 'job_name' in cluster_data.columns:
            unique_job_names = cluster_data['job_name'].nunique()
            job_name_consistency = 1.0 / max(1, unique_job_names)
            consistency_scores.append(job_name_consistency)
        
        # 检查资源配置一致性
        if 'cpu_count_norm' in cluster_data.columns:
            cpu_std = cluster_data['cpu_count_norm'].std()
            cpu_consistency = 1.0 / (1.0 + cpu_std)
            consistency_scores.append(cpu_consistency)
        
        # 检查时间模式一致性
        if 'submit_hour' in cluster_data.columns:
            hour_std = cluster_data['submit_hour'].std()
            time_consistency = 1.0 / (1.0 + hour_std / 24.0)
            consistency_scores.append(time_consistency)
        
        return np.mean(consistency_scores) if consistency_scores else 0.0

    def _validate_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证推断结果"""
        df = df.copy()

        # 简单验证：检查推断结果的合理性
        if 'inferred_user_id' in df.columns:
            inferred_mask = df['inference_confidence'].isin(['high', 'medium', 'low', 'singleton'])
            inferred_count = inferred_mask.sum()

            logger.info(f"推断验证: {inferred_count:,} 条记录完成推断")

            # 统计各置信度级别的数量
            for confidence in ['high', 'medium', 'low', 'singleton']:
                count = (df['inference_confidence'] == confidence).sum()
                if count > 0:
                    logger.info(f"  {confidence}置信度: {count:,} 条")

        return df

    def _save_results_no_inference(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """保存无需推断的结果"""
        output_files = {}

        output_file = os.path.join(output_dir, "user_inference_complete.csv")
        df.to_csv(output_file, index=False)
        output_files['complete_data'] = output_file

        logger.info(f"保存完整数据: {len(df):,} 条 -> {output_file}")
        return output_files

    def _save_results(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """保存推断结果"""
        output_files = {}

        # 清理临时列
        temp_columns = ['job_name_hash', 'working_dir_hash', 'command_hash', 'queue_name_clean',
                       'cpu_count_norm', 'memory_norm', 'gpu_count_norm', 'submit_hour',
                       'submit_weekday', 'duration_norm', 'success_rate', 'cluster_id']

        result_df = df.drop(columns=temp_columns, errors='ignore')

        # 合并原始用户ID和推断用户ID
        if 'user_id' in result_df.columns and 'inferred_user_id' in result_df.columns:
            result_df['final_user_id'] = result_df['user_id'].fillna(result_df['inferred_user_id'])
            # 对于仍然为空的记录，分配默认用户ID
            still_missing = result_df['final_user_id'].isna()
            if still_missing.any():
                logger.warning(f"发现 {still_missing.sum()} 条记录无法推断用户ID，分配默认ID")
                result_df.loc[still_missing, 'final_user_id'] = 'unknown_user'
        elif 'inferred_user_id' in result_df.columns:
            result_df['final_user_id'] = result_df['inferred_user_id'].fillna('unknown_user')
        else:
            result_df['final_user_id'] = 'unknown_user'

        # 保存主要结果
        main_output_file = os.path.join(output_dir, "user_inference_complete.csv")
        result_df.to_csv(main_output_file, index=False)
        output_files['complete_data'] = main_output_file

        logger.info(f"保存推断结果: {len(result_df):,} 条 -> {main_output_file}")

        # 保存推断映射表
        if 'inferred_user_id' in df.columns:
            inference_mapping = df[df['inference_confidence'].isin(['high', 'medium', 'low', 'singleton'])][
                ['inferred_user_id', 'inference_confidence', 'cluster_id']
            ].drop_duplicates()

            mapping_file = os.path.join(output_dir, "user_inference_mapping.csv")
            inference_mapping.to_csv(mapping_file, index=False)
            output_files['mapping'] = mapping_file

            logger.info(f"保存推断映射: {len(inference_mapping):,} 条 -> {mapping_file}")

        return output_files

    def _generate_inference_report(self, output_dir: str):
        """生成推断报告"""
        report_file = os.path.join(output_dir, "user_inference_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 用户ID推断报告 ===\n\n")
            f.write(f"总作业数: {self.stats['total_jobs']:,}\n")
            f.write(f"用户ID缺失: {self.stats['user_id_missing']:,} ({self.stats['user_id_missing']/self.stats['total_jobs']*100:.2f}%)\n\n")

            f.write("聚类结果:\n")
            f.write(f"  发现聚类: {self.stats['clusters_found']:,} 个\n\n")

            f.write("推断结果:\n")
            f.write(f"  高置信度推断: {self.stats['high_confidence_inferred']:,} 条\n")
            f.write(f"  中等置信度推断: {self.stats['medium_confidence_inferred']:,} 条\n")
            f.write(f"  低置信度推断: {self.stats['low_confidence_inferred']:,} 条\n")
            f.write(f"  单例作业: {self.stats['singleton_jobs']:,} 条\n\n")

            total_inferred = (self.stats['high_confidence_inferred'] +
                            self.stats['medium_confidence_inferred'] +
                            self.stats['low_confidence_inferred'] +
                            self.stats['singleton_jobs'])

            f.write(f"推断覆盖率: {total_inferred/self.stats['user_id_missing']*100:.2f}%\n")

        logger.info(f"推断报告已保存: {report_file}")
