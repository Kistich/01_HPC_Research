#!/usr/bin/env python3

import pandas as pd
import os
import re
from collections import defaultdict
from tqdm import tqdm
import logging
import numpy as np
import multiprocessing
from functools import partial
import gc

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClusterClassifier:
    def __init__(self):
        # 初始化集群分类字典
        self.cluster_generation = {}
        
        # 定义要检查的列
        self.resource_columns = [
            'res_req', 'exec_hosts', 'first_exec_host', 'from_host',
            'job_name', 'application', 'queue', 'command'
        ]
        
        # 从提交脚本模板直接提取的特征
        # 一期集群特征(UniScheduler系统)
        self.gen1_patterns = [
            # 脚本指令和环境
            r'#BSUB\b',                    # 作业脚本标识符
            r'\bjsub\b',                   # 作业提交命令
            r'\bbsub\b',                   # 作业提交命令
            r'-q\s+\w+',                   # 队列指定参数
            
            # 特有环境变量
            r'\$MPI_HOSTFILE\b',           # 机器文件环境变量
            r'\$OMPI_HOSTFILE\b',          # OpenMPI机器文件环境变量
            r'\$JH_HOSTFILE\b',            # JH机器文件环境变量
            r'profile\.unischeduler\b',    # UniScheduler环境
            r'/opt/jhinno',                # jhinno路径
            
            # 一期节点命名格式
            r'(cpu|gpu)[0-9]{2}(?!-)',     # 例如: cpu01, gpu02 (后面不跟-)
            r'\b(gnode-|cnode-)[0-9]+\b',  # 特殊一期节点命名
        ]
        
        # 二期集群特征(SLURM系统)
        self.gen2_patterns = [
            # 脚本指令和命令
            r'#SBATCH\b',                  # 作业脚本标识符
            r'\bsbatch\b',                 # 作业提交命令
            r'\bsrun\b',                   # 交互式作业命令
            
            # SLURM特有参数
            r'-p\s+\w+',                   # 分区参数
            r'--partition=\w+',            # 分区参数长格式
            r'--gres=gpu',                 # GPU资源请求
            r'--ntasks-per-node',          # 每节点任务数
            
            # SLURM环境变量
            r'\$SLURM_STEP_NUM_TASKS\b',   # SLURM任务数环境变量
            r'\$SLURM_NTASKS\b',           # SLURM任务数环境变量
            r'\$SLURM_JOB_NODELIST\b',     # SLURM节点列表
            
            # 二期特有分区名
            r'\b(i64m512[ur]|a128m512u|i96m3tu|i64m1tg)\b',
            
            # 二期节点命名格式
            r'(cpu|gpu|train)[0-9]+-[0-9]+',  # 例如: cpu1-14, gpu2-08
            
            # 国产AI平台特征
            r'ascend',                     # 华为昇腾
            r'npu',                        # 神经网络处理单元
            r'mindspore',                  # 华为深度学习框架
            r'atlas',                      # 华为Atlas系列产品
        ]
        
        # 特定集群的强制规则
        self.special_rules = {
            "jhcluster": {
                "patterns": [r'#BSUB', r'jsub', r'bsub', r'/opt/jhinno', r'unischeduler', r'\$JH_HOSTFILE'],
                "generation": 1  # 一期集群
            },
            "ASCEND-AI": {
                "patterns": [r'ascend', r'npu', r'mindspore', r'huawei', r'atlas'],
                "generation": 2  # 二期集群
            }
        }
        
        # 数据类型优化
        self.dtype_optimizations = {
            'job_id': 'int64',
            'user_name': 'category',
            'queue': 'category',
            'from_host': 'category',
            'exec_host': 'category',
            'first_exec_host': 'category',
            'job_name': 'category',
            'application': 'category',
            'command': 'object',
            'res_req': 'object'
        }
    
    def _process_chunk(self, chunk, job_ids):
        """并行处理数据块"""
        results = []
        match_info_list = []
        
        for _, row in chunk.iterrows():
            gen, match_info = self._identify_generation(row)
            results.append(gen)
            match_info_list.append(match_info)
        
        return list(zip(job_ids, results, match_info_list))
        
    def _identify_generation(self, row):
        """根据脚本特征判断集群代次"""
        gen1_matches = 0
        gen2_matches = 0
        
        # 先检查是否有特殊规则
        if pd.notna(row.get('cluster_name')):
            cluster = row['cluster_name']
            if cluster in self.special_rules:
                # 检查是否满足特殊规则的模式
                rule = self.special_rules[cluster]
                pattern_matches = 0
                total_patterns = len(rule["patterns"])
                
                # 检查所有可能包含信息的列
                for col in self.resource_columns:
                    if col not in row or pd.isna(row[col]):
                        continue
                        
                    value = str(row[col]).lower()
                    
                    # 检查规则中的模式
                    for pattern in rule["patterns"]:
                        if re.search(pattern, value, re.IGNORECASE):
                            pattern_matches += 1
                            break  # 每个模式只计算一次
                
                # 如果匹配度超过30%，应用特殊规则
                if pattern_matches > 0 and pattern_matches / total_patterns >= 0.3:
                    return rule["generation"], {
                        'gen1_matches': gen1_matches,
                        'gen2_matches': gen2_matches,
                        'special_rule_applied': True,
                        'special_rule_matches': pattern_matches,
                        'special_rule_total': total_patterns
                    }
        
        # 检查所有可能包含信息的列
        for col in self.resource_columns:
            if col not in row or pd.isna(row[col]):
                continue
                
            value = str(row[col]).lower()
            
            # 检查一期特征
            for pattern in self.gen1_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    gen1_matches += 1
            
            # 检查二期特征
            for pattern in self.gen2_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    gen2_matches += 1
        
        # 保存匹配数量，用于后续分析
        match_info = {
            'gen1_matches': gen1_matches,
            'gen2_matches': gen2_matches
        }
        
        # 基于特征匹配数量判断代次
        if gen1_matches > 0 and gen2_matches == 0:
            return 1, match_info
        elif gen2_matches > 0 and gen1_matches == 0:
            return 2, match_info
        elif gen1_matches > gen2_matches:
            return 1, match_info
        elif gen2_matches > gen1_matches:
            return 2, match_info
        elif gen1_matches > 0 or gen2_matches > 0:
            # 如果匹配数相等但不为零，返回两者都有
            return "both", match_info
        else:
            return None, match_info
    
    def classify_jobs(self, input_file, num_processes=None):
        """对全量作业进行分类，使用多进程加速"""
        logger.info(f"读取输入文件: {input_file}")
        
        # 如果未指定进程数，则使用系统CPU核心数的75%
        if num_processes is None:
            num_processes = max(1, int(multiprocessing.cpu_count() * 0.75))
        
        logger.info(f"将使用 {num_processes} 个进程进行并行处理")
        
        # 读取全量数据，应用数据类型优化
        logger.info("读取全量数据...")
        df = pd.read_csv(input_file, low_memory=False, dtype=self.dtype_optimizations)
        
        total_rows = len(df)
        logger.info(f"总行数: {total_rows:,}")
        
        # 按集群名分组统计
        cluster_counts = df['cluster_name'].value_counts()
        logger.info(f"发现 {len(cluster_counts)} 个集群")
        
        for cluster, count in cluster_counts.items():
            logger.info(f"集群 {cluster}: {count:,} 作业 ({count/total_rows*100:.2f}%)")
        
        # 分析每个作业的集群代次
        logger.info("开始分析全量数据...")
        
        # 创建新列来存储匹配信息
        df['gen1_matches'] = 0
        df['gen2_matches'] = 0
        
        # 将数据分块处理
        chunk_size = max(1, total_rows // (num_processes * 4))  # 每个进程处理多个块
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        job_ids_chunks = [chunk.index.tolist() for chunk in chunks]
        
        logger.info(f"数据已分为 {len(chunks)} 个块，每块约 {chunk_size:,} 条记录")
        
        # 使用多进程处理数据
        with multiprocessing.Pool(processes=num_processes) as pool:
            all_results = []
            
            # 并行处理所有块
            process_func = partial(self._process_chunk)
            
            # 使用tqdm显示进度
            for result in tqdm(
                pool.starmap(process_func, zip(chunks, job_ids_chunks)),
                total=len(chunks),
                desc="处理数据块"
            ):
                all_results.extend(result)
            
            # 释放内存
            del chunks
            gc.collect()
        
        # 整理结果
        logger.info("整理分析结果...")
        results_dict = {job_id: (gen, info) for job_id, gen, info in all_results}
        
        # 添加结果到DataFrame
        df['cluster_gen'] = [results_dict[job_id][0] for job_id in df.index]
        
        # 添加匹配信息
        for job_id, row in df.iterrows():
            if job_id in results_dict:
                info = results_dict[job_id][1]
                df.loc[job_id, 'gen1_matches'] = info['gen1_matches']
                df.loc[job_id, 'gen2_matches'] = info['gen2_matches']
        
        # 分析每个集群的主要代次
        cluster_gen_count = defaultdict(lambda: defaultdict(int))
        for _, row in df.iterrows():
            if pd.notna(row['cluster_gen']) and row['cluster_gen'] != "both":
                cluster = row['cluster_name']
                gen = row['cluster_gen']
                cluster_gen_count[cluster][gen] += 1
        
        # 为每个集群确定主要代次
        for cluster, counts in cluster_gen_count.items():
            if counts:
                majority_gen, count = max(counts.items(), key=lambda x: x[1])
                
                total = sum(counts.values())
                confidence = count / total
                self.cluster_generation[cluster] = majority_gen
                
                gen1_count = counts.get(1, 0)
                gen2_count = counts.get(2, 0)
                
                logger.info(f"集群 {cluster}: 第{majority_gen}期 (置信度: {confidence:.2f})")
                logger.info(f"  - 样本详情: 一期样本: {gen1_count} ({gen1_count/total*100:.1f}%), 二期样本: {gen2_count} ({gen2_count/total*100:.1f}%)")
        
        # 标记异常记录 (与集群主要代次不符的记录)
        df['is_anomaly'] = False
        
        for i, row in df.iterrows():
            if pd.notna(row['cluster_gen']) and row['cluster_gen'] != "both":
                cluster = row['cluster_name']
                # 如果该集群有确定的主要代次
                if cluster in self.cluster_generation:
                    # 如果该行的代次与集群主要代次不符，标记为异常
                    if row['cluster_gen'] != self.cluster_generation[cluster]:
                        df.loc[i, 'is_anomaly'] = True
            elif row['cluster_gen'] == "both":
                # 同时具有一期和二期特征的也标记为异常
                df.loc[i, 'is_anomaly'] = True
        
        logger.info("分类完成!")
        return df
    
    def save_classified_jobs(self, df, output_dir):
        """保存分类结果"""
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 根据分类结果分割数据 - 排除异常值
        first_gen = df[(df['cluster_gen'] == 1) | 
                     ((df['cluster_name'].isin(self.cluster_generation)) & 
                     (df['cluster_name'].map(self.cluster_generation) == 1)) &
                    (df['is_anomaly'] == False)]  # 排除异常记录
        
        second_gen = df[(df['cluster_gen'] == 2) | 
                      ((df['cluster_name'].isin(self.cluster_generation)) & 
                      (df['cluster_name'].map(self.cluster_generation) == 2)) &
                     (df['is_anomaly'] == False)]  # 排除异常记录
        
        unknown_gen = df[~df.index.isin(first_gen.index) & 
                       ~df.index.isin(second_gen.index) &
                       (df['is_anomaly'] == False)]  # 排除异常记录
        
        # 筛选异常记录
        anomaly_records = df[df['is_anomaly'] == True]
        
        # 保存文件路径
        first_gen_file = os.path.join(output_dir, 'first_generation_jobs.csv')
        second_gen_file = os.path.join(output_dir, 'second_generation_jobs.csv')
        unknown_gen_file = os.path.join(output_dir, 'unknown_generation_jobs.csv')
        anomaly_file = os.path.join(output_dir, 'anomaly_records.csv')
        
        # 删除临时列
        columns_to_drop = ['gen1_matches', 'gen2_matches', 'is_anomaly']
        
        # 保存为CSV
        logger.info(f"保存分类结果到 {output_dir}...")
        
        if len(first_gen) > 0:
            first_gen_out = first_gen.drop(columns=columns_to_drop, errors='ignore')
            first_gen_out.to_csv(first_gen_file, index=False)
        
        if len(second_gen) > 0:
            second_gen_out = second_gen.drop(columns=columns_to_drop, errors='ignore')
            second_gen_out.to_csv(second_gen_file, index=False)
        
        if len(unknown_gen) > 0:
            unknown_gen_out = unknown_gen.drop(columns=columns_to_drop, errors='ignore')
            unknown_gen_out.to_csv(unknown_gen_file, index=False)
        
        # 保存异常记录(保留匹配信息列)
        if len(anomaly_records) > 0:
            anomaly_records.to_csv(anomaly_file, index=False)
            logger.info(f"发现 {len(anomaly_records):,} 条异常记录 ({len(anomaly_records)/len(df)*100:.2f}%)")
        
        logger.info(f"\n结果已保存到 {output_dir}")
        logger.info(f"一期集群作业: {len(first_gen):,} 行 ({len(first_gen)/len(df)*100:.2f}%)")
        logger.info(f"二期集群作业: {len(second_gen):,} 行 ({len(second_gen)/len(df)*100:.2f}%)")
        if len(unknown_gen) > 0:
            logger.info(f"未知集群作业: {len(unknown_gen):,} 行 ({len(unknown_gen)/len(df)*100:.2f}%)")
        
        # 保存分类统计信息
        stats_file = os.path.join(output_dir, 'classification_stats.txt')
        total_jobs = len(df)
        
        # 创建汇总表，按集群和代次分组
        summary_df = df.groupby(['cluster_name', 'cluster_gen']).size().reset_index(name='count')
        summary_df['percentage'] = summary_df['count'] / total_jobs * 100
        
        with open(stats_file, 'w') as f:
            f.write("=== 集群分类统计 ===\n\n")
            f.write(f"总分析作业数: {total_jobs:,}\n")
            f.write(f"一期集群作业: {len(first_gen):,} ({len(first_gen)/total_jobs*100:.2f}%)\n")
            f.write(f"二期集群作业: {len(second_gen):,} ({len(second_gen)/total_jobs*100:.2f}%)\n")
            if len(unknown_gen) > 0:
                f.write(f"未知集群作业: {len(unknown_gen):,} ({len(unknown_gen)/total_jobs*100:.2f}%)\n")
            
            f.write(f"异常记录数: {len(anomaly_records):,} ({len(anomaly_records)/total_jobs*100:.2f}%)\n\n")
            
            f.write("\n=== 按集群名称和代次的详细统计 ===\n\n")
            for _, row in summary_df.sort_values(['cluster_gen', 'count'], ascending=[True, False]).iterrows():
                gen = int(row['cluster_gen']) if pd.notna(row['cluster_gen']) and row['cluster_gen'] != "both" else "混合特征"
                f.write(f"集群: {row['cluster_name']}, 代次: {gen}, 作业数: {row['count']:,} ({row['percentage']:.2f}%)\n")
            
            f.write("\n=== 集群代次映射关系 ===\n\n")
            for cluster, gen in self.cluster_generation.items():
                f.write(f"{cluster}: 第{gen}期\n")
                
            # 添加判断依据说明
            f.write("\n=== 代次判断依据 ===\n\n")
            f.write("一期集群特征(UniScheduler):\n")
            for pattern in self.gen1_patterns:
                f.write(f"- {pattern}\n")
            
            f.write("\n二期集群特征(SLURM):\n")
            for pattern in self.gen2_patterns:
                f.write(f"- {pattern}\n")
            
            f.write("\n特定集群强制规则:\n")
            for cluster, rule in self.special_rules.items():
                f.write(f"- {cluster}: 归属于第{rule['generation']}期集群\n")
                f.write(f"  模式: {', '.join(rule['patterns'])}\n")
            
            f.write("\n判断逻辑:\n")
            f.write("1. 首先检查特定集群的强制规则\n")
            f.write("2. 检查每一行数据是否包含一期或二期特征\n")
            f.write("3. 统计匹配到的一期和二期特征数量\n")
            f.write("4. 根据匹配特征数量的多少决定代次\n")
            f.write("5. 对于特征匹配数相等的情况，标记为'混合特征'\n")
            f.write("6. 与集群主要代次不符的记录被标记为异常并单独收集\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="集群作业分类工具")
    parser.add_argument("--input", "-i", default="/mnt/raid/liuhongbin/job_analysis/job_analysis/DATA_DC/jobinfo_20250224_113534.csv", 
                       help="输入的jobinfo CSV文件路径")
    parser.add_argument("--output-dir", "-o", default="/mnt/raid/liuhongbin/job_analysis/job_analysis/submit_cluster_classification", 
                       help="输出目录")
    parser.add_argument("--processes", "-p", type=int, default=None,
                       help="并行处理的进程数量，默认为CPU核心数的75%")
    
    args = parser.parse_args()
    
    # 初始化分类器
    classifier = ClusterClassifier()
    
    try:
        # 分类作业
        classified_df = classifier.classify_jobs(args.input, num_processes=args.processes)
        
        # 保存结果
        classifier.save_classified_jobs(classified_df, args.output_dir)
        
        logger.info("分类完成!")
        
    except Exception as e:
        logger.error(f"分类过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 