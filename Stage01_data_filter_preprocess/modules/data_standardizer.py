#!/usr/bin/env python3
"""
数据标准化模块 - 阶段6
将所有处理后的数据标准化为Trace Analysis兼容格式
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, Any
import yaml

logger = logging.getLogger(__name__)

class DataStandardizer:
    """数据标准化器"""
    
    def __init__(self, config_path: str):
        """初始化数据标准化器"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.stats = {}
    
    def standardize_data(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        标准化数据为Trace Analysis兼容格式
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径字典
        """
        logger.info("开始数据标准化...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info(f"加载数据: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"原始数据: {len(df):,} 条记录")
        
        # 步骤1: 字段标准化和重命名
        logger.info("步骤1: 字段标准化和重命名...")
        df = self._standardize_field_names(df)
        
        # 步骤2: 数据类型统一
        logger.info("步骤2: 数据类型统一...")
        df = self._unify_data_types(df)
        
        # 步骤3: 添加Trace Analysis必需字段
        logger.info("步骤3: 添加Trace Analysis必需字段...")
        df = self._add_required_fields(df)
        
        # 步骤4: 最终数据质量验证
        logger.info("步骤4: 最终数据质量验证...")
        df = self._final_quality_validation(df)
        
        # 步骤5: 输出数据规范化
        logger.info("步骤5: 输出数据规范化...")
        df = self._normalize_output_format(df)
        
        # 保存结果
        output_files = self._save_results(df, output_dir)
        
        # 生成标准化报告
        self._generate_standardization_report(output_dir)
        
        logger.info("数据标准化完成")
        return output_files
    
    def _standardize_field_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化字段名称"""
        logger.info("标准化字段名称...")
        
        df = df.copy()
        
        # 字段重命名映射（使用actual_*字段，它们是基于exec_hosts解析的准确值）
        field_mapping = {
            'final_user_id': 'user_id',           # 使用推断补全后的用户ID
            'actual_cpu_cores': 'cpu_num',        # 使用实际分配的CPU核心数
            'actual_gpu_count': 'gpu_num',        # 使用实际分配的GPU数量
            'actual_node_count': 'node_num',      # 使用实际分配的节点数量
            'helios_status': 'state',             # 标准化状态字段
            'subcluster_type': 'primary_subcluster'  # 子集群类型
        }

        # 执行重命名（优先使用推断后的完美数据）
        for old_name, new_name in field_mapping.items():
            if old_name in df.columns:
                if new_name in df.columns:
                    # 如果目标字段已存在，删除原始字段，保留推断后的字段，然后重命名
                    logger.info(f"字段 {new_name} 已存在，删除原始字段，使用推断后的 {old_name}")
                    df = df.drop(columns=[new_name])  # 删除原始的有缺陷数据
                    df = df.rename(columns={old_name: new_name})  # 重命名推断后的完美数据
                    logger.info(f"字段重命名: {old_name} → {new_name}")
                else:
                    # 正常重命名
                    df = df.rename(columns={old_name: new_name})
                    logger.info(f"字段重命名: {old_name} → {new_name}")
        
        # 确保关键字段存在
        required_fields = ['job_id', 'user_id', 'submit_time', 'start_time', 'end_time', 
                          'duration', 'queue_time', 'exec_hosts', 'state', 'cluster_name']
        
        missing_fields = [field for field in required_fields if field not in df.columns]
        if missing_fields:
            logger.warning(f"缺失关键字段: {missing_fields}")
        
        return df
    
    def _unify_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一数据类型"""
        logger.info("统一数据类型...")
        
        df = df.copy()
        
        # 时间字段转换
        time_fields = ['submit_time', 'start_time', 'end_time']
        for field in time_fields:
            if field in df.columns:
                df[field] = pd.to_datetime(df[field], errors='coerce')
        
        # 数值字段转换
        numeric_fields = ['job_id', 'cpu_num', 'gpu_num', 'node_num', 'duration', 'queue_time']
        for field in numeric_fields:
            if field in df.columns:
                try:
                    df[field] = pd.to_numeric(df[field], errors='coerce')
                except Exception as e:
                    logger.warning(f"无法转换字段 {field} 为数值类型: {e}")

        # 字符串字段转换
        string_fields = ['user_id', 'exec_hosts', 'state', 'cluster_name']
        for field in string_fields:
            if field in df.columns:
                try:
                    df[field] = df[field].astype(str)
                except Exception as e:
                    logger.warning(f"无法转换字段 {field} 为字符串类型: {e}")
        
        return df
    
    def _add_required_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加Trace Analysis必需的字段"""
        logger.info("添加必需字段...")
        
        df = df.copy()
        
        # 添加is_gpu_job字段
        if 'is_gpu_job' not in df.columns:
            if 'gpu_num' in df.columns:
                df['is_gpu_job'] = pd.to_numeric(df['gpu_num'], errors='coerce').fillna(0) > 0
            else:
                df['is_gpu_job'] = df['exec_hosts'].str.contains('gpu', case=False, na=False)
        
        # 添加parsed_hosts字段（简化版）
        if 'parsed_hosts' not in df.columns:
            df['parsed_hosts'] = df['exec_hosts'].apply(self._parse_hosts_simple)
        
        # 确保状态字段有默认值
        if 'state' in df.columns:
            df['state'] = df['state'].fillna('Unknown')
        
        return df
    
    def _parse_hosts_simple(self, exec_hosts_str: str) -> str:
        """简单解析exec_hosts为JSON格式"""
        if pd.isna(exec_hosts_str) or exec_hosts_str == '':
            return '[]'
        
        try:
            # 简单的主机列表解析
            if ' ' in exec_hosts_str and '+' not in exec_hosts_str:
                hosts = exec_hosts_str.split()
            elif '+' in exec_hosts_str:
                hosts = exec_hosts_str.split('+')
            else:
                hosts = [exec_hosts_str]
            
            # 返回JSON格式的主机列表
            import json
            return json.dumps([h.strip() for h in hosts if h.strip()])
        except:
            return '[]'
    
    def _final_quality_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """最终数据质量验证"""
        logger.info("最终数据质量验证...")
        
        df = df.copy()
        original_count = len(df)
        
        # 移除关键字段缺失的记录
        required_for_analysis = ['job_id', 'user_id', 'submit_time']
        for field in required_for_analysis:
            if field in df.columns:
                before_count = len(df)
                df = df[df[field].notna()]
                after_count = len(df)
                if before_count != after_count:
                    logger.info(f"移除{field}缺失记录: {before_count - after_count:,} 条")
        
        # Duration质量验证已在阶段2完成，此处不再重复验证
        # 只进行基本的数据完整性检查
        if 'duration' in df.columns:
            invalid_duration = df[df['duration'] <= 0].shape[0]
            if invalid_duration > 0:
                logger.warning(f"发现 {invalid_duration:,} 条无效duration记录，这些应该在阶段2被移除")
                # 为了数据一致性，仍然移除这些记录
                before_count = len(df)
                df = df[df['duration'] > 0]
                after_count = len(df)
                logger.info(f"移除残留的无效duration记录: {before_count - after_count:,} 条")
        
        # 统计最终质量
        final_count = len(df)
        self.stats['original_count'] = original_count
        self.stats['final_count'] = final_count
        self.stats['quality_retention_rate'] = final_count / original_count if original_count > 0 else 0
        
        logger.info(f"质量验证完成: {original_count:,} → {final_count:,} 条记录 "
                   f"(保留率: {self.stats['quality_retention_rate']*100:.2f}%)")
        
        return df
    
    def _normalize_output_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """规范化输出格式"""
        logger.info("规范化输出格式...")
        
        df = df.copy()
        
        # 确保字段顺序
        preferred_order = [
            'job_id', 'user_id', 'cluster_name', 'submit_time', 'start_time', 'end_time',
            'duration', 'queue_time', 'exec_hosts', 'cpu_num', 'gpu_num', 'node_num',
            'state', 'is_gpu_job', 'primary_subcluster', 'gpu_type'
        ]
        
        # 重新排列列顺序
        existing_preferred = [col for col in preferred_order if col in df.columns]
        other_columns = [col for col in df.columns if col not in preferred_order]
        final_columns = existing_preferred + other_columns
        
        df = df[final_columns]
        
        return df
    
    def _save_results(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """保存标准化结果"""
        output_files = {}
        
        # 保存标准化数据
        standardized_file = os.path.join(output_dir, "standardized_data.csv")
        df.to_csv(standardized_file, index=False)
        output_files['standardized_data'] = standardized_file
        
        logger.info(f"标准化数据已保存: {standardized_file}")
        logger.info(f"最终数据: {len(df):,} 条记录, {len(df.columns)} 个字段")
        
        return output_files
    
    def _generate_standardization_report(self, output_dir: str):
        """生成标准化报告"""
        report_file = os.path.join(output_dir, "standardization_report.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("数据标准化报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"原始记录数: {self.stats.get('original_count', 0):,}\n")
            f.write(f"最终记录数: {self.stats.get('final_count', 0):,}\n")
            f.write(f"数据保留率: {self.stats.get('quality_retention_rate', 0)*100:.2f}%\n\n")
            
            f.write("标准化完成，数据已准备好用于Trace Analysis分析。\n")
        
        logger.info(f"标准化报告已保存: {report_file}")
