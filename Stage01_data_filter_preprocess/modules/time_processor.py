#!/usr/bin/env python3
"""
时间字段处理器
处理时间字段缺失、估计和验证
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from parallel_processor import ParallelProcessor
from progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class TimeProcessor:
    """时间字段处理器"""
    
    def __init__(self, config_path: str):
        """
        初始化时间处理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.processor = ParallelProcessor(
            max_cores=self.config['parallel_processing']['max_cores']
        )
        
        # 统计信息
        self.stats = {
            'total_jobs': 0,
            'submit_time_missing': 0,
            'start_time_missing': 0,
            'end_time_missing': 0,
            'submit_time_estimated': 0,
            'time_logic_errors': 0,
            'jobs_removed': 0,
            'jobs_retained': 0
        }
        
        logger.info("时间字段处理器初始化完成")
    
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
    
    def process_time_fields(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        处理时间字段
        
        Args:
            input_file: 输入文件路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径字典
        """
        logger.info(f"开始时间字段处理: {input_file}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据
        logger.info("加载数据...")
        df = pd.read_csv(input_file)
        self.stats['total_jobs'] = len(df)
        
        logger.info(f"数据加载完成: {len(df):,} 条记录")
        
        # 步骤1: 时间字段完整性检查
        logger.info("步骤1: 时间字段完整性检查...")
        df = self._check_time_completeness(df)
        
        # 步骤2: 时间格式标准化
        logger.info("步骤2: 时间格式标准化...")
        df = self._standardize_time_formats(df)
        
        # 步骤3: submit_time智能估计
        logger.info("步骤3: submit_time智能估计...")
        df = self._estimate_submit_time(df)
        
        # 步骤4: start_time和end_time处理
        logger.info("步骤4: start_time和end_time缺失处理...")
        df = self._process_start_end_time(df)
        
        # 步骤5: 时间逻辑验证
        logger.info("步骤5: 时间逻辑验证...")
        df = self._validate_time_logic(df)
        
        # 步骤6: 计算衍生时间字段
        logger.info("步骤6: 计算衍生时间字段...")
        df = self._calculate_derived_fields(df)

        # 步骤7: 作业状态映射
        logger.info("步骤7: 作业状态映射...")
        df = self._map_job_status(df)

        # 步骤8: Duration质量验证
        logger.info("步骤8: Duration质量验证...")
        df = self._validate_duration_quality(df)

        # 保存结果
        output_files = self._save_results(df, output_dir)

        # 生成处理报告
        self._generate_processing_report(output_dir)

        logger.info("时间字段处理完成")
        return output_files
    
    def _check_time_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        """检查时间字段完整性"""
        df = df.copy()
        
        # 检查各时间字段的缺失情况
        time_fields = self.config['time_processing']['time_fields']['required_fields']
        
        for field in time_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                self.stats[f'{field}_missing'] = missing_count
                
                if missing_count > 0:
                    logger.info(f"{field} 缺失: {missing_count:,} 条 ({missing_count/len(df)*100:.2f}%)")
        
        # 添加缺失标记字段
        df['submit_time_missing'] = df['submit_time'].isna()
        df['start_time_missing'] = df['start_time'].isna()
        df['end_time_missing'] = df['end_time'].isna()
        
        return df
    
    def _standardize_time_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化时间格式"""
        df = df.copy()
        
        time_formats = self.config['time_processing']['time_fields']['time_formats']
        time_fields = ['submit_time', 'start_time', 'end_time']
        
        for field in time_fields:
            if field in df.columns:
                logger.info(f"标准化 {field} 格式...")
                
                # 尝试多种时间格式
                for fmt in time_formats:
                    try:
                        # 只转换非空值
                        mask = df[field].notna()
                        if mask.any():
                            df.loc[mask, field] = pd.to_datetime(df.loc[mask, field], format=fmt, errors='coerce')
                            break
                    except:
                        continue
                
                # 如果所有格式都失败，使用自动推断
                try:
                    mask = df[field].notna()
                    if mask.any():
                        df.loc[mask, field] = pd.to_datetime(df.loc[mask, field], errors='coerce')
                except:
                    logger.warning(f"{field} 时间格式转换失败")
        
        return df
    
    def _estimate_submit_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """智能估计submit_time"""
        df = df.copy()
        
        if 'submit_time' not in df.columns:
            return df
        
        missing_mask = df['submit_time'].isna()
        missing_count = missing_mask.sum()
        
        if missing_count == 0:
            logger.info("submit_time 无缺失，跳过估计")
            return df
        
        logger.info(f"开始估计 {missing_count:,} 个缺失的submit_time...")
        
        # 方法1: 基于用户模式估计
        df = self._estimate_by_user_pattern(df)
        
        # 方法2: 基于作业序列估计
        df = self._estimate_by_job_sequence(df)
        
        # 方法3: 基于批次分析估计
        df = self._estimate_by_batch_analysis(df)
        
        # 统计估计结果
        estimated_count = df['submit_time_estimated'].sum() if 'submit_time_estimated' in df.columns else 0
        self.stats['submit_time_estimated'] = estimated_count
        
        logger.info(f"submit_time 估计完成: {estimated_count:,} 个")
        
        return df
    
    def _estimate_by_user_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于用户模式估计submit_time"""
        df = df.copy()
        
        if 'user_id' not in df.columns:
            return df
        
        # 为每个用户分析提交模式
        user_patterns = {}
        
        for user_id in df['user_id'].unique():
            if pd.isna(user_id):
                continue
            
            user_data = df[df['user_id'] == user_id].copy()
            valid_submits = user_data[user_data['submit_time'].notna()]
            
            if len(valid_submits) >= 10:  # 至少10个有效提交时间
                # 计算提交间隔模式
                submit_times = pd.to_datetime(valid_submits['submit_time']).sort_values()
                intervals = submit_times.diff().dt.total_seconds().dropna()
                
                if len(intervals) > 0:
                    user_patterns[user_id] = {
                        'median_interval': intervals.median(),
                        'mean_interval': intervals.mean(),
                        'std_interval': intervals.std()
                    }
        
        # 基于模式估计缺失的submit_time
        estimated_count = 0
        for user_id, pattern in user_patterns.items():
            user_mask = (df['user_id'] == user_id) & df['submit_time'].isna()
            
            if user_mask.any():
                # 简单估计：使用最近的有效提交时间加上平均间隔
                user_data = df[df['user_id'] == user_id]
                last_valid_submit = user_data[user_data['submit_time'].notna()]['submit_time'].max()
                
                if pd.notna(last_valid_submit):
                    estimated_time = pd.to_datetime(last_valid_submit) + timedelta(seconds=pattern['mean_interval'])
                    df.loc[user_mask, 'submit_time'] = estimated_time
                    estimated_count += user_mask.sum()
        
        if 'submit_time_estimated' not in df.columns:
            df['submit_time_estimated'] = False
        
        df.loc[df['submit_time_missing'] & df['submit_time'].notna(), 'submit_time_estimated'] = True
        
        logger.info(f"基于用户模式估计: {estimated_count} 个")
        return df
    
    def _estimate_by_job_sequence(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于作业序列估计submit_time"""
        df = df.copy()
        
        if 'job_id' not in df.columns:
            return df
        
        # 尝试基于job_id的数值部分进行排序
        try:
            # 提取job_id中的数值部分
            df['job_id_numeric'] = df['job_id'].astype(str).str.extract(r'(\d+)').astype(float)
            
            # 按数值排序
            df_sorted = df.sort_values('job_id_numeric')
            
            # 线性插值估计缺失的submit_time
            valid_mask = df_sorted['submit_time'].notna()
            if valid_mask.sum() >= 2:
                df_sorted['submit_time'] = df_sorted['submit_time'].interpolate(method='time')
                
                # 将结果映射回原DataFrame
                df['submit_time'] = df_sorted['submit_time']
        
        except Exception as e:
            logger.warning(f"基于作业序列估计失败: {e}")
        
        return df
    
    def _estimate_by_batch_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于批次分析估计submit_time"""
        df = df.copy()
        
        # 检测可能的批次提交
        if 'start_time' in df.columns:
            # 使用start_time作为参考 - 向量化操作
            df_with_start = df[df['start_time'].notna()].copy()

            if len(df_with_start) > 0:
                # 假设submit_time通常在start_time之前 - 向量化处理
                missing_submit_mask = df['submit_time'].isna() & df['start_time'].notna()
                if missing_submit_mask.any():
                    # 向量化估计：start_time前5分钟
                    estimated_submit = pd.to_datetime(df.loc[missing_submit_mask, 'start_time']) - timedelta(minutes=5)
                    df.loc[missing_submit_mask, 'submit_time'] = estimated_submit
        
        return df

    def _process_start_end_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理start_time和end_time缺失"""
        df = df.copy()

        # 检查可恢复性
        start_missing = df['start_time'].isna()
        end_missing = df['end_time'].isna()

        logger.info(f"start_time缺失: {start_missing.sum():,} 条")
        logger.info(f"end_time缺失: {end_missing.sum():,} 条")

        # 尝试从duration字段恢复
        if 'duration' in df.columns:
            # 从duration和start_time推算end_time
            mask = end_missing & df['start_time'].notna() & df['duration'].notna()
            if mask.any():
                df.loc[mask, 'end_time'] = pd.to_datetime(df.loc[mask, 'start_time']) + pd.to_timedelta(df.loc[mask, 'duration'], unit='s')
                logger.info(f"从duration恢复end_time: {mask.sum():,} 条")

            # 从duration和end_time推算start_time
            mask = start_missing & df['end_time'].notna() & df['duration'].notna()
            if mask.any():
                df.loc[mask, 'start_time'] = pd.to_datetime(df.loc[mask, 'end_time']) - pd.to_timedelta(df.loc[mask, 'duration'], unit='s')
                logger.info(f"从duration恢复start_time: {mask.sum():,} 条")

        # 标记不可恢复的记录
        unrecoverable = (df['start_time'].isna()) | (df['end_time'].isna())
        df['time_unrecoverable'] = unrecoverable

        return df

    def _validate_time_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证时间逻辑"""
        df = df.copy()

        # 检查时间顺序
        time_order_error = (
            (df['submit_time'] > df['start_time']) |
            (df['start_time'] > df['end_time']) |
            (df['submit_time'] > df['end_time'])
        )

        self.stats['time_logic_errors'] = time_order_error.sum()
        df['time_logic_error'] = time_order_error

        if time_order_error.any():
            logger.warning(f"时间逻辑错误: {time_order_error.sum():,} 条")

        return df

    def _calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生时间字段"""
        df = df.copy()

        # 计算duration
        if 'start_time' in df.columns and 'end_time' in df.columns:
            mask = df['start_time'].notna() & df['end_time'].notna()
            df.loc[mask, 'duration'] = (pd.to_datetime(df.loc[mask, 'end_time']) - pd.to_datetime(df.loc[mask, 'start_time'])).dt.total_seconds()

        # 计算queue_time
        if 'submit_time' in df.columns and 'start_time' in df.columns:
            mask = df['submit_time'].notna() & df['start_time'].notna()
            df.loc[mask, 'queue_time'] = (pd.to_datetime(df.loc[mask, 'start_time']) - pd.to_datetime(df.loc[mask, 'submit_time'])).dt.total_seconds()

        return df

    def _validate_duration_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Duration质量验证 - 移除无效的duration记录"""
        logger.info("开始Duration质量验证...")

        df = df.copy()
        original_count = len(df)

        # 统计原始duration分布
        if 'duration' in df.columns:
            duration_stats = df['duration'].describe()
            logger.info(f"原始duration统计: min={duration_stats['min']:.1f}s, max={duration_stats['max']:.1f}s, mean={duration_stats['mean']:.1f}s")

            # 1. 移除无效duration (≤0)
            invalid_duration = df[df['duration'] <= 0].shape[0]
            if invalid_duration > 0:
                logger.info(f"发现无效duration(≤0): {invalid_duration:,} 条 ({invalid_duration/len(df)*100:.2f}%)")
                df = df[df['duration'] > 0]
                logger.info(f"移除无效duration: {invalid_duration:,} 条")

            # 2. 检查但不移除长duration作业 (HPC环境中长时间作业是正常的)
            # 从配置文件读取最大duration限制
            max_duration_days = self.config.get('time_processing', {}).get('start_end_time_processing', {}).get('logic_validation', {}).get('max_duration_days', 365)  # 默认365天
            max_duration = max_duration_days * 24 * 3600  # 转换为秒

            abnormal_long = df[df['duration'] > max_duration].shape[0]
            if abnormal_long > 0:
                logger.warning(f"发现超长duration(>{max_duration_days}天): {abnormal_long:,} 条 ({abnormal_long/len(df)*100:.2f}%)")
                logger.warning(f"注意: 这些长时间作业在HPC环境中可能是正常的，不会被移除")
                # 不移除这些记录，只记录统计信息
                # df = df[df['duration'] <= max_duration]  # 注释掉移除操作
                logger.info(f"保留所有长时间作业，仅记录统计信息")

            # 3. 统计最终结果
            final_count = len(df)
            removed_count = original_count - final_count
            retention_rate = final_count / original_count if original_count > 0 else 0

            # 更新统计信息
            self.stats['duration_validation'] = {
                'original_count': original_count,
                'invalid_duration_removed': invalid_duration,
                'abnormal_long_detected': abnormal_long,  # 改为detected，因为没有移除
                'abnormal_long_removed': 0,  # 实际移除数为0
                'total_removed': removed_count,
                'final_count': final_count,
                'retention_rate': retention_rate,
                'max_duration_days': max_duration_days  # 记录配置的最大天数
            }

            logger.info(f"Duration质量验证完成: {original_count:,} → {final_count:,} 条记录")
            logger.info(f"总移除: {removed_count:,} 条 (保留率: {retention_rate*100:.2f}%)")

            # 最终duration统计
            if len(df) > 0:
                final_stats = df['duration'].describe()
                logger.info(f"最终duration统计: min={final_stats['min']:.1f}s, max={final_stats['max']:.1f}s, mean={final_stats['mean']:.1f}s")
        else:
            logger.warning("未找到duration字段，跳过质量验证")

        return df

    def _map_job_status(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射作业状态到Helios兼容格式"""
        logger.info("映射作业状态...")

        df = df.copy()

        # 获取状态映射配置
        status_config = self.config.get('job_status_mapping', {})
        helios_map = status_config.get('helios_status_map', {})
        jstatus_map = status_config.get('jstatus_map', {})
        exit_status_rules = status_config.get('exit_status_rules', {})

        # 初始化状态字段
        df['helios_status'] = 'Unknown'
        df['status_confidence'] = 0.0

        # 1. 优先使用job_status_str
        if 'job_status_str' in df.columns:
            mask = df['job_status_str'].notna()
            for original_status, helios_status in helios_map.items():
                status_mask = mask & (df['job_status_str'].str.upper() == original_status)
                df.loc[status_mask, 'helios_status'] = helios_status
                df.loc[status_mask, 'status_confidence'] = 0.9

        # 2. 使用jstatus作为补充
        if 'jstatus' in df.columns:
            unknown_mask = df['helios_status'] == 'Unknown'
            jstatus_notna = df['jstatus'].notna() & unknown_mask

            for jstatus_code, helios_status in jstatus_map.items():
                jstatus_mask = jstatus_notna & (df['jstatus'] == jstatus_code)
                df.loc[jstatus_mask, 'helios_status'] = helios_status
                df.loc[jstatus_mask, 'status_confidence'] = 0.8

        # 3. 使用exit_status作为最后补充
        if 'exit_status' in df.columns:
            unknown_mask = df['helios_status'] == 'Unknown'
            exit_status_notna = df['exit_status'].notna() & unknown_mask

            success_codes = exit_status_rules.get('success_codes', [0])
            failure_codes = exit_status_rules.get('failure_codes', [1, 2, 130, 143])

            # 成功状态
            success_mask = exit_status_notna & df['exit_status'].isin(success_codes)
            df.loc[success_mask, 'helios_status'] = 'Pass'
            df.loc[success_mask, 'status_confidence'] = 0.7

            # 失败状态
            failure_mask = exit_status_notna & df['exit_status'].isin(failure_codes)
            df.loc[failure_mask, 'helios_status'] = 'Failed'
            df.loc[failure_mask, 'status_confidence'] = 0.7

        # 统计状态映射结果
        status_dist = df['helios_status'].value_counts()
        logger.info(f"状态映射完成: {dict(status_dist)}")

        return df

    def _save_results(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """保存处理结果"""
        output_files = {}

        # 分离可用和不可用的数据
        usable_mask = ~df.get('time_unrecoverable', False) & ~df.get('time_logic_error', False)

        usable_df = df[usable_mask].copy()
        unusable_df = df[~usable_mask].copy()

        # 清理临时列
        temp_columns = ['submit_time_missing', 'start_time_missing', 'end_time_missing',
                       'submit_time_estimated', 'time_unrecoverable', 'time_logic_error', 'job_id_numeric']

        for col in temp_columns:
            usable_df = usable_df.drop(columns=[col], errors='ignore')
            unusable_df = unusable_df.drop(columns=[col], errors='ignore')

        # 保存可用数据
        if len(usable_df) > 0:
            usable_file = os.path.join(output_dir, "time_processed_clean.csv")
            usable_df.to_csv(usable_file, index=False)
            output_files['clean_data'] = usable_file
            self.stats['jobs_retained'] = len(usable_df)
            logger.info(f"保存可用数据: {len(usable_df):,} 条 -> {usable_file}")

        # 保存不可用数据
        if len(unusable_df) > 0:
            unusable_file = os.path.join(output_dir, "time_processed_filtered.csv")
            unusable_df.to_csv(unusable_file, index=False)
            output_files['filtered_data'] = unusable_file
            self.stats['jobs_removed'] = len(unusable_df)
            logger.info(f"保存被过滤数据: {len(unusable_df):,} 条 -> {unusable_file}")

        return output_files

    def _generate_processing_report(self, output_dir: str):
        """生成处理报告"""
        report_file = os.path.join(output_dir, "time_processing_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 时间字段处理报告 ===\n\n")
            f.write(f"总作业数: {self.stats['total_jobs']:,}\n\n")

            f.write("缺失情况:\n")
            f.write(f"  submit_time缺失: {self.stats['submit_time_missing']:,} ({self.stats['submit_time_missing']/self.stats['total_jobs']*100:.2f}%)\n")
            f.write(f"  start_time缺失: {self.stats['start_time_missing']:,} ({self.stats['start_time_missing']/self.stats['total_jobs']*100:.2f}%)\n")
            f.write(f"  end_time缺失: {self.stats['end_time_missing']:,} ({self.stats['end_time_missing']/self.stats['total_jobs']*100:.2f}%)\n\n")

            f.write("处理结果:\n")
            f.write(f"  submit_time估计: {self.stats['submit_time_estimated']:,} 个\n")
            f.write(f"  时间逻辑错误: {self.stats['time_logic_errors']:,} 个\n")
            f.write(f"  保留作业: {self.stats['jobs_retained']:,} ({self.stats['jobs_retained']/self.stats['total_jobs']*100:.2f}%)\n")
            f.write(f"  移除作业: {self.stats['jobs_removed']:,} ({self.stats['jobs_removed']/self.stats['total_jobs']*100:.2f}%)\n")

        logger.info(f"处理报告已保存: {report_file}")
