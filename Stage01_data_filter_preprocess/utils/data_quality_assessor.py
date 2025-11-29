#!/usr/bin/env python3
"""
数据质量评估器
提供数据质量评估和验证功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DataQualityAssessor:
    """数据质量评估器"""
    
    def __init__(self):
        """初始化数据质量评估器"""
        self.quality_metrics = {}
        logger.info("数据质量评估器初始化完成")
    
    def assess_data_quality(self, df: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        评估数据质量
        
        Args:
            df: 待评估的数据框
            config: 评估配置
            
        Returns:
            质量评估结果
        """
        logger.info(f"开始数据质量评估: {len(df):,} 条记录")
        
        quality_report = {
            'basic_stats': self._calculate_basic_stats(df),
            'completeness': self._assess_completeness(df),
            'consistency': self._assess_consistency(df),
            'validity': self._assess_validity(df),
            'uniqueness': self._assess_uniqueness(df),
            'timeliness': self._assess_timeliness(df),
            'overall_score': 0.0,
            'quality_grade': 'Unknown'
        }
        
        # 计算综合质量分数
        quality_report['overall_score'] = self._calculate_overall_score(quality_report)
        quality_report['quality_grade'] = self._get_quality_grade(quality_report['overall_score'])
        
        logger.info(f"数据质量评估完成: 综合分数 {quality_report['overall_score']:.3f}")
        return quality_report
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算基础统计信息"""
        return {
            'total_records': len(df),
            'total_fields': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_fields': len(df.select_dtypes(include=[np.number]).columns),
            'text_fields': len(df.select_dtypes(include=['object']).columns),
            'datetime_fields': len(df.select_dtypes(include=['datetime64']).columns)
        }
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据完整性"""
        completeness = {}
        
        # 整体完整性
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        overall_completeness = (total_cells - missing_cells) / total_cells
        
        completeness['overall_completeness'] = overall_completeness
        completeness['missing_cells'] = missing_cells
        completeness['total_cells'] = total_cells
        
        # 字段级完整性
        field_completeness = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            field_completeness[column] = {
                'completeness_ratio': (len(df) - missing_count) / len(df),
                'missing_count': missing_count,
                'present_count': len(df) - missing_count
            }
        
        completeness['field_completeness'] = field_completeness
        
        # 完整性等级分布
        complete_fields = sum(1 for stats in field_completeness.values() if stats['completeness_ratio'] == 1.0)
        high_complete_fields = sum(1 for stats in field_completeness.values() if 0.9 <= stats['completeness_ratio'] < 1.0)
        medium_complete_fields = sum(1 for stats in field_completeness.values() if 0.5 <= stats['completeness_ratio'] < 0.9)
        low_complete_fields = sum(1 for stats in field_completeness.values() if stats['completeness_ratio'] < 0.5)
        
        completeness['completeness_distribution'] = {
            'complete_fields': complete_fields,
            'high_complete_fields': high_complete_fields,
            'medium_complete_fields': medium_complete_fields,
            'low_complete_fields': low_complete_fields
        }
        
        return completeness
    
    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据一致性"""
        consistency = {}
        
        # 数据类型一致性
        type_consistency = {}
        for column in df.columns:
            if df[column].dtype == 'object':
                # 检查文本字段的格式一致性
                non_null_values = df[column].dropna()
                if len(non_null_values) > 0:
                    # 简单检查：是否都是字符串
                    string_ratio = sum(isinstance(val, str) for val in non_null_values) / len(non_null_values)
                    type_consistency[column] = string_ratio
                else:
                    type_consistency[column] = 1.0
            else:
                type_consistency[column] = 1.0  # 数值类型默认一致
        
        consistency['type_consistency'] = type_consistency
        
        # 时间字段一致性
        time_consistency = self._check_time_consistency(df)
        consistency['time_consistency'] = time_consistency
        
        # 数值范围一致性
        numeric_consistency = self._check_numeric_consistency(df)
        consistency['numeric_consistency'] = numeric_consistency
        
        return consistency
    
    def _check_time_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查时间字段一致性"""
        time_fields = ['submit_time', 'start_time', 'end_time']
        time_consistency = {}
        
        for field in time_fields:
            if field in df.columns:
                try:
                    # 尝试转换为时间格式
                    time_series = pd.to_datetime(df[field], errors='coerce')
                    valid_ratio = time_series.notna().sum() / len(df[field].dropna())
                    time_consistency[field] = valid_ratio
                except:
                    time_consistency[field] = 0.0
        
        # 检查时间逻辑一致性
        if all(field in df.columns for field in time_fields):
            try:
                submit_time = pd.to_datetime(df['submit_time'], errors='coerce')
                start_time = pd.to_datetime(df['start_time'], errors='coerce')
                end_time = pd.to_datetime(df['end_time'], errors='coerce')
                
                valid_mask = submit_time.notna() & start_time.notna() & end_time.notna()
                if valid_mask.any():
                    logical_consistency = (
                        (submit_time <= start_time) & 
                        (start_time <= end_time)
                    )[valid_mask].mean()
                    time_consistency['time_logic_consistency'] = logical_consistency
            except:
                time_consistency['time_logic_consistency'] = 0.0
        
        return time_consistency
    
    def _check_numeric_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数值字段一致性"""
        numeric_consistency = {}
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            if len(series) > 0:
                # 检查是否有异常值
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_ratio = ((series < lower_bound) | (series > upper_bound)).mean()
                consistency_score = 1.0 - min(outlier_ratio, 1.0)
                numeric_consistency[column] = consistency_score
        
        return numeric_consistency
    
    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据有效性"""
        validity = {}
        
        # 检查关键字段的有效性
        key_fields_validity = {}
        
        # 用户ID有效性
        if 'user_id' in df.columns or 'final_user_id' in df.columns:
            user_field = 'final_user_id' if 'final_user_id' in df.columns else 'user_id'
            non_null_users = df[user_field].dropna()
            if len(non_null_users) > 0:
                # 检查用户ID格式的合理性
                valid_user_ratio = sum(len(str(uid)) > 0 for uid in non_null_users) / len(non_null_users)
                key_fields_validity['user_id'] = valid_user_ratio
        
        # 作业ID有效性
        if 'job_id' in df.columns:
            non_null_jobs = df['job_id'].dropna()
            if len(non_null_jobs) > 0:
                valid_job_ratio = sum(len(str(jid)) > 0 for jid in non_null_jobs) / len(non_null_jobs)
                key_fields_validity['job_id'] = valid_job_ratio
        
        # 退出状态有效性
        if 'exit_status' in df.columns:
            non_null_status = df['exit_status'].dropna()
            if len(non_null_status) > 0:
                # 检查退出状态是否为合理的数值
                try:
                    numeric_status = pd.to_numeric(non_null_status, errors='coerce')
                    valid_status_ratio = numeric_status.notna().mean()
                    key_fields_validity['exit_status'] = valid_status_ratio
                except:
                    key_fields_validity['exit_status'] = 0.0
        
        validity['key_fields_validity'] = key_fields_validity
        
        return validity
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据唯一性"""
        uniqueness = {}
        
        # 整体重复率
        duplicate_rows = df.duplicated().sum()
        uniqueness['duplicate_rows'] = duplicate_rows
        uniqueness['uniqueness_ratio'] = (len(df) - duplicate_rows) / len(df)
        
        # 关键字段唯一性
        key_uniqueness = {}
        if 'job_id' in df.columns:
            unique_jobs = df['job_id'].nunique()
            total_jobs = df['job_id'].notna().sum()
            key_uniqueness['job_id'] = unique_jobs / total_jobs if total_jobs > 0 else 0.0
        
        uniqueness['key_uniqueness'] = key_uniqueness
        
        return uniqueness
    
    def _assess_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据时效性"""
        timeliness = {}
        
        # 检查时间范围
        time_fields = ['submit_time', 'start_time', 'end_time']
        time_ranges = {}
        
        for field in time_fields:
            if field in df.columns:
                try:
                    time_series = pd.to_datetime(df[field], errors='coerce').dropna()
                    if len(time_series) > 0:
                        time_ranges[field] = {
                            'min_time': time_series.min(),
                            'max_time': time_series.max(),
                            'time_span_days': (time_series.max() - time_series.min()).days
                        }
                except:
                    pass
        
        timeliness['time_ranges'] = time_ranges
        
        # 数据新鲜度（如果有时间字段）
        if time_ranges:
            latest_time = max(ranges['max_time'] for ranges in time_ranges.values())
            days_old = (datetime.now() - latest_time).days
            freshness_score = max(0, 1 - days_old / 365)  # 一年内的数据认为是新鲜的
            timeliness['freshness_score'] = freshness_score
        
        return timeliness
    
    def _calculate_overall_score(self, quality_report: Dict[str, Any]) -> float:
        """计算综合质量分数"""
        scores = []
        weights = []
        
        # 完整性分数 (权重: 0.3)
        completeness_score = quality_report['completeness']['overall_completeness']
        scores.append(completeness_score)
        weights.append(0.3)
        
        # 一致性分数 (权重: 0.25)
        type_consistency = quality_report['consistency']['type_consistency']
        if type_consistency:
            consistency_score = np.mean(list(type_consistency.values()))
            scores.append(consistency_score)
            weights.append(0.25)
        
        # 有效性分数 (权重: 0.25)
        key_validity = quality_report['validity']['key_fields_validity']
        if key_validity:
            validity_score = np.mean(list(key_validity.values()))
            scores.append(validity_score)
            weights.append(0.25)
        
        # 唯一性分数 (权重: 0.2)
        uniqueness_score = quality_report['uniqueness']['uniqueness_ratio']
        scores.append(uniqueness_score)
        weights.append(0.2)
        
        # 加权平均
        if scores and weights:
            overall_score = np.average(scores, weights=weights)
        else:
            overall_score = 0.0
        
        return overall_score
    
    def _get_quality_grade(self, score: float) -> str:
        """获取质量等级"""
        if score >= 0.9:
            return 'A (优秀)'
        elif score >= 0.8:
            return 'B (良好)'
        elif score >= 0.7:
            return 'C (中等)'
        elif score >= 0.6:
            return 'D (较差)'
        else:
            return 'F (很差)'
    
    def generate_quality_report(self, quality_assessment: Dict[str, Any], output_file: str):
        """生成质量报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 数据质量评估报告 ===\n\n")
            
            # 基础统计
            basic = quality_assessment['basic_stats']
            f.write("1. 基础统计信息\n")
            f.write(f"   总记录数: {basic['total_records']:,}\n")
            f.write(f"   总字段数: {basic['total_fields']}\n")
            f.write(f"   内存使用: {basic['memory_usage_mb']:.2f} MB\n")
            f.write(f"   数值字段: {basic['numeric_fields']} 个\n")
            f.write(f"   文本字段: {basic['text_fields']} 个\n")
            f.write(f"   时间字段: {basic['datetime_fields']} 个\n\n")
            
            # 质量评估
            f.write("2. 质量评估结果\n")
            f.write(f"   综合质量分数: {quality_assessment['overall_score']:.3f}\n")
            f.write(f"   质量等级: {quality_assessment['quality_grade']}\n")
            f.write(f"   数据完整性: {quality_assessment['completeness']['overall_completeness']:.3f}\n")
            f.write(f"   数据唯一性: {quality_assessment['uniqueness']['uniqueness_ratio']:.3f}\n\n")
            
            # 完整性分布
            dist = quality_assessment['completeness']['completeness_distribution']
            f.write("3. 字段完整性分布\n")
            f.write(f"   完全完整字段: {dist['complete_fields']} 个\n")
            f.write(f"   高完整性字段: {dist['high_complete_fields']} 个\n")
            f.write(f"   中等完整性字段: {dist['medium_complete_fields']} 个\n")
            f.write(f"   低完整性字段: {dist['low_complete_fields']} 个\n\n")
            
            # 质量建议
            f.write("4. 质量改进建议\n")
            if quality_assessment['overall_score'] >= 0.8:
                f.write("   ✓ 数据质量良好，可以直接用于分析\n")
            elif quality_assessment['overall_score'] >= 0.6:
                f.write("   ⚠ 数据质量中等，建议进行适当清洗\n")
            else:
                f.write("   ❌ 数据质量较差，需要重点清洗和改进\n")
        
        logger.info(f"质量报告已保存: {output_file}")
