#!/usr/bin/env python3
"""
缺失数据分析器
全面分析数据缺失模式和质量评估
"""

import pandas as pd
import numpy as np
import yaml
import logging
from typing import Dict, List, Tuple, Optional, Any
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 添加utils路径
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from parallel_processor import ParallelProcessor
from progress_tracker import ProgressTracker

logger = logging.getLogger(__name__)

class MissingAnalyzer:
    """缺失数据分析器"""
    
    def __init__(self, config_path: str):
        """
        初始化缺失数据分析器
        
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
            'total_fields': 0,
            'missing_patterns': {},
            'field_quality_scores': {},
            'overall_quality_score': 0.0
        }
        
        logger.info("缺失数据分析器初始化完成")
    
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
    
    def analyze_missing_data(self, input_file: str, output_dir: str) -> Dict[str, str]:
        """
        分析缺失数据 - 纯分析阶段，不修改主数据流

        Args:
            input_file: 输入文件路径
            output_dir: 输出目录

        Returns:
            分析报告文件路径字典
        """
        logger.info(f"开始缺失数据分析: {input_file}")

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 加载数据
        logger.info("加载数据...")
        df = pd.read_csv(input_file, low_memory=False)
        self.stats['total_jobs'] = len(df)
        self.stats['total_fields'] = len(df.columns)

        logger.info(f"数据加载完成: {len(df):,} 条记录, {len(df.columns)} 个字段")

        # 步骤1: 基础缺失统计
        logger.info("步骤1: 基础缺失统计...")
        missing_stats = self._calculate_basic_missing_stats(df)
        logger.info(f"基础缺失统计完成: {len(missing_stats)} 个字段")

        # 步骤2: 缺失模式分析
        logger.info("步骤2: 缺失模式分析...")
        missing_patterns = self._analyze_missing_patterns(df)
        logger.info("缺失模式分析完成")

        # 步骤3: 字段重要性评估
        logger.info("步骤3: 字段重要性评估...")
        field_importance = self._assess_field_importance(df)
        logger.info("字段重要性评估完成")

        # 步骤4: 数据质量评分
        logger.info("步骤4: 数据质量评分...")
        quality_scores = self._calculate_quality_scores(df, missing_stats, field_importance)
        overall_quality = quality_scores.get('overall_score', 0.0)
        logger.info(f"数据质量评分完成: 整体质量分数 {overall_quality:.3f}")

        # 步骤5: 缺失数据可视化
        logger.info("步骤5: 生成可视化...")
        self._generate_visualizations(df, missing_stats, output_dir)
        logger.info("可视化图表生成完成")

        # 保存分析报告 (仅保存分析报告，不保存修改后的数据)
        report_files = self._save_analysis_reports(
            missing_stats, missing_patterns, field_importance, quality_scores, output_dir
        )
        logger.info(f"分析报告保存完成: {len(report_files)} 个文件")

        # 生成综合报告
        comprehensive_report = self._generate_comprehensive_report(
            missing_stats, missing_patterns, field_importance, quality_scores, output_dir
        )
        report_files['comprehensive_report'] = comprehensive_report
        logger.info(f"综合报告已保存: {comprehensive_report}")

        logger.info("缺失数据分析完成")
        return report_files
    
    def _calculate_basic_missing_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算基础缺失统计"""
        missing_stats = {}
        
        for column in df.columns:
            missing_count = df[column].isna().sum()
            missing_ratio = missing_count / len(df)
            
            missing_stats[column] = {
                'missing_count': missing_count,
                'missing_ratio': missing_ratio,
                'present_count': len(df) - missing_count,
                'present_ratio': 1 - missing_ratio,
                'data_type': str(df[column].dtype),
                'unique_values': df[column].nunique() if missing_count < len(df) else 0
            }
        
        # 排序：按缺失比例降序
        missing_stats = dict(sorted(missing_stats.items(), key=lambda x: x[1]['missing_ratio'], reverse=True))
        
        logger.info(f"基础缺失统计完成: {len(missing_stats)} 个字段")
        return missing_stats
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """分析缺失模式"""
        patterns = {}
        
        # 1. 完全缺失的字段
        completely_missing = []
        for column in df.columns:
            if df[column].isna().all():
                completely_missing.append(column)
        patterns['completely_missing'] = completely_missing
        
        # 2. 完全存在的字段
        completely_present = []
        for column in df.columns:
            if df[column].notna().all():
                completely_present.append(column)
        patterns['completely_present'] = completely_present
        
        # 3. 高缺失字段 (>50%)
        high_missing = []
        for column in df.columns:
            missing_ratio = df[column].isna().sum() / len(df)
            if 0.5 < missing_ratio < 1.0:
                high_missing.append((column, missing_ratio))
        patterns['high_missing'] = sorted(high_missing, key=lambda x: x[1], reverse=True)
        
        # 4. 中等缺失字段 (10%-50%)
        medium_missing = []
        for column in df.columns:
            missing_ratio = df[column].isna().sum() / len(df)
            if 0.1 <= missing_ratio <= 0.5:
                medium_missing.append((column, missing_ratio))
        patterns['medium_missing'] = sorted(medium_missing, key=lambda x: x[1], reverse=True)
        
        # 5. 低缺失字段 (<10%)
        low_missing = []
        for column in df.columns:
            missing_ratio = df[column].isna().sum() / len(df)
            if 0 < missing_ratio < 0.1:
                low_missing.append((column, missing_ratio))
        patterns['low_missing'] = sorted(low_missing, key=lambda x: x[1], reverse=True)
        
        # 6. 缺失组合模式
        patterns['combination_patterns'] = self._analyze_combination_patterns(df)
        
        self.stats['missing_patterns'] = patterns
        logger.info("缺失模式分析完成")
        return patterns
    
    def _analyze_combination_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """分析缺失组合模式"""
        # 简化实现：分析常见的缺失组合
        combination_patterns = []
        
        # 检查时间字段的组合缺失
        time_fields = ['submit_time', 'start_time', 'end_time']
        existing_time_fields = [field for field in time_fields if field in df.columns]
        
        if len(existing_time_fields) > 1:
            for i, field1 in enumerate(existing_time_fields):
                for field2 in existing_time_fields[i+1:]:
                    both_missing = (df[field1].isna() & df[field2].isna()).sum()
                    if both_missing > 0:
                        combination_patterns.append({
                            'fields': [field1, field2],
                            'both_missing_count': both_missing,
                            'both_missing_ratio': both_missing / len(df)
                        })
        
        # 检查资源字段的组合缺失
        resource_fields = ['cpu_num', 'mem_req', 'gpu_num']
        existing_resource_fields = [field for field in resource_fields if field in df.columns]
        
        if len(existing_resource_fields) > 1:
            all_resource_missing = df[existing_resource_fields].isna().all(axis=1).sum()
            if all_resource_missing > 0:
                combination_patterns.append({
                    'fields': existing_resource_fields,
                    'all_missing_count': all_resource_missing,
                    'all_missing_ratio': all_resource_missing / len(df)
                })
        
        return combination_patterns
    
    def _assess_field_importance(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """评估字段重要性"""
        field_importance_config = self.config['missing_analysis']['field_importance']
        field_importance = {}

        # 构建字段到类别和权重的映射
        field_mapping = {}
        for category in ['core_fields', 'important_fields', 'auxiliary_fields', 'redundant_fields']:
            if category in field_importance_config:
                for field_info in field_importance_config[category]:
                    field_name = field_info['field']
                    field_mapping[field_name] = {
                        'category': category,
                        'weight': field_info['weight'],
                        'description': field_info.get('description', '')
                    }

        for column in df.columns:
            # 确定字段类别和权重
            if column in field_mapping:
                category = field_mapping[column]['category']
                weight = field_mapping[column]['weight']
                description = field_mapping[column]['description']
            else:
                category = 'auxiliary_fields'  # 默认类别
                weight = 1.0
                description = 'Unknown field'
            
            # 计算重要性分数
            base_score = weight / 5.0  # 标准化到0-1范围

            # 根据数据完整性调整分数
            missing_ratio = df[column].isna().sum() / len(df)
            completeness_bonus = (1 - missing_ratio) * 0.3

            # 根据数据唯一性调整分数
            if missing_ratio < 1.0:
                uniqueness_ratio = df[column].nunique() / (len(df) - df[column].isna().sum())
                uniqueness_bonus = min(uniqueness_ratio, 1.0) * 0.2
            else:
                uniqueness_bonus = 0.0

            final_score = min(1.0, base_score + completeness_bonus + uniqueness_bonus)

            field_importance[column] = {
                'category': category,
                'weight': weight,
                'description': description,
                'base_score': base_score,
                'completeness_bonus': completeness_bonus,
                'uniqueness_bonus': uniqueness_bonus,
                'final_score': final_score,
                'missing_ratio': missing_ratio
            }
        
        logger.info("字段重要性评估完成")
        return field_importance
    
    def _calculate_quality_scores(self, df: pd.DataFrame, missing_stats: Dict[str, Any], 
                                field_importance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """计算数据质量评分"""
        quality_scores = {}
        
        # 字段级质量评分
        for column in df.columns:
            missing_ratio = missing_stats[column]['missing_ratio']
            importance_score = field_importance[column]['final_score']
            
            # 质量分数 = (1 - 缺失比例) * 重要性权重
            field_quality = (1 - missing_ratio) * importance_score
            quality_scores[column] = field_quality
        
        # 整体质量评分
        total_weighted_quality = sum(quality_scores.values())
        total_weights = sum(field_importance[col]['final_score'] for col in df.columns)
        overall_quality = total_weighted_quality / total_weights if total_weights > 0 else 0.0
        
        self.stats['field_quality_scores'] = quality_scores
        self.stats['overall_quality_score'] = overall_quality
        
        logger.info(f"数据质量评分完成: 整体质量分数 {overall_quality:.3f}")
        return {
            'field_scores': quality_scores,
            'overall_score': overall_quality,
            'quality_grade': self._get_quality_grade(overall_quality)
        }
    
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

    def _generate_visualizations(self, df: pd.DataFrame, missing_stats: Dict[str, Any], output_dir: str):
        """生成可视化图表"""
        try:
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            # 1. 缺失比例条形图
            fig, ax = plt.subplots(figsize=(12, 8))

            fields = list(missing_stats.keys())[:20]  # 只显示前20个字段
            missing_ratios = [missing_stats[field]['missing_ratio'] for field in fields]

            bars = ax.barh(fields, missing_ratios)
            ax.set_xlabel('缺失比例')
            ax.set_title('字段缺失比例分析')
            ax.set_xlim(0, 1)

            # 添加数值标签
            for i, (bar, ratio) in enumerate(zip(bars, missing_ratios)):
                ax.text(ratio + 0.01, i, f'{ratio:.2%}', va='center')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'missing_ratio_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 缺失模式热图
            if len(df.columns) <= 50:  # 只有字段数不太多时才生成热图
                fig, ax = plt.subplots(figsize=(15, 10))

                # 创建缺失模式矩阵
                missing_matrix = df.isnull().astype(int)

                # 随机采样1000行用于可视化
                if len(missing_matrix) > 1000:
                    sample_indices = np.random.choice(len(missing_matrix), 1000, replace=False)
                    missing_matrix = missing_matrix.iloc[sample_indices]

                sns.heatmap(missing_matrix.T, cmap='RdYlBu_r', cbar_kws={'label': '缺失 (1) / 存在 (0)'})
                ax.set_title('缺失模式热图 (随机采样1000条记录)')
                ax.set_xlabel('记录索引')
                ax.set_ylabel('字段')

                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'missing_pattern_heatmap.png'), dpi=300, bbox_inches='tight')
                plt.close()

            logger.info("可视化图表生成完成")

        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")

    def _save_analysis_reports(self, missing_stats: Dict[str, Any],
                             missing_patterns: Dict[str, Any], field_importance: Dict[str, Dict[str, Any]],
                             quality_scores: Dict[str, Any], output_dir: str) -> Dict[str, str]:
        """保存分析报告 (不保存修改后的数据)"""
        report_files = {}

        # 1. 保存详细缺失统计
        missing_stats_df = pd.DataFrame.from_dict(missing_stats, orient='index')
        missing_stats_file = os.path.join(output_dir, "detailed_missing_statistics.csv")
        missing_stats_df.to_csv(missing_stats_file)
        report_files['missing_statistics'] = missing_stats_file

        # 2. 保存字段重要性评估
        importance_df = pd.DataFrame.from_dict(field_importance, orient='index')
        importance_file = os.path.join(output_dir, "field_importance_assessment.csv")
        importance_df.to_csv(importance_file)
        report_files['field_importance'] = importance_file

        # 3. 保存质量评分
        if 'field_scores' in quality_scores:
            quality_df = pd.DataFrame.from_dict(quality_scores['field_scores'], orient='index', columns=['quality_score'])
            quality_file = os.path.join(output_dir, "data_quality_scores.csv")
            quality_df.to_csv(quality_file)
            report_files['quality_scores'] = quality_file

        # 注意: 不再保存 missing_analysis_complete.csv
        # 主数据流应该继续使用 user_inference_complete.csv

        logger.info(f"分析报告保存完成: {len(report_files)} 个文件")
        return report_files

    def _generate_comprehensive_report(self, missing_stats: Dict[str, Any], missing_patterns: Dict[str, Any],
                                     field_importance: Dict[str, Dict[str, Any]], quality_scores: Dict[str, Any],
                                     output_dir: str) -> str:
        """生成综合报告"""
        report_file = os.path.join(output_dir, "comprehensive_missing_analysis_report.txt")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=== 缺失数据综合分析报告 ===\n\n")

            # 基础统计
            f.write("1. 基础统计信息\n")
            f.write(f"   总记录数: {self.stats['total_jobs']:,}\n")
            f.write(f"   总字段数: {self.stats['total_fields']}\n")
            f.write(f"   整体质量分数: {quality_scores['overall_score']:.3f}\n")
            f.write(f"   质量等级: {quality_scores['quality_grade']}\n\n")

            # 缺失模式统计
            f.write("2. 缺失模式分析\n")
            f.write(f"   完全缺失字段: {len(missing_patterns['completely_missing'])} 个\n")
            f.write(f"   完全存在字段: {len(missing_patterns['completely_present'])} 个\n")
            f.write(f"   高缺失字段 (>50%): {len(missing_patterns['high_missing'])} 个\n")
            f.write(f"   中等缺失字段 (10%-50%): {len(missing_patterns['medium_missing'])} 个\n")
            f.write(f"   低缺失字段 (<10%): {len(missing_patterns['low_missing'])} 个\n\n")

            # 字段重要性分布
            f.write("3. 字段重要性分布\n")
            category_counts = defaultdict(int)
            for field_info in field_importance.values():
                category_counts[field_info['category']] += 1

            for category, count in category_counts.items():
                f.write(f"   {category}: {count} 个字段\n")
            f.write("\n")

            # 高缺失重要字段警告
            f.write("4. 重要字段缺失警告\n")
            important_missing = []
            for field, stats in missing_stats.items():
                if (field_importance[field]['category'] in ['core', 'important'] and
                    stats['missing_ratio'] > 0.1):
                    important_missing.append((field, stats['missing_ratio'], field_importance[field]['category']))

            if important_missing:
                f.write("   以下重要字段存在显著缺失:\n")
                for field, ratio, category in sorted(important_missing, key=lambda x: x[1], reverse=True):
                    f.write(f"     {field} ({category}): {ratio:.2%} 缺失\n")
            else:
                f.write("   ✓ 所有重要字段缺失率都在可接受范围内\n")
            f.write("\n")

            # 数据质量建议
            f.write("5. 数据质量改进建议\n")
            if quality_scores['overall_score'] >= 0.8:
                f.write("   ✓ 数据质量良好，可以直接用于分析\n")
            elif quality_scores['overall_score'] >= 0.6:
                f.write("   ⚠ 数据质量中等，建议:\n")
                f.write("     - 重点关注高缺失的重要字段\n")
                f.write("     - 考虑数据补全或替代方案\n")
            else:
                f.write("   ❌ 数据质量较差，强烈建议:\n")
                f.write("     - 数据清洗和补全\n")
                f.write("     - 重新评估数据收集流程\n")
                f.write("     - 考虑使用更完整的数据源\n")

        logger.info(f"综合报告已保存: {report_file}")
        return report_file
