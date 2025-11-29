#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HPC工作负载分析主程序 - Helios兼容版本
基于Helios项目的分析方法，严格按照Helios标准进行HPC集群数据分析
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 解决numpy兼容性问题（与test_cpu_gpu_analysis.py中的处理一致）
try:
    import numpy as np
    # 为了兼容旧版本pickle文件，添加numpy._core的别名
    if not hasattr(np, '_core'):
        import numpy.core as _core
        np._core = _core
        # 确保numpy._core在sys.modules中
        sys.modules['numpy._core'] = _core
        # 添加更多可能需要的子模块
        if hasattr(_core, 'multiarray'):
            sys.modules['numpy._core.multiarray'] = _core.multiarray
        if hasattr(_core, 'umath'):
            sys.modules['numpy._core.umath'] = _core.umath
        if hasattr(_core, 'numeric'):
            sys.modules['numpy._core.numeric'] = _core.numeric
        if hasattr(_core, '_multiarray_umath'):
            sys.modules['numpy._core._multiarray_umath'] = _core._multiarray_umath
except ImportError as e:
    print(f"Numpy导入失败: {e}")
    sys.exit(1)

from modules.data_preprocessing.data_preprocessor import HeliosCompatibleDataPreprocessor
from modules.philly_comparison.philly_comparison_analyzer import PhillyComparisonAnalyzer
from modules.cluster_characterization.cluster_analyzer import ClusterCharacterizationAnalyzer
from modules.job_characterization.job_analyzer import JobCharacterizationAnalyzer
from modules.user_characterization.user_analyzer import UserBehaviorAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('helios_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class HeliosCompatibleAnalyzer:
    """Helios兼容的HPC工作负载分析器"""

    def __init__(self, cluster_config_path: str = "config/cluster_config.yaml"):
        """
        初始化分析器

        Args:
            cluster_config_path: 集群配置文件路径（相对于脚本所在目录）
        """
        logger.info("初始化Helios兼容HPC工作负载分析器...")

        # 获取脚本所在目录
        script_dir = Path(__file__).parent

        # 如果配置路径是相对路径，转换为基于脚本目录的绝对路径
        config_path = Path(cluster_config_path)
        if not config_path.is_absolute():
            config_path = (script_dir / config_path).resolve()

        # 创建输出目录（基于脚本目录）
        self.output_base = script_dir / "output"
        self.output_paths = {
            'philly_comparison': self.output_base / 'philly_comparison',
            'cluster_characterization': self.output_base / 'cluster_characterization',
            'job_characterization': self.output_base / 'job_characterization',
            'user_characterization': self.output_base / 'user_characterization'
        }

        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # 初始化数据预处理器
        self.data_preprocessor = HeliosCompatibleDataPreprocessor(str(config_path))
        
        # 简单的可视化器
        class SimpleVisualizer:
            def _save_figure(self, fig, path):
                fig.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
        
        self.visualizer = SimpleVisualizer()
        
        logger.info("分析器初始化完成")
    
    def load_and_prepare_data(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        加载和准备Helios兼容数据
        
        Args:
            force_reload: 是否强制重新处理数据
            
        Returns:
            预处理后的完整数据集（包含Helios兼容格式）
        """
        logger.info("加载和准备Helios兼容数据...")
        processed_data = self.data_preprocessor.load_and_preprocess_all_data(force_reload)
        logger.info("Helios兼容数据加载完成")
        return processed_data
    
    def run_philly_comparison(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行Philly对比分析（保持4张图结构）"""
        logger.info("运行Philly对比分析...")
        
        config = {'philly_data_path': 'data/external/philly_trace_sample.csv'}
        analyzer = PhillyComparisonAnalyzer(config, self.output_paths, self.visualizer)
        results = analyzer.analyze(processed_data)
        
        logger.info("Philly对比分析完成")
        return results
    
    def run_cluster_characterization(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行集群特征分析（Helios风格）"""
        logger.info("运行集群特征分析（Helios风格）...")
        
        config = {}
        analyzer = ClusterCharacterizationAnalyzer(config, self.output_paths, self.visualizer)
        results = analyzer.analyze(processed_data)
        
        logger.info("集群特征分析完成")
        return results
    
    def run_job_characterization(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行作业特征分析（Helios风格）"""
        logger.info("运行作业特征分析（Helios风格）...")
        
        config = {}
        analyzer = JobCharacterizationAnalyzer(config, self.output_paths, self.visualizer)
        results = analyzer.analyze(processed_data)
        
        logger.info("作业特征分析完成")
        return results
    
    def run_user_characterization(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行用户特征分析（Helios风格）"""
        logger.info("运行用户特征分析（Helios风格）...")
        
        config = {}
        analyzer = UserBehaviorAnalyzer(config, self.output_paths, self.visualizer)
        results = analyzer.analyze(processed_data)
        
        logger.info("用户特征分析完成")
        return results
    
    def run_all_analyses(self, force_reload: bool = False) -> Dict[str, Any]:
        """运行所有Helios兼容分析模块"""
        logger.info("开始完整的Helios兼容工作负载分析...")
        
        # 加载和预处理数据
        processed_data = self.load_and_prepare_data(force_reload)
        
        # 运行各个分析模块
        all_results = {
            'philly_comparison': self.run_philly_comparison(processed_data),
            'cluster_characterization': self.run_cluster_characterization(processed_data),
            'job_characterization': self.run_job_characterization(processed_data),
            'user_characterization': self.run_user_characterization(processed_data)
        }
        
        # 生成综合报告
        self._generate_summary_report(all_results, processed_data)
        
        logger.info("所有Helios兼容分析完成")
        return all_results
    
    def _generate_summary_report(self, results: Dict[str, Any], processed_data: Dict[str, Any]):
        """生成综合分析报告"""
        logger.info("生成Helios兼容分析综合报告...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = self.output_base / 'helios_analysis_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HPC集群工作负载Helios兼容分析报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"分析方法: 严格按照Helios项目标准\n\n")
            
            # Helios数据统计
            if 'helios_data' in processed_data:
                helios_data = processed_data['helios_data']
                f.write("Helios兼容数据统计:\n")
                f.write("-" * 30 + "\n")
                
                if 'cluster_log' in helios_data:
                    cluster_log = helios_data['cluster_log']
                    total_jobs = len(cluster_log)
                    gpu_jobs = len(cluster_log[cluster_log['gpu_num'] > 0])
                    cpu_jobs = total_jobs - gpu_jobs
                    
                    f.write(f"总作业数: {total_jobs:,}\n")
                    f.write(f"GPU作业数: {gpu_jobs:,} ({gpu_jobs/total_jobs*100:.1f}%)\n")
                    f.write(f"CPU作业数: {cpu_jobs:,} ({cpu_jobs/total_jobs*100:.1f}%)\n")
                
                if 'cluster_user' in helios_data:
                    cluster_user = helios_data['cluster_user']
                    f.write(f"总用户数: {len(cluster_user):,}\n")
                
                f.write("\n")
            
            # 分析模块结果摘要
            f.write("分析模块结果:\n")
            f.write("-" * 30 + "\n")
            
            for module_name, module_results in results.items():
                if module_results:
                    f.write(f"✓ {module_name}: 分析完成\n")
                    f.write(f"  结果类别: {list(module_results.keys())}\n")
                else:
                    f.write(f"✗ {module_name}: 分析失败或跳过\n")
            
            f.write("\n")
            f.write("生成的图表文件:\n")
            f.write("-" * 30 + "\n")
            f.write("• Philly比较: job_type_distribution.png, gpu_job_count_status.png, gpu_duration_cdf.png, gpu_time_status.png\n")
            f.write("• 集群特征: cluster_characterization_helios.png\n")
            f.write("• 作业特征: job_characterization_helios.png, job_status_distribution_helios.png\n")
            f.write("• 用户特征: user_resource_cdf_helios.png, user_behavior_patterns_helios.png\n")
            f.write("\n")
            f.write("所有图表均严格按照Helios论文风格生成。\n")
        
        logger.info(f"Helios兼容分析报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HPC工作负载Helios兼容分析工具')
    parser.add_argument('--module', 
                       choices=['philly', 'cluster', 'job', 'user', 'all'],
                       default='all', 
                       help='要运行的分析模块')
    parser.add_argument('--cluster-config', 
                       default='config/cluster_config.yaml',
                       help='集群配置文件路径')
    parser.add_argument('--force-reload', 
                       action='store_true',
                       help='强制重新处理数据')
    
    args = parser.parse_args()
    
    try:
        # 初始化分析器
        analyzer = HeliosCompatibleAnalyzer(args.cluster_config)
        
        if args.module == 'all':
            # 运行所有分析
            results = analyzer.run_all_analyses(args.force_reload)
        else:
            # 运行特定模块
            processed_data = analyzer.load_and_prepare_data(args.force_reload)
            
            if args.module == 'philly':
                results = analyzer.run_philly_comparison(processed_data)
            elif args.module == 'cluster':
                results = analyzer.run_cluster_characterization(processed_data)
            elif args.module == 'job':
                results = analyzer.run_job_characterization(processed_data)
            elif args.module == 'user':
                results = analyzer.run_user_characterization(processed_data)
        
        logger.info("分析完成！请查看output目录中的结果。")
        
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
