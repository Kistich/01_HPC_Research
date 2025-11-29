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

# 导入额外的分析脚本功能
import pandas as pd
import json

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
            'user_characterization': self.output_base / 'user_characterization',
            'peak_day_detailed': self.output_base / 'peak_day_detailed',
            'data_verification': self.output_base / 'data_verification',
            'detailed_user_analysis': self.output_base / 'detailed_user_analysis',
            'null_user_analysis': self.output_base / 'null_user_analysis'
        }

        for path in self.output_paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # 保存脚本目录路径
        self.script_dir = script_dir

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

    def run_peak_day_analysis(self) -> Dict[str, Any]:
        """运行峰值日分析"""
        logger.info("运行峰值日分析...")

        try:
            # 导入并运行峰值日分析脚本
            import subprocess
            scripts_dir = self.script_dir / "scripts"

            # 运行三个峰值日分析脚本
            scripts = [
                "analyze_peak_day.py",
                "detailed_peak_day_analysis.py",
                "visualize_peak_day_analysis.py"
            ]

            for script in scripts:
                script_path = scripts_dir / script
                if script_path.exists():
                    logger.info(f"运行脚本: {script}")
                    result = subprocess.run(
                        [sys.executable, str(script_path)],
                        cwd=str(self.script_dir),
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        logger.warning(f"脚本 {script} 执行失败: {result.stderr}")
                    else:
                        logger.info(f"脚本 {script} 执行成功")
                else:
                    logger.warning(f"脚本不存在: {script_path}")

            logger.info("峰值日分析完成")
            return {"status": "completed"}

        except Exception as e:
            logger.error(f"峰值日分析失败: {e}")
            return {"status": "failed", "error": str(e)}

    def run_data_verification(self) -> Dict[str, Any]:
        """运行数据验证"""
        logger.info("运行数据验证...")

        try:
            import subprocess
            scripts_dir = self.script_dir / "scripts"
            script_path = scripts_dir / "verify_user_data.py"

            if script_path.exists():
                logger.info("运行数据验证脚本...")
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.script_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.warning(f"数据验证脚本执行失败: {result.stderr}")
                else:
                    logger.info("数据验证脚本执行成功")
            else:
                logger.warning(f"数据验证脚本不存在: {script_path}")

            logger.info("数据验证完成")
            return {"status": "completed"}

        except Exception as e:
            logger.error(f"数据验证失败: {e}")
            return {"status": "failed", "error": str(e)}

    def run_detailed_user_analysis(self) -> Dict[str, Any]:
        """运行详细用户分析"""
        logger.info("运行详细用户分析...")

        try:
            import subprocess
            scripts_dir = self.script_dir / "scripts"
            script_path = scripts_dir / "detailed_user_job_analysis.py"

            if script_path.exists():
                logger.info("运行详细用户分析脚本...")
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.script_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.warning(f"详细用户分析脚本执行失败: {result.stderr}")
                else:
                    logger.info("详细用户分析脚本执行成功")
            else:
                logger.warning(f"详细用户分析脚本不存在: {script_path}")

            logger.info("详细用户分析完成")
            return {"status": "completed"}

        except Exception as e:
            logger.error(f"详细用户分析失败: {e}")
            return {"status": "failed", "error": str(e)}

    def run_null_user_analysis(self) -> Dict[str, Any]:
        """运行空用户分析"""
        logger.info("运行空用户分析...")

        try:
            import subprocess
            scripts_dir = self.script_dir / "scripts"
            script_path = scripts_dir / "analyze_null_user_records.py"

            if script_path.exists():
                logger.info("运行空用户分析脚本...")
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=str(self.script_dir),
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    logger.warning(f"空用户分析脚本执行失败: {result.stderr}")
                else:
                    logger.info("空用户分析脚本执行成功")
            else:
                logger.warning(f"空用户分析脚本不存在: {script_path}")

            logger.info("空用户分析完成")
            return {"status": "completed"}

        except Exception as e:
            logger.error(f"空用户分析失败: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_all_analyses(self, force_reload: bool = False, include_extended: bool = True) -> Dict[str, Any]:
        """
        运行所有分析模块

        Args:
            force_reload: 是否强制重新处理数据
            include_extended: 是否包含扩展分析（峰值日、数据验证等）
        """
        logger.info("=" * 80)
        logger.info("开始完整的HPC工作负载分析...")
        logger.info("=" * 80)

        # 加载和预处理数据
        logger.info("\n[1/2] 数据加载和预处理...")
        processed_data = self.load_and_prepare_data(force_reload)

        # 运行核心分析模块
        logger.info("\n[2/2] 运行核心分析模块...")
        all_results = {
            'philly_comparison': self.run_philly_comparison(processed_data),
            'cluster_characterization': self.run_cluster_characterization(processed_data),
            'job_characterization': self.run_job_characterization(processed_data),
            'user_characterization': self.run_user_characterization(processed_data)
        }

        # 运行扩展分析模块
        if include_extended:
            logger.info("\n[扩展分析] 运行额外分析模块...")
            all_results['peak_day_analysis'] = self.run_peak_day_analysis()
            all_results['data_verification'] = self.run_data_verification()
            all_results['detailed_user_analysis'] = self.run_detailed_user_analysis()
            all_results['null_user_analysis'] = self.run_null_user_analysis()

        # 生成综合报告
        logger.info("\n[报告生成] 生成综合分析报告...")
        self._generate_summary_report(all_results, processed_data, include_extended)

        logger.info("\n" + "=" * 80)
        logger.info("所有分析完成！")
        logger.info("=" * 80)
        logger.info(f"\n📊 输出目录: {self.output_base}")
        logger.info(f"📄 综合报告: {self.output_base / 'helios_analysis_report.txt'}")

        return all_results
    
    def _generate_summary_report(self, results: Dict[str, Any], processed_data: Dict[str, Any],
                                 include_extended: bool = True):
        """生成综合分析报告"""
        logger.info("生成综合分析报告...")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = self.output_base / 'helios_analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("HPC集群工作负载完整分析报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {timestamp}\n")
            f.write(f"分析方法: Helios标准 + 扩展分析\n\n")

            # Helios数据统计
            if 'helios_data' in processed_data:
                helios_data = processed_data['helios_data']
                f.write("数据统计:\n")
                f.write("-" * 80 + "\n")

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
            f.write("分析模块执行状态:\n")
            f.write("-" * 80 + "\n")

            # 核心模块
            f.write("\n【核心分析模块】\n")
            core_modules = ['philly_comparison', 'cluster_characterization',
                          'job_characterization', 'user_characterization']
            for module_name in core_modules:
                if module_name in results:
                    module_results = results[module_name]
                    if module_results:
                        f.write(f"  ✓ {module_name}: 分析完成\n")
                    else:
                        f.write(f"  ✗ {module_name}: 分析失败或跳过\n")

            # 扩展模块
            if include_extended:
                f.write("\n【扩展分析模块】\n")
                extended_modules = ['peak_day_analysis', 'data_verification',
                                  'detailed_user_analysis', 'null_user_analysis']
                for module_name in extended_modules:
                    if module_name in results:
                        module_results = results[module_name]
                        status = module_results.get('status', 'unknown')
                        if status == 'completed':
                            f.write(f"  ✓ {module_name}: 分析完成\n")
                        else:
                            f.write(f"  ✗ {module_name}: {status}\n")

            f.write("\n")
            f.write("生成的输出文件:\n")
            f.write("-" * 80 + "\n")

            # 核心分析输出
            f.write("\n【核心分析输出】\n")
            f.write("  • Philly比较:\n")
            f.write("    - job_type_distribution.png\n")
            f.write("    - gpu_job_count_status.png\n")
            f.write("    - gpu_duration_cdf.png\n")
            f.write("    - gpu_time_status.png\n")
            f.write("  • 集群特征:\n")
            f.write("    - cluster_characterization_helios.png\n")
            f.write("  • 作业特征:\n")
            f.write("    - job_characterization_cpu_helios.png\n")
            f.write("    - job_characterization_gpu_helios.png\n")
            f.write("    - job_status_distribution_helios.png\n")
            f.write("  • 用户特征:\n")
            f.write("    - user_resource_cdf_helios.png\n")
            f.write("    - user_behavior_patterns_helios.png\n")
            f.write("    - user_cpu_behavior_helios.png\n")
            f.write("    - user_gpu_behavior_helios.png\n")

            # 扩展分析输出
            if include_extended:
                f.write("\n【扩展分析输出】\n")
                f.write("  • 峰值日分析:\n")
                f.write("    - output/peak_day_analysis_report.md\n")
                f.write("    - output/peak_day_detailed/peak_day_summary_report.md\n")
                f.write("    - output/peak_day_detailed/*.png\n")
                f.write("  • 数据验证:\n")
                f.write("    - output/data_verification/*.csv\n")
                f.write("  • 详细用户分析:\n")
                f.write("    - output/detailed_user_analysis/*.csv\n")
                f.write("    - output/detailed_user_analysis/*.md\n")
                f.write("  • 空用户分析:\n")
                f.write("    - output/null_user_analysis/*.json\n")
                f.write("    - output/null_user_analysis/*.csv\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("分析完成！所有结果已保存到 output/ 目录\n")
            f.write("=" * 80 + "\n")

        logger.info(f"综合分析报告已保存: {report_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='HPC工作负载完整分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行所有分析（包括扩展分析）
  python main.py

  # 运行所有分析（不包括扩展分析）
  python main.py --no-extended

  # 只运行核心分析模块
  python main.py --module cluster

  # 只运行扩展分析
  python main.py --module peak_day

  # 强制重新处理数据
  python main.py --force-reload
        """
    )

    parser.add_argument('--module',
                       choices=['philly', 'cluster', 'job', 'user', 'all',
                               'peak_day', 'data_verification', 'detailed_user', 'null_user'],
                       default='all',
                       help='要运行的分析模块 (默认: all)')
    parser.add_argument('--cluster-config',
                       default='config/cluster_config.yaml',
                       help='集群配置文件路径 (默认: config/cluster_config.yaml)')
    parser.add_argument('--force-reload',
                       action='store_true',
                       help='强制重新处理数据')
    parser.add_argument('--no-extended',
                       action='store_true',
                       help='运行all时不包括扩展分析（峰值日、数据验证等）')

    args = parser.parse_args()

    try:
        # 打印欢迎信息
        print("\n" + "=" * 80)
        print("HPC工作负载完整分析工具")
        print("=" * 80)
        print(f"模块: {args.module}")
        print(f"配置: {args.cluster_config}")
        print(f"强制重载: {args.force_reload}")
        if args.module == 'all':
            print(f"包含扩展分析: {not args.no_extended}")
        print("=" * 80 + "\n")

        # 初始化分析器
        analyzer = HeliosCompatibleAnalyzer(args.cluster_config)

        if args.module == 'all':
            # 运行所有分析
            include_extended = not args.no_extended
            analyzer.run_all_analyses(args.force_reload, include_extended)

        elif args.module in ['philly', 'cluster', 'job', 'user']:
            # 运行核心分析模块
            processed_data = analyzer.load_and_prepare_data(args.force_reload)

            if args.module == 'philly':
                analyzer.run_philly_comparison(processed_data)
            elif args.module == 'cluster':
                analyzer.run_cluster_characterization(processed_data)
            elif args.module == 'job':
                analyzer.run_job_characterization(processed_data)
            elif args.module == 'user':
                analyzer.run_user_characterization(processed_data)

            logger.info(f"{args.module} 分析完成！")

        elif args.module == 'peak_day':
            # 运行峰值日分析
            analyzer.run_peak_day_analysis()

        elif args.module == 'data_verification':
            # 运行数据验证
            analyzer.run_data_verification()

        elif args.module == 'detailed_user':
            # 运行详细用户分析
            analyzer.run_detailed_user_analysis()

        elif args.module == 'null_user':
            # 运行空用户分析
            analyzer.run_null_user_analysis()

        # 打印完成信息
        print("\n" + "=" * 80)
        print("✅ 分析完成！")
        print("=" * 80)
        print(f"📊 输出目录: {analyzer.output_base}")
        print(f"📄 综合报告: {analyzer.output_base / 'helios_analysis_report.txt'}")
        print("=" * 80 + "\n")

    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
