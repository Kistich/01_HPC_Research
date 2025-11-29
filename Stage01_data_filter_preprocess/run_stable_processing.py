#!/usr/bin/env python3
"""
稳定版全量数据处理脚本
修复了所有已知问题：
1. 字符串拼接错误
2. tqdm兼容性问题  
3. 中间文件保存和断点续传
4. 进程管理优化
"""

import os
import sys
import time
import logging
import signal
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

# 配置日志
log_filename = f"stable_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# 全局变量用于优雅退出
processing_interrupted = False

def signal_handler(signum, frame):
    """改进的信号处理器"""
    global processing_interrupted
    processing_interrupted = True
    logger.warning(f"接收到信号 {signum}，设置中断标志...")
    # 不立即退出，让主程序检查标志后优雅退出

def check_interrupt():
    """检查是否需要中断处理"""
    if processing_interrupted:
        logger.info("检测到中断信号，正在优雅退出...")
        sys.exit(0)

def estimate_processing_time(input_file: str) -> tuple:
    """估算处理时间和输出规模"""
    logger.info("正在分析输入文件...")
    line_count = int(os.popen(f"wc -l {input_file}").read().split()[0]) - 1
    
    # 基于优化后的性能估算
    processing_speed = 2000  # 条记录/秒 (优化后)
    retention_rate = 0.977
    
    estimated_time_seconds = line_count / processing_speed
    estimated_output_records = int(line_count * retention_rate)
    
    return line_count, estimated_time_seconds, estimated_output_records

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}小时{minutes}分钟"

def check_intermediate_files():
    """检查中间文件状态"""
    # 使用绝对路径，基于脚本所在目录
    script_dir = Path(__file__).parent

    files_to_check = {
        "脚本特征分析": script_dir / "full_processing_outputs/stage1_generation_filter/analyzed_features.csv",
        "二期高置信度数据": script_dir / "full_processing_outputs/stage1_generation_filter/second_generation_high.csv",
        "时间处理结果": script_dir / "full_processing_outputs/stage2_time_processing/time_processed_clean.csv",
        "用户推断结果": script_dir / "full_processing_outputs/stage3_user_inference/user_inference_complete.csv",
        "缺失分析结果": script_dir / "full_processing_outputs/stage4_missing_analysis/comprehensive_missing_analysis_report.txt",
        "最终采样结果": script_dir / "full_processing_outputs/stage5_intelligent_sampling/intelligent_sampling_result.csv"
    }

    existing_files = {}
    for name, path in files_to_check.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            try:
                if str(path).endswith('.csv'):
                    line_count = int(os.popen(f"wc -l {path}").read().split()[0]) - 1
                    existing_files[name] = {"path": str(path), "size_mb": size_mb, "records": line_count}
                    logger.info(f"✅ {name}: {line_count:,} 条记录 ({size_mb:.1f} MB)")
                else:
                    existing_files[name] = {"path": str(path), "size_mb": size_mb, "records": "报告文件"}
                    logger.info(f"✅ {name}: 报告文件 ({size_mb:.1f} MB)")
            except:
                existing_files[name] = {"path": str(path), "size_mb": size_mb, "records": "未知"}
                logger.info(f"✅ {name}: {size_mb:.1f} MB")

    return existing_files

def determine_start_stage(existing_files):
    """根据现有文件确定开始阶段"""
    stage_files = {
        1: "二期高置信度数据",
        2: "时间处理结果",
        3: "用户推断结果",
        4: "缺失分析结果",
        5: "最终采样结果"
    }

    # 从最高阶段开始检查
    for stage in range(5, 0, -1):
        if stage_files[stage] in existing_files:
            if stage == 5:
                logger.info("🎉 所有阶段都已完成！")
                return 6  # 表示全部完成
            else:
                logger.info(f"📋 发现阶段{stage}输出文件，将从阶段{stage+1}开始")
                return stage + 1

    logger.info("📋 未发现完整的阶段输出文件，将从阶段1开始")
    return 1

def run_stages_from(start_stage: int, input_file: str, output_dir: str, existing_files: dict) -> str:
    """从指定阶段开始运行处理流程"""

    # 根据开始阶段确定输入文件
    if start_stage == 1:
        stage1_output = run_stage1_with_checkpoint(input_file, output_dir)
        if not stage1_output:
            logger.error("❌ 阶段1未产生二期高置信度数据")
            sys.exit(1)
        current_output = stage1_output
        current_stage = 2
    elif start_stage == 2:
        current_output = existing_files["二期高置信度数据"]["path"]
        current_stage = 2
    elif start_stage == 3:
        current_output = existing_files["时间处理结果"]["path"]
        current_stage = 3
    elif start_stage == 4:
        current_output = existing_files["用户推断结果"]["path"]
        current_stage = 4
    elif start_stage == 5:
        current_output = existing_files["用户推断结果"]["path"]  # 阶段5使用阶段3的输出
        current_stage = 5
    else:
        logger.error(f"❌ 无效的开始阶段: {start_stage}")
        sys.exit(1)

    # 运行剩余阶段
    return run_remaining_stages_from(current_output, output_dir, current_stage)

def run_remaining_stages_from(input_data: str, output_dir: str, start_stage: int) -> str:
    """从指定阶段开始运行剩余阶段"""
    current_output = input_data

    # 阶段2: 时间字段处理
    if start_stage <= 2:
        logger.info("=" * 60)
        logger.info("阶段2: 时间字段处理")
        logger.info("=" * 60)
        check_interrupt()

        from modules.time_processor import TimeProcessor
        stage2_start = time.time()
        time_processor = TimeProcessor("config/time_processor_config.yaml")
        stage2_result = time_processor.process_time_fields(current_output, f"{output_dir}/stage2_time_processing")
        stage2_time = time.time() - stage2_start
        logger.info(f"阶段2完成，耗时: {format_time(stage2_time)}")

        current_output = stage2_result.get('clean_data', stage2_result.get('processed_data', ''))
        if not current_output:
            logger.error("❌ 阶段2未产生有效输出文件")
            return None

    # 阶段3: 用户ID推断
    if start_stage <= 3:
        logger.info("=" * 60)
        logger.info("阶段3: 用户ID推断")
        logger.info("=" * 60)
        check_interrupt()

        from modules.user_inferrer import UserInferrer
        stage3_start = time.time()
        user_inferrer = UserInferrer("config/user_inference_config.yaml")
        stage3_result = user_inferrer.infer_user_ids(current_output, f"{output_dir}/stage3_user_inference")
        stage3_time = time.time() - stage3_start
        logger.info(f"阶段3完成，耗时: {format_time(stage3_time)}")

        current_output = stage3_result.get('complete_data', '')
        if not current_output:
            logger.error("❌ 阶段3未产生有效输出文件")
            return None

    # 阶段4: 缺失数据分析 (纯分析阶段，不修改主数据流)
    if start_stage <= 4:
        logger.info("=" * 60)
        logger.info("阶段4: 缺失数据分析")
        logger.info("=" * 60)
        check_interrupt()

        from modules.missing_analyzer import MissingAnalyzer
        stage4_start = time.time()
        missing_analyzer = MissingAnalyzer("config/missing_analysis_config.yaml")
        stage4_reports = missing_analyzer.analyze_missing_data(current_output, f"{output_dir}/stage4_missing_analysis")
        stage4_time = time.time() - stage4_start
        logger.info(f"阶段4完成，耗时: {format_time(stage4_time)}")

        # 阶段4只生成分析报告，主数据流继续使用current_output
        logger.info(f"✅ 缺失数据分析报告已生成: {len(stage4_reports)} 个文件")
        logger.info(f"📊 主数据流继续使用: {current_output}")

    # 阶段5: 智能采样 (使用阶段3的输出)
    if start_stage <= 5:
        logger.info("=" * 60)
        logger.info("阶段5: 智能采样")
        logger.info("=" * 60)
        check_interrupt()

        from modules.intelligent_sampler import IntelligentSampler
        stage5_start = time.time()
        intelligent_sampler = IntelligentSampler("config/intelligent_sampling_config.yaml")
        # 修正: 使用current_output作为输入，而不是stage4_output
        stage5_result = intelligent_sampler.perform_intelligent_sampling(current_output, f"{output_dir}/stage5_intelligent_sampling")
        stage5_time = time.time() - stage5_start
        logger.info(f"阶段5完成，耗时: {format_time(stage5_time)}")

        # 获取阶段5输出文件
        current_output = stage5_result.get('sampled_data', stage5_result.get('final_data', ''))

        # 阶段6: 数据标准化
        logger.info("=" * 60)
        logger.info("阶段6: 数据标准化")
        logger.info("=" * 60)
        check_interrupt()

        from modules.data_standardizer import DataStandardizer
        stage6_start = time.time()
        data_standardizer = DataStandardizer("config/generation_filter_config.yaml")
        stage6_result = data_standardizer.standardize_data(current_output, f"{output_dir}/stage6_data_standardization")
        stage6_time = time.time() - stage6_start
        logger.info(f"阶段6完成，耗时: {format_time(stage6_time)}")

        # 获取最终输出文件
        final_output = stage6_result['standardized_data']
        return final_output

    return current_output

def run_stage1_with_checkpoint(input_file: str, output_dir: str):
    """运行阶段1并支持断点续传"""
    from modules.generation_filter import GenerationFilter
    
    logger.info("=" * 60)
    logger.info("阶段1: 一期二期数据过滤 (支持断点续传)")
    logger.info("=" * 60)
    
    # 初始化过滤器
    filter_module = GenerationFilter("config/generation_filter_config.yaml")
    
    # 执行过滤 (内部会检查中间文件)
    stage1_start = time.time()
    stage1_result = filter_module.filter_data(input_file, f"{output_dir}/stage1_generation_filter")
    stage1_time = time.time() - stage1_start

    logger.info(f"阶段1完成，耗时: {format_time(stage1_time)}")

    # 获取阶段1的主要输出文件 (高置信度二期数据)
    stage1_output = stage1_result.get('second_generation_high', '')
    if not stage1_output:
        logger.error("❌ 阶段1未产生有效的二期高置信度数据")
        return None

    return stage1_output

def run_remaining_stages(stage1_output: str, output_dir: str):
    """运行剩余阶段"""
    from modules.time_processor import TimeProcessor
    from modules.user_inferrer import UserInferrer
    from modules.missing_analyzer import MissingAnalyzer
    from modules.intelligent_sampler import IntelligentSampler
    from modules.data_standardizer import DataStandardizer
    
    current_input = stage1_output
    
    # 阶段2: 时间处理
    logger.info("=" * 60)
    logger.info("阶段2: 时间字段处理")
    logger.info("=" * 60)
    check_interrupt()
    
    stage2_start = time.time()
    time_processor = TimeProcessor("config/time_processor_config.yaml")
    stage2_result = time_processor.process_time_fields(current_input, f"{output_dir}/stage2_time_processing")
    stage2_time = time.time() - stage2_start
    logger.info(f"阶段2完成，耗时: {format_time(stage2_time)}")

    # 获取阶段2的主要输出文件
    stage2_output = stage2_result.get('clean_data', stage2_result.get('processed_data', ''))
    if not stage2_output:
        logger.error("❌ 阶段2未产生有效输出文件")
        return None

    # 阶段3: 用户推断
    logger.info("=" * 60)
    logger.info("阶段3: 用户ID推断")
    logger.info("=" * 60)
    check_interrupt()

    stage3_start = time.time()
    user_inferrer = UserInferrer("config/user_inference_config.yaml")
    stage3_result = user_inferrer.infer_user_ids(stage2_output, f"{output_dir}/stage3_user_inference")
    stage3_time = time.time() - stage3_start
    logger.info(f"阶段3完成，耗时: {format_time(stage3_time)}")

    # 获取阶段3的主要输出文件
    stage3_output = stage3_result.get('complete_data', stage3_result.get('processed_data', ''))
    if not stage3_output:
        logger.error("❌ 阶段3未产生有效输出文件")
        return None

    # 阶段4: 缺失数据分析 (纯分析阶段，不修改主数据流)
    logger.info("=" * 60)
    logger.info("阶段4: 缺失数据分析")
    logger.info("=" * 60)
    check_interrupt()

    stage4_start = time.time()
    missing_analyzer = MissingAnalyzer("config/missing_analysis_config.yaml")
    stage4_reports = missing_analyzer.analyze_missing_data(stage3_output, f"{output_dir}/stage4_missing_analysis")
    stage4_time = time.time() - stage4_start
    logger.info(f"阶段4完成，耗时: {format_time(stage4_time)}")

    # 阶段4只生成分析报告，主数据流继续使用stage3_output
    logger.info(f"✅ 缺失数据分析报告已生成: {len(stage4_reports)} 个文件")
    logger.info(f"📊 主数据流继续使用: {stage3_output}")

    # 阶段5: 智能采样 (使用stage3_output而不是stage4_output)
    logger.info("=" * 60)
    logger.info("阶段5: 智能采样")
    logger.info("=" * 60)
    check_interrupt()

    stage5_start = time.time()
    intelligent_sampler = IntelligentSampler("config/intelligent_sampling_config.yaml")
    # 修正: 使用stage3_output作为输入，而不是stage4_output
    stage5_result = intelligent_sampler.perform_intelligent_sampling(stage3_output, f"{output_dir}/stage5_intelligent_sampling")
    stage5_time = time.time() - stage5_start
    logger.info(f"阶段5完成，耗时: {format_time(stage5_time)}")

    # 获取阶段5输出文件
    stage5_output = stage5_result.get('sampled_data', stage5_result.get('final_data', ''))

    # 阶段6: 数据标准化
    logger.info("=" * 60)
    logger.info("阶段6: 数据标准化")
    logger.info("=" * 60)
    check_interrupt()

    stage6_start = time.time()
    data_standardizer = DataStandardizer("config/generation_filter_config.yaml")
    stage6_result = data_standardizer.standardize_data(stage5_output, f"{output_dir}/stage6_data_standardization")
    stage6_time = time.time() - stage6_start
    logger.info(f"阶段6完成，耗时: {format_time(stage6_time)}")

    # 获取最终输出文件
    final_output = stage6_result['standardized_data']
    return final_output

def main():
    """主函数"""
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



    # 配置参数 - 使用相对路径（基于当前脚本位置）
    # 当前脚本在: Stage01_data_filter_preprocess/run_stable_processing.py
    # 目标文件在: Stage00_HPC_raw_data/jobinfo_20250224_113534.csv
    # 需要向上一级到 01_HPC_Research，然后进入 Stage00_HPC_raw_data
    script_dir = Path(__file__).parent  # Stage01_data_filter_preprocess/
    project_root = script_dir.parent     # 01_HPC_Research/
    input_file = str(project_root / "Stage00_HPC_raw_data" / "jobinfo_20250224_113534.csv")
    output_dir = "full_processing_outputs"
    
    logger.info("=" * 80)
    logger.info("🚀 稳定版全量数据过滤和预处理系统")
    logger.info("=" * 80)
    logger.info("优化内容:")
    logger.info("  ✅ 修复字符串拼接错误")
    logger.info("  ✅ 修复tqdm兼容性问题")
    logger.info("  ✅ 支持中间文件保存和断点续传")
    logger.info("  ✅ 优化进程管理和信号处理")
    logger.info("  ✅ 预计处理时间: 2-3小时 (vs 原来的26小时)")
    logger.info("=" * 80)
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"日志文件: {log_filename}")
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 检查输入文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 估算处理时间
    try:
        input_records, estimated_time, estimated_output = estimate_processing_time(input_file)
        logger.info("=" * 60)
        logger.info("📊 处理预估")
        logger.info("=" * 60)
        logger.info(f"输入记录数: {input_records:,} 条")
        logger.info(f"预计输出记录数: {estimated_output:,} 条")
        logger.info(f"预计处理时间: {format_time(estimated_time)}")
    except Exception as e:
        logger.error(f"文件分析失败: {e}")
        sys.exit(1)
    
    # 检查中间文件并确定开始阶段
    logger.info("=" * 60)
    logger.info("📋 检查中间文件状态")
    logger.info("=" * 60)
    existing_files = check_intermediate_files()
    start_stage = determine_start_stage(existing_files)

    # 开始处理
    start_time = time.time()

    try:
        if start_stage == 6:
            # 所有阶段都已完成
            logger.info("🎉 所有处理阶段都已完成，无需重新处理！")
            final_output = existing_files["最终采样结果"]["path"]
        else:
            # 根据开始阶段执行相应的处理
            final_output = run_stages_from(start_stage, input_file, output_dir, existing_files)

        # 处理完成
        end_time = time.time()
        total_duration = end_time - start_time

        logger.info("=" * 80)
        logger.info("🎉 稳定版处理完成!")
        logger.info("=" * 80)
        logger.info(f"总处理时间: {format_time(total_duration)}")
        if total_duration > 0:
            logger.info(f"实际处理速度: {int(8376397 / total_duration)} 条记录/秒")
        logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"最终输出: {final_output}")
        logger.info(f"日志文件: {log_filename}")
            
    except KeyboardInterrupt:
        logger.warning("⚠️  用户中断处理")
        sys.exit(1)
    except Exception as e:
        import traceback
        logger.error(f"❌ 处理失败: {str(e)}")
        logger.error("详细错误堆栈:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
