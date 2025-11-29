#!/usr/bin/env python3
"""
只运行智能采样阶段的脚本
用于重新生成采样前后对比图表
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

# 配置日志
log_filename = f"intelligent_sampling_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

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

def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("🚀 智能采样阶段重新运行")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 设置输入输出路径
    input_file = "full_processing_outputs/stage3_user_inference/user_inference_complete.csv"
    output_dir = "full_processing_outputs/stage5_intelligent_sampling"
    
    # 检查输入文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取输入文件信息
    try:
        line_count = int(os.popen(f"wc -l {input_file}").read().split()[0]) - 1
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        logger.info(f"输入文件: {input_file}")
        logger.info(f"记录数: {line_count:,} 条")
        logger.info(f"文件大小: {file_size_mb:.1f} MB")
    except Exception as e:
        logger.error(f"文件分析失败: {e}")
        sys.exit(1)
    
    # 开始智能采样
    start_time = time.time()
    
    try:
        logger.info("=" * 60)
        logger.info("阶段5: 智能采样")
        logger.info("=" * 60)
        
        from modules.intelligent_sampler import IntelligentSampler
        
        intelligent_sampler = IntelligentSampler("config/intelligent_sampling_config.yaml")
        stage5_result = intelligent_sampler.perform_intelligent_sampling(input_file, output_dir)
        
        # 处理完成
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("🎉 智能采样完成!")
        logger.info("=" * 80)
        logger.info(f"处理时间: {format_time(total_duration)}")
        if total_duration > 0:
            logger.info(f"处理速度: {int(line_count / total_duration)} 条记录/秒")
        logger.info(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示输出文件
        for key, file_path in stage5_result.items():
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                logger.info(f"输出文件 ({key}): {file_path} ({size_mb:.1f} MB)")
        
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
