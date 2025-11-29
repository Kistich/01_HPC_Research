#!/usr/bin/env python3
"""
32核并行处理工具
提供高效的数据并行处理能力
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import gc
import logging
from typing import Callable, List, Any, Optional, Tuple
import time
import math

logger = logging.getLogger(__name__)

class ParallelProcessor:
    """32核并行处理器"""
    
    def __init__(self, max_cores: int = 32, memory_limit_gb: float = 64):
        """
        初始化并行处理器
        
        Args:
            max_cores: 最大核心数
            memory_limit_gb: 内存限制(GB)
        """
        self.max_cores = min(max_cores, cpu_count())
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.current_memory_usage = 0
        
        logger.info(f"并行处理器初始化: {self.max_cores}核心, {memory_limit_gb}GB内存限制")
    
    def get_optimal_chunk_size(self, data_size: int, base_chunk_size: int = 50000) -> int:
        """
        计算最优数据块大小
        
        Args:
            data_size: 数据总大小
            base_chunk_size: 基础块大小
            
        Returns:
            最优块大小
        """
        # 基于核心数和内存限制调整块大小
        memory_per_core = self.memory_limit_bytes / self.max_cores
        estimated_row_size = 1024  # 估计每行1KB
        max_rows_per_core = int(memory_per_core / estimated_row_size)
        
        # 确保每个核心至少有一个块
        min_chunk_size = max(1000, data_size // (self.max_cores * 4))
        max_chunk_size = min(max_rows_per_core, base_chunk_size)
        
        optimal_size = max(min_chunk_size, min(max_chunk_size, base_chunk_size))
        
        logger.debug(f"最优块大小: {optimal_size} (数据大小: {data_size})")
        return optimal_size
    
    def split_dataframe(self, df: pd.DataFrame, chunk_size: Optional[int] = None) -> List[pd.DataFrame]:
        """
        分割DataFrame为多个块
        
        Args:
            df: 输入DataFrame
            chunk_size: 块大小，如果为None则自动计算
            
        Returns:
            DataFrame块列表
        """
        if chunk_size is None:
            chunk_size = self.get_optimal_chunk_size(len(df))
        
        chunks = []
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size].copy()
            chunks.append(chunk)
        
        logger.info(f"数据分割完成: {len(chunks)}个块, 每块约{chunk_size}行")
        return chunks
    
    def process_chunks_with_pool(self, 
                                chunks: List[pd.DataFrame], 
                                process_func: Callable,
                                func_args: Tuple = (),
                                progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        使用进程池处理数据块
        
        Args:
            chunks: 数据块列表
            process_func: 处理函数
            func_args: 函数额外参数
            progress_callback: 进度回调函数
            
        Returns:
            处理结果列表
        """
        results = []
        
        with Pool(self.max_cores) as pool:
            # 准备任务参数
            tasks = [(chunk, *func_args) for chunk in chunks]
            
            # 提交所有任务
            async_results = []
            for task in tasks:
                async_result = pool.apply_async(process_func, task)
                async_results.append(async_result)
            
            # 收集结果并更新进度
            for i, async_result in enumerate(async_results):
                try:
                    result = async_result.get(timeout=300)  # 5分钟超时
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, len(async_results))
                        
                except Exception as e:
                    logger.error(f"处理块{i}时出错: {e}")
                    results.append(None)
        
        # 过滤None结果
        valid_results = [r for r in results if r is not None]
        logger.info(f"并行处理完成: {len(valid_results)}/{len(chunks)}个块成功")
        
        return valid_results
    
    def process_chunks_with_executor(self,
                                   chunks: List[pd.DataFrame],
                                   process_func: Callable,
                                   func_args: Tuple = (),
                                   progress_callback: Optional[Callable] = None) -> List[Any]:
        """
        使用ProcessPoolExecutor处理数据块
        
        Args:
            chunks: 数据块列表
            process_func: 处理函数
            func_args: 函数额外参数
            progress_callback: 进度回调函数
            
        Returns:
            处理结果列表
        """
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_cores) as executor:
            # 提交所有任务
            future_to_index = {}
            for i, chunk in enumerate(chunks):
                future = executor.submit(process_func, chunk, *func_args)
                future_to_index[future] = i
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_index):
                try:
                    result = future.result(timeout=300)
                    index = future_to_index[future]
                    results.append((index, result))
                    
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, len(chunks))
                        
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"处理块{index}时出错: {e}")
                    results.append((index, None))
        
        # 按索引排序并提取结果
        results.sort(key=lambda x: x[0])
        valid_results = [r[1] for r in results if r[1] is not None]
        
        logger.info(f"并行处理完成: {len(valid_results)}/{len(chunks)}个块成功")
        return valid_results
    
    def merge_results(self, results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        合并处理结果
        
        Args:
            results: 结果DataFrame列表
            
        Returns:
            合并后的DataFrame
        """
        if not results:
            logger.warning("没有结果需要合并")
            return pd.DataFrame()
        
        logger.info(f"开始合并{len(results)}个结果...")
        
        # 分批合并以节省内存
        batch_size = 10
        merged_batches = []
        
        for i in range(0, len(results), batch_size):
            batch = results[i:i+batch_size]
            merged_batch = pd.concat(batch, ignore_index=True)
            merged_batches.append(merged_batch)
            
            # 清理内存
            del batch
            gc.collect()
        
        # 最终合并
        final_result = pd.concat(merged_batches, ignore_index=True)
        
        logger.info(f"合并完成: {len(final_result)}行数据")
        return final_result
    
    def monitor_memory_usage(self) -> float:
        """
        监控内存使用情况
        
        Returns:
            当前内存使用率
        """
        memory_info = psutil.virtual_memory()
        usage_ratio = memory_info.used / memory_info.total
        
        if usage_ratio > 0.9:
            logger.warning(f"内存使用率过高: {usage_ratio:.1%}")
            gc.collect()  # 强制垃圾回收
        
        return usage_ratio
    
    def estimate_processing_time(self, 
                               data_size: int, 
                               sample_processing_time: float,
                               sample_size: int = 1000) -> float:
        """
        估算处理时间
        
        Args:
            data_size: 数据总大小
            sample_processing_time: 样本处理时间
            sample_size: 样本大小
            
        Returns:
            估算的总处理时间(秒)
        """
        # 基于样本估算单行处理时间
        time_per_row = sample_processing_time / sample_size
        
        # 考虑并行效率(通常为70-80%)
        parallel_efficiency = 0.75
        
        # 估算总时间
        sequential_time = data_size * time_per_row
        parallel_time = sequential_time / (self.max_cores * parallel_efficiency)
        
        # 加上开销时间(约10%)
        overhead_factor = 1.1
        estimated_time = parallel_time * overhead_factor
        
        logger.info(f"估算处理时间: {estimated_time:.1f}秒 ({estimated_time/60:.1f}分钟)")
        return estimated_time


def parallel_apply(df: pd.DataFrame, 
                  func: Callable, 
                  max_cores: int = 32,
                  chunk_size: Optional[int] = None,
                  progress_callback: Optional[Callable] = None) -> pd.DataFrame:
    """
    并行应用函数到DataFrame
    
    Args:
        df: 输入DataFrame
        func: 处理函数
        max_cores: 最大核心数
        chunk_size: 块大小
        progress_callback: 进度回调
        
    Returns:
        处理后的DataFrame
    """
    processor = ParallelProcessor(max_cores=max_cores)
    
    # 分割数据
    chunks = processor.split_dataframe(df, chunk_size)
    
    # 并行处理
    results = processor.process_chunks_with_pool(
        chunks, func, progress_callback=progress_callback
    )
    
    # 合并结果
    return processor.merge_results(results)
