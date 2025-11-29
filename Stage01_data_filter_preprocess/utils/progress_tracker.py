#!/usr/bin/env python3
"""
进度条管理器
提供详细的进度跟踪和性能监控
"""

from tqdm import tqdm
import time
import psutil
import threading
from typing import Optional, Callable, Dict, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, 
                 total: int,
                 description: str = "Processing",
                 unit: str = "items",
                 show_eta: bool = True,
                 show_speed: bool = True,
                 show_memory: bool = True,
                 update_interval: float = 1.0):
        """
        初始化进度跟踪器
        
        Args:
            total: 总数量
            description: 描述信息
            unit: 单位
            show_eta: 显示预估完成时间
            show_speed: 显示处理速度
            show_memory: 显示内存使用
            update_interval: 更新间隔(秒)
        """
        self.total = total
        self.description = description
        self.unit = unit
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.show_memory = show_memory
        self.update_interval = update_interval
        
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processed_count = 0
        self.speed_history = []
        self.memory_history = []
        
        # 创建进度条 - 确保total是整数
        total_int = int(total) if total is not None else None
        self.pbar = tqdm(
            total=total_int,
            desc=description,
            unit=unit,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )
        
        # 启动监控线程
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info(f"进度跟踪器启动: {description} (总计: {total} {unit})")
    
    def update(self, n: int = 1, **kwargs):
        """
        更新进度
        
        Args:
            n: 增加的数量
            **kwargs: 额外的状态信息
        """
        self.processed_count += n
        current_time = time.time()
        
        # 计算速度
        if current_time - self.last_update_time >= self.update_interval:
            elapsed_time = current_time - self.last_update_time
            speed = n / elapsed_time if elapsed_time > 0 else 0
            self.speed_history.append(speed)
            
            # 保持速度历史在合理范围内
            if len(self.speed_history) > 60:  # 保留最近60个数据点
                self.speed_history.pop(0)
            
            self.last_update_time = current_time
        
        # 构建状态信息
        postfix = self._build_postfix(**kwargs)
        
        # 更新进度条
        self.pbar.update(n)
        if postfix:
            self.pbar.set_postfix_str(postfix)
    
    def _build_postfix(self, **kwargs) -> str:
        """构建状态信息字符串"""
        postfix_parts = []
        
        # 添加速度信息
        if self.show_speed and self.speed_history:
            avg_speed = sum(self.speed_history) / len(self.speed_history)
            postfix_parts.append(f"速度: {avg_speed:.1f} {self.unit}/s")
        
        # 添加内存信息
        if self.show_memory:
            memory_usage = psutil.virtual_memory().percent
            postfix_parts.append(f"内存: {memory_usage:.1f}%")
        
        # 添加自定义信息
        for key, value in kwargs.items():
            if isinstance(value, float):
                postfix_parts.append(f"{key}: {value:.2f}")
            else:
                postfix_parts.append(f"{key}: {value}")
        
        return " | ".join(postfix_parts)
    
    def _monitor_performance(self):
        """性能监控线程"""
        while self.monitoring:
            try:
                # 记录内存使用
                memory_usage = psutil.virtual_memory().percent
                self.memory_history.append(memory_usage)
                
                # 保持内存历史在合理范围内
                if len(self.memory_history) > 300:  # 保留最近5分钟的数据
                    self.memory_history.pop(0)
                
                # 检查内存使用是否过高
                if memory_usage > 90:
                    logger.warning(f"内存使用率过高: {memory_usage:.1f}%")
                
                time.sleep(1.0)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"性能监控出错: {e}")
                break
    
    def set_description(self, description: str):
        """设置描述信息"""
        self.description = description
        self.pbar.set_description(description)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        stats = {
            'total': self.total,
            'processed': self.processed_count,
            'remaining': self.total - self.processed_count,
            'progress_percent': (self.processed_count / self.total * 100) if self.total > 0 else 0,
            'elapsed_time': elapsed_time,
            'start_time': datetime.fromtimestamp(self.start_time),
            'current_time': datetime.fromtimestamp(current_time)
        }
        
        # 计算速度统计
        if self.speed_history:
            stats['avg_speed'] = sum(self.speed_history) / len(self.speed_history)
            stats['max_speed'] = max(self.speed_history)
            stats['min_speed'] = min(self.speed_history)
            stats['current_speed'] = self.speed_history[-1] if self.speed_history else 0
            
            # 估算剩余时间
            if stats['avg_speed'] > 0:
                remaining_time = stats['remaining'] / stats['avg_speed']
                stats['eta'] = datetime.fromtimestamp(current_time + remaining_time)
                stats['remaining_time'] = timedelta(seconds=remaining_time)
        
        # 内存统计
        if self.memory_history:
            stats['avg_memory'] = sum(self.memory_history) / len(self.memory_history)
            stats['max_memory'] = max(self.memory_history)
            stats['current_memory'] = self.memory_history[-1] if self.memory_history else 0
        
        return stats
    
    def close(self):
        """关闭进度跟踪器"""
        self.monitoring = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        self.pbar.close()
        
        # 输出最终统计
        stats = self.get_statistics()
        logger.info(f"进度跟踪完成: {self.description}")
        logger.info(f"  处理数量: {stats['processed']}/{stats['total']} ({stats['progress_percent']:.1f}%)")
        logger.info(f"  总耗时: {stats['elapsed_time']:.1f}秒")
        if 'avg_speed' in stats:
            logger.info(f"  平均速度: {stats['avg_speed']:.1f} {self.unit}/s")
        if 'avg_memory' in stats:
            logger.info(f"  平均内存: {stats['avg_memory']:.1f}%")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MultiStageProgressTracker:
    """多阶段进度跟踪器"""
    
    def __init__(self, stages: Dict[str, int], overall_description: str = "Multi-stage Processing"):
        """
        初始化多阶段进度跟踪器
        
        Args:
            stages: 阶段字典 {阶段名: 总数量}
            overall_description: 总体描述
        """
        self.stages = stages
        self.overall_description = overall_description
        self.current_stage = None
        self.current_tracker = None
        self.completed_stages = set()
        
        # 计算总体进度
        self.total_items = sum(stages.values())
        self.completed_items = 0
        
        # 创建总体进度条
        self.overall_pbar = tqdm(
            total=self.total_items,
            desc=overall_description,
            unit="items",
            position=0,
            ncols=120
        )
        
        logger.info(f"多阶段进度跟踪器启动: {len(stages)}个阶段, 总计{self.total_items}项")
    
    def start_stage(self, stage_name: str, description: Optional[str] = None) -> ProgressTracker:
        """
        开始新阶段
        
        Args:
            stage_name: 阶段名称
            description: 阶段描述
            
        Returns:
            阶段进度跟踪器
        """
        if stage_name not in self.stages:
            raise ValueError(f"未知阶段: {stage_name}")
        
        if self.current_tracker:
            self.current_tracker.close()
        
        self.current_stage = stage_name
        stage_total = self.stages[stage_name]
        stage_desc = description or f"阶段: {stage_name}"
        
        self.current_tracker = ProgressTracker(
            total=stage_total,
            description=stage_desc,
            unit="items",
            show_eta=True,
            show_speed=True,
            show_memory=True
        )
        
        logger.info(f"开始阶段: {stage_name} ({stage_total}项)")
        return self.current_tracker
    
    def update_stage(self, n: int = 1, **kwargs):
        """更新当前阶段进度"""
        if self.current_tracker:
            self.current_tracker.update(n, **kwargs)
            self.completed_items += n
            self.overall_pbar.update(n)
    
    def complete_stage(self):
        """完成当前阶段"""
        if self.current_stage:
            self.completed_stages.add(self.current_stage)
            logger.info(f"阶段完成: {self.current_stage}")
            
            if self.current_tracker:
                self.current_tracker.close()
                self.current_tracker = None
            
            self.current_stage = None
    
    def get_overall_progress(self) -> Dict[str, Any]:
        """获取总体进度"""
        return {
            'total_stages': len(self.stages),
            'completed_stages': len(self.completed_stages),
            'current_stage': self.current_stage,
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'overall_progress': (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
        }
    
    def close(self):
        """关闭多阶段进度跟踪器"""
        if self.current_tracker:
            self.current_tracker.close()
        
        self.overall_pbar.close()
        
        progress = self.get_overall_progress()
        logger.info(f"多阶段处理完成: {progress['completed_stages']}/{progress['total_stages']}阶段")
        logger.info(f"总体进度: {progress['completed_items']}/{progress['total_items']} ({progress['overall_progress']:.1f}%)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
