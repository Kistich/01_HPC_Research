#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化工具模块
严格按照Helios项目的可视化风格和图表类型
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class HeliosStyleVisualizer:
    """Helios风格可视化器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置
        """
        self.config = config
        self.setup_style()
        
    def setup_style(self):
        """设置Helios风格的图表样式"""
        # 设置matplotlib参数
        plt.rcParams['figure.figsize'] = (
            self.config['figure_size']['width'],
            self.config['figure_size']['height']
        )
        plt.rcParams['font.family'] = self.config['fonts']['family']
        plt.rcParams['font.size'] = self.config['fonts']['size']
        plt.rcParams['axes.titlesize'] = self.config['fonts']['title_size']
        plt.rcParams['figure.dpi'] = self.config['dpi']
        
        # 设置seaborn样式
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def plot_cdf(self, data: pd.Series, title: str, xlabel: str, 
                 output_path: str, log_scale: bool = False) -> None:
        """
        绘制累积分布函数图 (Helios风格)
        
        Args:
            data: 数据序列
            title: 图表标题
            xlabel: X轴标签
            output_path: 输出路径
            log_scale: 是否使用对数刻度
        """
        fig, ax = plt.subplots()
        
        # 移除无效值
        clean_data = data.dropna()
        clean_data = clean_data[clean_data > 0] if log_scale else clean_data
        
        if len(clean_data) == 0:
            logger.warning(f"数据为空，跳过绘制: {title}")
            return
        
        # 计算CDF
        sorted_data = np.sort(clean_data)
        y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF曲线
        ax.plot(sorted_data, y, linewidth=2, color=self.config['colors']['primary'])
        
        # 设置坐标轴
        if log_scale:
            ax.set_xscale('log')
        ax.set_ylabel('CDF')
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        median = np.median(clean_data)
        p95 = np.percentile(clean_data, 95)
        ax.axvline(median, color='red', linestyle='--', alpha=0.7, label=f'Median: {median:.1f}')
        ax.axvline(p95, color='orange', linestyle='--', alpha=0.7, label=f'95th: {p95:.1f}')
        ax.legend()
        
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close()
    
    def plot_time_series(self, data: pd.DataFrame, time_col: str, value_col: str,
                        title: str, ylabel: str, output_path: str) -> None:
        """
        绘制时间序列图
        
        Args:
            data: 数据DataFrame
            time_col: 时间列名
            value_col: 数值列名
            title: 图表标题
            ylabel: Y轴标签
            output_path: 输出路径
        """
        fig, ax = plt.subplots()
        
        # 确保时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(data[time_col]):
            data[time_col] = pd.to_datetime(data[time_col])
        
        # 绘制时间序列
        ax.plot(data[time_col], data[value_col], 
               linewidth=1.5, color=self.config['colors']['primary'])
        
        # 设置时间轴格式
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        plt.xticks(rotation=45)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close()
    
    def plot_hourly_heatmap(self, data: pd.DataFrame, time_col: str,
                           title: str, output_path: str) -> None:
        """
        绘制24小时热力图 (类似Helios的diurnal pattern)
        
        Args:
            data: 数据DataFrame
            time_col: 时间列名
            title: 图表标题
            output_path: 输出路径
        """
        fig, ax = plt.subplots()
        
        # 提取小时和星期几
        data = data.copy()
        data['hour'] = pd.to_datetime(data[time_col]).dt.hour
        data['weekday'] = pd.to_datetime(data[time_col]).dt.dayofweek
        
        # 创建透视表
        pivot_table = data.groupby(['weekday', 'hour']).size().unstack(fill_value=0)
        
        # 绘制热力图
        sns.heatmap(pivot_table, annot=False, cmap='YlOrRd', ax=ax)
        
        # 设置标签
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Day of Week')
        ax.set_title(title)
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close()
    
    def plot_bar_chart(self, data: Dict[str, float], title: str, xlabel: str,
                      ylabel: str, output_path: str, sort_values: bool = True) -> None:
        """
        绘制条形图
        
        Args:
            data: 数据字典
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            output_path: 输出路径
            sort_values: 是否按值排序
        """
        fig, ax = plt.subplots()
        
        # 准备数据
        if sort_values:
            sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
            labels, values = zip(*sorted_items)
        else:
            labels, values = zip(*data.items())
        
        # 绘制条形图
        bars = ax.bar(labels, values, color=self.config['colors']['primary'], alpha=0.8)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}', ha='center', va='bottom')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 旋转X轴标签以避免重叠
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close()
    
    def plot_dual_axis(self, data: pd.DataFrame, x_col: str, y1_col: str, y2_col: str,
                      title: str, y1_label: str, y2_label: str, output_path: str) -> None:
        """
        绘制双Y轴图表 (类似Helios的CPU/GPU对比图)
        
        Args:
            data: 数据DataFrame
            x_col: X轴列名
            y1_col: 左Y轴列名
            y2_col: 右Y轴列名
            title: 图表标题
            y1_label: 左Y轴标签
            y2_label: 右Y轴标签
            output_path: 输出路径
        """
        fig, ax1 = plt.subplots()
        
        # 左Y轴
        color1 = self.config['colors']['primary']
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y1_label, color=color1)
        line1 = ax1.plot(data[x_col], data[y1_col], color=color1, linewidth=2, label=y1_label)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # 右Y轴
        ax2 = ax1.twinx()
        color2 = self.config['colors']['secondary']
        ax2.set_ylabel(y2_label, color=color2)
        line2 = ax2.plot(data[x_col], data[y2_col], color=color2, linewidth=2, label=y2_label)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 添加图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, output_path)
        plt.close()
    
    def _save_figure(self, fig, output_path: str):
        """保存图表到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为多种格式
        for fmt in self.config['output_formats']:
            file_path = output_path.with_suffix(f'.{fmt}')
            fig.savefig(file_path, format=fmt, dpi=self.config['dpi'], 
                       bbox_inches='tight', facecolor='white')
            logger.info(f"图表已保存: {file_path}")


def create_output_directories(base_path: str, config: Dict[str, Any]) -> Dict[str, Path]:
    """
    创建输出目录结构
    
    Args:
        base_path: 基础路径
        config: 输出配置
        
    Returns:
        目录路径字典
    """
    base_path = Path(base_path)
    subdirs = config.get('subdirs', {})
    
    paths = {}
    for name, subdir in subdirs.items():
        path = base_path / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
        logger.info(f"创建输出目录: {path}")
    
    return paths


def create_modular_output_directories(base_path: str) -> Dict[str, Dict[str, Path]]:
    """
    为所有模块创建模块化输出目录结构

    Args:
        base_path: 基础输出路径

    Returns:
        模块化输出路径字典
    """
    base_path = Path(base_path)

    modules = [
        'philly_comparison',
        'data_overview',
        'cluster_characterization',
        'temporal_analysis',
        'job_characterization',
        'user_characterization'
    ]

    modular_paths = {}

    # 为每个模块创建目录
    for module in modules:
        module_paths = {
            'figures': base_path / 'figures' / module,
            'reports': base_path / 'reports' / module,
            'data': base_path / 'data' / module,
            'metrics': base_path / 'metrics' / module
        }

        # 创建模块目录
        for path in module_paths.values():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建模块输出目录: {path}")

        modular_paths[module] = module_paths

    # 也创建通用目录
    general_paths = {
        'figures': base_path / 'figures',
        'reports': base_path / 'reports',
        'data': base_path / 'data',
        'metrics': base_path / 'metrics'
    }

    for path in general_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    modular_paths['general'] = general_paths

    return modular_paths
