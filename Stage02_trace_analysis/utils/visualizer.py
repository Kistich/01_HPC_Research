#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
简单的可视化器模块
为job_characterization分析提供基础可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class Visualizer:
    """简单的可视化器"""
    
    def __init__(self):
        """初始化可视化器"""
        self.setup_style()
        
    def setup_style(self):
        """设置基础图表样式"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_cdf(self, data, title, xlabel, output_path, log_scale=False):
        """绘制累积分布函数图"""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # 计算CDF
            sorted_data = np.sort(data)
            y = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
            
            ax.plot(sorted_data, y, linewidth=2)
            
            if log_scale:
                ax.set_xscale('log')
            
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Percentage (%)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"CDF图已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"绘制CDF图失败: {e}")
            
    def plot_bar_chart(self, data, title, xlabel, ylabel, output_path):
        """绘制条形图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if isinstance(data, dict):
                keys = list(data.keys())
                values = list(data.values())
            else:
                keys = data.index
                values = data.values
            
            ax.bar(keys, values)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # 旋转x轴标签如果太多
            if len(keys) > 10:
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"条形图已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"绘制条形图失败: {e}")
