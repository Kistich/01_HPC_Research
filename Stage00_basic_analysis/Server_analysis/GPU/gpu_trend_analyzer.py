#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU趋势分析器 - 解析Excel工作表，按群组聚合分析GPU指标随时间的变化趋势
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import datetime
import warnings
import sys
import time
from tqdm import tqdm

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib为非交互式后端并使用英文字体
plt.switch_backend('agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_gpu_data():
    """分析GPU数据文件，为每个指标生成服务器组趋势图"""

    # Get the script directory and construct paths relative to it
    from pathlib import Path
    script_dir = Path(__file__).parent.resolve()
    # Go up to 01_HPC_Research directory
    hpc_research_dir = script_dir.parent.parent.parent
    raw_data_dir = hpc_research_dir / "Stage00_HPC_raw_data"

    # GPU数据文件路径
    gpu_files = {
        'GPUUtilization': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx'),
        'MemoryUtilization': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_MemoryUtilization_20250221_105207.xlsx'),
        'GraphicsClockSpeed': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235.xlsx'),
        'MemoryClockSpeed': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_MemoryClockSpeed_20250221_113253.xlsx'),
        'SMClockSpeed': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_SMClockSpeed_20250221_112610.xlsx'),
        'VideoClockSpeed': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_VideoClockSpeed_20250221_111923.xlsx'),
        'PowerDraw': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_当前GPU卡的PowerDraw_20250221_110550.xlsx'),
        'AverageUtilization': str(raw_data_dir / 'prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx')
    }

    # 指标英文名称映射
    metric_names = {
        'GPUUtilization': 'GPU Utilization (%)',
        'MemoryUtilization': 'Memory Utilization (%)',
        'GraphicsClockSpeed': 'Graphics Clock Speed (MHz)',
        'MemoryClockSpeed': 'Memory Clock Speed (MHz)',
        'SMClockSpeed': 'SM Clock Speed (MHz)',
        'VideoClockSpeed': 'Video Clock Speed (MHz)',
        'PowerDraw': 'Power Draw (W)',
        'AverageUtilization': 'Average GPU Utilization (%)'
    }

    # 输出基础目录
    base_output_dir = str(script_dir / 'trend')
    
    # 处理每个文件
    for i, (metric_key, file_path) in enumerate(gpu_files.items()):
        print(f"[{i+1}/{len(gpu_files)}] 处理文件: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在!")
            continue
        
        # 为每个指标创建输出目录
        output_dir = os.path.join(base_output_dir, metric_key)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 读取Excel文件中的所有工作表并提取群组信息
            group_data = read_all_worksheets(file_path)
            
            if not group_data:
                print(f"警告: 未能提取有效数据")
                continue
                
            # 生成群组趋势对比图
            generate_group_comparison_plot(group_data, metric_names[metric_key], output_dir)
            
            print(f"分析完成. 结果保存到: {output_dir}")
            
        except Exception as e:
            print(f"处理错误: {str(e)}")
    
    print("所有GPU指标分析完成.")

def extract_group_from_sheet_name(sheet_name):
    """从工作表名提取群组信息，支持中英文名称"""
    # 中英文名称映射
    chinese_to_english = {
        '训练': 'training',
        '推理': 'inference',
        'GPU服务器1': 'GPU1',
        'GPU服务器2': 'GPU2',
        'GPU服务器3': 'GPU3'
    }
    
    # 常见的群组前缀(英文)
    known_groups = ['GPU1', 'GPU2', 'GPU3', 'training', 'inference']
    
    # 检查是否包含中文群组名称
    sheet_lower = sheet_name.lower()
    for cn_name, en_name in chinese_to_english.items():
        if cn_name in sheet_name:
            return en_name
    
    # 检查英文名称匹配
    for group in known_groups:
        pattern = re.compile(r'\b' + re.escape(group) + r'\b', re.IGNORECASE)
        if pattern.search(sheet_name):
            return group
    
    # 尝试从服务器名称中提取GPU组信息
    if 'gpu' in sheet_lower:
        match = re.search(r'gpu\s*(\d+)', sheet_lower)
        if match:
            return f'GPU{match.group(1)}'
    
    # 检查是否包含特定关键词
    if 'train' in sheet_lower:
        return 'training'
    if 'infer' in sheet_lower:
        return 'inference'
    
    # 无法识别群组
    return None

def read_all_worksheets(excel_file):
    """读取Excel文件中的所有工作表，并按群组整理数据"""
    # 获取Excel文件中的所有工作表名
    xl = pd.ExcelFile(excel_file)
    sheet_names = xl.sheet_names
    
    # 初始化群组数据字典
    group_data = {}
    
    # 创建进度条
    with tqdm(total=len(sheet_names), desc="处理工作表", unit="sheet") as pbar:
        # 逐个处理工作表
        for sheet_name in sheet_names:
            # 从工作表名提取群组信息
            group = extract_group_from_sheet_name(sheet_name)
            if not group:
                pbar.update(1)
                continue
            
            # 读取工作表数据
            try:
                df = xl.parse(sheet_name)
                
                # 检查数据格式
                if df.empty:
                    pbar.update(1)
                    continue
                
                # 确保有时间列和值列
                if df.shape[1] < 2:
                    pbar.update(1)
                    continue
                
                # 提取时间和值列
                time_col = df.columns[0]
                value_col = df.columns[1]
                
                # 创建标准化的数据帧
                processed_df = pd.DataFrame({
                    'time': pd.to_datetime(df[time_col]),
                    'value': pd.to_numeric(df[value_col], errors='coerce')
                })
                
                # 过滤NaN值
                processed_df = processed_df.dropna()
                
                if processed_df.empty:
                    pbar.update(1)
                    continue
                
                # 添加时间相关特征
                processed_df['date'] = processed_df['time'].dt.date
                processed_df['hour'] = processed_df['time'].dt.hour
                processed_df['day'] = processed_df['time'].dt.day
                processed_df['month'] = processed_df['time'].dt.month_name()
                processed_df['year'] = processed_df['time'].dt.year
                processed_df['day_of_week'] = processed_df['time'].dt.day_name()
                
                # 将工作表数据添加到群组
                if group not in group_data:
                    group_data[group] = {
                        'raw': processed_df,
                        'servers': [sheet_name]
                    }
                else:
                    # 合并同一组的数据
                    group_data[group]['raw'] = pd.concat([group_data[group]['raw'], processed_df])
                    group_data[group]['servers'].append(sheet_name)
            
            except Exception as e:
                # 处理数据时出错，继续处理下一个工作表
                pass
            
            pbar.update(1)
    
    # 对每个群组数据进行日期聚合
    for group in group_data:
        if 'raw' in group_data[group]:
            # 按日期聚合
            daily_data = group_data[group]['raw'].groupby('date')['value'].agg(['mean', 'min', 'max', 'std']).reset_index()
            daily_data.columns = ['date', 'value', 'min', 'max', 'std']
            group_data[group]['daily'] = daily_data
            
            # 计算群组统计信息
            group_data[group]['stats'] = {
                'mean': group_data[group]['raw']['value'].mean(),
                'min': group_data[group]['raw']['value'].min(),
                'max': group_data[group]['raw']['value'].max(),
                'std': group_data[group]['raw']['value'].std(),
                'count': len(group_data[group]['servers'])
            }
    
    return group_data

def generate_group_comparison_plot(group_data, metric_name, output_dir):
    """生成服务器组性能指标随时间变化的趋势图"""
    if not group_data:
        return
    
    # 设置组名显示映射
    group_display_names = {
        'GPU1': 'GPU Server Group 1',
        'GPU2': 'GPU Server Group 2',
        'GPU3': 'GPU Server Group 3',
        'training': 'Training Servers',
        'inference': 'Inference Servers'
    }
    
    # 设置颜色映射
    color_map = {
        'GPU1': '#1f77b4',  # 蓝色
        'GPU2': '#ff7f0e',  # 橙色
        'GPU3': '#2ca02c',  # 绿色
        'training': '#d62728',  # 红色
        'inference': '#9467bd',  # 紫色
    }
    
    # 创建趋势图
    plt.figure(figsize=(14, 8))
    
    # 绘制每个组的时间序列
    legend_entries = []
    
    for group, group_info in group_data.items():
        if 'daily' not in group_info:
            continue
            
        daily_data = group_info['daily']
        
        # 确保日期是按时间顺序排列的
        daily_data = daily_data.sort_values('date')
        
        # 获取显示名称
        display_name = group_display_names.get(group, group)
        
        # 绘制趋势线
        line, = plt.plot(daily_data['date'], daily_data['value'], '-', 
                  color=color_map.get(group, '#808080'), linewidth=2, 
                  label=f"{display_name}")
        
        legend_entries.append(line)
        
        # 打印简单统计信息
        min_val = daily_data['value'].min()
        max_val = daily_data['value'].max()
        mean_val = daily_data['value'].mean()
        print(f"  - 群组 '{display_name}': 最小值={min_val:.2f}, 最大值={max_val:.2f}, 平均值={mean_val:.2f}")
    
    # 添加标签和标题
    plt.title(f"Server Groups {metric_name} Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend(handles=legend_entries)
    
    # 改进x轴日期显示
    plt.gcf().autofmt_xdate()
    
    # 保存图表
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'server_groups_trend.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 生成箱线图比较不同组的分布情况
    plt.figure(figsize=(14, 8))
    
    # 准备箱线图数据
    boxplot_data = []
    boxplot_labels = []
    
    for group, group_info in group_data.items():
        if 'daily' not in group_info:
            continue
            
        daily_data = group_info['daily']
        boxplot_data.append(daily_data['value'])
        
        display_name = group_display_names.get(group, group)
        boxplot_labels.append(f"{display_name}\n(avg: {daily_data['value'].mean():.1f})")
    
    # 创建箱线图
    box = plt.boxplot(
        boxplot_data,
        labels=boxplot_labels,
        patch_artist=True,
        showfliers=True,
        whis=1.5
    )
    
    # 自定义箱线图颜色
    for i, patch in enumerate(box['boxes']):
        group = list(group_data.keys())[i]
        patch.set_facecolor(color_map.get(group, '#808080'))
        patch.set_alpha(0.6)
    
    plt.title(f"Server Groups {metric_name} Distribution")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 保存箱线图
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'server_groups_distribution.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    # 生成简单分析报告
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"GPU指标分析报告 - {metric_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("各群组指标统计:\n")
        f.write("-" * 50 + "\n")
        
        for group, group_info in group_data.items():
            if 'stats' not in group_info:
                continue
                
            stats = group_info['stats']
            display_name = group_display_names.get(group, group)
            
            f.write(f"\n● {display_name}:\n")
            f.write(f"  平均值: {stats['mean']:.2f}\n")
            f.write(f"  最大值: {stats['max']:.2f}\n")
            f.write(f"  最小值: {stats['min']:.2f}\n")
            f.write(f"  标准差: {stats['std']:.2f}\n")
            f.write(f"  服务器数量: {stats['count']}\n")

if __name__ == "__main__":
    try:
        start_time = time.time()
        analyze_gpu_data()
        end_time = time.time()
        print(f"总执行时间: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 