#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统负载趋势分析器 - 解析Excel工作表，按群组聚合分析系统负载指标随时间的变化趋势
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

# 预定义已知的服务器群组模式
SERVER_GROUPS = {
    'cpu1': r'^cpu1-\d+$',
    'cpu2': r'^cpu2-\d+$', 
    'cpu3': r'^cpu3-\d+$',  # 添加可能的cpu3群组
    'gpu1': r'^gpu1-\d+$',
    'gpu2': r'^gpu2-\d+$',
    'gpu3': r'^gpu3-\d+$',
    'bigmem': r'^bigmen-\d+$',  # 注意这里是"bigmen"，不是"bigmem"
    'inference': r'^推理服务器\d+$',
    'training': r'^训练服务器\d+$'
}

# 群组显示名称映射 - 改为全英文
GROUP_DISPLAY_NAMES = {
    'cpu1': 'CPU Group 1',
    'cpu2': 'CPU Group 2',
    'cpu3': 'CPU Group 3',
    'gpu1': 'GPU Group 1', 
    'gpu2': 'GPU Group 2',
    'gpu3': 'GPU Group 3',
    'bigmem': 'Big Memory Servers',
    'inference': 'Inference Servers',
    'training': 'Training Servers'
}

# 群组颜色映射
COLOR_MAP = {
    'cpu1': '#1f77b4',  # 蓝色
    'cpu2': '#2ca02c',  # 绿色
    'cpu3': '#ff7f0e',  # 橙色
    'gpu1': '#d62728',  # 红色
    'gpu2': '#8c564b',  # 棕色
    'gpu3': '#e377c2',  # 粉色
    'bigmem': '#7f7f7f', # 灰色
    'inference': '#bcbd22', # 黄色
    'training': '#17becf'  # 青色
}

def analyze_load_data():
    """分析系统负载数据文件，为每个指标生成服务器组趋势图"""
    
    # 系统负载数据文件路径
    load_files = {
        'load_1min': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_系统平均负载_1分钟负载（%）_20250220_160120.xlsx',
        'load_5min': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_系统平均负载_5分钟负载（%）_20250220_160458.xlsx',
        'load_15min': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_系统平均负载_15分钟负载（%）_20250220_160842.xlsx'
    }
    
    # 指标英文名称映射
    metric_names = {
        'load_1min': 'System Load (1min) %',
        'load_5min': 'System Load (5min) %',
        'load_15min': 'System Load (15min) %'
    }
    
    # 输出基础目录
    base_output_dir = '/mnt/raid/liuhongbin/job_analysis/job_analysis/Server_analysis/Load'
    
    # 处理每个文件
    for i, (metric_key, file_path) in enumerate(load_files.items()):
        print(f"[{i+1}/{len(load_files)}] 处理文件: {os.path.basename(file_path)}")
        
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在!")
            continue
        
        # 为每个指标创建输出目录
        output_dir = os.path.join(base_output_dir, metric_key)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 读取Excel文件中的所有工作表并提取群组信息
            group_data = read_excel_file(file_path)
            
            if not group_data:
                print(f"警告: 未能提取有效数据")
                continue
                
            # 识别异常值
            identify_anomalies(group_data, output_dir)
            
            # 生成不同视角的图表
            generate_visualizations(group_data, metric_names[metric_key], output_dir)
            
            print(f"分析完成. 结果保存到: {output_dir}")
            
        except Exception as e:
            print(f"处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("所有系统负载指标分析完成.")

def identify_server_group(sheet_name):
    """识别工作表所属的服务器群组"""
    for group, pattern in SERVER_GROUPS.items():
        if re.match(pattern, sheet_name):
            return group
    return None

def read_excel_file(file_path):
    """读取Excel文件中的所有工作表，按服务器群组聚合数据"""
    
    xl = pd.ExcelFile(file_path, engine='openpyxl')
    sheet_names = xl.sheet_names
    group_data = {}
    
    # 用于统计
    total_sheets = len(sheet_names)
    processed_sheets = 0
    grouped_sheets = 0
    
    print(f"从Excel文件中读取数据...")
    print(f"总工作表数: {total_sheets}")
    
    # 按群组整理工作表
    sheet_groups = {}
    for sheet_name in sheet_names:
        group = identify_server_group(sheet_name)
        if group:
            if group not in sheet_groups:
                sheet_groups[group] = []
            sheet_groups[group].append(sheet_name)
    
    print(f"已识别服务器群组: {len(sheet_groups)}")
    
    # 处理每个群组的工作表
    for group, sheets in tqdm(sheet_groups.items(), desc="处理服务器群组"):
        all_timestamps = []
        all_values = []
        server_count = 0
        
        # 处理每个工作表
        for sheet_name in tqdm(sheets, desc=f"处理 {group} 群组的工作表", leave=False):
            try:
                # 读取工作表数据
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                
                if len(df) == 0 or 'timestamp' not in df.columns or 'value' not in df.columns:
                    print(f"  警告: 工作表 {sheet_name} 格式无效，跳过")
                    continue
                
                # 收集数据
                all_timestamps.extend(df['timestamp'])
                all_values.extend(df['value'])
                server_count += 1
                processed_sheets += 1
                
            except Exception as e:
                print(f"  警告: 读取工作表 {sheet_name} 时出错: {str(e)}")
        
        if server_count > 0:
            grouped_sheets += server_count
            
            # 创建数据框
            raw_data = pd.DataFrame({
                'timestamp': all_timestamps,
                'value': all_values
            })
            
            # 确保timestamp是datetime类型
            if not pd.api.types.is_datetime64_dtype(raw_data['timestamp']):
                raw_data['timestamp'] = pd.to_datetime(raw_data['timestamp'])
            
            # 按日期重采样以获得每日统计
            daily_data = resample_to_daily(raw_data, 'mean')
            
            # 计算统计信息
            stats = {
                'mean': raw_data['value'].mean(),
                'max': raw_data['value'].max(),
                'min': raw_data['value'].min(),
                'std': raw_data['value'].std(),
                'count': server_count
            }
            
            # 存储群组数据
            group_data[group] = {
                'raw': raw_data,
                'daily': daily_data,
                'stats': stats
            }
    
    print(f"数据读取完成.")
    print(f"从 {total_sheets} 个工作表中提取了 {len(group_data)} 个服务器群组")
    print(f"处理的工作表: {processed_sheets}, 已分组: {grouped_sheets}")
    
    return group_data

def resample_to_daily(df, method='mean'):
    """按日期重采样数据"""
    # 设置索引为时间戳
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    
    # 按日重采样
    if method == 'mean':
        daily = df_copy.resample('D').mean().reset_index()
    elif method == 'max':
        daily = df_copy.resample('D').max().reset_index()
    else:
        daily = df_copy.resample('D').mean().reset_index()
        
    return daily

def identify_anomalies(group_data, output_dir):
    """识别并记录异常值"""
    anomaly_file = os.path.join(output_dir, 'anomalies_report.txt')
    
    with open(anomaly_file, 'w') as f:
        f.write("系统负载异常值报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for group, data in group_data.items():
            if 'raw' not in data:
                continue
                
            raw_values = data['raw']['value'].values
            timestamps = data['raw']['timestamp'].values
            
            # 计算IQR (四分位距)
            q1 = np.percentile(raw_values, 25)
            q3 = np.percentile(raw_values, 75)
            iqr = q3 - q1
            
            # 定义异常阈值 (更严格的阈值以捕获极端值)
            upper_threshold = q3 + 3 * iqr
            extreme_threshold = 500  # 500%负载认为是极端异常
            
            # 找出异常点
            anomaly_indices = np.where(raw_values > upper_threshold)[0]
            
            if len(anomaly_indices) == 0:
                f.write(f"\n● {GROUP_DISPLAY_NAMES.get(group, group)}:\n")
                f.write("  未发现异常值\n")
                continue
            
            # 按值排序找出最高的异常
            sorted_anomaly_indices = anomaly_indices[np.argsort(-raw_values[anomaly_indices])]
            
            # 显示最多20个最高异常值
            top_n = min(20, len(sorted_anomaly_indices))
            
            f.write(f"\n● {GROUP_DISPLAY_NAMES.get(group, group)}:\n")
            f.write(f"  检测到 {len(anomaly_indices)} 个异常值 (高于 {upper_threshold:.2f}%)\n")
            f.write(f"  最高的 {top_n} 个异常值:\n")
            
            for i in range(top_n):
                idx = sorted_anomaly_indices[i]
                ts = timestamps[idx]
                value = raw_values[idx]
                
                # 格式化时间戳
                if isinstance(ts, np.datetime64):
                    ts = pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                elif hasattr(ts, 'strftime'):
                    ts = ts.strftime('%Y-%m-%d %H:%M:%S')
                
                severity = "极端异常" if value > extreme_threshold else "异常"
                f.write(f"    - {ts}: {value:.2f}% ({severity})\n")

def generate_visualizations(group_data, metric_name, output_dir):
    """生成不同类型的可视化图表"""
    print(f"生成图表中...")
    
    # 生成不同类型的趋势图
    generate_trend_plot(group_data, metric_name, output_dir)
    generate_limited_trend_plot(group_data, metric_name, output_dir, y_limit=200)
    generate_percentile_trend_plot(group_data, metric_name, output_dir, percentile=99)
    
    # 生成分布图
    generate_enhanced_boxplot(group_data, metric_name, output_dir)
    
    # 生成热图
    try:
        generate_heatmap(group_data, metric_name, output_dir)
    except Exception as e:
        print(f"生成热图时出错: {str(e)}")
    
    print(f"图表生成完成")

def generate_trend_plot(group_data, metric_name, output_dir):
    """生成服务器群组趋势图（对数刻度以显示异常值）"""
    plt.figure(figsize=(16, 9))
    
    for group, group_info in group_data.items():
        if 'daily' not in group_info:
            continue
            
        daily_data = group_info['daily']
        display_name = GROUP_DISPLAY_NAMES.get(group, group)
        
        # 绘制数据线
        plt.plot(
            daily_data['timestamp'], 
            daily_data['value'],
            label=f"{display_name} (Avg: {group_info['stats']['mean']:.1f}%)",
            color=COLOR_MAP.get(group, None),
            alpha=0.8
        )
    
    # 使用对数刻度以便同时显示正常值和异常值
    plt.yscale('log')
    
    # 添加100%参考线
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, label="100% Load")
    
    # 设置图表标题和标签
    plt.title(f"Server Groups {metric_name} Trends (Log Scale)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(f"{metric_name} (Log Scale)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 保存趋势图
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'server_groups_trends_log.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_limited_trend_plot(group_data, metric_name, output_dir, y_limit=200):
    """生成Y轴限制在指定值以内的趋势图，超出范围的点标记为特殊符号"""
    plt.figure(figsize=(16, 9))
    
    # 用于记录超出范围的点
    outlier_markers = []
    
    for group, group_info in group_data.items():
        if 'daily' not in group_info:
            continue
            
        daily_data = group_info['daily']
        display_name = GROUP_DISPLAY_NAMES.get(group, group)
        
        # 拷贝数据并限制y值
        x = daily_data['timestamp'].values
        y = daily_data['value'].values.copy()
        
        # 确定超出范围的点
        outliers = y > y_limit
        outlier_x = x[outliers]
        outlier_y = np.full_like(y[outliers], y_limit * 0.95)  # 在上限下方显示
        
        # 限制y值以便于图表显示
        y[y > y_limit] = y_limit
        
        # 绘制主数据线
        plt.plot(
            x, 
            y,
            label=f"{display_name} (Avg: {group_info['stats']['mean']:.1f}%)",
            color=COLOR_MAP.get(group, None),
            alpha=0.8
        )
        
        # 标记超出范围的点
        if len(outlier_x) > 0:
            marker = plt.plot(
                outlier_x, 
                outlier_y,
                'v',  # 向下三角形
                color=COLOR_MAP.get(group, None),
                markersize=8,
                alpha=0.7
            )
            outlier_markers.append(marker[0])
    
    # 添加100%参考线
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, label="100% Load")
    
    # 如果有超出范围的点，添加说明
    if outlier_markers:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.figtext(0.5, 0.01, f"▼ 表示超过 {y_limit}% 的数据点", ha='center')
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 设置图表标题和标签
    plt.title(f"Server Groups {metric_name} Trends (Limited to {y_limit}%)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 设置Y轴范围
    plt.ylim(0, y_limit)
    
    # 保存趋势图
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'server_groups_trends_{y_limit}limit.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_percentile_trend_plot(group_data, metric_name, output_dir, percentile=99):
    """生成只显示指定百分位数以下数据的趋势图"""
    plt.figure(figsize=(16, 9))
    
    y_max = 0
    
    for group, group_info in group_data.items():
        if 'daily' not in group_info:
            continue
            
        daily_data = group_info['daily']
        display_name = GROUP_DISPLAY_NAMES.get(group, group)
        
        # 计算百分位数阈值
        threshold = np.percentile(daily_data['value'], percentile)
        
        # 过滤数据
        mask = daily_data['value'] <= threshold
        filtered_x = daily_data['timestamp'][mask]
        filtered_y = daily_data['value'][mask]
        
        # 更新y轴最大值
        current_max = filtered_y.max()
        if current_max > y_max:
            y_max = current_max
        
        # 绘制过滤后的数据
        plt.plot(
            filtered_x, 
            filtered_y,
            label=f"{display_name} (Avg: {group_info['stats']['mean']:.1f}%)",
            color=COLOR_MAP.get(group, None),
            alpha=0.8
        )
    
    # 添加100%参考线
    if y_max >= 100:
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, label="100% Load")
    
    # 设置图表标题和标签
    plt.title(f"Server Groups {metric_name} Trends (Only {percentile}th Percentile)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # 设置Y轴范围
    plt.ylim(0, y_max*1.05)
    
    # 保存趋势图
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'server_groups_trends_{percentile}percentile.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_enhanced_boxplot(group_data, metric_name, output_dir):
    """生成增强版箱线图，使用对数刻度显示异常值"""
    plt.figure(figsize=(14, 8))
    
    boxplot_data = []
    boxplot_labels = []
    
    for group, group_info in group_data.items():
        if 'raw' not in group_info:
            continue
            
        raw_data = group_info['raw']['value']
        boxplot_data.append(raw_data)
        
        display_name = GROUP_DISPLAY_NAMES.get(group, group)
        boxplot_labels.append(f"{display_name}\n(avg: {raw_data.mean():.1f}%)")
    
    # 创建箱线图
    box = plt.boxplot(
        boxplot_data,
        labels=boxplot_labels,
        patch_artist=True,
        showfliers=True,
        whis=1.5
    )
    
    # 绘制100%参考线
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.5)
    
    # 自定义箱线图颜色
    for i, patch in enumerate(box['boxes']):
        group = list(group_data.keys())[i]
        patch.set_facecolor(COLOR_MAP.get(group, '#808080'))
        patch.set_alpha(0.6)
    
    # 使用对数刻度显示大范围的值
    plt.yscale('log')
    
    # 设置Y轴下限，确保小值可见
    plt.ylim(bottom=0.1)
    
    plt.title(f"Server Groups {metric_name} Distribution (Log Scale)", fontsize=16)
    plt.ylabel(f"{metric_name} (Log Scale)", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(fontsize=12)
    
    # 保存箱线图
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'server_groups_distribution_log.png')
    plt.savefig(output_path, dpi=150)
    plt.close()

def generate_heatmap(group_data, metric_name, output_dir):
    """生成按小时和工作日显示负载模式的热图"""
    for group, group_info in group_data.items():
        if 'raw' not in group_info:
            continue
            
        # 获取数据
        df = pd.DataFrame({
            'timestamp': group_info['raw']['timestamp'],
            'value': group_info['raw']['value']
        })
        
        # 确保timestamp是datetime类型
        if not pd.api.types.is_datetime64_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 添加小时和工作日列
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        
        # 创建热图数据 - 按小时和星期几聚合
        try:
            pivot_table = df.pivot_table(
                values='value', 
                index='hour',
                columns='dayofweek',
                aggfunc='mean'
            )
            
            if pivot_table.empty:
                print(f"  警告: {group} 组的热图数据为空")
                continue
                
            heatmap_data = pivot_table.values
            
            # 创建热图
            plt.figure(figsize=(10, 8))
            
            display_name = GROUP_DISPLAY_NAMES.get(group, group)
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            hours = list(range(24))
            
            # 确保数据和刻度匹配
            valid_cols = min(len(weekdays), heatmap_data.shape[1])
            valid_rows = min(len(hours), heatmap_data.shape[0])
            
            plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            plt.colorbar(label=metric_name)
            
            plt.title(f"{display_name} - {metric_name} by Hour and Day", fontsize=16)
            plt.xlabel('Day of Week', fontsize=14)
            plt.ylabel('Hour of Day', fontsize=14)
            
            # 设置刻度标签
            plt.xticks(range(valid_cols), weekdays[:valid_cols], rotation=45)
            plt.yticks(range(valid_rows), hours[:valid_rows])
            
            # 保存热图
            plt.tight_layout()
            output_dir_heatmaps = os.path.join(output_dir, 'heatmaps')
            os.makedirs(output_dir_heatmaps, exist_ok=True)
            output_path = os.path.join(output_dir_heatmaps, f'{group}_heatmap.png')
            plt.savefig(output_path, dpi=150)
            plt.close()
        except Exception as e:
            print(f"  警告: 生成 {group} 热图时出错: {str(e)}")

if __name__ == "__main__":
    try:
        start_time = time.time()
        analyze_load_data()
        end_time = time.time()
        print(f"总执行时间: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 