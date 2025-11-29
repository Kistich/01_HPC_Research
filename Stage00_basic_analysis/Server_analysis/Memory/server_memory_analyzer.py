#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
服务器内存分析器 - 解析Excel工作表，按群组聚合分析指标随时间的变化趋势
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

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib为非交互式后端并使用英文字体
plt.switch_backend('agg')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_memory_data():
    """分析内存数据文件，为每个指标生成服务器组趋势图"""
    
    # 文件路径 - 使用提供的确切路径
    memory_files = {
        'available': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_内存使用率_可用（G）_20250221_083923.xlsx',
        'utilization': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_内存使用率_使用率（%）_20250221_085433.xlsx',
        'used': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_内存使用率_已用（G）_20250221_085041.xlsx',
        'total': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_内存使用率_总内存（G）_20250221_084630.xlsx'
    }
    
    # 指标英文名称映射
    metric_names = {
        'available': 'Available Memory (GB)',
        'utilization': 'Memory Utilization (%)',
        'used': 'Used Memory (GB)',
        'total': 'Total Memory (GB)'
    }
    
    # 输出基础目录
    base_output_dir = '/mnt/raid/liuhongbin/job_analysis/job_analysis/Server_analysis/Memory'
    
    # 处理每个文件
    for metric_key, file_path in memory_files.items():
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在!")
            continue
            
        print(f"\n-------------------------------------")
        print(f"处理 {metric_key} 数据...")
        print(f"文件: {file_path}")
        
        # 为每个指标创建输出目录
        output_dir = os.path.join(base_output_dir, metric_key)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 读取Excel文件中的所有工作表并提取群组信息
            print(f"读取Excel文件中的所有工作表...")
            group_data = read_all_worksheets(file_path)
            
            if not group_data:
                print(f"警告: 未能从文件中提取到任何有效数据")
                continue
                
            # 生成群组趋势对比图
            generate_group_comparison_plot(group_data, metric_names[metric_key], output_dir)
            
            print(f"{metric_key} 分析完成. 输出保存到 {output_dir}")
            
        except Exception as e:
            print(f"处理 {metric_key} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n所有分析完成.")

def read_all_worksheets(excel_file):
    """读取Excel文件中的所有工作表，并按群组整理数据"""
    print(f"开始读取Excel文件: {excel_file}")
    
    # 获取Excel文件中的所有工作表名
    xl = pd.ExcelFile(excel_file)
    sheet_names = xl.sheet_names
    print(f"发现 {len(sheet_names)} 个工作表")
    
    # 初始化群组数据字典
    group_data = {}
    
    # 逐个处理工作表
    for sheet_name in sheet_names:
        print(f"处理工作表: {sheet_name}")
        
        # 从工作表名提取群组信息
        group = extract_group_from_sheet_name(sheet_name)
        if not group:
            print(f"  - 警告: 无法从工作表名 '{sheet_name}' 提取群组信息，跳过")
            continue
            
        print(f"  - 归类到群组: {group}")
        
        # 读取工作表数据
        try:
            df = xl.parse(sheet_name)
            
            # 检查数据格式
            if df.empty:
                print(f"  - 警告: 工作表 '{sheet_name}' 为空，跳过")
                continue
                
            print(f"  - 读取 {len(df)} 行数据")
            
            # 检查必要的列
            if 'time' not in df.columns or 'value' not in df.columns:
                print(f"  - 警告: 工作表 '{sheet_name}' 缺少必要的列 (time/value)，尝试猜测列名")
                
                # 尝试猜测时间列
                time_col = None
                for col in df.columns:
                    if 'time' in col.lower() or 'date' in col.lower() or '时间' in col:
                        time_col = col
                        break
                        
                # 尝试猜测值列
                value_col = None
                for col in df.columns:
                    if 'value' in col.lower() or 'val' in col.lower() or '值' in col:
                        value_col = col
                        break
                
                if time_col is None or value_col is None:
                    print(f"  - 错误: 无法识别时间和值列，跳过该工作表")
                    continue
                    
                print(f"  - 使用列: 时间='{time_col}'，值='{value_col}'")
                
                # 创建新的DataFrame，只包含时间和值列
                df = df[[time_col, value_col]].copy()
                df.columns = ['time', 'value']
            
            # 确保time列是datetime类型
            if not pd.api.types.is_datetime64_dtype(df['time']):
                print(f"  - 转换时间列为datetime格式")
                df['time'] = pd.to_datetime(df['time'], errors='coerce')
            
            # 确保value列是数值类型
            if not pd.api.types.is_numeric_dtype(df['value']):
                print(f"  - 转换值列为数值类型")
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
            
            # 删除无效数据
            df = df.dropna()
            
            # 添加到对应的群组
            if group not in group_data:
                group_data[group] = []
            
            group_data[group].append(df)
            print(f"  - 成功添加到群组 '{group}'，有效数据点: {len(df)}")
            
        except Exception as e:
            print(f"  - 处理工作表 '{sheet_name}' 时出错: {str(e)}")
            continue
    
    # 整合每个群组的数据
    print("整合群组数据...")
    for group in list(group_data.keys()):
        if not group_data[group]:
            print(f"  - 群组 '{group}' 没有有效数据，移除")
            del group_data[group]
            continue
            
        # 合并同一群组的所有DataFrame
        combined_df = pd.concat(group_data[group])
        print(f"  - 群组 '{group}': 合并 {len(group_data[group])} 个工作表，共 {len(combined_df)} 行数据")
        
        # 添加日期列
        combined_df['date'] = combined_df['time'].dt.date
        
        # 按日期分组计算平均值
        daily_avg = combined_df.groupby('date')['value'].mean().reset_index()
        print(f"  - 群组 '{group}': 计算 {len(daily_avg)} 个日均值")
        
        # 用每日平均值替换原始数据
        group_data[group] = daily_avg
    
    print(f"完成数据整理，共提取 {len(group_data)} 个群组的数据")
    return group_data

def extract_group_from_sheet_name(sheet_name):
    """从工作表名提取群组信息，支持中英文名称"""
    # 中英文名称映射
    chinese_to_english = {
        '训练': 'training',
        '推理': 'inference',
        '大内存': 'bigmen',
        'CPU服务器1': 'cpu1',
        'CPU服务器2': 'cpu2',
        'CPU服务器3': 'cpu3',
        'GPU服务器1': 'gpu1',
        'GPU服务器2': 'gpu2',
        'GPU服务器3': 'gpu3'
    }
    
    # 常见的群组前缀(英文)
    known_groups = ['cpu1', 'cpu2', 'cpu3', 'gpu1', 'gpu2', 'gpu3', 'bigmen', 'training', 'inference']
    
    # 检查是否包含中文群组名称
    sheet_lower = sheet_name.lower()
    for cn_name, en_name in chinese_to_english.items():
        if cn_name in sheet_name:
            print(f"  - 检测到中文群组名 '{cn_name}'，映射为 '{en_name}'")
            return en_name
    
    # 尝试直接匹配英文前缀
    for group in known_groups:
        if sheet_lower.startswith(group):
            return group
    
    # 尝试使用正则表达式提取
    patterns = [
        r'^(cpu\d+)',  # 匹配cpu后跟数字
        r'^(gpu\d+)',  # 匹配gpu后跟数字
        r'^(bigmen)',  # 匹配bigmen
        r'^(training)', # 匹配training
        r'^(inference)', # 匹配inference
        r'^(\w+)[-_]', # 匹配前缀直到连字符或下划线
    ]
    
    for pattern in patterns:
        match = re.match(pattern, sheet_lower)
        if match:
            return match.group(1)
    
    # 如果都没匹配到，尝试提取数字前的部分作为群组名
    match = re.match(r'^([a-zA-Z]+)', sheet_lower)
    if match:
        return match.group(1)
    
    # 最后返回一个安全的默认值
    print(f"  - 警告: 无法识别群组名，使用默认名称 'other'")
    return 'other'

def generate_group_comparison_plot(group_data, metric_name, output_dir):
    """生成服务器群组趋势对比图"""
    print(f"生成群组趋势对比图: {metric_name}")
    
    plt.figure(figsize=(15, 8))
    
    # 设置颜色映射
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_map = {group: colors[i % len(colors)] for i, group in enumerate(group_data.keys())}
    
    # 组名显示映射 - 确保显示更加友好的名称
    group_display_names = {
        'cpu1': 'CPU1',
        'cpu2': 'CPU2',
        'cpu3': 'CPU3',
        'gpu1': 'GPU1',
        'gpu2': 'GPU2', 
        'gpu3': 'GPU3',
        'bigmen': 'BigMem',
        'training': 'Training',
        'inference': 'Inference',
        'other': 'Other'
    }
    
    # 图例项
    legend_entries = []
    
    # 绘制每个群组的趋势线
    for group, daily_data in group_data.items():
        # 获取显示名称
        display_name = group_display_names.get(group, group)
        
        # 绘制趋势线
        line, = plt.plot(daily_data['date'], daily_data['value'], '-', 
                  color=color_map[group], linewidth=2, 
                  label=f"{display_name}")
        
        legend_entries.append(line)
        
        # 打印数据范围信息
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
    
    print(f"群组趋势对比图已保存到 {output_path}")

if __name__ == "__main__":
    try:
        start_time = time.time()
        analyze_memory_data()
        end_time = time.time()
        print(f"总执行时间: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
                
                #