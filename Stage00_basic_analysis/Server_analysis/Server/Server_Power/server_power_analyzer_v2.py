#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
服务器功耗分析器 v2 - 解析Excel工作表，按群组聚合分析指标随时间的变化趋势
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

def analyze_power_data():
    """分析功耗数据文件，为每个指标生成服务器组趋势图"""
    
    # 更新为用户提供的真实文件路径
    power_files = {
        'total_power': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_电源总功率_20250221_095857.xlsx',
        'power_watts': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_PowerReadingInWatts_20250221_102521.xlsx',
        'power_state': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_PowerState_20250221_100543.xlsx',
        'power_status': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_PowerStatus_20250221_100203.xlsx',
        'voltage_state': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_ReportedStateOfAVoltageSensor_20250221_104536.xlsx',
        'voltage_reading': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_VoltageReadingInVolts_20250221_113609.xlsx'
    }
    
    # 更新指标英文名称映射
    metric_names = {
        'total_power': 'Total Power (W)',
        'power_watts': 'Power Reading (W)',
        'power_state': 'Power State',
        'power_status': 'Power Status',
        'voltage_state': 'Voltage Sensor State',
        'voltage_reading': 'Voltage Reading (V)'
    }
    
    # 输出基础目录
    base_output_dir = '/mnt/raid/liuhongbin/job_analysis/job_analysis/Server_analysis/Server/Server_Power'
    
    # 处理每个文件
    for i, (metric_key, file_path) in enumerate(power_files.items()):
        print(f"[{i+1}/6] 处理文件: {os.path.basename(file_path)}")
        
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
    
    print("所有分析完成.")

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
                
                # 检查必要的列
                if 'time' not in df.columns or 'value' not in df.columns:
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
                        pbar.update(1)
                        continue
                    
                    # 重命名列以便后续处理
                    df = df.rename(columns={time_col: 'time', value_col: 'value'})
                
                # 确保时间列是datetime格式
                if not pd.api.types.is_datetime64_any_dtype(df['time']):
                    df['time'] = pd.to_datetime(df['time'], errors='coerce')
                
                # 确保value列是数值类型
                if not pd.api.types.is_numeric_dtype(df['value']):
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                
                # 删除无效数据
                df = df.dropna()
                
                # 添加到对应的群组
                if group not in group_data:
                    group_data[group] = []
                
                group_data[group].append(df)
                
            except Exception:
                pass
            
            finally:
                pbar.update(1)
    
    # 整合每个群组的数据
    for group in list(group_data.keys()):
        if not group_data[group]:
            del group_data[group]
            continue
            
        # 合并同一群组的所有DataFrame
        combined_df = pd.concat(group_data[group])
        
        # 添加日期列
        combined_df['date'] = combined_df['time'].dt.date
        
        # 按日期分组计算平均值
        daily_avg = combined_df.groupby('date')['value'].mean().reset_index()
        
        # 用每日平均值替换原始数据
        group_data[group] = daily_avg
    
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
    return 'other'

def generate_group_comparison_plot(group_data, metric_name, output_dir):
    """生成服务器群组趋势对比图"""
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
    
    for group, daily_data in group_data.items():
        boxplot_data.append(daily_data['value'])
        display_name = group_display_names.get(group, group)
        
        # 添加适当的单位，根据指标类型
        if 'Volt' in metric_name:
            boxplot_labels.append(f"{display_name}\n(avg: {daily_data['value'].mean():.2f}V)")
        elif 'Power' in metric_name or 'Watt' in metric_name:
            boxplot_labels.append(f"{display_name}\n(avg: {daily_data['value'].mean():.1f}W)")
        else:
            boxplot_labels.append(f"{display_name}\n(avg: {daily_data['value'].mean():.2f})")
    
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
        patch.set_facecolor(color_map[group])
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
        f.write(f"服务器能耗分析报告 - {metric_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("各群组能耗统计:\n")
        f.write("-" * 50 + "\n")
        
        for group, daily_data in group_data.items():
            display_name = group_display_names.get(group, group)
            f.write(f"\n● {display_name}:\n")
            
            # 根据指标类型添加适当的单位
            unit = "W" if ("Power" in metric_name or "Watt" in metric_name) else "V" if "Volt" in metric_name else ""
            
            f.write(f"  平均值: {daily_data['value'].mean():.2f} {unit}\n")
            f.write(f"  最大值: {daily_data['value'].max():.2f} {unit}\n")
            f.write(f"  最小值: {daily_data['value'].min():.2f} {unit}\n")
            f.write(f"  标准差: {daily_data['value'].std():.2f} {unit}\n")
            f.write(f"  变异系数: {(daily_data['value'].std() / daily_data['value'].mean()) * 100:.2f}%\n")

if __name__ == "__main__":
    try:
        start_time = time.time()
        analyze_power_data()
        end_time = time.time()
        print(f"总执行时间: {end_time - start_time:.2f} 秒")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)