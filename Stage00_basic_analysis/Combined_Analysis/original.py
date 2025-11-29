#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
资源与任务整合分析工具 - 将任务提交数据与服务器资源监控数据进行匹配和分析
"""

import os
import pandas as pd
import numpy as np
import re
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import sys
import time
import concurrent.futures
import multiprocessing
from functools import partial
import threading
from collections import defaultdict

# 忽略警告
warnings.filterwarnings('ignore')

# 设置matplotlib字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 预定义要排除的国产服务器模式
EXCLUDE_PATTERNS = [
    r'.*推理服务器.*',
    r'.*训练服务器.*',
    r'.*training.*',
    r'.*reasoning.*',
    r'.*inferencing.*'
]

# 服务器组映射
SERVER_GROUP_PATTERNS = {
    'cpu1': r'^cpu1-\d+$',
    'cpu2': r'^cpu2-\d+$', 
    'cpu3': r'^cpu3-\d+$',
    'gpu1': r'^gpu1-\d+$',
    'gpu2': r'^gpu2-\d+$',
    'gpu3': r'^gpu3-\d+$',
    'bigmem': r'^bigmen-\d+$',
}

# 资源数据文件路径
DATA_PATHS = {
    # 任务提交记录
    'job_data': '/mnt/raid/liuhongbin/job_analysis/job_analysis/User_analysis/llm_results/analysis_20250309_115130/all_classifications_20250309_115130.csv',
    
    # CPU数据
    'cpu_total': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_CPU使用率_总使用率（%）_20250221_120300.xlsx',
    'cpu_system': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_CPU使用率_系统使用率（%）_20250221_104358.xlsx',
    
    # GPU数据
    'gpu_avg': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx',
    'gpu_util': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx',
    
    # 内存数据
    'mem_percent': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_内存使用率_使用率（%）_20250221_085433.xlsx',
    'mem_used': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_内存使用率_已用（G）_20250221_085041.xlsx',
    
    # 负载数据
    'load_1min': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_系统平均负载_1分钟负载（%）_20250220_160120.xlsx',
    'load_15min': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_系统平均负载_15分钟负载（%）_20250220_160842.xlsx',
    
    # 能源与温度数据
    'power': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器能耗_PowerReadingInWatts_20250221_102521.xlsx',
    'temperature': '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_服务器温度_Temperatures_20250221_111956.xlsx'
}

# 输出目录
OUTPUT_DIR = '/mnt/raid/liuhongbin/job_analysis/job_analysis/Combined_Analysis/results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 配置多进程/多线程参数
MAX_WORKERS_PROCESSES = max(1, multiprocessing.cpu_count() - 1)  # 保留一个核心给系统
MAX_WORKERS_THREADS = max(2, multiprocessing.cpu_count() * 2)    # IO密集型任务用更多线程

# 在文件顶部常量定义区域添加
JOB_COLUMNS = ['job_id', 'user_id', 'exec_host', 'queue_time', 'duration_time', 'server', 'resource_exists']

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='资源与任务整合分析工具')
    parser.add_argument('--test', action='store_true', help='是否运行测试模式（只使用前N行数据）')
    parser.add_argument('--rows', type=int, default=100, 
                      help='测试模式下每个文件读取的行数')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                      help='输出目录路径')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS_PROCESSES,
                      help='并行处理的工作进程数')
    parser.add_argument('--sequential', action='store_true', 
                      help='禁用并行处理，使用顺序执行模式')
    return parser.parse_args()

def parse_datetime(dt_str):
    """解析日期时间字符串"""
    if pd.isna(dt_str):
        return None
    
    try:
        # 尝试多种日期格式
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"]:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        
        # 如果是Unix时间戳
        if isinstance(dt_str, (int, float)) or (isinstance(dt_str, str) and dt_str.isdigit()):
            return datetime.fromtimestamp(float(dt_str))
        
        # 默认使用pandas解析
        return pd.to_datetime(dt_str)
    
    except Exception:
        return None

def should_exclude_server(server_name):
    """检查服务器名称是否应该被排除"""
    if server_name is None:
        return True
        
    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, server_name, re.IGNORECASE):
            return True
    return False

def identify_server_group(server_name):
    """识别服务器所属的组"""
    if server_name is None:
        return "unknown"
        
    for group, pattern in SERVER_GROUP_PATTERNS.items():
        if re.match(pattern, server_name, re.IGNORECASE):
            return group
    
    # 根据名称前缀进行猜测
    if server_name.startswith('cpu'):
        return 'cpu'
    elif server_name.startswith('gpu'):
        return 'gpu'
    elif server_name.startswith('bigmem'):
        return 'bigmem'
    
    return "other"

def extract_server_id_from_exec_hosts(exec_hosts):
    """从exec_hosts字段提取服务器ID列表"""
    if pd.isna(exec_hosts) or exec_hosts == '':
        return []
    
    # 处理可能的格式: "server1 server2 server3" 或 "server1:8 server2:4"
    servers = []
    
    try:
        # 分割字符串
        hosts = str(exec_hosts).split()
        
        for host in hosts:
            # 处理可能带有端口/槽数的格式，如 "server1:8"
            server = host.split(':')[0]
            
            # 过滤掉不需要的服务器
            if not should_exclude_server(server):
                servers.append(server)
    except Exception:
        pass
    
    return servers

def read_job_data(file_path, test_mode=False, max_rows=100):
    """读取任务提交数据"""
    filename = os.path.basename(file_path)
    print(f"\n正在读取任务数据: {filename}")
    start_time = time.time()
    
    try:
        # 读取CSV文件，测试模式只读取前N行
        if test_mode:
            print(f"  测试模式: 只读取前 {max_rows} 行数据")
            df = pd.read_csv(file_path, nrows=max_rows)
        else:
            df = pd.read_csv(file_path)
            
        print(f"  原始任务数据读取完成。记录数: {len(df)}")
        
        # 检查必要列
        required_cols = ['job_id', 'user_id', 'submit_time', 'start_time', 'end_time']
        for col in required_cols:
            if col not in df.columns:
                print(f"警告: 任务数据缺少必要列 '{col}'")
                return None
        
        # 转换时间字段为datetime
        print("  转换时间字段...")
        time_cols = ['submit_time', 'start_time', 'end_time']
        
        for col in tqdm(time_cols, desc="时间字段转换"):
            df[col] = df[col].apply(parse_datetime)
        
        # 提取服务器ID列表
        print("  提取服务器ID列表...")
        df['server_list'] = df['exec_hosts'].apply(extract_server_id_from_exec_hosts)
        
        # 计算排队时间和执行时间（分钟）
        print("  计算任务相关时间...")
        df['queue_time'] = ((df['start_time'] - df['submit_time']).dt.total_seconds() / 60)
        df['run_time'] = ((df['end_time'] - df['start_time']).dt.total_seconds() / 60)
        
        # 处理请求的资源
        df['requested_processors'] = df['num_processors'].fillna(0).astype(int)
        df['requested_gpu'] = df['gpu_num'].fillna(0).astype(int)
        
        # 过滤无效任务（结束时间为空或负的运行时间）
        valid_jobs = df.dropna(subset=['end_time']).copy()
        valid_jobs = valid_jobs[valid_jobs['run_time'] > 0]
        
        elapsed_time = time.time() - start_time
        print(f"任务数据处理完成。耗时: {elapsed_time:.2f}秒")
        print(f"有效任务总数: {len(valid_jobs)}")
        return valid_jobs
    
    except Exception as e:
        print(f"读取任务数据时出错: {str(e)}")
        return None

def _read_single_sheet_safe(excel_path, sheet_name, resource_type, test_mode=False, rows_limit=None):
    """安全地读取单个工作表的数据 - 使用文件路径而不是Excel对象，避免并发访问"""
    try:
        # 决定是否限制读取行数
        nrows = rows_limit if test_mode and rows_limit else None
        
        # 读取工作表数据 - 直接从文件路径读取特定工作表
        df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=nrows)
        
        # 检查必要列
        required_cols = ['timestamp', 'value']
        for col in required_cols:
            if col not in df.columns:
                print(f"  工作表 {sheet_name} 缺少必要列: {col}")
                return None
        
        # 确保timestamp列为日期时间类型
        if df['timestamp'].dtype != 'datetime64[ns]':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except:
                print(f"  工作表 {sheet_name} 时间戳列转换失败")
                return None
        
        # 确保value列为数值类型
        if not pd.api.types.is_numeric_dtype(df['value']):
            try:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
            except:
                print(f"  工作表 {sheet_name} 值列转换失败")
                return None
        
        # 添加指标名称列(如果不存在)
        if 'metric' not in df.columns:
            df['metric'] = resource_type
        
        return df
    except Exception as e:
        print(f"  读取工作表 {sheet_name} 时出错: {str(e)}")
        return None

def _process_resource_file(file_path, resource_type, test_mode=False, rows_limit=None):
    """处理单个资源文件的所有工作表 - 每个文件一个线程，避免并发访问冲突"""
    filename = os.path.basename(file_path)
    print(f"处理{resource_type}文件: {filename}")
    
    try:
        # 先安全地获取工作表列表
        try:
            # 使用pandas安全地获取工作表名称列表
            sheet_names = pd.ExcelFile(file_path).sheet_names
        except Exception as e:
            print(f"  获取工作表列表失败: {str(e)}")
            return {}
            
        # 筛选要处理的工作表
        valid_sheets = []
        for sheet in sheet_names:
            # 跳过不符合要求的工作表
            should_exclude = False
            for pattern in EXCLUDE_PATTERNS:
                if re.match(pattern, sheet):
                    should_exclude = True
                    break
            
            if not should_exclude:
                valid_sheets.append(sheet)
        
        if test_mode and rows_limit:
            print(f"  测试模式: 只处理每个工作表的前 {rows_limit} 行")
        
        # 从此处开始，我们顺序处理每个工作表，避免并行访问同一个Excel文件
        server_data = {}
        
        # 顺序处理工作表
        progress_bar = tqdm(valid_sheets, 
                           desc=f"{resource_type} 工作表", 
                           position=0,
                           leave=True)
        
        for sheet_name in progress_bar:
            try:
                # 直接从文件路径读取特定工作表
                data = _read_single_sheet_safe(file_path, sheet_name, resource_type, test_mode, rows_limit)
                
                # 确保结果是有效的DataFrame
                if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                    # 对服务器名称进行标准化处理
                    normalized_name = _normalize_server_name(sheet_name)
                    if normalized_name:  # 确保标准化后的名称不为空
                        server_data[normalized_name] = data
            except Exception as e:
                print(f"  处理工作表 {sheet_name} 时出错: {str(e)}")
            
        # 添加详细日志，显示读取到的服务器名称
        if server_data:
            print(f"  成功读取 {len(server_data)} 台服务器的{resource_type}数据")
            if len(server_data) <= 10:  # 如果服务器数量较少，全部显示
                print(f"  服务器列表: {', '.join(server_data.keys())}")
            else:  # 仅显示前5个和后5个
                server_names = list(server_data.keys())
                print(f"  服务器示例: {', '.join(server_names[:5])} ... {', '.join(server_names[-5:])}")
        else:
            print(f"  警告: 未找到任何有效的服务器{resource_type}数据")
        
        return server_data
    
    except Exception as e:
        print(f"处理{resource_type}文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

def read_excel_resource_data(file_path, resource_type, test_mode=False, rows_limit=None, use_parallel=True):
    """读取Excel资源监控数据 - 外层处理函数"""
    # 直接调用单个文件处理函数，不再进行内部并行化
    return _process_resource_file(file_path, resource_type, test_mode, rows_limit)

def _normalize_server_name(server_name):
    """标准化服务器名称，确保与任务记录中的格式一致"""
    # 去除可能的空格和特殊字符
    name = server_name.strip().lower()
    
    # 根据需要添加特定的转换规则
    # 例如，如果存在"node-"前缀但任务记录中没有，则去除
    # 如果Excel中是"bigmen-1"但任务记录中是"bigmem-1"，进行修正
    if name.startswith('bigmen-'):
        name = 'bigmem-' + name[7:]
    
    return name

def parse_exec_hosts(exec_hosts):
    """精确解析exec_hosts字段"""
    if pd.isna(exec_hosts) or not exec_hosts.strip():
        return []
    
    # 标准化处理
    hosts = []
    seen = set()
    for part in re.split(r'[,\s;|]+', exec_hosts.strip()):
        host = part.split(':')[0].lower().strip()
        if host and host not in seen:
            seen.add(host)
            hosts.append(host)
    return hosts

def _auto_generate_group_mapping(server_names):
    """根据服务器名称自动生成实例到组的映射"""
    group_map = {}
    pattern = re.compile(r'^([a-zA-Z]+)(\d+)(-\d+)?$')
    
    for name in server_names:
        # 尝试匹配常见命名模式
        match = pattern.match(name)
        if match:
            base = match.group(1).lower()
            group_num = match.group(2)
            instance_num = match.group(3) or ''
            
            # 生成组名 (如gpu1、cpu2)
            group_name = f"{base}{group_num}"
            
            # 特殊处理bigmem类型
            if 'bigmem' in name.lower():
                group_name = 'bigmem'
            
            group_map[name] = group_name
        else:
            # 无法自动分组的保持原名称
            group_map[name] = name
    return group_map

def analyze_resource_usage(matched_df, resource_groups):
    """分析资源使用情况（分实例和组两个层级）"""
    print("\n开始资源使用分析...")
    
    # 实例级分析
    instance_analysis = matched_df.groupby(['server_instance', 'metric']).agg({
        'value': ['mean', 'max', 'min'],
        'job_id': 'count'
    }).reset_index()
    instance_analysis.columns = ['Server Instance', 'Metric', 'Avg Usage', 'Peak Usage', 'Min Usage', 'Job Count']
    
    # 组级分析
    matched_df['server_group'] = matched_df['server_instance'].map(resource_groups)
    group_analysis = matched_df.groupby(['server_group', 'metric']).agg({
        'value': ['mean', 'max', 'min'],
        'job_id': 'count'
    }).reset_index()
    group_analysis.columns = ['Server Group', 'Metric', 'Avg Usage', 'Peak Usage', 'Min Usage', 'Job Count']
    
    return instance_analysis, group_analysis

def match_jobs_with_resources(jobs_df, resources, test_mode=False, rows_limit=None, use_parallel=False):
    """将任务数据与资源监控数据匹配（分实例和组两个层级）"""
    print("开始任务与资源数据匹配...")
    match_start = time.time()
    
    # 获取所有资源服务器实例
    all_resource_servers = set()
    for resource_data in resources.values():
        all_resource_servers.update(resource_data.keys())
    
    # 自动生成组映射
    group_mapping = _auto_generate_group_mapping(all_resource_servers)
    inverse_group_mapping = defaultdict(list)
    for instance, group in group_mapping.items():
        inverse_group_mapping[group].append(instance)
    
    # 处理任务数据
    matched_records = []
    missing_servers = set()
    
    for _, job in tqdm(jobs_df.iterrows(), total=len(jobs_df), desc="匹配任务"):
        job_servers = parse_exec_hosts(job['exec_hosts'])
        valid_servers = []
        
        for s in job_servers:
            # 第一层：精确匹配实例
            if s in all_resource_servers:
                valid_servers.append(s)
                continue
                
            # 第二层：尝试匹配组
            group_servers = inverse_group_mapping.get(s, [])
            valid_group_servers = [server for server in group_servers if server in all_resource_servers]
            
            if valid_group_servers:
                valid_servers.extend(valid_group_servers)
            else:
                missing_servers.add(s)
        
        # 记录匹配结果
        if valid_servers:
            for server in valid_servers:
                for metric, resource_data in resources.items():
                    if server in resource_data:
                        server_df = resource_data[server]
                        # 时间窗口匹配逻辑（保持原有实现）
                        # ... 原有时间匹配代码 ...
                        
                        # 添加组信息
                        record = {
                            'job_id': job['job_id'],
                            'user_id': job['user_id'],
                            'submit_time': job['submit_time'],
                            'start_time': job['start_time'],
                            'end_time': job['end_time'],
                            'server_instance': server,
                            'server_group': group_mapping[server],
                            'metric': metric,
                            'value': server_df['value'].iloc[0],
                            'timestamp': server_df['timestamp'].iloc[0],
                            'requested_processors': job['requested_processors'],
                            'requested_gpu': job['requested_gpu']
                        }
                        matched_records.append(record)
    
    # 生成最终DataFrame
    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        
        # 输出匹配统计
        print(f"\n匹配统计:")
        print(f"- 总任务数: {len(jobs_df)}")
        print(f"- 成功匹配的任务数: {len(matched_df['job_id'].unique())}")
        print(f"- 未匹配的服务器引用数: {len(missing_servers)}")
        if missing_servers:
            print(f"- 示例未匹配服务器: {sorted(missing_servers)[:10]}...")
        
        return matched_df
    else:
        print("错误: 没有匹配到任何有效数据")
        return None

def verify_data_consistency(matched_df):
    """验证数据一致性，用于测试模式"""
    print("\n==== 数据一致性检查 ====")
    
    if matched_df is None or matched_df.empty:
        print("错误: 没有匹配数据，无法进行一致性检查")
        return False
    
    # 检查关键列是否存在
    required_cols = ['timestamp', 'job_id', 'user_id', 'server', 'server_group']
    missing_cols = [col for col in required_cols if col not in matched_df.columns]
    
    if missing_cols:
        print(f"错误: 缺少关键列: {', '.join(missing_cols)}")
        return False
    
    # 检查资源数据列
    resource_cols = []
    for col in matched_df.columns:
        if col.startswith(('cpu_', 'gpu_', 'mem_', 'load_')) or col in ['power', 'temperature']:
            resource_cols.append(col)
    
    if not resource_cols:
        print("警告: 未找到任何资源数据列")
        return False
    
    print(f"数据列检查通过，找到 {len(resource_cols)} 个资源数据列")
    
    # 检查服务器组分布
    server_groups = matched_df['server_group'].value_counts()
    print("\n服务器组分布:")
    for group, count in server_groups.items():
        print(f"  - {group}: {count} 条记录")
    
    # 检查时间范围
    time_range = matched_df['timestamp'].max() - matched_df['timestamp'].min()
    time_hours = time_range.total_seconds() / 3600
    print(f"\n时间范围: {time_hours:.1f} 小时")
    
    # 检查是否有缺失值
    na_count = matched_df[resource_cols].isna().sum()
    if na_count.sum() > 0:
        print("\n警告: 资源数据存在缺失值:")
        for col, count in na_count[na_count > 0].items():
            print(f"  - {col}: {count} 缺失值")
    else:
        print("\n资源数据完整性检查通过，无缺失值")
    
    # 检查基本统计数据
    print("\n基本统计信息:")
    print(f"  - 总记录数: {len(matched_df)}")
    print(f"  - 唯一任务数: {matched_df['job_id'].nunique()}")
    print(f"  - 唯一服务器数: {matched_df['server'].nunique()}")
    print(f"  - 唯一用户数: {matched_df['user_id'].nunique()}")
    
    print("\n数据一致性检查完成!")
    return True

def analyze_combined_data(matched_df, output_dir):
    """分析整合后的数据集"""
    print("\n开始执行数据分析...")
    
    # 创建分析结果目录
    analysis_dir = os.path.join(output_dir, "analysis_results")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # 基础统计
    with open(os.path.join(analysis_dir, 'basic_stats.txt'), 'w') as f:
        f.write("=== 基础统计 ===\n")
        f.write(f"总记录数: {len(matched_df)}\n")
        f.write(f"涉及任务数: {matched_df['job_id'].nunique()}\n")
        f.write(f"服务器实例数量: {matched_df['server_instance'].nunique()}\n")
        f.write(f"服务器组数量: {matched_df['server_group'].nunique()}\n")
    
    # 时间范围分析（修改列名）
    time_analysis = matched_df.groupby(pd.Grouper(key='time_window_start', freq='D')).agg({
        'job_id': 'nunique',
        'server_instance': 'nunique'  # 修改列名
    }).reset_index()
    time_analysis.columns = ['日期', '任务数', '涉及服务器实例数']
    time_analysis.to_csv(os.path.join(analysis_dir, 'daily_stats.csv'), index=False)
    
    # 资源使用分析（修改列名）
    def _plot_usage(df, metric, title_prefix):
        plt.figure(figsize=(12, 6))
        
        # 实例级
        ax1 = plt.subplot(121)
        instance_stats = df.groupby('server_instance')[metric].mean().sort_values(ascending=False)
        instance_stats.head(10).plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title(f'{title_prefix} - 实例级TOP10')
        ax1.set_ylabel(metric)
        
        # 组级
        ax2 = plt.subplot(122)
        group_stats = df.groupby('server_group')[metric].mean().sort_values(ascending=False)
        group_stats.head(10).plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title(f'{title_prefix} - 组级TOP10')
        
        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, f'{metric}_usage.png'))
        plt.close()
    
    # 对每个指标生成图表
    for metric in ['cpu_usage', 'gpu_usage', 'mem_usage']:
        if metric in matched_df.columns:
            _plot_usage(matched_df, metric, metric.capitalize())
    
    print("数据分析完成。结果保存在:", analysis_dir)

def generate_resource_job_correlation_plots(matched_df, output_dir):
    """生成资源使用与任务数量关系图"""
    if matched_df is None or matched_df.empty:
        return
    
    print("生成资源使用与任务数量关系图...")
    
    # 创建关系图目录
    correlation_dir = os.path.join(output_dir, 'correlation_plots')
    os.makedirs(correlation_dir, exist_ok=True)
    
    # 按时间和服务器组聚合数据
    time_agg = matched_df.groupby([pd.Grouper(key='timestamp', freq='1H'), 'server_group']).agg({
        'job_id': 'nunique',  # 每个时间单位的任务数
        'cpu_total': 'mean',  # CPU平均使用率
        'gpu_util': 'mean',   # GPU平均使用率
        'load_1min': 'mean',  # 平均1分钟负载
        'power': 'mean'       # 平均功率
    }).reset_index()
    
    # 为每个服务器组绘制关系图
    for group in time_agg['server_group'].unique():
        group_data = time_agg[time_agg['server_group'] == group]
        
        # CPU使用率与任务数关系
        if 'cpu_total' in group_data.columns and not group_data['cpu_total'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.scatter(group_data['job_id'], group_data['cpu_total'], alpha=0.7)
            
            # 添加拟合曲线
            if len(group_data) > 1:
                try:
                    z = np.polyfit(group_data['job_id'], group_data['cpu_total'], 1)
                    p = np.poly1d(z)
                    plt.plot(group_data['job_id'], p(group_data['job_id']), "r--")
                    plt.text(0.95, 0.05, f"y = {z[0]:.2f}x + {z[1]:.2f}", 
                             transform=plt.gca().transAxes, ha='right')
                except Exception:
                    pass
            
            plt.title(f'{group} 服务器组 - 任务数量与CPU使用率关系')
            plt.xlabel('任务数量')
            plt.ylabel('CPU使用率 (%)')
            plt.grid(True)
            plt.tight_layout()
            
            cpu_corr_plot = os.path.join(correlation_dir, f'{group}_cpu_job_correlation.png')
            plt.savefig(cpu_corr_plot)
            plt.close()
        
        # GPU使用率与任务数关系（如果有GPU数据）
        if 'gpu_util' in group_data.columns and not group_data['gpu_util'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.scatter(group_data['job_id'], group_data['gpu_util'], alpha=0.7)
            
            # 添加拟合曲线
            if len(group_data) > 1:
                try:
                    z = np.polyfit(group_data['job_id'], group_data['gpu_util'], 1)
                    p = np.poly1d(z)
                    plt.plot(group_data['job_id'], p(group_data['job_id']), "r--")
                    plt.text(0.95, 0.05, f"y = {z[0]:.2f}x + {z[1]:.2f}", 
                             transform=plt.gca().transAxes, ha='right')
                except Exception:
                    pass
            
            plt.title(f'{group} 服务器组 - 任务数量与GPU使用率关系')
            plt.xlabel('任务数量')
            plt.ylabel('GPU使用率 (%)')
            plt.grid(True)
            plt.tight_layout()
            
            gpu_corr_plot = os.path.join(correlation_dir, f'{group}_gpu_job_correlation.png')
            plt.savefig(gpu_corr_plot)
            plt.close()
        
        # 负载与任务数关系
        if 'load_1min' in group_data.columns and not group_data['load_1min'].isna().all():
            plt.figure(figsize=(10, 6))
            plt.scatter(group_data['job_id'], group_data['load_1min'], alpha=0.7)
            
            # 添加拟合曲线
            if len(group_data) > 1:
                try:
                    z = np.polyfit(group_data['job_id'], group_data['load_1min'], 1)
                    p = np.poly1d(z)
                    plt.plot(group_data['job_id'], p(group_data['job_id']), "r--")
                    plt.text(0.95, 0.05, f"y = {z[0]:.2f}x + {z[1]:.2f}", 
                             transform=plt.gca().transAxes, ha='right')
                except Exception:
                    pass
            
            plt.title(f'{group} 服务器组 - 任务数量与系统负载关系')
            plt.xlabel('任务数量')
            plt.ylabel('系统负载 (%)')
            plt.grid(True)
            plt.tight_layout()
            
            load_corr_plot = os.path.join(correlation_dir, f'{group}_load_job_correlation.png')
            plt.savefig(load_corr_plot)
            plt.close()
    
    print(f"关系图已保存到: {correlation_dir}")

def generate_user_resource_analysis(matched_df, output_dir):
    """生成用户资源使用分析"""
    if matched_df is None or matched_df.empty:
        return
    
    # 创建用户分析目录
    user_dir = os.path.join(output_dir, 'user_analysis')
    os.makedirs(user_dir, exist_ok=True)
    
    # 按用户聚合数据
    user_agg = matched_df.groupby('user_id').agg({
        'job_id': 'nunique',  # 每个用户的任务数
        'window_coverage': 'sum',  # 总的时间窗口覆盖
        'cpu_total': 'mean',  # CPU平均使用率
        'gpu_util': 'mean',   # GPU平均使用率
        'load_1min': 'mean',  # 平均1分钟负载
        'power': 'mean',      # 平均功率
        'requested_processors': 'mean',  # 平均请求CPU数
        'requested_gpu': 'mean'  # 平均请求GPU数
    }).reset_index()
    
    # 对用户按任务数量排序
    user_agg_sorted = user_agg.sort_values('job_id', ascending=False)
    
    # 保存用户资源使用统计信息
    user_stats_file = os.path.join(user_dir, 'user_resource_stats.csv')
    user_agg_sorted.to_csv(user_stats_file, index=False)
    
    # 绘制前20个用户的任务数量
    plt.figure(figsize=(14, 8))
    
    top_users = user_agg_sorted.head(20)
    plt.bar(top_users['user_id'].astype(str), top_users['job_id'])
    
    plt.title('前20名用户的任务数量')
    plt.xlabel('用户ID')
    plt.ylabel('任务数量')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    
    user_jobs_plot = os.path.join(user_dir, 'top_users_by_jobs.png')
    plt.savefig(user_jobs_plot)
    plt.close()
    
    # 绘制用户的平均CPU和GPU请求
    plt.figure(figsize=(14, 8))
    
    width = 0.35
    x = np.arange(len(top_users))
    
    plt.bar(x - width/2, top_users['requested_processors'], width, label='平均CPU请求')
    plt.bar(x + width/2, top_users['requested_gpu'], width, label='平均GPU请求')
    
    plt.title('前20名用户的平均资源请求')
    plt.xlabel('用户ID')
    plt.ylabel('平均请求资源数')
    plt.xticks(x, top_users['user_id'].astype(str), rotation=45)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    
    user_requests_plot = os.path.join(user_dir, 'top_users_resource_requests.png')
    plt.savefig(user_requests_plot)
    plt.close()
    
    print(f"用户资源使用分析已保存到: {user_dir}")

def generate_server_group_patterns(matched_df, output_dir):
    """生成服务器组使用模式分析"""
    if matched_df is None or matched_df.empty:
        return
    
    print("生成服务器组使用模式分析...")
    
    # 创建服务器组分析目录
    group_dir = os.path.join(output_dir, 'server_group_analysis')
    os.makedirs(group_dir, exist_ok=True)
    
    # 添加星期和小时列
    matched_df['dayofweek'] = matched_df['timestamp'].dt.dayofweek
    matched_df['hour'] = matched_df['timestamp'].dt.hour
    
    # 按服务器组、星期和小时聚合数据
    time_pattern_agg = matched_df.groupby(['server_group', 'dayofweek', 'hour']).agg({
        'job_id': 'nunique',  # 每个时间单位的任务数
        'cpu_total': 'mean',  # CPU平均使用率
        'gpu_util': 'mean',   # GPU平均使用率
        'load_1min': 'mean',  # 平均1分钟负载
        'power': 'mean'       # 平均功率
    }).reset_index()
    
    # 为每个服务器组创建热图
    for group in time_pattern_agg['server_group'].unique():
        group_data = time_pattern_agg[time_pattern_agg['server_group'] == group]
        
        # 创建星期/小时热图数据
        job_pivot = group_data.pivot(index='hour', columns='dayofweek', values='job_id')
        cpu_pivot = group_data.pivot(index='hour', columns='dayofweek', values='cpu_total')
        
        # 检查是否有GPU数据
        has_gpu = 'gpu_util' in group_data.columns and not group_data['gpu_util'].isna().all()
        if has_gpu:
            gpu_pivot = group_data.pivot(index='hour', columns='dayofweek', values='gpu_util')
        
        # 检查是否有负载数据
        has_load = 'load_1min' in group_data.columns and not group_data['load_1min'].isna().all()
        if has_load:
            load_pivot = group_data.pivot(index='hour', columns='dayofweek', values='load_1min')
        
        # 检查是否有功率数据
        has_power = 'power' in group_data.columns and not group_data['power'].isna().all()
        if has_power:
            power_pivot = group_data.pivot(index='hour', columns='dayofweek', values='power')
        
        # 绘制任务数热图
        plt.figure(figsize=(10, 8))
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        
        ax = plt.gca()
        im = ax.imshow(job_pivot.values, cmap='YlOrRd', aspect='auto')
        
        plt.colorbar(im, label='任务数量')
        plt.title(f'{group} 服务器组 - 任务数量按星期/小时分布')
        plt.xlabel('星期')
        plt.ylabel('小时')
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(weekdays)))
        ax.set_xticklabels(weekdays)
        ax.set_yticks(np.arange(24))
        ax.set_yticklabels(np.arange(24))
        
        plt.tight_layout()
        job_heatmap = os.path.join(group_dir, f'{group}_job_heatmap.png')
        plt.savefig(job_heatmap)
        plt.close()
        
        # 绘制CPU使用率热图
        plt.figure(figsize=(10, 8))
        
        ax = plt.gca()
        im = ax.imshow(cpu_pivot.values, cmap='YlOrRd', aspect='auto')
        
        plt.colorbar(im, label='CPU使用率 (%)')
        plt.title(f'{group} 服务器组 - CPU使用率按星期/小时分布')
        plt.xlabel('星期')
        plt.ylabel('小时')
        
        # 设置坐标轴标签
        ax.set_xticks(np.arange(len(weekdays)))
        ax.set_xticklabels(weekdays)
        ax.set_yticks(np.arange(24))
        ax.set_yticklabels(np.arange(24))
        
        plt.tight_layout()
        cpu_heatmap = os.path.join(group_dir, f'{group}_cpu_heatmap.png')
        plt.savefig(cpu_heatmap)
        plt.close()
        
        # 如果有GPU数据，绘制GPU使用率热图
        if has_gpu:
            plt.figure(figsize=(10, 8))
            
            ax = plt.gca()
            im = ax.imshow(gpu_pivot.values, cmap='YlOrRd', aspect='auto')
            
            plt.colorbar(im, label='GPU使用率 (%)')
            plt.title(f'{group} 服务器组 - GPU使用率按星期/小时分布')
            plt.xlabel('星期')
            plt.ylabel('小时')
            
            # 设置坐标轴标签
            ax.set_xticks(np.arange(len(weekdays)))
            ax.set_xticklabels(weekdays)
            ax.set_yticks(np.arange(24))
            ax.set_yticklabels(np.arange(24))
            
            plt.tight_layout()
            gpu_heatmap = os.path.join(group_dir, f'{group}_gpu_heatmap.png')
            plt.savefig(gpu_heatmap)
            plt.close()
        
        # 如果有负载数据，绘制系统负载热图
        if has_load:
            plt.figure(figsize=(10, 8))
            
            ax = plt.gca()
            im = ax.imshow(load_pivot.values, cmap='YlOrRd', aspect='auto')
            
            plt.colorbar(im, label='系统负载 (%)')
            plt.title(f'{group} 服务器组 - 系统负载按星期/小时分布')
            plt.xlabel('星期')
            plt.ylabel('小时')
            
            # 设置坐标轴标签
            ax.set_xticks(np.arange(len(weekdays)))
            ax.set_xticklabels(weekdays)
            ax.set_yticks(np.arange(24))
            ax.set_yticklabels(np.arange(24))
            
            plt.tight_layout()
            load_heatmap = os.path.join(group_dir, f'{group}_load_heatmap.png')
            plt.savefig(load_heatmap)
            plt.close()
    
    print(f"服务器组使用模式分析已保存到: {group_dir}")

def generate_resource_efficiency_analysis(matched_df, output_dir):
    """生成资源效率分析"""
    if matched_df is None or matched_df.empty:
        return
    
    # 创建资源效率分析目录
    efficiency_dir = os.path.join(output_dir, 'resource_efficiency')
    os.makedirs(efficiency_dir, exist_ok=True)
    
    # 计算资源请求与实际使用效率
    matched_df['cpu_efficiency'] = matched_df['cpu_total'] / 100  # 实际使用率(0-1)
    
    # 有GPU的任务
    gpu_jobs = matched_df[matched_df['requested_gpu'] > 0].copy()
    if not gpu_jobs.empty and 'gpu_util' in gpu_jobs.columns:
        gpu_jobs['gpu_efficiency'] = gpu_jobs['gpu_util'] / 100  # 实际使用率(0-1)
    
    # 按任务类型聚合资源效率
    if 'application' in matched_df.columns:
        app_efficiency = matched_df.groupby('application').agg({
            'cpu_efficiency': 'mean',
            'job_id': 'nunique',
            'requested_processors': 'mean',
            'window_coverage': 'sum'
        }).reset_index()
        
        # 保存应用程序资源效率
        app_eff_file = os.path.join(efficiency_dir, 'application_efficiency.csv')
        app_efficiency.to_csv(app_eff_file, index=False)
        
        # 绘制应用程序CPU效率
        top_apps = app_efficiency.sort_values('job_id', ascending=False).head(10)
        
        plt.figure(figsize=(12, 8))
        plt.bar(top_apps['application'], top_apps['cpu_efficiency'] * 100)
        plt.title('前10个应用程序的CPU使用效率')
        plt.xlabel('应用程序')
        plt.ylabel('平均CPU使用效率 (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        
        app_cpu_eff_plot = os.path.join(efficiency_dir, 'top_apps_cpu_efficiency.png')
        plt.savefig(app_cpu_eff_plot)
        plt.close()
    
    # 计算每小时的资源利用率
    hourly_usage = matched_df.groupby(['server_group', pd.Grouper(key='timestamp', freq='1H')]).agg({
        'job_id': 'nunique',
        'cpu_total': 'mean',
        'requested_processors': 'sum',
        'window_coverage': 'sum'
    }).reset_index()
    
    # 添加请求与实际使用比较
    hourly_usage['cpu_saturation'] = hourly_usage['requested_processors'] / hourly_usage['window_coverage']
    hourly_usage['cpu_usage'] = hourly_usage['cpu_total']
    
    # 保存每小时资源利用数据
    hourly_file = os.path.join(efficiency_dir, 'hourly_resource_usage.csv')
    hourly_usage.to_csv(hourly_file, index=False)
    
    # 为每个服务器组绘制资源饱和度与实际使用对比
    for group in hourly_usage['server_group'].unique():
        group_data = hourly_usage[hourly_usage['server_group'] == group]
        
        plt.figure(figsize=(14, 8))
        
        plt.plot(group_data['timestamp'], group_data['cpu_saturation'], 
                 label='请求CPU饱和度', color='blue', marker='o', alpha=0.7, markersize=3)
        plt.plot(group_data['timestamp'], group_data['cpu_usage'], 
                 label='实际CPU使用率', color='red', marker='x', alpha=0.7, markersize=3)
        
        plt.title(f'{group} 服务器组 - CPU请求vs实际使用')
        plt.xlabel('时间')
        plt.ylabel('CPU使用率/饱和度')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(200, group_data['cpu_saturation'].max() * 1.1))
        
        # 添加100%参考线
        plt.axhline(y=100, color='green', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        group_eff_plot = os.path.join(efficiency_dir, f'{group}_cpu_efficiency.png')
        plt.savefig(group_eff_plot)
        plt.close()
    
    print(f"资源效率分析已保存到: {efficiency_dir}")

def generate_time_windows(resources):
    """生成统一的时间窗口序列"""
    all_times = []
    for resource_type in resources.values():
        for server_data in resource_type.values():
            if not server_data.empty:
                all_times.extend(server_data['time_window_start'].tolist())
    
    if not all_times:
        return pd.DataFrame()
    
    min_time = pd.to_datetime(min(all_times))
    max_time = pd.to_datetime(max(all_times))
    return pd.DataFrame({
        'time_window_start': pd.date_range(start=min_time.floor('H'), 
                                         end=max_time.ceil('H'),
                                         freq='H')
    })

def merge_resource_data(base_df, resources):
    """合并资源数据（每个服务器单独行）"""
    merged = []
    for metric, servers in resources.items():
        for server, data in servers.items():
            df = data[['time_window_start', 'value']].copy()
            df['server'] = server
            df.rename(columns={'value': metric}, inplace=True)
            merged.append(df)
    
    # 合并所有资源指标
    resource_df = pd.concat(merged)
    return resource_df.groupby(['time_window_start', 'server']).first().reset_index()

def match_jobs_to_windows(jobs_df, time_windows, resources):
    """将任务匹配到时间窗口（每个服务器生成独立行）"""
    matched = []
    
    for _, job in jobs_df.iterrows():
        # 解析服务器列表
        servers = parse_exec_hosts(job['exec_hosts'])
        if not servers:
            continue
        
        # 计算任务时间
        job_start = pd.to_datetime(job['start_time'])
        job_end = pd.to_datetime(job['end_time'])
        
        # 找到重叠的时间窗口
        mask = (time_windows['time_window_start'] <= job_end) & \
               (time_windows['time_window_end'] > job_start)
        windows = time_windows[mask].copy()
        
        if windows.empty:
            continue
        
        # 为每个服务器生成记录
        for server in servers:
            # 复制时间窗口数据
            server_windows = windows.copy()
            server_windows['server'] = server
            
            # 添加任务信息
            server_windows['job_id'] = job['job_id']
            server_windows['user_id'] = job['user_id']
            server_windows['exec_host'] = job['exec_host']  # 保留原始字段
            server_windows['queue_time'] = (job_start - pd.to_datetime(job['submit_time'])).total_seconds()
            server_windows['duration_time'] = (job_end - job_start).total_seconds()
            
            # 添加资源存在性标记
            server_windows['resource_exists'] = server_windows.apply(
                lambda row: server in resources.get('cpu_total', {}), 
                axis=1
            )
            
            matched.append(server_windows)
    
    return pd.concat(matched) if matched else pd.DataFrame()

def integrate_data(resources, jobs_df):
    """整合资源和任务数据"""
    print("\n开始数据整合...")
    
    # 生成统一时间窗口
    time_windows = generate_time_windows(resources)
    if time_windows.empty:
        print("错误: 无法生成时间窗口")
        return None
    
    # 合并资源数据
    resource_df = merge_resource_data(time_windows.copy(), resources)
    
    # 合并任务数据
    jobs_matched = match_jobs_to_windows(jobs_df, time_windows, resources)
    
    # 最终合并
    final_df = pd.merge(resource_df, jobs_matched,
                       on=['time_window_start', 'server'],
                       how='left')
    
    # 列排序
    ordered_cols = [
        'time_window_start', 'time_window_end', 'server',
        'resource_exists', 'cpu_total', 'cpu_system', 'gpu_avg', 'gpu_util',
        'mem_percent', 'mem_used', 'power', 'temperature', 'load_1min', 'load_15min'
    ] + JOB_COLUMNS
    
    return final_df[ordered_cols]

def main():
    """主函数，执行整合分析流程"""
    try:
        # 记录全局开始时间
        global_start_time = time.time()
        
        # 解析命令行参数
        args = parse_args()
        
        # 更新输出目录
        global OUTPUT_DIR
        OUTPUT_DIR = args.output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # 使用并行还是顺序处理
        use_parallel = not args.sequential
        
        # 测试模式信息
        if args.test:
            print(f"=== 运行快速测试模式 (每个文件前 {args.rows} 行) ===")
            # 创建测试专用输出目录
            test_output = os.path.join(OUTPUT_DIR, f"test_rows{args.rows}")
            os.makedirs(test_output, exist_ok=True)
            OUTPUT_DIR = test_output
            print(f"测试输出目录: {OUTPUT_DIR}")
        
        print("开始资源与任务整合分析...")
        
        # 读取任务数据
        job_df = read_job_data(DATA_PATHS['job_data'], args.test, args.rows)
        if job_df is None or job_df.empty:
            print("错误: 任务数据为空，无法继续分析")
            return 1
        
        # 读取资源监控数据
        print("\n开始读取所有资源数据...")
        resources_start_time = time.time()
        
        resources = {}
        resource_types = [
            ('cpu_total', 'CPU总使用率'),
            ('cpu_system', 'CPU系统使用率'),
            ('gpu_avg', 'GPU平均使用率'),
            ('gpu_util', 'GPU使用率'),
            ('mem_percent', '内存使用率'),
            ('mem_used', '内存使用量'),
            ('load_1min', '1分钟负载'),
            ('load_15min', '15分钟负载'),
            ('power', '服务器功率'),
            ('temperature', '服务器温度')
        ]
        
        # 并行读取资源文件 - 每个文件一个线程，但文件内部顺序处理
        if use_parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(resource_types)) as executor:
                future_to_resource = {}
                for key, label in resource_types:
                    future = executor.submit(
                        _process_resource_file,
                        DATA_PATHS[key],
                        label,
                        args.test,
                        args.rows
                    )
                    future_to_resource[future] = key
                
                for future in tqdm(concurrent.futures.as_completed(future_to_resource), 
                                  total=len(resource_types),
                                  desc="读取资源文件"):
                    key = future_to_resource[future]
                    try:
                        resources[key] = future.result()
                    except Exception as e:
                        print(f"读取 {key} 数据出错: {str(e)}")
                        resources[key] = {}
        else:
            # 顺序读取
            for idx, (key, label) in enumerate(tqdm(resource_types, desc="读取资源数据")):
                # 显示处理进度和预计剩余时间
                if idx > 0:
                    elapsed = time.time() - resources_start_time
                    remaining = (elapsed / idx) * (len(resource_types) - idx)
                    print(f"处理进度: {idx}/{len(resource_types)} - 预计剩余: {int(remaining/60)}分{int(remaining%60)}秒")
                
                resources[key] = _process_resource_file(
                    DATA_PATHS[key], 
                    label, 
                    args.test, 
                    args.rows
                )
        
        # 计算资源数据读取总时间
        resources_time = time.time() - resources_start_time
        print(f"\n资源数据读取完成。耗时: {int(resources_time/60)}分{int(resources_time%60)}秒")
        
        # 检查是否至少有一种资源数据
        if all(not data for data in resources.values()):
            print("错误: 所有资源数据都为空，无法继续分析")
            return 1
        
        # 显示有效资源数据统计
        valid_resources = {k: v for k, v in resources.items() if v}
        print(f"\n成功读取的资源数据: {len(valid_resources)}/{len(resources)}")
        for resource_name, resource_data in valid_resources.items():
            server_count = len(resource_data)
            record_count = sum(len(data) for server, data in resource_data.items())
            print(f"  - {resource_name}: {server_count} 台服务器, 总计 {record_count} 条记录")
        
        # 将任务数据与资源数据关联
        print("\n开始任务与资源数据匹配...")
        match_start_time = time.time()
        matched_df = match_jobs_with_resources(job_df, resources, args.test, args.rows, use_parallel)
        match_time = time.time() - match_start_time
        print(f"匹配完成。耗时: {int(match_time/60)}分{int(match_time%60)}秒")
        
        if matched_df is None or matched_df.empty:
            print("错误: 无法匹配任务与资源数据")
            return 1
        
        # 生成整合数据集
        integrated_file = os.path.join(OUTPUT_DIR, 'integrated_data.csv')
        matched_df.to_csv(integrated_file, index=False)
        print(f"整合数据集已保存到: {integrated_file}")
        
        # 测试模式下验证数据一致性
        if args.test:
            if not verify_data_consistency(matched_df):
                print("警告: 数据一致性检查未通过，分析结果可能不可靠")
            else:
                print("数据一致性检查通过，继续分析...")
        
        # 执行各种分析
        print("\n开始执行数据分析...")
        analysis_start_time = time.time()
        
        print("1/2 基础分析中...")
        analyze_combined_data(matched_df, OUTPUT_DIR)
        
        print("2/2 资源效率分析中...")
        generate_resource_efficiency_analysis(matched_df, OUTPUT_DIR)
        
        analysis_time = time.time() - analysis_start_time
        print(f"分析完成。耗时: {int(analysis_time/60)}分{int(analysis_time%60)}秒")
        
        # 计算总执行时间
        total_time = time.time() - global_start_time
        print(f"\n===== 整个流程完成 =====")
        print(f"总执行时间: {int(total_time/60)}分{int(total_time%60)}秒")
        print(f"- 读取资源数据: {int(resources_time/60)}分{int(resources_time%60)}秒 ({resources_time/total_time*100:.1f}%)")
        print(f"- 匹配任务数据: {int(match_time/60)}分{int(match_time%60)}秒 ({match_time/total_time*100:.1f}%)")
        print(f"- 执行数据分析: {int(analysis_time/60)}分{int(analysis_time%60)}秒 ({analysis_time/total_time*100:.1f}%)")
        
        if args.test:
            print(f"\n=== 测试模式运行完成 ===")
            print(f"测试数据已保存到: {OUTPUT_DIR}")
        
        # 显示CPU核心使用情况和内存占用
        import psutil
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        print(f"\n系统资源使用情况:")
        print(f"- CPU使用率: {cpu_percent:.1f}%")
        print(f"- 内存使用率: {memory_percent:.1f}%")
        
    except Exception as e:
        print(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())