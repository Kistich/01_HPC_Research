#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
资源与任务整合分析工具 - 将任务提交数据与服务器资源监控数据进行匹配和分析
"""

# 导入标准库
import os
import re
import sys
import gc
import glob
import time
import json
import warnings
import traceback
import argparse
import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial, reduce

# 导入第三方库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

# 自定义日志记录类，同时输出到控制台和文件
class TeeLogger:
    """同时将日志输出到控制台和文件的自定义日志记录类"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, "w", encoding="utf-8")
        self.start_time = datetime.now()
    
    def write(self, message):
        self.terminal.write(message)
        # 添加时间戳到日志文件
        if message.strip():
            timestamp = datetime.now()
            elapsed = timestamp - self.start_time
            elapsed_str = f"{elapsed.total_seconds():.2f}s"
            self.log_file.write(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')} +{elapsed_str}] {message}")
        else:
            self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

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
    'bigman': r'^bigman-\d+$',
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


# 在文件顶部常量定义区域添加
JOB_COLUMNS = ['job_id', 'user_id', 'exec_hosts', 'queue_time', 'duration_time', 'server', 'resource_exists']


def parse_datetime(dt_str):
    """
    增强版日期时间字符串解析函数
    
    支持多种格式：
    1. 标准时间格式: "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M" 等
    2. Unix时间戳（整数或浮点数）
    3. 其他pandas支持的时间格式
    
    参数:
        dt_str: 日期时间字符串或数字
    
    返回:
        datetime对象或None（如果解析失败）
    """
    # 处理空值
    if pd.isna(dt_str) or dt_str == "":
        return None
    
    # 如果已经是datetime对象，直接返回
    if isinstance(dt_str, (datetime, pd.Timestamp)):
        return dt_str if isinstance(dt_str, datetime) else dt_str.to_pydatetime()
    
    try:
        # 统一处理字符串
        if isinstance(dt_str, str):
            dt_str = dt_str.strip()
            
            # 检查空字符串
            if not dt_str:
                return None
        
        # 处理Unix时间戳（整数、浮点数或数字字符串）
        if isinstance(dt_str, (int, float)) or (isinstance(dt_str, str) and dt_str.replace('.', '', 1).isdigit()):
            timestamp = float(dt_str)
            # 验证时间戳合理性（介于1970年和2100年之间）
            if 0 < timestamp < 4102444800:  # 2100年的时间戳
                return datetime.fromtimestamp(timestamp)
        
        # 尝试常见的日期格式
        if isinstance(dt_str, str):
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M", 
                         "%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
        
        # 最后尝试使用pandas自动检测格式
        result = pd.to_datetime(dt_str, errors='coerce')
        if pd.isnull(result):
            return None
        return result.to_pydatetime()
        
    except Exception as e:
        # 更详细的错误日志可以在调试时启用
        # import logging
        # logging.debug(f"Failed to parse datetime '{dt_str}': {str(e)}")
        return None

def should_exclude_server(server_name):
    """
    检查服务器名称是否应该被排除
    
    参数:
        server_name: 服务器名称字符串
        
    返回:
        True: 如果服务器应该被排除
        False: 如果服务器不应该被排除
    """
    # 处理空值
    if server_name is None or not isinstance(server_name, str) or not server_name.strip():
        return True
    
    # 标准化为小写以提高匹配效率
    server_name = server_name.lower().strip()
    
    # 排除明显无效的服务器名称
    if server_name in ["none", "null", "na", "n/a", "-", ""]:
        return True
        
    # 如果服务器名称只包含特殊字符，也应该排除
    if all(c in '[]()-_,;:|/' for c in server_name):
        return True
        
    # 检查是否匹配排除模式
    for pattern in EXCLUDE_PATTERNS:
        try:
            if re.search(pattern, server_name, re.IGNORECASE):
                return True
        except re.error:
            # 如果正则表达式有语法错误，记录日志并继续
            print(f"  警告: 无效的排除模式: '{pattern}'")
            continue
            
    return False

def identify_server_group(server_name):
    """
    识别服务器所属的组
    
    参数:
        server_name: 服务器名称字符串
        
    返回:
        服务器组名称字符串
    """
    # 处理空值或非字符串输入
    if server_name is None or not isinstance(server_name, str) or not server_name.strip():
        return "unknown"
    
    # 标准化为小写并去除两端空白
    server_name = server_name.lower().strip()
    
    # 使用定义的模式进行匹配
    for group, pattern in SERVER_GROUP_PATTERNS.items():
        try:
            if re.match(pattern, server_name, re.IGNORECASE):
                return group
        except re.error:
            # 如果正则表达式有语法错误，继续
            continue
    
    # 根据名称前缀进行智能猜测
    # 引入更精细的服务器类型检测逻辑
    if server_name.startswith('cpu'):
        # 分析cpu类型，例如cpu1, cpu2等
        match = re.match(r'cpu(\d+)', server_name)
        if match:
            return f'cpu{match.group(1)}'
        return 'cpu'
    elif server_name.startswith('gpu'):
        # 分析gpu类型，例如gpu1, gpu2等
        match = re.match(r'gpu(\d+)', server_name)
        if match:
            return f'gpu{match.group(1)}'
        return 'gpu'
    elif server_name.startswith('bigman') or server_name.startswith('bigmen'):
        # 标准化bigman/bigmen类型
        return 'bigman'
    elif 'node' in server_name:
        # 分析节点类型
        return 'node'
    
    # 如果没有匹配任何已知类型
    return "other"

def extract_server_id_from_exec_hosts(exec_hosts, row_num=None):
    """
    从exec_hosts字段中提取服务器ID列表
    支持各种服务器名称格式和范围表示法
    参数：
        exec_hosts: exec_hosts字段的内容
        row_num: 原始数据中的行号，用于调试和错误跟踪
    返回：
        解析出的服务器ID列表
    """
    if pd.isna(exec_hosts) or exec_hosts == '':
        return []
    
    try:
        # 标准化格式
        hosts_str = str(exec_hosts).strip()
        
        # 清理格式，移除可能的前缀
        hosts_str = hosts_str.replace('exec_hosts=', '')
        
        # 使用改进的parse_exec_hosts函数解析服务器列表
        return parse_exec_hosts(hosts_str, row_num)
        
    except Exception as e:
        error_msg = f"提取服务器ID时出错: {str(e)}"
        if row_num is not None:
            error_msg += f" （行号: {row_num}）"
        print(error_msg)
        return []

def parse_exec_hosts(exec_hosts, row_num=None):
    """
    精确解析exec_hosts字段，支持各种服务器名称格式和范围表示法
    支持的格式包括：
    1. 基本范围: 'cpu1-[41-43]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43']
    2. 逗号分隔的多个值: 'cpu1-[41,43,45]' -> ['cpu1-41', 'cpu1-43', 'cpu1-45']
    3. 复合范围: 'cpu1-[41-43,45,47-48]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43', 'cpu1-45', 'cpu1-47', 'cpu1-48']
    4. 多个服务器: 'cpu1-[41-43],cpu2-[1-3]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43', 'cpu2-1', 'cpu2-2', 'cpu2-3']
    """
    if pd.isna(exec_hosts) or not exec_hosts.strip():
        return []
    
    # 标准化处理
    hosts = []
    seen = set()
    
    # 使用更智能的分割方式，避免将范围表示法错误分割
    # 例如，避免将 'cpu1-[7-9]' 分割为 'cpu1-[7' 和 '9]'
    
    # 首先检查是否包含范围表示法
    if '[' in exec_hosts and (']' in exec_hosts or exec_hosts.count('[') > exec_hosts.count(']')):
        # 包含范围表示法，需要更小心地处理
        
        # 先尝试处理逗号分隔的多个服务器
        # 但要避免将方括号内的逗号当作分隔符
        parts = []
        current_part = ""
        bracket_count = 0
        
        for char in exec_hosts:
            if char == '[':
                bracket_count += 1
                current_part += char
            elif char == ']':
                bracket_count -= 1
                current_part += char
            elif char == ',' and bracket_count == 0:
                # 只有在方括号外的逗号才是服务器分隔符
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        
        # 添加最后一部分
        if current_part.strip():
            parts.append(current_part.strip())
        
        # 处理每一部分
        for part in parts:
            if '[' in part:
                # 包含范围表示法，使用_parse_server_range函数解析
                try:
                    parsed_servers = _parse_server_range(part, row_num)
                    for server in parsed_servers:
                        server_clean = server.split(':')[0].lower().strip()
                        if server_clean and server_clean not in seen:
                            seen.add(server_clean)
                            hosts.append(server_clean)
                except Exception as e:
                    print(f"    警告: 解析服务器范围 '{part}' 出错: {str(e)} （行号: {row_num}）")
                    # 如果解析失败，使用原始方式处理
                    server_clean = part.split(':')[0].lower().strip()
                    if server_clean and server_clean not in seen:
                        seen.add(server_clean)
                        hosts.append(server_clean)
            else:
                # 普通服务器名称
                server_clean = part.split(':')[0].lower().strip()
                if server_clean and server_clean not in seen:
                    seen.add(server_clean)
                    hosts.append(server_clean)
    else:
        # 没有范围表示法，使用原始分割方式
        for part in re.split(r'[,\s;|]+', exec_hosts.strip()):
            host = part.split(':')[0].lower().strip()
            if host and host not in seen:
                seen.add(host)
                hosts.append(host)
    
    # 输出解析结果统计
    if row_num is not None and hosts:
        if len(hosts) > 1:
            print(f"    解析服务器列表成功: 共 {len(hosts)} 台服务器 （行号: {row_num}）")
    
    return hosts

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
    if not server_name or not isinstance(server_name, str):
        return ""
    
    # 转为小写
    name = server_name.lower().strip()
    
    # 例如，如果存在"node-"前缀但任务记录中没有，则去除
    # 如果Excel中是"bigmen-1"但任务记录中是"bigman-1"，进行修正
    if name.startswith('bigmen-'):
        name = 'bigman-' + name[7:]
    elif name.startswith('bigman-'):  # 确保bigman保持不变
        name = name
    
    return name

def read_job_data(file_path, test_mode=False, max_rows=100, filter_month=None):
    """读取任务提交数据"""
    filename = os.path.basename(file_path)
    print(f"\n正在读取任务数据: {filename}")
    start_time = time.time()
    
    try:
        # 读取CSV文件，测试模式下使用高效的随机抽样方法
        if test_mode:
            print(f"  测试模式: 随机抽取约 {max_rows} 行数据")
            
            total_rows = 8877313
            print(f"  估计文件总行数: 约 {total_rows:,} 行")
            
            # 如果文件很大，使用跳跃式读取
            if total_rows > max_rows * 10:
                # 计算抽样间隔
                skip_interval = max(1, total_rows // max_rows)
                # 生成要读取的行号
                rows_to_read = list(range(0, total_rows, skip_interval))[:max_rows]
                
                # 读取指定行
                chunks = []
                for i, chunk in enumerate(pd.read_csv(file_path, chunksize=1, skiprows=lambda x: x not in rows_to_read and x > 0)):
                    chunks.append(chunk)
                    if i >= max_rows:
                        break
                
                df = pd.concat(chunks) if chunks else pd.DataFrame()
            else:
                # 文件不是很大，读取全部然后抽样
                df = pd.read_csv(file_path)
                if len(df) > max_rows:
                    df = df.sample(max_rows, random_state=42)
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
        # 使用apply和lambda传递行号参数
        df['server_list'] = df.apply(lambda row: extract_server_id_from_exec_hosts(row['exec_hosts'], row.name), axis=1)
        
        # 预先解析每个任务的服务器列表
        print("  解析服务器列表...")
        df['parsed_servers'] = None  # 初始化存储服务器列表的列
        
        def parse_servers(row):
            """解析单个任务的服务器列表"""
            host_string = row.get('exec_hosts', '')
            row_num = row.name  # 获取行号
            
            # 如果exec_hosts为空，尝试使用first_exec_host
            if (pd.isna(host_string) or not host_string) and 'first_exec_host' in row and not pd.isna(row['first_exec_host']):
                return [row['first_exec_host']]
                
            # 使用ResourceJobIntegrator中的解析逻辑来解析服务器列表
            # 这与_parse_exec_hosts函数使用相同的逻辑
            hosts = []
            seen = set()
            
            if pd.isna(host_string) or not host_string:
                return []
            
            # 检查是否包含范围表示法
            if '[' in host_string and (']' in host_string or host_string.count('[') > host_string.count(']')):
                # 包含范围表示法，使用智能分割
                parts = []
                current_part = ""
                bracket_count = 0
                
                # 处理逗号分隔的服务器列表，保留方括号内的逗号
                for char in host_string:
                    if char == '[':
                        bracket_count += 1
                        current_part += char
                    elif char == ']':
                        bracket_count -= 1
                        current_part += char
                    elif char == ',' and bracket_count == 0:
                        if current_part.strip():
                            parts.append(current_part.strip())
                        current_part = ""
                    else:
                        current_part += char
                
                if current_part.strip():
                    parts.append(current_part.strip())
                    
                # 处理每一部分
                for part in parts:
                    if '[' in part:
                        # 包含范围表示法
                        try:
                            # 使用与_parse_server_range函数相同的逻辑
                            # 解析形如 "cpu1-[41-43]" 的服务器范围
                            prefix = part.split('[')[0]
                            ranges = part.split('[')[1].split(']')[0].split(',')
                            
                            for r in ranges:
                                if '-' in r:
                                    # 处理范围，如 "41-43"
                                    start, end = map(int, r.split('-'))
                                    for i in range(start, end + 1):
                                        server = f"{prefix}{i}"
                                        if server not in seen:
                                            seen.add(server)
                                            hosts.append(server)
                                else:
                                    # 单一值，如 "41"
                                    server = f"{prefix}{r}"
                                    if server not in seen:
                                        seen.add(server)
                                        hosts.append(server)
                        except Exception:
                            # 如果解析失败，将原始字符串作为服务器名称
                            server = part.split(':')[0].lower().strip()
                            if server and server not in seen:
                                seen.add(server)
                                hosts.append(server)
                    else:
                        # 普通服务器名称
                        server = part.split(':')[0].lower().strip()
                        if server and server not in seen:
                            seen.add(server)
                            hosts.append(server)
            else:
                # 没有范围表示法，使用简单分割
                for part in re.split(r'[,\s;|]+', host_string.strip()):
                    host = part.split(':')[0].lower().strip()
                    if host and host not in seen:
                        seen.add(host)
                        hosts.append(host)
            
            return hosts
        
        # 使用tqdm显示解析进度
        print("  正在解析服务器列表...")
        tqdm.pandas(desc="解析服务器列表")
        df['parsed_servers'] = df.progress_apply(parse_servers, axis=1)
        
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
        traceback.print_exc()
        return None

def _auto_generate_group_mapping(server_names):
    """
    根据服务器名称自动生成实例到组的映射
    
    支持多种服务器命名格式，包括：
    1. 标准格式: 'gpu1-15', 'cpu2-43' -> 组为'gpu1', 'cpu2'
    2. 范围表示法: 已展开的服务器名称，如'cpu1-7'来自'cpu1-[7-9]'
    3. 复合范围: 已展开的服务器名称，如'gpu2-43'来自'gpu2-[41-43,45]'
    4. 非标准命名: 如'node103', 'bigman01', '等
    
    参数:
        server_names: 要分组的服务器名称列表
        
    返回:
        服务器名称到组名称的映射字典
    """
    if not server_names:
        return {}
    
    group_map = {}
    
    # 定义多种命名模式
    patterns = [
        # 标准格式: gpu1-15, cpu2-43 (先字母再数字然后可能带的实例标识)
        re.compile(r'^([a-zA-Z]+)(\d+)(?:-(\d+))?$'),
        # 扩展格式1: node103, server45 (字母后跟数字)
        re.compile(r'^([a-zA-Z]+)(\d+)$'),
        # 扩展格式2: cn-g001-35 (带连字符的服务器名称)
        re.compile(r'^([a-zA-Z]+-[a-zA-Z0-9]+)-(\d+)$'),
        # 扩展格式3: h01n03, r05n02（多层次编号）
        re.compile(r'^([a-zA-Z])(\d+)n(\d+)$'),
    ]
    
    # 预处理: 转换所有服务器名称为小写并去除空白
    normalized_names = [name.lower().strip() if isinstance(name, str) else str(name) for name in server_names]
    
    # 特殊类别映射
    special_categories = {
        'bigman': ['bigman', 'bigmen', 'large'],
        'gpu': ['gpu', 'graphic', 'nvidia', 'rtx', 'v100', 'a100', 'p100'],
        'cpu': ['cpu', 'xeon', 'compute', 'c-node'],
        'node': ['node', 'server', 'host'],
        'storage': ['storage', 'disk', 'sto-', 'backup', 'bck-'],
        'network': ['network', 'net-', 'router', 'switch']
    }
    
    # 分析服务器名称汇总信息，提取模式
    server_patterns = {}
    special_count = {category: 0 for category in special_categories}
    
    for orig_name, name in zip(server_names, normalized_names):
        if not name or name in ['none', 'null', 'na', 'n/a', '-']:
            group_map[orig_name] = 'unknown'
            continue
        
        # 判断是否属于特殊类别
        matched_category = None
        for category, keywords in special_categories.items():
            if any(keyword in name for keyword in keywords):
                matched_category = category
                special_count[category] += 1
                break
        
        if matched_category:
            # 尝试精确匹配格式
            matched = False
            for pattern in patterns:
                match = pattern.match(name)
                if match:
                    # 提取主要组件以生成组名
                    if len(match.groups()) >= 2:
                        base = match.group(1).lower()
                        number = match.group(2)
                        
                        # 特殊情况处理
                        if matched_category == 'bigman':
                            group_map[orig_name] = 'bigman'
                        else:
                            # 生成组名: 如gpu1, cpu2
                            group_name = f"{matched_category}{number}" if number else matched_category
                            group_map[orig_name] = group_name
                            
                            # 记录模式以便后续分析
                            if matched_category not in server_patterns:
                                server_patterns[matched_category] = {}
                            if group_name not in server_patterns[matched_category]:
                                server_patterns[matched_category][group_name] = 0
                            server_patterns[matched_category][group_name] += 1
                            
                        matched = True
                        break
            
            # 如果没有匹配到特定模式，使用类别作为组名
            if not matched:
                group_map[orig_name] = matched_category
        else:
            # 尝试使用模式
            matched = False
            for pattern in patterns:
                match = pattern.match(name)
                if match:
                    if len(match.groups()) >= 2:
                        base = match.group(1).lower()
                        number = match.group(2)
                        # 生成组名
                        group_name = f"{base}{number}"
                        group_map[orig_name] = group_name
                        matched = True
                        break
            
            # 没有匹配任何模式，使用服务器名称作为组名
            if not matched:
                group_map[orig_name] = orig_name
    
    # 打印服务器组分析信息
    print(f"  服务器组映射生成完成: 生成了 {len(set(group_map.values()))} 个组")
    
    # 显示特殊类别统计
    for category, count in special_count.items():
        if count > 0:
            print(f"    - {category}: {count} 个服务器")
    
    return group_map

def analyze_resource_usage(matched_df, resource_groups):
    """
    分析资源使用情况（分实例和组两个层级）
    
    参数:
        matched_df: 已匹配的数据帧，包含任务和资源信息
        resource_groups: 服务器实例到组的映射字典
        
    返回:
        instance_analysis: 实例级别的资源分析结果
        group_analysis: 组级别的资源分析结果
    """
    print("\n开始资源使用分析...")
    
    # 验证输入数据
    if matched_df is None or matched_df.empty:
        print("  警告: 匹配数据为空，无法进行资源分析")
        # 返回空的DataFrame以保持一致的返回类型
        empty_df = pd.DataFrame(columns=['Server Instance', 'Metric', 'Avg Usage', 'Peak Usage', 'Min Usage', 'Job Count'])
        return empty_df, empty_df.copy()
    
    # 确保必要的列存在
    required_cols = ['server_instance', 'metric', 'value', 'job_id']
    missing_cols = [col for col in required_cols if col not in matched_df.columns]
    if missing_cols:
        print(f"  错误: 分析数据缺失必要列: {missing_cols}")
        empty_df = pd.DataFrame(columns=['Server Instance', 'Metric', 'Avg Usage', 'Peak Usage', 'Min Usage', 'Job Count'])
        return empty_df, empty_df.copy()
    
    # 实例级分析 - 使用更高效的聚合方法
    try:
        # 优化职业度量计算
        metrics_to_compute = {
            'value': ['mean', 'max', 'min', 'std'],  # 增加标准差信息
            'job_id': 'nunique'  # 使用nunique而不是count以获取唯一任务数
        }
        
        # 实现分组计算以提高性能
        instance_analysis = matched_df.groupby(['server_instance', 'metric']).agg(metrics_to_compute)
        
        # 重构多级索引
        instance_analysis.columns = ['_'.join(col).strip() for col in instance_analysis.columns.values]
        instance_analysis = instance_analysis.reset_index()
        
        # 重命名列，确保一致的命名风格
        instance_analysis.rename(columns={
            'server_instance': 'Server Instance',
            'metric': 'Metric',
            'value_mean': 'Avg Usage',
            'value_max': 'Peak Usage',
            'value_min': 'Min Usage',
            'value_std': 'Std Dev',
            'job_id_nunique': 'Job Count'
        }, inplace=True)
        
        # 添加利用率列（平均使用量/峰值）
        instance_analysis['Usage Ratio'] = instance_analysis['Avg Usage'] / instance_analysis['Peak Usage']
        instance_analysis['Usage Ratio'].fillna(0, inplace=True)  # 处理可能的划零情况
        
        # 组级分析
        # 首先确保数据帧包含server_group列
        if 'server_group' not in matched_df.columns:
            print("  添加服务器组映射...")
            # 如果提供了resource_groups字典，使用它来映射服务器组
            if resource_groups:
                matched_df['server_group'] = matched_df['server_instance'].map(resource_groups)
                # 处理没有对应组的服务器
                matched_df['server_group'].fillna('unknown', inplace=True)
            else:
                # 如果没有提供映射，则使用identify_server_group函数
                matched_df['server_group'] = matched_df['server_instance'].apply(identify_server_group)
        
        # 计算组级分析
        group_analysis = matched_df.groupby(['server_group', 'metric']).agg(metrics_to_compute)
        group_analysis.columns = ['_'.join(col).strip() for col in group_analysis.columns.values]
        group_analysis = group_analysis.reset_index()
        
        # 与实例分析保持一致的列名
        group_analysis.rename(columns={
            'server_group': 'Server Group',
            'metric': 'Metric',
            'value_mean': 'Avg Usage',
            'value_max': 'Peak Usage',
            'value_min': 'Min Usage',
            'value_std': 'Std Dev',
            'job_id_nunique': 'Job Count'
        }, inplace=True)
        
        # 添加利用率
        group_analysis['Usage Ratio'] = group_analysis['Avg Usage'] / group_analysis['Peak Usage']
        group_analysis['Usage Ratio'].fillna(0, inplace=True)
        
        # 添加组大小信息（每个组包含的服务器数量）
        group_sizes = matched_df.groupby('server_group')['server_instance'].nunique().to_dict()
        group_analysis['Group Size'] = group_analysis['Server Group'].map(group_sizes)
        
        # 返回结果前输出摘要信息
        print(f"  分析了 {len(instance_analysis['Server Instance'].unique())} 个服务器实例")
        print(f"  分析了 {len(group_analysis['Server Group'].unique())} 个服务器组")
        
        return instance_analysis, group_analysis
        
    except Exception as e:
        print(f"  警告: 分析资源使用时出错: {str(e)}")
        traceback.print_exc()
        # 出错时返回空的DataFrame
        empty_df = pd.DataFrame(columns=['Server Instance', 'Metric', 'Avg Usage', 'Peak Usage', 'Min Usage', 'Job Count'])
        return empty_df, empty_df.copy()

def match_jobs_with_resources(jobs_df, resources, test_mode=False, rows_limit=None, use_parallel=False):
    """
    将任务数据与资源监控数据匹配（分实例和组两个层级）
    
    实现了批量处理和快速时间窗口匹配逻辑
    
    参数:
        jobs_df: 任务数据帧
        resources: 资源数据字典 {metric_name: {server_name: dataframe}}
        test_mode: 是否为测试模式
        rows_limit: 测试模式下处理的行数限制
        use_parallel: 是否使用并行处理（大数据集推荐）
        
    返回:
        matched_df: 匹配结果数据帧
    """
    print("开始任务与资源数据匹配...")
    match_start = time.time()
    
    # 验证输入
    if jobs_df is None or jobs_df.empty:
        print("错误: 任务数据为空")
        return pd.DataFrame()
        
    if not resources or not any(resources.values()):
        print("错误: 资源数据为空")
        return pd.DataFrame()
    
    # 如果在测试模式下，限制处理的任务数
    if test_mode and rows_limit and rows_limit < len(jobs_df):
        print(f"  测试模式: 限制处理任务数量为 {rows_limit} 条")
        jobs_df = jobs_df.head(rows_limit)
    
    # 预处理时间字段，确保类型一致性
    for time_field in ['start_time', 'end_time', 'submit_time']:
        if time_field in jobs_df.columns:
            jobs_df[time_field] = jobs_df[time_field].apply(parse_datetime)
    
    # 1. 先获取所有资源服务器实例
    all_resource_servers = set()
    resource_time_ranges = {}
    
    # 收集每个服务器的时间范围信息，用于快速时间范围检查
    for metric, resource_data in resources.items():
        for server, df in resource_data.items():
            all_resource_servers.add(server)
            if 'timestamp' in df.columns and not df.empty:
                min_time = df['timestamp'].min()
                max_time = df['timestamp'].max()
                if server not in resource_time_ranges:
                    resource_time_ranges[server] = {}
                resource_time_ranges[server][metric] = (min_time, max_time)
    
    # 2. 生成服务器组映射
    group_mapping = _auto_generate_group_mapping(all_resource_servers)
    inverse_group_mapping = defaultdict(list)
    for instance, group in group_mapping.items():
        inverse_group_mapping[group].append(instance)
    
    print(f"  识别到 {len(all_resource_servers)} 个资源服务器实例, {len(group_mapping)} 个服务器组")
    
    # 3. 处理任务数据
    matched_records = []
    missing_servers = set()
    no_resource_time_servers = set()
    time_out_of_range = 0
    
    for _, job in tqdm(jobs_df.iterrows(), total=len(jobs_df), desc="匹配任务"):
        # 解析任务的服务器列表
        job_servers = parse_exec_hosts(job['exec_hosts']) if 'exec_hosts' in job and job['exec_hosts'] else []
        if not job_servers:
            continue
            
        # 处理任务时间
        job_start = job.get('start_time')
        job_end = job.get('end_time')
        
        # 跳过没有有效时间的任务
        if job_start is None or job_end is None:
            continue
        
        valid_servers = []
        
        # 服务器匹配逻辑
        for server in job_servers:
            # 判断是否应该排除该服务器
            if should_exclude_server(server):
                continue
                
            # 尝试情况1: 精确匹配服务器实例
            if server in all_resource_servers:
                valid_servers.append(server)
                continue
            
            # 尝试情况2: 匹配组级别
            server_group = identify_server_group(server)
            group_servers = inverse_group_mapping.get(server_group, [])
            valid_group_servers = [srv for srv in group_servers if srv in all_resource_servers]
            
            if valid_group_servers:
                valid_servers.extend(valid_group_servers)
            else:
                # 尝试情况3: 类似服务器名称匹配
                # 实现模糊匹配 - 将服务器名称根据核心部分进行匹配
                normalized_server = _normalize_server_name(server)
                if normalized_server and normalized_server in all_resource_servers:
                    valid_servers.append(normalized_server)
                    continue
                    
                # 记录未匹配的服务器
                missing_servers.add(server)
        
        # 如果找到匹配的服务器，创建记录
        if valid_servers:
            # 针对每个服务器和指标进行细粒度匹配
            for server in valid_servers:
                server_group = group_mapping.get(server, "unknown")
                
                for metric, resource_data in resources.items():
                    if server in resource_data:
                        server_df = resource_data[server]
                        
                        # 检查服务器的时间范围
                        server_time_range = resource_time_ranges.get(server, {}).get(metric, None)
                        if not server_time_range:
                            no_resource_time_servers.add(server)
                            continue
                            
                        min_time, max_time = server_time_range
                        
                        # 判断任务时间是否在资源时间范围内
                        # 任务执行时间区间与资源监控时间区间至少有部分重叠
                        if (job_start <= max_time and job_end >= min_time):
                            # 找到最适合的时间点来匹配资源
                            # 逻辑：先尝试匹配任务开始时间，如果超出范围则选用最近的时间点
                            matched_times = []
                            
                            # 1. 任务开始时间
                            if min_time <= job_start <= max_time:
                                # 找与任务开始时间最接近的所有资源数据点
                                closest_idx = server_df['timestamp'].searchsorted(job_start)
                                if closest_idx < len(server_df):
                                    matched_times.append(closest_idx)
                            
                            # 2. 任务结束时间
                            if min_time <= job_end <= max_time:
                                closest_idx = server_df['timestamp'].searchsorted(job_end)
                                if closest_idx < len(server_df):
                                    matched_times.append(closest_idx)
                                    
                            # 3. 任务中间点
                            mid_time = job_start + (job_end - job_start) / 2
                            if min_time <= mid_time <= max_time:
                                closest_idx = server_df['timestamp'].searchsorted(mid_time)
                                if closest_idx < len(server_df):
                                    matched_times.append(closest_idx)
                            
                            # 如果没有找到指定时间点的匹配，但时间范围不为空，选取最近的时间点
                            if not matched_times:
                                # 找出最接近任务时间范围的点
                                if job_start < min_time:  # 任务开始早于资源记录
                                    matched_times.append(0)  # 选用最早的资源记录
                                elif job_end > max_time:  # 任务结束晚于资源记录
                                    matched_times.append(len(server_df) - 1)  # 选用最晚的资源记录
                            
                            # 为匹配的每个时间点创建记录
                            for idx in set(matched_times):  # 使用set删除重复的索引
                                if 0 <= idx < len(server_df):
                                    resource_row = server_df.iloc[idx]
                                    record = {
                                        'job_id': job['job_id'],
                                        'user_id': job['user_id'],
                                        'submit_time': job['submit_time'],
                                        'start_time': job_start,
                                        'end_time': job_end,
                                        'server_instance': server,
                                        'server_group': server_group,
                                        'metric': metric,
                                        'value': resource_row['value'],
                                        'timestamp': resource_row['timestamp'],
                                        'requested_processors': job.get('requested_processors', 0),
                                        'requested_gpu': job.get('requested_gpu', 0),
                                        'queue': job.get('queue', ''),
                                        'job_duration': (job_end - job_start).total_seconds() if job_end and job_start else 0
                                    }
                                    matched_records.append(record)
                        else:
                            time_out_of_range += 1
    
    # 生成最终DataFrame
    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        match_end = time.time()
        
        # 输出匹配统计
        print(f"\n匹配统计:")
        print(f"- 总任务数: {len(jobs_df)}")
        print(f"- 成功匹配的任务数: {len(matched_df['job_id'].unique())}")
        print(f"- 产生的匹配记录数: {len(matched_df)}")
        print(f"- 未匹配的服务器数: {len(missing_servers)}")
        if missing_servers:
            print(f"- 示例未匹配服务器: {sorted(list(missing_servers)[:10])}...")
        print(f"- 由于时间范围不匹配而跳过的记录数: {time_out_of_range}")
        print(f"- 匹配耗时: {match_end - match_start:.2f} 秒")
        
        # 添加数据验证
        if test_mode:
            try:
                verify_data_consistency(matched_df)
            except Exception as e:
                print(f"\n警告: 验证匹配数据时出错: {e}")
        
        return matched_df
    else:
        print("错误: 没有匹配到任何有效数据")
        # 返回空的DataFrame而不是None，以保持一致的返回类型
        return pd.DataFrame()

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
    hourly_usage = matched_df.groupby([pd.Grouper(key='timestamp', freq='1H'), 'server_group']).agg({
        'job_id': 'nunique',
        'cpu_total': 'mean',
        'requested_processors': 'sum',
        'gpu_util': 'mean',
        'requested_gpu': 'sum'
    }).reset_index()
    
    # 保存每小时的资源使用数据
    hourly_file = os.path.join(output_dir, 'hourly_resource_usage.csv')
    print(f"  保存每小时资源使用数据: {os.path.basename(hourly_file)}")
    hourly_usage.to_csv(hourly_file, index=False)
    
    # 为每个服务器组绘制资源饱和度与实际使用对比
    for group in hourly_usage['server_group'].unique():
        group_data = hourly_usage[hourly_usage['server_group'] == group]
        
        plt.figure(figsize=(14, 8))
        
        plt.plot(group_data['timestamp'], group_data['cpu_total'], 
                 label='实际CPU使用率', color='blue', marker='o', alpha=0.7, markersize=3)
        plt.plot(group_data['timestamp'], group_data['requested_processors'] / group_data['job_id'], 
                 label='请求CPU饱和度', color='red', marker='x', alpha=0.7, markersize=3)
        
        plt.title(f'{group} 服务器组 - CPU请求vs实际使用')
        plt.xlabel('时间')
        plt.ylabel('CPU使用率/饱和度')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, max(200, group_data['requested_processors'].max() / group_data['job_id'].max() * 1.1))
        
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
            if not server_data.empty and 'time_window_start' in server_data.columns:
                all_times.extend(server_data['time_window_start'].tolist())
    
    if not all_times:
        return pd.DataFrame()
    
    min_time = pd.to_datetime(min(all_times))
    max_time = pd.to_datetime(max(all_times))
    return pd.DataFrame({
        'time_window_start': pd.date_range(start=min_time.floor('H'), 
                                         end=max_time.ceil('H'),
                                         freq='H'),
        'time_window_end': lambda x: x['time_window_start'] + pd.Timedelta(hours=1)
    })

def merge_resource_data(base_df, resources):
    """合并资源数据（每个服务器单独行）
    
    参数:
        base_df: 基础DataFrame，包含时间窗口信息（目前在实现中未使用）
        resources: 资源数据字典 {metric_name: {server_name: dataframe}}
        
    返回:
        资源数据合并后的DataFrame
    """
    merged = []
    
    # 检查base_df是否包含时间窗口信息（记录在注释中，但保持原有逻辑不变）
    has_base_windows = (base_df is not None and not base_df.empty 
                      and 'time_window_start' in base_df.columns)
    if has_base_windows:
        print(f"  发现基础时间窗口: {len(base_df)} 个时间点")
    
    # 处理所有资源指标和服务器
    for metric_name, servers in resources.items():
        for server, data in servers.items():
            if 'time_window_start' not in data.columns:
                print(f"  警告: {server}的{metric_name}数据缺少time_window_start列")
                continue
                
            df = data[['time_window_start', 'value']].copy()
            # 确保值列包含数值
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['value'])  # 删除无法转换为数值的行
            df['server'] = server
            df.rename(columns={'value': metric_name}, inplace=True)
            merged.append(df)
    
    # 合并所有资源指标
    if not merged:
        print("  警告: 没有有效的资源数据可合并")
        return pd.DataFrame()
        
    # 使用concat合并所有资源数据框
    resource_df = pd.concat(merged, ignore_index=True)
    
    # 按时间窗口和服务器分组，取每组的第一个值
    return resource_df.groupby(['time_window_start', 'server']).first().reset_index()

def match_jobs_to_windows(jobs_df, time_windows, resources):
    """将任务匹配到时间窗口（每个服务器生成独立行）
    
    参数:
        jobs_df: 任务数据帧
        time_windows: 时间窗口数据帧
        resources: 资源数据字典
        
    返回:
        匹配后的数据帧, 包含任务与时间窗口的关联
    """
    print("\
开始匹配任务到时间窗口...")
    matched = []
    job_count = 0
    server_match_count = 0
    
    # 定义预期的列
    expected_columns = [
        'time_window_start', 'time_window_end', 'server', 'job_id', 'user_id', 
        'exec_hosts', 'queue_time', 'duration_time'
    ]
    
    # 验证输入数据
    if jobs_df.empty:
        print("  警告: 任务数据为空")
        empty_df = pd.DataFrame(columns=expected_columns)
        empty_df['time_window_start'] = pd.Series(dtype='datetime64[ns]')
        empty_df['time_window_end'] = pd.Series(dtype='datetime64[ns]')
        return empty_df
        
    if time_windows.empty:
        print("  警告: 时间窗口数据为空")
        empty_df = pd.DataFrame(columns=expected_columns)
        empty_df['time_window_start'] = pd.Series(dtype='datetime64[ns]')
        empty_df['time_window_end'] = pd.Series(dtype='datetime64[ns]')
        return empty_df
    
    # 确保关键列存在
    required_job_cols = ['job_id', 'user_id', 'exec_hosts', 'start_time', 'end_time', 'submit_time']
    missing_cols = [col for col in required_job_cols if col not in jobs_df.columns]
    if missing_cols:
        print(f"  错误: 任务数据缺少必要列: {', '.join(missing_cols)}")
        empty_df = pd.DataFrame(columns=expected_columns)
        empty_df['time_window_start'] = pd.Series(dtype='datetime64[ns]')
        empty_df['time_window_end'] = pd.Series(dtype='datetime64[ns]')
        return empty_df
    
    required_window_cols = ['time_window_start', 'time_window_end']
    missing_cols = [col for col in required_window_cols if col not in time_windows.columns]
    if missing_cols:
        print(f"  错误: 时间窗口数据缺少必要列: {', '.join(missing_cols)}")
        empty_df = pd.DataFrame(columns=expected_columns)
        empty_df['time_window_start'] = pd.Series(dtype='datetime64[ns]')
        empty_df['time_window_end'] = pd.Series(dtype='datetime64[ns]')
        return empty_df
    
    # 开始匹配处理
    total_jobs = len(jobs_df)
    print(f"  总计{total_jobs}个任务待匹配")
    
    # 预处理时间窗口，确保是时间类型
    time_windows = time_windows.copy()
    if not pd.api.types.is_datetime64_dtype(time_windows['time_window_start']):
        time_windows['time_window_start'] = pd.to_datetime(time_windows['time_window_start'])
    if not pd.api.types.is_datetime64_dtype(time_windows['time_window_end']):
        time_windows['time_window_end'] = pd.to_datetime(time_windows['time_window_end'])
    
    for _, job in jobs_df.iterrows():
        try:
            # 解析服务器列表
            servers = parse_exec_hosts(job['exec_hosts'])
            if not servers:
                continue
            
            # 计算任务时间
            try:
                job_start = pd.to_datetime(job['start_time'])
                job_end = pd.to_datetime(job['end_time'])
                submit_time = pd.to_datetime(job['submit_time'])
            except (ValueError, TypeError) as e:
                print(f"  警告: 任务 {job.get('job_id', 'unknown')} 的时间格式无效: {str(e)}")
                continue
            
            # 检查时间有效性
            if job_start > job_end:
                print(f"  警告: 任务 {job.get('job_id', 'unknown')} 的结束时间早于开始时间")
                continue
            
            # 找到重叠的时间窗口
            mask = (time_windows['time_window_start'] <= job_end) & \
                   (time_windows['time_window_end'] > job_start)
            windows = time_windows[mask].copy()
            
            if windows.empty:
                continue
            
            # 成功匹配一个任务
            job_count += 1
            
            # 为每个服务器生成记录
            for server in servers:
                server_windows = windows.copy()
                server_windows['server'] = server
                
                # 添加任务信息
                server_windows['job_id'] = job['job_id']
                server_windows['user_id'] = job['user_id']
                server_windows['exec_hosts'] = job['exec_hosts']
                queue_seconds = (job_start - submit_time).total_seconds()
                duration_seconds = (job_end - job_start).total_seconds()
                
                # 检查计算的值是否合理
                if queue_seconds < 0:
                    print(f"  警告: 任务 {job['job_id']} 的排队时间为负值 ({queue_seconds}s)")
                    queue_seconds = 0
                if duration_seconds < 0:
                    print(f"  警告: 任务 {job['job_id']} 的运行时间为负值 ({duration_seconds}s)")
                    duration_seconds = 0
                    
                server_windows['queue_time'] = queue_seconds
                server_windows['duration_time'] = duration_seconds
                
                matched.append(server_windows)
                server_match_count += len(server_windows)
        except Exception as e:
            print(f"  警告: 处理任务 {job.get('job_id', 'unknown')} 时发生错误: {str(e)}")
            continue
    
    # 如果有匹配数据，合并并返回
    if matched:
        print(f"  匹配结果: {job_count}/{total_jobs} 个任务匹配成功, 生成 {server_match_count} 条服务器时间窗口记录")
        result_df = pd.concat(matched, ignore_index=True)
        # 因为有可能生成重复记录，所以这里去重
        result_df = result_df.drop_duplicates()
        return result_df
    else:
        # 返回空的DataFrame，但包含必要的列
        print("  警告: 没有任务与时间窗口匹配成功")
        empty_df = pd.DataFrame(columns=expected_columns)
        # 确保时间列有正确的类型
        empty_df['time_window_start'] = pd.Series(dtype='datetime64[ns]')
        empty_df['time_window_end'] = pd.Series(dtype='datetime64[ns]')
        return empty_df

def integrate_data(resources, jobs_df):
    """
    整合资源和任务数据，生成统一的分析数据框
    
    参数:
        resources: 资源数据字典，格式为 {metric_name: {server_name: dataframe}}
        jobs_df: 包含任务信息的DataFrame
    
    返回:
        final_df: 整合了资源和任务数据的DataFrame，按指定列排序
        None: 如果处理过程中发生严重错误
    """
    print("\n开始数据整合...")
    
    # 验证输入数据
    if not resources:
        print("  错误: 资源数据为空")
        return None
        
    if jobs_df is None or jobs_df.empty:
        print("  警告: 任务数据为空，将只包含资源数据")
        jobs_df = pd.DataFrame()  # 确保为空DataFrame而非None
    
    # 生成统一时间窗口
    print("  1. 生成统一时间窗口...")
    time_windows = generate_time_windows(resources)
    if time_windows.empty:
        print("  错误: 无法生成时间窗口，可能资源数据中没有有效的时间点")
        return None
    print(f"    共生成 {len(time_windows)} 个时间窗口")
    
    # 合并资源数据
    print("  2. 合并资源数据...")
    resource_df = merge_resource_data(time_windows.copy(), resources)
    if resource_df.empty:
        print("  错误: 合并资源数据失败，无法继续")
        return None
    print(f"    资源数据合并完成: {len(resource_df)} 行 x {len(resource_df.columns)} 列")
    
    # 检查必要的列是否存在
    required_resource_cols = ['time_window_start', 'server']
    missing_cols = [col for col in required_resource_cols if col not in resource_df.columns]
    if missing_cols:
        print(f"  错误: 资源数据缺少必要列: {', '.join(missing_cols)}")
        return None
    
    # 合并任务数据
    print("  3. 匹配任务数据到时间窗口...")
    jobs_matched = match_jobs_to_windows(jobs_df, time_windows, resources)
    if jobs_matched is None:  # 区分空DataFrame和None
        print("  错误: 任务匹配过程发生严重错误")
        return None
    
    # 显示数据统计信息
    print("\n数据统计:")
    print(f"  - 资源数据: {len(resource_df)} 行, {len(resource_df.columns)} 列")
    print(f"  - 任务匹配数据: {len(jobs_matched)} 行, {len(jobs_matched.columns)} 列")
    
    # 显示列信息（更简洁）
    resource_cols = ", ".join(sorted(resource_df.columns.tolist()))
    jobs_cols = ", ".join(sorted(jobs_matched.columns.tolist()))
    print(f"  - 资源数据列: {resource_cols}")
    print(f"  - 任务数据列: {jobs_cols}")
    
    # 确保关键列存在
    merge_cols = ['time_window_start', 'server']
    for col in merge_cols:
        if col not in resource_df.columns:
            print(f"  错误: 资源数据缺少合并键 '{col}'")
            return None
        if col not in jobs_matched.columns and not jobs_matched.empty:
            print(f"  错误: 任务数据缺少合并键 '{col}'")
            return None
    
    print("\n  4. 执行最终合并...")
    # 最终合并 - 使用left join保留所有资源数据点
    try:
        if jobs_matched.empty:
            print("    没有匹配到任务数据，最终结果将只包含资源指标")
            final_df = resource_df
        else:
            final_df = pd.merge(resource_df, jobs_matched,
                               on=merge_cols,
                               how='left')
            print(f"    合并完成: {len(final_df)} 行 x {len(final_df.columns)} 列")
    except Exception as e:
        print(f"  错误: 合并数据时出错: {str(e)}")
        return None
    
    # 定义详细的列排序，确保exec_hosts位于queue之后，job_distribution放在最后
    print("  5. 整理列顺序...")
    base_ordered_cols = [
        # 时间和服务器信息
        'time_window_start', 'time_window_end', 'server',
        # 资源监控数据
        'resource_exists', 'cpu_total', 'cpu_system', 'gpu_avg', 'gpu_util',
        'mem_percent', 'mem_used', 'power', 'temperature', 'load_1min', 'load_15min',
        # 任务基本信息
        'job_id', 'job_name', 'job_user', 'job_status', 'queue', 
        # exec_hosts字段放在queue之后
        'exec_hosts',
        # 任务时间和其他信息
        'submit_time', 'start_time', 'end_time', 'job_duration', 'queue_time', 'duration_time',
        # 任务分布信息放在最后
        'job_distribution'
    ]
    
    # 只保留实际存在于数据框中的列
    ordered_cols = [col for col in base_ordered_cols if col in final_df.columns]
    
    # 添加数据框中存在但未列举的其他列
    other_cols = [col for col in final_df.columns if col not in ordered_cols]
    if other_cols:
        print(f"    发现 {len(other_cols)} 个未在预定义排序中的列，将添加到结果末尾")
        print(f"    额外列: {', '.join(other_cols)}")
        ordered_cols.extend(sorted(other_cols))  # 按字母顺序排序额外列
    
    # 确保最终返回的DataFrame包含所有列并且按照指定顺序排列
    print(f"\n数据整合完成: 最终结果包含 {len(final_df)} 行 x {len(ordered_cols)} 列")
    return final_df[ordered_cols]

def extract_job_metadata(jobs_df):
    """从已读取的任务数据中提取服务器列表和时间范围"""
    print("从任务数据提取元数据...")
    
    all_servers = set()
    time_range = [pd.Timestamp.max, pd.Timestamp.min]
    
    try:
        # 解析服务器列表
        for _, row in jobs_df.iterrows():
            if 'exec_hosts' in row and not pd.isna(row['exec_hosts']):
                servers = parse_exec_hosts(row['exec_hosts'])
                all_servers.update([s for s in servers if s])
        
        # 解析时间范围
        if 'start_time' in jobs_df.columns and 'end_time' in jobs_df.columns:
            time_range[0] = jobs_df['start_time'].min()
            time_range[1] = jobs_df['end_time'].max()
        
        return {
            'servers': list(all_servers),
            'time_range': (time_range[0].floor('D'), time_range[1].ceil('D'))
        }
    except Exception as e:
        print(f"提取任务元数据时发生错误: {str(e)}")
        traceback.print_exc()
        return {'servers': [], 'time_range': (pd.Timestamp.now(), pd.Timestamp.now())}

def load_target_resources(pre_data):
    """按需加载目标资源数据"""
    target_servers = pre_data['servers']
    time_range = pre_data['time_range']
    
    if not target_servers:
        print("警告: 没有需要监控的服务器")
        return {}
    
    # 预处理服务器列表，检测是否包含范围表示法
    expanded_servers = []
    range_notation_found = False
    
    for server in target_servers:
        if '[' in server:
            range_notation_found = True
            try:
                # 尝试解析范围表示法
                parsed_servers = _parse_server_range(server)
                if parsed_servers:
                    print(f"服务器范围解析成功: '{server}' -> {len(parsed_servers)}个服务器")
                    expanded_servers.extend(parsed_servers)
                    continue
            except Exception as e:
                print(f"警告: 解析服务器范围 '{server}' 出错: {str(e)}")
        
        # 如果不是范围或解析失败，直接添加原始服务器名称
        expanded_servers.append(server)
    
    # 去除重复项
    unique_servers = []
    seen = set()
    for server in expanded_servers:
        if server not in seen:
            seen.add(server)
            unique_servers.append(server)
    
    if len(unique_servers) < len(expanded_servers):
        print(f"信息: 去除了 {len(expanded_servers) - len(unique_servers)} 个重复的服务器名称")
    
    # 更新目标服务器列表
    target_servers = unique_servers
    
    print(f"\n按需加载资源数据（{len(target_servers)}台服务器）...")
    if len(target_servers) <= 20:
        print(f"需要查找的服务器: {', '.join(target_servers)}")
    else:
        print(f"需要查找的服务器: {len(target_servers)}台服务器 (前5个: {', '.join(target_servers[:5])}...)")
    
    if range_notation_found:
        print(f"信息: 检测到服务器范围表示法，已展开为具体服务器名称")
    
    # 创建结果字典
    resources = {}
    
    # 加载各类资源数据
    start_time = time.time()
    
    # 调试输出时间范围
    print(f"查询时间范围: {time_range[0]} 至 {time_range[1]}")
    
    # 定义要加载的资源类型
    resource_types = [
        'cpu_total', 'cpu_system', 'gpu_avg', 'gpu_util', 
        'mem_percent', 'mem_used', 'load_1min', 'load_15min',
        'power', 'temperature'
    ]
    
    # 加载所有资源类型
    for resource_type in resource_types:
        print(f"\n加载 {resource_type} 资源数据...")
        resources[resource_type] = _load_resource_data(resource_type, DATA_PATHS[resource_type], target_servers, time_range)
    
    # 检查是否有任何有效资源数据
    has_data = False
    for resource_type, data in resources.items():
        if data:  # 如果至少有一个服务器的数据
            has_data = True
            break
    
    if not has_data:
        print("错误: 所有资源数据都为空，无法继续分析")
    
    elapsed_time = time.time() - start_time
    print(f"资源数据加载完成。耗时: {int(elapsed_time/60)}分{int(elapsed_time%60)}秒")
    
    # 输出资源数据统计
    valid_resources = {k: v for k, v in resources.items() if v}
    if valid_resources:
        print(f"\n成功加载的资源类型: {len(valid_resources)}/{len(resources)}")
        
        # 计算各资源类型的服务器覆盖率
        for resource_name, resource_data in valid_resources.items():
            server_count = len(resource_data)
            coverage = server_count / len(target_servers) * 100 if target_servers else 0
            record_count = sum(len(data) for server, data in resource_data.items())
            avg_records = record_count / server_count if server_count else 0
            
            print(f"  - {resource_name}: {server_count}/{len(target_servers)} 台服务器 (覆盖率: {coverage:.1f}%), 总计 {record_count} 条记录, 平均 {avg_records:.1f} 条/服务器")
            
            # 如果服务器数量少，显示具体服务器名称
            if server_count <= 10:
                print(f"    成功加载的服务器: {', '.join(sorted(resource_data.keys()))}")
            
            # 显示缺失的服务器
            missing_servers = set(target_servers) - set(resource_data.keys())
            if missing_servers and len(missing_servers) <= 10:
                print(f"    缺失的服务器: {', '.join(sorted(missing_servers))}")
            elif missing_servers:
                print(f"    缺失 {len(missing_servers)} 台服务器")
    
    return resources

def _load_resource_data(metric_name, file_path, target_servers, time_range):
    """加载特定类型的资源数据"""
    if not os.path.exists(file_path):
        print(f"  警告: 资源数据文件不存在: {file_path}")
        return {}
    
    print(f"  加载 {metric_name} 数据: {os.path.basename(file_path)}")
    
    try:
        # 读取Excel文件的工作表列表
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        print(f"    可用工作表数量: {len(sheet_names)}")
        
        # 创建结果字典
        result = {}
        
        # 预处理目标服务器列表，展开范围表示法
        expanded_servers = []
        for server in target_servers:
            # 检查是否包含范围表示法 [x-y]
            if '[' in server and ']' in server:
                try:
                    # 尝试解析范围表示法
                    parsed_servers = _parse_server_range(server)
                    if parsed_servers:
                        print(f"    范围解析成功: '{server}' -> {len(parsed_servers)}个服务器 {', '.join(parsed_servers[:3])}{' ...' if len(parsed_servers) > 3 else ''}")
                        expanded_servers.extend(parsed_servers)
                        continue
                except Exception as e:
                    print(f"    警告: 解析服务器范围 '{server}' 出错: {str(e)}")
            
            # 检查是否包含连字符范围表示法，如 'cpu1-103-104'
            if re.search(r'[a-zA-Z]+[0-9]*-[0-9]+-[0-9]+$', server):
                try:
                    # 尝试解析连字符范围表示法
                    parsed_servers = _parse_hyphen_range(server)
                    if len(parsed_servers) > 1:  # 确认是范围，不是原始服务器名称
                        expanded_servers.extend(parsed_servers)
                        continue
                except Exception as e:
                    print(f"    警告: 解析连字符范围 '{server}' 出错: {str(e)}")
            
            # 如果不是范围或解析失败，直接添加原始服务器名称
            expanded_servers.append(server)
        
        print(f"    展开后的服务器数量: {len(expanded_servers)}")
        
        # 遍历展开后的目标服务器，查找匹配的工作表
        matched_count = 0
        for server in expanded_servers:
            # 查找匹配的工作表
            matching_sheet = _find_matching_sheet(server, sheet_names)
            
            if matching_sheet:
                matched_count += 1
                
                try:
                    # 读取工作表数据
                    df = pd.read_excel(file_path, sheet_name=matching_sheet)
                    
                    # 确保所有列都被处理为适当的类型
                    if 'value' in df.columns:
                        # 尝试将value列转换为数值类型
                        df['value'] = pd.to_numeric(df['value'], errors='coerce')
                        # 丢弃无法转换为数值的行
                        df = df.dropna(subset=['value'])
                    
                    # 确保有时间列
                    if 'timestamp' not in df.columns:
                        print(f"    警告: 工作表 {matching_sheet} 缺少时间列")
                        continue
                    
                    # 标准化时间列名
                    time_col = None
                    for col in ['timestamp', 'time_window_start']:
                        if col in df.columns:
                            time_col = col
                            break
                    
                    if time_col is None:
                        print(f"    警告: 工作表 {matching_sheet} 无法识别时间列")
                        continue
                    
                    # 转换时间列
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    
                    # 过滤时间范围
                    start_time, end_time = time_range
                    filtered_df = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
                    
                    if len(filtered_df) > 0:
                        # 保存到结果字典
                        result[server] = filtered_df
                        print(f"    成功加载 {server} 数据: {len(filtered_df)} 条记录")
                    else:
                        print(f"    警告: 服务器 {server} 在指定时间范围内没有数据")
                
                except Exception as e:
                    print(f"    读取服务器 {server} 的工作表 {matching_sheet} 时出错: {str(e)}")
            else:
                # 记录未匹配的服务器，但不需要为每个都打印日志
                # if len(expanded_servers) <= 20 or random.random() < 0.1:  # 如果服务器很多，只随机打印部分
                print(f"    未找到匹配: 服务器 {server} 在 {metric_name} 数据中不存在匹配的工作表")
        
        # 检查是否有成功加载的数据
        if not result:
            print(f"  警告: 没有成功加载任何 {metric_name} 数据")
        else:
            print(f"  成功加载 {metric_name} 数据: {len(result)}/{len(expanded_servers)} 台服务器 (匹配率: {matched_count}/{len(expanded_servers)} = {matched_count/len(expanded_servers)*100:.1f}%)")
        
        return result
    
    except Exception as e:
        print(f"  加载 {metric_name} 数据时出错: {str(e)}")
        traceback.print_exc()
        return {}

def _parse_server_range(server_name, row_num=None):
    """
    增强版服务器范围表示法解析器，处理多种复杂格式
    支持的格式包括：
    1. 基本范围: 'cpu1-[7-9]' -> ['cpu1-7', 'cpu1-8', 'cpu1-9']
    2. 逗号分隔的多个值: 'cpu1-[13,15,17]' -> ['cpu1-13', 'cpu1-15', 'cpu1-17']
    3. 复合范围: 'cpu1-[15,96-97]' -> ['cpu1-15', 'cpu1-96', 'cpu1-97']
    4. 不完整范围: 'cpu1-[7-9' -> ['cpu1-7', 'cpu1-8', 'cpu1-9'] (缺少右括号)
    5. 更复杂的复合范围: 'cpu1-[13-15,17,19-20]'
    """
    # 如果不包含范围表示法，直接返回原始名称
    if '[' not in server_name:
        return [server_name]
    
    row_info = f" （行号: {row_num}）" if row_num is not None else ""
    
    try:
        # 提取服务器基本名称前缀
        prefix = server_name[:server_name.find('[')]
        
        # 检查是否存在不完整的范围表示法
        if '[' in server_name and ']' not in server_name:
            # 不完整范围，如 'cpu1-[7-9'
            range_str = server_name[server_name.find('[')+1:]
            print(f"    不完整范围解析: '{server_name}' -> 前缀='{prefix}', 范围='{range_str}'{row_info}")
        else:
            # 正常范围表示法
            range_str = server_name[server_name.find('[')+1:server_name.find(']')]
        
        # 检查是否包含逗号，决定使用哪种解析方式
        if ',' in range_str:
            # 复合范围，如 'cpu1-[15,96-97]' 或 'cpu1-[34,39,43]'
            return _parse_compound_range(server_name, prefix, range_str, row_num)
        elif '-' in range_str:
            # 基本范围，如 'cpu1-[7-9]'
            return _parse_simple_range(server_name, prefix, range_str, row_num)
        else:
            # 单值，如 'cpu1-[7]'
            try:
                value = int(range_str)
                result = [f"{prefix}{value}"]
                print(f"    单值解析成功: '{server_name}' -> '{result[0]}'{row_info}")
                return result
            except ValueError:
                print(f"    警告: 无法解析的单值 '{range_str}'{row_info}")
                return [server_name]  # 返回原始名称
    except Exception as e:
        print(f"    警告: 解析服务器范围 '{server_name}' 时出错: {str(e)}{row_info}")
        return [server_name]  # 如果解析失败，返回原始服务器名称

def _parse_simple_range(server_name, prefix, range_str, row_num=None):
    """解析简单的范围表示法，如 'cpu1-[7-9]'"""
    row_info = f" （行号: {row_num}）" if row_num is not None else ""
    
    try:
        # 分隔起始和结束编号
        start, end = map(int, range_str.split('-'))
        result = [f"{prefix}{i}" for i in range(start, end + 1)]
        
        servers_str = '、'.join([f"'{s}'" for s in result[:5]])
        if len(result) > 5:
            servers_str += f"等{len(result)}个服务器"
        
        print(f"    简单范围解析成功: '{server_name}' -> {len(result)}个服务器 {servers_str}{row_info}")
        return result
    except Exception as e:
        print(f"    警告: 解析简单范围 '{range_str}' 失败: {str(e)}{row_info}")
        return [server_name]

def _parse_compound_range(server_name, prefix, range_str, row_num=None):
    """解析复合范围表示法，如 'cpu1-[15,96-97]' 或 'cpu1-[34,39,43]'"""
    row_info = f" （行号: {row_num}）" if row_num is not None else ""
    
    result = []
    # 处理逗号分隔的多个范围或单值
    parts = [p.strip() for p in range_str.split(',') if p.strip()]
    
    for part in parts:
        if '-' in part:
            # 范围表示法，如 '96-97'
            try:
                start, end = map(int, part.split('-'))
                for i in range(start, end + 1):
                    result.append(f"{prefix}{i}")
            except ValueError as ve:
                print(f"    警告: 复合范围解析失败 '{part}': {str(ve)}{row_info}")
                # 如果解析失败，使用原始格式
                result.append(f"{prefix}{part}")
        else:
            # 单个值，如 '15'
            try:
                # 尝试将其解析为数字
                int(part)
                result.append(f"{prefix}{part}")
            except ValueError:
                print(f"    警告: 复合范围中的非数字值 '{part}'{row_info}")
                result.append(f"{prefix}{part}")
    
    if result:
        servers_str = '、'.join([f"'{s}'" for s in result[:5]])
        if len(result) > 5:
            servers_str += f"等{len(result)}个服务器"
        print(f"    复合范围解析成功: '{server_name}' -> {len(result)}个服务器 {servers_str}{row_info}")
    
    return result


def _parse_hyphen_range(server_name, row_num=None):
    """
    解析形如 'cpu1-103-104' 的服务器范围表示法
    将其转换为 ['cpu1-103', 'cpu1-104']
    """
    row_info = f" （行号: {row_num}）" if row_num is not None else ""
    
    # 检查是否符合 prefix-num1-num2 格式
    pattern = re.match(r'([a-zA-Z]+[0-9]*-[0-9]+)-([0-9]+)$', server_name)
    if not pattern:
        return [server_name]  # 不符合格式，返回原始名称
    
    try:
        first_part, end_num = pattern.groups()
        # 提取前缀和起始数字
        prefix_pattern = re.match(r'([a-zA-Z]+[0-9]*)-([0-9]+)', first_part)
        if not prefix_pattern:
            return [server_name]
            
        prefix, start_num = prefix_pattern.groups()
        
        # 转换为整数进行范围检查
        start = int(start_num)
        end = int(end_num)
        
        # 生成范围内的所有服务器名称
        result = []
        for i in range(start, end + 1):
            result.append(f"{prefix}-{i}")
        
        servers_str = '、'.join([f"'{s}'" for s in result])
        print(f"    连字符范围解析成功: '{server_name}' -> {len(result)}个服务器 {servers_str}{row_info}")
        
        return result
    except Exception as e:
        print(f"    警告: 解析连字符范围 '{server_name}' 时出错: {str(e)}{row_info}")
        return [server_name]


def _find_matching_sheet(server_name, sheet_names, row_num=None):
    """
    查找匹配的工作表
    支持多种匹配策略，包括精确匹配、部分匹配、基本名称匹配等
    参数：
        server_name: 服务器名称
        sheet_names: 可用的工作表名称列表
        row_num: 行号，用于跟踪和调试
    返回：
        匹配的工作表名称，如果没有匹配则返回 None
    """
    # 行号信息用于输出
    row_info = f" （行号: {row_num}）" if row_num is not None else ""
    
    # 记录原始服务器名称用于日志
    original_name = server_name
    
    # 检查是否是异常数据
    if not server_name or pd.isna(server_name):
        print(f"    警告: 服务器名称为空{row_info}")
        return None
    
    # 标准化处理
    server_name = str(server_name).strip()
    
    # 检查是否是极端异常数据（纯数字或只有括号的数字）
    if re.match(r'^[\[\]0-9\-]+$', server_name):
        print(f"    警告: 检测到极端异常服务器名称 '{server_name}'{row_info}，标记为服务器数据缺失")
        return None
    
    # 生成多种可能的服务器名称变体
    server_variations = [
        server_name.lower(),                    # 原始名称小写
        server_name.upper(),                    # 全大写
        server_name.replace('-', '_'),          # 破折号替换为下划线
        server_name.replace('_', '-'),          # 下划线替换为破折号
        # 添加更多可能的变体
        re.sub(r'[\[\]\(\)\{\}]', '', server_name),  # 移除所有括号
    ]
    
    # 去除重复项并过滤空值
    server_variations = list(set([v for v in server_variations if v]))
    
    # 1. 尝试精确匹配
    for variation in server_variations:
        if variation in sheet_names:
            print(f"    精确匹配成功: '{original_name}' -> '{variation}'{row_info}")
            return variation
    
    # 2. 尝试部分匹配（工作表名包含服务器名）
    for variation in server_variations:
        for sheet in sheet_names:
            if variation in sheet.lower():
                print(f"    部分匹配成功: '{original_name}' -> '{sheet}'{row_info}")
                return sheet
    
    # 3. 尝试提取服务器名称的基本部分（去除数字后缀）
    base_pattern = re.match(r'([a-zA-Z]+[0-9]*)[-_]([0-9]+)', server_name.lower())
    if base_pattern:
        base_name, number = base_pattern.groups()
        
        # 尝试匹配基本名称+数字
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            # 检查基本名称和数字是否都在工作表名中
            if base_name in sheet_lower and number in sheet_lower:
                print(f"    基本名称+数字匹配成功: '{original_name}' -> '{sheet}'{row_info}")
                return sheet
        
        # 尝试仅匹配基本名称
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            if (base_name == sheet_lower or 
                f"{base_name}-" in sheet_lower or 
                f"{base_name}_" in sheet_lower):
                print(f"    基本名称匹配成功: '{original_name}' -> '{sheet}'{row_info}")
                return sheet
    
    # 4. 尝试更灵活的匹配：提取服务器名称的字母部分和数字部分
    server_pattern = re.match(r'([a-zA-Z]+)([0-9-_\[\]]+)', server_name.lower())
    if server_pattern:
        prefix, suffix = server_pattern.groups()
        
        # 尝试匹配前缀和后缀的一部分
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            if prefix in sheet_lower:
                # 提取后缀中的所有数字
                numbers = re.findall(r'\d+', suffix)
                if numbers and any(num in sheet_lower for num in numbers):
                    print(f"    前缀+数字匹配成功: '{original_name}' -> '{sheet}'{row_info}")
                    return sheet
    
    # 5. 尝试匹配服务器名称的任何数字部分
    # 这是最后的尝试，可能会导致错误匹配，但比没有匹配好
    numbers = re.findall(r'\d+', server_name)
    if numbers:
        for sheet in sheet_names:
            sheet_lower = sheet.lower()
            # 如果工作表名包含服务器名称中的任何数字
            if any(num in sheet_lower for num in numbers):
                # 检查是否还包含服务器名称中的任何字母
                letters = re.findall(r'[a-zA-Z]+', server_name.lower())
                if letters and any(letter in sheet_lower for letter in letters):
                    print(f"    字母+数字匹配成功: '{original_name}' -> '{sheet}'{row_info}")
                    return sheet
    
    print(f"    未找到匹配: '{original_name}'{row_info}")
    return None

class ResourceJobAnalyzer:
    def __init__(self, jobs_data, resource_data, output_path):
        """
        初始化分析器
        :param jobs_data: 预处理后的任务数据 (DataFrame)
        :param resource_data: 加载的资源数据 (dict)
        :param output_path: 结果输出目录
        """
        self.jobs = jobs_data
        self.resources = resource_data
        self.output_dir = output_path
        os.makedirs(output_path, exist_ok=True)

    def process(self):
        """处理资源数据和任务数据，生成集成分析结果"""
        
        # 初始化日志记录
        log_file = os.path.join(self.output_dir, 'log.txt')
        original_stdout = sys.stdout
        
        try:
            # 将标准输出重定向到日志记录器
            sys.stdout = TeeLogger(log_file)
            
            # 记录运行环境信息
            print("\n=== 资源与任务整合分析工具运行日志 ===\n")
            print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"CPU核心数: {multiprocessing.cpu_count()}")
            print("\n===================================\n")
            
            print("开始处理资源和任务数据...")
            
            # 处理任务数据，为每个任务匹配资源使用情况
            result = self._task_based_integration()
            
            if result is not None and not result.empty:
                # 保存结果
                self._save_output(result)
                
                print("\n\n=== 处理完成统计信息 ===\n")
                print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("\n=== 日志记录结束 ===\n")
                
                return True
            else:
                print("处理失败: 未能生成有效的集成数据")
                return False
        
        finally:
            # 确保日志文件关闭并恢复原始标准输出
            if isinstance(sys.stdout, TeeLogger):
                sys.stdout.close()
            sys.stdout = original_stdout
            print(f"日志已保存至: {log_file}")

    def _process_job_chunk(self, job_chunk, resource_index, resource_columns):
        """处理一批任务数据，使用优化的向量化操作和批量处理"""
        result_rows = []
        
        # 预处理：一次性转换时间列，避免重复转换
        job_chunk = job_chunk.copy()
        
        # 确保时间列是datetime类型
        for time_col in ['start_time', 'end_time', 'submit_time']:
            if time_col in job_chunk.columns:
                mask = job_chunk[time_col].apply(lambda x: isinstance(x, str))
                if mask.any():
                    job_chunk.loc[mask, time_col] = pd.to_datetime(job_chunk.loc[mask, time_col])
        
        # 过滤掉没有有效时间信息的任务
        valid_jobs = job_chunk.dropna(subset=['start_time', 'end_time'])
        
        # 转换必要的列和预计算值
        valid_jobs['job_id_str'] = valid_jobs['job_id'].astype(str)
        valid_jobs['job_duration'] = (valid_jobs['end_time'] - valid_jobs['start_time']).dt.total_seconds()
        
        # 处理排队时间
        valid_jobs['queue_time'] = float('nan')
        queue_mask = ~valid_jobs['submit_time'].isna()
        if queue_mask.any():
            valid_jobs.loc[queue_mask, 'queue_time'] = (
                valid_jobs.loc[queue_mask, 'start_time'] - 
                valid_jobs.loc[queue_mask, 'submit_time']
            ).dt.total_seconds()
        
        # 一次性处理用户ID
        valid_jobs['job_user'] = 'n/a'
        user_mask = ~valid_jobs['user_id'].isna()
        if user_mask.any():
            valid_jobs.loc[user_mask, 'job_user'] = valid_jobs.loc[user_mask, 'user_id'].apply(self._hash_user_id)
        
        for _, job in valid_jobs.iterrows():
            job_id = job['job_id_str']
            job_start = job['start_time']
            job_end = job['end_time']
            job_submit = job['submit_time']
            job_duration = job['job_duration']
            queue_time = job['queue_time']
            job_user = job['job_user']
            
            # 使用预解析的服务器列表
            servers = job.get('parsed_servers', [])
            if not servers:
                # 如果没有预解析的服务器列表，则回退到原始的解析方法
                servers = self._parse_exec_hosts(job.get('exec_hosts', ''))
                if not servers and 'first_exec_host' in job and not pd.isna(job['first_exec_host']):
                    servers = [job['first_exec_host']]
            
            if not servers:
                # 没有服务器信息，创建一行使用n/a作为服务器
                base_row = {
                    'job_id': job_id,
                    'job_name': job.get('job_name', 'n/a'),
                    'job_user': job_user,
                    'job_status': job.get('job_status_str', 'n/a'),
                    'queue': job.get('queue', 'n/a'),
                    'exec_hosts': job.get('exec_hosts', 'n/a'),
                    'submit_time': job_submit,
                    'start_time': job_start,
                    'end_time': job_end,
                    'job_duration': job_duration,
                    'queue_time': queue_time,
                    'server': 'n/a',
                    'time_window_start': job_start.floor('H'),
                    'time_window_end': job_start.floor('H') + pd.Timedelta(hours=1)
                }
                
                # 批量添加所有资源列为n/a
                base_row.update({col: 'n/a' for col in resource_columns})
                result_rows.append(base_row)
                continue
            
            # 为每个服务器和任务时间窗口创建行
            for server in servers:
                # 预计算所有时间窗口，避免在循环中重复计算
                current_time = job_start.floor('H')
                end_time = job_end.ceil('H')
                time_windows = []
                
                while current_time < end_time:
                    window_end = current_time + pd.Timedelta(hours=1)
                    time_windows.append((current_time, window_end))
                    current_time = window_end
                
                # 创建基础行数据（所有时间窗口共享的部分）
                common_data = {
                    'job_id': job_id,
                    'job_name': job.get('job_name', 'n/a'),
                    'job_user': job_user,
                    'job_status': job.get('job_status_str', 'n/a'),
                    'queue': job.get('queue', 'n/a'),
                    'exec_hosts': job.get('exec_hosts', 'n/a'),
                    'submit_time': job_submit,
                    'start_time': job_start,
                    'end_time': job_end,
                    'job_duration': job_duration,
                    'queue_time': queue_time,
                    'server': server,
                }
                
                # 提前检查服务器是否在资源索引中
                server_in_index = server in resource_index
                
                # 批量处理每个时间窗口
                for window_start, window_end in time_windows:
                    # 复制共享数据
                    row_data = common_data.copy()
                    row_data.update({
                        'time_window_start': window_start,
                        'time_window_end': window_end
                    })
                    
                    # 添加资源使用信息
                    if server_in_index and window_start in resource_index[server]:
                        for resource_type in resource_columns:
                            # 获取资源值
                            value = resource_index[server][window_start].get(resource_type, 'n/a')
                            
                            # 确保值是数值类型
                            if value != 'n/a':
                                try:
                                    # 尝试转换为浮点数
                                    value = float(value)
                                except (ValueError, TypeError):
                                    # 如果无法转换为数值，设为n/a
                                    value = 'n/a'
                            
                            row_data[resource_type] = value
                    else:
                        # 批量设置所有资源列为n/a
                        row_data.update({col: 'n/a' for col in resource_columns})
                    
                    result_rows.append(row_data)
        
        return result_rows

    def _task_based_integration(self):
        """以任务为中心进行数据集成，使用多核并行处理，带有错误处理和顺序处理回退机制 - 优化版本"""
        # 导入垃圾回收模块，以便在处理大块数据时手动进行内存管理
        import gc
        import psutil  # 用于监控系统资源
        
        # 保存开始时间，用于性能分析
        start_time = time.time()
        
        # 检查任务数据
        if self.jobs is None or len(self.jobs) == 0:
            print("错误: 没有任务数据可以与资源关联")
            return None
        
        print("\n开始准备任务与资源匹配...")
        
        # 准备资源数据索引，按服务器和时间窗口组织
        print("\n步骤1: 创建资源数据索引")
        resource_index = self._create_resource_index()
        
        # 打印系统资源使用情况
        mem_info = psutil.virtual_memory()
        print(f"\n当前内存使用情况: {mem_info.percent}% ({mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB)")
        
        # 定义所有资源列，确保即使没有数据也会有这些列
        resource_columns = [
            'cpu_total', 'cpu_system', 'gpu_avg', 'gpu_util', 
            'mem_percent', 'mem_used', 'load_1min', 'load_15min', 
            'power', 'temperature'
        ]
        
        # 步骤2: 准备任务分块
        print("\n步骤2: 准备并行处理任务数据")
        
        # 计算每个进程处理的任务数量
        total_jobs = len(self.jobs)
        print(f"任务总数: {total_jobs}")
        
        # 自动检测系统资源和任务规模，智能调整并行度
        # 获取CPU核心数量
        num_cores = multiprocessing.cpu_count()
        
        # 根据内存使用情况和任务数量动态调整并行度
        if mem_info.percent > 85:  # 内存使用率极高
            print(f"检测到内存使用率极高({mem_info.percent}%)，将使用保守的并行设置")
            # 使用保守的设置，每块更大以减少进程间通信
            num_workers = max(32, int(num_cores * 0.3))  # 使用至少32个核心，约30%的核心数
            chunk_size = max(1000, min(10000, total_jobs // (num_workers * 2)))  # 更大的块大小
        elif total_jobs > 1000000:  # 大型任务集
            print(f"检测到大型任务集({total_jobs}个任务)，使用自适应并行设置")
            # 对于大型任务集，使用更大的块，较高并行度
            num_workers = max(48, int(num_cores * 0.5))  # 使用至少48个核心，约50%的核心数
            chunk_size = max(5000, min(20000, total_jobs // (num_workers * 4)))  # 适合大数据集的块大小
        else:  # 正常情况
            print(f"检测到常规任务集({total_jobs}个任务)，使用标准并行设置")
            # 一般情况下的高并行度设置
            num_workers = max(64, int(num_cores * 0.75))  # 使用至少64个核心，约75%的核心数
            chunk_size = max(1000, min(5000, total_jobs // (num_workers * 4)))  # 优化的块大小
        
        # 在系统资源过低时限制并行度
        if num_cores < 4:  # 低核心数系统
            num_workers = 1  # 不使用并行
            chunk_size = max(500, min(2000, total_jobs // 4))  # 更小的块大小
        
        # 分割任务数据
        job_chunks = [self.jobs.iloc[i:i+chunk_size] for i in range(0, total_jobs, chunk_size)]
        print(f"数据已分割为 {len(job_chunks)} 个块，每块约 {chunk_size} 个任务")
        print(f"检测到 {num_cores} 个CPU核心，将使用 {num_workers} 个并行进程")
        
        # 设置处理标志，以便在并行处理失败时回退到顺序处理
        use_parallel = True
        all_results = []
        
        # 步骤3: 并行处理任务数据
        print("\n步骤3: 开始处理任务数据")
        
        # 尝试并行处理
        if use_parallel and num_workers > 1:
            try:
                print(f"使用并行模式处理数据 (进程数: {num_workers}, 块数: {len(job_chunks)})")
                
                # 记录成功和失败的块数
                successful_chunks = 0
                failed_chunks = 0
                
                # 使用tqdm显示总体进度
                with tqdm(total=len(job_chunks), desc="并行处理任务块", unit="块") as pbar:
                    # 使用进程池处理任务
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        # 分批提交任务，避免一次性创建太多future对象
                        batch_size = min(num_workers * 4, len(job_chunks))  # 增大批次大小以提高吞吐量
                        processed_chunks = 0
                        timeout_occurred = False
                        
                        # 分批处理所有块
                        while processed_chunks < len(job_chunks) and not timeout_occurred:
                            # 自适应批大小调整，根据已处理的块的平均时间来估算
                            if successful_chunks > 0 and processed_chunks > batch_size * 2:
                                # 动态调整批大小以平衡性能和内存使用
                                current_memory_usage = psutil.virtual_memory().percent
                                if current_memory_usage > 85:  # 内存压力大时减小批大小
                                    batch_size = max(1, batch_size // 2)
                                    print(f"\n警告: 内存使用过高 ({current_memory_usage}%), 减小批大小至 {batch_size}")
                                elif current_memory_usage < 60:  # 内存充足时增加批大小
                                    batch_size = min(num_workers * 8, batch_size * 2)
                            
                            # 计算当前批次范围
                            current_batch = job_chunks[processed_chunks:processed_chunks+batch_size]
                            batch_futures = {}
                            
                            # 提交当前批次的任务
                            for i, chunk in enumerate(current_batch):
                                chunk_index = processed_chunks + i
                                future = executor.submit(self._process_job_chunk, chunk, resource_index, resource_columns)
                                batch_futures[future] = chunk_index
                            
                            # 处理当前批次的结果
                            try:
                                # 设置动态超时，根据批次大小并考虑最小和最大范围
                                batch_timeout = max(600, min(3600, len(batch_futures) * 120))  # 每个块平均给予120秒，最多1小时
                                
                                # 等待所有任务完成或超时
                                for future in as_completed(batch_futures.keys(), timeout=batch_timeout):
                                    try:
                                        # 为单个任务设置超时
                                        single_task_timeout = max(240, min(1200, batch_timeout // len(batch_futures) * 3))
                                        result = future.result(timeout=single_task_timeout)
                                        all_results.extend(result)
                                        successful_chunks += 1
                                        pbar.update(1)
                                    except Exception as e:
                                        chunk_idx = batch_futures.get(future, "未知")
                                        print(f"\n警告: 处理任务块 {chunk_idx} 时出错: {type(e).__name__}: {str(e)}")
                                        failed_chunks += 1
                                        pbar.update(1)  # 即使失败也更新进度条
                            except concurrent.futures.TimeoutError:
                                timeout_occurred = True
                                print(f"\n批处理超时({batch_timeout}秒), 切换到顺序处理模式处理剩余任务")
                                # 标记当前批次中未完成的任务
                                incomplete_chunks = [job_chunks[batch_futures[f]] for f in batch_futures if not f.done()]
                                # 将未完成的块添加到剩余块中
                                remaining_chunks = incomplete_chunks + job_chunks[processed_chunks+len(current_batch):]
                                print(f"  - 有 {len(incomplete_chunks)} 个块在本批次中未能完成")
                                print(f"  - 还有 {len(job_chunks) - (processed_chunks+len(current_batch))} 个块尚未处理")
                                
                                # 继续顺序处理
                                print(f"\n切换到顺序处理模式，准备处理剩余的 {len(remaining_chunks)} 个任务块")
                                with tqdm(total=len(remaining_chunks), desc="顺序处理剩余任务块", unit="块") as seq_pbar:
                                    for i, chunk in enumerate(remaining_chunks):
                                        try:
                                            # 单线程处理每个块
                                            result = self._process_job_chunk(chunk, resource_index, resource_columns)
                                            all_results.extend(result)
                                            successful_chunks += 1
                                            seq_pbar.update(1)
                                            
                                            # 每处理10个块执行一次垃圾回收
                                            if (i + 1) % 10 == 0:
                                                gc.collect()
                                        except Exception as e:
                                            print(f"\n警告: 顺序处理任务块 {i} 时出错: {type(e).__name__}: {str(e)}")
                                            failed_chunks += 1
                                            seq_pbar.update(1)  # 即使失败也更新进度条
                                break
                            
                            # 更新已处理的块数量
                            processed_chunks += len(current_batch)
                            
                            # 手动回收内存
                            gc.collect()
                
                # 显示并行处理总结
                print(f"\n并行处理统计: 处理 {processed_chunks}/{len(job_chunks)} 块, 成功 {successful_chunks} 块, 失败 {failed_chunks} 块")
                
            except concurrent.futures.process.BrokenProcessPool as e:
                print(f"\n并行处理失败: 进程池被中断: {str(e)}")
                print("将回退到顺序处理模式")
                use_parallel = False
            except MemoryError as e:
                print(f"\n内存不足错误: {str(e)}")
                print("将回退到顺序处理模式并减小块大小")
                # 减小块大小以减轻内存干折
                chunk_size = max(100, chunk_size // 10)
                job_chunks = [self.jobs.iloc[i:i+chunk_size] for i in range(0, total_jobs, chunk_size)]
                print(f"重新分割数据为 {len(job_chunks)} 个块，每块约 {chunk_size} 个任务")
                use_parallel = False
            except Exception as e:
                print(f"\n并行处理时发生未预期错误: {type(e).__name__}: {str(e)}")
                print("将回退到顺序处理模式并输出详细错误")
                traceback.print_exc()
                use_parallel = False
        
        # 如果并行处理失败或未完成，使用顺序处理
        if not use_parallel:
            successful_sequential_chunks = 0
            failed_sequential_chunks = 0
            print("\n步骤4: 使用顺序处理模式，需要处理 {len(job_chunks)} 个任务块")
            
            # 清空之前的结果，避免重复
            all_results = []
            
            # 使用tqdm显示进度
            with tqdm(total=len(job_chunks), desc="顺序处理任务块", unit="块") as pbar:
                for i, chunk in enumerate(job_chunks):
                    try:
                        # 监控内存使用率，过高时优化内存
                        current_memory_usage = psutil.virtual_memory().percent
                        if current_memory_usage > 90:  # 内存危险，强制回收
                            print(f"\n警告: 内存使用率过高 ({current_memory_usage}%)，执行强制内存回收")
                            gc.collect()
                            # 如果内存仍然过高，尝试减小块大小
                            if psutil.virtual_memory().percent > 90 and i < len(job_chunks) - 1:
                                # 重新分割剩余数据为更小的块
                                remaining_index = i + 1
                                remaining_jobs = pd.concat([chunk for chunk in job_chunks[remaining_index:]])
                                smaller_chunk_size = max(100, chunk_size // 5)
                                new_chunks = [remaining_jobs.iloc[j:j+smaller_chunk_size] for j in range(0, len(remaining_jobs), smaller_chunk_size)]
                                # 替换剩余的块
                                job_chunks = job_chunks[:remaining_index] + new_chunks
                                print(f"  已重新分割剩余任务为更小的块，将处理 {len(job_chunks) - remaining_index} 个小块")
                        
                        # 处理当前块
                        result = self._process_job_chunk(chunk, resource_index, resource_columns)
                        all_results.extend(result)
                        successful_sequential_chunks += 1
                        pbar.update(1)
                        
                        # 自适应垃圾回收策略，根据内存压力动态调整
                        if current_memory_usage > 75:  # 内存压力大时更频繁回收
                            if (i + 1) % 5 == 0:  # 每5个块执行一次垃圾回收
                                gc.collect()
                        else:  # 正常情况下的回收频率
                            if (i + 1) % 10 == 0:  # 每10个块执行一次垃圾回收
                                gc.collect()
                    except MemoryError as e:
                        print(f"\n内存不足错误在块 {i}: {str(e)}")
                        # 立即回收内存
                        gc.collect()
                        # 尝试将当前块分割为更小的块继续处理
                        try:
                            print("  尝试将当前块分割为更小的单元继续处理...")
                            smaller_size = max(50, len(chunk) // 10)
                            sub_chunks = [chunk.iloc[j:j+smaller_size] for j in range(0, len(chunk), smaller_size)]
                            # 处理小块
                            for sub_i, sub_chunk in enumerate(sub_chunks):
                                try:
                                    sub_result = self._process_job_chunk(sub_chunk, resource_index, resource_columns)
                                    all_results.extend(sub_result)
                                    successful_sequential_chunks += 1/len(sub_chunks)  # 分数应用
                                    # 每处理完一个小块就立即回收内存
                                    gc.collect()
                                except Exception as sub_e:
                                    print(f"  处理小块 {sub_i} 时出错: {type(sub_e).__name__}: {str(sub_e)}")
                                    failed_sequential_chunks += 1/len(sub_chunks)  # 分数应用
                            # 注意: 不更新主进度条，因为这是对当前块的特殊处理
                            pbar.update(1)  # 建议更新一次，表示原块已处理完毕
                        except Exception as retry_e:
                            print(f"\n尝试分割处理块 {i} 时仍然失败: {type(retry_e).__name__}: {str(retry_e)}")
                            failed_sequential_chunks += 1
                            pbar.update(1)  # 即使失败也更新进度条
                    except Exception as e:
                        print(f"\n处理任务块 {i} 时出错: {type(e).__name__}: {str(e)}")
                        traceback.print_exc()
                        failed_sequential_chunks += 1
                        pbar.update(1)  # 即使失败也更新进度条
            
            # 显示顺序处理的统计信息
            print(f"\n顺序处理统计: 处理 {len(job_chunks)} 块, 成功 {successful_sequential_chunks:.1f} 块, 失败 {failed_sequential_chunks:.1f} 块")
        
        # 步骤5: 结果处理和报告生成
        print("\n步骤5: 结果处理和报告生成")
        
        # 计算总运行时间
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒"
        
        # 在所有任务处理完成后处理结果
        if len(all_results) == 0:
            print("\n错误: 所有处理块失败，无法生成结果列表")
            return pd.DataFrame()
        
        # 将所有结果合并为一个DataFrame
        try:
            # 最终垃圾回收，以减少内存压力
            gc.collect()
            
            # 记录内存使用情况
            mem_before = psutil.virtual_memory()
            print(f"\n合并结果前的内存使用情况: {mem_before.percent}% ({mem_before.used/1024/1024/1024:.1f}GB/{mem_before.total/1024/1024/1024:.1f}GB)")
            
            # 创建pandas DataFrame
            print(f"\n正在合并 {len(all_results)} 个结果记录...")
            merged_df = pd.DataFrame(all_results)
            
            # 记录内存使用情况
            mem_after = psutil.virtual_memory()
            mem_increase = mem_after.used - mem_before.used
            print(f"\n合并结果后的内存使用情况: {mem_after.percent}% ({mem_after.used/1024/1024/1024:.1f}GB/{mem_after.total/1024/1024/1024:.1f}GB)")
            print(f"\n内存增加: {mem_increase/1024/1024/1024:.1f}GB")
            
            # 显示统计信息
            merged_jobs_count = len(merged_df)
            server_count = len(resource_index) if resource_index else 0
            success_rate = merged_jobs_count / total_jobs * 100
            
            # 生成性能报告
            print("\n" + "="*80)
            print("\n任务与资源匹配性能报告")
            print("="*80)
            print(f"\n处理概况:")
            print(f"  - 总任务数: {total_jobs:,}")
            print(f"  - 成功匹配数: {merged_jobs_count:,} ({success_rate:.1f}%)")
            print(f"  - 使用的服务器数: {server_count:,}")
            print(f"  - 总运行时间: {time_str}")
            print(f"  - 平均处理速度: {total_jobs/total_time:.1f} 条/秒")
            print("\n" + "="*80)
            
            # 返回结果
            return merged_df
        except MemoryError as e:
            print(f"\n内存不足错误：无法合并结果: {str(e)}")
            print("\n重要提示: 请使用更大内存的服务器或开启内存交换文件再尝试运行。")
            return pd.DataFrame()
        except Exception as e:
            print(f"\n生成最终结果时出错: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def _create_resource_index(self):
        """创建资源数据索引，按服务器和时间窗口组织 - 并行处理优化版本
        使用多进程并行处理加速索引创建，大幅减少处理时间
        """
        import gc  # 导入垃圾回收模块
        import psutil  # 用于监控系统资源
        import multiprocessing
        import concurrent.futures
        
        if not self.resources:
            return {}
        
        print("创建资源数据索引...")
        start_time = time.time()
        
        # 获取当前系统状态
        mem_info = psutil.virtual_memory()
        print(f"当前系统内存使用: {mem_info.percent}% ({mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB)")
        
        # 计算并行处理的数量
        num_cores = multiprocessing.cpu_count()
        num_resource_types = len(self.resources)
        
        # 安全的并行处理数量 - 根据系统资源动态调整
        if mem_info.percent > 80:  # 内存使用非常高时调低并行度
            num_workers = max(1, min(num_cores // 2, 32))  # 使用至少一半的核心，最少32个
        else:  # 内存充足时
            num_workers = max(64, int(num_cores * 0.75))  # 使用至少64个核心，最多使用75%的核心
            
        # 如果资源类型少于并行数，则调低并行数
        num_workers = min(num_workers, num_resource_types)
        
        print(f"系统有 {num_cores} 个核心, 将使用 {num_workers} 个并行进程处理 {num_resource_types} 种资源类型")
        
        # 使用预先解析的服务器列表
        all_servers = set()
        if self.jobs is not None:
            # 使用tqdm显示进度
            with tqdm(total=len(self.jobs), desc="提取服务器名称", unit="任务") as pbar:
                for _, job in self.jobs.iterrows():
                    # 尝试使用预解析的服务器列表
                    if 'parsed_servers' in job and job['parsed_servers']:
                        all_servers.update(job['parsed_servers'])
                    elif 'exec_hosts' in job and not pd.isna(job['exec_hosts']):
                        # 如果没有预解析的服务器列表，则使用原始的解析方法
                        parsed_hosts = self._parse_exec_hosts(job['exec_hosts'])
                        all_servers.update(parsed_hosts)
                    pbar.update(1)
                    
                    # 定期回收内存
                    if pbar.n % 10000 == 0:
                        gc.collect()
        
        # 转换为小写以避免大小写不匹配问题
        all_servers = {s.lower() for s in all_servers if s}
        print(f"任务中使用的所有服务器数量: {len(all_servers)}")
        
        # 创建资源索引 - 使用嵌套字典结构
        # 结构: resource_index[server][time_window] = {resource1: value1, resource2: value2, ...}
        resource_index = {}
        
        # 预先为所有服务器创建字典，避免动态扩展
        for server in all_servers:
            resource_index[server] = {}
        
        # 跟踪加载的每种资源类型的数据量
        resource_stats = {}
        
        # 记录上次内存检查时间，用于定期监控内存
        last_mem_check = time.time()
        mem_check_interval = 60  # 每60秒检查一次内存状态
        
        # 使用并行处理加速资源索引创建
        print(f"开始并行处理资源数据...")
        
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 创建并行任务
                futures = {}
                for resource_type, server_data_dict in self.resources.items():
                    futures[executor.submit(self._process_resource_type, resource_type, server_data_dict, all_servers)] = resource_type
                
                # 使用tqdm来显示进度
                with tqdm(total=len(futures), desc="并行处理资源类型", unit="类型") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        resource_type = futures[future]
                        try:
                            # 获取并行处理结果
                            processed_resource_type, sub_index, stats = future.result()
                            
                            # 将子索引合并到主索引
                            for server, windows in sub_index.items():
                                for window_start, resource_data in windows.items():
                                    # 创建空的服务器和时间窗口索引（如果不存在）
                                    if server not in resource_index:
                                        resource_index[server] = {}
                                    if window_start not in resource_index[server]:
                                        resource_index[server][window_start] = {
                                            'window_end': window_start + pd.Timedelta(hours=1)
                                        }
                                    
                                    # 将资源数据合并到主索引
                                    for key, value in resource_data.items():
                                        if key != 'window_end':  # 避免覆盖window_end
                                            resource_index[server][window_start][key] = value
                            
                            # 保存统计信息
                            resource_stats[processed_resource_type] = stats
                            
                        except Exception as e:
                            print(f"警告: 处理资源类型 {resource_type} 时出错: {type(e).__name__}: {str(e)}")
                        
                        # 更新进度条
                        pbar.update(1)
                        
                        # 定期检查内存并执行垃圾回收
                        current_time = time.time()
                        if current_time - last_mem_check > mem_check_interval:
                            gc.collect()  # 强制垃圾回收
                            mem_info = psutil.virtual_memory()
                            print(f"\n当前内存使用: {mem_info.percent}% ({mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB)")
                            last_mem_check = current_time
        except Exception as e:
            print(f"\n并行处理时发生错误: {type(e).__name__}: {str(e)}")
            print("切换到顺序处理模式...")
            
            # 如果并行处理失败，回退到顺序处理
            resource_stats = {}
            for resource_type, server_data_dict in tqdm(self.resources.items(), desc="顺序处理资源类型", unit="类型"):
                try:
                    _, sub_index, stats = self._process_resource_type(resource_type, server_data_dict, all_servers)
                    
                    # 合并索引
                    for server, windows in sub_index.items():
                        for window_start, resource_data in windows.items():
                            if server not in resource_index:
                                resource_index[server] = {}
                            if window_start not in resource_index[server]:
                                resource_index[server][window_start] = {
                                    'window_end': window_start + pd.Timedelta(hours=1)
                                }
                            
                            for key, value in resource_data.items():
                                if key != 'window_end':
                                    resource_index[server][window_start][key] = value
                    
                    resource_stats[resource_type] = stats
                except Exception as e:
                    print(f"警告: 顺序处理 {resource_type} 时出错: {type(e).__name__}: {str(e)}")
                
                # 定期回收内存
                gc.collect()
        
        # 显示资源数据统计
        print("\n资源数据索引统计:")
        for resource_type, stats in resource_stats.items():
            print(f"  - {resource_type}: {stats['server_count']} 台服务器, {stats['data_points']} 个数据点")
        
        # 统计索引大小
        server_count = len(resource_index)
        window_count = sum(len(windows) for server, windows in resource_index.items())
        
        # 输出创建时间
        elapsed_time = time.time() - start_time
        print(f"资源索引创建完成。包含 {server_count} 台服务器, {window_count} 个时间窗口。耗时: {elapsed_time:.2f} 秒")
        
        # 手动进行垃圾回收
        gc.collect()
        
        return resource_index

    def _process_resource_type(self, resource_type, server_data_dict, all_servers):
        """并行处理单个资源类型的数据并返回子索引"""
        sub_index = {}
        count = 0
        server_count = 0
        
        # 处理每个服务器的数据
        for server, df in server_data_dict.items():
            # 规范化服务器名称为小写
            server_lower = server.lower()
            
            # 跳过不在任务中使用的服务器数据
            if server_lower not in all_servers:
                continue
                
            try:
                # 检查数据帧是否有效
                if df is None or df.empty:
                    continue
                    
                # 查找时间戳列和值列
                timestamp_col = None
                value_col = 'value'  # 明确指定值列为 'value'
                
                # 检查value列是否存在
                if 'value' not in df.columns:
                    # 如果没有value列，尝试按之前的逻辑查找可能的值列
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'time' in col_lower or 'date' in col_lower:
                            timestamp_col = col
                        elif col_lower != 'server' and 'server' not in col_lower and col_lower != 'metric':
                            value_col = col
                    
                    # 如果找不到值列，跳过这个服务器
                    if value_col is None:
                        continue
                else:
                    # 确保值列中的数据是数值
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    # 删除无效的非数值行
                    df = df.dropna(subset=['value'])
                    
                # 查找时间列
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'time' in col_lower or 'date' in col_lower:
                        timestamp_col = col
                        break
                
                if timestamp_col is None or value_col is None:
                    continue
                
                # 确保时间戳格式正确
                try:
                    # 如果尚未转换为datetime，则进行转换
                    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                        df = df.dropna(subset=[timestamp_col])  # 删除无效时间戳
                except Exception as e:
                    continue
                
                # 使用更高效的向量化操作，避免逐行处理
                # 1. 计算每个时间戳对应的小时时间窗口
                df['hour_window'] = df[timestamp_col].dt.floor('H')
                
                # 2. 直接将时间窗口和对应的值转换为字典
                # 确保数据是数值类型
                if not pd.api.types.is_numeric_dtype(df[value_col]):
                    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                    df = df.dropna(subset=[value_col])
                resource_dict = df.set_index('hour_window')[value_col].to_dict()
                
                # 3. 将结果添加到资源索引
                for window_start, value in resource_dict.items():
                    # 确保服务器的索引存在
                    if server_lower not in sub_index:
                        sub_index[server_lower] = {}
                        
                    # 确保时间窗口的索引存在
                    if window_start not in sub_index[server_lower]:
                        sub_index[server_lower][window_start] = {
                            'window_end': window_start + pd.Timedelta(hours=1)
                        }
                    
                    # 添加资源值
                    sub_index[server_lower][window_start][resource_type] = value
                    count += 1
                    
                server_count += 1
                
            except Exception as e:
                print(f"警告: 处理 {server} 的 {resource_type} 数据时出错: {str(e)}")
        
        return resource_type, sub_index, {'server_count': server_count, 'data_points': count}
    
    def _parse_exec_hosts(self, host_string, row_num=None):
        """
        解析任务使用的服务器列表，处理各种复杂的服务器范围表示法
        支持的格式包括：
        1. 基本范围: 'cpu1-[41-43]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43']
        2. 逗号分隔的多个值: 'cpu1-[41,43,45]' -> ['cpu1-41', 'cpu1-43', 'cpu1-45']
        3. 复合范围: 'cpu1-[41-43,45,47-48]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43', 'cpu1-45', 'cpu1-47', 'cpu1-48']
        4. 多个服务器: 'cpu1-[41-43],cpu2-[1-3]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43', 'cpu2-1', 'cpu2-2', 'cpu2-3']
        5. 不完整范围: 'cpu1-[41-43' -> ['cpu1-41', 'cpu1-42', 'cpu1-43']
        """
        if pd.isna(host_string) or not host_string:
            return []
        
        # 标准化处理
        hosts = []
        seen = set()
        
        # 使用更智能的分割方式，避免将范围表示法错误分割
        # 例如，避免将 'cpu1-[7-9]' 分割为 'cpu1-[7' 和 '9]'
        
        # 首先检查是否包含范围表示法
        if '[' in host_string and (']' in host_string or host_string.count('[') > host_string.count(']')):
            # 包含范围表示法，需要更小心地处理
            
            # 先尝试处理逗号分隔的多个服务器
            # 但要避免将方括号内的逗号当作分隔符
            parts = []
            current_part = ""
            bracket_count = 0
            
            for char in host_string:
                if char == '[':
                    bracket_count += 1
                    current_part += char
                elif char == ']':
                    bracket_count -= 1
                    current_part += char
                elif char == ',' and bracket_count == 0:
                    # 只有在方括号外的逗号才是服务器分隔符
                    if current_part.strip():
                        parts.append(current_part.strip())
                    current_part = ""
                else:
                    current_part += char
            
            # 添加最后一部分
            if current_part.strip():
                parts.append(current_part.strip())
            
            # 处理每一部分
            for part in parts:
                if '[' in part:
                    # 包含范围表示法，使用_parse_server_range函数解析
                    try:
                        parsed_servers = _parse_server_range(part, row_num)
                        for server in parsed_servers:
                            server_clean = server.split(':')[0].lower().strip()
                            if server_clean and server_clean not in seen:
                                seen.add(server_clean)
                                hosts.append(server_clean)
                    except Exception as e:
                        print(f"    警告: 解析服务器范围 '{part}' 出错: {str(e)} （行号: {row_num}）")
                        # 如果解析失败，使用原始方式处理
                        server_clean = part.split(':')[0].lower().strip()
                        if server_clean and server_clean not in seen:
                            seen.add(server_clean)
                            hosts.append(server_clean)
                else:
                    # 普通服务器名称
                    server_clean = part.split(':')[0].lower().strip()
                    if server_clean and server_clean not in seen:
                        seen.add(server_clean)
                        hosts.append(server_clean)
        else:
            # 没有范围表示法，使用原始分割方式
            for part in re.split(r'[,\s;|]+', host_string.strip()):
                host = part.split(':')[0].lower().strip()
                if host and host not in seen:
                    seen.add(host)
                    hosts.append(host)
        
        # 输出解析结果统计
        if row_num is not None and hosts:
            if len(hosts) > 1:
                print(f"    解析服务器列表成功: 共 {len(hosts)} 台服务器 （行号: {row_num}）")
        
        return hosts

    def _hash_user_id(self, user_id):
        """对用户ID进行hash处理以保护隐私"""
        if pd.isna(user_id):
            return "n/a"
            
        try:
            # 将user_id转换为字符串并哈希
            user_str = str(user_id)
            # 使用md5生成固定长度哈希
            hashed = hashlib.md5(user_str.encode()).hexdigest()
            # 只取前8位，避免过长
            return hashed[:8]
        except Exception as e:
            print(f"警告: 用户ID哈希失败: {e}")
            return "hash_error"

    def _save_chunk_to_csv(self, chunk, file_path, header, mode):
        """将数据块保存到CSV文件"""
        chunk.to_csv(file_path, index=False, header=header, mode=mode)
        return len(chunk)

    def _save_output(self, df):
        """保存结果文件，使用分块和并行处理加速大型数据集的保存"""
        # 定义资源列表
        resource_columns = [
            'cpu_total', 'cpu_system', 'gpu_avg', 'gpu_util', 
            'mem_percent', 'mem_used', 'load_1min', 'load_15min', 
            'power', 'temperature'
        ]
        
        # 过滤数据：移除所有资源值都为N/A的记录（代表时间范围外的任务）
        original_rows = len(df)
        
        # 优化：使用向量化操作代替apply，显著提高性能
        # 创建一个布尔矩阵判断每一个资源列值是否为'n/a'
        na_matrix = df[resource_columns] == 'n/a'
        # 对每行进行全部元素是否为True的判断
        all_na_mask = na_matrix.all(axis=1)
        
        # 过滤掉所有资源都是n/a的行
        df_filtered = df[~all_na_mask]
        filtered_rows = original_rows - len(df_filtered)
        
        print(f"原始数据: {original_rows} 行，过滤掉时间范围外记录: {filtered_rows} 行 ({filtered_rows/original_rows*100:.1f}%)")
        
        # 优化：对每个任务添加job_distribution列，标识是多服务器任务还是多时间段任务
        print("正在分析任务分布特征...")
        
        # 优化：使用groupby一次性计算所有任务ID的统计信息，而不是循环处理
        print("  计算任务服务器和时间窗口统计...")
        start_time = time.time()
        
        # 对每个任务ID分组计算统计信息
        server_counts = df_filtered.groupby('job_id')['server'].nunique().to_dict()
        time_window_counts = df_filtered.groupby('job_id')['time_window_start'].nunique().to_dict()
        
        print(f"  统计计算完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"  处理 {len(server_counts)} 个独立任务ID")
        
        # 创建任务分布信息映射字典
        print("  创建任务分布信息映射...")
        start_time = time.time()
        
        # 优化：预先创建映射字典而不是使用apply函数
        distribution_mapping = {}
        for job_id in server_counts.keys():
            s_count = server_counts.get(job_id, 0)
            t_count = time_window_counts.get(job_id, 0)
            
            if s_count > 1 and t_count > 1:
                distribution_mapping[job_id] = f"多服务器多时段: {s_count}台服务器, {t_count}个时间段"
            elif s_count > 1:
                distribution_mapping[job_id] = f"多服务器: {s_count}台服务器"
            elif t_count > 1:
                distribution_mapping[job_id] = f"多时间段: {t_count}个时间段"
            else:
                distribution_mapping[job_id] = "单一服务器单一时间段"
        
        print(f"  映射创建完成，耗时: {time.time() - start_time:.2f}秒")
        
        # 优化：使用映射函数替代apply，显著提升性能
        print("  添加任务分布特征列...")
        start_time = time.time()
        
        # 使用映射函数
        df_filtered['job_distribution'] = df_filtered['job_id'].map(distribution_mapping).fillna("未知")
        
        print(f"  特征列添加完成，耗时: {time.time() - start_time:.2f}秒")
        
        total_rows = len(df_filtered)
        print(f"开始保存 {total_rows} 行数据...")
        
        # 判断数据集大小
        large_dataset = total_rows > 1000000  # 超过100万行认为是大型数据集
        
        # CSV文件路径
        csv_file = os.path.join(self.output_dir, 'integrated_data.csv')
        
        # 无论数据集大小，都保存CSV和Parquet格式
        if large_dataset:
            # 对于大型数据集，使用分块处理
            print(f"检测到大型数据集 ({total_rows} 行)，使用分块并行处理进行保存")
            
            # 获取CPU核心数量
            num_cores = multiprocessing.cpu_count()
            num_workers = max(64, int(num_cores * 0.75))  # 使用至少64个核心或核心数的75%
            
            # 优化：增大块大小，减少IO操作次数
            # 计算分块大小，根据数据量自适应调整
            chunk_size = min(1000000, max(200000, total_rows // (num_workers * 2)))
            num_chunks = (total_rows + chunk_size - 1) // chunk_size
            
            print(f"数据将分为 {num_chunks} 个块进行处理，每块约 {chunk_size} 行")
            
            # 先创建空文件并写入列名
            with open(csv_file, 'w') as f:
                f.write(','.join(df_filtered.columns) + '\n')
            
            # 优化：分块处理CSV文件，显示每秒写入速度
            processed_rows = 0
            start_time = time.time()
            with tqdm(total=total_rows, desc="保存CSV数据", unit="行") as pbar:
                for i in range(0, total_rows, chunk_size):
                    chunk_start = time.time()
                    # 获取当前块
                    end_idx = min(i + chunk_size, total_rows)
                    chunk = df_filtered.iloc[i:end_idx]
                    
                    # 写入数据，不包含列名（除了第一块）
                    chunk.to_csv(csv_file, index=False, header=False, mode='a')
                    
                    # 更新进度条
                    chunk_rows = len(chunk)
                    processed_rows += chunk_rows
                    chunk_time = time.time() - chunk_start
                    rows_per_sec = chunk_rows / max(0.1, chunk_time)
                    pbar.set_postfix({"速度": f"{rows_per_sec:.1f}行/秒"})
                    pbar.update(chunk_rows)
            
            print(f"CSV格式结果已保存至: {csv_file}")
            
        else:
            # 对于小型数据集，直接保存CSV
            print(f"正在保存CSV格式数据 ({total_rows} 行)...")
            df_filtered.to_csv(csv_file, index=False)
            print(f"CSV格式结果已保存至: {csv_file}")
        
        # 无论数据集大小，都保存Parquet格式
        parquet_file = os.path.join(self.output_dir, 'integrated_data.parquet')
        print(f"正在保存Parquet格式数据...")
        
        # 优化：使用分块保存Parquet格式，避免内存溢出
        if total_rows > 5000000:  # 对于特别大的数据集，分块保存Parquet
            print(f"数据量过大，使用分块保存Parquet格式...")
            # 创建一个临时目录存放分块Parquet文件
            temp_parquet_dir = os.path.join(self.output_dir, 'temp_parquet_chunks')
            os.makedirs(temp_parquet_dir, exist_ok=True)
            
            # 分块保存
            parquet_chunks = []
            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk = df_filtered.iloc[i:end_idx]
                chunk_file = os.path.join(temp_parquet_dir, f"chunk_{i//chunk_size}.parquet")
                chunk.to_parquet(chunk_file, engine='pyarrow', compression='snappy')
                parquet_chunks.append(chunk_file)
            
            # 合并所有分块
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            # 读取并合并所有分块
            tables = [pq.read_table(chunk) for chunk in parquet_chunks]
            combined_table = pa.concat_tables(tables)
            
            # 写入最终文件
            pq.write_table(combined_table, parquet_file, compression='snappy')
            
            # 清理临时文件
            for chunk_file in parquet_chunks:
                os.remove(chunk_file)
            os.rmdir(temp_parquet_dir)
        else:
            # 使用PyArrow引擎和Snappy压缩算法提高效率
            df_filtered.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
        
        print(f"Parquet格式结果已保存至: {parquet_file}")
        
        # 结束统计
        total_time = time.time() - self.start_time
        print(f"所有数据保存完成，共 {total_rows} 行，总处理时间: {total_time:.2f}秒")
        

    def _load_resource_data(self, resource_type, file_path):
        """
        加载特定类型的资源数据 - 优化版本
        仅加载必要的列，支持时间范围过滤，减少内存占用
        支持各种服务器名称格式和范围表示法
        """
        if not os.path.exists(file_path):
            print(f"错误: 资源文件不存在: {file_path}")
            return {}
        
        servers_data = {}
        try:
            # 导入gc模块，以便在处理大量数据时手动进行内存管理
            import gc
            
            # 获取所有工作表名称
            with pd.ExcelFile(file_path) as xls:
                sheet_names = xls.sheet_names
                
            # 不显示所有工作表名称，避免输出过多
            print(f"    可用工作表数量: {len(sheet_names)}")
            
            # 记录匹配成功和失败的服务器数量
            matched_count = 0
            failed_count = 0
            
            # 创建低内存加载选项
            excel_options = {
                'engine': 'openpyxl',  # 使用openpyxl引擎提高兼容性
            }
            
            for idx, server in enumerate(self.target_servers):
                # 添加行号参数，便于跟踪和调试
                row_num = idx + 1
                
                # 首先解析可能包含范围表示法的服务器名称
                try:
                    parsed_servers = self._parse_exec_hosts(server, row_num)
                except Exception as e:
                    print(f"    警告: 解析服务器名称 '{server}' 出错: {str(e)} （行号: {row_num}）")
                    failed_count += 1
                    continue
                
                # 如果没有解析出任何服务器，跳过
                if not parsed_servers:
                    print(f"    警告: 无法解析服务器名称 '{server}' （行号: {row_num}）")
                    failed_count += 1
                    continue
                
                # 如果解析出多个服务器，为每个服务器单独加载数据
                if len(parsed_servers) > 1:
                    print(f"    解析服务器范围 '{server}' 为 {len(parsed_servers)} 台服务器 （行号: {row_num}）")
                    
                    # 记录当前范围的匹配成功数
                    range_matched = 0
                    
                    for parsed_server in parsed_servers:
                        # 为每个解析后的服务器查找对应的工作表
                        matching_sheet = _find_matching_sheet(parsed_server, sheet_names, row_num)
                        
                        if matching_sheet:
                            print(f"    匹配成功: 服务器 {parsed_server} -> 工作表 {matching_sheet} （行号: {row_num}）")
                            try:
                                # 首先读取表头以确定必要的列
                                df_headers = pd.read_excel(file_path, sheet_name=matching_sheet, nrows=0)
                                
                                # 查找时间戳列和值列
                                timestamp_col = None
                                value_col = None
                                
                                for col in df_headers.columns:
                                    col_lower = str(col).lower()
                                    if 'time' in col_lower or 'date' in col_lower:
                                        timestamp_col = col
                                    elif col_lower != 'server' and 'server' not in col_lower:
                                        value_col = col
                                
                                if timestamp_col is None or value_col is None:
                                    print(f"    警告: {matching_sheet} 工作表缺少时间戳列或值列")
                                    failed_count += 1
                                    continue
                                
                                # 只加载必要的列以减少内存占用
                                df = pd.read_excel(
                                    file_path, 
                                    sheet_name=matching_sheet,
                                    usecols=[timestamp_col, value_col],
                                    **excel_options
                                )
                                
                                # 过滤掉缺失值
                                df = df.dropna(subset=[timestamp_col, value_col])
                                
                                if df.empty:
                                    print(f"    警告: {matching_sheet} 工作表没有有效数据 （行号: {row_num}）")
                                    failed_count += 1
                                    continue
                                
                                # 转换时间戳格式
                                df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                                df = df.dropna(subset=[timestamp_col])  # 删除无效时间戳
                                
                                # 排序数据
                                df = df.sort_values(by=timestamp_col)
                                
                                # 仅保留时间戳和值列
                                df = df[[timestamp_col, value_col]]
                                
                                # 保存到服务器数据字典
                                servers_data[parsed_server] = df
                                print(f"    成功加载 {parsed_server} 数据: {len(df)} 条记录")
                                range_matched += 1
                                matched_count += 1
                                
                                # 手动回收内存
                                if idx % 10 == 0:
                                    gc.collect()
                                
                            except Exception as e:
                                print(f"    错误: 无法读取 {matching_sheet} 工作表: {str(e)} （行号: {row_num}）")
                                failed_count += 1
                        else:
                            print(f"    警告: 服务器 {parsed_server} 没有匹配的工作表 （行号: {row_num}）")
                            failed_count += 1
                    
                    # 显示范围匹配统计
                    if range_matched > 0:
                        print(f"    范围匹配统计: '{server}' -> 成功匹配 {range_matched}/{len(parsed_servers)} 台服务器 （行号: {row_num}）")
                else:
                    # 单个服务器的情况
                    actual_server = parsed_servers[0]
                    
                    # 查找匹配的工作表
                    matching_sheet = _find_matching_sheet(actual_server, sheet_names, row_num)
                    
                    if matching_sheet:
                        print(f"    匹配成功: 服务器 {server} -> 工作表 {matching_sheet} （行号: {row_num}）")
                        try:
                            # 首先读取表头以确定必要的列
                            df_headers = pd.read_excel(file_path, sheet_name=matching_sheet, nrows=0)
                            
                            # 查找时间戳列和值列
                            timestamp_col = None
                            value_col = None
                            
                            for col in df_headers.columns:
                                col_lower = str(col).lower()
                                if 'time' in col_lower or 'date' in col_lower:
                                    timestamp_col = col
                                elif col_lower != 'server' and 'server' not in col_lower:
                                    value_col = col
                                
                            if timestamp_col is None or value_col is None:
                                print(f"    警告: {matching_sheet} 工作表缺少时间戳列或值列")
                                failed_count += 1
                                continue
                            
                            # 只加载必要的列以减少内存占用
                            df = pd.read_excel(
                                file_path, 
                                sheet_name=matching_sheet,
                                usecols=[timestamp_col, value_col],
                                **excel_options
                            )
                            
                            # 过滤掉缺失值
                            df = df.dropna(subset=[timestamp_col, value_col])
                            
                            if df.empty:
                                print(f"    警告: {matching_sheet} 工作表没有有效数据 （行号: {row_num}）")
                                failed_count += 1
                                continue
                            
                            # 转换时间戳格式
                            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
                            df = df.dropna(subset=[timestamp_col])  # 删除无效时间戳
                            
                            # 排序数据
                            df = df.sort_values(by=timestamp_col)
                            
                            # 仅保留时间戳和值列
                            df = df[[timestamp_col, value_col]]
                            
                            # 保存到服务器数据字典
                            servers_data[actual_server] = df
                            print(f"    成功加载 {actual_server} 数据: {len(df)} 条记录")
                            matched_count += 1
                            
                            # 手动回收内存
                            if idx % 10 == 0:
                                gc.collect()
                                
                        except Exception as e:
                            print(f"    错误: 无法读取 {matching_sheet} 工作表: {str(e)} （行号: {row_num}）")
                            failed_count += 1
                    else:
                        print(f"    警告: 服务器 {server} 没有匹配的工作表 （行号: {row_num}）")
                        failed_count += 1
            
            # 显示最终统计信息
            total = len(self.target_servers)
            print(f"\n    服务器匹配统计: 成功 {matched_count}/{total} 台 (成功率: {matched_count/total*100:.1f}%), 失败 {failed_count} 台")
            
            # 最终内存回收
            gc.collect()
            
            return servers_data
        except Exception as e:
            print(f"错误: 无法加载资源文件 {file_path}: {str(e)}")
            traceback.print_exc()
            return {}

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='资源与任务整合分析工具')
    parser.add_argument('--test', action='store_true', help='运行测试模式（每个文件只读取前N行）')
    parser.add_argument('--rows', type=int, default=100, help='测试模式下读取的行数')
    parser.add_argument('--output', type=str, help='指定输出目录')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output if args.output else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if args.test:
        output_dir = os.path.join(output_dir, f'test_rows{args.rows}')
        print(f"=== 运行快速测试模式 (每个文件前 {args.rows} 行) ===")
    else:
        output_dir = os.path.join(output_dir, f'full_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        print("=== 运行完整分析模式 ===")
    
    print(f"测试输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 只读取一次任务数据
    print("开始资源与任务整合分析...")
    jobs = read_job_data(DATA_PATHS['job_data'], args.test, args.rows)
    if jobs is None or len(jobs) == 0:
        print("错误: 任务数据为空，无法继续分析")
        return 1
    
    # 从已读取的DataFrame中提取元数据
    pre_data = extract_job_metadata(jobs)
    print(f"需要处理的服务器数量: {len(pre_data['servers'])}")
    print(f"任务时间范围: {pre_data['time_range'][0]} 至 {pre_data['time_range'][1]}")
    
    # 修复服务器名称匹配问题 (bigmem -> bigmen)
    corrected_servers = []
    for server in pre_data['servers']:
        if 'bigmem' in server:
            corrected_servers.append(server.replace('bigmem', 'bigmen'))
        else:
            corrected_servers.append(server)
    pre_data['servers'] = corrected_servers
    
    # 加载目标资源数据
    resources = load_target_resources(pre_data)
    
    # 检查是否有有效的资源数据
    has_data = False
    for resource_type, data in resources.items():
        if data:  # 如果至少有一个服务器的数据
            has_data = True
            break
    
    if not has_data:
        print("错误: 所有资源数据都为空，无法继续分析")
        print("可能原因: 1) 任务时间范围内无资源数据 2) 服务器名称不匹配")
        print(f"建议: 检查 {pre_data['time_range'][0]} 至 {pre_data['time_range'][1]} 期间是否有资源数据")
        return 1
    
    # 初始化分析器 (使用已读取的任务数据和output_dir)
    analyzer = ResourceJobAnalyzer(jobs, resources, output_dir)
    
    result = analyzer.process()
    
    print(f"分析完成。结果已保存到: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())