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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 调试输出函数
def print_debug(message, verbose=False, prefix="DEBUG"):
    """根据设置决定是否输出调试信息
    
    参数:
        message: 要输出的消息
        verbose: 是否处于详细输出模式
        prefix: 输出前缀，默认为"DEBUG"
    """
    if verbose:
        print(f"[{prefix}] {message}")

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
    增强版exec_hosts字段解析器，支持各种服务器名称和范围格式
    支持的格式包括：
    1. 基本范围: 'cpu1-[41-43]' -> ['cpu1-41', 'cpu1-42', 'cpu1-43']
    2. 逗号分隔值: 'cpu1-[41,43,45]' -> ['cpu1-41', 'cpu1-43', 'cpu1-45']
    3. 复合范围: 'cpu1-[41-43,45,47-48]' -> ['cpu1-41', ..., 'cpu1-48']
    4. 多个服务器: 'cpu1-[41-43],cpu2-[1-3]' -> ['cpu1-41', ..., 'cpu2-3']
    5. 特殊嵌套格式: 'cpu1-1+cpu1-[2,3]+cpu1-[4-6]' -> ['cpu1-1', 'cpu1-2', 'cpu1-3', 'cpu1-4', 'cpu1-5', 'cpu1-6']
    6. JSON格式: '{"nodes":["cpu1-1","cpu1-2"]}' -> ['cpu1-1', 'cpu1-2']
    7. 字典格式: "host=cpu1-1 task=3" -> ['cpu1-1']
    
    参数:
        exec_hosts: 原始服务器字符串
        row_num: 行号，用于调试跟踪
    返回:
        标准化的服务器名称列表
    """
    # 设置默认的verbose值
    verbose = getattr(parse_exec_hosts, 'verbose', False)
    # 详细的解析调试信息
    if verbose:
        print(f"[parse_exec_hosts] 开始解析: '{exec_hosts}' {'(行:'+str(row_num)+')' if row_num is not None else ''}")
    if pd.isna(exec_hosts) or not exec_hosts or not str(exec_hosts).strip():
        if verbose:
            print(f"[parse_exec_hosts] 输入为空或无效")
        return []
    
    # 处理结果
    hosts = []
    seen = set()
    
    # 尝试处理可能的特殊格式
    # 1. 检查JSON格式
    if exec_hosts.strip().startswith('{') and exec_hosts.strip().endswith('}'):
        try:
            json_data = json.loads(exec_hosts)
            if 'nodes' in json_data and isinstance(json_data['nodes'], list):
                hosts = [host.lower() for host in json_data['nodes'] if host]

                return hosts
        except json.JSONDecodeError:
            pass
    
    # 2. 检查字典格式字符串（如"host=cpu1-1 task=3"）
    if '=' in exec_hosts:
        host_matches = re.findall(r'(?:host|node)=([^\s,;]+)', exec_hosts, re.IGNORECASE)
        if host_matches:
            hosts = [host.lower() for host in host_matches if host]

            return hosts
            
    # 3. 检查加号连接的多个服务器（如"cpu1-1+cpu1-2"）
    if '+' in exec_hosts:
        parts = exec_hosts.split('+')
        hosts = []
        for part in parts:
            part_hosts = parse_exec_hosts(part, row_num)  # 递归解析每一部分
            hosts.extend(part_hosts)
        if hosts:

            return hosts
    
    # 标准格式处理：智能分割支持范围表示法，避免将 'cpu1-[7-9]' 分割为 'cpu1-[7' 和 '9]'
    if '[' in exec_hosts:
        # 先尝试处理逗号分隔的多个服务器，避免将方括号内的逗号当作分隔符
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
                except Exception:
                    # 解析失败时处理原始字符串
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
        # 没有范围表示法，使用简单分割
        for part in re.split(r'[,\s;|]+', exec_hosts.strip()):
            host = part.split(':')[0].lower().strip()
            if host and host not in seen:
                seen.add(host)
                hosts.append(host)
    
    return hosts

def _read_single_sheet_safe(excel_path, sheet_name, resource_type, test_mode=False, rows_limit=None):
    """安全地读取单个工作表的数据 - 优化版
    使用文件路径而不是Excel对象，避免并发访问，并使用数据类型规范和高效的数据处理
    """
    try:
        # 决定是否限制读取行数
        nrows = rows_limit if test_mode and rows_limit else None
        
        # 优化：为timestamp和value指定数据类型，更高效地读取
        dtype_dict = {'value': 'float32'}  # 使用更低精度的数据类型以减少内存使用
        usecols = ['timestamp', 'value']   # 只读取必要的列
        
        # 先尝试获取文件大小以决定读取策略
        file_size = os.path.getsize(excel_path)
        large_file = file_size > 50 * 1024 * 1024  # 50MB
        
        # 对于非常大的Excel文件，执行额外的优化
        if large_file and not test_mode:
            # 大文件先尝试读取前几行确定数据结构
            header_df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=5)
            if 'timestamp' not in header_df.columns or 'value' not in header_df.columns:
                print(f"  工作表 {sheet_name} 缺少必要列: timestamp或value")
                return None
                
            try:
                # 分批读取数据以减少内存压力
                chunks = []
                for chunk in pd.read_excel(
                    excel_path, 
                    sheet_name=sheet_name,
                    usecols=usecols,
                    dtype=dtype_dict,
                    chunksize=100000,  # 每批10万行
                    nrows=nrows
                ):
                    # 立即处理时间戳列和值列，然后追加
                    try:
                        # 转换时间戳
                        if chunk['timestamp'].dtype != 'datetime64[ns]':
                            chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], errors='coerce')
                            
                        # 确保值为数值类型
                        if not pd.api.types.is_numeric_dtype(chunk['value']):
                            chunk['value'] = pd.to_numeric(chunk['value'], errors='coerce')
                        
                        # 过滤掉无效值
                        chunk = chunk.dropna(subset=['timestamp', 'value'])
                        chunks.append(chunk)
                    except Exception as e:
                        print(f"  处理工作表 {sheet_name} 数据块时出错: {str(e)}")
                        continue
                    
                    # 主动垃圾回收
                    gc.collect()
                
                if not chunks:
                    return None
                    
                # 合并所有数据块
                df = pd.concat(chunks, ignore_index=True)
                del chunks
                gc.collect()
            except Exception as e:
                print(f"  分批读取工作表 {sheet_name} 时出错: {str(e)}")
                # 失败时回退到标准读取
                df = pd.read_excel(
                    excel_path, 
                    sheet_name=sheet_name, 
                    usecols=usecols,
                    dtype=dtype_dict,
                    nrows=nrows
                )
        else:
            # 中小文件标准读取
            df = pd.read_excel(
                excel_path, 
                sheet_name=sheet_name, 
                usecols=usecols,
                dtype=dtype_dict,
                nrows=nrows
            )
        
        # 检查必要列
        required_cols = ['timestamp', 'value']
        for col in required_cols:
            if col not in df.columns:
                print(f"  工作表 {sheet_name} 缺少必要列: {col}")
                return None
        
        # 确保timestamp列为日期时间类型
        if df['timestamp'].dtype != 'datetime64[ns]':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                # 过滤掉无效的时间戳
                df = df.dropna(subset=['timestamp'])
            except Exception as e:
                print(f"  工作表 {sheet_name} 时间戳列转换失败: {str(e)}")
                return None
        
        # 确保value列为数值类型
        if not pd.api.types.is_numeric_dtype(df['value']):
            try:
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df.dropna(subset=['value'])
            except Exception as e:
                print(f"  工作表 {sheet_name} 值列转换失败: {str(e)}")
                return None
        
        # 删除重复数据 - 有些资源数据可能存在重复测量值
        df = df.drop_duplicates(['timestamp']).reset_index(drop=True)
        
        # 对时间戳进行排序，以确保数据按时间顺序排列
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        # 添加指标名称列(如果不存在)
        if 'metric' not in df.columns:
            df['metric'] = resource_type
        
        # 降低内存使用 - 优化数据类型
        df['value'] = df['value'].astype('float32')
        
        return df
    except Exception as e:
        print(f"  读取工作表 {sheet_name} 时出错: {type(e).__name__}: {str(e)}")
        return None

def _process_resource_file(file_path, resource_type, test_mode=False, rows_limit=None):
    """处理单个资源文件的所有工作表 - 优化版
    实现高效的Excel文件处理，包括并行加载工作表、内存优化和进度跟踪
    """
    filename = os.path.basename(file_path)
    print(f"处理{resource_type}文件: {filename}")
    start_time = time.time()
    
    try:
        # 检查文件是否存在且可读
        if not os.path.exists(file_path) or not os.access(file_path, os.R_OK):
            print(f"  错误: 文件不存在或无法访问: {file_path}")
            return {}
            
        # 获取文件大小，以便决定加载策略
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        print(f"  文件大小: {file_size:.2f} MB")
        
        # 先安全地获取工作表列表
        try:
            # 使用context manager确保Excel文件正确关闭
            with pd.ExcelFile(file_path) as xls:
                sheet_names = xls.sheet_names
        except Exception as e:
            print(f"  获取工作表列表失败: {type(e).__name__}: {str(e)}")
            return {}
        
        print(f"  发现 {len(sheet_names)} 个工作表")
            
        # 筛选要处理的工作表
        valid_sheets = []
        for sheet in sheet_names:
            # 跳过不符合要求的工作表
            should_exclude = False
            for pattern in EXCLUDE_PATTERNS:
                try:
                    if re.match(pattern, sheet):
                        should_exclude = True
                        break
                except Exception:
                    continue  # 如果正则表达式出错，跳过此模式
            
            # 只添加有效的工作表
            if not should_exclude:
                # 预先过滤掉明显无效的名称
                normalized_name = _normalize_server_name(sheet)
                if normalized_name:  # 确保标准化后的名称不为空
                    valid_sheets.append(sheet)
        
        print(f"  有效工作表: {len(valid_sheets)} 个")
        
        if test_mode and rows_limit:
            print(f"  测试模式: 只处理每个工作表的前 {rows_limit} 行")
        
        # 创建一个字典来存储服务器数据
        server_data = {}
        
        # 决定是否使用并行处理
        use_parallel = len(valid_sheets) > 20 and not test_mode and file_size < 500  # 500MB以下的文件启用并行
        
        if use_parallel:
            # 实现并行工作表处理 - 适用于大量小型工作表的情况
            print(f"  使用并行处理加载 {len(valid_sheets)} 个工作表")
            
            # 定义工作表处理函数
            def process_sheet(sheet_name):
                try:
                    data = _read_single_sheet_safe(file_path, sheet_name, resource_type, test_mode, rows_limit)
                    if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                        normalized_name = _normalize_server_name(sheet_name)
                        if normalized_name:  # 确保标准化后的名称不为空
                            return (normalized_name, data)
                except Exception as e:
                    print(f"  处理工作表 {sheet_name} 时出错: {type(e).__name__}: {str(e)}")
                return None
            
            # 确定最佳线程数，不超过CPU核心数的一半
            num_workers = min(len(valid_sheets), multiprocessing.cpu_count() // 2 or 1)
            
            # 使用线程池并行处理工作表
            results = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # 提交所有任务
                future_to_sheet = {executor.submit(process_sheet, sheet_name): sheet_name for sheet_name in valid_sheets}
                
                # 收集结果
                for future in tqdm(as_completed(future_to_sheet), total=len(valid_sheets), 
                                   desc=f"{resource_type} 工作表处理"):
                    sheet_name = future_to_sheet[future]
                    try:
                        result = future.result()
                        if result is not None:
                            normalized_name, data = result
                            server_data[normalized_name] = data
                    except Exception as e:
                        print(f"  处理工作表 {sheet_name} 结果时出错: {type(e).__name__}: {str(e)}")
        else:
            # 顺序处理工作表 - 适用于少量大型工作表或测试模式
            with tqdm(valid_sheets, desc=f"{resource_type} 工作表处理", position=0, leave=True) as progress_bar:
                for sheet_name in progress_bar:
                    try:
                        # 更新进度条描述
                        progress_bar.set_description(f"处理: {sheet_name[:15]}..." if len(sheet_name) > 15 else f"处理: {sheet_name}")
                        
                        # 读取工作表数据
                        data = _read_single_sheet_safe(file_path, sheet_name, resource_type, test_mode, rows_limit)
                        
                        # 如果成功读取，添加到结果字典
                        if data is not None and isinstance(data, pd.DataFrame) and not data.empty:
                            normalized_name = _normalize_server_name(sheet_name)
                            if normalized_name:  # 确保标准化后的名称不为空
                                server_data[normalized_name] = data
                        
                        # 定期进行垃圾回收
                        if len(server_data) % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        print(f"  处理工作表 {sheet_name} 时出错: {type(e).__name__}: {str(e)}")
        
        # 添加详细日志，显示读取到的服务器名称
        elapsed_time = time.time() - start_time
        if server_data:
            print(f"  成功读取 {len(server_data)} 台服务器的{resource_type}数据，耗时: {elapsed_time:.2f}秒")
            
            # 报告内存使用情况
            mem_usage = 0
            for server, data in server_data.items():
                mem_usage += data.memory_usage(deep=True).sum()
            mem_usage_mb = mem_usage / (1024 * 1024)
            print(f"  数据内存占用: {mem_usage_mb:.2f} MB")
            
            if len(server_data) <= 10:  # 如果服务器数量较少，全部显示
                print(f"  服务器列表: {', '.join(server_data.keys())}")
            else:  # 仅显示前5个和后5个
                server_names = list(server_data.keys())
                print(f"  服务器示例: {', '.join(server_names[:5])} ... {', '.join(server_names[-5:])}")
        else:
            print(f"  警告: 未找到任何有效的服务器{resource_type}数据，耗时: {elapsed_time:.2f}秒")
        
        return server_data
    
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"处理{resource_type}文件时出错: {type(e).__name__}: {str(e)}，耗时: {elapsed_time:.2f}秒")
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
    """读取任务提交数据 - 优化版本
    仅加载必要的列，使用数据类型规范，并支持大文件的分块处理
    """
    filename = os.path.basename(file_path)
    print(f"\n正在读取任务数据: {filename}")
    start_time = time.time()
    
    try:
        # 定义需要加载的列和数据类型
        needed_columns = [
            'job_id', 'user_id', 'submit_time', 'start_time', 'end_time',
            'job_name', 'exec_hosts', 'first_exec_host', 'job_status',
            'queue', 'num_processors', 'gpu_num'
        ]
        
        # 定义数据类型字典，提高加载效率和降低内存使用
        dtype_dict = {
            'job_id': str,
            'user_id': str,
            'job_name': str,
            'exec_hosts': str,
            'first_exec_host': str,
            'queue': str,
            'job_status': str
        }
        
        # 估计文件大小，决定是否使用分块处理
        file_size = os.path.getsize(file_path)
        large_file_threshold = 100 * 1024 * 1024  # 100MB
        use_chunked_processing = file_size > large_file_threshold and not test_mode
        
        if use_chunked_processing:
            print(f"  检测到大文件 ({file_size/1024/1024:.1f} MB)，使用分块处理")
            chunk_size = 250000  # 每批25万行
            
            # 先加载头部确定可用列
            header_df = pd.read_csv(file_path, nrows=5)
            usecols = [col for col in needed_columns if col in header_df.columns]
            
            # 创建分块处理进度条
            chunks = []
            with tqdm(desc="读取CSV数据块", unit="块") as pbar:
                # 使用分块读取CSV
                for chunk in pd.read_csv(
                    file_path, 
                    usecols=usecols,
                    dtype={col: dtype_dict.get(col) for col in usecols if col in dtype_dict},
                    chunksize=chunk_size,
                    low_memory=False
                ):
                    # 测试模式下限制行数
                    if test_mode and len(chunks) * chunk_size + len(chunk) >= max_rows:
                        chunk = chunk.head(max_rows - len(chunks) * chunk_size)
                        chunks.append(chunk)
                        break
                    
                    chunks.append(chunk)
                    pbar.update(1)
                    
                    # 测试模式下检查是否已达到最大行数
                    if test_mode and len(chunks) * chunk_size >= max_rows:
                        break
                        
            # 合并所有数据块
            df = pd.concat(chunks, ignore_index=True)
            # 释放内存
            del chunks
            gc.collect()
        else:
            if test_mode:
                print(f"  测试模式: 读取约 {max_rows} 行数据")
                
                # 对于测试模式，我们可以使用更高效的抽样方法
                estimated_total_rows = 8877313 if file_size > 100000000 else 1000000
                
                # 如果文件很大，使用跳跃式读取
                if estimated_total_rows > max_rows * 10:
                    # 计算抽样间隔
                    skip_interval = max(1, estimated_total_rows // max_rows)
                    
                    # 先加载头部确定可用列
                    header_df = pd.read_csv(file_path, nrows=5)
                    usecols = [col for col in needed_columns if col in header_df.columns]
                    
                    # 使用随机抽样行
                    np.random.seed(42)  # 设置随机种子
                    skip_rows = sorted(np.random.choice(
                        range(1, estimated_total_rows), 
                        estimated_total_rows - max_rows, 
                        replace=False
                    ))
                    
                    df = pd.read_csv(
                        file_path,
                        usecols=usecols,
                        dtype={col: dtype_dict.get(col) for col in usecols if col in dtype_dict},
                        skiprows=skip_rows,
                        low_memory=False
                    )
                    df = df.head(max_rows)  # 确保不超过最大行数
                else:
                    # 直接读取然后抽样
                    header_df = pd.read_csv(file_path, nrows=5)
                    usecols = [col for col in needed_columns if col in header_df.columns]
                    
                    df = pd.read_csv(
                        file_path,
                        usecols=usecols,
                        dtype={col: dtype_dict.get(col) for col in usecols if col in dtype_dict},
                        low_memory=False
                    )
                    if len(df) > max_rows:
                        df = df.sample(max_rows, random_state=42)
            else:
                # 正常模式，读取全部数据
                header_df = pd.read_csv(file_path, nrows=5)
                usecols = [col for col in needed_columns if col in header_df.columns]
                
                df = pd.read_csv(
                    file_path,
                    usecols=usecols,
                    dtype={col: dtype_dict.get(col) for col in usecols if col in dtype_dict},
                    low_memory=False
                )
        
        print(f"  原始任务数据读取完成。记录数: {len(df)}")
        
        # 检查必要列
        required_cols = ['job_id', 'user_id', 'submit_time', 'start_time', 'end_time']
        for col in required_cols:
            if col not in df.columns:
                print(f"警告: 任务数据缺少必要列 '{col}'")
                return None
        
        # 使用向量化操作快速转换时间字段
        print("  转换时间字段...")
        time_cols = ['submit_time', 'start_time', 'end_time']
        
        # 使用并行处理转换时间列
        def convert_time_column(df, col):
            return pd.to_datetime(df[col], errors='coerce')
        
        with ThreadPoolExecutor(max_workers=min(len(time_cols), 3)) as executor:
            # 提交所有转换任务
            future_to_col = {executor.submit(convert_time_column, df, col): col for col in time_cols}
            
            # 收集结果并更新DataFrame
            for future in tqdm(as_completed(future_to_col), total=len(time_cols), desc="时间字段转换"):
                col = future_to_col[future]
                try:
                    df[col] = future.result()
                except Exception as e:
                    print(f"转换时间列 {col} 时出错: {str(e)}")
        
        # 创建服务器名称解析缓存
        server_parse_cache = {}
        
        def parse_servers_optimized(host_string, row_num=None, first_exec_host=None):
            """优化版服务器解析函数，使用缓存提高效率"""
            # 检查缓存
            if pd.notna(host_string) and host_string in server_parse_cache:
                return server_parse_cache[host_string]
                
            # 如果exec_hosts为空，尝试使用first_exec_host
            if (pd.isna(host_string) or not host_string) and pd.notna(first_exec_host):
                result = [first_exec_host]
                server_parse_cache[str(host_string)] = result
                return result
                
            # 使用现有的extract_server_id_from_exec_hosts函数
            result = extract_server_id_from_exec_hosts(host_string, row_num)
            
            # 更新缓存
            if pd.notna(host_string):
                server_parse_cache[str(host_string)] = result
                
            return result
        
        # 提取服务器ID列表
        print("  解析服务器列表...")
        
        # 使用优化的向量化操作批量处理
        total_rows = len(df)
        
        # 根据内存情况动态调整批处理大小
        mem_info = {}
        try:
            with open('/proc/meminfo') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        mem_info[key.strip()] = int(value.strip().split()[0])  # 以KB为单位
            
            available_mem = mem_info.get('MemAvailable', 0) / 1024 / 1024  # 转换为GB
            print(f"  当前可用内存: {available_mem:.1f}GB")
            
            # 根据可用内存调整批大小
            if available_mem < 50:  # 小于50GB
                batch_size = 2000  # 约减为原月2000行
                print("  ⚠️ 内存非常紧张，采用极小批处理大小: 2000行")
            elif available_mem < 100:  # 小于100GB
                batch_size = 5000
                print("  ⚠️ 内存紧张，采用较小批处理大小: 5000行")
            else:
                batch_size = 10000  # 每批处理1万行
                print("  内存充足，使用标准批处理大小: 10000行")
        except Exception as e:
            print(f"  警告: 获取内存信息失败: {e}，使用安全的小批量大小")
            batch_size = 5000  # 失败时使用安全的小值
        
        num_batches = (total_rows + batch_size - 1) // batch_size
        print(f"  将分{num_batches}批处理数据，每批{batch_size}行")
        
        # 初始化存储解析结果的列
        df['parsed_servers'] = None
        
        with tqdm(total=total_rows, desc="解析服务器列表") as pbar:
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_rows)
                batch = df.iloc[start_idx:end_idx]
                
                # 使用预处理的函数添加服务器解析
                batch_results = []
                for idx, row in batch.iterrows():
                    host_string = row.get('exec_hosts', '')
                    first_host = row.get('first_exec_host', None)
                    result = parse_servers_optimized(host_string, idx, first_host)
                    # 将服务器列表转换为JSON字符串以避免形状不一致的错误
                    batch_results.append(json.dumps(result))
                
                # 更新结果
                df.loc[start_idx:end_idx-1, 'parsed_servers'] = batch_results
                pbar.update(end_idx - start_idx)
                
                # 周期性释放内存
                if i % 10 == 0:
                    gc.collect()
        
        # 计算任务相关时间 - 使用向量化操作
        print("  计算任务相关时间...")
        df['queue_time'] = ((df['start_time'] - df['submit_time']).dt.total_seconds() / 60)
        df['run_time'] = ((df['end_time'] - df['start_time']).dt.total_seconds() / 60)
        
        # 处理请求的资源 - 使用向量化操作替代fillna
        for col in ['num_processors', 'gpu_num']:
            if col in df.columns:
                # 使用mask操作更高效
                mask = df[col].isna()
                if mask.any():
                    df.loc[mask, col] = 0
        
        # 转换为整数类型
        if 'num_processors' in df.columns:
            df['requested_processors'] = df['num_processors'].astype('int32')
        else:
            df['requested_processors'] = 0
            
        if 'gpu_num' in df.columns:
            df['requested_gpu'] = df['gpu_num'].astype('int32')
        else:
            df['requested_gpu'] = 0
        
        # 过滤无效任务（结束时间为空或负的运行时间）
        valid_jobs = df.dropna(subset=['end_time']).copy()
        valid_jobs = valid_jobs[valid_jobs['run_time'] > 0]
        
        # 显示内存使用情况
        mem_usage = valid_jobs.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"  任务数据内存占用: {mem_usage:.2f} MB")
        
        elapsed_time = time.time() - start_time
        print(f"任务数据处理完成。耗时: {elapsed_time:.2f}秒")
        print(f"有效任务总数: {len(valid_jobs):,}")
        return valid_jobs
    
    except Exception as e:
        print(f"读取任务数据时出错: {type(e).__name__}: {str(e)}")
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
    统一的服务器范围表示法解析器，处理各种格式
    支持的格式包括：
    1. 方括号范围: 'cpu1-[7-9]' -> ['cpu1-7', 'cpu1-8', 'cpu1-9']
    2. 逗号分隔值: 'cpu1-[13,15,17]' -> ['cpu1-13', 'cpu1-15', 'cpu1-17']
    3. 复合范围: 'cpu1-[15,96-97]' -> ['cpu1-15', 'cpu1-96', 'cpu1-97']
    4. 连字符范围: 'cpu1-103-104' -> ['cpu1-103', 'cpu1-104']
    5. 不完整范围: 'cpu1-[7-9' -> ['cpu1-7', 'cpu1-8', 'cpu1-9']
    """
    # 验证输入
    if not server_name:
        return []
    
    # 调试信息前缀
    row_info = f" (行 {row_num})" if row_num is not None else ""
    
    try:
        # 处理方括号范围表示法
        if '[' in server_name:
            # 提取服务器基本名称前缀
            prefix = server_name[:server_name.find('[')]
            
            # 处理可能缺失右括号的情况
            if ']' not in server_name:
                range_str = server_name[server_name.find('[')+1:]
            else:
                range_str = server_name[server_name.find('[')+1:server_name.find(']')]
            
            result = []
            # 处理逗号分隔的多个范围或单值
            parts = [p.strip() for p in range_str.split(',') if p.strip()]
            
            for part in parts:
                if '-' in part:
                    # 范围表示法，如 '7-9'
                    try:
                        start, end = map(int, part.split('-'))
                        for i in range(start, end + 1):
                            result.append(f"{prefix}{i}")
                    except ValueError:
                        # 当解析失败时，保留原始字符串
                        result.append(f"{prefix}{part}")
                else:
                    # 单个值，如 '7'
                    try:
                        # 尝试数字解析，如果成功则使用数字
                        int(part)
                        result.append(f"{prefix}{part}")
                    except ValueError:
                        # 非数字，直接添加
                        result.append(f"{prefix}{part}")
            
            return result if result else [server_name]
            
        # 处理连字符范围表示法，如 'cpu1-103-104'
        elif '-' in server_name:
            # 检查是否符合 prefix-num1-num2 格式
            pattern = re.match(r'([a-zA-Z]+[0-9]*-[0-9]+)-([0-9]+)$', server_name)
            if pattern:
                first_part, end_num = pattern.groups()
                # 提取前缀和起始数字
                prefix_pattern = re.match(r'([a-zA-Z]+[0-9]*)-([0-9]+)', first_part)
                if prefix_pattern:
                    prefix, start_num = prefix_pattern.groups()
                    
                    # 转换为整数并生成范围
                    try:
                        start = int(start_num)
                        end = int(end_num)
                        return [f"{prefix}-{i}" for i in range(start, end + 1)]
                    except ValueError:
                        pass
            
            # 如果上面的模式不匹配或处理失败，尝试其他格式
            # 如这种格式: 'gpu-node1-3' -> ['gpu-node1', 'gpu-node2', 'gpu-node3']
            parts = server_name.rsplit('-', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_part = parts[0]
                end_num = int(parts[1])
                
                # 尝试提取前缀中的数字部分 'gpu-node1' -> 'gpu-node', '1'
                match = re.search(r'(.*?)(\d+)$', base_part)
                if match:
                    prefix = match.group(1)  # 非数字部分
                    start_num = int(match.group(2))  # 起始数字
                    
                    # 生成范围内的服务器
                    if start_num <= end_num:
                        return [f"{prefix}{i}" for i in range(start_num, end_num + 1)]
        
        # 未匹配任何特殊格式，返回原始服务器名称
        return [server_name]
        
    except Exception as e:
        # 出现异常时返回原始名称
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
    def __init__(self, jobs_data, resource_data, output_path, pre_extracted_servers=None):
        """
        初始化分析器
        :param jobs_data: 预处理后的任务数据 (DataFrame)
        :param resource_data: 加载的资源数据 (dict)
        :param output_path: 结果输出目录
        :param pre_extracted_servers: 预先提取的服务器名称列表，避免重复提取
        """
        self.jobs = jobs_data
        self.resources = resource_data
        self.output_dir = output_path
        self.pre_extracted_servers = pre_extracted_servers or []
        self.start_time = time.time()  # 记录初始化时间作为开始时间
        self.verbose = False  # 默认不输出详细的调试信息
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
        # 调试信息
        if self.verbose:
            print_debug(f"资源索引服务器列表: {list(resource_index.keys())[:5]}...", True)
            
            if resource_index:
                first_server = next(iter(resource_index.keys()))
                if resource_index[first_server]:
                    first_time = next(iter(resource_index[first_server].keys()))
                    print_debug(f"资源数据示例: {resource_index[first_server][first_time]}", True)
                    
            print_debug(f"资源列: {resource_columns}", True)
        
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
            
            # 使用预解析的服务器列表 (存储为JSON字符串)
            parsed_servers_value = job.get('parsed_servers', '[]')
            
            # 如果是字符串格式，尝试解析JSON
            if isinstance(parsed_servers_value, str):
                try:
                    servers = json.loads(parsed_servers_value)
                except json.JSONDecodeError:
                    servers = []
            else:
                servers = parsed_servers_value or []
                
            if not servers:
                # 如果没有预解析的服务器列表，则回退到原始的解析方法
                exec_hosts_str = job.get('exec_hosts', '')
                # 添加调试信息

                
                # 确保对服务器列表进行正确解析，返回列表而不是字符串
                servers = self._parse_exec_hosts(exec_hosts_str)
                
                # 调试信息 - 检查解析结果

                
                # 确保 servers 是列表而不是字符串
                if isinstance(servers, str):

                    servers = [servers]  # 将字符串转换为单元素列表
                
                # 如果没有解析到服务器，试回getting first_exec_host
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
            
            # 去除重复的服务器名称
            unique_servers = list(dict.fromkeys(servers)) if isinstance(servers, list) else [servers]
            print(f"DEBUG: 唯一服务器列表: {unique_servers}, 类型: {type(unique_servers)}")
            
            # 再次检查unique_servers是否是字符串，如果是，将其转换为列表
            if isinstance(unique_servers, str):
                print(f"DEBUG: 警告! unique_servers是字符串而非列表: '{unique_servers}'")
                # 将字符串转为单元素列表，而不是字符列表
                unique_servers = [unique_servers]
            
            # 保存原始的exec_hosts值，避免使用解析后的值
            original_exec_hosts = job.get('exec_hosts', 'n/a')
            if self.verbose:
                print_debug(f"开始遍历 {len(unique_servers)} 个服务器", True)
            
            # 为每个服务器和任务时间窗口创建行
            # 特别保障措施：确保我们不会逐字符迭代字符串
            if len(unique_servers) == 1 and isinstance(unique_servers[0], str) and len(unique_servers[0]) > 1:
                if self.verbose:
                    print_debug(f"使用安全模式遍历服务器列表，防止逐字符迭代", True)
                servers_to_process = unique_servers
            else:
                servers_to_process = unique_servers
                
            for server in servers_to_process:
                # 确保服务器名称是完整的名称，而不是单个字符
                if self.verbose:
                    print_debug(f"处理服务器: '{server}', 类型: {type(server)}", True)
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
                    'exec_hosts': original_exec_hosts,  # 使用原始exec_hosts值
                    'submit_time': job_submit,
                    'start_time': job_start,
                    'end_time': job_end,
                    'job_duration': job_duration,
                    'queue_time': queue_time,
                    'server': server,  # 保存完整的服务器名称
                }
                
                # 提前检查服务器是否在资源索引中
                # 增强服务器名称处理逻辑
                # 检查原始服务器名称
                print(f"DEBUG: 原始服务器名称: '{server}', 类型: {type(server)}")
                
                # 检查是否需要解析服务器范围表示法
                expanded_servers = []
                
                if isinstance(server, str) and ('[' in server and ']' in server):
                    # 可能是范围表示法，如 "cpu1-[7-9]" 或 "cpu1-[15,96-97]"
                    print(f"DEBUG: 检测到可能的服务器范围表示法: '{server}'")
                    # 在这里可以解析范围，当前暂时跳过
                
                # 标准化服务器名称，确保格式一致性
                server_normalized = server.lower().strip() if isinstance(server, str) else str(server).lower().strip()
                print(f"DEBUG: 标准化后的服务器名称: '{server_normalized}'")
                
                # 检查resource_index中的键数量和类型
                print(f"DEBUG: resource_index包含 {len(resource_index)} 个服务器的数据")
                if len(resource_index) > 0:
                    first_key = next(iter(resource_index.keys()))
                    print(f"DEBUG: resource_index第一个键示例: '{first_key}', 类型: {type(first_key)}")
                    # 显示更多资源索引信息以便调试
                    sample_keys = list(resource_index.keys())[:5]
                    print(f"DEBUG: resource_index的前5个键: {sample_keys}")
                
                # 尝试精确匹配
                server_in_index = server_normalized in resource_index
                print(f"DEBUG: 精确匹配结果: {server_normalized} 在索引中? {server_in_index}")
                
                # 批量处理每个时间窗口
                for window_start, window_end in time_windows:
                    # 复制共享数据
                    row_data = common_data.copy()
                    row_data.update({
                        'time_window_start': window_start,
                        'time_window_end': window_end
                    })
                    
                    # 添加资源使用信息 - 增强版匹配逻辑
                    # 首先批量初始化所有资源列为n/a
                    row_data.update({col: 'n/a' for col in resource_columns})
                    
                    # 尝试匹配资源数据 - 增强版时间窗口匹配
                    print(f"DEBUG: 检查服务器 '{server}' 是否在索引中: {server_in_index}")
                    print(f"DEBUG: resource_index中共有 {len(resource_index)} 个服务器")
                    
                    # 详细分析时间窗口匹配问题
                    time_window_match = False
                    matching_window_key = None
                    
                    if server_in_index:
                        # 检查时间窗口的类型和格式
                        print(f"DEBUG: 当前窗口: {window_start}, 类型: {type(window_start)}")
                        
                        # 获取服务器的所有时间窗口
                        time_keys = list(resource_index[server].keys())
                        time_key_count = len(time_keys)
                        print(f"DEBUG: 服务器 '{server}' 有 {time_key_count} 个时间窗口")
                        
                        # 显示应查找的时间窗口
                        if time_key_count > 0:
                            sample_time_keys = time_keys[:min(3, time_key_count)]
                            print(f"DEBUG: 前{len(sample_time_keys)}个时间窗口示例: {sample_time_keys}")
                            
                            # 显示时间窗口类型以便调试
                            first_time_key = time_keys[0]
                            print(f"DEBUG: 第一个时间窗口类型: {type(first_time_key)}")
                            
                            # 尝试直接匹配
                            time_window_match = window_start in resource_index[server]
                            print(f"DEBUG: 直接时间窗口匹配结果: {time_window_match}")
                            
                            # 如果直接匹配失败，尝试其他匹配方式
                            if not time_window_match:
                                print(f"DEBUG: 尝试更灵活的时间窗口匹配方式")
                                
                                # 尝试将时间窗口转换为字符串进行匹配
                                window_start_str = str(window_start)
                                for tk in time_keys:
                                    tk_str = str(tk)
                                    if window_start_str == tk_str:
                                        print(f"DEBUG: 时间窗口字符串匹配成功: '{window_start_str}' 匹配 '{tk_str}'")
                                        time_window_match = True
                                        matching_window_key = tk
                                        break
                                
                                # 如果还是失败，尝试时间格式转换
                                if not time_window_match and isinstance(window_start, pd.Timestamp):
                                    window_dt = window_start.to_pydatetime()
                                    for tk in time_keys:
                                        # 根据索引中可能的时间格式进行转换和比较
                                        if isinstance(tk, str) and window_start_str in tk:
                                            print(f"DEBUG: 时间窗口部分匹配成功: '{window_start_str}' 包含在 '{tk}'")
                                            time_window_match = True
                                            matching_window_key = tk
                                            break
                                        elif isinstance(tk, (datetime, pd.Timestamp)):
                                            tk_dt = tk.to_pydatetime() if isinstance(tk, pd.Timestamp) else tk
                                            # 检查是否时间相同(忽略毫秒)
                                            time_diff = abs((window_dt - tk_dt).total_seconds())
                                            if time_diff < 2:  # 允许2秒的偏差
                                                print(f"DEBUG: 时间窗口相近匹配: '{window_dt}' 和 '{tk_dt}' (偏差: {time_diff}秒)")
                                                time_window_match = True
                                                matching_window_key = tk
                                                break
                        else:
                            print(f"DEBUG: 警告! 服务器 '{server}' 在索引中没有时间窗口数据")
                    else:
                        # 输出部分resource_index键以帮助调试
                        if len(resource_index) > 0:
                            key_examples = list(resource_index.keys())[:5]
                            print(f"DEBUG: resource_index前5个键示例: {key_examples}")
                        print(f"DEBUG: 服务器 '{server}' 不在资源索引中")
                    
                    # 定义一个辅助函数来获取资源值，包含详细调试
                    def custom_get(resource_type, resource_values):
                        if resource_type in resource_values:
                            value = resource_values[resource_type]
                            print(f"DEBUG: 成功获取资源 {resource_type} = {value}")
                            return value
                        else:
                            print(f"DEBUG: 资源类型 {resource_type} 不在可用资源中, 可用资源: {list(resource_values.keys())}")
                            return 'n/a'
                    
                    # 增强版资源值获取逻辑，与时间窗口匹配协同工作
                    if server_in_index and len(server_normalized) > 1:  # 确保服务器名称有效
                        window_key = matching_window_key if time_window_match else window_start
                        
                        # 显示详细的调试信息
                        print(f"DEBUG: 尝试使用窗口键 '{window_key}' 找到资源值")
                        
                        # 先显示服务器的时间窗口情况
                        time_keys = list(resource_index[server].keys()) if server in resource_index else []
                        if len(time_keys) > 0:
                            print(f"DEBUG: 服务器 '{server}' 有 {len(time_keys)} 个时间窗口")
                            print(f"DEBUG: 时间窗口示例(前3个): {time_keys[:min(3, len(time_keys))]}")
                        
                        # 确保 window_key 不为 None
                        if window_key is None:
                            print(f"DEBUG: 警告! window_key 为 None，尝试使用第一个可用的时间窗口")
                            if time_keys:
                                window_key = time_keys[0]
                                print(f"DEBUG: 使用第一个时间窗口作为替代: {window_key}")
                            else:
                                print(f"DEBUG: 服务器 '{server}' 没有可用的时间窗口，跳过资源值获取")
                                continue
                        
                        # 检查是否可以获取资源值
                        if server in resource_index and window_key in resource_index[server]:
                            resource_values = resource_index[server][window_key]
                            print(f"DEBUG: 成功! 使用窗口键 '{window_key}' 找到资源值: {len(resource_values)} 个指标")
                            print(f"DEBUG: 资源值示例: {list(resource_values.items())[:3]}")
                            
                            # 检查资源类型
                            for resource_type in list(resource_values.keys())[:5]:
                                print(f"DEBUG: 检查资源类型 {resource_type}: {resource_type in resource_values}")
                                
                            # 记录调试信息
                            if self.verbose and len(result_rows) < 3:
                                print(f"\n成功匹配资源数据:")
                                print(f"  服务器: {server} (原始: {common_data['server']})")
                                print(f"  时间窗口: {window_key} (原始请求: {window_start})")
                                print(f"  资源值数量: {len(resource_values)}")
                                print(f"  资源类型: {list(resource_values.keys())}")
                                print(f"  匹配方式: {'直接匹配' if window_key == window_start else '智能匹配'}")
                        else:
                            print(f"DEBUG: 服务器 '{server}' 存在，但时间窗口 '{window_key}' 不存在")
                            continue
                        
                        # 定义安全获取函数
                        def custom_get(key, values_dict, default='n/a'):
                            # 尝试多种键名形式
                            key_with_value = f"{key}_value" if not key.endswith('_value') else key
                            key_without_value = key.replace('_value', '') if key.endswith('_value') else key
                            
                            print(f"DEBUG: custom_get 函数尝试获取键 {key}")
                            print(f"DEBUG: 尝试的键形式: 1) {key_with_value}, 2) {key_without_value}")
                            print(f"DEBUG: values_dict 类型: {type(values_dict)}, 可用键: {list(values_dict.keys()) if isinstance(values_dict, dict) else '不是字典'}")
                            
                            if isinstance(values_dict, dict):
                                # 先尝试带_value后缀的键
                                if key_with_value in values_dict:
                                    value = values_dict[key_with_value]
                                    print(f"DEBUG: 成功获取资源值 {key_with_value}: {value}")
                                    return value
                                # 再尝试不带_value后缀的键
                                elif key_without_value in values_dict:
                                    value = values_dict[key_without_value]
                                    print(f"DEBUG: 成功获取资源值 {key_without_value}: {value}")
                                    return value
                                # 最后直接尝试原始键
                                elif key in values_dict:
                                    value = values_dict[key]
                                    print(f"DEBUG: 成功获取资源值 {key}: {value}")
                                    return value
                            
                            print(f"DEBUG: 未找到资源值, 返回默认值 {default}")
                            return default
                            
                        # 更新资源值 - 只在资源值存在时才执行
                        if 'resource_values' in locals():
                            # 为每个资源列设置值
                            for res_column in resource_columns:
                                # 提取真正的资源类型名称（移除_value后缀）
                                res_type = res_column.replace('_value', '') if res_column.endswith('_value') else res_column
                                
                                # 使用改进的custom_get函数，尝试各种可能的键名形式
                                resource_value = custom_get(res_type, resource_values)
                                
                                # 设置资源列名称（确保带_value后缀）
                                column_name = res_column if res_column.endswith('_value') else f"{res_column}_value"
                                row_data[column_name] = resource_value
                                print(f"DEBUG: 设置资源列 {column_name} = {resource_value}")
                            
                            # 所有资源值处理已在当前循环中完成
                    # 2. 如果第一种方法失败，尝试服务器名称的其他变种
                    else:
                        # 日志信息 - 对小数据量详细记录
                        if self.verbose and len(result_rows) < 100:
                            print(f"\n调试: 未找到直接匹配资源 - 服务器: {server}, 时间: {window_start}")
                    
                    result_rows.append(row_data)
        
        return result_rows

    def _task_based_integration(self):
        """以任务为中心进行数据集成，使用多核并行处理和高级内存管理处理超大规模任务集 - 终极优化版"""
        # 导入必要的库
        import gc  # 垃圾回收
        import psutil  # 系统资源监控
        import traceback  # 详细错误报告
        import concurrent.futures
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from tqdm import tqdm  # 进度条
        from collections import defaultdict
        
        # 记录分析开始时间
        start_time = time.time()
        print(f"\n===== 任务与资源匹配处理开始 ===== [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        
        # 详细系统资源信息
        mem_info = psutil.virtual_memory()
        cpu_count = multiprocessing.cpu_count()
        print(f"系统资源信息:")
        print(f"  - CPU: {cpu_count} 核心")
        print(f"  - 内存: {mem_info.total/1024/1024/1024:.1f}GB 总计, {mem_info.percent}% 使用中")
        print(f"  - 可用内存: {mem_info.available/1024/1024/1024:.1f}GB")
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                print(f"  - GPU: {len(gpus)} 个, 型号: {gpus[0].name if gpus else 'N/A'}")
        except ImportError:
            pass  # GPUtil不可用，忽略GPU信息
            
        # 检查任务数据
        if self.jobs is None or len(self.jobs) == 0:
            print("错误: 没有任务数据可以与资源关联")
            return None
        
        # 步骤1: 创建资源数据索引
        print("\n步骤1: 创建资源数据索引")
        gc.collect()  # 先清理内存
        resource_index = self._create_resource_index()
        
        # 二次检查资源索引
        if not resource_index:
            print("警告: 资源索引为空，任务将无法获取对应资源数据")
            # 如果仍需继续处理，则只返回任务数据而没有资源信息
            resource_index = {}
            
        # 重新获取内存状态
        mem_after_index = psutil.virtual_memory()
        print(f"资源索引创建后内存使用: {mem_after_index.percent}% ({mem_after_index.used/1024/1024/1024:.1f}GB)")
        mem_increase = mem_after_index.used - mem_info.used
        print(f"资源索引内存占用: {mem_increase/1024/1024/1024:.2f}GB")
        
        # 获取或合成所有资源列名
        resource_columns = []
        if self.resources:
            for resource_type in self.resources.keys():
                resource_columns.append(f"{resource_type}_value")
        else:
            # 如果没有资源数据，使用默认列名确保输出格式一致
            resource_columns = [
                'cpu_total_value', 'cpu_system_value', 'gpu_avg_value', 'gpu_util_value', 
                'mem_percent_value', 'mem_used_value', 'load_1min_value', 'load_15min_value', 
                'power_value', 'temperature_value'
            ]
        print(f"将处理的资源指标: {len(resource_columns)} 项")
        
        # 步骤2: 准备任务分块
        print("\n步骤2: 准备并行处理任务数据")
        total_jobs = len(self.jobs)
        print(f"任务总数: {total_jobs:,}")
        
        # 自适应智能配置 - 根据任务规模、内存使用和CPU可用性进行动态优化
        num_cores = cpu_count  # 使用之前已经获取的CPU核心数
        current_mem_percent = mem_after_index.percent  # 使用最新的内存使用率
        available_mem_gb = mem_after_index.available / (1024**3)  # 可用内存GB
        
        # 1. 内存状态评估
        if current_mem_percent > 90:  # 内存极度紧张
            print("⚠️ 警告: 内存使用极高，采用极度保守设置")
            mem_factor = 0.1  # 极低内存系数
        elif current_mem_percent > 80:  # 内存紧张
            print("⚠️ 内存使用率高，采用保守并行设置")
            mem_factor = 0.3  # 低内存系数
        elif current_mem_percent > 70:  # 内存稍紧张
            print("ℹ️ 内存使用中等，采用平衡设置")
            mem_factor = 0.5  # 中等内存系数
        else:  # 内存充足
            print("✓ 内存状态良好，采用高性能设置")
            mem_factor = 0.8  # 高内存系数
        
        # 2. 任务规模评估 - 对不同任务量采用不同策略
        if total_jobs > 10_000_000:  # 超大规模任务
            print(f"⚠️ 检测到超大规模任务集 ({total_jobs:,} 个任务)")
            scale_factor = 0.3  # 超大数据集使用非常保守的参数
            base_chunk_size = 10000
            max_chunk_size = 50000
        elif total_jobs > 1_000_000:  # 大规模任务
            print(f"ℹ️ 检测到大规模任务集 ({total_jobs:,} 个任务)")
            scale_factor = 0.5  # 大数据集使用保守的参数
            base_chunk_size = 5000
            max_chunk_size = 30000
        elif total_jobs > 100_000:  # 中等规模任务
            print(f"ℹ️ 检测到中等规模任务集 ({total_jobs:,} 个任务)")
            scale_factor = 0.7  # 中等数据集使用平衡参数
            base_chunk_size = 2000
            max_chunk_size = 10000
        else:  # 小规模任务
            print(f"✓ 检测到小规模任务集 ({total_jobs:,} 个任务)")
            scale_factor = 1.0  # 小数据集可以使用最优性能参数
            base_chunk_size = 1000
            max_chunk_size = 5000
        
        # 3. 系统状态智能配置计算
        # 结合内存和规模因子计算最终的资源使用比例
        final_factor = min(mem_factor, scale_factor)
        
        # 计算工作进程数 - 针对极高内存使用环境的特殊优化
        ideal_workers = int(num_cores * final_factor)
        
        # 检查当前内存压力，更激进地限制并行度
        try:
            # 获取最新内存状态
            current_mem = psutil.virtual_memory()
            mem_percent = current_mem.percent
            available_mem_gb = current_mem.available / (1024**3)  # GB
            
            print(f"内存使用检查 - 使用率: {mem_percent:.1f}%, 可用: {available_mem_gb:.1f}GB")
            
            # 极端内存压力下的超保守设置
            if mem_percent > 93 or available_mem_gb < 32:  # 接近现在观察到的值
                print("⚠️ 危险: 内存使用极高(>93%)，采用超保守设置")
                num_workers = max(1, min(2, ideal_workers // 4))  # 极度减少工作进程
                # 主动触发内存回收
                gc.collect()
            elif mem_percent > 85 or available_mem_gb < 50:
                print("⚠️ 警告: 内存使用高(>85%)，采用保守设置")
                num_workers = max(1, min(4, ideal_workers // 3))  # 大幅减少工作进程
            elif available_mem_gb < 80:  # 可用内存适中但不充足
                num_workers = max(2, min(8, ideal_workers // 2))  # 减半工作进程
            else:  # 内存相对充足
                num_workers = max(4, min(num_cores-4, ideal_workers))  # 保留至少4个核心给系统使用
        except Exception as e:
            print(f"内存检查异常: {e}，默认使用保守设置")
            # 出错时使用保守设置
            num_workers = max(2, min(6, ideal_workers // 2))
        
        # 4. 针对极高内存使用率环境的块大小计算优化
        # 根据内存使用率动态调整块大小和分割策略
        try:
            # 定义一个估算每个任务的内存使用量(假设平均每个任务使用15MB)
            estimated_mem_per_task_mb = 15
            
            # 计算安全内存使用限制
            if mem_percent > 90:
                # 超高内存使用率，只使用可用内存的20%
                safe_mem_gb = max(1, available_mem_gb * 0.2)
                print(f"⚠️ 超高内存使用率，安全内存限制: {safe_mem_gb:.1f}GB")
            elif mem_percent > 85:
                # 高内存使用率，只使用可用内存的30%
                safe_mem_gb = max(2, available_mem_gb * 0.3)
                print(f"⚠️ 高内存使用率，安全内存限制: {safe_mem_gb:.1f}GB")
            else:
                # 一般内存压力，可使用可用内存的50%
                safe_mem_gb = max(4, available_mem_gb * 0.5)
                print(f"内存算法限制: {safe_mem_gb:.1f}GB")
            
            # 计算安全内存下能容纳的最大任务数
            max_tasks_in_memory = int((safe_mem_gb * 1024) / estimated_mem_per_task_mb)
            print(f"内存安全任务限制: {max_tasks_in_memory} 个任务")
            
            # 计算增强的分块策略
            min_chunks_per_worker = 8 if mem_percent > 90 else 4  # 内存高压力时增加分割数
            total_chunks = max(num_workers * min_chunks_per_worker * 2, 1)  # 增大分割数
            
            # 根据内存安全限制计算块大小
            if mem_percent > 90:
                chunk_size = min(500, max_tasks_in_memory // (num_workers * 2))
            elif mem_percent > 85:
                chunk_size = min(1000, max_tasks_in_memory // (num_workers * 1.5))
            else:
                chunk_size = min(max_chunk_size, 
                               max(base_chunk_size, 
                                  min(max_tasks_in_memory // num_workers, total_jobs // total_chunks)))
        except Exception as e:
            print(f"计算块大小时出错: {e}，使用安全的默认值")
            # 出错时使用保守的默认值
            min_chunks_per_worker = 4
            total_chunks = max(num_workers * min_chunks_per_worker, 1)
            chunk_size = max(500, min(1000, total_jobs // total_chunks))
        
        # 5. 极端情况特殊处理
        if num_cores < 4:  # 低核心数系统
            print("ℹ️ 系统CPU核心数少于4个，采用单线程模式")
            num_workers = 1  # 单线程模式
            chunk_size = max(500, min(2000, total_jobs // 4))  # 合理的块大小
        
        # 最终参数显示
        print(f"☑️ 并行处理参数: {num_workers} 个工作进程, {chunk_size} 个任务/块")
        print(f"☑️ 预计将分割为 {total_jobs // chunk_size + (1 if total_jobs % chunk_size else 0)} 个处理块")
        
        # 更高效的数据分割 - 使用生成器避免一次性创建所有分块以节省内存
        def chunk_generator(df, chunk_size):
            for i in range(0, len(df), chunk_size):
                yield df.iloc[i:i+chunk_size]
        
        # 计算分块数量而不实际创建块
        num_chunks = (total_jobs + chunk_size - 1) // chunk_size  # 向上取整
        print(f"✓ 数据将分割为 {num_chunks} 个处理块, 每块约 {chunk_size} 个任务")
        print(f"✓ 系统资源: {num_cores} 个CPU核心, 将使用 {num_workers} 个工作进程")
        
        # 强制内存回收
        gc.collect()
        
        # 设置处理标志和性能监控变量
        use_parallel = True  # 临时禁用并行处理，强制使用单进程模式进行调试
        all_results = []
        processed_chunks = 0  # 初始化已处理块计数
        successful_chunks = 0  # 初始化成功块计数 
        failed_chunks = 0      # 初始化失败块计数
        parallel_start_time = time.time()
        
        # 步骤3: 开始任务处理
        print(f"\n步骤3: 开始处理任务数据 [{datetime.now().strftime('%H:%M:%S')}]")
        
        # 尝试并行处理
        if use_parallel and num_workers > 1:
            try:
                print(f"🚀 启动并行处理模式 (进程池大小: {num_workers}, 块数: {num_chunks})")
                
                # 预分配进度和统计变量
                processed_items = 0  # 已处理的任务数
                processing_rate = 0   # 处理速率
                
                # 重置性能监控变量
                total_processed_items = 0
                
                # 创建任务分块生成器
                job_chunks_gen = chunk_generator(self.jobs, chunk_size)
                
                # 使用tqdm显示总体进度并记录处理速度
                with tqdm(total=num_chunks, desc="并行处理任务块", unit="块") as pbar:
                    # 创建工作进程池
                    with ProcessPoolExecutor(max_workers=num_workers, 
                                            mp_context=multiprocessing.get_context('spawn')) as executor:
                        # 使用spawn比fork更加稳定，特别是对大型数据集
                        
                        # 动态批处理 - 避免一次性创建过多的Future对象
                        optimal_batch_size = max(2, min(num_workers * 2, num_chunks // 10 + 1))  # 初始批大小
                        processed_chunks = 0
                        timeout_occurred = False
                        processing_times = []  # 记录每批处理时间以优化后续批次
                        
                        # 主处理循环 - 分批处理所有块
                        timeout_occurred = False  # 初始化超时标志
                        while processed_chunks < num_chunks and not timeout_occurred:
                            # 自适应批大小调整 - 基于性能和内存状态
                            if successful_chunks > 0 and processed_chunks > optimal_batch_size:
                                # 检查当前内存使用情况
                                current_memory = psutil.virtual_memory()
                                current_memory_usage = current_memory.percent
                                
                                # 内存压力下的批大小调整策略
                                if current_memory_usage > 90:  # 内存严重不足
                                    optimal_batch_size = max(1, optimal_batch_size // 4)  # 大幅减小批大小
                                    print(f"\n⚠️ 严重警告: 内存接近耗尽 ({current_memory_usage}%)，大幅减小批大小至 {optimal_batch_size}")
                                    # 强制垃圾回收
                                    gc.collect()
                                elif current_memory_usage > 80:  # 内存紧张
                                    optimal_batch_size = max(1, optimal_batch_size // 2)  # 减半批大小
                                    print(f"\n⚠️ 警告: 内存使用率高 ({current_memory_usage}%)，减小批大小至 {optimal_batch_size}")
                                    # 尝试垃圾回收
                                    gc.collect()
                                elif current_memory_usage < 60 and len(processing_times) > 2:  # 内存充足且有足够的性能数据
                                    # 计算处理速度趋势
                                    if len(processing_times) >= 3 and sum(processing_times[-3:]) < sum(processing_times[-6:-3]):
                                        # 性能在提升，可以增加批大小
                                        optimal_batch_size = min(num_workers * 4, optimal_batch_size * 2)
                            
                            # 获取当前批次的数据块
                            batch_start_time = time.time()
                            current_batch = []
                            batch_count = 0
                            
                            # 从生成器中获取指定数量的块
                            try:
                                for _ in range(min(optimal_batch_size, num_chunks - processed_chunks)):
                                    current_batch.append(next(job_chunks_gen))
                                    batch_count += 1
                            except StopIteration:
                                # 生成器结束，处理最后的块
                                pass
                            
                            # 跟踪当前内存使用状态
                            current_memory = psutil.virtual_memory()
                            if batch_count > 0 and current_memory.percent > 75:
                                print(f"\nℹ️ 内存监控: 批次拥有 {batch_count} 个块，当前内存使用率: {current_memory.percent}%")
                            
                            # 创建并跟踪任务的Future对象
                            batch_futures = {}
                            future_chunks = {}  # 跟踪每个块的缓存和索引
                            
                            # 提交当前批次的任务并优化内存使用
                            for i, chunk in enumerate(current_batch):
                                chunk_index = processed_chunks + i
                                chunk_size_count = len(chunk)
                                # 使用精确的逻辑提交任务
                                future = executor.submit(self._process_job_chunk, chunk, resource_index, resource_columns)
                                batch_futures[future] = chunk_index
                                future_chunks[future] = (chunk_index, chunk_size_count)  # 保存块大小以计算速率
                                
                                # 释放块引用以节约内存
                                if i > 0:  # 保留最后一个块的引用以防生成器提前退出
                                    current_batch[i-1] = None
                            
                            # 计算动态超时时间 - 基于批次大小和历史性能
                            if len(processing_times) > 0:
                                avg_time_per_chunk = sum(processing_times) / len(processing_times)
                                expected_time = avg_time_per_chunk * len(batch_futures) * 1.5  # 增加50%的缓冲
                                batch_timeout = max(600, min(3600, int(expected_time)))
                            else:
                                # 首批或没有历史数据，设置合理默认值
                                batch_timeout = max(600, min(3600, len(batch_futures) * 150))  # 每块平均150秒
                            
                            # 显示批次信息
                            if processed_chunks == 0:
                                print(f"ℹ️ 首批处理: {len(batch_futures)} 个块, 超时时间: {batch_timeout//60}分{batch_timeout%60}秒")
                            
                            # 处理当前批次的结果
                            batch_item_count = 0
                            try:
                                # 等待所有任务完成或超时
                                for future in as_completed(batch_futures.keys(), timeout=batch_timeout):
                                    try:
                                        # 计算单个任务超时时间 - 根据块大小动态分配
                                        chunk_idx, chunk_items = future_chunks.get(future, ("unknown", 0))
                                        per_item_time = 0.5  # 假设每个任务项需要平均500ms
                                        single_timeout = max(240, min(1800, int(chunk_items * per_item_time)))
                                        
                                        # 获取结果并更新统计
                                        result = future.result(timeout=single_timeout)
                                        result_count = len(result)
                                        batch_item_count += result_count
                                        all_results.extend(result)
                                        successful_chunks += 1
                                        total_processed_items += result_count
                                        pbar.update(1)
                                        
                                        # 定期验证内存状态
                                        if successful_chunks % 10 == 0:
                                            gc.collect()
                                    except Exception as e:
                                        chunk_idx = batch_futures.get(future, "未知")
                                        print(f"\n⚠️ 处理任务块 {chunk_idx} 时出错: {type(e).__name__}: {str(e)}")
                                        failed_chunks += 1
                                        pbar.update(1)
                            except concurrent.futures.TimeoutError:
                                # 批处理超时处理
                                timeout_occurred = True
                                batch_end_time = time.time()
                                batch_elapsed = batch_end_time - batch_start_time
                                
                                print(f"\n⚠️ 批处理超时 ({batch_timeout//60}分{batch_timeout%60}秒), 实际运行: {batch_elapsed:.1f}秒")
                                print(f"⚠️ 恢复处理: 切换到自适应顺序处理模式")
                                
                                # 统计未完成的任务
                                incomplete_futures = [f for f in batch_futures if not f.done()]
                                done_futures = [f for f in batch_futures if f.done()]
                                
                                # 尝试收集已完成任务的结果
                                for future in done_futures:
                                    try:
                                        result = future.result(timeout=1)  # 短超时只等待已完成的任务
                                        all_results.extend(result)
                                        successful_chunks += 1
                                        pbar.update(1)
                                    except Exception:
                                        # 忽略已完成但有错误的任务
                                        failed_chunks += 1
                                        pbar.update(1)
                                
                                # 重新创建任务生成器，包含未完成的块
                                incomplete_indices = [batch_futures[f] for f in incomplete_futures]
                                # 使用更安全的方式创建未完成的块
                                incomplete_chunks = []
                                for i in incomplete_indices:
                                    start_idx = i * chunk_size
                                    end_idx = min((i + 1) * chunk_size, len(self.jobs))
                                    if start_idx < len(self.jobs):
                                        incomplete_chunks.append(self.jobs.iloc[start_idx:end_idx])
                                
                                # 计算剩余未处理的任务总数
                                remaining_tasks_start = (processed_chunks + len(current_batch)) * chunk_size
                                remaining_tasks = max(0, total_jobs - remaining_tasks_start)
                                remaining_chunks_count = (remaining_tasks + chunk_size - 1) // chunk_size
                                
                                print(f"  • 当前批次: {len(done_futures)} 完成, {len(incomplete_futures)} 超时")
                                print(f"  • 未完成块: {len(incomplete_chunks)} 个块需要重试")
                                print(f"  • 未处理任务: 还有约 {remaining_tasks:,} 个任务 (约 {remaining_chunks_count} 个块)")
                                
                                # 新的顺序处理切块大小 - 减小块大小以提高成功率
                                sequential_chunk_size = max(chunk_size // 5, 100)  # 显著减小块大小但不小于100
                                print(f"\n实施自适应顺序处理: 新块大小 = {sequential_chunk_size}, 总剩余任务数 = {remaining_tasks + len(incomplete_chunks)*chunk_size:,}")
                                
                                # 1. 先处理未完成的块
                                if incomplete_chunks:
                                    print(f"\n第1步: 处理 {len(incomplete_chunks)} 个超时块")
                                    with tqdm(total=len(incomplete_chunks), desc="处理超时块", unit="块") as seq_pbar:
                                        for i, chunk in enumerate(incomplete_chunks):
                                            try:
                                                # 将大块拆分成小块处理以提高成功率
                                                for j in range(0, len(chunk), sequential_chunk_size):
                                                    sub_chunk = chunk.iloc[j:j+sequential_chunk_size]
                                                    result = self._process_job_chunk(sub_chunk, resource_index, resource_columns)
                                                    all_results.extend(result)
                                                
                                                successful_chunks += 1
                                                seq_pbar.update(1)
                                                
                                                # 每处理一个块就强制内存回收
                                                gc.collect()
                                            except Exception as e:
                                                print(f"\n⚠️ 处理超时块 {i} 时出错: {type(e).__name__}: {str(e)}")
                                                failed_chunks += 1
                                                seq_pbar.update(1)
                                
                                # 2. 再处理剩余的任务
                                if remaining_tasks > 0:
                                    print(f"\n第2步: 处理剩余 {remaining_tasks:,} 个任务 (约 {(remaining_tasks+sequential_chunk_size-1)//sequential_chunk_size} 个小块)")
                                    with tqdm(total=remaining_tasks, desc="处理剩余任务", unit="任务") as seq_pbar:
                                        # 直接遍历剩余的任务数据而不是原来的大块
                                        remaining_df = self.jobs.iloc[remaining_tasks_start:]
                                        
                                        for j in range(0, len(remaining_df), sequential_chunk_size):
                                            try:
                                                # 使用更小的块处理
                                                sub_chunk = remaining_df.iloc[j:j+sequential_chunk_size]
                                                result = self._process_job_chunk(sub_chunk, resource_index, resource_columns)
                                                all_results.extend(result)
                                                successful_chunks += (len(sub_chunk) / chunk_size)  # 计算相当于多少个原始块
                                                seq_pbar.update(len(sub_chunk))
                                                
                                                # 每处理一定数量的小块进行内存回收
                                                if j % (sequential_chunk_size * 5) == 0:
                                                    gc.collect()
                                            except Exception as e:
                                                print(f"\n⚠️ 处理剩余任务块 {j//sequential_chunk_size} 时出错: {type(e).__name__}: {str(e)}")
                                                failed_chunks += (len(sub_chunk) / chunk_size)  # 计算相当于多少个原始块
                                                seq_pbar.update(len(sub_chunk))
                                
                                # 跨过剩余的并行处理步骤
                                break
                            
                            # 更新已处理的块数量
                            processed_chunks += len(current_batch)
                            
                            # 手动回收内存
                            gc.collect()
                
                # 显示并行处理总结
                print(f"\n并行处理统计: 处理 {processed_chunks}/{num_chunks} 块, 成功 {successful_chunks} 块, 失败 {failed_chunks} 块")
                
            except concurrent.futures.process.BrokenProcessPool as e:
                # 进程池损坏 - 可能由于内存问题或系统资源限制
                parallel_end_time = time.time()
                parallel_duration = parallel_end_time - parallel_start_time
                print(f"\n⚠️ 并行处理失败: 进程池被中断: {str(e)}")
                print(f"⚠️ 并行处理已运行: {parallel_duration:.1f}秒, 完成 {successful_chunks} 个块, 失败 {failed_chunks} 个块")
                print(f"↪️ 正在切换到优化的顺序处理模式...")
                
                # 记录进程池损坏的细节，有助于日后优化
                try:
                    import platform
                    system_info = {
                        "system": platform.system(),
                        "release": platform.release(),
                        "memory_percent": psutil.virtual_memory().percent,
                        "memory_available": psutil.virtual_memory().available / (1024**3),
                        "cpu_count": num_cores,
                        "workers": num_workers,
                        "chunks_processed": processed_chunks
                    }
                    print(f"ℹ️ 系统状态: {system_info}")
                except Exception:
                    pass  # 忽略信息收集错误
                
                # 强制清理内存，准备回退处理
                gc.collect()
                use_parallel = False
                
            except MemoryError as e:
                # 内存不足错误处理 - 更积极地减少内存使用
                parallel_end_time = time.time()
                print(f"\n⚠️ 内存不足错误: {str(e)}")
                print(f"⚠️ 内存状态: {psutil.virtual_memory().percent}% 使用, {psutil.virtual_memory().available/(1024**3):.1f}GB 可用")
                print(f"↪️ 切换到内存优化模式处理剩余任务...")
                
                # 积极清理内存
                all_results_len = len(all_results)
                print(f"ℹ️ 当前已有 {all_results_len} 条结果记录")
                gc.collect()
                
                # 使用更保守的块大小
                new_chunk_size = max(50, chunk_size // 20)  # 极大减小块大小
                num_remaining = total_jobs - (processed_chunks * chunk_size)
                num_new_chunks = (num_remaining + new_chunk_size - 1) // new_chunk_size
                
                print(f"ℹ️ 内存优化策略: 块大小从 {chunk_size} 减至 {new_chunk_size}, 剩余 {num_remaining:,} 任务将分为 {num_new_chunks} 个小块")
                
                # 不再使用job_chunks列表，而是使用按需生成的迭代器
                use_parallel = False
                chunk_size = new_chunk_size
                
            except Exception as e:
                # 通用错误处理 - 提供详细诊断
                parallel_end_time = time.time()
                parallel_duration = parallel_end_time - parallel_start_time
                print(f"\n⚠️ 并行处理发生未预期错误: {type(e).__name__}: {str(e)}")
                print(f"⚠️ 并行处理已运行: {parallel_duration:.1f}秒, 完成 {successful_chunks} 个块")
                print(f"↪️ 详细错误信息:")
                traceback.print_exc()
                
                # 添加更多诊断信息
                print(f"\nℹ️ 诊断信息:")
                print(f"  - Python版本: {sys.version}")
                print(f"  - 内存使用: {psutil.virtual_memory().percent}%")
                print(f"  - 已处理块: {processed_chunks}/{num_chunks}")
                print(f"  - 成功/失败: {successful_chunks}/{failed_chunks}")
                
                # 强制清理资源
                gc.collect()
                print(f"↪️ 将切换到顺序处理模式继续处理...")
                use_parallel = False
        
        # 顺序处理模式 - 如果并行处理失败或被跳过
        if not use_parallel:
            sequential_start_time = time.time()
            successful_sequential_chunks = 0
            failed_sequential_chunks = 0
            
            # 保留已处理的结果，除非检测到需要从头开始
            if len(all_results) > 0 and successful_chunks > 0:
                print(f"\n✓ 保留并行模式已处理的 {len(all_results)} 条结果记录 (来自 {successful_chunks} 个块)")
                initial_results_count = len(all_results)
            else:
                print("\n↪️ 清空之前的不完整结果，从头开始处理")
                all_results = []
                initial_results_count = 0
            
            # 计算需要处理的任务范围 - 仅处理尚未处理的部分
            # 如果已经处理了一些块，计算出起始位置
            start_index = processed_chunks * chunk_size if processed_chunks > 0 else 0
            remaining_count = total_jobs - start_index
            
            # 更高效的小块处理大小
            sequential_chunk_size = max(50, min(500, chunk_size // 10))
            num_sequential_chunks = (remaining_count + sequential_chunk_size - 1) // sequential_chunk_size
            
            # 显示顺序处理计划
            print(f"\n↪️ 顺序处理阶段: 将处理 {remaining_count:,} 个剩余任务 ({start_index:,} → {total_jobs:,})")
            print(f"↪️ 优化策略: 使用 {sequential_chunk_size} 个任务/块, 共 {num_sequential_chunks} 个小块")
            
            # 强制回收内存
            gc.collect()
            
            # 使用更高效的生成器模式处理数据 - 避免创建大列表
            sequential_progress = tqdm(total=remaining_count, desc="顺序处理剩余任务", unit="任务")
            
            # 顺序处理主循环 - 使用小块分段处理
            current_index = start_index
            completed_items = 0
            
            try:
                while current_index < total_jobs:
                    # 自适应确定当前块的结束位置
                    next_index = min(current_index + sequential_chunk_size, total_jobs)
                    chunk_size_actual = next_index - current_index
                    
                    try:
                        # 监控内存使用情况
                        current_memory = psutil.virtual_memory()
                        critical_memory = current_memory.percent > 90
                        
                        # 内存危急时的特殊处理
                        if critical_memory:
                            # 强制垃圾回收
                            print(f"\n⚠️ 内存压力极大 ({current_memory.percent}%), 执行紧急优化")
                            gc.collect()
                            
                            # 极端情况: 将当前块分割成更小的块处理
                            if current_memory.percent > 95 and chunk_size_actual > 10:
                                # 超小块处理模式
                                micro_chunk_size = max(1, sequential_chunk_size // 10)
                                print(f"⚠️ 启动超小块处理模式: {micro_chunk_size} 个任务/块")
                                
                                # 逐个小块处理当前区间
                                for micro_idx in range(current_index, next_index, micro_chunk_size):
                                    micro_end = min(micro_idx + micro_chunk_size, next_index)
                                    micro_chunk = self.jobs.iloc[micro_idx:micro_end]
                                    
                                    try:
                                        # 处理微型块
                                        micro_result = self._process_job_chunk(micro_chunk, resource_index, resource_columns)
                                        all_results.extend(micro_result)
                                        completed_items += len(micro_chunk)
                                        sequential_progress.update(len(micro_chunk))
                                        
                                        # 每个微型块后强制回收
                                        gc.collect()
                                    except Exception as me:
                                        print(f"⚠️ 微型块处理出错 ({micro_idx}→{micro_end}): {type(me).__name__}: {str(me)}")
                                        failed_sequential_chunks += len(micro_chunk) / sequential_chunk_size
                                        sequential_progress.update(len(micro_chunk))
                                
                                # 跳过常规处理
                                current_index = next_index
                                continue
                        
                        # 标准块处理 - 提取当前块数据
                        current_chunk = self.jobs.iloc[current_index:next_index]
                        
                        # 处理当前块
                        result = self._process_job_chunk(current_chunk, resource_index, resource_columns)
                        all_results.extend(result)
                        successful_sequential_chunks += 1
                        completed_items += len(current_chunk)
                        sequential_progress.update(len(current_chunk))
                        
                        # 动态内存管理策略
                        if current_memory.percent > 80:  # 高内存压力
                            gc.collect()  # 每块后立即回收
                        elif current_memory.percent > 60:  # 中等内存压力
                            if (successful_sequential_chunks % 5) == 0:
                                gc.collect()  # 每5块回收一次
                        else:  # 低内存压力
                            if (successful_sequential_chunks % 20) == 0:
                                gc.collect()  # 每20块回收一次
                    except MemoryError as e:
                        current_memory_pct = psutil.virtual_memory().percent
                        print(f"\n⚠️ 内存不足错误 ({current_index}→{next_index}): {str(e)}")
                        print(f"⚠️ 当前内存使用: {current_memory_pct}%, 可用内存: {psutil.virtual_memory().available/(1024**3):.2f} GB")
                        
                        # 极端内存优化措施
                        gc.collect()
                        
                        # 如果内存仍然极度紧张，且当前处理的块较大，则减少块大小再试
                        if current_memory_pct > 95 and chunk_size_actual > 5:
                            # 缩小块大小并重试
                            reduced_size = max(1, chunk_size_actual // 5)
                            print(f"↪️ 尝试减小块大小 {chunk_size_actual} → {reduced_size} 并重试...")
                            
                            # 重试使用更小的块
                            next_attempt_idx = current_index
                            while next_attempt_idx < next_index:
                                attempt_end = min(next_attempt_idx + reduced_size, next_index)
                                mini_chunk = self.jobs.iloc[next_attempt_idx:attempt_end]
                                
                                try:
                                    # 处理极小块
                                    result = self._process_job_chunk(mini_chunk, resource_index, resource_columns)
                                    all_results.extend(result)
                                    completed_items += len(mini_chunk)
                                    sequential_progress.update(len(mini_chunk))
                                    gc.collect()  # 每个最小块后强制回收
                                except Exception as mini_e:
                                    # 跳过此极小块
                                    print(f"⚠️ 极小块处理失败 ({next_attempt_idx}→{attempt_end}): {type(mini_e).__name__}")
                                    failed_sequential_chunks += len(mini_chunk) / sequential_chunk_size
                                    sequential_progress.update(len(mini_chunk))
                                
                                next_attempt_idx = attempt_end
                            
                            # 跳过常规错误处理，继续下一个完整块
                            current_index = next_index
                            continue
                        else:
                            # 常规错误处理 - 跳过当前块
                            failed_sequential_chunks += 1
                            sequential_progress.update(chunk_size_actual)
                            current_index = next_index
                            continue
                            
                    except Exception as e:
                        # 常规错误处理
                        print(f"\n⚠️ 处理区块出错 ({current_index}→{next_index}): {type(e).__name__}: {str(e)}")
                        
                        # 只在严重错误时输出完整堆格
                        if not isinstance(e, (KeyError, ValueError, TypeError, IndexError)):
                            traceback.print_exc()
                            
                        # 尝试记录更多上下文信息，帮助诊断
                        try:
                            chunk_info = {
                                "range": f"{current_index}→{next_index}",
                                "size": chunk_size_actual,
                                "memory": f"{psutil.virtual_memory().percent}%", 
                                "job_count": len(self.jobs)
                            }
                            print(f"ℹ️ 错误上下文: {chunk_info}")
                        except Exception:
                            pass
                        
                        failed_sequential_chunks += 1
                        sequential_progress.update(chunk_size_actual)
                        current_index = next_index  # 继续处理下一个块
            
                    # 处理下一个块
                    current_index = next_index
            
                # 最终强制清理内存
                gc.collect()
                sequential_end_time = time.time()
                sequential_duration = sequential_end_time - sequential_start_time
                
                # 显示顺序处理的统计信息
                print(f"\n✓ 顺序处理完成: 耗时 {sequential_duration:.1f}秒")
                print(f"✓ 处理统计: 成功块 {successful_sequential_chunks}, 失败块 {failed_sequential_chunks:.1f}, 完成项 {completed_items:,}")
                
            except Exception as major_e:
                # 捕获顺序处理循环中可能出现的意外错误
                print(f"\n⚠️ 顺序处理模式发生关键错误: {type(major_e).__name__}: {str(major_e)}")
                traceback.print_exc()
                print("⚠️ 尽管出现错误，将尝试返回已处理的数据...")
            finally:
                # 关闭进度条如果存在
                if 'sequential_progress' in locals() and sequential_progress is not None:
                    sequential_progress.close()
        
        # 步骤5: 执行结果后处理和优化
        print("\n➡️ 步骤5: 处理结果数据并准备返回...")
        
        # 计算总运行时间
        end_time = time.time()
        total_time = end_time - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒"
        
        # 检查是否有结果
        if not all_results:
            print("⚠️ 警告: 未获取到任何结果数据")
            return pd.DataFrame()
        
        # 结果优化 - 在转换为DataFrame前执行一些优化
        result_count = len(all_results)
        print(f"ℹ️ 原始结果数: {result_count:,} 条记录")
        
        # 记录内存使用情况
        try:
            current_process = psutil.Process()
            pre_df_memory = current_process.memory_info().rss / (1024 * 1024)  # MB
            current_memory = psutil.virtual_memory()
            
            print(f"ℹ️ 当前内存状态: {current_memory.percent}% 使用, {current_memory.available/(1024**3):.1f}GB 可用")
            print(f"ℹ️ 当前进程内存: {pre_df_memory:.1f}MB")
            
            # 继续前执行全面的内存清理
            gc.collect()
            
            # 转换过程中的内存优化 - 分批处理大型结果集
            result_df = None
            ultra_large_dataset = result_count > 1000000
            very_large_dataset = result_count > 500000
            large_dataset = result_count > 100000
            
            if ultra_large_dataset:
                # 极大规模数据集特殊处理
                print(f"⚠️ 检测到超大规模结果集 ({result_count:,} 条记录)，使用多批次处理...")
                batch_size = 200000  # 每批处理的记录数
                batches = (result_count + batch_size - 1) // batch_size
                
                print(f"ℹ️ 将分为 {batches} 个批次处理，每批 {batch_size:,} 条记录")
                
                with tqdm(total=batches, desc="分批处理结果数据", unit="批") as batch_pbar:
                    result_batches = []
                    for i in range(0, result_count, batch_size):
                        batch = all_results[i:i+batch_size]
                        batch_df = pd.DataFrame(batch)
                        
                        # 预处理每个批次的数据类型
                        for col in batch_df.select_dtypes(include=['float64']).columns:
                            batch_df[col] = batch_df[col].astype('float32')  # 减少浮点数精度
                            
                        for col in ['server_id', 'job_id', 'user_id']:
                            if col in batch_df.columns and batch_df[col].nunique() < len(batch_df) * 0.5:
                                batch_df[col] = batch_df[col].astype('category')  # 使用分类类型
                        
                        # 添加准备好的批次
                        result_batches.append(batch_df)
                        
                        # 清理原始数据以减少内存压力
                        del batch
                        gc.collect()
                        batch_pbar.update(1)
                
                # 合并所有处理后的批次
                print("ℹ️ 合并所有处理后的批次...")
                result_df = pd.concat(result_batches, ignore_index=True)
                
                # 清理中间数据
                del result_batches
                gc.collect()
                
            elif very_large_dataset:
                # 大规模数据集优化处理
                print(f"ℹ️ 检测到大规模结果集 ({result_count:,} 条记录)，使用优化处理...")
                
                # 直接创建 DataFrame 但使用更优化的数据类型
                result_df = pd.DataFrame(all_results)
                
                # 立即释放原始数据
                del all_results
                gc.collect()
                
                # 高效优化数据类型
                for col in result_df.select_dtypes(include=['float64']).columns:
                    result_df[col] = result_df[col].astype('float32')  # 减少浮点数精度
                    
                for col in ['server_id', 'job_id', 'user_id']:
                    if col in result_df.columns and result_df[col].nunique() < len(result_df) * 0.5:
                        result_df[col] = result_df[col].astype('category')  # 使用分类类型
                
            else:
                # 标准处理 - 直接转换
                print(f"ℹ️ 处理标准规模结果集 ({result_count:,} 条记录)")
                result_df = pd.DataFrame(all_results)
                
                # 释放原始数据
                del all_results
                gc.collect()
                
                # 仅对较大的数据集进行数据类型优化
                if large_dataset:
                    for col in result_df.select_dtypes(include=['float64']).columns:
                        result_df[col] = result_df[col].astype('float32')
            
            # 垂直方向的数据优化 - 记录内存影响
            post_df_memory = current_process.memory_info().rss / (1024 * 1024)  # MB
            memory_diff = post_df_memory - pre_df_memory
            
            # 结果统计
            merged_jobs_count = len(result_df)
            server_count = len(resource_index) if resource_index else 0
            success_rate = merged_jobs_count / total_jobs * 100 if total_jobs > 0 else 0
            
            # 生成高级性能报告
            print("\n" + "═"*50)
            print("✨ 任务与资源匹配完成报告 ✨")
            print("═"*50)
            print(f"\n📊 处理统计:")
            print(f"  • 总任务数: {total_jobs:,}")
            print(f"  • 成功匹配数: {merged_jobs_count:,} ({success_rate:.1f}%)")
            print(f"  • 处理的服务器数: {server_count:,}")
            print(f"\n⏱ 性能指标:")
            print(f"  • 总运行时间: {time_str}")
            print(f"  • 平均处理速度: {total_jobs/total_time:.1f} 条/秒")
            print(f"  • 高峰内存使用: {post_df_memory:.1f}MB (变化: {memory_diff:+.1f}MB)")
            print(f"  • 当前系统内存使用: {psutil.virtual_memory().percent}%")
            print("\n" + "═"*50)
            
            # 强制最终垃圾回收
            gc.collect()
            
            return result_df
            
        except MemoryError as e:
            print(f"\n❗ 严重内存错误: {str(e)}")
            print("❗ 无法将结果转换为DataFrame，内存不足")
            print(f"ℹ️ 当前内存状态: {psutil.virtual_memory().percent}% 使用, {psutil.virtual_memory().available/(1024**3):.1f}GB 可用")
            print("\nℹ️ 建议解决方案:")
            print("  1. 使用更大内存的服务器")
            print("  2. 增加交换空间 (swap)")
            print("  3. 减小输入数据集大小")
            print("  4. 修改代码以使用流式处理或图形化接口")
            return pd.DataFrame()
        except Exception as e:
            print(f"\n⚠️ 结果处理错误: {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            print("ℹ️ 尝试恢复并返回部分结果...")
            try:
                # 尝试恢复 - 返回部分结果
                if all_results and len(all_results) > 0:
                    # 假设我们可以处理至少一部分数据
                    recovery_size = min(len(all_results), 100000)  # 限制恢复大小
                    print(f"ℹ️ 恢复模式: 尝试使用前 {recovery_size:,} 条记录...")
                    partial_results = all_results[:recovery_size]
                    return pd.DataFrame(partial_results)
                else:
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame()

    def _create_resource_index(self):
        """创建资源数据索引，按服务器和时间窗口组织 - 超级优化版
        使用多进程并行处理和高级内存管理加速索引创建，大幅提高效率
        """
        import gc  # 导入垃圾回收模块
        import psutil  # 用于监控系统资源
        import multiprocessing
        import concurrent.futures
        from collections import defaultdict
        
        if not self.resources:
            return {}
        
        print("创建资源数据索引...")
        start_time = time.time()
        
        # 获取当前系统状态
        mem_info = psutil.virtual_memory()
        print(f"当前系统内存使用: {mem_info.percent}% ({mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB)")
        
        # 性能参数自动配置 - 基于系统资源智能调整
        num_cores = multiprocessing.cpu_count()
        num_resource_types = len(self.resources)
        
        # 计算最佳并行度 - 动态平衡性能与内存使用
        if mem_info.percent > 85:  # 内存严重不足时
            # 使用较低的并行度，保留系统稳定性
            num_workers = max(1, min(8, num_cores // 8))  
            print("⚠️ 系统内存紧张，使用低并行模式")
        elif mem_info.percent > 70:  # 内存紧张但可用
            # 限制并行度以避免内存溢出
            num_workers = max(4, min(16, num_cores // 4))  
            print("⚠️ 系统内存较紧张，使用中等并行模式")
        else:  # 内存充足
            # 大规模并行，最多使用75%的核心
            num_workers = max(min(32, num_cores // 2), int(num_cores * 0.75))  
            print("✓ 系统内存充足，使用高并行模式")
        
        # 根据资源类型数量调整并行度
        num_workers = min(num_workers, num_resource_types)
        print(f"系统有 {num_cores} 个核心, 将使用 {num_workers} 个并行进程处理 {num_resource_types} 种资源类型")
        
        # 直接使用预先提取的服务器列表，不再重复提取
        if hasattr(self, 'pre_extracted_servers') and self.pre_extracted_servers:
            print(f"DEBUG: 直接使用预先提取的服务器列表, 包含 {len(self.pre_extracted_servers)} 个服务器")
            all_servers = set(self.pre_extracted_servers)
        else:
            # 如果没有预先提取的服务器列表，创建一个空集合
            all_servers = set()
            print("警告: 未提供预先提取的服务器列表，请确保在调用规耲0中传递服务器名称")
        
        num_servers = len(all_servers)
        print(f"任务中使用的有效服务器数量: {num_servers}")
        
        # 调试输出
        if num_servers == 0:
            print("警告: 未能从任务数据中提取有效服务器名称，请检查exec_hosts字段格式和parse_exec_hosts函数实现")
        
        # 优化：使用defaultdict简化索引创建
        # 使用更高效的数据结构，避免重复检查键是否存在
        resource_index = defaultdict(lambda: defaultdict(dict))
        
        # 跟踪处理的资源类型统计
        resource_stats = {}
        
        # 优化：动态监控系统资源
        last_mem_check = time.time()
        mem_check_interval = 30  # 每30秒检查一次内存状态
        
        # 使用并行处理大幅加速资源索引创建
        print(f"开始并行处理资源数据...")
        
        try:
            # 使用多进程并行处理 - 性能提升显著
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 创建并行任务
                futures = {}
                # 将all_servers转换为列表，以确保能被正确序列化和传递给子进程
                all_servers_list = list(all_servers)
                print(f"DEBUG: 传递给子进程的服务器名称集合大小: {len(all_servers_list)}")
                if len(all_servers_list) > 0:
                    print(f"DEBUG: 服务器名称示例: {all_servers_list[:min(5, len(all_servers_list))]}")
                
                # 如果服务器列表为空，输出警告信息
                if len(all_servers_list) == 0:
                    print("警告: 服务器列表为空，无法进行资源索引创建。请确保第一次解析exec_hosts字段时已正确提取服务器名称。")
                
                for resource_type, server_data_dict in self.resources.items():
                    # 使用列表而不是集合传递给子进程，以避免序列化问题
                    # 如果服务器列表为空，则跳过处理
                    if not all_servers_list:
                        print(f"警告: 跳过处理资源类型 {resource_type}，因为服务器列表为空")
                        continue
                    futures[executor.submit(self._process_resource_type, resource_type, server_data_dict, all_servers_list)] = resource_type
                
                # 使用tqdm跟踪进度
                processed_count = 0
                
                error_count = 0
                with tqdm(total=len(futures), desc="并行处理资源类型", unit="类型") as pbar:
                    for future in concurrent.futures.as_completed(futures):
                        resource_type = futures[future]
                        try:
                            # 获取并行处理结果
                            processed_resource_type, sub_index, stats = future.result()
                            processed_count += 1
                            
                            # 优化：使用批量更新而非逐项更新
                            batch_updates = defaultdict(dict)
                            
                            # 预处理要更新的数据
                            for server, windows in sub_index.items():
                                for window_start, resource_data in windows.items():
                                    if 'window_end' not in resource_data:
                                        resource_data['window_end'] = window_start + pd.Timedelta(hours=1)
                                    batch_updates[server][window_start] = resource_data
                            
                            # 批量更新资源索引
                            for server, windows in batch_updates.items():
                                for window_start, resource_data in windows.items():
                                    # 如果窗口不存在，则创建新窗口
                                    if window_start not in resource_index[server]:
                                        resource_index[server][window_start] = {
                                            'window_end': resource_data['window_end']
                                        }
                                    
                                    # 一次性更新所有资源数据
                                    for key, value in resource_data.items():
                                        if key != 'window_end':  # 避免覆盖window_end
                                            resource_index[server][window_start][key] = value
                            
                            # 保存统计信息
                            resource_stats[processed_resource_type] = stats
                            
                        except Exception as e:
                            error_count += 1
                            print(f"\n警告: 处理资源类型 {resource_type} 时出错: {type(e).__name__}: {str(e)}")
                            if error_count > len(futures) // 3:  # 如果超过1/3的任务失败，终止并行处理
                                print("错误率过高，切换到顺序处理...")
                                break
                        
                        # 更新进度条
                        pbar.update(1)
                        
                        # 优化：自适应内存监控与清理
                        current_time = time.time()
                        if current_time - last_mem_check > mem_check_interval:
                            # 强制垃圾回收
                            gc.collect()
                            mem_info = psutil.virtual_memory()
                            # 只在内存使用超过阈值时显示警告
                            if mem_info.percent > 75:
                                print(f"\n⚠️ 内存使用率较高: {mem_info.percent}% ({mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB)")
                            last_mem_check = current_time
                            
                            # 动态调整内存检查频率
                            if mem_info.percent > 85:
                                mem_check_interval = 15  # 内存紧张时更频繁检查
                            elif mem_info.percent < 60:
                                mem_check_interval = 60  # 内存充足时减少检查频率
        
        except Exception as e:
            print(f"\n并行处理时发生错误: {type(e).__name__}: {str(e)}")
            print("切换到顺序处理模式...")
            
            # 优化：顺序处理的备用方案，使用批处理减少内存压力
            resource_stats = {}
            for resource_type, server_data_dict in tqdm(self.resources.items(), desc="顺序处理资源类型", unit="类型"):
                try:
                    _, sub_index, stats = self._process_resource_type(resource_type, server_data_dict, all_servers)
                    
                    # 高效率批量合并索引
                    for server, windows in sub_index.items():
                        for window_start, resource_data in windows.items():
                            # 如果窗口不存在，则使用默认值创建
                            if 'window_end' not in resource_index[server][window_start]:
                                resource_index[server][window_start]['window_end'] = window_start + pd.Timedelta(hours=1)
                            
                            # 更新资源数据
                            for key, value in resource_data.items():
                                if key != 'window_end':
                                    resource_index[server][window_start][key] = value
                    
                    resource_stats[resource_type] = stats
                    
                    # 每个资源类型处理后进行一次垃圾回收
                    gc.collect()
                except Exception as e:
                    print(f"警告: 顺序处理 {resource_type} 时出错: {type(e).__name__}: {str(e)}")
        
        # 优化：转换defaultdict为普通dict以便序列化
        final_index = {}
        for server, windows in resource_index.items():
            final_index[server] = dict(windows)
        
        # 显示资源数据统计，按数据点数量排序
        print("\n资源数据索引统计:")
        sorted_stats = sorted(resource_stats.items(), key=lambda x: x[1]['data_points'], reverse=True)
        for resource_type, stats in sorted_stats:
            print(f"  - {resource_type}: {stats['server_count']} 台服务器, {stats['data_points']:,} 个数据点")
        
        # 统计索引大小
        server_count = len(final_index)
        window_count = sum(len(windows) for server, windows in final_index.items())
        
        # 输出创建时间和完整统计
        elapsed_time = time.time() - start_time
        print(f"\n✅ 资源索引创建完成: {server_count} 台服务器, {window_count:,} 个时间窗口, {len(resource_stats)} 种资源类型")
        print(f"总耗时: {elapsed_time:.2f} 秒, 平均每秒处理 {int(window_count/max(1,elapsed_time)):,} 个时间窗口")
        
        # 最终内存清理
        gc.collect()
        mem_info = psutil.virtual_memory()
        print(f"当前内存使用: {mem_info.percent}% ({mem_info.used/1024/1024/1024:.1f}GB/{mem_info.total/1024/1024/1024:.1f}GB)")
        
        return final_index

    def _process_resource_type(self, resource_type, server_data_dict, all_servers):
        """并行处理单个资源类型的数据并返回子索引 - 优化版
        使用更高效的数据处理方法和内存管理策略
        
        参数：
            resource_type: 资源类型
            server_data_dict: 服务器数据字典
            all_servers: 所有服务器名称集合
        返回：
            处理后的资源类型、子索引和统计信息
        """
        import gc
        
        # 使用defaultdict减少条件检查
        from collections import defaultdict
        sub_index = defaultdict(dict)
        count = 0
        server_count = 0
        
        # 预先定义常量以提高性能
        HOUR_TIMEDELTA = pd.Timedelta(hours=1)
        
        # 初始化调试计数器
        matched_servers = 0
        processed_windows = 0
        
        # 以批处理方式处理服务器数据，避免内存溢出
        server_items = list(server_data_dict.items())
        batch_size = 20  # 每批处理20台服务器
        
        for batch_start in range(0, len(server_items), batch_size):
            batch_end = min(batch_start + batch_size, len(server_items))
            batch = server_items[batch_start:batch_end]
            
            # 在批处理开始时添加调试信息
            if batch_start == 0:
                print(f"DEBUG: 当前处理资源类型 {resource_type}, 样本服务器: {[s for s, _ in batch[:3]]}")
                # 确保all_servers在子进程中有效
                print(f"DEBUG: 在子进程中收到的服务器列表类型: {type(all_servers)}, 大小: {len(all_servers)}")
                if len(all_servers) > 0:
                    print(f"DEBUG: 从任务数据中提取的有效服务器示例: {list(all_servers)[:5] if isinstance(all_servers, set) else all_servers[:5]}")
                else:
                    print("警告: 没有收到有效的服务器名称列表!")
                
            # 确保all_servers是集合类型，以提高查找效率
            all_servers_set = set(all_servers) if not isinstance(all_servers, set) else all_servers
                
            for server, df in batch:
                # 规范化服务器名称为小写
                server_lower = server.lower()
                matched_server = None
                
                # 首先尝试直接匹配
                if server_lower in all_servers_set:
                    matched_server = server_lower
                else:
                    # 已删除子串匹配代码，只保留精确匹配
                    matched_server = None
                
                # 如果没有匹配到任何服务器，跳过处理
                if matched_server is None:
                    # 如果这是第一批且某些服务器无法匹配，打印调试信息
                    if batch_start == 0 and server_count < 3:
                        print(f"DEBUG: 服务器 '{server}' 未在任务使用的服务器列表中找到，已跳过")
                    continue
                    
                try:
                    # 快速检查数据有效性
                    if df is None or df.empty or len(df) < 2:  # 至少需要2个数据点才有意义
                        continue
                        
                    # 检查并优化值列
                    # 初始化timestamp_col以避免UnboundLocalError
                    timestamp_col = None
                    
                    if 'value' in df.columns:
                        value_col = 'value'
                        
                        # 使用向量化操作和高效数据类型转换
                        # 统一使用float32减少内存使用
                        if not pd.api.types.is_numeric_dtype(df[value_col]):
                            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                            df = df.dropna(subset=[value_col])
                            # 转换为更低精度的浮点数以节省内存
                            df[value_col] = df[value_col].astype('float32')
                    else:
                        # 智能检测可能的值列并保留第一个找到的
                        value_col = None
                        timestamp_col = None
                        
                        # 单次迭代检查所有列
                        for col in df.columns:
                            col_lower = str(col).lower()
                            if timestamp_col is None and ('time' in col_lower or 'timestamp' in col_lower):
                                timestamp_col = col
                            elif value_col is None and col_lower != 'value' and 'server' not in col_lower and col_lower != 'metric':
                                # 尝试将此列转换为数值以验证它是否确实是值列
                                try:
                                    pd.to_numeric(df[col], errors='raise')
                                    value_col = col
                                except:
                                    continue
                        
                        # 如果找不到值列，跳过这个服务器
                        if value_col is None:
                            continue
                    
                    # 检查并获取时间戳列
                    if timestamp_col is None:
                        # 没有预先找到时间戳列，尝试查找
                        for col in df.columns:
                            col_lower = str(col).lower()
                            if 'time' in col_lower or 'date' in col_lower:
                                timestamp_col = col
                                break
                    
                    # 确保找到时间戳列
                    if timestamp_col is None or timestamp_col not in df.columns:
                        # 尝试找名为'timestamp'的列
                        if 'timestamp' in df.columns:
                            timestamp_col = 'timestamp'
                        else:
                            continue  # 没有时间戳列，跳过此服务器
                    
                    # 优化数据格式转换
                    try:
                        # 使用更高效的日期时间转换
                        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                            # 使用缓存和并行处理加速时间转换
                            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce', cache=True)
                            # 移除无效时间戳记录
                            df = df.dropna(subset=[timestamp_col])
                            if df.empty:
                                continue
                    except Exception as e:
                        continue  # 时间戳转换失败，跳过
                    
                    # 优化：使用resample替代逐行操作，大幅提高处理速度
                    try:
                        # 设置时间索引
                        df = df.set_index(timestamp_col)
                        
                        # 对数值列确保为数值类型
                        if not pd.api.types.is_numeric_dtype(df[value_col]):
                            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
                            df = df.dropna(subset=[value_col])
                        
                        # 使用resample按小时聚合数据
                        # 使用平均值聚合同一小时内的多个测量值
                        hourly_data = df.resample('H')[value_col].mean()
                        
                        # 将结果直接转换为字典，这比单独添加每个值更高效
                        time_value_dict = hourly_data.to_dict()
                        
                        # 一次性批量添加到索引中
                        windows_processed = 0
                        for window_start, value in time_value_dict.items():
                            # 跳过NaN值
                            if pd.isna(value):
                                continue
                                
                            # 使用defaultdict简化代码
                            if window_start not in sub_index[server_lower]:
                                sub_index[server_lower][window_start] = {
                                    'window_end': window_start + HOUR_TIMEDELTA
                                }
                            
                            # 存储资源值（不再添加_value后缀，直接使用资源类型名称）
                            sub_index[server_lower][window_start][resource_type] = float(value)  # 确保为基本浮点类型
                            count += 1
                            windows_processed += 1
                            processed_windows += 1
                        
                        if windows_processed > 0:
                            # 记录已成功处理的服务器
                            matched_servers += 1
                            server_count += 1
                            
                            # 提供首几个服务器的处理详情
                            if server_count <= 3:
                                print(f"DEBUG: 匹配成功 - 资源服务器: '{server}' 处理了 {windows_processed} 个时间窗口")
                            
                    except Exception as e:
                        # 详细记录错误信息以便调试
                        print(f"警告: 处理 {server} 的 {resource_type} resample时出错: {type(e).__name__}: {str(e)}")
                except Exception as e:
                    print(f"警告: 处理 {server} 的 {resource_type} 数据时出错: {type(e).__name__}: {str(e)}")
            
            # 批处理后主动释放内存
            gc.collect()
        
        # 转换defaultdict为普通dict以便序列化
        final_sub_index = dict(sub_index)
        
        # 收集更详细的统计信息
        stats = {
            'server_count': server_count,
            'data_points': count,
            'matched_servers': matched_servers,
            'processed_windows': processed_windows,
            'all_servers_count': len(all_servers_set) if isinstance(all_servers_set, set) else len(all_servers)
        }
        
        # 输出详细的调试统计信息
        print(f"DEBUG: 资源类型 {resource_type} 统计信息:")
        print(f"  - 从任务中提取的服务器总数: {len(all_servers)}")
        print(f"  - 实际匹配的服务器数: {matched_servers}")
        print(f"  - 处理的时间窗口总数: {processed_windows}")
        print(f"  - 创建的子索引包含 {len(final_sub_index)} 台服务器")
        
        # 打印索引中服务器的示例
        if len(final_sub_index) > 0:
            print(f"  - 子索引中的服务器示例: {list(final_sub_index.keys())[:3]}")
            
            # 查看第一个服务器的窗口数量
            first_server = list(final_sub_index.keys())[0]
            print(f"  - 第一个服务器 '{first_server}' 有 {len(final_sub_index[first_server])} 个时间窗口")
        else:
            print("  - 警告: 子索引为空, 没有匹配到任何服务器!")
        
        return resource_type, final_sub_index, stats
    
    def _parse_exec_hosts(self, host_string, row_num=None):
        """
        解析服务器列表字符串。
        调用全局parse_exec_hosts函数以统一解析逻辑。
        
        参数:
            host_string: 原始服务器字符串
            row_num: 可选的调试行号
            
        返回:
            标准化的服务器名称列表
        """
        # 直接使用全局函数来处理所有服务器解析逻辑
        return parse_exec_hosts(host_string, row_num)

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
            'cpu_total_value', 'cpu_system_value', 'gpu_avg_value', 'gpu_util_value', 
            'mem_percent_value', 'mem_used_value', 'load_1min_value', 'load_15min_value', 
            'power_value', 'temperature_value'
        ]
        
        # 改进过滤数据逻辑: 更灵活地处理资源值为'n/a'的情况
        original_rows = len(df)
        
        # 先打印原始数据的基本信息
        print(f"\n===== 数据过滤过程 =====")
        print(f"原始数据: {original_rows} 行")
        
        # 检查数据中的服务器列情况
        server_stats = df['server'].value_counts()
        non_na_servers = server_stats[server_stats.index != 'n/a'].sum()
        print(f"服务器列统计: 有效服务器名称数: {non_na_servers}/{len(df)} ({non_na_servers/len(df)*100:.1f}%)")
        if 'n/a' in server_stats.index:
            print(f"  'n/a'服务器数量: {server_stats['n/a']} ({server_stats['n/a']/len(df)*100:.1f}%)")
        
        # 检查exec_hosts列情况
        exec_hosts_na_count = (df['exec_hosts'] == 'n/a').sum()
        print(f"exec_hosts列统计: 'n/a'数量: {exec_hosts_na_count}/{len(df)} ({exec_hosts_na_count/len(df)*100:.1f}%)")
        
        # 为每个资源列创建布尔矩阵，标记哪些值是'n/a'
        na_matrix = df[resource_columns] == 'n/a'
        
        # 统计每列的n/a数量
        na_count_by_column = na_matrix.sum()
        print(f"\n资源列'n/a'分布:")
        for col, count in na_count_by_column.items():
            print(f"  - {col}: {count}/{len(df)} ({count/len(df)*100:.1f}%)")
        
        # 统计每一行有多少个资源列是n/a
        na_counts_per_row = na_matrix.sum(axis=1)
        # 统计不同数量的n/a列的行数
        na_dist = na_counts_per_row.value_counts().sort_index()
        print(f"\n行级别'n/a'分布:")
        for count, rows in na_dist.items():
            print(f"  - {count}个'n/a'资源列: {rows}行 ({rows/len(df)*100:.1f}%)")
            
        # 标记所有资源列都是n/a的行
        all_na_mask = na_matrix.all(axis=1)
        all_na_count = all_na_mask.sum()
        print(f"\n所有资源列都是'n/a'的行数: {all_na_count}/{len(df)} ({all_na_count/len(df)*100:.1f}%)")
        
        # 智能过滤决策 - 姻合多种情况
        if all_na_count == len(df):
            # 情况1: 所有行的所有资源值都是n/a
            print("\n⚠️ 关键情况: 所有记录的所有资源列都是'n/a'")
            print("  执行策略: 跳过过滤步骤以避免空结果，保留所有原始数据")
            df_filtered = df
            filtered_rows = 0
        elif all_na_count > len(df) * 0.9:
            # 情况2: 超过90%的行所有资源值都是n/a
            print(f"\n⚠️ 关键情况: {all_na_count/len(df)*100:.1f}%的记录所有资源值均为'n/a'")
            print("  执行策略: 这可能是资源匹配问题或服务器名称格式问题导致，保留所有数据")
            df_filtered = df
            filtered_rows = 0
        else:
            # 情况3: 正常情况，过滤掉所有资源列都是n/a的行
            print("\n✔️ 正常过滤: 将过滤掉所有资源列都是'n/a'的行")
            df_filtered = df[~all_na_mask]
            filtered_rows = original_rows - len(df_filtered)
            
            # 安全检查 - 如果过滤后数据太少，回退到原始数据
            if len(df_filtered) < len(df) * 0.1 and len(df) > 0:
                print(f"\n⚠️ 危险情况: 过滤后仅保留了{len(df_filtered)}/{len(df)}条记录({len(df_filtered)/len(df)*100:.1f}%)")
                print("  执行策略: 为避免过度数据丢失，回退并保留原始数据")
                df_filtered = df
                filtered_rows = 0
        
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
        
        # 处理exec_hosts列中重复的服务器名称
        if 'exec_hosts' in df_filtered.columns:
            print("优化exec_hosts列中的服务器名称...")
            def clean_server_names(hosts_str):
                if pd.isna(hosts_str) or hosts_str == 'n/a':
                    return hosts_str
                    
                # 分割服务器名称
                servers = hosts_str.split()
                
                # 去除重复项并保持原始顺序
                unique_servers = []
                for server in servers:
                    if server not in unique_servers:
                        unique_servers.append(server)
                
                # 如果服务器名称大于3个，显示数量而不是全部列出
                if len(unique_servers) > 3:
                    # 如果所有服务器名称都相同
                    if len(set(unique_servers)) == 1:
                        return f"{unique_servers[0]} (×{len(servers)})"
                    else:
                        # 列出前3个不同的服务器名称，后面用等表示
                        return f"{', '.join(unique_servers[:3])} 等 {len(unique_servers)} 台服务器"
                
                # 如果服务器名称不多，直接返回不重复的列表
                return ' '.join(unique_servers)
            
            # 应用处理函数
            df_filtered['exec_hosts'] = df_filtered['exec_hosts'].apply(clean_server_names)
            print(f"  - 服务器名称优化完成")
        
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
            # 统一数据类型，解决Parquet保存错误
            print("数据类型预处理...")
            # 资源值列
            resource_value_columns = [col for col in df_filtered.columns if col.endswith('_value')]
            
            for col in resource_value_columns:
                # 将'n/a'转换为浮点型NaN
                try:
                    df_filtered[col] = pd.to_numeric(df_filtered[col].replace('n/a', np.nan), errors='coerce')
                    print(f"  - 列 {col} 转换为数值类型")
                except Exception as e:
                    print(f"  - 警告: 列 {col} 类型转换出错: {e}")
                    # 如果无法转换为数字，则保持为字符串
                    df_filtered[col] = df_filtered[col].astype(str)
                    print(f"  - 列 {col} 转换为字符串类型")
            
            # 使用PyArrow引擎和Snappy压缩算法提高效率
            try:
                df_filtered.to_parquet(parquet_file, engine='pyarrow', compression='snappy')
            except Exception as e:
                print(f"\n错误: 保存Parquet文件失败: {e}")
                # 备用方案: 保存为网络安全格式
                print("\n尝试保存为临时CSV格式...")
                temp_csv = os.path.join(self.output_dir, 'integrated_data_backup.csv.gz')
                df_filtered.to_csv(temp_csv, index=False, compression='gzip')
                print(f"临时文件已保存至: {temp_csv}")
        
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
                                if 'timestamps' in col_lower or 'date' in col_lower:
                                    timestamp_col = col
                                elif col_lower != 'value' and 'value_column' not in col_lower:
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
    parser.add_argument('--row', '--rows', type=int, default=100, help='测试模式下读取的行数')
    parser.add_argument('--output', type=str, help='指定输出目录')
    parser.add_argument('--verbose', action='store_true', help='输出详细的调试信息')
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = args.output if args.output else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if args.test:
        output_dir = os.path.join(output_dir, f'test_rows{args.row}')
        print(f"=== 运行快速测试模式 (每个文件前 {args.row} 行) ===")
    else:
        output_dir = os.path.join(output_dir, f'full_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        print("=== 运行完整分析模式 ===")
    
    print(f"测试输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 只读取一次任务数据
    print("开始资源与任务整合分析...")
    jobs = read_job_data(DATA_PATHS['job_data'], args.test, args.row)
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
    # 传递预先提取的服务器名称，避免重复提取
    analyzer = ResourceJobAnalyzer(jobs, resources, output_dir, pre_extracted_servers=pre_data['servers'])
    # 设置verbose模式
    analyzer.verbose = args.verbose
    
    # 设置全局函数的verbose参数
    # 这样parse_exec_hosts函数可以访问verbose设置
    global parse_exec_hosts
    parse_exec_hosts.verbose = args.verbose
    
    result = analyzer.process()
    
    print(f"分析完成。结果已保存到: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())