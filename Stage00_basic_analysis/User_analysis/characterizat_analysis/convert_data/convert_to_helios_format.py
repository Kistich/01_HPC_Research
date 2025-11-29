#!/usr/bin/env python3

import pandas as pd
import os
import hashlib
from datetime import datetime
import numpy as np

# 创建输出目录
output_dir = "/mnt/raid/liuhongbin/job_analysis/job_analysis/User_behavior_analysis/convert_data/all_data"
os.makedirs(output_dir, exist_ok=True)

# 读取二期集群作业数据
second_gen_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/llm_results/analysis_20250309_115130/all_classifications_20250309_115130.csv"
df = pd.read_csv(second_gen_path)

print(f"读取到 {len(df)} 条二期集群作业记录")
print(f"原始数据列: {df.columns.tolist()}")

# 创建用户ID映射(保持一致性)
def hash_value(value):
    return hashlib.md5(str(value).encode()).hexdigest()[:4]

# 根据提供的列名信息，确认使用user_id
if 'user_id' in df.columns:
    users = df['user_id'].dropna().unique()
    user_map = {user: f"u{hash_value(user)}" for user in users}
    print(f"使用user_id列进行用户映射，共有{len(users)}个用户")
else:
    users = df['user'].dropna().unique()
    user_map = {user: f"u{hash_value(user)}" for user in users}
    print(f"使用user列进行用户映射，共有{len(users)}个用户")

# 将数据转换为HeliosData格式
helios_df = pd.DataFrame()

# 1. job_id - 作业唯一ID (直接使用job_id字段)
helios_df['job_id'] = df['job_id']

# 2. user - 用户的哈希ID (使用user_id字段并应用哈希映射)
if 'user_id' in df.columns:
    helios_df['user'] = df['user_id'].map(user_map)
else:
    helios_df['user'] = df['user'].map(user_map)

# 3. gpu_num - 作业需要的GPU数量 (直接使用gpu_num字段)
helios_df['gpu_num'] = df['gpu_num']

# 4. cpu_num - 作业需要的CPU数量 (直接使用num_processors字段)
helios_df['cpu_num'] = df['num_processors']

# 5. node_num - 作业使用的节点数量 (直接使用num_exec_hosts字段)
helios_df['node_num'] = df['num_exec_hosts']

# 6. state - 作业终止状态
# 根据列表中提供的信息，我们应该使用job_status_str字段
state_map = {
    'PEND': 'PENDING', 
    'RUN': 'RUNNING',
    'DONE': 'COMPLETED',
    'EXIT': 'FAILED',
    'USUSP': 'CANCELLED',
    'PSUSP': 'CANCELLED',
    'ZOMBI': 'FAILED',
    'UNKWN': 'FAILED'
}

# 使用job_status_str或jstatus映射状态
if 'job_status_str' in df.columns:
    helios_df['state'] = df['job_status_str'].map(state_map).fillna('FAILED')
    print("使用job_status_str映射状态")
elif 'jstatus' in df.columns:
    # 如果jstatus是数字代码，需要先映射到状态字符串
    jstatus_str_map = {
        1: 'PEND',   # 通常1表示等待中
        2: 'RUN',    # 通常2表示运行中
        32: 'DONE',  # 通常32表示已完成
        64: 'EXIT'   # 通常64表示失败
    }
    temp_status = df['jstatus'].map(jstatus_str_map).fillna('UNKWN')
    helios_df['state'] = temp_status.map(state_map).fillna('FAILED')
    print("使用jstatus映射状态")

# 根据HeliosData要求，PENDING和RUNNING状态视为FAILED（因为这些作业没有正常完成）
helios_df['state'] = helios_df['state'].replace(['PENDING', 'RUNNING'], 'FAILED')

# 7,8,9. submit_time, start_time, end_time - 时间字段
def parse_time(time_str):
    if pd.isna(time_str):
        return None
    
    try:
        # 尝试解析不同格式的时间
        formats = [
            '%Y-%m-%d %H:%M:%S', 
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%Y%m%d%H%M%S'
        ]
        
        # 字符串格式时间
        for fmt in formats:
            try:
                return datetime.strptime(str(time_str).strip(), fmt)
            except:
                continue
                
        # Unix时间戳(秒)
        try:
            return datetime.fromtimestamp(float(time_str))
        except:
            pass
    except:
        pass
        
    return None

print("处理时间字段...")
# 直接使用匹配的时间字段
time_fields = ['submit_time', 'start_time', 'end_time']
for field in time_fields:
    helios_df[field] = df[field].apply(parse_time)
    print(f"处理{field}字段")

# 10. duration - 作业执行时间(秒)
# 11. queue - 作业队列时间(秒)
def calc_duration(row):
    try:
        if pd.notna(row['start_time']) and pd.notna(row['end_time']):
            return max(0, int((row['end_time'] - row['start_time']).total_seconds()))
    except:
        pass
    return 0

def calc_queue_time(row):
    try:
        if pd.notna(row['submit_time']) and pd.notna(row['start_time']):
            return max(0, int((row['start_time'] - row['submit_time']).total_seconds()))
    except:
        pass
    return 0

print("计算duration和queue字段")
helios_df['duration'] = helios_df.apply(calc_duration, axis=1)
helios_df['queue'] = helios_df.apply(calc_queue_time, axis=1)

# 过滤掉无效记录 - 确保关键时间字段非空
valid_records = helios_df['submit_time'].notna() & helios_df['start_time'].notna() & helios_df['end_time'].notna()
print(f"有{(~valid_records).sum()}条记录因时间字段缺失而被过滤")
helios_df = helios_df[valid_records]

# 转换时间列为字符串格式
for col in time_fields:
    helios_df[col] = helios_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

print(f"转换完成，保留了 {len(helios_df)} 条有效记录")

# 保存结果
output_path = os.path.join(output_dir, "second_gen_helios_format.csv")
helios_df.to_csv(output_path, index=False)
print(f"数据已保存到 {output_path}")

# 输出转换结果统计
print("\n=== 转换结果统计 ===")
print(f"总记录数: {len(helios_df)}")
print(f"GPU作业占比: {(helios_df['gpu_num'] > 0).mean()*100:.2f}%")
print(f"多节点作业占比: {(helios_df['node_num'] > 1).mean()*100:.2f}%")
print(f"作业状态分布:")
for state, count in helios_df['state'].value_counts().items():
    print(f"  {state}: {count} ({count/len(helios_df)*100:.2f}%)")

# 生成转换结果总结，并保存至convert_summary.txt
def generate_conversion_summary(helios_df, df, output_dir):
    """生成数据转换总结并保存为文本文件"""
    
    summary_path = os.path.join(output_dir, "convert_summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("# 二期集群数据转换为HeliosData格式总结报告\n\n")
        
        # 基本数据信息
        f.write("## 1. 基本数据信息\n\n")
        f.write(f"* 原始数据记录总数: {len(df):,}\n")
        f.write(f"* 转换后有效记录总数: {len(helios_df):,}\n")
        f.write(f"* 数据过滤率: {(1 - len(helios_df)/len(df))*100:.2f}%\n")
        
        # 时间范围计算
        try:
            min_date = min(pd.to_datetime(helios_df['submit_time'])).date()
            max_date = max(pd.to_datetime(helios_df['end_time'])).date()
            f.write(f"* 时间范围: {min_date} 至 {max_date}\n\n")
        except:
            f.write("* 时间范围: 无法确定\n\n")
        
        # 用户信息
        f.write("## 2. 用户统计\n\n")
        original_users = df['user_id'].nunique() if 'user_id' in df.columns else df['user'].nunique()
        f.write(f"* 原始唯一用户数: {original_users:,}\n")
        f.write(f"* 转换后唯一用户数: {helios_df['user'].nunique():,}\n")
        
        # 显示前10个最活跃的用户及其作业数量
        user_counts = helios_df['user'].value_counts()
        f.write("\n### 最活跃用户（前10名）\n\n")
        f.write("| 用户 | 作业数量 | 占比 |\n")
        f.write("|------|----------|------|\n")
        for user, count in user_counts.head(10).items():
            percentage = count/len(helios_df)*100
            f.write(f"| {user} | {count:,} | {percentage:.2f}% |\n")
        
        # 资源使用统计
        f.write("\n## 3. 资源使用统计\n\n")
        
        # GPU统计
        f.write("### GPU使用情况\n\n")
        gpu_jobs = helios_df[helios_df['gpu_num'] > 0]
        f.write(f"* 使用GPU的作业数量: {len(gpu_jobs):,} ({len(gpu_jobs)/len(helios_df)*100:.2f}%)\n")
        f.write(f"* 平均每个作业使用的GPU数量: {helios_df['gpu_num'].mean():.2f}\n")
        
        # GPU使用量分布
        gpu_dist = helios_df['gpu_num'].value_counts().sort_index()
        f.write("\n#### GPU数量分布\n\n")
        f.write("| GPU数量 | 作业数 | 占比 |\n")
        f.write("|---------|--------|------|\n")
        for gpu_num, count in gpu_dist.items():
            percentage = count/len(helios_df)*100
            f.write(f"| {gpu_num} | {count:,} | {percentage:.2f}% |\n")
        
        # 节点使用统计
        f.write("\n### 节点使用情况\n\n")
        multi_node_jobs = helios_df[helios_df['node_num'] > 1]
        f.write(f"* 多节点作业数量: {len(multi_node_jobs):,} ({len(multi_node_jobs)/len(helios_df)*100:.2f}%)\n")
        f.write(f"* 平均每个作业使用的节点数量: {helios_df['node_num'].mean():.2f}\n")
        
        # 节点数量分布 (显示前10项)
        node_dist = helios_df['node_num'].value_counts().sort_index()
        f.write("\n#### 节点数量分布\n\n")
        f.write("| 节点数量 | 作业数 | 占比 |\n")
        f.write("|----------|--------|------|\n")
        for node_num, count in node_dist.head(10).items():
            percentage = count/len(helios_df)*100
            f.write(f"| {node_num} | {count:,} | {percentage:.2f}% |\n")
        
        # 作业执行情况
        f.write("\n## 4. 作业执行情况\n\n")
        
        # 作业状态分布
        f.write("### 作业状态分布\n\n")
        f.write("| 状态 | 作业数 | 占比 |\n")
        f.write("|------|--------|------|\n")
        for state, count in helios_df['state'].value_counts().items():
            percentage = count/len(helios_df)*100
            f.write(f"| {state} | {count:,} | {percentage:.2f}% |\n")
        
        # 时间统计
        f.write("\n### 执行时间统计\n\n")
        f.write(f"* 平均执行时间: {helios_df['duration'].mean()/3600:.2f} 小时\n")
        f.write(f"* 中位执行时间: {helios_df['duration'].median()/3600:.2f} 小时\n")
        f.write(f"* 最长执行时间: {helios_df['duration'].max()/3600:.2f} 小时\n")
        
        f.write("\n### 队列时间统计\n\n")
        f.write(f"* 平均队列时间: {helios_df['queue'].mean()/60:.2f} 分钟\n")
        f.write(f"* 中位队列时间: {helios_df['queue'].median()/60:.2f} 分钟\n")
        f.write(f"* 最长队列时间: {helios_df['queue'].max()/3600:.2f} 小时\n")
        
        # 数据转换过程说明
        f.write("\n## 5. 数据转换过程说明\n\n")
        f.write("### 转换映射关系\n\n")
        f.write("| HeliosData字段 | 原始数据字段 |\n")
        f.write("|---------------|-------------|\n")
        f.write("| job_id | job_id |\n")
        f.write(f"| user | {'user_id' if 'user_id' in df.columns else 'user'} (已哈希) |\n")
        f.write("| gpu_num | gpu_num |\n")
        f.write("| cpu_num | num_processors |\n")
        f.write("| node_num | num_exec_hosts |\n")
        f.write("| state | job_status_str (已映射) |\n")
        f.write("| submit_time | submit_time |\n")
        f.write("| start_time | start_time |\n")
        f.write("| end_time | end_time |\n")
        f.write("| duration | 从start_time和end_time计算 |\n")
        f.write("| queue | 从submit_time和start_time计算 |\n")
        
        f.write("\n### 状态映射关系\n\n")
        f.write("| 原始状态 | HeliosData状态 |\n")
        f.write("|---------|---------------|\n")
        for orig, mapped in state_map.items():
            f.write(f"| {orig} | {mapped} |\n")
        f.write("| PENDING | FAILED |\n")
        f.write("| RUNNING | FAILED |\n")
        
        # 过滤规则说明
        f.write("\n### 数据过滤规则\n\n")
        f.write("* 过滤了submit_time, start_time, end_time任一字段为空的记录\n")
        
    print(f"转换总结已保存到: {summary_path}")

# 生成转换总结
generate_conversion_summary(helios_df, df, output_dir) 