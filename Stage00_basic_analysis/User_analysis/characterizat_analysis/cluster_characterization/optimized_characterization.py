import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
from tqdm import tqdm  # 添加进度条

# 设置可视化样式
sns.set_style("ticks")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
sns.set_context("paper", font_scale=1.6, rc={"lines.linewidth": 3, "lines.markersize": 10})

# 数据加载与预处理
def load_data(filepath):
    """加载作业数据并进行必要的预处理"""
    print(f"Loading data from: {filepath}")
    # 修复列类型警告
    df = pd.read_csv(filepath, parse_dates=['submit_time', 'start_time', 'end_time'], low_memory=False)
    
    # 计算必要的派生字段
    df['month'] = df['submit_time'].dt.month
    df['hour'] = df['submit_time'].dt.hour
    df['day'] = df['submit_time'].dt.day
    df['gpu_time'] = df['gpu_num'] * df['duration']
    
    print(f"Loaded {len(df)} job records")
    return df

# 1. 昼夜趋势分析 - 优化版
def analyze_diurnal_trends(df, save=False, output_path=None):
    """分析集群使用的昼夜趋势 - 基于全量数据"""
    print("Analyzing diurnal trends using full data...")
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), constrained_layout=True)
    
    # 为避免内存问题，使用高效的小时粒度聚合
    # 统计不同小时的GPU利用数据
    hourly_stats = {}
    
    # 遍历不同的日期进行统计
    dates = df['submit_time'].dt.date.unique()
    print(f"Processing {len(dates)} unique dates for diurnal analysis...")
    
    for date in tqdm(dates):
        for hour in range(24):
            key = hour
            if key not in hourly_stats:
                hourly_stats[key] = {'total_gpu': 0, 'active_samples': 0}
            
            # 查找当前日期+小时活跃的作业
            hour_start = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            hour_end = hour_start + pd.Timedelta(hours=1)
            
            # 找出这个时间窗口内活跃的GPU任务
            active_jobs = df[(df['start_time'] < hour_end) & 
                            (df['end_time'] > hour_start) & 
                            (df['gpu_num'] > 0)]
            
            if len(active_jobs) > 0:
                total_gpu = active_jobs['gpu_num'].sum()
                hourly_stats[key]['total_gpu'] += total_gpu
                hourly_stats[key]['active_samples'] += 1
    
    # 计算每小时平均GPU使用
    hours = range(24)
    avg_gpus = []
    
    # 估算集群总GPU数量
    gpu_counts = df[df['gpu_num'] > 0]['gpu_num'].value_counts()
    max_common_gpus = gpu_counts.index[0] if not gpu_counts.empty else 1
    
    # 查找并发峰值作为基数
    peak_concurrent = 0
    for hour, stats in hourly_stats.items():
        if stats['active_samples'] > 0:
            avg_gpu = stats['total_gpu'] / stats['active_samples']
            peak_concurrent = max(peak_concurrent, avg_gpu)
    
    # 假设集群总GPU数是峰值的1.2倍
    total_gpus = max(int(peak_concurrent * 1.2), max_common_gpus * 10)
    
    # 计算利用率
    for hour in hours:
        if hour in hourly_stats and hourly_stats[hour]['active_samples'] > 0:
            avg_gpu = hourly_stats[hour]['total_gpu'] / hourly_stats[hour]['active_samples']
            utilization = min(avg_gpu / total_gpus, 1.0) * 100
            avg_gpus.append(utilization)
        else:
            avg_gpus.append(0)
    
    # 绘制每小时平均利用率
    ax1.plot(hours, avg_gpus, 'o-', alpha=0.8)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Average Utilization (%)")
    ax1.set_xticks(range(0, 24, 4))
    ax1.set_xlim(-1, 24)
    ax1.grid(axis="y", linestyle=":")
    
    # 2. 每小时GPU作业提交数分析
    print("Calculating hourly job submissions...")
    df_gpu = df[df['gpu_num'] > 0].copy()
    hourly_submissions = df_gpu.groupby('hour').size()
    
    ax2.plot(range(24), hourly_submissions, 'o-', alpha=0.8)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Total Submitted\nGPU Jobs")
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xlim(-1, 24)
    ax2.grid(axis="y", linestyle=":")
    
    if save and output_path:
        plt.savefig(f"{output_path}/diurnal_trends.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/diurnal_trends.png", bbox_inches="tight", dpi=300)
    
    print("Diurnal trends analysis complete")
    return fig

# 2. 月度趋势分析 - 修正版
def analyze_monthly_trends(df, save=False, output_path=None):
    """分析集群使用的月度趋势 - 使用实际配置数据"""
    print("Analyzing monthly trends using full data...")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # 找出数据中的所有月份
    months = sorted(df['month'].unique())
    
    # 准备单GPU和多GPU作业数据
    single_gpu_jobs = []
    multi_gpu_jobs = []
    utilization = []
    
    # 设置真实的GPU总数（根据集群实际配置）
    # NVIDIA GPU: 65个A800节点(8卡/节点) + 15个A40节点(4卡/节点) = 580卡
    # 加上国产AI平台GPU，总数约为600+
    TOTAL_GPU = 600  # 根据集群实际情况设置
    
    print("Processing month by month...")
    for month in tqdm(months):
        month_df = df[df['month'] == month]
        
        # 计算单GPU和多GPU作业数
        gpu_jobs = month_df[month_df['gpu_num'] > 0]
        single_gpu = len(gpu_jobs[gpu_jobs['gpu_num'] == 1])
        multi_gpu = len(gpu_jobs[gpu_jobs['gpu_num'] > 1])
        
        single_gpu_jobs.append(single_gpu)
        multi_gpu_jobs.append(multi_gpu)
        
        # 计算每月GPU时间总和
        total_gpu_time = gpu_jobs['gpu_time'].sum()
        
        # 计算月实际小时数（考虑2023年9月后正式运营）
        if month < 9 and year == 2023:  # 假设数据包含年份信息
            # 9月前数据可能是测试阶段或来自其他集群
            month_hours = 24 * 30  # 使用标准月长度
        else:
            # 获取该月的实际天数
            year = month_df['submit_time'].dt.year.iloc[0] if not month_df.empty else 2023
            days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
            month_hours = 24 * days_in_month
        
        # 正确计算GPU利用率，不设置人为上限
        month_util = total_gpu_time / (month_hours * TOTAL_GPU * 3600)  # 转换为小时单位
        utilization.append(month_util * 100)  # 转为百分比
    
    # 绘制月度作业数量和利用率
    width = 0.35
    x = np.arange(len(months))
    
    # 堆叠柱状图：单GPU和多GPU作业
    p1 = ax.bar(x, multi_gpu_jobs, width, label='Multi-GPU Jobs', color='skyblue', 
               edgecolor='black', hatch='//')
    p2 = ax.bar(x, single_gpu_jobs, width, bottom=multi_gpu_jobs, 
               label='Single-GPU Jobs', color='skyblue', edgecolor='black')
    
    # 右Y轴显示利用率
    ax2 = ax.twinx()
    ax2.set_ylabel("GPU Utilization (%)")
    ax2.set_ylim(0, max(utilization) * 1.2)  # 动态设置上限，留出20%余量
    p3 = ax2.plot(x, utilization, 'r--o', label='GPU Utilization')
    
    # 设置图表标签和样式
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_ylabel('GPU Job Number')
    ax.set_xlabel('Month')
    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m-1] for m in months])
    
    # 组合图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # 添加网格线
    ax.grid(axis='y', linestyle=':')
    
    if save and output_path:
        plt.savefig(f"{output_path}/monthly_trends.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/monthly_trends.png", bbox_inches="tight", dpi=300)
    
    print("Monthly trends analysis complete")
    return fig

# 3. GPU数量分布分析 - 基于全量数据
def analyze_gpu_distribution(df, save=False, output_path=None):
    """分析GPU数量分布
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    print("Analyzing GPU distribution...")
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    
    # 只分析GPU作业
    gpu_jobs = df[df['gpu_num'] > 0].copy()
    
    # 计算每个GPU数量的作业数
    gpu_counts = gpu_jobs['gpu_num'].value_counts().sort_index()
    
    # 计算GPU时间
    gpu_jobs['gpu_time'] = gpu_jobs['gpu_num'] * gpu_jobs['duration']
    
    # 按GPU数量分组计算GPU时间
    gpu_time_by_count = gpu_jobs.groupby('gpu_num')['gpu_time'].sum()
    
    # 绘制作业数量分布
    ax1.bar(gpu_counts.index, gpu_counts.values)
    ax1.set_xlabel('GPU Count')
    ax1.set_ylabel('Number of Jobs')
    ax1.set_title('Job Count by GPU Number')
    ax1.set_yscale('log')  # 对数坐标更好地显示分布
    
    # 绘制GPU时间分布
    ax2.bar(gpu_time_by_count.index, gpu_time_by_count.values)
    ax2.set_xlabel('GPU Count')
    ax2.set_ylabel('GPU Time (seconds)')
    ax2.set_title('GPU Time by GPU Number')
    ax2.set_yscale('log')  # 对数坐标
    
    plt.tight_layout()
    
    if save and output_path:
        plt.savefig(f"{output_path}/gpu_distribution.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/gpu_distribution.png", bbox_inches="tight", dpi=300)
    
    print("GPU distribution analysis complete")
    return fig

# 4. 作业状态分析 - 基于全量数据
def analyze_job_status(df, save=False, output_path=None):
    """分析作业状态分布
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    print("Analyzing job status...")
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 分析作业状态分布
    status_counts = df['state'].value_counts()
    
    # 状态过多时，只显示前7个最常见状态
    if len(status_counts) > 7:
        top_states = status_counts.head(7)
        other_count = status_counts[7:].sum()
        status_counts = pd.concat([top_states, pd.Series([other_count], index=['Other'])])
    
    # 使用派生的颜色列表
    colors = plt.cm.tab10(np.linspace(0, 1, len(status_counts)))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        status_counts, 
        labels=status_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        wedgeprops={'edgecolor': 'w', 'linewidth': 1}
    )
    
    # 美化文本
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
    
    ax.set_title('Job Status Distribution')
    ax.axis('equal')  # 确保饼图是圆形
    
    if save and output_path:
        plt.savefig(f"{output_path}/job_status.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/job_status.png", bbox_inches="tight", dpi=300)
    
    print("Job status analysis complete")
    return fig

# 5. 等待时间vs运行时间分析 - 使用原始散点图方式
def analyze_wait_vs_run_time(df, save=False, output_path=None):
    """分析作业等待时间与运行时间的关系
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    print("Analyzing wait time vs run time...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 筛选有效数据
    valid_jobs = df[(df['queue'] > 0) & (df['duration'] > 0)]
    
    # 绘制散点图，完全保持原始分析方式
    gpu_jobs = valid_jobs[valid_jobs['gpu_num'] > 0]
    cpu_jobs = valid_jobs[valid_jobs['gpu_num'] == 0]
    
    # 使用较小的点大小和更低的透明度来处理大数据集
    ax.scatter(cpu_jobs['duration'], cpu_jobs['queue'], 
              alpha=0.3, label='CPU Jobs', s=10)
    ax.scatter(gpu_jobs['duration'], gpu_jobs['queue'], 
              alpha=0.3, label='GPU Jobs', s=10)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Run Time (seconds)')
    ax.set_ylabel('Wait Time (seconds)')
    ax.set_title('Wait Time vs Run Time')
    ax.grid(True, linestyle=':')
    ax.legend()
    
    if save and output_path:
        plt.savefig(f"{output_path}/wait_vs_run.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/wait_vs_run.png", bbox_inches="tight", dpi=300)
    
    print("Wait vs run time analysis complete")
    return fig

# 主程序入口点
def main():
    """主程序入口点，处理命令行参数"""
    import argparse
    import os
    import time
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='集群作业数据特征分析工具')
    parser.add_argument('--data_path', type=str, required=True,
                        help='作业数据CSV文件路径')
    parser.add_argument('--output_path', type=str, default='./cluster_analysis_output',
                        help='分析结果输出目录')
    parser.add_argument('--analyses', type=str, default='all',
                        help='要执行的分析，逗号分隔: diurnal,monthly,gpu,status,wait_run,all')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"错误: 数据文件不存在: {args.data_path}")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    print(f"开始分析，时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    df = load_data(args.data_path)
    
    # 确定要执行哪些分析
    analyses = args.analyses.lower().split(',')
    run_all = 'all' in analyses
    
    # 执行选定的分析
    results = {}
    
    # 1. 昼夜趋势分析
    if run_all or 'diurnal' in analyses:
        print("\n== 分析昼夜趋势 ==")
        results['diurnal_fig'] = analyze_diurnal_trends(df, save=True, output_path=args.output_path)
    
    # 2. 月度趋势分析
    if run_all or 'monthly' in analyses:
        print("\n== 分析月度趋势 ==")
        results['monthly_fig'] = analyze_monthly_trends(df, save=True, output_path=args.output_path)
    
    # 3. GPU数量分布分析
    if run_all or 'gpu' in analyses:
        print("\n== 分析GPU分布 ==")
        results['gpu_dist_fig'] = analyze_gpu_distribution(df, save=True, output_path=args.output_path)
    
    # 4. 作业状态分析
    if run_all or 'status' in analyses:
        print("\n== 分析作业状态 ==")
        results['status_fig'] = analyze_job_status(df, save=True, output_path=args.output_path)
    
    # 5. 等待时间vs运行时间分析
    if run_all or 'wait_run' in analyses:
        print("\n== 分析等待时间vs运行时间 ==")
        results['wait_run_fig'] = analyze_wait_vs_run_time(df, save=True, output_path=args.output_path)
    
    # 记录结束时间
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n分析完成！")
    print(f"总运行时间: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    print(f"结果保存在: {args.output_path}")
    
    return 0

# 程序入口
if __name__ == "__main__":
    import sys
    sys.exit(main()) 