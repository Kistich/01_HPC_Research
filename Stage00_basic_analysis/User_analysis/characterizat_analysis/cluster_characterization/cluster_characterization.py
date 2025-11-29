import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# 设置可视化样式
sns.set_style("ticks")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "Bitstream Vera Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_context("paper", font_scale=1.6, rc={"lines.linewidth": 3, "lines.markersize": 10})

# 数据加载与预处理
def load_data(filepath):
    """加载作业数据并进行必要的预处理"""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['submit_time', 'start_time', 'end_time'])
    
    # 计算必要的派生字段
    df['month'] = df['submit_time'].dt.month
    df['hour'] = df['submit_time'].dt.hour
    df['day'] = df['submit_time'].dt.day
    df['gpu_time'] = df['gpu_num'] * df['duration']
    
    print(f"Loaded {len(df)} job records")
    return df

# 1. 昼夜趋势分析
def analyze_diurnal_trends(df, save=False, output_path=None):
    """分析集群使用的昼夜趋势
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), constrained_layout=True)
    
    # 1. 每小时平均GPU利用率分析
    # 我们需要先重建利用率序列
    # 创建时间范围（以小时为单位）
    min_date = df['submit_time'].min().date()
    max_date = df['end_time'].max().date()
    date_range = pd.date_range(start=min_date, end=max_date, freq='H')
    
    # 创建时间序列DataFrame
    hourly_df = pd.DataFrame(date_range, columns=['time'])
    hourly_df['running_gpu'] = 0
    hourly_df['total_gpu'] = 0  # 这需要根据实际情况设置
    
    # 估算集群总GPU数（取运行中最大GPU数的1.2倍）
    max_concurrent_gpus = 0
    for _, job in df.iterrows():
        concurrent_jobs = df[(df['start_time'] <= job['end_time']) & 
                            (df['end_time'] >= job['start_time'])]
        concurrent_gpus = concurrent_jobs['gpu_num'].sum()
        max_concurrent_gpus = max(max_concurrent_gpus, concurrent_gpus)
    
    total_gpus = int(max_concurrent_gpus * 1.2)
    hourly_df['total_gpu'] = total_gpus
    
    # 计算每个时间点的运行中GPU数
    for _, job in df[df['gpu_num'] > 0].iterrows():
        mask = (hourly_df['time'] >= job['start_time']) & (hourly_df['time'] <= job['end_time'])
        hourly_df.loc[mask, 'running_gpu'] += job['gpu_num']
    
    # 确保运行中GPU不超过总数
    hourly_df['running_gpu'] = hourly_df[['running_gpu', 'total_gpu']].min(axis=1)
    
    # 计算利用率
    hourly_df['utilization'] = hourly_df['running_gpu'] / hourly_df['total_gpu']
    
    # 按小时聚合
    hourly_df['hour'] = hourly_df['time'].dt.hour
    hour_util_mean = hourly_df.groupby('hour')['utilization'].mean() * 100
    
    # 绘制每小时平均利用率
    ax1.plot(range(24), hour_util_mean, 'o-', alpha=0.8)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Average Utilization (%)")
    ax1.set_xticks(range(0, 24, 4))
    ax1.set_xlim(-1, 24)
    ax1.grid(axis="y", linestyle=":")
    
    # 2. 每小时GPU作业提交数分析
    df_gpu = df[df['gpu_num'] > 0].copy()
    hourly_submissions = df_gpu.groupby('hour').size()
    
    ax2.plot(range(24), hourly_submissions, 'o-', alpha=0.8)
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Average Submitted\nGPU Job Number")
    ax2.set_xticks(range(0, 24, 4))
    ax2.set_xlim(-1, 24)
    ax2.grid(axis="y", linestyle=":")
    
    if save and output_path:
        plt.savefig(f"{output_path}/diurnal_trends.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/diurnal_trends.png", bbox_inches="tight", dpi=300)
    
    return fig

# 2. 月度趋势分析
def analyze_monthly_trends(df, save=False, output_path=None):
    """分析集群使用的月度趋势
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    # 找出数据中的所有月份
    months = sorted(df['month'].unique())
    
    # 准备单GPU和多GPU作业数据
    single_gpu_jobs = []
    multi_gpu_jobs = []
    utilization = []
    
    for month in months:
        month_df = df[df['month'] == month]
        
        # 计算单GPU和多GPU作业数
        gpu_jobs = month_df[month_df['gpu_num'] > 0]
        single_gpu_jobs.append(len(gpu_jobs[gpu_jobs['gpu_num'] == 1]))
        multi_gpu_jobs.append(len(gpu_jobs[gpu_jobs['gpu_num'] > 1]))
        
        # 计算月平均利用率（需要先计算每天的利用率）
        # 这里我们使用一个简化的方法：总GPU时间/（总时间×总GPU数）
        month_days = pd.DatetimeIndex(month_df['submit_time']).days_in_month[0]
        month_hours = month_days * 24
        total_gpu_time = gpu_jobs['gpu_time'].sum()
        
        # 估算总GPU数量（如果没有准确数据）
        if 'total_gpu' in locals():
            total_gpu = locals()['total_gpu']
        else:
            # 使用运行中最大GPU数的1.2倍作为估计
            max_concurrent_gpus = 0
            for _, job in gpu_jobs.iterrows():
                concurrent_jobs = gpu_jobs[(gpu_jobs['start_time'] <= job['end_time']) & 
                                          (gpu_jobs['end_time'] >= job['start_time'])]
                concurrent_gpus = concurrent_jobs['gpu_num'].sum()
                max_concurrent_gpus = max(max_concurrent_gpus, concurrent_gpus)
            total_gpu = int(max_concurrent_gpus * 1.2)
        
        # 计算利用率（简化方法）
        if month_hours * total_gpu > 0:
            month_util = total_gpu_time / (month_hours * total_gpu)
            utilization.append(month_util * 100)  # 转为百分比
        else:
            utilization.append(0)
    
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
    p3 = ax2.plot(x, utilization, 'r--o', label='GPU Utilization')
    
    # 设置图表标签和样式
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_ylabel('GPU Job Number')
    ax.set_xlabel('Month')
    ax.set_xticks(x)
    ax.set_xticklabels([month_names[m-1] for m in months])
    
    ax2.set_ylabel('GPU Utilization (%)')
    
    # 组合图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left')
    
    # 添加网格线
    ax.grid(axis='y', linestyle=':')
    
    if save and output_path:
        plt.savefig(f"{output_path}/monthly_trends.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/monthly_trends.png", bbox_inches="tight", dpi=300)
    
    return fig

# 3. GPU数量分布分析
def analyze_gpu_distribution(df, save=False, output_path=None):
    """分析GPU数量分布
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
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
    
    return fig

# 4. 作业处理状态分析
def analyze_job_status(df, save=False, output_path=None):
    """分析作业状态分布
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 计算不同状态的作业数量
    status_counts = df['state'].value_counts()
    
    # 绘制饼图
    ax.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%',
           startangle=90, shadow=True)
    ax.axis('equal')  # 保持圆形
    ax.set_title('Job Status Distribution')
    
    if save and output_path:
        plt.savefig(f"{output_path}/job_status.pdf", bbox_inches="tight", dpi=600)
        plt.savefig(f"{output_path}/job_status.png", bbox_inches="tight", dpi=300)
    
    return fig

# 5. 等待时间vs运行时间分析
def analyze_wait_vs_run_time(df, save=False, output_path=None):
    """分析作业等待时间与运行时间的关系
    
    Args:
        df: 包含作业数据的DataFrame
        save: 是否保存图表
        output_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 筛选有效数据
    valid_jobs = df[(df['queue'] > 0) & (df['duration'] > 0)]
    
    # 绘制散点图
    gpu_jobs = valid_jobs[valid_jobs['gpu_num'] > 0]
    cpu_jobs = valid_jobs[valid_jobs['gpu_num'] == 0]
    
    ax.scatter(cpu_jobs['duration'], cpu_jobs['queue'], 
              alpha=0.5, label='CPU Jobs', s=20)
    ax.scatter(gpu_jobs['duration'], gpu_jobs['queue'], 
              alpha=0.5, label='GPU Jobs', s=20)
    
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
    
    return fig

# 主函数
def analyze_cluster(data_path, output_path='./cluster_analysis'):
    """执行完整的集群分析
    
    Args:
        data_path: 数据文件路径
        output_path: 输出目录
    """
    import os
    os.makedirs(output_path, exist_ok=True)
    
    # 加载数据
    df = load_data(data_path)
    
    # 1. 昼夜趋势分析
    print("Analyzing diurnal trends...")
    diurnal_fig = analyze_diurnal_trends(df, save=True, output_path=output_path)
    
    # 2. 月度趋势分析
    print("Analyzing monthly trends...")
    monthly_fig = analyze_monthly_trends(df, save=True, output_path=output_path)
    
    # 3. GPU数量分布分析
    print("Analyzing GPU distribution...")
    gpu_dist_fig = analyze_gpu_distribution(df, save=True, output_path=output_path)
    
    # 4. 作业状态分析
    print("Analyzing job status...")
    status_fig = analyze_job_status(df, save=True, output_path=output_path)
    
    # 5. 等待时间vs运行时间分析
    print("Analyzing wait time vs run time...")
    wait_run_fig = analyze_wait_vs_run_time(df, save=True, output_path=output_path)
    
    print(f"Analysis complete. Results saved to {output_path}")
    
    return {
        "diurnal_fig": diurnal_fig,
        "monthly_fig": monthly_fig,
        "gpu_dist_fig": gpu_dist_fig,
        "status_fig": status_fig,
        "wait_run_fig": wait_run_fig
    }

# 示例调用
if __name__ == "__main__":
    # 设置路径
    data_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/convert_data/second_gen_helios_format.csv"
    output_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/cluster_characterization"
    
    # 执行分析
    analyze_cluster(data_path, output_path)
