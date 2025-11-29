#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化2024-04-14峰值日分析结果
生成图表展示用户行为、时间分布和资源利用情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from datetime import datetime

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')
sns.set_palette("husl")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PeakDayVisualizer:
    """峰值日可视化器"""

    def __init__(self):
        # 使用相对路径，基于脚本所在位置
        script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
        project_root = script_dir.parent  # 01_HPC_Research/

        self.output_dir = script_dir / "output" / "peak_day_detailed"
        self.data_path = project_root / "Stage01_data_filter_preprocess" / "full_processing_outputs" / "stage6_data_standardization" / "standardized_data.csv"
        self.peak_day_data = None
        
    def load_peak_day_data(self):
        """加载峰值日数据"""
        logger.info("加载峰值日数据...")
        
        # 分块读取数据
        chunk_size = 100000
        peak_day_chunks = []
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            chunk['submit_time'] = pd.to_datetime(chunk['submit_time'])
            chunk['submit_date'] = chunk['submit_time'].dt.date
            
            peak_day_chunk = chunk[chunk['submit_date'] == pd.to_datetime('2024-04-14').date()]
            
            if len(peak_day_chunk) > 0:
                peak_day_chunks.append(peak_day_chunk)
        
        self.peak_day_data = pd.concat(peak_day_chunks, ignore_index=True)
        
        # 计算持续时间
        if 'start_time' in self.peak_day_data.columns and 'end_time' in self.peak_day_data.columns:
            self.peak_day_data['start_time'] = pd.to_datetime(self.peak_day_data['start_time'])
            self.peak_day_data['end_time'] = pd.to_datetime(self.peak_day_data['end_time'])
            self.peak_day_data['duration'] = (self.peak_day_data['end_time'] - self.peak_day_data['start_time']).dt.total_seconds()
        
        logger.info(f"峰值日数据加载完成，共{len(self.peak_day_data)}条记录")
    
    def plot_user_behavior(self):
        """绘制用户行为分析图"""
        logger.info("生成用户行为分析图...")
        
        # 读取用户统计数据
        user_stats = pd.read_csv(self.output_dir / 'user_behavior_stats.csv')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 用户作业数量分布
        top_users = user_stats.head(10)
        bars1 = ax1.bar(range(len(top_users)), top_users['job_count'], color='skyblue', alpha=0.8)
        ax1.set_xlabel('用户排名')
        ax1.set_ylabel('作业数量')
        ax1.set_title('前10名用户作业数量分布')
        ax1.set_xticks(range(len(top_users)))
        ax1.set_xticklabels([f'User{i+1}' for i in range(len(top_users))], rotation=45)
        
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=9)
        
        # 2. 用户平均作业持续时间
        bars2 = ax2.bar(range(len(top_users)), top_users['avg_duration'], color='lightcoral', alpha=0.8)
        ax2.set_xlabel('用户排名')
        ax2.set_ylabel('平均持续时间 (秒)')
        ax2.set_title('前10名用户平均作业持续时间')
        ax2.set_xticks(range(len(top_users)))
        ax2.set_xticklabels([f'User{i+1}' for i in range(len(top_users))], rotation=45)
        
        # 3. 用户提交时间跨度
        bars3 = ax3.bar(range(len(top_users)), top_users['submit_span_hours'], color='lightgreen', alpha=0.8)
        ax3.set_xlabel('用户排名')
        ax3.set_ylabel('提交时间跨度 (小时)')
        ax3.set_title('前10名用户提交时间跨度')
        ax3.set_xticks(range(len(top_users)))
        ax3.set_xticklabels([f'User{i+1}' for i in range(len(top_users))], rotation=45)
        
        # 4. 用户资源使用情况
        ax4.scatter(top_users['avg_cpu'], top_users['avg_gpu'], 
                   s=top_users['job_count']/100, alpha=0.6, c=range(len(top_users)), cmap='viridis')
        ax4.set_xlabel('平均CPU数量')
        ax4.set_ylabel('平均GPU数量')
        ax4.set_title('用户资源使用模式 (气泡大小=作业数量)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'user_behavior_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"用户行为分析图已保存: {self.output_dir / 'user_behavior_analysis.png'}")
    
    def plot_temporal_patterns(self):
        """绘制时间模式分析图"""
        logger.info("生成时间模式分析图...")
        
        if self.peak_day_data is None:
            self.load_peak_day_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 小时级提交分布
        self.peak_day_data['submit_hour'] = self.peak_day_data['submit_time'].dt.hour
        hourly_counts = self.peak_day_data['submit_hour'].value_counts().sort_index()
        
        bars1 = ax1.bar(hourly_counts.index, hourly_counts.values, color='steelblue', alpha=0.8)
        ax1.set_xlabel('小时')
        ax1.set_ylabel('作业提交数量')
        ax1.set_title('24小时作业提交分布')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # 标记峰值
        peak_hour = hourly_counts.idxmax()
        peak_count = hourly_counts.max()
        ax1.annotate(f'峰值: {peak_hour}点\n{peak_count:,}个作业', 
                    xy=(peak_hour, peak_count), xytext=(peak_hour+2, peak_count*0.8),
                    arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')
        
        # 2. 分钟级高频提交时间点
        self.peak_day_data['submit_minute'] = self.peak_day_data['submit_time'].dt.floor('min')
        minute_counts = self.peak_day_data['submit_minute'].value_counts().sort_values(ascending=False)
        top_minutes = minute_counts.head(20)
        
        bars2 = ax2.bar(range(len(top_minutes)), top_minutes.values, color='orange', alpha=0.8)
        ax2.set_xlabel('时间点排名')
        ax2.set_ylabel('单分钟作业提交数量')
        ax2.set_title('前20个高频提交分钟')
        ax2.set_xticks(range(0, len(top_minutes), 2))
        
        # 3. 持续时间分布
        valid_duration = self.peak_day_data[self.peak_day_data['duration'] > 0]['duration']
        duration_bins = [0, 1, 2, 5, 10, 30, 60, 300, 1800, 3600, float('inf')]
        duration_labels = ['<1s', '1-2s', '2-5s', '5-10s', '10-30s', '30s-1min', '1-5min', '5-30min', '30min-1h', '>1h']
        
        duration_categories = pd.cut(valid_duration, bins=duration_bins, labels=duration_labels)
        duration_dist = duration_categories.value_counts()
        
        bars3 = ax3.bar(range(len(duration_dist)), duration_dist.values, color='green', alpha=0.8)
        ax3.set_xlabel('持续时间区间')
        ax3.set_ylabel('作业数量')
        ax3.set_title('作业持续时间分布')
        ax3.set_xticks(range(len(duration_dist)))
        ax3.set_xticklabels(duration_dist.index, rotation=45)
        
        # 添加百分比标签
        total_jobs = duration_dist.sum()
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            percentage = height / total_jobs * 100
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{percentage:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # 4. 资源请求模式
        resource_patterns = self.peak_day_data.groupby(['num_processors', 'gpu_num']).size().sort_values(ascending=False)
        top_patterns = resource_patterns.head(10)
        
        bars4 = ax4.bar(range(len(top_patterns)), top_patterns.values, color='purple', alpha=0.8)
        ax4.set_xlabel('资源配置排名')
        ax4.set_ylabel('作业数量')
        ax4.set_title('前10种资源请求模式')
        ax4.set_xticks(range(len(top_patterns)))
        
        # 添加资源配置标签
        labels = [f'CPU:{cpu}\nGPU:{gpu}' for (cpu, gpu) in top_patterns.index]
        ax4.set_xticklabels(labels, rotation=45, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"时间模式分析图已保存: {self.output_dir / 'temporal_patterns_analysis.png'}")
    
    def plot_efficiency_analysis(self):
        """绘制效率分析图"""
        logger.info("生成效率分析图...")
        
        if self.peak_day_data is None:
            self.load_peak_day_data()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 作业有效性分析
        effective_jobs = len(self.peak_day_data[self.peak_day_data['duration'] > 10])
        ineffective_jobs = len(self.peak_day_data[self.peak_day_data['duration'] <= 10])
        
        labels = ['有效作业\n(>10秒)', '无效作业\n(≤10秒)']
        sizes = [effective_jobs, ineffective_jobs]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'作业有效性分析\n总计: {len(self.peak_day_data):,}个作业')
        
        # 2. 用户作业数量分布
        user_job_counts = self.peak_day_data['user_id'].value_counts()
        
        # 按作业数量分组
        bins = [0, 10, 100, 1000, 10000, float('inf')]
        labels_bins = ['1-10', '11-100', '101-1000', '1001-10000', '>10000']
        user_categories = pd.cut(user_job_counts, bins=bins, labels=labels_bins)
        user_dist = user_categories.value_counts()
        
        bars2 = ax2.bar(range(len(user_dist)), user_dist.values, color='skyblue', alpha=0.8)
        ax2.set_xlabel('单用户作业数量区间')
        ax2.set_ylabel('用户数量')
        ax2.set_title('用户作业数量分布')
        ax2.set_xticks(range(len(user_dist)))
        ax2.set_xticklabels(user_dist.index)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom')
        
        # 3. 异常用户分析
        top_user = user_job_counts.index[0]
        top_user_data = self.peak_day_data[self.peak_day_data['user_id'] == top_user]
        
        # 该用户的小时级提交分布
        top_user_hourly = top_user_data['submit_time'].dt.hour.value_counts().sort_index()
        
        bars3 = ax3.bar(top_user_hourly.index, top_user_hourly.values, color='red', alpha=0.7)
        ax3.set_xlabel('小时')
        ax3.set_ylabel('作业提交数量')
        ax3.set_title(f'异常用户(ID:{top_user})的24小时提交分布\n总计: {len(top_user_data):,}个作业')
        ax3.set_xticks(range(0, 24, 2))
        ax3.grid(True, alpha=0.3)
        
        # 4. 资源浪费分析
        # 计算CPU小时浪费
        total_cpu_hours = (self.peak_day_data['duration'] * self.peak_day_data['num_processors']).sum() / 3600
        ineffective_data = self.peak_day_data[self.peak_day_data['duration'] <= 10]
        wasted_cpu_hours = (ineffective_data['duration'] * ineffective_data['num_processors']).sum() / 3600
        
        effective_cpu_hours = total_cpu_hours - wasted_cpu_hours
        
        labels_waste = ['有效CPU小时', '浪费CPU小时']
        sizes_waste = [effective_cpu_hours, wasted_cpu_hours]
        colors_waste = ['lightblue', 'orange']
        
        wedges, texts, autotexts = ax4.pie(sizes_waste, labels=labels_waste, colors=colors_waste, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'CPU资源利用效率\n总计: {total_cpu_hours:.1f}小时')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"效率分析图已保存: {self.output_dir / 'efficiency_analysis.png'}")
    
    def generate_summary_report(self):
        """生成总结报告"""
        logger.info("生成总结报告...")
        
        if self.peak_day_data is None:
            self.load_peak_day_data()
        
        # 计算关键指标
        total_jobs = len(self.peak_day_data)
        unique_users = self.peak_day_data['user_id'].nunique()
        
        # 用户行为分析
        user_job_counts = self.peak_day_data['user_id'].value_counts()
        top_user_jobs = user_job_counts.iloc[0]
        top_user_id = user_job_counts.index[0]
        
        # 时间分析
        hourly_counts = self.peak_day_data['submit_time'].dt.hour.value_counts()
        peak_hour = hourly_counts.idxmax()
        peak_hour_jobs = hourly_counts.max()
        
        # 持续时间分析
        short_jobs = len(self.peak_day_data[self.peak_day_data['duration'] <= 10])
        short_job_percentage = short_jobs / total_jobs * 100
        
        # 资源分析
        gpu_jobs = len(self.peak_day_data[self.peak_day_data['gpu_num'] > 0])
        cpu_jobs = len(self.peak_day_data[self.peak_day_data['gpu_num'] == 0])
        
        report = f"""
# 2024-04-14 峰值日作业提交分析总结报告

## 关键发现

### 1. 基本统计
- **总作业数**: {total_jobs:,}
- **涉及用户数**: {unique_users}
- **平均每用户作业数**: {total_jobs/unique_users:.1f}

### 2. 异常用户行为
- **最高频用户**: 用户ID {top_user_id}
- **该用户作业数**: {top_user_jobs:,} ({top_user_jobs/total_jobs*100:.1f}%)
- **异常特征**: 单用户提交了超过6万个作业，远超正常水平

### 3. 时间集中性
- **峰值小时**: {peak_hour}点
- **峰值小时作业数**: {peak_hour_jobs:,}
- **时间集中度**: 存在多个单分钟提交超过1万个作业的异常时间点

### 4. 作业质量问题
- **超短作业数**: {short_jobs:,}
- **超短作业占比**: {short_job_percentage:.1f}%
- **质量评估**: 近30%的作业持续时间≤10秒，可能为测试或错误提交

### 5. 资源使用模式
- **CPU作业**: {cpu_jobs:,} ({cpu_jobs/total_jobs*100:.1f}%)
- **GPU作业**: {gpu_jobs:,} ({gpu_jobs/total_jobs*100:.1f}%)
- **主要配置**: 99.7%的作业使用1个CPU，0个GPU

## 数据质量评估

### 问题识别
1. **用户行为异常**: 单个用户提交了{top_user_jobs:,}个作业，占总数的{top_user_jobs/total_jobs*100:.1f}%
2. **时间集中异常**: 存在明显的批量提交模式
3. **持续时间异常**: {short_job_percentage:.1f}%的作业持续时间过短
4. **重复性高**: 大量相同配置的重复作业

### 数据清洗建议
1. **过滤超短作业**: 移除持续时间≤10秒的{short_jobs:,}个作业
2. **限制用户频率**: 对单用户单日作业数设置合理上限（如1000个）
3. **去重处理**: 识别并处理相同用户的相同配置重复提交
4. **时间窗口过滤**: 过滤明显的批量提交时间窗口

### 预期效果
- **可移除作业数**: 约{short_jobs + top_user_jobs:,}个（{(short_jobs + top_user_jobs)/total_jobs*100:.1f}%）
- **清洗后数据量**: 约{total_jobs - short_jobs - top_user_jobs:,}个作业
- **数据质量提升**: 显著改善作业持续时间分布和用户行为合理性

## 结论

2024年4月14日的异常峰值主要由以下因素造成：
1. 单个用户的大量重复提交（可能是自动化脚本或测试）
2. 大量超短时长作业（可能是失败的快速重试）
3. 时间集中的批量提交模式

建议对此类数据进行清洗处理，以获得更真实的集群使用模式。

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        with open(self.output_dir / 'peak_day_summary_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"总结报告已保存: {self.output_dir / 'peak_day_summary_report.md'}")
        
        return report
    
    def run_all_visualizations(self):
        """运行所有可视化"""
        logger.info("开始生成所有可视化图表...")
        
        self.plot_user_behavior()
        self.plot_temporal_patterns()
        self.plot_efficiency_analysis()
        summary = self.generate_summary_report()
        
        logger.info(f"所有可视化完成，结果保存在: {self.output_dir}")
        
        return summary


def main():
    """主函数"""
    visualizer = PeakDayVisualizer()
    summary = visualizer.run_all_visualizations()
    
    print("\n" + "="*50)
    print("峰值日分析完成！")
    print("="*50)
    print(summary)


if __name__ == "__main__":
    main()
