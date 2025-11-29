import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

class TraceAnalyzer:
    def __init__(self, data_path):
        """初始化分析器
        Args:
            data_path: 数据文件路径
        """
        print(f"Loading data from: {data_path}")
        self.df = pd.read_csv(data_path, parse_dates=['submit_time', 'start_time', 'end_time'])
        print(f"Loaded {len(self.df)} job records")
        
        # 数据基本属性验证
        expected_fields = ['job_id', 'user', 'gpu_num', 'cpu_num', 'node_num', 
                           'state', 'submit_time', 'start_time', 'end_time', 
                           'duration', 'queue']
        
        missing_fields = [field for field in expected_fields if field not in self.df.columns]
        if missing_fields:
            print(f"Warning: Missing expected fields: {missing_fields}")
        
        # 基本数据统计
        self.total_jobs = len(self.df)
        self.gpu_jobs = len(self.df[self.df['gpu_num'] > 0])
        self.cpu_only_jobs = len(self.df[self.df['gpu_num'] == 0])
        
        print(f"Total jobs: {self.total_jobs}")
        print(f"GPU jobs: {self.gpu_jobs} ({self.gpu_jobs/self.total_jobs*100:.2f}%)")
        print(f"CPU-only jobs: {self.cpu_only_jobs} ({self.cpu_only_jobs/self.total_jobs*100:.2f}%)")
    
    def analyze_resource_usage(self, start_date=None, end_date=None, interval='1h', total_gpus=None):
        """分析资源使用时间序列
        Args:
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔 (例如 '1min', '1h')
            total_gpus: 总GPU数量 (若为None，将使用观察到的最大值)
        Returns:
            资源使用时间序列DataFrame
        """
        # 设置分析时间范围
        if start_date is None:
            start_date = self.df['submit_time'].min()
        if end_date is None:
            end_date = self.df['end_time'].max()
            
        print(f"Analyzing resource usage from {start_date} to {end_date}")
        
        # 创建时间序列DataFrame
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        sequence_df = pd.DataFrame(date_range, columns=['time'])
        sequence_df[['running_jobs', 'running_gpujob_num', 'running_gpu_num', 
                    'running_gpu_multi', 'running_gpu_single',
                    'running_nodes']] = 0
        
        # 逐作业分析资源使用
        for _, job in self.df.iterrows():
            # 计算作业运行期间各个时间点
            mask = (sequence_df['time'] >= job['start_time']) & (sequence_df['time'] <= job['end_time'])
            
            # 更新作业计数
            sequence_df.loc[mask, 'running_jobs'] += 1
            sequence_df.loc[mask, 'running_nodes'] += job['node_num']
            
            # GPU相关计数
            if job['gpu_num'] > 0:
                sequence_df.loc[mask, 'running_gpujob_num'] += 1
                sequence_df.loc[mask, 'running_gpu_num'] += job['gpu_num']
                
                # 区分单GPU和多GPU作业
                if job['gpu_num'] > 1:
                    sequence_df.loc[mask, 'running_gpu_multi'] += job['gpu_num']
                else:
                    sequence_df.loc[mask, 'running_gpu_single'] += job['gpu_num']
        
        # 设置总GPU数
        if total_gpus is None:
            # 如果未提供总GPU数，使用观察到的最大GPU数作为估计值
            observed_max = sequence_df['running_gpu_num'].max()
            total_gpus = int(observed_max * 1.2)  # 假设最大利用率约为80-90%
            print(f"Estimated total GPUs: {total_gpus}")
        
        sequence_df['total_gpu_num'] = total_gpus
        
        # 计算利用率
        sequence_df['gpu_utilization'] = (sequence_df['running_gpu_num'] / 
                                        sequence_df['total_gpu_num']).round(3)
        sequence_df['gpu_utilization_multi'] = (sequence_df['running_gpu_multi'] / 
                                             sequence_df['total_gpu_num']).round(3)
        sequence_df['gpu_utilization_single'] = (sequence_df['running_gpu_single'] / 
                                              sequence_df['total_gpu_num']).round(3)
        
        sequence_df.set_index('time', inplace=True)
        return sequence_df
    
    def analyze_throughput(self, interval='1h'):
        """分析作业吞吐量
        Args:
            interval: 时间间隔 (例如 '10min', '1h')
        Returns:
            吞吐量时间序列DataFrame
        """
        # 设置时间范围
        start_date = self.df['submit_time'].min()
        end_date = self.df['end_time'].max()
        
        print(f"Analyzing throughput from {start_date} to {end_date} (interval: {interval})")
        
        # 创建时间序列
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        throughput_df = pd.DataFrame(date_range, columns=['time'])
        
        # 初始化计数列
        throughput_columns = ['submit_job_all', 'start_job_all', 'end_job_all',
                             'submit_gpu_job', 'start_gpu_job', 'end_gpu_job',
                             'submit_gpu_num', 'start_gpu_num', 'end_gpu_num']
        throughput_df[throughput_columns] = 0
        
        # 按时间间隔统计
        for i in range(len(throughput_df)):
            current_time = throughput_df.loc[i, 'time']
            next_time = current_time + pd.Timedelta(interval)
            
            # 所有作业
            throughput_df.loc[i, 'submit_job_all'] = len(
                self.df[(self.df['submit_time'] >= current_time) & 
                        (self.df['submit_time'] < next_time)])
            
            throughput_df.loc[i, 'start_job_all'] = len(
                self.df[(self.df['start_time'] >= current_time) & 
                        (self.df['start_time'] < next_time)])
            
            throughput_df.loc[i, 'end_job_all'] = len(
                self.df[(self.df['end_time'] >= current_time) & 
                        (self.df['end_time'] < next_time)])
            
            # GPU作业
            gpu_jobs = self.df[self.df['gpu_num'] > 0]
            
            throughput_df.loc[i, 'submit_gpu_job'] = len(
                gpu_jobs[(gpu_jobs['submit_time'] >= current_time) & 
                         (gpu_jobs['submit_time'] < next_time)])
            
            throughput_df.loc[i, 'start_gpu_job'] = len(
                gpu_jobs[(gpu_jobs['start_time'] >= current_time) & 
                         (gpu_jobs['start_time'] < next_time)])
            
            throughput_df.loc[i, 'end_gpu_job'] = len(
                gpu_jobs[(gpu_jobs['end_time'] >= current_time) & 
                         (gpu_jobs['end_time'] < next_time)])
            
            # GPU数量
            throughput_df.loc[i, 'submit_gpu_num'] = gpu_jobs[
                (gpu_jobs['submit_time'] >= current_time) & 
                (gpu_jobs['submit_time'] < next_time)]['gpu_num'].sum()
            
            throughput_df.loc[i, 'start_gpu_num'] = gpu_jobs[
                (gpu_jobs['start_time'] >= current_time) & 
                (gpu_jobs['start_time'] < next_time)]['gpu_num'].sum()
            
            throughput_df.loc[i, 'end_gpu_num'] = gpu_jobs[
                (gpu_jobs['end_time'] >= current_time) & 
                (gpu_jobs['end_time'] < next_time)]['gpu_num'].sum()
        
        throughput_df.set_index('time', inplace=True)
        return throughput_df
    
    def analyze_users(self):
        """分析用户行为
        Returns:
            用户统计数据DataFrame
        """
        print("Analyzing user behavior...")
        
        users = self.df['user'].unique()
        print(f"Total users: {len(users)}")
        
        user_stats = pd.DataFrame(index=users)
        
        # 逐用户分析
        for user in users:
            user_jobs = self.df[self.df['user'] == user]
            user_gpu_jobs = user_jobs[user_jobs['gpu_num'] > 0]
            user_cpu_jobs = user_jobs[user_jobs['gpu_num'] == 0]
            
            # 基本作业统计
            user_stats.loc[user, 'job_num'] = len(user_jobs)
            user_stats.loc[user, 'gpu_job_num'] = len(user_gpu_jobs)
            user_stats.loc[user, 'cpu_job_num'] = len(user_cpu_jobs)
            
            # 资源使用
            user_stats.loc[user, 'avg_duration'] = user_jobs['duration'].mean()
            user_stats.loc[user, 'max_duration'] = user_jobs['duration'].max()
            user_stats.loc[user, 'avg_gpu_duration'] = user_gpu_jobs['duration'].mean() if len(user_gpu_jobs) > 0 else 0
            user_stats.loc[user, 'avg_cpu_duration'] = user_cpu_jobs['duration'].mean() if len(user_cpu_jobs) > 0 else 0
            
            user_stats.loc[user, 'avg_queue'] = user_jobs['queue'].mean()
            user_stats.loc[user, 'max_queue'] = user_jobs['queue'].max()
            user_stats.loc[user, 'avg_gpu_queue'] = user_gpu_jobs['queue'].mean() if len(user_gpu_jobs) > 0 else 0
            user_stats.loc[user, 'avg_cpu_queue'] = user_cpu_jobs['queue'].mean() if len(user_cpu_jobs) > 0 else 0
            
            # GPU/CPU资源计算
            user_stats.loc[user, 'total_gpu_time'] = (user_gpu_jobs['gpu_num'] * user_gpu_jobs['duration']).sum()
            user_stats.loc[user, 'total_cpu_time'] = (user_jobs['cpu_num'] * user_jobs['duration']).sum()
            user_stats.loc[user, 'total_node_time'] = (user_jobs['node_num'] * user_jobs['duration']).sum()
            
            # 作业完成率分析
            if len(user_jobs) > 0:
                for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
                    user_stats.loc[user, f'{state.lower()}_percent'] = len(user_jobs[user_jobs['state'] == state]) / len(user_jobs)
            
            if len(user_gpu_jobs) > 0:
                for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
                    user_stats.loc[user, f'{state.lower()}_gpu_percent'] = len(user_gpu_jobs[user_gpu_jobs['state'] == state]) / len(user_gpu_jobs)
            
            if len(user_cpu_jobs) > 0:
                for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
                    user_stats.loc[user, f'{state.lower()}_cpu_percent'] = len(user_cpu_jobs[user_cpu_jobs['state'] == state]) / len(user_cpu_jobs)
        
        # 按作业数量排序
        user_stats.sort_values(by='job_num', ascending=False, inplace=True)
        
        # 填充NaN值
        user_stats.fillna(0, inplace=True)
        
        return user_stats

    def plot_resource_usage(self, sequence_df=None, save_path=None):
        """绘制资源使用时间序列图
        Args:
            sequence_df: 资源使用时间序列数据 (如果为None则调用analyze_resource_usage)
            save_path: 保存路径
        """
        if sequence_df is None:
            sequence_df = self.analyze_resource_usage()
        
        print("Plotting resource usage...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 绘制GPU数量
        ax1.plot(sequence_df.index, sequence_df['running_gpu_num'], label='Running GPUs')
        ax1.plot(sequence_df.index, sequence_df['running_gpu_multi'], label='Multi-GPU Jobs')
        ax1.plot(sequence_df.index, sequence_df['running_gpu_single'], label='Single-GPU Jobs')
        ax1.plot(sequence_df.index, sequence_df['total_gpu_num'], 'k--', label='Total GPUs')
        
        ax1.set_ylabel('GPU Count')
        ax1.set_title('GPU Usage Over Time')
        ax1.legend()
        ax1.grid(True, linestyle=':')
        
        # 绘制GPU利用率
        ax2.plot(sequence_df.index, sequence_df['gpu_utilization'] * 100, label='Total Utilization')
        ax2.plot(sequence_df.index, sequence_df['gpu_utilization_multi'] * 100, label='Multi-GPU Jobs')
        ax2.plot(sequence_df.index, sequence_df['gpu_utilization_single'] * 100, label='Single-GPU Jobs')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('GPU Utilization (%)')
        ax2.set_title('GPU Utilization Over Time')
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved resource usage plot to {save_path}")
        
        return fig
    
    def plot_throughput(self, throughput_df=None, save_path=None):
        """绘制作业吞吐量图
        Args:
            throughput_df: 吞吐量数据 (如果为None则调用analyze_throughput)
            save_path: 保存路径
        """
        if throughput_df is None:
            throughput_df = self.analyze_throughput()
        
        print("Plotting job throughput...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 绘制作业数量
        ax1.plot(throughput_df.index, throughput_df['submit_job_all'], label='Submitted Jobs')
        ax1.plot(throughput_df.index, throughput_df['start_job_all'], label='Started Jobs')
        ax1.plot(throughput_df.index, throughput_df['end_job_all'], label='Completed Jobs')
        
        ax1.set_ylabel('Job Count')
        ax1.set_title('Job Throughput Over Time')
        ax1.legend()
        ax1.grid(True, linestyle=':')
        
        # 绘制GPU作业数量
        ax2.plot(throughput_df.index, throughput_df['submit_gpu_job'], label='Submitted GPU Jobs')
        ax2.plot(throughput_df.index, throughput_df['start_gpu_job'], label='Started GPU Jobs')
        ax2.plot(throughput_df.index, throughput_df['end_gpu_job'], label='Completed GPU Jobs')
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('GPU Job Count')
        ax2.set_title('GPU Job Throughput Over Time')
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved throughput plot to {save_path}")
        
        return fig
    
    def plot_user_stats(self, user_stats=None, top_n=10, save_path=None):
        """绘制用户统计数据图
        Args:
            user_stats: 用户统计数据 (如果为None则调用analyze_users)
            top_n: 显示前N名用户
            save_path: 保存路径
        """
        if user_stats is None:
            user_stats = self.analyze_users()
        
        print(f"Plotting top {top_n} user statistics...")
        
        # 选择前N名用户
        top_users = user_stats.head(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 作业数量分布
        top_users[['job_num', 'gpu_job_num', 'cpu_job_num']].plot(
            kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title(f'Job Count by Top {top_n} Users')
        axes[0, 0].set_ylabel('Job Count')
        axes[0, 0].set_xlabel('User')
        axes[0, 0].grid(axis='y', linestyle=':')
        
        # 2. 平均运行时间
        top_users[['avg_duration', 'avg_gpu_duration', 'avg_cpu_duration']].plot(
            kind='bar', ax=axes[0, 1], width=0.8)
        axes[0, 1].set_title('Average Job Duration')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].set_xlabel('User')
        axes[0, 1].grid(axis='y', linestyle=':')
        
        # 3. 资源使用
        resource_data = top_users[['total_gpu_time', 'total_node_time']]
        resource_data.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('Resource Usage')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_xlabel('User')
        axes[1, 0].grid(axis='y', linestyle=':')
        
        # 4. 完成率
        completion_data = top_users[['completed_percent', 'cancelled_percent', 'failed_percent']]
        completion_data.plot(kind='bar', stacked=True, ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title('Job Completion Status')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_xlabel('User')
        axes[1, 1].grid(axis='y', linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved user statistics plot to {save_path}")
        
        return fig
    
    def analyze_all(self, output_dir='./trace_analysis'):
        """执行所有分析并保存结果
        Args:
            output_dir: 输出目录
        """
        print(f"Performing complete trace analysis, results will be saved to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 资源使用分析
        sequence_df = self.analyze_resource_usage()
        sequence_df.to_csv(f"{output_dir}/resource_usage.csv")
        self.plot_resource_usage(sequence_df, save_path=f"{output_dir}/resource_usage.png")
        
        # 2. 吞吐量分析
        throughput_df = self.analyze_throughput()
        throughput_df.to_csv(f"{output_dir}/throughput.csv")
        self.plot_throughput(throughput_df, save_path=f"{output_dir}/throughput.png")
        
        # 3. 用户行为分析
        user_stats = self.analyze_users()
        user_stats.to_csv(f"{output_dir}/user_stats.csv")
        self.plot_user_stats(user_stats, save_path=f"{output_dir}/user_stats.png")
        
        # 4. 基本统计
        stats = {
            'total_jobs': self.total_jobs,
            'gpu_jobs': self.gpu_jobs,
            'cpu_only_jobs': self.cpu_only_jobs,
            'gpu_job_percentage': self.gpu_jobs / self.total_jobs * 100,
            'total_users': len(user_stats),
            'avg_job_duration': self.df['duration'].mean(),
            'avg_gpu_job_duration': self.df[self.df['gpu_num'] > 0]['duration'].mean(),
            'avg_queue_time': self.df['queue'].mean(),
            'max_gpu_per_job': self.df['gpu_num'].max(),
            'avg_gpu_per_job': self.df[self.df['gpu_num'] > 0]['gpu_num'].mean(),
        }
        
        pd.Series(stats).to_csv(f"{output_dir}/basic_stats.csv")
        
        print(f"Analysis complete. Results saved to {output_dir}")
        return stats

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出路径
    data_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/User_behavior_analysis/convert_data/all_data/second_gen_helios_format.csv"
    output_dir = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/trace_characterization/all_data"
    
    # 创建分析器
    analyzer = TraceAnalyzer(data_path)
    
    # 执行完整分析
    analyzer.analyze_all(output_dir=output_dir)
    
    # 或者单独执行特定分析
    # sequence_df = analyzer.analyze_resource_usage(interval='1h')
    # analyzer.plot_resource_usage(sequence_df, save_path=f"{output_dir}/hourly_resource_usage.png")
