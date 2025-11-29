import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# 设置matplotlib样式
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
sns.set_context("paper", font_scale=1.6, rc={"lines.linewidth": 2, "lines.markersize": 10})

class TraceAnalyzer:
    """作业轨迹分析器 - 优化版"""
    
    def __init__(self, data_path):
        """初始化分析器
        
        Args:
            data_path: 作业数据文件路径
        """
        self.data_path = data_path
        print(f"Loading data from {data_path}...")
        
        # 加载数据
        start_time = time.time()
        self.df = pd.read_csv(data_path, parse_dates=['submit_time', 'start_time', 'end_time'], low_memory=False)
        
        # 计算必要的派生字段
        if 'duration' not in self.df.columns:
            self.df['duration'] = (self.df['end_time'] - self.df['start_time']).dt.total_seconds()
        
        if 'queue' not in self.df.columns:
            self.df['queue'] = (self.df['start_time'] - self.df['submit_time']).dt.total_seconds()
        
        # 确保gpu_num为数值型
        self.df['gpu_num'] = pd.to_numeric(self.df['gpu_num'], errors='coerce').fillna(0).astype(int)
        
        # 预计算的属性
        self.total_jobs = len(self.df)
        self.gpu_jobs = len(self.df[self.df['gpu_num'] > 0])
        self.cpu_only_jobs = self.total_jobs - self.gpu_jobs
        
        # 预计算基本统计
        self.basic_stats = self._calculate_basic_stats()
        
        print(f"Loaded {self.total_jobs} jobs ({self.gpu_jobs} GPU jobs) in {time.time() - start_time:.2f} seconds")
    
    def _calculate_basic_stats(self):
        """计算基本统计信息"""
        stats = {
            'total_jobs': self.total_jobs,
            'gpu_jobs': self.gpu_jobs,
            'cpu_jobs': self.cpu_only_jobs,
            'gpu_job_ratio': self.gpu_jobs / self.total_jobs if self.total_jobs > 0 else 0,
            'total_gpus_used': self.df['gpu_num'].sum(),
            'avg_gpus_per_job': self.df['gpu_num'].mean(),
            'median_gpus_per_job': self.df[self.df['gpu_num'] > 0]['gpu_num'].median(),
            'max_gpus_per_job': self.df['gpu_num'].max(),
            'total_nodes_used': self.df['node_num'].sum() if 'node_num' in self.df.columns else 0,
            'avg_nodes_per_job': self.df['node_num'].mean() if 'node_num' in self.df.columns else 0,
            'avg_queue_time': self.df['queue'].mean(),
            'median_queue_time': self.df['queue'].median(),
            'avg_duration': self.df['duration'].mean(),
            'median_duration': self.df['duration'].median(),
            'time_range_days': (self.df['end_time'].max() - self.df['submit_time'].min()).days,
        }
        return pd.Series(stats)
    
    @staticmethod
    def _process_chunk(chunk_dates, df, gpu_jobs):
        """处理一组时间点的资源使用情况
        
        Args:
            chunk_dates: 时间点列表
            df: 完整作业数据集
            gpu_jobs: 仅GPU作业的数据集
            
        Returns:
            结果列表，每个元素为(时间点, 结果字典)
        """
        results = []
        for t in chunk_dates:
            # 查找当前运行的作业
            running_jobs = df[(df['start_time'] <= t) & (df['end_time'] >= t)]
            running_gpus = running_jobs[running_jobs['gpu_num'] > 0]
            
            # 计算统计值
            single_gpu = running_gpus[running_gpus['gpu_num'] == 1]
            multi_gpu = running_gpus[running_gpus['gpu_num'] > 1]
            
            # 收集当前时间点的结果
            result = {
                'running_jobs': len(running_jobs),
                'running_gpujob_num': len(running_gpus),
                'running_gpu_num': running_gpus['gpu_num'].sum(),
                'running_gpu_single': single_gpu['gpu_num'].sum(),
                'running_gpu_multi': multi_gpu['gpu_num'].sum(),
                'running_nodes': running_jobs['node_num'].sum() if 'node_num' in running_jobs.columns else 0
            }
            
            results.append((t, result))
        return results
    
    def analyze_resource_usage(self, start_date=None, end_date=None, interval='1h', total_gpus=None, max_workers=32):
        """分析资源使用时间序列 - 并行优化版"""
        # 设置分析时间范围
        if start_date is None:
            start_date = self.df['submit_time'].min()
        if end_date is None:
            end_date = self.df['end_time'].max()
        
        print(f"Analyzing resource usage from {start_date} to {end_date}")
        print(f"Using time interval: {interval}")
        
        # 创建时间序列
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        print(f"Created {len(date_range)} time points for analysis")
        
        # 预先筛选所有GPU作业
        gpu_jobs = self.df[self.df['gpu_num'] > 0]
        
        # 分割时间点为多个块
        print(f"Using {max_workers} CPU cores for parallel processing")
        time_chunks = np.array_split(date_range, max_workers)
        
        # 并行处理
        all_results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk in time_chunks:
                futures.append(executor.submit(
                    self._process_chunk,
                    chunk, 
                    self.df, 
                    gpu_jobs
                ))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                all_results.extend(future.result())
        
        # 按时间排序并转换为DataFrame
        all_results.sort(key=lambda x: x[0])
        result_df = pd.DataFrame([r[1] for r in all_results], index=[r[0] for r in all_results])
        result_df.index.name = 'time'
        
        # 设置总GPU数
        if total_gpus is None:
            observed_max = result_df['running_gpu_num'].max()
            total_gpus = max(int(observed_max * 1.2), 1)  # 假设最大利用率约为80-90%
            print(f"Estimated total GPUs: {total_gpus}")
        
        result_df['total_gpu_num'] = total_gpus
        
        # 计算利用率
        result_df['gpu_utilization'] = (result_df['running_gpu_num'] / 
                                        result_df['total_gpu_num']).round(3)
        result_df['gpu_utilization_multi'] = (result_df['running_gpu_multi'] / 
                                             result_df['total_gpu_num']).round(3)
        result_df['gpu_utilization_single'] = (result_df['running_gpu_single'] / 
                                              result_df['total_gpu_num']).round(3)
        
        return result_df
    
    @staticmethod
    def _create_events(df, time_col, event_type, is_gpu=False):
        """创建事件数据框"""
        events = df[[time_col, 'gpu_num']].copy()
        events.rename(columns={time_col: 'time'}, inplace=True)
        events['event_type'] = event_type
        events['is_gpu'] = is_gpu
        return events
    
    @staticmethod
    def _process_throughput_chunk(time_chunk, all_events):
        """处理一组时间点的吞吐量数据"""
        chunk_results = pd.DataFrame(index=range(len(time_chunk)))
        chunk_results['time'] = time_chunk
        
        # 初始化计数列
        throughput_columns = ['submit_job_all', 'start_job_all', 'end_job_all',
                             'submit_gpu_job', 'start_gpu_job', 'end_gpu_job',
                             'submit_gpu_num', 'start_gpu_num', 'end_gpu_num']
        for col in throughput_columns:
            chunk_results[col] = 0
        
        # 处理时间间隔
        for i in range(len(chunk_results) - 1):
            current_time = chunk_results.loc[i, 'time']
            next_time = chunk_results.loc[i+1, 'time']
            
            # 筛选当前时间间隔内的事件
            interval_events = all_events[(all_events['time'] >= current_time) & 
                                   (all_events['time'] < next_time)]
            
            # 更新统计数据
            for _, event in interval_events.iterrows():
                if event['event_type'] == 'submit':
                    if event['is_gpu']:
                        chunk_results.loc[i, 'submit_gpu_job'] += 1
                        chunk_results.loc[i, 'submit_gpu_num'] += event['gpu_num']
                    chunk_results.loc[i, 'submit_job_all'] += 1
                elif event['event_type'] == 'start':
                    if event['is_gpu']:
                        chunk_results.loc[i, 'start_gpu_job'] += 1
                        chunk_results.loc[i, 'start_gpu_num'] += event['gpu_num']
                    chunk_results.loc[i, 'start_job_all'] += 1
                elif event['event_type'] == 'end':
                    if event['is_gpu']:
                        chunk_results.loc[i, 'end_gpu_job'] += 1
                        chunk_results.loc[i, 'end_gpu_num'] += event['gpu_num']
                    chunk_results.loc[i, 'end_job_all'] += 1
        
        return chunk_results
    
    @staticmethod
    def _process_throughput_chunk_optimized(time_indices, date_range, all_events):
        """处理一组时间点的吞吐量数据 - 优化版"""
        chunk_results = []
        
        for i in time_indices:
            current_time = date_range[i]
            next_time = date_range[i+1] if i < len(date_range)-1 else date_range[i] + pd.Timedelta('1h')
            
            # 筛选当前时间间隔内的事件
            interval_events = all_events[(all_events['time'] >= current_time) & 
                                   (all_events['time'] < next_time)]
            
            # 初始化计数
            result = {
                'time': current_time,
                'submit_job_all': 0, 'start_job_all': 0, 'end_job_all': 0,
                'submit_gpu_job': 0, 'start_gpu_job': 0, 'end_gpu_job': 0,
                'submit_gpu_num': 0, 'start_gpu_num': 0, 'end_gpu_num': 0
            }
            
            # 计算统计值
            for _, event in interval_events.iterrows():
                if event['event_type'] == 'submit':
                    if event['is_gpu']:
                        result['submit_gpu_job'] += 1
                        result['submit_gpu_num'] += event['gpu_num']
                    result['submit_job_all'] += 1
                elif event['event_type'] == 'start':
                    if event['is_gpu']:
                        result['start_gpu_job'] += 1
                        result['start_gpu_num'] += event['gpu_num']
                    result['start_job_all'] += 1
                elif event['event_type'] == 'end':
                    if event['is_gpu']:
                        result['end_gpu_job'] += 1
                        result['end_gpu_num'] += event['gpu_num']
                    result['end_job_all'] += 1
            
            chunk_results.append(result)
            
        return chunk_results

    def analyze_throughput(self, interval='1h', max_workers=32):
        """分析作业吞吐量 - 进一步优化版"""
        # 设置时间范围
        start_date = self.df['submit_time'].min()
        end_date = self.df['end_time'].max()
        
        print(f"Analyzing throughput from {start_date} to {end_date} (interval: {interval})")
        
        # 创建时间序列
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        print(f"Created {len(date_range)} time points for analysis")
        
        # 预先筛选GPU作业
        gpu_jobs = self.df[self.df['gpu_num'] > 0]
        
        # 预计算所有作业的状态变化
        print("Preprocessing job events...")
        events = []
        
        # 为所有作业添加提交、开始和结束事件
        for col, event_type in [('submit_time', 'submit'), ('start_time', 'start'), ('end_time', 'end')]:
            events.append(self._create_events(self.df, col, event_type, is_gpu=False))
            events.append(self._create_events(gpu_jobs, col, event_type, is_gpu=True))
        
        # 合并所有事件
        all_events = pd.concat(events)
        all_events.sort_values('time', inplace=True)
        
        # 计算均衡工作负载
        print(f"Using {max_workers} CPU cores for parallel processing")
        
        # 简化分割方式
        balanced_chunks = np.array_split(range(len(date_range)), max_workers)
        
        # 并行处理
        print("Processing throughput data in parallel...")
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for chunk in balanced_chunks:
                futures.append(executor.submit(
                    self._process_throughput_chunk_optimized, 
                    chunk,
                    date_range,
                    all_events
                ))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.extend(future.result())
        
        # 排序结果并转换为DataFrame
        results.sort(key=lambda x: x['time'])
        throughput_df = pd.DataFrame(results)
        throughput_df.set_index('time', inplace=True)
        
        return throughput_df
    
    @staticmethod
    def _process_user_chunk(users_chunk, df):
        """处理一组用户的统计数据
        
        Args:
            users_chunk: 用户ID列表
            df: 完整作业DataFrame
        
        Returns:
            用户统计DataFrame
        """
        columns = [
            'job_num', 'gpu_job_num', 'cpu_job_num', 
            'avg_duration', 'max_duration', 'avg_gpu_duration', 'avg_cpu_duration',
            'avg_queue', 'max_queue', 'avg_gpu_queue', 'avg_cpu_queue',
            'total_gpu_time', 'total_cpu_time', 'total_node_time'
        ]
        
        # 添加完成率相关列
        for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
            columns.extend([
                f'{state.lower()}_percent',
                f'{state.lower()}_gpu_percent',
                f'{state.lower()}_cpu_percent'
            ])
        
        chunk_stats = pd.DataFrame(index=users_chunk, columns=columns)
        
        for user in users_chunk:
            user_jobs = df[df['user'] == user]
            user_gpu_jobs = user_jobs[user_jobs['gpu_num'] > 0]
            user_cpu_jobs = user_jobs[user_jobs['gpu_num'] == 0]
            
            # 基本作业统计
            chunk_stats.loc[user, 'job_num'] = len(user_jobs)
            chunk_stats.loc[user, 'gpu_job_num'] = len(user_gpu_jobs)
            chunk_stats.loc[user, 'cpu_job_num'] = len(user_cpu_jobs)
            
            # 资源使用
            chunk_stats.loc[user, 'avg_duration'] = user_jobs['duration'].mean() if len(user_jobs) > 0 else 0
            chunk_stats.loc[user, 'max_duration'] = user_jobs['duration'].max() if len(user_jobs) > 0 else 0
            chunk_stats.loc[user, 'avg_gpu_duration'] = user_gpu_jobs['duration'].mean() if len(user_gpu_jobs) > 0 else 0
            chunk_stats.loc[user, 'avg_cpu_duration'] = user_cpu_jobs['duration'].mean() if len(user_cpu_jobs) > 0 else 0
            
            chunk_stats.loc[user, 'avg_queue'] = user_jobs['queue'].mean() if len(user_jobs) > 0 else 0
            chunk_stats.loc[user, 'max_queue'] = user_jobs['queue'].max() if len(user_jobs) > 0 else 0
            chunk_stats.loc[user, 'avg_gpu_queue'] = user_gpu_jobs['queue'].mean() if len(user_gpu_jobs) > 0 else 0
            chunk_stats.loc[user, 'avg_cpu_queue'] = user_cpu_jobs['queue'].mean() if len(user_cpu_jobs) > 0 else 0
            
            # GPU/CPU资源计算
            chunk_stats.loc[user, 'total_gpu_time'] = (user_gpu_jobs['gpu_num'] * user_gpu_jobs['duration']).sum()
            chunk_stats.loc[user, 'total_cpu_time'] = (user_jobs['cpu_num'] * user_jobs['duration']).sum() if 'cpu_num' in user_jobs.columns else 0
            chunk_stats.loc[user, 'total_node_time'] = (user_jobs['node_num'] * user_jobs['duration']).sum() if 'node_num' in user_jobs.columns else 0
            
            # 作业完成率分析
            if len(user_jobs) > 0:
                for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
                    chunk_stats.loc[user, f'{state.lower()}_percent'] = len(user_jobs[user_jobs['state'] == state]) / len(user_jobs)
            
            if len(user_gpu_jobs) > 0:
                for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
                    chunk_stats.loc[user, f'{state.lower()}_gpu_percent'] = len(user_gpu_jobs[user_gpu_jobs['state'] == state]) / len(user_gpu_jobs)
            
            if len(user_cpu_jobs) > 0:
                for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
                    chunk_stats.loc[user, f'{state.lower()}_cpu_percent'] = len(user_cpu_jobs[user_cpu_jobs['state'] == state]) / len(user_cpu_jobs)
        
        return chunk_stats
    
    def analyze_users(self, max_workers=32):
        """分析用户行为 - 并行优化版"""
        print("Analyzing user behavior...")
        
        users = self.df['user'].unique()
        print(f"Total users: {len(users)}")
        
        # 优化1: 预先按用户分组
        user_groups = self.df.groupby('user')
        
        # 优化2: 预先准备结果DataFrame
        columns = [
            'job_num', 'gpu_job_num', 'cpu_job_num', 
            'avg_duration', 'max_duration', 'avg_gpu_duration', 'avg_cpu_duration',
            'avg_queue', 'max_queue', 'avg_gpu_queue', 'avg_cpu_queue',
            'total_gpu_time', 'total_cpu_time', 'total_node_time'
        ]
        
        # 添加完成率相关列
        for state in ['COMPLETED', 'CANCELLED', 'FAILED']:
            columns.extend([
                f'{state.lower()}_percent',
                f'{state.lower()}_gpu_percent',
                f'{state.lower()}_cpu_percent'
            ])
        
        user_stats = pd.DataFrame(index=users, columns=columns)
        
        # 优化3: 并行处理用户分析
        user_chunks = np.array_split(users, min(max_workers, len(users)))
        results = []
        
        print("Processing user data in parallel...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for chunk in user_chunks:
                futures.append(executor.submit(
                    self._process_user_chunk, 
                    chunk, 
                    self.df
                ))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())
        
        # 合并结果
        for user_chunk_stats in results:
            for user in user_chunk_stats.index:
                user_stats.loc[user] = user_chunk_stats.loc[user]
        
        # 按作业数量排序
        user_stats.sort_values(by='job_num', ascending=False, inplace=True)
        
        # 填充NaN值
        user_stats.fillna(0, inplace=True)
        
        return user_stats
    
    @staticmethod
    def _process_queue_chunk(df_chunk):
        """并行处理队列时间分析的数据块"""
        # 统计基本队列时间
        queue_stats = {
            'avg_queue_time': df_chunk['queue'].mean(),
            'median_queue_time': df_chunk['queue'].median(),
            'max_queue_time': df_chunk['queue'].max(),
            'min_queue_time': df_chunk['queue'].min(),
        }
        
        # 计算百分位
        for p in [90, 95, 99]:
            queue_stats[f'p{p}_queue_time'] = df_chunk['queue'].quantile(p/100)
        
        # 按时间段统计
        queue_stats['short_queue_ratio'] = (df_chunk['queue'] < 60).mean()  # 1分钟以内
        queue_stats['medium_queue_ratio'] = ((df_chunk['queue'] >= 60) & (df_chunk['queue'] < 3600)).mean()  # 1小时以内
        queue_stats['long_queue_ratio'] = (df_chunk['queue'] >= 3600).mean()  # 1小时以上
        
        return queue_stats

    def analyze_queue_time(self, max_workers=32):
        """分析作业队列时间 - 并行优化版"""
        print("Analyzing queue time statistics...")
        
        # 确保队列时间列存在
        if 'queue' not in self.df.columns:
            self.df['queue'] = (self.df['start_time'] - self.df['submit_time']).dt.total_seconds()
        
        # 过滤无效数据
        valid_df = self.df[self.df['queue'] >= 0].copy()
        
        # 区分GPU和CPU作业
        gpu_jobs = valid_df[valid_df['gpu_num'] > 0]
        cpu_jobs = valid_df[valid_df['gpu_num'] == 0]
        
        print(f"Analyzing queue time for {len(valid_df)} jobs ({len(gpu_jobs)} GPU jobs)")
        
        # 准备并行处理的数据块
        data_chunks = [valid_df, gpu_jobs, cpu_jobs]
        labels = ['all', 'gpu', 'cpu']
        
        # 并行处理
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_queue_chunk, chunk): label 
                      for chunk, label in zip(data_chunks, labels)}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                label = futures[future]
                results[label] = future.result()
        
        # 转换为DataFrame
        queue_stats = pd.DataFrame(results).T
        queue_stats.index.name = 'job_type'
        
        return queue_stats
    
    @staticmethod
    def _process_duration_chunk(df_chunk):
        """并行处理作业持续时间分析的数据块"""
        # 统计基本持续时间
        duration_stats = {
            'avg_duration': df_chunk['duration'].mean(),
            'median_duration': df_chunk['duration'].median(),
            'max_duration': df_chunk['duration'].max(),
            'min_duration': df_chunk['duration'].min(),
        }
        
        # 计算百分位
        for p in [90, 95, 99]:
            duration_stats[f'p{p}_duration'] = df_chunk['duration'].quantile(p/100)
        
        # 按时长统计
        duration_stats['short_job_ratio'] = (df_chunk['duration'] < 300).mean()  # 5分钟以内
        duration_stats['medium_job_ratio'] = ((df_chunk['duration'] >= 300) & 
                                             (df_chunk['duration'] < 3600)).mean()  # 1小时以内
        duration_stats['long_job_ratio'] = ((df_chunk['duration'] >= 3600) & 
                                           (df_chunk['duration'] < 86400)).mean()  # 1天以内
        duration_stats['very_long_job_ratio'] = (df_chunk['duration'] >= 86400).mean()  # 1天以上
        
        return duration_stats

    def analyze_job_duration(self, max_workers=32):
        """分析作业持续时间 - 并行优化版"""
        print("Analyzing job duration statistics...")
        
        # 过滤无效数据
        valid_df = self.df[self.df['duration'] > 0].copy()
        
        # 区分GPU和CPU作业
        gpu_jobs = valid_df[valid_df['gpu_num'] > 0]
        cpu_jobs = valid_df[valid_df['gpu_num'] == 0]
        
        print(f"Analyzing duration for {len(valid_df)} jobs ({len(gpu_jobs)} GPU jobs)")
        
        # 准备并行处理的数据块
        data_chunks = [valid_df, gpu_jobs, cpu_jobs]
        labels = ['all', 'gpu', 'cpu']
        
        # 并行处理
        results = {}
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._process_duration_chunk, chunk): label 
                      for chunk, label in zip(data_chunks, labels)}
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                label = futures[future]
                results[label] = future.result()
        
        # 转换为DataFrame
        duration_stats = pd.DataFrame(results).T
        duration_stats.index.name = 'job_type'
        
        return duration_stats
    
    @staticmethod
    def _process_completion_chunk(df_chunk, total_jobs, job_type):
        """并行处理作业完成状态的数据块"""
        # 统计各状态作业数量
        status_counts = df_chunk['state'].value_counts()
        
        # 计算各状态占比
        result = {}
        for state in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY']:
            count = status_counts.get(state, 0)
            result[f'{state.lower()}_count'] = count
            result[f'{state.lower()}_ratio'] = count / total_jobs if total_jobs > 0 else 0
        
        # 添加作业类型标记
        result['job_type'] = job_type
        result['total_jobs'] = total_jobs
        
        return result

    def analyze_completion_status(self, max_workers=32):
        """分析作业完成状态 - 并行优化版"""
        print("Analyzing job completion status...")
        
        # 区分GPU和CPU作业
        gpu_jobs = self.df[self.df['gpu_num'] > 0]
        cpu_jobs = self.df[self.df['gpu_num'] == 0]
        
        # 准备并行处理的数据块
        all_total = len(self.df)
        gpu_total = len(gpu_jobs)
        cpu_total = len(cpu_jobs)
        
        data_chunks = [(self.df, all_total, 'all'), 
                       (gpu_jobs, gpu_total, 'gpu'), 
                       (cpu_jobs, cpu_total, 'cpu')]
        
        # 并行处理
        results = []
        with ProcessPoolExecutor(max_workers=min(max_workers, 3)) as executor:  # 只有3个任务，限制进程数
            futures = []
            
            for chunk, total, job_type in data_chunks:
                futures.append(executor.submit(
                    self._process_completion_chunk, 
                    chunk, 
                    total,
                    job_type
                ))
            
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.append(future.result())
        
        # 转换为DataFrame
        completion_stats = pd.DataFrame(results)
        completion_stats.set_index('job_type', inplace=True)
        
        return completion_stats
    
    def plot_resource_usage(self, sequence_df=None, save_path=None):
        """绘制资源使用时间序列图"""
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
        """绘制吞吐量时间序列图"""
        if throughput_df is None:
            throughput_df = self.analyze_throughput()
        
        print("Plotting throughput...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # 绘制作业提交/开始/结束
        ax1.plot(throughput_df.index, throughput_df['submit_job_all'], 'b-', label='Submitted Jobs')
        ax1.plot(throughput_df.index, throughput_df['start_job_all'], 'g-', label='Started Jobs')
        ax1.plot(throughput_df.index, throughput_df['end_job_all'], 'r-', label='Ended Jobs')
        
        ax1.set_ylabel('Job Count')
        ax1.set_title('Job Throughput Over Time')
        ax1.legend()
        ax1.grid(True, linestyle=':')
        
        # 绘制GPU作业吞吐量
        ax2.plot(throughput_df.index, throughput_df['submit_gpu_job'], 'b-', label='Submitted GPU Jobs')
        ax2.plot(throughput_df.index, throughput_df['start_gpu_job'], 'g-', label='Started GPU Jobs')
        ax2.plot(throughput_df.index, throughput_df['end_gpu_job'], 'r-', label='Ended GPU Jobs')
        
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
        """绘制用户统计图"""
        if user_stats is None:
            user_stats = self.analyze_users()
        
        print(f"Plotting top {top_n} user statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 选取前N名用户
        top_users = user_stats.head(top_n)
        
        # 1. 作业数量
        ax1 = axes[0, 0]
        top_users[['cpu_job_num', 'gpu_job_num']].plot(
            kind='barh', stacked=True, ax=ax1, 
            color=['skyblue', 'orange']
        )
        ax1.set_title(f'Top {top_n} Users by Job Count')
        ax1.set_xlabel('Number of Jobs')
        ax1.set_ylabel('User')
        ax1.legend(['CPU Jobs', 'GPU Jobs'])
        ax1.grid(axis='x', linestyle=':')
        
        # 2. GPU时间使用
        ax2 = axes[0, 1]
        top_users['total_gpu_time'].plot(kind='barh', ax=ax2, color='orange')
        ax2.set_title(f'Top {top_n} Users by GPU Time Usage')
        ax2.set_xlabel('GPU Hours')
        ax2.set_ylabel('User')
        ax2.grid(axis='x', linestyle=':')
        
        # 3. 平均等待时间
        ax3 = axes[1, 0]
        top_users[['avg_queue', 'avg_gpu_queue']].plot(
            kind='barh', ax=ax3,
            color=['skyblue', 'orange']
        )
        ax3.set_title(f'Top {top_n} Users by Queue Time')
        ax3.set_xlabel('Average Queue Time (seconds)')
        ax3.set_ylabel('User')
        ax3.legend(['All Jobs', 'GPU Jobs'])
        ax3.grid(axis='x', linestyle=':')
        
        # 4. 作业完成率
        ax4 = axes[1, 1]
        completion_data = top_users[['completed_percent', 'cancelled_percent', 'failed_percent']]
        completion_data.plot(
            kind='barh', stacked=True, ax=ax4,
            color=['green', 'orange', 'red']
        )
        ax4.set_title(f'Top {top_n} Users by Job Completion Rate')
        ax4.set_xlabel('Completion Rate')
        ax4.set_ylabel('User')
        ax4.legend(['Completed', 'Cancelled', 'Failed'])
        ax4.grid(axis='x', linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved user statistics plot to {save_path}")
        
        return fig
    
    def plot_queue_distribution(self, save_path=None):
        """绘制队列时间分布图"""
        print("Plotting queue time distribution...")
        
        # 过滤有效数据
        valid_df = self.df[self.df['queue'] >= 0].copy()
        gpu_jobs = valid_df[valid_df['gpu_num'] > 0]
        cpu_jobs = valid_df[valid_df['gpu_num'] == 0]
        
        # 创建分布图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 整体队列时间直方图（对数刻度）
        ax1 = axes[0, 0]
        valid_df['queue'].hist(ax=ax1, bins=50, alpha=0.7, color='blue')
        ax1.set_title('Queue Time Distribution (All Jobs)')
        ax1.set_xlabel('Queue Time (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')
        ax1.grid(True, linestyle=':')
        
        # 2. GPU vs CPU作业队列时间对比
        ax2 = axes[0, 1]
        gpu_jobs['queue'].hist(ax=ax2, bins=50, alpha=0.7, label='GPU Jobs', color='orange')
        cpu_jobs['queue'].hist(ax=ax2, bins=50, alpha=0.7, label='CPU Jobs', color='skyblue')
        ax2.set_title('Queue Time: GPU vs CPU Jobs')
        ax2.set_xlabel('Queue Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        # 3. 队列时间CDF
        ax3 = axes[1, 0]
        gpu_jobs_sorted = np.sort(gpu_jobs['queue'])
        cpu_jobs_sorted = np.sort(cpu_jobs['queue'])
        all_jobs_sorted = np.sort(valid_df['queue'])
        
        gpu_y = np.arange(1, len(gpu_jobs_sorted) + 1) / len(gpu_jobs_sorted)
        cpu_y = np.arange(1, len(cpu_jobs_sorted) + 1) / len(cpu_jobs_sorted)
        all_y = np.arange(1, len(all_jobs_sorted) + 1) / len(all_jobs_sorted)
        
        ax3.plot(all_jobs_sorted, all_y, label='All Jobs', color='blue')
        ax3.plot(gpu_jobs_sorted, gpu_y, label='GPU Jobs', color='orange')
        ax3.plot(cpu_jobs_sorted, cpu_y, label='CPU Jobs', color='skyblue')
        
        ax3.set_title('Queue Time CDF')
        ax3.set_xlabel('Queue Time (seconds)')
        ax3.set_ylabel('CDF')
        ax3.legend()
        ax3.grid(True, linestyle=':')
        
        # 4. 按周分组的队列时间箱线图
        ax4 = axes[1, 1]
        valid_df['weekday'] = valid_df['submit_time'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        sns.boxplot(x='weekday', y='queue', data=valid_df, ax=ax4, order=weekday_order)
        ax4.set_title('Queue Time by Weekday')
        ax4.set_xlabel('Submission Day')
        ax4.set_ylabel('Queue Time (seconds)')
        ax4.set_yscale('log')
        ax4.set_xticklabels(weekday_order, rotation=45)
        ax4.grid(True, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved queue distribution plot to {save_path}")
        
        return fig
    
    def plot_duration_distribution(self, save_path=None):
        """绘制作业持续时间分布图"""
        print("Plotting job duration distribution...")
        
        # 过滤有效数据
        valid_df = self.df[self.df['duration'] > 0].copy()
        gpu_jobs = valid_df[valid_df['gpu_num'] > 0]
        cpu_jobs = valid_df[valid_df['gpu_num'] == 0]
        
        # 创建分布图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 整体持续时间直方图（对数刻度）
        ax1 = axes[0, 0]
        valid_df['duration'].hist(ax=ax1, bins=50, alpha=0.7, color='blue')
        ax1.set_title('Job Duration Distribution (All Jobs)')
        ax1.set_xlabel('Duration (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_yscale('log')
        ax1.grid(True, linestyle=':')
        
        # 2. GPU vs CPU作业持续时间对比
        ax2 = axes[0, 1]
        gpu_jobs['duration'].hist(ax=ax2, bins=50, alpha=0.7, label='GPU Jobs', color='orange')
        cpu_jobs['duration'].hist(ax=ax2, bins=50, alpha=0.7, label='CPU Jobs', color='skyblue')
        ax2.set_title('Duration: GPU vs CPU Jobs')
        ax2.set_xlabel('Duration (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, linestyle=':')
        
        # 3. 持续时间CDF
        ax3 = axes[1, 0]
        gpu_jobs_sorted = np.sort(gpu_jobs['duration'])
        cpu_jobs_sorted = np.sort(cpu_jobs['duration'])
        all_jobs_sorted = np.sort(valid_df['duration'])
        
        gpu_y = np.arange(1, len(gpu_jobs_sorted) + 1) / len(gpu_jobs_sorted)
        cpu_y = np.arange(1, len(cpu_jobs_sorted) + 1) / len(cpu_jobs_sorted)
        all_y = np.arange(1, len(all_jobs_sorted) + 1) / len(all_jobs_sorted)
        
        ax3.plot(all_jobs_sorted, all_y, label='All Jobs', color='blue')
        ax3.plot(gpu_jobs_sorted, gpu_y, label='GPU Jobs', color='orange')
        ax3.plot(cpu_jobs_sorted, cpu_y, label='CPU Jobs', color='skyblue')
        
        ax3.set_title('Job Duration CDF')
        ax3.set_xlabel('Duration (seconds)')
        ax3.set_ylabel('CDF')
        ax3.legend()
        ax3.grid(True, linestyle=':')
        
        # 4. 多GPU vs 单GPU持续时间箱线图
        ax4 = axes[1, 1]
        multi_gpu = gpu_jobs[gpu_jobs['gpu_num'] > 1]
        single_gpu = gpu_jobs[gpu_jobs['gpu_num'] == 1]
        
        data = []
        labels = []
        
        if len(single_gpu) > 0:
            data.append(single_gpu['duration'])
            labels.append('Single GPU')
        
        if len(multi_gpu) > 0:
            data.append(multi_gpu['duration'])
            labels.append('Multi GPU')
            
        if data:
            ax4.boxplot(data, labels=labels)
            ax4.set_title('Duration: Single vs Multi GPU Jobs')
            ax4.set_ylabel('Duration (seconds)')
            ax4.set_yscale('log')
            ax4.grid(True, linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved duration distribution plot to {save_path}")
        
        return fig
    
    def plot_completion_status(self, completion_stats=None, save_path=None):
        """绘制作业完成状态统计图"""
        if completion_stats is None:
            completion_stats = self.analyze_completion_status()
        
        print("Plotting job completion status...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 状态计数条形图
        status_cols = ['completed_count', 'failed_count', 'cancelled_count', 'timeout_count', 'out_of_memory_count']
        status_names = ['Completed', 'Failed', 'Cancelled', 'Timeout', 'OOM']
        
        # 转置数据以便绘图
        status_counts = completion_stats[status_cols].T
        status_counts.index = status_names
        
        ax1 = axes[0]
        status_counts.plot(kind='bar', ax=ax1)
        ax1.set_title('Job Completion Status Counts')
        ax1.set_xlabel('Status')
        ax1.set_ylabel('Count')
        ax1.set_yscale('log')
        ax1.legend(title='Job Type')
        ax1.grid(axis='y', linestyle=':')
        
        # 2. 状态比例条形图
        ratio_cols = ['completed_ratio', 'failed_ratio', 'cancelled_ratio', 'timeout_ratio', 'out_of_memory_ratio']
        ratio_df = completion_stats[ratio_cols] * 100  # 转换为百分比
        ratio_df.columns = status_names
        
        ax2 = axes[1]
        ratio_df.plot(kind='bar', stacked=True, ax=ax2, 
                     color=['green', 'red', 'orange', 'purple', 'brown'])
        ax2.set_title('Job Completion Status Ratio')
        ax2.set_xlabel('Job Type')
        ax2.set_ylabel('Percentage (%)')
        ax2.legend(title='Status')
        ax2.grid(axis='y', linestyle=':')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved completion status plot to {save_path}")
        
        return fig
    
    def analyze_all(self, output_dir=None, use_daily_interval=False, max_workers=32):
        """执行完整的轨迹分析并生成报告
        
        Args:
            output_dir: 输出目录
            use_daily_interval: 是否使用天作为时间间隔（默认False，使用小时间隔）
            max_workers: 最大并行工作进程数（默认32）
        """
        # 设置输出目录
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Performing complete trace analysis, results will be saved to {output_dir}")
        
        # 1. 资源使用分析
        print("\n=== Resource Usage Analysis ===")
        
        # 根据数据量选择合适的时间间隔
        interval = '1D' if use_daily_interval else '1h'
        print(f"Using {interval} as time interval")
        
        sequence_df = self.analyze_resource_usage(interval=interval, max_workers=max_workers)
        sequence_df.to_csv(f"{output_dir}/resource_usage.csv")
        
        # 绘制资源使用图表
        self.plot_resource_usage(sequence_df, save_path=f"{output_dir}/resource_usage.png")
        
        # 2. 吞吐量分析
        print("\n=== Throughput Analysis ===")
        throughput_df = self.analyze_throughput(interval=interval, max_workers=max_workers)
        throughput_df.to_csv(f"{output_dir}/throughput.csv")
        
        # 绘制吞吐量图表
        self.plot_throughput(throughput_df, save_path=f"{output_dir}/throughput.png")
        
        # 3. 用户行为分析
        print("\n=== User Behavior Analysis ===")
        user_stats = self.analyze_users(max_workers=max_workers)
        user_stats.to_csv(f"{output_dir}/user_stats.csv")
        
        # 绘制用户统计图
        self.plot_user_stats(user_stats, save_path=f"{output_dir}/user_stats.png")
        
        # 4. 基本统计信息
        print("\n=== Basic Statistics ===")
        self.basic_stats.to_csv(f"{output_dir}/basic_stats.csv")
        
        # 5. 作业等待时间分析
        print("\n=== Queue Time Analysis ===")
        queue_stats = self.analyze_queue_time(max_workers=max_workers)
        queue_stats.to_csv(f"{output_dir}/queue_stats.csv")
        
        # 绘制队列时间分布
        self.plot_queue_distribution(save_path=f"{output_dir}/queue_distribution.png")
        
        # 6. 作业运行时间分析 
        print("\n=== Job Duration Analysis ===")
        duration_stats = self.analyze_job_duration(max_workers=max_workers)
        duration_stats.to_csv(f"{output_dir}/duration_stats.csv")
        
        # 绘制运行时间分布
        self.plot_duration_distribution(save_path=f"{output_dir}/duration_distribution.png")
        
        # 7. 作业完成状态分析
        print("\n=== Job Completion Status Analysis ===")
        completion_stats = self.analyze_completion_status(max_workers=max_workers)
        completion_stats.to_csv(f"{output_dir}/completion_stats.csv")
        
        # 绘制完成状态统计
        self.plot_completion_status(completion_stats, save_path=f"{output_dir}/completion_status.png")
        
        print(f"\nAnalysis complete! All results saved to {output_dir}")

# 使用示例
if __name__ == "__main__":
    # 设置输入和输出路径
    data_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/convert_data/second_gen_helios_format.csv"
    output_dir = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/trace_characterization"
    
    # 创建分析器
    analyzer = TraceAnalyzer(data_path)
    
    # 执行完整分析 - 默认使用小时作为间隔
    analyzer.analyze_all(output_dir=output_dir, use_daily_interval=False, max_workers=32) 