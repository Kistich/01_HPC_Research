import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import timedelta

class UserCharacterizationAnalyzer:
    """User Characterization Analyzer"""
    
    def __init__(self, data_path):
        """Initialize analyzer
        
        Args:
            data_path: Job data file path
        """
        self.data_path = data_path
        print(f"Loading data from {data_path}...")
        
        # 1. Load raw job data
        self.jobs_df = pd.read_csv(data_path, parse_dates=['submit_time', 'start_time', 'end_time'])
        
        # 2. Calculate necessary derived fields
        if 'duration' not in self.jobs_df.columns:
            self.jobs_df['duration'] = (self.jobs_df['end_time'] - self.jobs_df['start_time']).dt.total_seconds()
        
        if 'queue' not in self.jobs_df.columns:
            self.jobs_df['queue'] = (self.jobs_df['start_time'] - self.jobs_df['submit_time']).dt.total_seconds()
        
        # 3. Create user statistics data
        self.user_df = self._generate_user_stats()
        print(f"Processed {len(self.jobs_df)} jobs from {len(self.user_df)} unique users")

    def _generate_user_stats(self):
        """Generate user-level statistics"""
        # Group by user
        user_stats = self.jobs_df.groupby('user').agg({
            'job_id': 'count',                           # Total job count
            'gpu_num': ['sum', 'mean'],                  # GPU total and average
            'cpu_num': ['sum', 'mean'],                  # CPU total and average
            'node_num': ['sum', 'mean'],                 # Node total and average
            'duration': ['sum', 'mean', 'median', 'max'],# Runtime statistics
            'queue': ['sum', 'mean', 'median', 'max']    # Queue time statistics
        })
        
        # Flatten column names
        user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        
        # Calculate GPU job statistics
        gpu_jobs = self.jobs_df[self.jobs_df['gpu_num'] > 0]
        gpu_stats = gpu_jobs.groupby('user').agg({
            'job_id': 'count',                           # GPU job count
            'gpu_num': ['sum', 'mean'],                  # GPU usage total and average
            'duration': 'sum',                           # GPU runtime total
            'queue': 'sum'                               # GPU queue time total
        })
        gpu_stats.columns = ['gpu_jobs_count', 'total_gpu_used', 'avg_gpu_per_job', 
                            'total_gpu_time', 'total_gpu_pend_time']
        
        # Calculate completion rates
        completed_jobs = self.jobs_df[self.jobs_df['state'] == 'COMPLETED']
        completed_stats = completed_jobs.groupby('user').agg({
            'job_id': 'count'  # Completed job count
        })
        completed_stats.columns = ['completed_jobs']
        
        # GPU job completion rate
        completed_gpu_jobs = gpu_jobs[gpu_jobs['state'] == 'COMPLETED']
        completed_gpu_stats = completed_gpu_jobs.groupby('user').agg({
            'job_id': 'count'  # Completed GPU job count
        })
        completed_gpu_stats.columns = ['completed_gpu_jobs']
        
        # Merge all user statistics
        user_df = user_stats.copy()
        
        # Use left join to keep all users
        user_df = user_df.join(gpu_stats, how='left')
        user_df = user_df.join(completed_stats, how='left')
        user_df = user_df.join(completed_gpu_stats, how='left')
        
        # Fill missing values
        user_df.fillna(0, inplace=True)
        
        # Calculate completion rates
        user_df['completed_percent'] = user_df['completed_jobs'] / user_df['job_id_count']
        user_df['completed_gpu_percent'] = np.where(
            user_df['gpu_jobs_count'] > 0,
            user_df['completed_gpu_jobs'] / user_df['gpu_jobs_count'],
            0
        )
        
        return user_df
    
    def plot_user_resource_distribution(self, save_path=None):
        """Plot user resource usage distribution CDF"""
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
        
        # 1. GPU time cumulative distribution
        gpu_time = self.user_df['total_gpu_time'].copy()
        gpu_time.sort_values(ascending=False, inplace=True)
        gpu_time_sum = gpu_time.sum()
        
        gpu_cdf = []
        for i in range(len(self.user_df)):
            gpu_cdf.append(gpu_time.iloc[:i+1].sum() / gpu_time_sum * 100)
        
        ax1.plot(np.linspace(0, 100, len(self.user_df)), gpu_cdf, '-', linewidth=2)
        ax1.set_xlabel('User Percentile (%)')
        ax1.set_ylabel('Cumulative GPU Time (%)')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.grid(linestyle=':')
        ax1.set_title('User GPU Time Distribution')
        
        # 2. CPU time cumulative distribution
        cpu_time = self.user_df['duration_sum'].copy()
        cpu_time.sort_values(ascending=False, inplace=True)
        cpu_time_sum = cpu_time.sum()
        
        cpu_cdf = []
        for i in range(len(self.user_df)):
            cpu_cdf.append(cpu_time.iloc[:i+1].sum() / cpu_time_sum * 100)
        
        ax2.plot(np.linspace(0, 100, len(self.user_df)), cpu_cdf, '-', linewidth=2)
        ax2.set_xlabel('User Percentile (%)')
        ax2.set_ylabel('Cumulative CPU Time (%)')
        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.grid(linestyle=':')
        ax2.set_title('User CPU Time Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_queue_and_completion(self, save_path=None):
        """Plot queue time CDF and job completion rate histogram"""
        fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
        
        # 1. Queue time cumulative distribution
        queue_time = self.user_df['total_gpu_pend_time'].copy()
        queue_time.sort_values(ascending=False, inplace=True)
        queue_time_sum = queue_time.sum()
        
        queue_cdf = []
        for i in range(len(self.user_df)):
            queue_cdf.append(queue_time.iloc[:i+1].sum() / queue_time_sum * 100)
        
        ax1.plot(np.linspace(0, 100, len(self.user_df)), queue_cdf, '-', linewidth=2)
        ax1.set_xlabel('User Percentile (%)')
        ax1.set_ylabel('Cumulative Queuing Time (%)')
        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1.grid(linestyle=':')
        ax1.set_title('User Queue Time Distribution')
        
        # 2. GPU job completion rate histogram
        completion_rates = self.user_df['completed_gpu_percent'] * 100
        sns.histplot(completion_rates, bins=10, ax=ax2, kde=False)
        ax2.set_xlabel('GPU Job Completion Rate (%)')
        ax2.set_ylabel('User Count')
        ax2.set_xlim(0, 100)
        ax2.grid(linestyle=':')
        ax2.set_title('GPU Job Completion Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def generate_user_report(self, output_dir="./user_analysis"):
        """Generate complete user analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save user statistics data
        self.user_df.to_csv(f"{output_dir}/user_stats.csv")
        
        # 2. Plot resource usage distribution
        fig1 = self.plot_user_resource_distribution(save_path=f"{output_dir}/user_resource_distribution.png")
        plt.close(fig1)
        
        # 3. Plot queue time and completion rate
        fig2 = self.plot_queue_and_completion(save_path=f"{output_dir}/user_queue_completion.png")
        plt.close(fig2)
        
        # 4. Generate user ranking report
        top_users = self.user_df.sort_values('total_gpu_time', ascending=False).head(20)
        top_users[['job_id_count', 'gpu_jobs_count', 'total_gpu_time', 'total_gpu_used', 
                  'completed_gpu_percent']].to_csv(f"{output_dir}/top_gpu_users.csv")
        
        print(f"User analysis report generated at: {output_dir}")

if __name__ == "__main__":
    # Set data path
    data_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/User_behavior_analysis/convert_data/all_data/second_gen_helios_format.csv"
    output_dir = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/user_characterization/all_data"
    
    # Create analyzer and execute analysis
    analyzer = UserCharacterizationAnalyzer(data_path)
    analyzer.generate_user_report(output_dir=output_dir)
