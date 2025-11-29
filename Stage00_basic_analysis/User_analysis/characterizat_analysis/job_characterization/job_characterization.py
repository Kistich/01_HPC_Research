# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager

# Set English font configuration
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans", "Bitstream Vera Sans"]
plt.rcParams["axes.unicode_minus"] = False  # Fix negative sign display issue


def load_data(filepath):
    """Load job data
    Args:
        filepath: CSV file path
    Returns:
        Processed DataFrame
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Convert time fields (if needed)
    time_columns = ['submit_time', 'start_time', 'end_time']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    
    # Ensure GPU field exists
    if 'gpu_num' not in df.columns and 'ngpus' in df.columns:
        df['gpu_num'] = df['ngpus']
    
    # Ensure duration field exists
    if 'duration' not in df.columns and 'start_time' in df.columns and 'end_time' in df.columns:
        df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()
    
    return df
   
def plot_cdf_job_time(df, save_path=None):
    """Plot job execution time CDF
    Args:
        df: Job data
        save_path: Save path
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    
    # Define time bins
    time_bins = [2 ** i for i in range(0, 22)]
    
    # Separate GPU and CPU jobs
    gpu_jobs = df[df['gpu_num'] > 0]
    cpu_jobs = df[df['gpu_num'] == 0]
    
    # Calculate GPU job CDF
    gpu_job_counts = []
    for t in time_bins:
        count = len(gpu_jobs[gpu_jobs['duration'] <= t])
        percentage = count / len(gpu_jobs) * 100 if len(gpu_jobs) > 0 else 0
        gpu_job_counts.append(percentage)
    
    # Calculate CPU job CDF
    cpu_job_counts = []
    for t in time_bins:
        count = len(cpu_jobs[cpu_jobs['duration'] <= t])
        percentage = count / len(cpu_jobs) * 100 if len(cpu_jobs) > 0 else 0
        cpu_job_counts.append(percentage)
    
    # Plot GPU job CDF
    ax1.plot(time_bins, gpu_job_counts, '-', linewidth=2)
    ax1.set_xscale('log')
    ax1.set_xlabel('GPU Job Duration (seconds)')
    ax1.set_ylabel('Job Percentage (%)')
    ax1.set_title('GPU Job Duration Distribution')
    ax1.grid(True, linestyle=':')
    
    # Plot CPU job CDF
    ax2.plot(time_bins, cpu_job_counts, '-', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('CPU Job Duration (seconds)')
    ax2.set_ylabel('Job Percentage (%)')
    ax2.set_title('CPU Job Duration Distribution')
    ax2.grid(True, linestyle=':')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
   
def plot_gpu_distribution(df, save_path=None):
    """Plot GPU count distribution
    Args:
        df: Job data
        save_path: Save path
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    
    # Only select GPU jobs
    gpu_jobs = df[df['gpu_num'] > 0].copy()
    
    # Calculate GPU time
    gpu_jobs['gpu_time'] = gpu_jobs['duration'] * gpu_jobs['gpu_num']
    total_gpu_time = gpu_jobs['gpu_time'].sum()
    
    # Define GPU count bins
    gpu_bins = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Calculate job count CDF
    job_counts = []
    for g in gpu_bins:
        count = len(gpu_jobs[gpu_jobs['gpu_num'] <= g])
        percentage = count / len(gpu_jobs) * 100
        job_counts.append(percentage)
    
    # Calculate GPU time CDF
    gpu_time_percents = []
    for g in gpu_bins:
        time_sum = gpu_jobs[gpu_jobs['gpu_num'] <= g]['gpu_time'].sum()
        percentage = time_sum / total_gpu_time * 100
        gpu_time_percents.append(percentage)
    
    # Plot job count CDF
    ax1.plot(gpu_bins, job_counts, '-', linewidth=2)
    ax1.set_xlabel('GPU Count')
    ax1.set_ylabel('Job Percentage (%)')
    ax1.set_title('Job Count Distribution')
    ax1.grid(True, linestyle=':')
    ax1.set_xticks(gpu_bins)
    ax1.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128+'])
    
    # Plot GPU time CDF
    ax2.plot(gpu_bins, gpu_time_percents, '-', linewidth=2)
    ax2.set_xlabel('GPU Count')
    ax2.set_ylabel('GPU Time Percentage (%)')
    ax2.set_title('GPU Time Distribution')
    ax2.grid(True, linestyle=':')
    ax2.set_xticks(gpu_bins)
    ax2.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128+'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
   
def plot_job_status(df, save_path=None):
    """Plot job status distribution
    Args:
        df: Job data
        save_path: Save path
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 4))
    
    # Count CPU and GPU job status
    gpu_jobs = df[df['gpu_num'] > 0].copy()  # Use copy() to avoid SettingWithCopyWarning
    cpu_jobs = df[df['gpu_num'] == 0].copy()  # Use copy() to avoid SettingWithCopyWarning
    
    # Map status names (adjust based on actual data)
    status_mapping = {
        'COMPLETED': 'Completed',
        'CANCELLED': 'Canceled',
        'FAILED': 'Failed',
        # Add other possible status mappings
    }
    
    # Apply mapping
    df['status_mapped'] = df['state'].map(status_mapping).fillna(df['state'])
    gpu_jobs['status_mapped'] = gpu_jobs['state'].map(status_mapping).fillna(gpu_jobs['state'])
    cpu_jobs['status_mapped'] = cpu_jobs['state'].map(status_mapping).fillna(cpu_jobs['state'])
    
    # Calculate CPU/GPU job status percentages
    cpu_status = cpu_jobs['status_mapped'].value_counts(normalize=True) * 100
    gpu_status = gpu_jobs['status_mapped'].value_counts(normalize=True) * 100
    
    # Plot CPU/GPU status comparison
    x = np.array([1, 2])
    width = 0.25
    
    # Ensure all statuses have data
    for status in ['Completed', 'Canceled', 'Failed']:
        if status not in cpu_status:
            cpu_status[status] = 0
        if status not in gpu_status:
            gpu_status[status] = 0
    
    # Sort data
    cpu_status = cpu_status.reindex(['Completed', 'Canceled', 'Failed'])
    gpu_status = gpu_status.reindex(['Completed', 'Canceled', 'Failed'])
    
    # Plot bar chart
    rects1 = ax1.bar(x[0]-width, cpu_status['Completed'], width, label='Completed')
    rects2 = ax1.bar(x[0], cpu_status['Canceled'], width, label='Canceled')
    rects3 = ax1.bar(x[0]+width, cpu_status['Failed'], width, label='Failed')
    
    rects4 = ax1.bar(x[1]-width, gpu_status['Completed'], width)
    rects5 = ax1.bar(x[1], gpu_status['Canceled'], width)
    rects6 = ax1.bar(x[1]+width, gpu_status['Failed'], width)
    
    # Add labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(['CPU Jobs', 'GPU Jobs'])
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('CPU vs GPU Job Status Comparison')
    ax1.legend()
    
    # Analyze status by GPU count
    gpu_bins = [1, 2, 4, 8, 16, 32, 64]
    statuses = ['Completed', 'Canceled', 'Failed']
    
    # Prepare data
    status_by_gpu = {status: [] for status in statuses}
    
    for i in range(len(gpu_bins)):
        if i < len(gpu_bins) - 1:
            bin_jobs = gpu_jobs[(gpu_jobs['gpu_num'] >= gpu_bins[i]) & (gpu_jobs['gpu_num'] < gpu_bins[i+1])]
        else:
            bin_jobs = gpu_jobs[gpu_jobs['gpu_num'] >= gpu_bins[i]]
        
        bin_total = len(bin_jobs)
        if bin_total > 0:
            for status in statuses:
                count = len(bin_jobs[bin_jobs['status_mapped'] == status])
                status_by_gpu[status].append(count / bin_total * 100)
        else:
            for status in statuses:
                status_by_gpu[status].append(0)
    
    # Ensure each status list length matches gpu_bins
    for status in statuses:
        if len(status_by_gpu[status]) != len(gpu_bins):
            print(f"Warning: Status '{status}' data length is {len(status_by_gpu[status])}, should be {len(gpu_bins)}")
            # If lengths don't match, pad or truncate to correct length
            status_by_gpu[status] = status_by_gpu[status][:len(gpu_bins)] + [0] * (len(gpu_bins) - len(status_by_gpu[status]))
    
    # Plot stacked bar chart
    bottom_values = np.zeros(len(gpu_bins))
    
    for i, status in enumerate(statuses):
        if i == 0:
            ax2.bar(gpu_bins, status_by_gpu[status], label=status)
            bottom_values = np.array(status_by_gpu[status])
        else:
            ax2.bar(gpu_bins, status_by_gpu[status], bottom=bottom_values, label=status)
            bottom_values = bottom_values + np.array(status_by_gpu[status])
    
    ax2.set_xlabel('GPU Count')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Job Status Distribution by GPU Count')
    ax2.set_xticks(gpu_bins)
    ax2.set_xticklabels(['1', '2', '4', '8', '16', '32', '≥64'])
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# Usage example
def analyze_helios_jobs(filepath, output_dir=None):
    """Complete job analysis workflow
    Args:
        filepath: Data file path
        output_dir: Output directory
    """
    import os
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analysis results will be saved to: {output_dir}")
    
    # Load data
    print(f"Loading data: {filepath}")
    df = load_data(filepath)
    print(f"Loaded {len(df)} job records")
    
    # Basic statistics
    print("\nBasic statistics:")
    print(f"Total jobs: {len(df)}")
    print(f"GPU jobs: {len(df[df['gpu_num'] > 0])} ({len(df[df['gpu_num'] > 0])/len(df)*100:.2f}%)")
    print(f"CPU jobs: {len(df[df['gpu_num'] == 0])} ({len(df[df['gpu_num'] == 0])/len(df)*100:.2f}%)")
    
    # Status statistics
    print("\nJob status distribution:")
    print(df['state'].value_counts(normalize=True).mul(100))
    
    try:
        # Execution time analysis
        print("\nPlotting job execution time CDF...")
        plot_cdf_job_time(df, save_path=f"{output_dir}/job_execution_time.png")
        
        # GPU distribution analysis
        print("\nPlotting GPU count distribution...")
        plot_gpu_distribution(df, save_path=f"{output_dir}/gpu_distribution.png")
        
        # Job status analysis
        print("\nPlotting job status distribution...")
        plot_job_status(df, save_path=f"{output_dir}/job_status.png")
        
        print(f"\nAnalysis complete, results saved to: {output_dir}")
    except Exception as e:
        import traceback
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    import os
    
    # Get current file directory, go up two levels to find convert_data directory
    data_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/User_behavior_analysis/convert_data/all_data/second_gen_helios_format.csv"
    output_dir = "/mnt/raid/liuhongbin/job_analysis/job_analysis/characterizat_analysis/job_characterization/all_data"
    
    print(f"Data file path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Analyze jobs
    analyze_helios_jobs(data_path, output_dir=output_dir) 