import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import time

class DiskIOAnalyzer:
    """Disk I/O Data Analyzer"""
    
    def __init__(self, excel_path):
        """Initialize analyzer"""
        self.excel_path = excel_path
        # Configure server groups based on actual data
        self.server_groups = {
            'CPU1': [f'cpu1-{i}' for i in range(1, 111)],  # 110 nodes
            'CPU2': [f'cpu2-{i}' for i in range(1, 31)],   # 30 nodes
            'CPU3': [f'cpu3-{i}' for i in range(1, 21)],   # 20 nodes
            'GPU1': [f'gpu1-{i}' for i in range(1, 51)],   # 50 nodes
            'GPU2': [f'gpu2-{i}' for i in range(1, 16)],   # 15 nodes
            'GPU3': [f'gpu3-{i}' for i in range(1, 15)],   # Only 14 nodes (adjusted from 15)
            'BIGMEM': [f'bigmen-{i}' for i in range(1, 7)], # 6 nodes
            'Inference': [f'推理服务器{i:02d}' for i in range(1, 3)], # 2 nodes
            'Training': [f'训练服务器{i:02d}' for i in range(1, 9)]  # 8 nodes
        }
        self.data = {}
        self.metrics = set()
        # Group colors
        self.group_colors = {
            'CPU1': '#1f77b4',      # blue
            'CPU2': '#ff7f0e',      # orange
            'CPU3': '#2ca02c',      # green
            'GPU1': '#d62728',      # red
            'GPU2': '#9467bd',      # purple
            'GPU3': '#8c564b',      # brown
            'BIGMEM': '#e377c2',    # pink
            'Inference': '#7f7f7f',  # gray
            'Training': '#bcbd22'    # olive
        }
        # Cache for loaded data
        self._cached_data = None

    def load_sheet(self, sheet_name):
        """Load specific worksheet data"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            # Ensure column names are consistent
            if list(df.columns) == ['timestamp', 'value', 'metric']:
                # Record all unique metric types
                self.metrics.update(df['metric'].unique())
                # Ensure timestamp is datetime type
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df
            else:
                print(f"Warning: Column names in {sheet_name} don't match expected format")
                return None
        except Exception as e:
            print(f"Error loading {sheet_name}: {str(e)}")
            return None
            
    def load_all_data(self):
        """Load all group data with caching"""
        if self._cached_data is not None:
            print("Using cached data")
            return self._cached_data
            
        print("Loading all data (this may take some time)...")
        start_time = time.time()
        
        all_data = {}
        for group_name, servers in self.server_groups.items():
            print(f"Loading {group_name} group data...")
            group_data = {}
            for server in servers:
                df = self.load_sheet(server)
                if df is not None:
                    group_data[server] = df
            all_data[group_name] = group_data
        
        self._cached_data = all_data
        end_time = time.time()
        print(f"Data loading completed in {end_time - start_time:.2f} seconds")
        return all_data
        
    def load_group_data(self, group_name):
        """Load data for a specific group with caching"""
        all_data = self.load_all_data()
        return all_data.get(group_name, {})
    
    def compare_monthly_patterns(self, output_dir='figures'):
        """Compare monthly I/O patterns across all groups"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load all data
        all_data = self.load_all_data()
        
        # Prepare monthly average data
        monthly_avgs = {}
        
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            # Combine all servers' data
            all_values = []
            
            for server, df in group_data.items():
                # Add group identifier
                df_copy = df.copy()
                df_copy['group'] = group_name
                df_copy['server'] = server
                all_values.append(df_copy)
            
            if all_values:
                combined_df = pd.concat(all_values)
                # Aggregate by month
                monthly_df = combined_df.set_index('timestamp')['value'].resample('M').mean()
                monthly_avgs[group_name] = monthly_df
        
        # Plot monthly comparison
        plt.figure(figsize=(18, 10))
        
        for group_name, monthly_data in monthly_avgs.items():
            plt.plot(monthly_data.index, monthly_data.values, 
                    label=f'{group_name} ({len(all_data[group_name])} nodes)', 
                    color=self.group_colors[group_name],
                    linewidth=2.5)
            
        plt.title('Monthly Average Disk I/O by Server Type', fontsize=18)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average I/O (MB/s)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set month format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_monthly_io.png', dpi=300)
        plt.close()
        
        return monthly_avgs
    
    def compare_weekly_patterns(self, output_dir='figures'):
        """Compare weekly I/O patterns across all groups"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cached data
        all_data = self.load_all_data()
        
        # Prepare weekly average data
        weekly_avgs = {}
        
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            # Combine all servers' data
            all_values = []
            
            for server, df in group_data.items():
                df_copy = df.copy()
                df_copy['group'] = group_name
                df_copy['server'] = server
                all_values.append(df_copy)
            
            if all_values:
                combined_df = pd.concat(all_values)
                # Aggregate by week
                weekly_df = combined_df.set_index('timestamp')['value'].resample('W').mean()
                weekly_avgs[group_name] = weekly_df
        
        # Plot weekly comparison
        plt.figure(figsize=(18, 10))
        
        for group_name, weekly_data in weekly_avgs.items():
            plt.plot(weekly_data.index, weekly_data.values, 
                    label=f'{group_name} ({len(all_data[group_name])} nodes)', 
                    color=self.group_colors[group_name],
                    linewidth=1.5)
            
        plt.title('Weekly Average Disk I/O by Server Type', fontsize=18)
        plt.xlabel('Week', fontsize=14)
        plt.ylabel('Average I/O (MB/s)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set week format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_weekly_io.png', dpi=300)
        plt.close()
        
        return weekly_avgs
    
    def create_group_comparison_plots(self, output_dir='figures'):
        """Create direct comparison plots between all 9 groups"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cached data
        all_data = self.load_all_data()
        
        # Calculate statistics for each group
        all_stats = {}
        server_counts = {}
        
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            # Combine all servers' data
            all_values = []
            for df in group_data.values():
                all_values.extend(df['value'].tolist())
            
            # Calculate stats
            all_stats[group_name] = {
                'mean': np.mean(all_values),
                'median': np.median(all_values),
                'max': np.max(all_values),
                'min': np.min(all_values),
                'std': np.std(all_values)
            }
            
            server_counts[group_name] = len(group_data)
        
        # 1. Average I/O comparison
        plt.figure(figsize=(14, 8))
        labels = [f"{group} ({count} nodes)" for group, count in server_counts.items()]
        means = [stats['mean'] for group, stats in all_stats.items()]
        colors = [self.group_colors[group] for group in all_stats.keys()]
        
        bars = plt.bar(labels, means, color=colors)
        plt.title('Average Disk I/O Comparison by Server Group', fontsize=16)
        plt.ylabel('Average I/O (MB/s)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_avg_io_comparison.png', dpi=300)
        plt.close()
        
        # 2. Maximum I/O comparison
        plt.figure(figsize=(14, 8))
        maxs = [stats['max'] for group, stats in all_stats.items()]
        
        bars = plt.bar(labels, maxs, color=colors)
        plt.title('Maximum Disk I/O Comparison by Server Group', fontsize=16)
        plt.ylabel('Maximum I/O (MB/s)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_max_io_comparison.png', dpi=300)
        plt.close()
        
        # 3. Box plot for I/O distribution comparison
        plt.figure(figsize=(16, 10))
        box_data = []
        box_labels = []
        
        for group_name in all_stats.keys():
            group_data = all_data[group_name]
            if not group_data:
                continue
                
            # Sample data points to avoid overcrowded plots
            sample_values = []
            for df in group_data.values():
                # Sample up to 1000 points from each server
                if len(df) > 1000:
                    sample = df['value'].sample(1000, random_state=42)
                else:
                    sample = df['value']
                sample_values.extend(sample.tolist())
            
            box_data.append(sample_values)
            box_labels.append(f"{group_name} ({server_counts[group_name]} nodes)")
        
        plt.boxplot(box_data, labels=box_labels, patch_artist=True,
                  boxprops=dict(alpha=0.7),
                  medianprops=dict(color='black'))
        
        # Add colors
        for i, box in enumerate(plt.gca().artists):
            box.set_facecolor(list(self.group_colors.values())[i])
        
        plt.title('Disk I/O Distribution Comparison by Server Group', fontsize=16)
        plt.ylabel('I/O (MB/s)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_io_distribution.png', dpi=300)
        plt.close()
        
        return all_stats
        
    def analyze_io_patterns(self, group_name, output_dir='figures'):
        """Analyze I/O patterns for a single group and generate individual charts"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        group_dir = os.path.join(output_dir, group_name)
        Path(group_dir).mkdir(exist_ok=True)
        
        # Use cached data
        group_data = self.load_group_data(group_name)
        if not group_data:
            print(f"Cannot analyze {group_name} group: No valid data found")
            return
            
        # Calculate basic statistics
        server_stats = {}
        for server, df in group_data.items():
            server_stats[server] = {
                'mean': df['value'].mean(),
                'max': df['value'].max(),
                'median': df['value'].median(),
                'std': df['value'].std(),
                'data_points': len(df)
            }
        
        # 1. Average I/O bar chart
        plt.figure(figsize=(20, 10))
        avg_values = {k: v['mean'] for k, v in server_stats.items()}
        bars = plt.bar(avg_values.keys(), avg_values.values(), color=self.group_colors[group_name])
        plt.title(f'Average Disk I/O for {group_name} Group', fontsize=16)
        plt.xticks(rotation=90, fontsize=10)
        plt.ylabel('Average I/O (MB/s)', fontsize=14)
        
        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{group_dir}/{group_name}_avg_io.png', dpi=300)
        plt.close()
        
        # 2. Maximum I/O bar chart
        plt.figure(figsize=(20, 10))
        max_values = {k: v['max'] for k, v in server_stats.items()}
        bars = plt.bar(max_values.keys(), max_values.values(), color=self.group_colors[group_name])
        plt.title(f'Maximum Disk I/O for {group_name} Group', fontsize=16)
        plt.xticks(rotation=90, fontsize=10)
        plt.ylabel('Maximum I/O (MB/s)', fontsize=14)
        
        # Add value labels to bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8, rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{group_dir}/{group_name}_max_io.png', dpi=300)
        plt.close()
        
        return {
            'server_stats': server_stats,
            'server_count': len(group_data)
        }
        
    def run_complete_analysis(self, output_dir='figures'):
        """Run complete analysis workflow"""
        print("Starting comprehensive disk I/O analysis...")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Load all data at once to avoid repeated loading
        self.load_all_data()
        
        # Analyze all groups' monthly and weekly patterns
        print("\n===== Analyzing time series patterns for all groups =====")
        monthly_avgs = self.compare_monthly_patterns(output_dir)
        weekly_avgs = self.compare_weekly_patterns(output_dir)
        results['monthly_avgs'] = monthly_avgs
        results['weekly_avgs'] = weekly_avgs
        
        # Create inter-group comparison charts
        print("\n===== Creating direct comparison charts for all groups =====")
        all_stats = self.create_group_comparison_plots(output_dir)
        results['all_stats'] = all_stats
        
        # Analyze each group's details (optional, for individual charts)
        print("\n===== Analyzing details for each group =====")
        for group in self.server_groups:
            print(f"Analyzing {group} group...")
            group_results = self.analyze_io_patterns(group, output_dir)
            results[group] = group_results
            
        # Generate summary report
        self.generate_summary_report(output_dir, results)
            
        return results
        
    def generate_summary_report(self, output_dir, results):
        """Generate a summary report of the analysis"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        report_path = os.path.join(output_dir, "analysis_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("===== DISK I/O ANALYSIS SUMMARY =====\n\n")
            f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Excel file: {os.path.abspath(self.excel_path)}\n\n")
            
            if 'all_stats' in results:
                all_stats = results['all_stats']
                
                # Sort by average I/O
                sorted_by_avg = sorted(all_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
                
                f.write("RANKING BY AVERAGE I/O:\n")
                for i, (group, stats) in enumerate(sorted_by_avg, 1):
                    f.write(f"{i}. {group}: {stats['mean']:.2f} MB/s\n")
                
                # Sort by maximum I/O
                sorted_by_max = sorted(all_stats.items(), key=lambda x: x[1]['max'], reverse=True)
                
                f.write("\nRANKING BY MAXIMUM I/O:\n")
                for i, (group, stats) in enumerate(sorted_by_max, 1):
                    f.write(f"{i}. {group}: {stats['max']:.2f} MB/s\n")
            
            f.write("\nDETAILED STATISTICS BY GROUP:\n")
            for group in self.server_groups:
                if group in results and results[group]:
                    f.write(f"\n{group} Group:\n")
                    f.write(f"  - Number of servers: {results[group]['server_count']}\n")
                    
                    # Calculate group-wide statistics
                    all_means = [stats['mean'] for stats in results[group]['server_stats'].values()]
                    all_maxs = [stats['max'] for stats in results[group]['server_stats'].values()]
                    
                    if all_means:
                        f.write(f"  - Average I/O across all servers: {np.mean(all_means):.2f} MB/s\n")
                        f.write(f"  - Highest average I/O: {np.max(all_means):.2f} MB/s\n")
                        f.write(f"  - Lowest average I/O: {np.min(all_means):.2f} MB/s\n")
                        f.write(f"  - Average maximum I/O: {np.mean(all_maxs):.2f} MB/s\n")
                        f.write(f"  - Highest maximum I/O: {np.max(all_maxs):.2f} MB/s\n")
            
            # Add timestamp
            import datetime
            f.write(f"\nReport generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze server disk I/O data')
    parser.add_argument('excel_path', help='Path to Excel file')
    parser.add_argument('--output', default='Server_analysis/CPU/disk_io/disk_io_figures', help='Output directory for charts')
    args = parser.parse_args()
    
    analyzer = DiskIOAnalyzer(args.excel_path)
    results = analyzer.run_complete_analysis(args.output)
    
    print(f"\nAnalysis complete! Charts saved to {os.path.abspath(args.output)} directory")
    
    # Print summary report
    print("\n===== ANALYSIS SUMMARY =====")
    if 'all_stats' in results:
        all_stats = results['all_stats']
        # Sort by average I/O
        sorted_by_avg = sorted(all_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        print("Ranking by average I/O:")
        for i, (group, stats) in enumerate(sorted_by_avg, 1):
            print(f"{i}. {group}: {stats['mean']:.2f} MB/s")
            
        # Sort by maximum I/O
        sorted_by_max = sorted(all_stats.items(), key=lambda x: x[1]['max'], reverse=True)
        
        print("\nRanking by maximum I/O:")
        for i, (group, stats) in enumerate(sorted_by_max, 1):
            print(f"{i}. {group}: {stats['max']:.2f} MB/s")

if __name__ == "__main__":
    main() 