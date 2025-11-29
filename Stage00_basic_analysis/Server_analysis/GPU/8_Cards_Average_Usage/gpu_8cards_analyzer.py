#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

class GPUUsageAnalyzer:
    """GPU 8 Cards Average Usage Analyzer"""
    
    def __init__(self, excel_path):
        """Initialize analyzer"""
        self.excel_path = excel_path
        # Configure server groups based on actual data in basic_info.txt
        self.server_groups = {
            'GPU1': [f'gpu1-{i}' for i in range(1, 51)],   # 50 nodes
            'GPU2': [f'gpu2-{i}' for i in range(1, 16)],   # 15 nodes
            'GPU3': [f'gpu3-{i}' for i in range(1, 15)]    # 14 nodes
        }
        self.data = {}
        self.metrics = set()
        # Group colors
        self.group_colors = {
            'GPU1': '#d62728',      # red
            'GPU2': '#9467bd',      # purple
            'GPU3': '#8c564b'       # brown
        }
        # Cache for loaded data
        self._cached_data = None
        
    def load_sheet(self, sheet_name):
        """Load specific worksheet data"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
            # Ensure column names are consistent
            if 'timestamp' not in df.columns or 'value' not in df.columns:
                print(f"Warning: Sheet {sheet_name} has unexpected columns: {df.columns}")
                return None
                
            # Convert timestamp to datetime if needed
            if not pd.api.types.is_datetime64_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Add calculated columns
            df['day_of_week'] = df['timestamp'].dt.day_name()
            df['hour'] = df['timestamp'].dt.hour
            df['date'] = df['timestamp'].dt.date
            df['week'] = df['timestamp'].dt.isocalendar().week
            df['month'] = df['timestamp'].dt.month
            
            # Add actual column with server name
            df['server'] = sheet_name
            
            # If metric column exists, collect unique metrics
            if 'metric' in df.columns:
                self.metrics.update(df['metric'].unique())
                
            return df
        except Exception as e:
            print(f"Error loading sheet {sheet_name}: {e}")
            return None

    def load_data(self):
        """Load all data for all server groups"""
        if self._cached_data is not None:
            print("Using cached data...")
            return self._cached_data
            
        print(f"Loading data from {self.excel_path}...")
        start_time = time.time()
        
        all_data = {}
        
        # Process each server group
        for group_name, servers in self.server_groups.items():
            group_data = {}
            
            for server in servers:
                # Try to load the server's data
                data = self.load_sheet(server)
                if data is not None:
                    group_data[server] = data
                
            if group_data:
                all_data[group_name] = group_data
                print(f"  Loaded {len(group_data)} servers for group {group_name}")
            
        print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
        
        self._cached_data = all_data
        self.data = all_data
        
        return all_data

    def compare_monthly_patterns(self, output_dir='figures'):
        """Compare monthly usage patterns across all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Processing monthly patterns for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Group by month and calculate average
            monthly_avg = combined_df.groupby('month')['value'].mean()
            
            # Plot
            plt.plot(monthly_avg.index, monthly_avg.values, 
                    marker='o', linewidth=2, label=group_name,
                    color=self.group_colors.get(group_name))
        
        plt.title('Monthly GPU Usage Patterns (8 Cards Average)', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average GPU Usage (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.savefig(f'{output_dir}/all_groups_monthly_gpu_usage.png', dpi=300)
        plt.close()
        
        print(f"Monthly pattern comparison saved to {output_dir}/all_groups_monthly_gpu_usage.png")

    def compare_weekly_patterns(self, output_dir='figures'):
        """Compare weekly usage patterns across all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 8))
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Processing weekly patterns for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Create a mapping of days to integers (0 = Monday, 6 = Sunday)
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            combined_df['day_num'] = combined_df['day_of_week'].map(day_map)
            
            # Group by day of week and calculate average
            weekly_avg = combined_df.groupby('day_num')['value'].mean()
            
            # Sort by day number
            weekly_avg = weekly_avg.sort_index()
            
            # Plot
            plt.plot(weekly_avg.index, weekly_avg.values, 
                    marker='o', linewidth=2, label=group_name,
                    color=self.group_colors.get(group_name))
        
        plt.title('Weekly GPU Usage Patterns (8 Cards Average)', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Average GPU Usage (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        plt.savefig(f'{output_dir}/all_groups_weekly_gpu_usage.png', dpi=300)
        plt.close()
        
        print(f"Weekly pattern comparison saved to {output_dir}/all_groups_weekly_gpu_usage.png")

    def analyze_daily_patterns(self, group_name=None, output_dir='figures'):
        """Analyze daily usage patterns for a specific group or all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If no group specified, process all groups
        groups_to_process = [group_name] if group_name else list(data.keys())
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                continue
                
            print(f"Analyzing daily patterns for {group}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(data[group].values())
            
            # Group by hour and calculate average
            hourly_avg = combined_df.groupby('hour')['value'].mean()
            
            plt.figure(figsize=(15, 6))
            plt.plot(hourly_avg.index, hourly_avg.values, 
                    marker='o', linewidth=2,
                    color=self.group_colors.get(group, '#1f77b4'))
            
            plt.title(f'Daily GPU Usage Pattern for {group} (8 Cards Average)', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Average GPU Usage (%)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(24))
            
            plt.savefig(f'{output_dir}/{group}_daily_gpu_usage.png', dpi=300)
            plt.close()
            
            print(f"Daily pattern for {group} saved to {output_dir}/{group}_daily_gpu_usage.png")

    def find_high_usage_periods(self, threshold=80, group_name=None, output_file='high_usage_periods.txt'):
        """Find periods of sustained high GPU usage"""
        data = self.load_data()
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # If no group specified, process all groups
        groups_to_process = [group_name] if group_name else list(data.keys())
        
        results = {}
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                continue
                
            print(f"Finding high usage periods for {group}...")
            
            # Process each server in the group
            group_results = {}
            
            for server, df in data[group].items():
                # Find periods where usage exceeds threshold
                high_usage = df[df['value'] > threshold].copy()
                
                if high_usage.empty:
                    continue
                    
                # Sort by timestamp
                high_usage = high_usage.sort_values('timestamp')
                
                # Find consecutive periods
                high_usage['time_diff'] = high_usage['timestamp'].diff()
                
                # Start a new group when the time difference is more than 1 hour
                high_usage['period_group'] = (high_usage['time_diff'] > pd.Timedelta(hours=1)).cumsum()
                
                # Group by period and calculate duration
                periods = high_usage.groupby('period_group').agg({
                    'timestamp': ['min', 'max', 'count'],
                    'value': ['mean', 'max']
                })
                
                # Flatten multi-index columns
                periods.columns = ['start_time', 'end_time', 'num_samples', 'avg_usage', 'max_usage']
                
                # Calculate duration in hours
                periods['duration_hours'] = (periods['end_time'] - periods['start_time']).dt.total_seconds() / 3600
                
                # Filter for periods longer than 1 hour
                significant_periods = periods[periods['duration_hours'] >= 1].copy()
                
                if not significant_periods.empty:
                    group_results[server] = significant_periods
            
            if group_results:
                results[group] = group_results
        
        # Write results to file
        with open(output_file, 'w') as f:
            f.write(f"High GPU Usage Periods (Threshold: {threshold}%)\n")
            f.write("=" * 80 + "\n\n")
            
            for group, group_results in results.items():
                f.write(f"\n{group} Server Group:\n")
                f.write("-" * 40 + "\n")
                
                for server, periods in group_results.items():
                    f.write(f"\n  {server}:\n")
                    
                    for idx, row in periods.iterrows():
                        f.write(f"    Period {idx+1}:\n")
                        f.write(f"      Start: {row['start_time']}\n")
                        f.write(f"      End: {row['end_time']}\n")
                        f.write(f"      Duration: {row['duration_hours']:.2f} hours\n")
                        f.write(f"      Average Usage: {row['avg_usage']:.2f}%\n")
                        f.write(f"      Maximum Usage: {row['max_usage']:.2f}%\n")
                        f.write("\n")
            
        print(f"High usage periods saved to {output_file}")
        return results

    def generate_heatmap(self, group_name=None, output_dir='figures'):
        """Generate heatmap of GPU usage by day of week and hour of day"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If no group specified, process all groups
        groups_to_process = [group_name] if group_name else list(data.keys())
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                continue
                
            print(f"Generating heatmap for {group}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(data[group].values())
            
            # Create pivot table: day of week vs hour of day
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create the pivot table
            heatmap_data = combined_df.pivot_table(
                values='value', 
                index='day_of_week', 
                columns='hour', 
                aggfunc='mean'
            )
            
            # Reorder days
            heatmap_data = heatmap_data.reindex(day_order)
            
            plt.figure(figsize=(15, 8))
            sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt=".1f", cbar_kws={'label': 'GPU Usage (%)'})
            
            plt.title(f'GPU Usage Heatmap for {group} (8 Cards Average)', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Day of Week', fontsize=14)
            
            plt.savefig(f'{output_dir}/{group}_heatmap.png', dpi=300)
            plt.close()
            
            print(f"Heatmap for {group} saved to {output_dir}/{group}_heatmap.png")

    def compare_all_groups(self, output_dir='figures'):
        """Compare key metrics across all server groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate key metrics for each group
        metrics = {
            'Group': [],
            'Avg_Usage': [],
            'Max_Usage': [],
            'Min_Usage': [],
            'Std_Dev': [],
            'Servers_Count': [],
            'Hours_Above_80': [],
            'Hours_Above_90': []
        }
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Calculating metrics for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Add to metrics
            metrics['Group'].append(group_name)
            metrics['Avg_Usage'].append(combined_df['value'].mean())
            metrics['Max_Usage'].append(combined_df['value'].max())
            metrics['Min_Usage'].append(combined_df['value'].min())
            metrics['Std_Dev'].append(combined_df['value'].std())
            metrics['Servers_Count'].append(len(group_data))
            
            # Calculate hours above thresholds
            hours_above_80 = combined_df[combined_df['value'] > 80].shape[0] / len(group_data)
            hours_above_90 = combined_df[combined_df['value'] > 90].shape[0] / len(group_data)
            
            metrics['Hours_Above_80'].append(hours_above_80)
            metrics['Hours_Above_90'].append(hours_above_90)
        
        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Sort by average usage
        metrics_df = metrics_df.sort_values('Avg_Usage', ascending=False)
        
        # Save to CSV
        metrics_df.to_csv(f'{output_dir}/group_comparison.csv', index=False)
        
        # Create bar charts for key metrics
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df['Group'], metrics_df['Avg_Usage'], 
               color=[self.group_colors.get(g, '#1f77b4') for g in metrics_df['Group']])
        
        plt.title('Average GPU Usage by Server Group (8 Cards Average)', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Average Usage (%)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(f'{output_dir}/group_comparison_avg.png', dpi=300)
        plt.close()
        
        # Create a second chart for high usage hours
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.35
        
        plt.bar(x - width/2, metrics_df['Hours_Above_80'], width, label='>80% Usage',
               color='orange')
        plt.bar(x + width/2, metrics_df['Hours_Above_90'], width, label='>90% Usage',
               color='red')
        
        plt.title('Hours of High GPU Usage by Server Group (8 Cards Average)', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Hours', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(x, metrics_df['Group'])
        plt.legend()
        
        plt.savefig(f'{output_dir}/group_comparison_high_usage.png', dpi=300)
        plt.close()
        
        print(f"Group comparison metrics saved to {output_dir}/group_comparison.csv")
        print(f"Group comparison charts saved to {output_dir}/")
        
        return metrics_df

    def generate_utilization_report(self, output_file='gpu_utilization_report.txt'):
        """Generate a comprehensive utilization report with recommendations"""
        data = self.load_data()
        metrics_df = self.compare_all_groups(output_dir='figures')
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("GPU 8 CARDS AVERAGE UTILIZATION REPORT\n")
            f.write("=====================================\n\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-----------------\n")
            
            # Calculate overall statistics
            all_data = pd.concat([pd.concat(group_data.values()) for group_data in data.values() if group_data])
            
            f.write(f"Overall Average GPU Utilization: {all_data['value'].mean():.2f}%\n")
            f.write(f"Maximum Observed Utilization: {all_data['value'].max():.2f}%\n")
            f.write(f"Minimum Observed Utilization: {all_data['value'].min():.2f}%\n")
            f.write(f"Standard Deviation: {all_data['value'].std():.2f}\n\n")
            
            # Group Summaries
            f.write("SERVER GROUP SUMMARIES\n")
            f.write("----------------------\n\n")
            
            for idx, row in metrics_df.iterrows():
                group = row['Group']
                f.write(f"{group}:\n")
                f.write(f"  Average Utilization: {row['Avg_Usage']:.2f}%\n")
                f.write(f"  Maximum Utilization: {row['Max_Usage']:.2f}%\n")
                f.write(f"  Minimum Utilization: {row['Min_Usage']:.2f}%\n")
                f.write(f"  Standard Deviation: {row['Std_Dev']:.2f}\n")
                f.write(f"  Number of Servers: {row['Servers_Count']}\n")
                f.write(f"  Hours Above 80% Utilization: {row['Hours_Above_80']:.2f}\n")
                f.write(f"  Hours Above 90% Utilization: {row['Hours_Above_90']:.2f}\n\n")
            
            # Identify most and least utilized groups
            most_utilized = metrics_df.iloc[0]['Group']
            least_utilized = metrics_df.iloc[-1]['Group']
            
            # Temporal Patterns
            f.write("TEMPORAL USAGE PATTERNS\n")
            f.write("----------------------\n\n")
            
            # Calculate daily patterns (hour of day)
            hourly_usage = all_data.groupby('hour')['value'].mean()
            peak_hour = hourly_usage.idxmax()
            lowest_hour = hourly_usage.idxmin()
            
            f.write(f"Peak Hour of the Day: {peak_hour}:00 (Average: {hourly_usage[peak_hour]:.2f}%)\n")
            f.write(f"Lowest Hour of the Day: {lowest_hour}:00 (Average: {hourly_usage[lowest_hour]:.2f}%)\n\n")
            
            # Calculate weekly patterns (day of week)
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            inv_day_map = {v: k for k, v in day_map.items()}
            
            all_data['day_num'] = all_data['day_of_week'].map(day_map)
            daily_usage = all_data.groupby('day_of_week')['value'].mean()
            
            peak_day_num = all_data.groupby('day_num')['value'].mean().idxmax()
            lowest_day_num = all_data.groupby('day_num')['value'].mean().idxmin()
            
            peak_day = inv_day_map[peak_day_num]
            lowest_day = inv_day_map[lowest_day_num]
            
            f.write(f"Peak Day of the Week: {peak_day} (Average: {daily_usage[peak_day]:.2f}%)\n")
            f.write(f"Lowest Day of the Week: {lowest_day} (Average: {daily_usage[lowest_day]:.2f}%)\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("--------------\n\n")
            
            # Analyze overall utilization
            overall_avg = all_data['value'].mean()
            if overall_avg < 50:
                f.write("1. Overall GPU utilization is low. Consider consolidating workloads or reducing active GPU servers.\n")
            elif overall_avg > 80:
                f.write("1. Overall GPU utilization is very high. Consider adding more GPU capacity or optimizing workloads.\n")
            else:
                f.write("1. Overall GPU utilization is moderate. Continue monitoring for changes in demand.\n")
            
            # Check for server group imbalances
            max_avg = metrics_df['Avg_Usage'].max()
            min_avg = metrics_df['Avg_Usage'].min()
            
            if max_avg - min_avg > 30:
                f.write(f"2. Significant imbalance between server groups. Consider redistributing workloads from {most_utilized} to {least_utilized}.\n")
            
            # Check for temporal patterns
            if hourly_usage.max() - hourly_usage.min() > 30:
                f.write(f"3. Strong daily usage pattern detected. Consider scheduling intensive batch jobs during low-usage periods (around {lowest_hour}:00).\n")
            
            # Check for day-of-week patterns
            if daily_usage.max() - daily_usage.min() > 15:
                f.write(f"4. Noticeable day-of-week pattern detected. Consider scheduling maintenance during {lowest_day}s when usage is typically lower.\n")
            
            # Check for underutilization
            underutilized_groups = metrics_df[metrics_df['Avg_Usage'] < 30]['Group'].tolist()
            if underutilized_groups:
                f.write(f"5. The following groups show low utilization and may be candidates for workload consolidation: {', '.join(underutilized_groups)}.\n")
            
            # Check for overutilization
            overutilized_groups = metrics_df[metrics_df['Hours_Above_90'] > 100]['Group'].tolist()
            if overutilized_groups:
                f.write(f"6. The following groups show extended periods of very high utilization and may need additional capacity: {', '.join(overutilized_groups)}.\n")
            
            # Final notes
            f.write("\nNOTES:\n")
            f.write("- This report analyzes the average utilization across 8 GPU cards per server.\n")
            f.write("- Individual GPU card usage may vary and could be analyzed separately for more detailed insights.\n")
            f.write("- Consider monitoring memory usage alongside GPU utilization for a more complete picture.\n")
        
        print(f"Utilization report saved to {output_file}")

def main():
    """Main function"""
    # Configure paths
    excel_path = '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx'
    output_dir = 'output'
    figures_dir = 'figures'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = GPUUsageAnalyzer(excel_path)
    
    # Run analyses
    print("Starting GPU 8 Cards Average Usage Analysis...")
    
    # Load data
    analyzer.load_data()
    
    # Generate temporal pattern analyses
    analyzer.compare_monthly_patterns(output_dir=figures_dir)
    analyzer.compare_weekly_patterns(output_dir=figures_dir)
    
    # Analyze daily patterns for each group
    for group in analyzer.server_groups.keys():
        analyzer.analyze_daily_patterns(group_name=group, output_dir=figures_dir)
    
    # Find high usage periods
    analyzer.find_high_usage_periods(
        threshold=80, 
        output_file=f'{output_dir}/high_usage_periods.txt'
    )
    
    # Generate heatmaps
    for group in analyzer.server_groups.keys():
        analyzer.generate_heatmap(group_name=group, output_dir=figures_dir)
    
    # Compare all groups
    analyzer.compare_all_groups(output_dir=figures_dir)
    
    # Generate comprehensive report
    analyzer.generate_utilization_report(
        output_file=f'{output_dir}/gpu_utilization_report.txt'
    )
    
    print("Analysis completed successfully!")

if __name__ == "__main__":
    main()
