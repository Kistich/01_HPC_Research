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

class GPUPowerDrawAnalyzer:
    """GPU Power Draw Analyzer"""
    
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
        """Compare monthly power draw patterns across all groups"""
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
        
        plt.title('Monthly GPU Power Draw Patterns', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average Power Draw (W)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.savefig(f'{output_dir}/all_groups_monthly_powerdraw.png', dpi=300)
        plt.close()
        
        print(f"Monthly pattern comparison saved to {output_dir}/all_groups_monthly_powerdraw.png")

    def compare_weekly_patterns(self, output_dir='figures'):
        """Compare weekly power draw patterns across all groups"""
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
        
        plt.title('Weekly GPU Power Draw Patterns', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Average Power Draw (W)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        plt.savefig(f'{output_dir}/all_groups_weekly_powerdraw.png', dpi=300)
        plt.close()
        
        print(f"Weekly pattern comparison saved to {output_dir}/all_groups_weekly_powerdraw.png")

    def analyze_daily_patterns(self, group_name=None, output_dir='figures'):
        """Analyze daily power draw patterns for a specific group or all groups"""
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
            
            plt.title(f'Daily GPU Power Draw Pattern for {group}', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Average Power Draw (W)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(24))
            
            plt.savefig(f'{output_dir}/{group}_daily_powerdraw.png', dpi=300)
            plt.close()
            
            print(f"Daily pattern for {group} saved to {output_dir}/{group}_daily_powerdraw.png")

    def find_high_power_periods(self, threshold=200, group_name=None, output_file='high_power_periods.txt'):
        """Find periods of sustained high GPU power draw (in Watts)"""
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
                
            print(f"Finding high power draw periods for {group}...")
            
            # Process each server in the group
            group_results = {}
            
            for server, df in data[group].items():
                # Find periods where power draw exceeds threshold
                high_power = df[df['value'] > threshold].copy()
                
                if high_power.empty:
                    continue
                    
                # Sort by timestamp
                high_power = high_power.sort_values('timestamp')
                
                # Find consecutive periods
                high_power['time_diff'] = high_power['timestamp'].diff()
                
                # Start a new group when the time difference is more than 1 hour
                high_power['period_group'] = (high_power['time_diff'] > pd.Timedelta(hours=1)).cumsum()
                
                # Group by period and calculate duration
                periods = high_power.groupby('period_group').agg({
                    'timestamp': ['min', 'max', 'count'],
                    'value': ['mean', 'max']
                })
                
                # Flatten multi-index columns
                periods.columns = ['start_time', 'end_time', 'num_samples', 'avg_power', 'max_power']
                
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
            f.write(f"High GPU Power Draw Periods (Threshold: {threshold}W)\n")
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
                        f.write(f"      Average Power: {row['avg_power']:.2f}W\n")
                        f.write(f"      Maximum Power: {row['max_power']:.2f}W\n")
                        f.write("\n")
            
        print(f"High power draw periods saved to {output_file}")
        return results

    def generate_heatmap(self, group_name=None, output_dir='figures'):
        """Generate heatmap of GPU power draw by day of week and hour of day"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If no group specified, process all groups
        groups_to_process = [group_name] if group_name else list(data.keys())
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                continue
                
            print(f"Generating power draw heatmap for {group}...")
            
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
            sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt=".1f", cbar_kws={'label': 'Power Draw (W)'})
            
            plt.title(f'GPU Power Draw Heatmap for {group}', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Day of Week', fontsize=14)
            
            plt.savefig(f'{output_dir}/{group}_powerdraw_heatmap.png', dpi=300)
            plt.close()
            
            print(f"Power draw heatmap for {group} saved to {output_dir}/{group}_powerdraw_heatmap.png")

    def compare_all_groups(self, output_dir='figures'):
        """Compare key metrics across all server groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate key metrics for each group
        metrics = {
            'Group': [],
            'Avg_Power': [],
            'Max_Power': [],
            'Min_Power': [],
            'Std_Dev': [],
            'Servers_Count': [],
            'Hours_Above_200W': [],
            'Hours_Above_250W': []
        }
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Calculating power metrics for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Add to metrics
            metrics['Group'].append(group_name)
            metrics['Avg_Power'].append(combined_df['value'].mean())
            metrics['Max_Power'].append(combined_df['value'].max())
            metrics['Min_Power'].append(combined_df['value'].min())
            metrics['Std_Dev'].append(combined_df['value'].std())
            metrics['Servers_Count'].append(len(group_data))
            
            # Calculate hours above thresholds
            hours_above_200 = combined_df[combined_df['value'] > 200].shape[0] / len(group_data)
            hours_above_250 = combined_df[combined_df['value'] > 250].shape[0] / len(group_data)
            
            metrics['Hours_Above_200W'].append(hours_above_200)
            metrics['Hours_Above_250W'].append(hours_above_250)
        
        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Sort by average power
        metrics_df = metrics_df.sort_values('Avg_Power', ascending=False)
        
        # Save to CSV
        metrics_df.to_csv(f'{output_dir}/group_power_comparison.csv', index=False)
        
        # Create bar charts for key metrics
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df['Group'], metrics_df['Avg_Power'], 
               color=[self.group_colors.get(g, '#1f77b4') for g in metrics_df['Group']])
        
        plt.title('Average GPU Power Draw by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Average Power Draw (W)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(f'{output_dir}/group_comparison_avg_power.png', dpi=300)
        plt.close()
        
        # Create a second chart for high power hours
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.35
        
        plt.bar(x - width/2, metrics_df['Hours_Above_200W'], width, label='>200W',
               color='orange')
        plt.bar(x + width/2, metrics_df['Hours_Above_250W'], width, label='>250W',
               color='red')
        
        plt.title('Hours of High GPU Power Draw by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Hours', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(x, metrics_df['Group'])
        plt.legend()
        
        plt.savefig(f'{output_dir}/group_comparison_high_power.png', dpi=300)
        plt.close()
        
        print(f"Group power comparison metrics saved to {output_dir}/group_power_comparison.csv")
        print(f"Group power comparison charts saved to {output_dir}/")
        
        return metrics_df

    def analyze_power_efficiency(self, output_dir='figures'):
        """Analyze power efficiency by examining power distribution"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Analyzing power efficiency for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Create histogram of power values
            plt.figure(figsize=(12, 6))
            sns.histplot(combined_df['value'], bins=50, kde=True, color=self.group_colors.get(group_name, '#1f77b4'))
            
            plt.title(f'GPU Power Draw Distribution for {group_name}', fontsize=16)
            plt.xlabel('Power Draw (W)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add vertical lines for typical thresholds
            plt.axvline(x=100, color='green', linestyle='--', label='Low Power (100W)')
            plt.axvline(x=200, color='orange', linestyle='--', label='Medium Power (200W)')
            plt.axvline(x=250, color='red', linestyle='--', label='High Power (250W)')
            
            plt.legend()
            
            plt.savefig(f'{output_dir}/{group_name}_power_distribution.png', dpi=300)
            plt.close()
            
            print(f"Power distribution for {group_name} saved to {output_dir}/{group_name}_power_distribution.png")
            
            # Analyze power state transitions
            if len(group_data) > 0:
                # Pick the first server for transition analysis (as an example)
                server_name = list(group_data.keys())[0]
                server_df = group_data[server_name].copy()
                
                # Sort by timestamp
                server_df = server_df.sort_values('timestamp')
                
                # Calculate power change between consecutive measurements
                server_df['power_change'] = server_df['value'].diff()
                
                # Plot power changes
                plt.figure(figsize=(15, 6))
                plt.hist(server_df['power_change'].dropna(), bins=100, alpha=0.7)
                
                plt.title(f'GPU Power Transitions for {server_name}', fontsize=16)
                plt.xlabel('Power Change Between Consecutive Measurements (W)', fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.savefig(f'{output_dir}/{group_name}_power_transitions.png', dpi=300)
                plt.close()
                
                print(f"Power transitions for {group_name} saved to {output_dir}/{group_name}_power_transitions.png")

    def generate_power_report(self, output_file='gpu_power_report.txt'):
        """Generate a comprehensive power draw report with recommendations"""
        data = self.load_data()
        metrics_df = self.compare_all_groups(output_dir='figures')
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("GPU POWER DRAW ANALYSIS REPORT\n")
            f.write("=============================\n\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-----------------\n")
            
            # Calculate overall statistics
            all_data = pd.concat([pd.concat(group_data.values()) for group_data in data.values() if group_data])
            
            f.write(f"Overall Average GPU Power Draw: {all_data['value'].mean():.2f}W\n")
            f.write(f"Maximum Observed Power Draw: {all_data['value'].max():.2f}W\n")
            f.write(f"Minimum Observed Power Draw: {all_data['value'].min():.2f}W\n")
            f.write(f"Standard Deviation: {all_data['value'].std():.2f}\n\n")
            
            # Calculate typical operating ranges
            percentiles = np.percentile(all_data['value'], [10, 25, 50, 75, 90])
            f.write(f"10th Percentile: {percentiles[0]:.2f}W\n")
            f.write(f"25th Percentile: {percentiles[1]:.2f}W\n")
            f.write(f"50th Percentile (Median): {percentiles[2]:.2f}W\n")
            f.write(f"75th Percentile: {percentiles[3]:.2f}W\n")
            f.write(f"90th Percentile: {percentiles[4]:.2f}W\n\n")
            
            # Group Summaries
            f.write("SERVER GROUP SUMMARIES\n")
            f.write("----------------------\n\n")
            
            for idx, row in metrics_df.iterrows():
                group = row['Group']
                f.write(f"{group}:\n")
                f.write(f"  Average Power Draw: {row['Avg_Power']:.2f}W\n")
                f.write(f"  Maximum Power Draw: {row['Max_Power']:.2f}W\n")
                f.write(f"  Minimum Power Draw: {row['Min_Power']:.2f}W\n")
                f.write(f"  Standard Deviation: {row['Std_Dev']:.2f}\n")
                f.write(f"  Number of Servers: {row['Servers_Count']}\n")
                f.write(f"  Hours Above 200W: {row['Hours_Above_200W']:.2f}\n")
                f.write(f"  Hours Above 250W: {row['Hours_Above_250W']:.2f}\n\n")
            
            # Identify most and least power-consuming groups
            most_power = metrics_df.iloc[0]['Group']
            least_power = metrics_df.iloc[-1]['Group']
            
            # Temporal Patterns
            f.write("TEMPORAL POWER PATTERNS\n")
            f.write("----------------------\n\n")
            
            # Calculate daily patterns (hour of day)
            hourly_power = all_data.groupby('hour')['value'].mean()
            peak_hour = hourly_power.idxmax()
            lowest_hour = hourly_power.idxmin()
            
            f.write(f"Peak Hour of the Day: {peak_hour}:00 (Average: {hourly_power[peak_hour]:.2f}W)\n")
            f.write(f"Lowest Hour of the Day: {lowest_hour}:00 (Average: {hourly_power[lowest_hour]:.2f}W)\n\n")
            
            # Calculate weekly patterns (day of week)
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            inv_day_map = {v: k for k, v in day_map.items()}
            
            all_data['day_num'] = all_data['day_of_week'].map(day_map)
            daily_power = all_data.groupby('day_of_week')['value'].mean()
            
            peak_day_num = all_data.groupby('day_num')['value'].mean().idxmax()
            lowest_day_num = all_data.groupby('day_num')['value'].mean().idxmin()
            
            peak_day = inv_day_map[peak_day_num]
            lowest_day = inv_day_map[lowest_day_num]
            
            f.write(f"Peak Day of the Week: {peak_day} (Average: {daily_power[peak_day]:.2f}W)\n")
            f.write(f"Lowest Day of the Week: {lowest_day} (Average: {daily_power[lowest_day]:.2f}W)\n\n")
            
            # Power Efficiency Analysis
            f.write("POWER EFFICIENCY ANALYSIS\n")
            f.write("-------------------------\n\n")
            
            # Calculate percentage of time in different power bands
            low_power_pct = (all_data['value'] < 100).mean() * 100
            med_power_pct = ((all_data['value'] >= 100) & (all_data['value'] < 200)).mean() * 100
            high_power_pct = ((all_data['value'] >= 200) & (all_data['value'] < 250)).mean() * 100
            very_high_power_pct = (all_data['value'] >= 250).mean() * 100
            
            f.write(f"Time in Low Power State (<100W): {low_power_pct:.2f}%\n")
            f.write(f"Time in Medium Power State (100-199W): {med_power_pct:.2f}%\n")
            f.write(f"Time in High Power State (200-249W): {high_power_pct:.2f}%\n")
            f.write(f"Time in Very High Power State (≥250W): {very_high_power_pct:.2f}%\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("--------------\n\n")
            
            # Analyze overall power draw
            overall_avg = all_data['value'].mean()
            if overall_avg < 120:
                f.write("1. Overall GPU power draw is low, suggesting GPUs are often idle or running light workloads. Consider workload consolidation or power management policies.\n")
            elif overall_avg > 220:
                f.write("1. Overall GPU power draw is very high. Review cooling systems and consider workload distribution to prevent potential thermal issues.\n")
            else:
                f.write("1. Overall GPU power draw is moderate. Continue monitoring, but current levels appear to be within expected operational range.\n")
            
            # Check for server group imbalances
            max_avg = metrics_df['Avg_Power'].max()
            min_avg = metrics_df['Avg_Power'].min()
            
            if max_avg - min_avg > 50:
                f.write(f"2. Significant power imbalance between server groups. {most_power} is drawing substantially more power than {least_power}, which may lead to cooling and performance variations.\n")
            
            # Check for temporal patterns
            if hourly_power.max() - hourly_power.min() > 50:
                f.write(f"3. Strong daily power pattern detected. Consider scheduling power-intensive workloads during {lowest_hour}:00 when overall power draw is lowest.\n")
            
            # Check for high power consumption
            high_power_groups = metrics_df[metrics_df['Avg_Power'] > 220]['Group'].tolist()
            if high_power_groups:
                f.write(f"4. The following groups show consistently high power draw and may benefit from workload optimization: {', '.join(high_power_groups)}.\n")
            
            # Check for power stability
            if all_data['value'].std() > 80:
                f.write("5. Large variations in power draw detected. This may indicate unstable workloads or power management issues. Consider implementing more consistent workload scheduling.\n")
            
            # Final notes
            f.write("\nNOTES:\n")
            f.write("- This report analyzes the power draw of GPUs across the server groups.\n")
            f.write("- Power efficiency can be improved by utilizing GPU processing capacity more consistently rather than in bursts.\n")
            f.write("- Consider comparing power draw with actual computational output to assess true efficiency.\n")
            f.write("- Modern GPUs generally operate most efficiently at moderate to high utilization levels with consistent workloads.\n")
        
        print(f"GPU Power report saved to {output_file}")

def main():
    """Main function"""
    # Configure paths
    excel_path = '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_当前GPU卡的PowerDraw_20250221_110550.xlsx'
    output_dir = 'output'
    figures_dir = 'figures'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = GPUPowerDrawAnalyzer(excel_path)
    
    # Run analyses
    print("Starting GPU Power Draw Analysis...")
    
    # Load data
    analyzer.load_data()
    
    # Generate temporal pattern analyses
    analyzer.compare_monthly_patterns(output_dir=figures_dir)
    analyzer.compare_weekly_patterns(output_dir=figures_dir)
    
    # Analyze daily patterns for each group
    for group in analyzer.server_groups.keys():
        analyzer.analyze_daily_patterns(group_name=group, output_dir=figures_dir)
    
    # Find high power periods
    analyzer.find_high_power_periods(
        threshold=220,  # Higher threshold for power analysis
        output_file=f'{output_dir}/high_power_periods.txt'
    )
    
    # Generate heatmaps
    for group in analyzer.server_groups.keys():
        analyzer.generate_heatmap(group_name=group, output_dir=figures_dir)
    
    # Compare all groups
    analyzer.compare_all_groups(output_dir=figures_dir)
    
    # Analyze power efficiency
    analyzer.analyze_power_efficiency(output_dir=figures_dir)
    
    # Generate comprehensive report
    analyzer.generate_power_report(
        output_file=f'{output_dir}/gpu_power_report.txt'
    )
    
    print("GPU Power Draw analysis completed successfully!")

if __name__ == "__main__":
    main()
