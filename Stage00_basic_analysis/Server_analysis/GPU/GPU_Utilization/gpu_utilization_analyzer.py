#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

class GPUUtilizationAnalyzer:
    """Analyzer for GPU utilization data from Excel sheets"""
    
    def __init__(self, excel_path):
        """Initialize with path to Excel file containing utilization data"""
        self.excel_path = excel_path
        self.data_cache = None
        
        # Define server groups and their colors for visualization
        self.server_groups = {
            'GPU1': [],
            'GPU2': [],
            'GPU3': []
        }
        
        # Define colors for each group
        self.group_colors = {
            'GPU1': '#1f77b4',  # blue
            'GPU2': '#ff7f0e',  # orange
            'GPU3': '#2ca02c',  # green
        }
    
    def load_data(self, force_reload=False):
        """Load data from all sheets and organize by server group"""
        if self.data_cache is not None and not force_reload:
            print("Using cached data...")
            return self.data_cache
        
        print(f"Loading data from {self.excel_path}...")
        start_time = datetime.now()
        
        # Get all sheet names
        xls = pd.ExcelFile(self.excel_path)
        all_sheets = xls.sheet_names
        
        # Initialize data structure for server groups
        data = {group: {} for group in self.server_groups.keys()}
        
        # Process each sheet (server)
        for sheet in all_sheets:
            # Determine the server group
            group = self._determine_server_group(sheet)
            if group:
                # Load this sheet
                try:
                    df = self.load_sheet(sheet)
                    if df is not None and not df.empty:
                        # Store in appropriate group
                        data[group][sheet] = df
                        # Keep track of servers in each group
                        if sheet not in self.server_groups[group]:
                            self.server_groups[group].append(sheet)
                except Exception as e:
                    print(f"  Error loading {sheet}: {str(e)}")
        
        # Print summary
        for group, servers in self.server_groups.items():
            print(f"  Loaded {len(servers)} servers for group {group}")
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        print(f"Data loading completed in {elapsed:.2f} seconds")
        
        # Cache data
        self.data_cache = data
        return data
    
    def load_sheet(self, sheet_name):
        """Load a specific sheet from the Excel file"""
        df = pd.read_excel(self.excel_path, sheet_name=sheet_name)
        
        # Ensure correct format (need timestamp and value)
        if 'timestamp' not in df.columns or 'value' not in df.columns:
            print(f"  Warning: Sheet {sheet_name} is missing required columns")
            return None
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add additional time-based columns for analysis
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        return df
    
    def _determine_server_group(self, server_name):
        """Determine which group a server belongs to based on name pattern"""
        # This is a simple example - adapt this to your actual naming scheme
        if re.search(r'gpu1|group1', server_name.lower()):
            return 'GPU1'
        elif re.search(r'gpu2|group2', server_name.lower()):
            return 'GPU2'
        elif re.search(r'gpu3|group3', server_name.lower()):
            return 'GPU3'
        
        # Default grouping based on first few characters if no pattern matches
        if server_name.startswith('a') or server_name.startswith('b'):
            return 'GPU1'
        elif server_name.startswith('c') or server_name.startswith('d'):
            return 'GPU2'
        else:
            return 'GPU3'
    
    def compare_monthly_patterns(self, output_dir='figures'):
        """Compare monthly utilization patterns across all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each group
        monthly_avgs = {}
        for group_name, group_data in data.items():
            print(f"Processing monthly patterns for {group_name}...")
            
            if not group_data:
                continue
                
            # Combine all servers in the group
            all_dfs = []
            for server, df in group_data.items():
                all_dfs.append(df)
            
            if not all_dfs:
                continue
                
            combined = pd.concat(all_dfs)
            
            # Group by month and calculate average
            monthly = combined.groupby('month')['value'].mean()
            monthly_avgs[group_name] = monthly
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        for group_name, monthly in monthly_avgs.items():
            plt.plot(monthly.index, monthly.values, 
                    marker='o', linestyle='-', linewidth=2, 
                    label=group_name, color=self.group_colors.get(group_name, 'blue'))
        
        plt.title('Monthly GPU Utilization by Server Group', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average Utilization (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.savefig(f'{output_dir}/all_groups_monthly_utilization.png', dpi=300)
        plt.close()
        
        print(f"Monthly pattern comparison saved to {output_dir}/all_groups_monthly_utilization.png")
    
    def compare_weekly_patterns(self, output_dir='figures'):
        """Compare weekly utilization patterns across all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each group
        weekly_avgs = {}
        for group_name, group_data in data.items():
            print(f"Processing weekly patterns for {group_name}...")
            
            if not group_data:
                continue
                
            # Combine all servers in the group
            all_dfs = []
            for server, df in group_data.items():
                all_dfs.append(df)
            
            if not all_dfs:
                continue
                
            combined = pd.concat(all_dfs)
            
            # Group by day of week and calculate average
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            
            combined['day_num'] = combined['day_of_week'].map(day_map)
            weekday = combined.groupby('day_num')['value'].mean()
            weekday = weekday.reindex(range(7))  # Ensure all days are included
            
            weekly_avgs[group_name] = weekday
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for group_name, weekly in weekly_avgs.items():
            plt.plot(weekly.index, weekly.values, 
                    marker='o', linestyle='-', linewidth=2, 
                    label=group_name, color=self.group_colors.get(group_name, 'blue'))
        
        plt.title('Weekly GPU Utilization by Server Group', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Average Utilization (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(7), days)
        
        plt.savefig(f'{output_dir}/all_groups_weekly_utilization.png', dpi=300)
        plt.close()
        
        print(f"Weekly pattern comparison saved to {output_dir}/all_groups_weekly_utilization.png")
    
    def analyze_daily_patterns(self, group_name=None, output_dir='figures'):
        """Analyze hourly utilization patterns for a specific group or all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If no group specified, use the first one
        if group_name is None:
            groups = list(self.server_groups.keys())
            if groups:
                group_name = groups[0]
            else:
                print("No server groups defined")
                return
        
        print(f"Analyzing daily patterns for {group_name}...")
        
        # Get data for the specified group
        group_data = data.get(group_name, {})
        
        if not group_data:
            print(f"No data for group {group_name}")
            return
        
        # Combine all servers in the group
        all_dfs = []
        for server, df in group_data.items():
            all_dfs.append(df)
        
        if not all_dfs:
            print(f"No data available for {group_name}")
            return
            
        combined = pd.concat(all_dfs)
        
        # Group by hour and calculate statistics
        hourly = combined.groupby('hour')['value'].agg(['mean', 'min', 'max', 'std'])
        
        # Create the plot
        plt.figure(figsize=(14, 7))
        
        # Plot mean with confidence interval
        plt.plot(hourly.index, hourly['mean'], 
                marker='o', linestyle='-', linewidth=2, 
                color=self.group_colors.get(group_name, 'blue'),
                label='Mean Utilization')
        
        # Add confidence interval
        plt.fill_between(hourly.index, 
                        hourly['mean'] - hourly['std'], 
                        hourly['mean'] + hourly['std'], 
                        alpha=0.2, color=self.group_colors.get(group_name, 'blue'),
                        label='±1 Std Dev')
        
        # Add min and max as dashed lines
        plt.plot(hourly.index, hourly['min'], 
                marker='x', linestyle='--', linewidth=1, 
                color='green', alpha=0.7, label='Minimum')
        
        plt.plot(hourly.index, hourly['max'], 
                marker='x', linestyle='--', linewidth=1, 
                color='red', alpha=0.7, label='Maximum')
        
        plt.title(f'Daily GPU Utilization Pattern for {group_name}', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Utilization (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(24))
        
        plt.savefig(f'{output_dir}/{group_name}_daily_utilization.png', dpi=300)
        plt.close()
        
        print(f"Daily pattern for {group_name} saved to {output_dir}/{group_name}_daily_utilization.png")
        
        return hourly
    
    def find_high_utilization_periods(self, threshold=80, group_name=None, output_file='high_utilization_periods.txt'):
        """Find periods of sustained high GPU utilization above the threshold"""
        data = self.load_data()
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process all groups or specific group
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        with open(output_file, 'w') as f:
            f.write(f"HIGH GPU UTILIZATION PERIODS (>{threshold}%)\n")
            f.write("="*50 + "\n\n")
            
            for group in groups_to_process:
                print(f"Finding high utilization periods for {group}...")
                f.write(f"GROUP: {group}\n")
                f.write("-"*40 + "\n\n")
                
                group_data = data.get(group, {})
                
                if not group_data:
                    f.write(f"No data available for {group}\n\n")
                    continue
                
                # Process each server in the group
                for server, df in group_data.items():
                    # Sort by timestamp
                    df = df.sort_values('timestamp')
                    
                    # Find periods where value exceeds threshold
                    high_util = df[df['value'] > threshold].copy()
                    
                    if high_util.empty:
                        continue
                    
                    # Group consecutive high utilization periods
                    high_util['time_diff'] = high_util['timestamp'].diff().dt.total_seconds()
                    high_util['group'] = (high_util['time_diff'] > 3600).cumsum()
                    
                    # Analyze each period
                    periods = []
                    for group_id, group_df in high_util.groupby('group'):
                        if len(group_df) > 1:  # At least two points
                            duration = (group_df['timestamp'].max() - group_df['timestamp'].min()).total_seconds() / 3600  # hours
                            avg_util = group_df['value'].mean()
                            max_util = group_df['value'].max()
                            
                            if duration >= 0.5:  # At least 30 minutes
                                periods.append({
                                    'start': group_df['timestamp'].min(),
                                    'end': group_df['timestamp'].max(),
                                    'duration_hours': duration,
                                    'avg_utilization': avg_util,
                                    'max_utilization': max_util
                                })
                    
                    if periods:
                        f.write(f"Server: {server}\n")
                        
                        # Sort by duration
                        periods.sort(key=lambda x: x['duration_hours'], reverse=True)
                        
                        for i, period in enumerate(periods):
                            f.write(f"  Period {i+1}:\n")
                            f.write(f"    Start: {period['start'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"    End:   {period['end'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write(f"    Duration: {period['duration_hours']:.2f} hours\n")
                            f.write(f"    Avg Utilization: {period['avg_utilization']:.2f}%\n")
                            f.write(f"    Max Utilization: {period['max_utilization']:.2f}%\n")
                            f.write("\n")
                
                f.write("\n")
        
        print(f"High utilization periods saved to {output_file}")
    
    def generate_heatmap(self, group_name=None, output_dir='figures'):
        """Generate heatmaps of GPU utilization by day of week and hour of day"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If no group specified, use the first one
        if group_name is None:
            groups = list(self.server_groups.keys())
            if groups:
                group_name = groups[0]
            else:
                print("No server groups defined")
                return
        
        print(f"Generating utilization heatmap for {group_name}...")
        
        # Get data for the specified group
        group_data = data.get(group_name, {})
        
        if not group_data:
            print(f"No data for group {group_name}")
            return
        
        # Combine all servers in the group
        all_dfs = []
        for server, df in group_data.items():
            all_dfs.append(df)
        
        if not all_dfs:
            print(f"No data available for {group_name}")
            return
            
        combined = pd.concat(all_dfs)
        
        # Map day names to numeric values for proper ordering
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        combined['day_num'] = combined['day_of_week'].map(day_map)
        
        # Create a pivot table for the heatmap
        heatmap_data = combined.pivot_table(
            values='value', 
            index='day_num',
            columns='hour',
            aggfunc='mean'
        )
        
        # Reindex to ensure we have all days and hours
        heatmap_data = heatmap_data.reindex(range(7))
        
        # Create heatmap
        plt.figure(figsize=(14, 7))
        
        ax = sns.heatmap(
            heatmap_data, 
            cmap='YlOrRd',
            linewidths=0.5,
            vmin=0, 
            vmax=100,  # Assuming 0-100% utilization
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Utilization (%)'}
        )
        
        # Set appropriate labels
        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        ax.set_yticklabels(day_labels, rotation=0)
        ax.set_xticklabels(range(24), rotation=0)  # Hours of the day
        
        plt.title(f'GPU Utilization Heatmap for {group_name}', fontsize=16)
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Day of Week', fontsize=14)
        
        plt.savefig(f'{output_dir}/{group_name}_utilization_heatmap.png', dpi=300)
        plt.close()
        
        print(f"Utilization heatmap for {group_name} saved to {output_dir}/{group_name}_utilization_heatmap.png")
    
    def compare_all_groups(self, output_dir='figures'):
        """Compare key metrics across all server groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate key metrics for each group
        metrics = {
            'Group': [],
            'Avg_Utilization': [],
            'Max_Utilization': [],
            'Min_Utilization': [],
            'Std_Dev': [],
            'Servers_Count': [],
            'Hours_Above_50pct': [],
            'Hours_Above_80pct': []
        }
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Calculating utilization metrics for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Add to metrics
            metrics['Group'].append(group_name)
            metrics['Avg_Utilization'].append(combined_df['value'].mean())
            metrics['Max_Utilization'].append(combined_df['value'].max())
            metrics['Min_Utilization'].append(combined_df['value'].min())
            metrics['Std_Dev'].append(combined_df['value'].std())
            metrics['Servers_Count'].append(len(group_data))
            
            # Calculate hours above thresholds
            hours_above_50 = combined_df[combined_df['value'] > 50].shape[0] / len(group_data)
            hours_above_80 = combined_df[combined_df['value'] > 80].shape[0] / len(group_data)
            
            metrics['Hours_Above_50pct'].append(hours_above_50)
            metrics['Hours_Above_80pct'].append(hours_above_80)
        
        # Create a DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        # Sort by average utilization
        metrics_df = metrics_df.sort_values('Avg_Utilization', ascending=False)
        
        # Save to CSV
        metrics_df.to_csv(f'{output_dir}/group_utilization_comparison.csv', index=False)
        
        # Create bar charts for key metrics
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df['Group'], metrics_df['Avg_Utilization'], 
               color=[self.group_colors.get(g, '#1f77b4') for g in metrics_df['Group']])
        
        plt.title('Average GPU Utilization by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Average Utilization (%)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(f'{output_dir}/group_comparison_avg_utilization.png', dpi=300)
        plt.close()
        
        # Create a second chart for high utilization hours
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.35
        
        plt.bar(x - width/2, metrics_df['Hours_Above_50pct'], width, label='>50%',
               color='orange')
        plt.bar(x + width/2, metrics_df['Hours_Above_80pct'], width, label='>80%',
               color='red')
        
        plt.title('Hours of High GPU Utilization by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Hours', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(x, metrics_df['Group'])
        plt.legend()
        
        plt.savefig(f'{output_dir}/group_comparison_high_utilization.png', dpi=300)
        plt.close()
        
        print(f"Group utilization comparison metrics saved to {output_dir}/group_utilization_comparison.csv")
        print(f"Group utilization comparison charts saved to {output_dir}/")
        
        return metrics_df
    
    def analyze_utilization_efficiency(self, output_dir='figures'):
        """Analyze utilization efficiency by examining utilization distribution"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for group_name, group_data in data.items():
            if not group_data:
                continue
                
            print(f"Analyzing utilization efficiency for {group_name}...")
            
            # Combine all servers in the group
            combined_df = pd.concat(group_data.values())
            
            # Create histogram of utilization values
            plt.figure(figsize=(12, 6))
            sns.histplot(combined_df['value'], bins=50, kde=True, color=self.group_colors.get(group_name, '#1f77b4'))
            
            plt.title(f'GPU Utilization Distribution for {group_name}', fontsize=16)
            plt.xlabel('Utilization (%)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add vertical lines for typical thresholds
            plt.axvline(x=20, color='green', linestyle='--', label='Low Utilization (20%)')
            plt.axvline(x=50, color='orange', linestyle='--', label='Medium Utilization (50%)')
            plt.axvline(x=80, color='red', linestyle='--', label='High Utilization (80%)')
            
            plt.legend()
            
            plt.savefig(f'{output_dir}/{group_name}_utilization_distribution.png', dpi=300)
            plt.close()
            
            print(f"Utilization distribution for {group_name} saved to {output_dir}/{group_name}_utilization_distribution.png")
            
            # Analyze utilization state transitions
            if len(group_data) > 0:
                # Pick the first server for transition analysis (as an example)
                server_name = list(group_data.keys())[0]
                server_df = group_data[server_name].copy()
                
                # Sort by timestamp
                server_df = server_df.sort_values('timestamp')
                
                # Calculate utilization change between consecutive measurements
                server_df['utilization_change'] = server_df['value'].diff()
                
                # Plot utilization changes
                plt.figure(figsize=(15, 6))
                plt.hist(server_df['utilization_change'].dropna(), bins=100, alpha=0.7)
                
                plt.title(f'GPU Utilization Transitions for {server_name}', fontsize=16)
                plt.xlabel('Utilization Change Between Consecutive Measurements (%)', fontsize=14)
                plt.ylabel('Frequency', fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.savefig(f'{output_dir}/{group_name}_utilization_transitions.png', dpi=300)
                plt.close()
                
                print(f"Utilization transitions for {group_name} saved to {output_dir}/{group_name}_utilization_transitions.png")
    
    def generate_utilization_report(self, output_file='gpu_utilization_report.txt'):
        """Generate a comprehensive utilization report with recommendations"""
        data = self.load_data()
        metrics_df = self.compare_all_groups(output_dir='figures')
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("GPU UTILIZATION ANALYSIS REPORT\n")
            f.write("===============================\n\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-----------------\n")
            
            # Calculate overall statistics
            all_data = pd.concat([pd.concat(group_data.values()) for group_data in data.values() if group_data])
            
            f.write(f"Overall Average GPU Utilization: {all_data['value'].mean():.2f}%\n")
            f.write(f"Maximum Observed Utilization: {all_data['value'].max():.2f}%\n")
            f.write(f"Minimum Observed Utilization: {all_data['value'].min():.2f}%\n")
            f.write(f"Standard Deviation: {all_data['value'].std():.2f}\n\n")
            
            # Calculate typical operating ranges
            percentiles = np.percentile(all_data['value'], [10, 25, 50, 75, 90])
            f.write(f"10th Percentile: {percentiles[0]:.2f}%\n")
            f.write(f"25th Percentile: {percentiles[1]:.2f}%\n")
            f.write(f"50th Percentile (Median): {percentiles[2]:.2f}%\n")
            f.write(f"75th Percentile: {percentiles[3]:.2f}%\n")
            f.write(f"90th Percentile: {percentiles[4]:.2f}%\n\n")
            
            # Group Summaries
            f.write("SERVER GROUP SUMMARIES\n")
            f.write("----------------------\n\n")
            
            for idx, row in metrics_df.iterrows():
                group = row['Group']
                f.write(f"{group}:\n")
                f.write(f"  Average Utilization: {row['Avg_Utilization']:.2f}%\n")
                f.write(f"  Maximum Utilization: {row['Max_Utilization']:.2f}%\n")
                f.write(f"  Minimum Utilization: {row['Min_Utilization']:.2f}%\n")
                f.write(f"  Standard Deviation: {row['Std_Dev']:.2f}\n")
                f.write(f"  Number of Servers: {row['Servers_Count']}\n")
                f.write(f"  Hours Above 50% Utilization: {row['Hours_Above_50pct']:.2f}\n")
                f.write(f"  Hours Above 80% Utilization: {row['Hours_Above_80pct']:.2f}\n\n")
            
            # Identify most and least utilized groups
            most_utilized = metrics_df.iloc[0]['Group']
            least_utilized = metrics_df.iloc[-1]['Group']
            
            # Temporal Patterns
            f.write("TEMPORAL UTILIZATION PATTERNS\n")
            f.write("----------------------------\n\n")
            
            # Calculate daily patterns (hour of day)
            hourly_util = all_data.groupby('hour')['value'].mean()
            peak_hour = hourly_util.idxmax()
            lowest_hour = hourly_util.idxmin()
            
            f.write(f"Peak Hour of the Day: {peak_hour}:00 (Average: {hourly_util[peak_hour]:.2f}%)\n")
            f.write(f"Lowest Hour of the Day: {lowest_hour}:00 (Average: {hourly_util[lowest_hour]:.2f}%)\n\n")
            
            # Calculate weekly patterns (day of week)
            day_map = {
                'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                'Friday': 4, 'Saturday': 5, 'Sunday': 6
            }
            inv_day_map = {v: k for k, v in day_map.items()}
            
            all_data['day_num'] = all_data['day_of_week'].map(day_map)
            daily_util = all_data.groupby('day_of_week')['value'].mean()
            
            peak_day_num = all_data.groupby('day_num')['value'].mean().idxmax()
            lowest_day_num = all_data.groupby('day_num')['value'].mean().idxmin()
            
            peak_day = inv_day_map[peak_day_num]
            lowest_day = inv_day_map[lowest_day_num]
            
            f.write(f"Peak Day of the Week: {peak_day} (Average: {daily_util[peak_day]:.2f}%)\n")
            f.write(f"Lowest Day of the Week: {lowest_day} (Average: {daily_util[lowest_day]:.2f}%)\n\n")
            
            # Utilization Efficiency Analysis
            f.write("UTILIZATION EFFICIENCY ANALYSIS\n")
            f.write("-------------------------------\n\n")
            
            # Calculate percentage of time in different utilization bands
            low_util_pct = (all_data['value'] < 20).mean() * 100
            med_util_pct = ((all_data['value'] >= 20) & (all_data['value'] < 50)).mean() * 100
            high_util_pct = ((all_data['value'] >= 50) & (all_data['value'] < 80)).mean() * 100
            very_high_util_pct = (all_data['value'] >= 80).mean() * 100
            
            f.write(f"Time in Low Utilization State (<20%): {low_util_pct:.2f}%\n")
            f.write(f"Time in Medium Utilization State (20-49%): {med_util_pct:.2f}%\n")
            f.write(f"Time in High Utilization State (50-79%): {high_util_pct:.2f}%\n")
            f.write(f"Time in Very High Utilization State (≥80%): {very_high_util_pct:.2f}%\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("--------------\n\n")
            
            # Analyze overall utilization
            overall_avg = all_data['value'].mean()
            if overall_avg < 30:
                f.write("1. Overall GPU utilization is low, suggesting GPUs are often idle. Consider consolidating workloads onto fewer GPUs or implementing an auto-scaling solution to optimize resource usage.\n")
            elif overall_avg > 70:
                f.write("1. Overall GPU utilization is very high. Consider adding more GPU resources or redistributing workloads to prevent potential performance bottlenecks.\n")
            else:
                f.write("1. Overall GPU utilization is moderate. Continue monitoring, but current levels appear to be within an optimal operational range.\n")
            
            # Check for server group imbalances
            max_avg = metrics_df['Avg_Utilization'].max()
            min_avg = metrics_df['Avg_Utilization'].min()
            
            if max_avg - min_avg > 30:
                f.write(f"2. Significant utilization imbalance between server groups. {most_utilized} is substantially more utilized than {least_utilized}, which may lead to performance variations and inefficient resource allocation.\n")
            
            # Check for temporal patterns
            if hourly_util.max() - hourly_util.min() > 30:
                f.write(f"3. Strong daily utilization pattern detected. Consider scheduling intensive workloads during {lowest_hour}:00 when overall utilization is lowest.\n")
            
            # Check for underutilization
            low_util_groups = metrics_df[metrics_df['Avg_Utilization'] < 20]['Group'].tolist()
            if low_util_groups:
                f.write(f"4. The following groups show consistently low utilization and may benefit from workload consolidation: {', '.join(low_util_groups)}.\n")
            
            # Check for overutilization
            high_util_groups = metrics_df[metrics_df['Avg_Utilization'] > 80]['Group'].tolist()
            if high_util_groups:
                f.write(f"5. The following groups show consistently high utilization and may need additional resources: {', '.join(high_util_groups)}.\n")
            
            # Check for utilization stability
            if all_data['value'].std() > 30:
                f.write("6. Large variations in utilization detected. This suggests bursty workloads, which often lead to inefficient resource usage. Consider implementing more consistent workload scheduling or batch processing.\n")
            
            # Final notes
            f.write("\nNOTES:\n")
            f.write("- This report analyzes the utilization of GPUs across the server groups.\n")
            f.write("- Optimal GPU utilization typically falls between 50-80%. Too low suggests wasted resources, while too high can lead to performance bottlenecks.\n")
            f.write("- Consider comparing utilization with actual job completion times to assess true efficiency.\n")
            f.write("- For better resource allocation, consider implementing dynamic scheduling of GPU workloads based on the patterns identified in this report.\n")
        
        print(f"GPU Utilization report saved to {output_file}")

def main():
    """Main function"""
    # Configure paths
    excel_path = '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx'
    output_dir = 'output'
    figures_dir = 'figures'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = GPUUtilizationAnalyzer(excel_path)
    
    # Run analyses
    print("Starting GPU Utilization Analysis...")
    
    # Load data
    analyzer.load_data()
    
    # Generate temporal pattern analyses
    analyzer.compare_monthly_patterns(output_dir=figures_dir)
    analyzer.compare_weekly_patterns(output_dir=figures_dir)
    
    # Analyze daily patterns for each group
    for group in analyzer.server_groups.keys():
        analyzer.analyze_daily_patterns(group_name=group, output_dir=figures_dir)
    
    # Find high utilization periods
    analyzer.find_high_utilization_periods(
        threshold=80,  # High utilization threshold at 80%
        output_file=f'{output_dir}/high_utilization_periods.txt'
    )
    
    # Generate heatmaps
    for group in analyzer.server_groups.keys():
        analyzer.generate_heatmap(group_name=group, output_dir=figures_dir)
    
    # Compare all groups
    analyzer.compare_all_groups(output_dir=figures_dir)
    
    # Analyze utilization efficiency
    analyzer.analyze_utilization_efficiency(output_dir=figures_dir)
    
    # Generate comprehensive report
    analyzer.generate_utilization_report(
        output_file=f'{output_dir}/gpu_utilization_report.txt'
    )
    
    print("GPU Utilization analysis completed successfully!")

if __name__ == "__main__":
    main()
