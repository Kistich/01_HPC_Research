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

class GPUClockSpeedAnalyzer:
    """Analyzer for GPU graphics clock speed data from Excel sheets"""
    
    def __init__(self, excel_path):
        """Initialize with path to Excel file containing clock speed data"""
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
        """Compare monthly clock speed patterns across all groups"""
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
        
        plt.title('Monthly GPU Clock Speed by Server Group', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average Clock Speed (MHz)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        plt.savefig(f'{output_dir}/all_groups_monthly_clockspeed.png', dpi=300)
        plt.close()
        
        print(f"Monthly pattern comparison saved to {output_dir}/all_groups_monthly_clockspeed.png")
    
    def compare_weekly_patterns(self, output_dir='figures'):
        """Compare weekly clock speed patterns across all groups"""
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
        
        plt.title('Weekly GPU Clock Speed by Server Group', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel('Average Clock Speed (MHz)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(7), days)
        
        plt.savefig(f'{output_dir}/all_groups_weekly_clockspeed.png', dpi=300)
        plt.close()
        
        print(f"Weekly pattern comparison saved to {output_dir}/all_groups_weekly_clockspeed.png")

    def analyze_daily_patterns(self, group_name=None, output_dir='figures'):
        """Analyze daily clock speed patterns for specific group or all groups"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                print(f"No data available for {group}")
                continue
            
            print(f"Analyzing daily patterns for {group}...")
            
            # Combine all servers in the group
            all_dfs = []
            for server, df in data[group].items():
                all_dfs.append(df)
            
            if not all_dfs:
                continue
                
            combined = pd.concat(all_dfs)
            
            # Group by hour and calculate average
            hourly = combined.groupby('hour')['value'].mean()
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            plt.plot(hourly.index, hourly.values, 
                    marker='o', linestyle='-', linewidth=2, 
                    color=self.group_colors.get(group, 'blue'))
            
            plt.title(f'Daily GPU Clock Speed Pattern for {group}', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Average Clock Speed (MHz)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(24))
            
            plt.savefig(f'{output_dir}/{group}_daily_clockspeed.png', dpi=300)
            plt.close()
            
            print(f"Daily pattern analysis saved to {output_dir}/{group}_daily_clockspeed.png")
    
    def find_high_clockspeed_periods(self, threshold=1500, group_name=None, output_file='high_clockspeed_periods.txt'):
        """Find periods where clock speed exceeds threshold consistently"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        # Open output file
        with open(output_file, 'w') as f:
            f.write(f"High Clock Speed Periods (> {threshold} MHz)\n")
            f.write("=" * 50 + "\n\n")
            
            for group in groups_to_process:
                if group not in data or not data[group]:
                    f.write(f"No data available for {group}\n\n")
                    continue
                
                # Process each server in the group
                for server, df in data[group].items():
                    # Sort by timestamp
                    df = df.sort_values('timestamp')
                    
                    # Find periods where value exceeds threshold
                    high_clock = df[df['value'] > threshold].copy()
                    
                    if high_clock.empty:
                        continue
                    
                    # Group consecutive high clock speed periods
                    high_clock['time_diff'] = high_clock['timestamp'].diff().dt.total_seconds()
                    high_clock['group'] = (high_clock['time_diff'] > 3600).cumsum()
                    
                    # Analyze each period
                    periods = []
                    for group_id, group_df in high_clock.groupby('group'):
                        if len(group_df) > 1:  # At least two points
                            duration = (group_df['timestamp'].max() - group_df['timestamp'].min()).total_seconds() / 3600  # hours
                            avg_clock = group_df['value'].mean()
                            max_clock = group_df['value'].max()
                            
                            if duration >= 0.5:  # At least 30 minutes
                                periods.append({
                                    'start': group_df['timestamp'].min(),
                                    'end': group_df['timestamp'].max(),
                                    'duration': duration,
                                    'avg_clock': avg_clock,
                                    'max_clock': max_clock
                                })
                    
                    if periods:
                        f.write(f"{server} (Group: {group}):\n")
                        for i, period in enumerate(periods, 1):
                            f.write(f"  Period {i}:\n")
                            f.write(f"    Start: {period['start']}\n")
                            f.write(f"    End: {period['end']}\n")
                            f.write(f"    Duration: {period['duration']:.2f} hours\n")
                            f.write(f"    Average Clock Speed: {period['avg_clock']:.2f} MHz\n")
                            f.write(f"    Maximum Clock Speed: {period['max_clock']:.2f} MHz\n\n")
        
        print(f"High clock speed periods saved to {output_file}")
    
    def generate_clockspeed_heatmap(self, group_name=None, output_dir='figures'):
        """Generate heatmaps of GPU clock speed by day of week and hour of day"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                print(f"No data available for {group}")
                continue
            
            print(f"Generating clock speed heatmap for {group}...")
            
            # Combine all servers in the group
            all_dfs = []
            for server, df in data[group].items():
                all_dfs.append(df)
            
            if not all_dfs:
                continue
                
            combined = pd.concat(all_dfs)
            
            # Create pivot table for heatmap data
            # Order days of week properly
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create the heatmap
            plt.figure(figsize=(12, 8))
            
            # Create pivot table: hour vs day of week
            heatmap_data = combined.pivot_table(
                values='value', 
                index='hour',
                columns='day_of_week',
                aggfunc='mean'
            )
            
            # Reorder columns to get days in correct sequence
            heatmap_data = heatmap_data.reindex(columns=days_order)
            
            # Create the heatmap
            sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt=".0f", linewidths=.5, cbar_kws={'label': 'Clock Speed (MHz)'})
            
            plt.title(f'GPU Clock Speed Heatmap for {group}', fontsize=16)
            plt.xlabel('Day of Week', fontsize=14)
            plt.ylabel('Hour of Day', fontsize=14)
            
            plt.savefig(f'{output_dir}/{group}_clockspeed_heatmap.png', dpi=300)
            plt.close()
            
            print(f"Clock speed heatmap saved to {output_dir}/{group}_clockspeed_heatmap.png")
    
    def analyze_clockspeed_efficiency(self, group_name=None, output_dir='figures'):
        """Analyze clock speed distribution and efficiency"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                print(f"No data available for {group}")
                continue
            
            print(f"Analyzing clock speed efficiency for {group}...")
            
            # Combine all servers in the group
            all_dfs = []
            for server, df in data[group].items():
                all_dfs.append(df)
            
            if not all_dfs:
                continue
                
            combined = pd.concat(all_dfs)
            
            # Create distribution histogram
            plt.figure(figsize=(12, 6))
            
            sns.histplot(combined['value'], bins=50, kde=True, color=self.group_colors.get(group, 'blue'))
            
            plt.title(f'GPU Clock Speed Distribution for {group}', fontsize=16)
            plt.xlabel('Clock Speed (MHz)', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.savefig(f'{output_dir}/{group}_clockspeed_distribution.png', dpi=300)
            plt.close()
            
            print(f"Clock speed distribution saved to {output_dir}/{group}_clockspeed_distribution.png")
            
            # Analyze transitions between different clock speed states
            combined = combined.sort_values('timestamp')
            combined['next_value'] = combined['value'].shift(-1)
            combined['change'] = combined['next_value'] - combined['value']
            
            # Remove rows with NaN
            combined = combined.dropna(subset=['change'])
            
            # Categorize changes
            combined['change_type'] = pd.cut(
                combined['change'],
                bins=[-float('inf'), -500, -100, 100, 500, float('inf')],
                labels=['Large Decrease', 'Small Decrease', 'Stable', 'Small Increase', 'Large Increase']
            )
            
            # Plot transitions
            plt.figure(figsize=(10, 6))
            
            sns.countplot(x='change_type', data=combined, palette='viridis')
            
            plt.title(f'GPU Clock Speed Transitions for {group}', fontsize=16)
            plt.xlabel('Clock Speed Change Type', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{group}_clockspeed_transitions.png', dpi=300)
            plt.close()
            
            print(f"Clock speed transitions analysis saved to {output_dir}/{group}_clockspeed_transitions.png")
    
    def compare_groups(self, output_dir='figures', output_file='output/clockspeed_group_comparison.txt'):
        """Compare clock speed metrics across all server groups"""
        data = self.load_data()
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Compute key metrics for each group
        metrics = {}
        for group, group_data in data.items():
            if not group_data:
                continue
                
            # Combine all servers in the group
            all_dfs = []
            for server, df in group_data.items():
                all_dfs.append(df)
            
            if not all_dfs:
                continue
                
            combined = pd.concat(all_dfs)
            
            # Calculate metrics
            metrics[group] = {
                'avg_clockspeed': combined['value'].mean(),
                'max_clockspeed': combined['value'].max(),
                'min_clockspeed': combined['value'].min(),
                'hours_above_1200': (combined['value'] > 1200).sum() / 60,  # Approx hours based on minute data
                'hours_above_1500': (combined['value'] > 1500).sum() / 60,
                'hours_above_1800': (combined['value'] > 1800).sum() / 60,
                'std_deviation': combined['value'].std(),
                'server_count': len(group_data)
            }
        
        # Write comparison to file
        with open(output_file, 'w') as f:
            f.write("GPU Clock Speed Comparison Across Server Groups\n")
            f.write("=" * 50 + "\n\n")
            
            for group, group_metrics in metrics.items():
                f.write(f"{group}:\n")
                f.write(f"  Server Count: {group_metrics['server_count']}\n")
                f.write(f"  Average Clock Speed: {group_metrics['avg_clockspeed']:.2f} MHz\n")
                f.write(f"  Maximum Clock Speed: {group_metrics['max_clockspeed']:.2f} MHz\n")
                f.write(f"  Minimum Clock Speed: {group_metrics['min_clockspeed']:.2f} MHz\n")
                f.write(f"  Standard Deviation: {group_metrics['std_deviation']:.2f} MHz\n")
                f.write(f"  Hours Above 1200 MHz: {group_metrics['hours_above_1200']:.2f}\n")
                f.write(f"  Hours Above 1500 MHz: {group_metrics['hours_above_1500']:.2f}\n")
                f.write(f"  Hours Above 1800 MHz: {group_metrics['hours_above_1800']:.2f}\n\n")
        
        print(f"Group comparison saved to {output_file}")
        
        # Create comparison visualizations
        
        # 1. Average clock speed comparison
        plt.figure(figsize=(10, 6))
        
        groups = list(metrics.keys())
        avg_values = [metrics[g]['avg_clockspeed'] for g in groups]
        
        bars = plt.bar(groups, avg_values, color=[self.group_colors.get(g, 'blue') for g in groups])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=12)
        
        plt.title('Average GPU Clock Speed by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Average Clock Speed (MHz)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.savefig(f'{output_dir}/group_comparison_avg_clockspeed.png', dpi=300)
        plt.close()
        
        # 2. Hours above threshold comparison
        plt.figure(figsize=(10, 6))
        
        thresholds = ['hours_above_1200', 'hours_above_1500', 'hours_above_1800']
        threshold_labels = ['> 1200 MHz', '> 1500 MHz', '> 1800 MHz']
        
        width = 0.25  # Width of bars
        x = np.arange(len(groups))
        
        for i, threshold in enumerate(thresholds):
            values = [metrics[g][threshold] for g in groups]
            plt.bar(x + (i - 1) * width, values, width, label=threshold_labels[i])
        
        plt.title('High Clock Speed Duration by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Hours', fontsize=14)
        plt.xticks(x, groups)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        plt.savefig(f'{output_dir}/group_comparison_high_clockspeed.png', dpi=300)
        plt.close()
        
        print(f"Group comparison visualizations saved to {output_dir}")
    
    def generate_report(self, output_file='output/gpu_clockspeed_report.txt'):
        """Generate a comprehensive report of GPU clock speed analysis"""
        data = self.load_data()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Compute overall statistics
        server_count = sum(len(group_data) for group_data in data.values())
        
        # Calculate overall metrics
        all_dfs = []
        for group_data in data.values():
            for df in group_data.values():
                all_dfs.append(df)
        
        if not all_dfs:
            print("No data available for report generation")
            return
            
        all_data = pd.concat(all_dfs)
        
        avg_clockspeed = all_data['value'].mean()
        max_clockspeed = all_data['value'].max()
        min_clockspeed = all_data['value'].min()
        
        # Find servers with highest average clock speed
        server_avgs = {}
        for group, group_data in data.items():
            for server, df in group_data.items():
                server_avgs[f"{server} ({group})"] = df['value'].mean()
        
        top_servers = sorted(server_avgs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Write report
        with open(output_file, 'w') as f:
            f.write("GPU Clock Speed Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Executive Summary\n")
            f.write("-" * 20 + "\n")
            f.write(f"This report analyzes GPU clock speed data from {server_count} servers across {len(data)} groups.\n\n")
            
            f.write("Key Findings\n")
            f.write("-" * 20 + "\n")
            f.write(f"1. Average Clock Speed: {avg_clockspeed:.2f} MHz\n")
            f.write(f"2. Maximum Clock Speed: {max_clockspeed:.2f} MHz\n")
            f.write(f"3. Minimum Clock Speed: {min_clockspeed:.2f} MHz\n\n")
            
            f.write("Top 5 Servers by Average Clock Speed\n")
            f.write("-" * 20 + "\n")
            for i, (server, avg) in enumerate(top_servers, 1):
                f.write(f"{i}. {server}: {avg:.2f} MHz\n")
            f.write("\n")
            
            f.write("Analysis by Server Group\n")
            f.write("-" * 20 + "\n")
            for group, group_data in data.items():
                if not group_data:
                    continue
                    
                # Combine all servers in the group
                group_dfs = []
                for df in group_data.values():
                    group_dfs.append(df)
                
                if not group_dfs:
                    continue
                    
                combined = pd.concat(group_dfs)
                
                avg_group = combined['value'].mean()
                max_group = combined['value'].max()
                hours_above_1500 = (combined['value'] > 1500).sum() / 60  # Approx hours
                
                f.write(f"\n{group} ({len(group_data)} servers):\n")
                f.write(f"  Average Clock Speed: {avg_group:.2f} MHz\n")
                f.write(f"  Maximum Clock Speed: {max_group:.2f} MHz\n")
                f.write(f"  Hours Above 1500 MHz: {hours_above_1500:.2f}\n")
            
            f.write("\nRecommendations\n")
            f.write("-" * 20 + "\n")
            
            # Generate recommendations based on the data
            if avg_clockspeed < 1000:
                f.write("1. Overall clock speeds are relatively low. Consider checking for thermal throttling or power limitations.\n")
            else:
                f.write("1. Overall clock speeds are within normal range.\n")
            
            # Add more recommendations based on patterns observed
            high_usage_groups = [group for group, data in server_avgs.items() if data > 1500]
            if high_usage_groups:
                f.write(f"2. The following servers have consistently high clock speeds and may be candidates for workload optimization: {', '.join(high_usage_groups[:3])}.\n")
            
            f.write("\nConclusion\n")
            f.write("-" * 20 + "\n")
            f.write("This analysis provides insights into GPU clock speed patterns across different server groups.\n")
            f.write("For detailed visualizations, refer to the figures directory which contains monthly, weekly, and daily pattern analysis,\n")
            f.write("as well as heatmaps showing clock speed distribution by day and hour.\n")
        
        print(f"Comprehensive report saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Path to the Excel file
    excel_path = '/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235.xlsx'
    
    # Create output directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    # Create analyzer instance
    analyzer = GPUClockSpeedAnalyzer(excel_path)
    
    # Run various analyses
    print("Starting GPU Clock Speed analysis...")
    
    # Load data (will be cached for subsequent analyses)
    analyzer.load_data()
    
    # Run analyses
    analyzer.compare_monthly_patterns()
    analyzer.compare_weekly_patterns()
    
    # Analyze daily patterns for each group
    for group in analyzer.server_groups.keys():
        analyzer.analyze_daily_patterns(group)
    
    # Generate heatmaps for each group
    for group in analyzer.server_groups.keys():
        analyzer.generate_clockspeed_heatmap(group)
    
    # Find high clock speed periods
    analyzer.find_high_clockspeed_periods(threshold=1500, output_file='output/high_clockspeed_periods.txt')
    
    # Analyze clock speed efficiency
    for group in analyzer.server_groups.keys():
        analyzer.analyze_clockspeed_efficiency(group)
    
    # Compare groups
    analyzer.compare_groups()
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    print("Analysis complete!")
