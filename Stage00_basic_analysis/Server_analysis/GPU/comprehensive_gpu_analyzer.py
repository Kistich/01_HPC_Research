#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive GPU Metrics Analyzer

This script analyzes multiple GPU metrics from various Excel files including:
- GPU Utilization
- Memory Utilization
- Graphics Clock Speed
- Memory Clock Speed
- SM Clock Speed
- Video Clock Speed
- Power Draw
- Memory Allocation
- Average Utilization across 8 GPUs

Each metric is analyzed separately and results are saved in their respective directories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
import time
import multiprocessing as mp
from tqdm import tqdm

class GPUMetricsAnalyzer:
    """Analyzer for various GPU metrics data from multiple Excel files"""
    
    def __init__(self):
        """Initialize the analyzer with data file paths and configurations"""
        # Base paths to Excel files
        self.excel_path_base = "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data"
        
        # Output directory path
        self.base_output_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/Server_analysis/GPU/Analysis_Results"
        
        # Excel files for each metric
        self.excel_files = {
            'GPUUtilization': "prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx",
            'MemoryUtilization': "prometheus_metrics_data_GPU使用率_MemoryUtilization_20250221_105207.xlsx",
            'GraphicsClockSpeed': "prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235.xlsx",
            'MemoryClockSpeed': "prometheus_metrics_data_GPU使用率_MemoryClockSpeed_20250221_113253.xlsx",
            'SMClockSpeed': "prometheus_metrics_data_GPU使用率_SMClockSpeed_20250221_112610.xlsx",
            'VideoClockSpeed': "prometheus_metrics_data_GPU使用率_VideoClockSpeed_20250221_111923.xlsx",
            'PowerDraw': "prometheus_metrics_data_GPU使用率_当前GPU卡的PowerDraw_20250221_110550.xlsx",
            'MemoryAllocation': "prometheus_metrics_data_GPU使用率_MemoryAllocation_20250221_105847.xlsx",
            'AverageUtilization': "prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx"
        }
        
        # Full paths to Excel files
        self.excel_paths = {metric: os.path.join(self.excel_path_base, filename) 
                            for metric, filename in self.excel_files.items()}
        
        # Store current data and metric
        self.current_data = None
        self.current_metric = None
        self.current_excel_file = None
        
        # Metric names for display
        self.metric_names = {
            'GPUUtilization': 'GPU Utilization',
            'MemoryUtilization': 'Memory Utilization',
            'GraphicsClockSpeed': 'Graphics Clock Speed',
            'MemoryClockSpeed': 'Memory Clock Speed',
            'SMClockSpeed': 'SM Clock Speed',
            'VideoClockSpeed': 'Video Clock Speed',
            'PowerDraw': 'Power Draw',
            'MemoryAllocation': 'Memory Allocation',
            'AverageUtilization': 'Average GPU Utilization'
        }
        
        # Units for each metric
        self.units = {
            'GPUUtilization': '%',
            'MemoryUtilization': '%',
            'GraphicsClockSpeed': 'MHz',
            'MemoryClockSpeed': 'MHz',
            'SMClockSpeed': 'MHz',
            'VideoClockSpeed': 'MHz',
            'PowerDraw': 'W',
            'MemoryAllocation': 'MB',
            'AverageUtilization': '%'
        }
        
        # Thresholds for each metric
        self.thresholds = {
            'GPUUtilization': 80,  # 80% utilization
            'MemoryUtilization': 80,  # 80% memory utilization
            'GraphicsClockSpeed': 1500,  # 1500 MHz
            'MemoryClockSpeed': 7000,  # 7000 MHz
            'SMClockSpeed': 1500,  # 1500 MHz
            'VideoClockSpeed': 1200,  # 1200 MHz
            'PowerDraw': 250,  # 250 Watts
            'MemoryAllocation': 8000,  # 8000 MB
            'AverageUtilization': 70  # 70% average utilization
        }
        
        # Server groups (will be populated from sheet names)
        self.server_groups = {
            'GPU1': [],
            'GPU2': [],
            'GPU3': []
        }
        
        # Colors for different server groups
        self.group_colors = {
            'GPU1': '#1f77b4',  # blue
            'GPU2': '#ff7f0e',  # orange
            'GPU3': '#2ca02c',  # green
        }
    
    def _determine_server_group(self, server_name):
        """确定服务器属于哪个组（基于名称模式）"""
        # 根据basic_info.txt中的工作表命名模式进行分组
        if server_name.startswith('gpu1-'):
            return 'GPU1'
        elif server_name.startswith('gpu2-'):
            return 'GPU2'
        elif server_name.startswith('gpu3-'):
            return 'GPU3'
        
    
    def _setup_output_directories(self, metric_type):
        """Setup output directories for analysis results"""
        # Use the Excel filename as the directory name
        if not self.current_excel_file:
            # Fallback to metric type if filename not available
            output_dir = os.path.join(self.base_output_path, metric_type)
        else:
            # Use Excel file name without the extension
            file_name = os.path.splitext(os.path.basename(self.current_excel_file))[0]
            output_dir = os.path.join(self.base_output_path, file_name)
        
        # Create subdirectories
        figures_dir = os.path.join(output_dir, "figures")
        reports_dir = os.path.join(output_dir, "reports")
        
        # Create directories if they don't exist
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)
        
        return output_dir, figures_dir, reports_dir
    
    def load_data(self, metric_type, excel_path=None):
        """Load data for a specific metric"""
        # Reset data
        self.current_data = {}
        self.current_metric = metric_type
        
        # Get the Excel file path
        if excel_path is None:
            excel_path = self.excel_paths.get(metric_type)
            if not excel_path:
                print(f"No Excel file defined for {metric_type}")
                return False
        
        self.current_excel_file = excel_path
        
        print(f"Loading data from {excel_path}")
        
        # Check if the Excel file exists
        if not os.path.isfile(excel_path):
            print(f"Excel file not found: {excel_path}")
            return False
        
        # Get all sheet names
        xls = pd.ExcelFile(excel_path)
        all_sheets = xls.sheet_names
        
        # Reset server groups for this metric
        for group in self.server_groups:
            self.server_groups[group] = []
        
        # Initialize data structure for server groups
        data = {group: {} for group in self.server_groups.keys()}
        
        # Process each sheet (server)
        for sheet in all_sheets:
            # Determine the server group
            group = self._determine_server_group(sheet)
            if group:
                # Load this sheet
                try:
                    df = self.load_sheet(excel_path, sheet)
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
        
        # Set current data
        self.current_data = data
        
        return True
    
    def load_sheet(self, excel_path, sheet_name):
        """Load a specific sheet from an Excel file"""
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        
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
    
    def load_data_for_metric(self, metric_type, excel_path=None):
        """加载特定指标类型的数据"""
        return self.load_data(metric_type, excel_path)

    def compare_monthly_patterns(self):
        """Compare monthly patterns across all groups for current metric"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        unit = self.units[metric_type]
        
        # Setup output directories
        _, figures_dir, _ = self._setup_output_directories(metric_type)
        
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
        
        plt.title(f'Monthly {metric_name} by Server Group', fontsize=16)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel(f'Average {metric_name} ({unit})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        
        output_file = os.path.join(figures_dir, f'all_groups_monthly_{metric_type.lower()}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Monthly pattern comparison saved to {output_file}")
    
    def compare_weekly_patterns(self):
        """Compare weekly patterns across all groups for current metric"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        unit = self.units[metric_type]
        
        # Setup output directories
        _, figures_dir, _ = self._setup_output_directories(metric_type)
        
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
        
        plt.title(f'Weekly {metric_name} by Server Group', fontsize=16)
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel(f'Average {metric_name} ({unit})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(range(7), days)
        
        output_file = os.path.join(figures_dir, f'all_groups_weekly_{metric_type.lower()}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Weekly pattern comparison saved to {output_file}")

    def analyze_daily_patterns(self, group_name=None):
        """Analyze daily patterns for specific group or all groups for current metric"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        unit = self.units[metric_type]
        
        # Setup output directories
        _, figures_dir, _ = self._setup_output_directories(metric_type)
        
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
            
            plt.title(f'Daily {metric_name} Pattern for {group}', fontsize=16)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel(f'Average {metric_name} ({unit})', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(24))
            
            output_file = os.path.join(figures_dir, f'{group}_daily_{metric_type.lower()}.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"Daily pattern analysis saved to {output_file}")
    
    def find_high_periods(self, group_name=None):
        """Find periods where metric exceeds threshold consistently"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        threshold = self.thresholds[metric_type]
        unit = self.units[metric_type]
        
        # Setup output directories
        _, _, output_dir = self._setup_output_directories(metric_type)
        
        # Output file
        output_file = os.path.join(output_dir, f'high_{metric_type.lower()}_periods.txt')
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        # Open output file
        with open(output_file, 'w') as f:
            f.write(f"High {metric_type} Periods (> {threshold} {unit})\n")
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
                    high_values = df[df['value'] > threshold].copy()
                    
                    if high_values.empty:
                        continue
                    
                    # Group consecutive high periods
                    high_values['time_diff'] = high_values['timestamp'].diff().dt.total_seconds()
                    high_values['group'] = (high_values['time_diff'] > 3600).cumsum()
                    
                    # Analyze each period
                    periods = []
                    for group_id, group_df in high_values.groupby('group'):
                        if len(group_df) > 1:  # At least two points
                            duration = (group_df['timestamp'].max() - group_df['timestamp'].min()).total_seconds() / 3600  # hours
                            avg_value = group_df['value'].mean()
                            max_value = group_df['value'].max()
                            
                            if duration >= 0.5:  # At least 30 minutes
                                periods.append({
                                    'start': group_df['timestamp'].min(),
                                    'end': group_df['timestamp'].max(),
                                    'duration': duration,
                                    'avg_value': avg_value,
                                    'max_value': max_value
                                })
                    
                    if periods:
                        f.write(f"{server} (Group: {group}):\n")
                        for i, period in enumerate(periods, 1):
                            f.write(f"  Period {i}:\n")
                            f.write(f"    Start: {period['start']}\n")
                            f.write(f"    End: {period['end']}\n")
                            f.write(f"    Duration: {period['duration']:.2f} hours\n")
                            f.write(f"    Average {metric_type}: {period['avg_value']:.2f} {unit}\n")
                            f.write(f"    Maximum {metric_type}: {period['max_value']:.2f} {unit}\n\n")
        
        print(f"High {metric_type} periods saved to {output_file}")
    
    def generate_heatmap(self, group_name=None):
        """Generate heatmaps of metric values by day of week and hour of day"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        
        # Setup output directories
        _, figures_dir, _ = self._setup_output_directories(metric_type)
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                print(f"No data available for {group}")
                continue
            
            print(f"Generating {metric_type} heatmap for {group}...")
            
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
            sns.heatmap(heatmap_data, cmap='YlOrRd', annot=False, fmt=".0f", linewidths=.5, 
                       cbar_kws={'label': f'{metric_name} ({self.units[metric_type]})'})
            
            plt.title(f'{metric_name} Heatmap for {group}', fontsize=16)
            plt.xlabel('Day of Week', fontsize=14)
            plt.ylabel('Hour of Day', fontsize=14)
            
            output_file = os.path.join(figures_dir, f'{group}_{metric_type.lower()}_heatmap.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"{metric_type} heatmap saved to {output_file}")

    def analyze_value_distribution(self, group_name=None):
        """Analyze metric value distribution and state transitions"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        unit = self.units[metric_type]
        
        # Setup output directories
        _, figures_dir, _ = self._setup_output_directories(metric_type)
        
        # Process specified group or all groups
        groups_to_process = [group_name] if group_name else self.server_groups.keys()
        
        for group in groups_to_process:
            if group not in data or not data[group]:
                print(f"No data available for {group}")
                continue
            
            print(f"Analyzing {metric_type} distribution for {group}...")
            
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
            
            plt.title(f'{metric_name} Distribution for {group}', fontsize=16)
            plt.xlabel(f'{metric_name} ({unit})', fontsize=14)
            plt.ylabel('Frequency', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            output_file = os.path.join(figures_dir, f'{group}_{metric_type.lower()}_distribution.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"{metric_type} distribution saved to {output_file}")
            
            # Analyze transitions between different value states
            combined = combined.sort_values('timestamp')
            combined['next_value'] = combined['value'].shift(-1)
            combined['change'] = combined['next_value'] - combined['value']
            
            # Remove rows with NaN
            combined = combined.dropna(subset=['change'])
            
            # Determine appropriate bins based on metric type
            if metric_type in ['GPUUtilization', 'MemoryUtilization', 'AverageUtilization']:
                # For percentage-based metrics
                bins = [-float('inf'), -30, -10, 10, 30, float('inf')]
                labels = ['Large Decrease', 'Small Decrease', 'Stable', 'Small Increase', 'Large Increase']
            elif metric_type in ['PowerDraw']:
                # For power metrics
                bins = [-float('inf'), -50, -10, 10, 50, float('inf')]
                labels = ['Large Decrease', 'Small Decrease', 'Stable', 'Small Increase', 'Large Increase']
            elif metric_type in ['MemoryAllocation']:
                # For memory allocation
                bins = [-float('inf'), -1000, -100, 100, 1000, float('inf')]
                labels = ['Large Decrease', 'Small Decrease', 'Stable', 'Small Increase', 'Large Increase']
            else:
                # For clock speed metrics
                bins = [-float('inf'), -200, -50, 50, 200, float('inf')]
                labels = ['Large Decrease', 'Small Decrease', 'Stable', 'Small Increase', 'Large Increase']
            
            # Categorize changes
            combined['change_type'] = pd.cut(
                combined['change'],
                bins=bins,
                labels=labels
            )
            
            # Plot transitions
            plt.figure(figsize=(10, 6))
            
            sns.countplot(x='change_type', hue='change_type', data=combined, palette='viridis', legend=False)
            
            plt.title(f'{metric_name} Transitions for {group}', fontsize=16)
            plt.xlabel(f'{metric_name} Change Type', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            output_file = os.path.join(figures_dir, f'{group}_{metric_type.lower()}_transitions.png')
            plt.savefig(output_file, dpi=300)
            plt.close()
            
            print(f"{metric_type} transitions analysis saved to {output_file}")
    
    def compare_groups(self):
        """Compare metric values across all server groups"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        unit = self.units[metric_type]
        threshold = self.thresholds[metric_type]
        
        # Setup output directories
        _, figures_dir, output_dir = self._setup_output_directories(metric_type)
        
        # Output file
        output_file = os.path.join(output_dir, f'{metric_type.lower()}_group_comparison.txt')
        
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
            
            # Calculate threshold-specific metrics based on metric type
            if metric_type in ['GPUUtilization', 'MemoryUtilization', 'AverageUtilization']:
                # For utilization metrics (percentage-based)
                t1 = threshold - 30
                t2 = threshold
                t3 = threshold + 10
            elif metric_type in ['PowerDraw']:
                # For power metrics
                t1 = threshold * 0.7
                t2 = threshold
                t3 = threshold * 1.2
            elif metric_type in ['MemoryAllocation']:
                # For memory allocation
                t1 = threshold * 0.7
                t2 = threshold
                t3 = threshold * 1.2
            else:
                # For clock speed metrics
                t1 = threshold * 0.8
                t2 = threshold
                t3 = threshold * 1.1
                
            # Calculate metrics
            metrics[group] = {
                'avg_value': combined['value'].mean(),
                'max_value': combined['value'].max(),
                'min_value': combined['value'].min(),
                'hours_above_t1': (combined['value'] > t1).sum() / 60,  # Approx hours based on minute data
                'hours_above_t2': (combined['value'] > t2).sum() / 60,
                'hours_above_t3': (combined['value'] > t3).sum() / 60,
                'std_deviation': combined['value'].std(),
                'server_count': len(group_data)
            }
        
        # Write comparison to file
        with open(output_file, 'w') as f:
            f.write(f"{metric_name} Comparison Across Server Groups\n")
            f.write("="*50 + "\n\n")
            
            for group, group_metrics in metrics.items():
                f.write(f"{group}:\n")
                f.write(f"  Server Count: {group_metrics['server_count']}\n")
                f.write(f"  Average {metric_name}: {group_metrics['avg_value']:.2f} {unit}\n")
                f.write(f"  Maximum {metric_name}: {group_metrics['max_value']:.2f} {unit}\n")
                f.write(f"  Minimum {metric_name}: {group_metrics['min_value']:.2f} {unit}\n")
                f.write(f"  Standard Deviation: {group_metrics['std_deviation']:.2f} {unit}\n")
                f.write(f"  Hours Above {t1} {unit}: {group_metrics['hours_above_t1']:.2f}\n")
                f.write(f"  Hours Above {t2} {unit}: {group_metrics['hours_above_t2']:.2f}\n")
                f.write(f"  Hours Above {t3} {unit}: {group_metrics['hours_above_t3']:.2f}\n\n")
        
        print(f"Group comparison saved to {output_file}")
        
        # Create comparison visualizations
        
        # 1. Average value comparison
        plt.figure(figsize=(10, 6))
        
        groups = list(metrics.keys())
        avg_values = [metrics[g]['avg_value'] for g in groups]
        
        bars = plt.bar(groups, avg_values, color=[self.group_colors.get(g, 'blue') for g in groups])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (max(avg_values) * 0.02),
                    f'{height:.0f}',
                    ha='center', va='bottom', fontsize=12)
        
        plt.title(f'Average {metric_name} by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel(f'Average {metric_name} ({unit})', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        output_file = os.path.join(figures_dir, f'group_comparison_avg_{metric_type.lower()}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        # 2. Hours above threshold comparison
        plt.figure(figsize=(10, 6))
        
        thresholds = ['hours_above_t1', 'hours_above_t2', 'hours_above_t3']
        threshold_labels = [f'> {t1} {unit}', f'> {t2} {unit}', f'> {t3} {unit}']
        
        width = 0.25  # Width of bars
        x = np.arange(len(groups))
        
        for i, threshold in enumerate(thresholds):
            values = [metrics[g][threshold] for g in groups]
            plt.bar(x + (i - 1) * width, values, width, label=threshold_labels[i])
        
        plt.title(f'High {metric_name} Duration by Server Group', fontsize=16)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Hours', fontsize=14)
        plt.xticks(x, groups)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        output_file = os.path.join(figures_dir, f'group_comparison_high_{metric_type.lower()}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Group comparison visualizations saved to {output_file}")
    
    def generate_report(self):
        """Generate a comprehensive report of the current metric analysis"""
        if self.current_metric is None or self.current_data is None:
            print("No data loaded. Please load data first.")
            return
            
        metric_type = self.current_metric
        data = self.current_data
        metric_name = self.metric_names[metric_type]
        unit = self.units[metric_type]
        
        # Setup output directories
        _, _, output_dir = self._setup_output_directories(metric_type)
        
        # Output file
        output_file = os.path.join(output_dir, f'{metric_type.lower()}_report.txt')
        
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
        
        avg_value = all_data['value'].mean()
        max_value = all_data['value'].max()
        min_value = all_data['value'].min()
        
        # Find servers with highest average values
        server_avgs = {}
        for group, group_data in data.items():
            for server, df in group_data.items():
                server_avgs[f"{server} ({group})"] = df['value'].mean()
        
        top_servers = sorted(server_avgs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Write report
        with open(output_file, 'w') as f:
            f.write(f"{metric_name} Analysis Report\n")
            f.write("="*50 + "\n\n")
            
            f.write("Executive Summary\n")
            f.write("-" * 20 + "\n")
            f.write(f"This report analyzes {metric_name} data from {server_count} servers across {len(data)} groups.\n\n")
            
            f.write("Key Findings\n")
            f.write("-" * 20 + "\n")
            f.write(f"1. Average {metric_name}: {avg_value:.2f} {unit}\n")
            f.write(f"2. Maximum {metric_name}: {max_value:.2f} {unit}\n")
            f.write(f"3. Minimum {metric_name}: {min_value:.2f} {unit}\n\n")
            
            f.write(f"Top 5 Servers by Average {metric_name}\n")
            f.write("-" * 20 + "\n")
            for i, (server, avg) in enumerate(top_servers, 1):
                f.write(f"{i}. {server}: {avg:.2f} {unit}\n")
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
                threshold = self.thresholds[metric_type]
                hours_above_threshold = (combined['value'] > threshold).sum() / 60  # Approx hours
                
                f.write(f"\n{group} ({len(group_data)} servers):\n")
                f.write(f"  Average {metric_name}: {avg_group:.2f} {unit}\n")
                f.write(f"  Maximum {metric_name}: {max_group:.2f} {unit}\n")
                f.write(f"  Hours Above {threshold} {unit}: {hours_above_threshold:.2f}\n")
            
            f.write("\nRecommendations\n")
            f.write("-" * 20 + "\n")
            
            # Generate metric-specific recommendations
            if metric_type in ['GPUUtilization', 'MemoryUtilization', 'AverageUtilization']:
                # For utilization metrics
                if avg_value < 40:
                    f.write("1. Overall utilization is relatively low. Consider consolidating workloads to fewer GPUs.\n")
                elif avg_value > 80:
                    f.write("1. Overall utilization is high. Consider monitoring for potential performance bottlenecks.\n")
                else:
                    f.write("1. Overall utilization is within a reasonable range.\n")
            elif metric_type in ['PowerDraw']:
                # For power metrics
                if avg_value > self.thresholds[metric_type] * 0.9:
                    f.write("1. Power consumption is consistently high. Consider workload optimization to reduce energy costs.\n")
                else:
                    f.write("1. Power consumption is within expected ranges.\n")
            elif metric_type in ['MemoryAllocation']:
                # For memory allocation
                if avg_value > self.thresholds[metric_type] * 0.9:
                    f.write("1. Memory allocation is consistently high. Monitor for potential memory issues.\n")
                else:
                    f.write("1. Memory allocation levels are appropriate for the workload.\n")
            else:
                # For clock speed metrics
                if avg_value < self.thresholds[metric_type] * 0.7:
                    f.write("1. Clock speeds are lower than expected. Check for thermal throttling or power limitations.\n")
                else:
                    f.write("1. Clock speeds are within expected ranges.\n")
            
            # Add more recommendations based on patterns observed
            high_usage_servers = [server for server, value in server_avgs.items() if value > self.thresholds[metric_type]]
            if high_usage_servers:
                f.write(f"2. The following servers have consistently high {metric_name.lower()} and may require attention: {', '.join(high_usage_servers[:3])}.\n")
            
            f.write("\nConclusion\n")
            f.write("-" * 20 + "\n")
            f.write(f"This analysis provides insights into {metric_name} patterns across different server groups.\n")
            f.write("For detailed visualizations, refer to the figures directory which contains monthly, weekly, and daily pattern analysis,\n")
            f.write(f"as well as heatmaps showing {metric_name.lower()} distribution by day and hour.\n")
        
        print(f"Comprehensive report saved to {output_file}")
        
    def run_all_analyses(self, metric_type, excel_path=None):
        """Run all analyses for a specific metric type"""
        
        # 如果提供了excel_path，使用它；否则，使用预定义的映射
        if excel_path is None:
            # Excel files for different metrics
            excel_files = {
                "GPUUtilization": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx",
                "MemoryUtilization": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_MemoryUtilization_20250221_105207.xlsx",
                "GraphicsClockSpeed": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235.xlsx",
                "MemoryClockSpeed": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_MemoryClockSpeed_20250221_113253.xlsx",
                "SMClockSpeed": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_SMClockSpeed_20250221_112610.xlsx",
                "VideoClockSpeed": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_VideoClockSpeed_20250221_111923.xlsx",
                "PowerDraw": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_当前GPU卡的PowerDraw_20250221_110550.xlsx",
                "MemoryAllocation": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_MemoryAllocation_20250221_105847.xlsx",
                "AverageUtilization": "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx"
            }
            
            # 获取对应的Excel文件路径
            if metric_type not in excel_files:
                print(f"Error: Unknown metric type '{metric_type}'")
                return False
            
            excel_path = excel_files[metric_type]
        
        # Get the Excel file name for logging purposes
        excel_file = os.path.basename(excel_path)
        print(f"\n{'='*80}\n Processing {metric_type} from {excel_file}\n{'='*80}")
        
        # Load data for this metric
        if not self.load_data_for_metric(metric_type, excel_path):
            return False
        
        # Run analyses
        self.compare_monthly_patterns()
        self.compare_weekly_patterns()
        
        # Analyze daily patterns for each group
        for group in self.server_groups.keys():
            if self.server_groups[group]:
                self.analyze_daily_patterns(group)
        
        # Generate heatmaps for each group
        for group in self.server_groups.keys():
            if self.server_groups[group]:
                self.generate_heatmap(group)
        
        # Find high value periods
        self.find_high_periods()
        
        # Analyze value distribution
        for group in self.server_groups.keys():
            if self.server_groups[group]:
                self.analyze_value_distribution(group)
        
        # Compare groups
        self.compare_groups()
        
        # Generate comprehensive report
        self.generate_report()
        
        print(f"Completed analysis for {metric_type} from {excel_file}")

# Main execution code
if __name__ == "__main__":
    import multiprocessing
    import time
    from tqdm import tqdm
    
    print("\n" + "="*100)
    print("\tComprehensive GPU Metrics Analysis")
    print("="*100 + "\n")
    
    # 确定CPU数量和要使用的进程数
    num_cpu = multiprocessing.cpu_count()
    num_processes = max(1, num_cpu // 14)  # 使用CPU核心数的约1/14，避免系统过载
    
    print(f"Using {num_processes} parallel processes for analysis (server has {num_cpu} CPU cores)")
    
    # Get the script directory and construct paths relative to it
    from pathlib import Path
    script_dir = Path(__file__).parent.resolve()
    # Go up to 01_HPC_Research directory
    hpc_research_dir = script_dir.parent.parent.parent
    raw_data_dir = hpc_research_dir / "Stage00_HPC_raw_data"

    # 要分析的所有指标及其Excel文件
    metrics_files = {
        "GPUUtilization": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_GPUUtilization_20250221_103520.xlsx"),
        "MemoryUtilization": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_MemoryUtilization_20250221_105207.xlsx"),
        "GraphicsClockSpeed": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_GraphicsClockSpeed_20250221_111235.xlsx"),
        "MemoryClockSpeed": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_MemoryClockSpeed_20250221_113253.xlsx"),
        "SMClockSpeed": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_SMClockSpeed_20250221_112610.xlsx"),
        "VideoClockSpeed": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_VideoClockSpeed_20250221_111923.xlsx"),
        "PowerDraw": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_当前GPU卡的PowerDraw_20250221_110550.xlsx"),
        "MemoryAllocation": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_MemoryAllocation_20250221_105847.xlsx"),
        "AverageUtilization": str(raw_data_dir / "prometheus_metrics_data_GPU使用率_8张GPU卡平均使用率_20250221_102538.xlsx")
    }
    
    def process_metric(metric_type, excel_path):
        """处理单个指标的函数，用于多进程调用"""
        try:
            analyzer = GPUMetricsAnalyzer()
            analyzer.run_all_analyses(metric_type, excel_path)
            return (metric_type, True, None)  # 成功
        except Exception as e:
            return (metric_type, False, str(e))  # 失败，返回错误信息
    
    # 创建进程池用于并行处理
    start_time = time.time()
    
    # 将任务添加到进度条
    print("\n进度概览:")
    results = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        tasks = [(metric, file) for metric, file in metrics_files.items()]
        total_tasks = len(tasks)
        
        # 使用tqdm创建进度条
        with tqdm(total=total_tasks, desc="整体进度", unit="文件") as pbar:
            # 提交所有任务
            for i, (metric, file) in enumerate(tasks):
                result = pool.apply_async(
                    process_metric, 
                    args=(metric, file),
                    callback=lambda _: pbar.update(1)
                )
                results.append(result)
            
            # 等待所有任务完成，同时更新进度条
            elapsed = 0
            while any(not r.ready() for r in results):
                time.sleep(0.5)
                elapsed += 0.5
                completed = sum(1 for r in results if r.ready())
                if completed > 0:
                    avg_time_per_task = elapsed / completed
                    remaining_tasks = total_tasks - completed
                    est_remaining = avg_time_per_task * remaining_tasks
                    pbar.set_postfix({
                        "已完成": f"{completed}/{total_tasks}",
                        "预计剩余时间": f"{int(est_remaining/60)}分{int(est_remaining%60)}秒"
                    })
    
    # 获取所有结果
    processed_results = [r.get() for r in results]
    
    # 分析结果
    successful = [r for r in processed_results if r[1]]
    failed = [r for r in processed_results if not r[1]]
    
    # 显示总结
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*100)
    print("\t执行摘要")
    print("="*100)
    print(f"总计处理文件: {len(metrics_files)}")
    print(f"成功: {len(successful)}")
    print(f"失败: {len(failed)}")
    print(f"总耗时: {int(total_time/60)}分{int(total_time%60)}秒 (平均每个文件 {int(total_time/len(metrics_files))}秒)")
    
    # 显示失败的任务详情
    if failed:
        print("\n失败的任务:")
        for metric, _, error in failed:
            print(f"- {metric}: {error}")
    
    print("\n" + "="*100)
    print("\tGPU分析完成")
    print("="*100)
