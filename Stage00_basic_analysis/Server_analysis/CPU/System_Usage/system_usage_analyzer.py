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

class SystemUsageAnalyzer:
    """System CPU Usage Analyzer"""
    
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
            'BIGMEM': [f'bigmen-{i}' for i in range(1, 7)]  # 6 nodes
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
            'BIGMEM': '#e377c2'     # pink
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
        """Compare monthly system usage patterns across all groups"""
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
            
        plt.title('Monthly Average System CPU Usage by Server Type', fontsize=18)
        plt.xlabel('Month', fontsize=14)
        plt.ylabel('Average CPU Usage (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set month format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_monthly_system_usage.png', dpi=300)
        plt.close()
        
        return monthly_avgs
    
    def compare_weekly_patterns(self, output_dir='figures'):
        """Compare weekly system usage patterns across all groups"""
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
            
        plt.title('Weekly Average System CPU Usage by Server Type', fontsize=18)
        plt.xlabel('Week', fontsize=14)
        plt.ylabel('Average CPU Usage (%)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set week format
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_weekly_system_usage.png', dpi=300)
        plt.close()
        
        return weekly_avgs

    def analyze_daily_patterns(self, output_dir='figures'):
        """Analyze daily system usage patterns for all server groups"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cached data
        all_data = self.load_all_data()
        
        # For each group, analyze the daily pattern
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            print(f"Analyzing daily patterns for {group_name}...")
            
            # Combine all servers' data
            all_values = []
            for server, df in group_data.items():
                df_copy = df.copy()
                df_copy['group'] = group_name
                df_copy['server'] = server
                # Extract hour from timestamp for daily pattern analysis
                df_copy['hour'] = df_copy['timestamp'].dt.hour
                all_values.append(df_copy)
            
            if not all_values:
                continue
                
            combined_df = pd.concat(all_values)
            
            # Group by hour and calculate average
            hourly_avg = combined_df.groupby('hour')['value'].mean()
            
            # Plot daily pattern
            plt.figure(figsize=(16, 8))
            plt.plot(hourly_avg.index, hourly_avg.values, 
                     marker='o', 
                     color=self.group_colors[group_name],
                     linewidth=2.5)
            
            plt.title(f'Daily System CPU Usage Pattern for {group_name}', fontsize=18)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Average CPU Usage (%)', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(0, 24), fontsize=12)
            plt.xlim(-0.5, 23.5)
            
            # Add horizontal lines for different thresholds
            plt.axhline(y=50, color='y', linestyle='--', label='50% Usage')
            plt.axhline(y=75, color='orange', linestyle='--', label='75% Usage')
            plt.axhline(y=90, color='r', linestyle='--', label='90% Usage')
            
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{group_name}_daily_pattern.png', dpi=300)
            plt.close()
    
    def find_high_usage_periods(self, threshold=80, duration_hours=2, output_dir='figures'):
        """Find periods of sustained high system usage"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cached data
        all_data = self.load_all_data()
        
        # Store results for all groups
        high_usage_results = {}
        
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            print(f"Finding high usage periods for {group_name}...")
            
            # Process each server in the group
            group_results = {}
            
            for server, df in group_data.items():
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
                
                # Flatten column hierarchy
                periods.columns = ['start_time', 'end_time', 'count', 'avg_usage', 'max_usage']
                
                # Calculate duration in hours
                periods['duration_hours'] = (periods['end_time'] - periods['start_time']).dt.total_seconds() / 3600
                
                # Filter for periods that meet the duration requirement
                long_periods = periods[periods['duration_hours'] >= duration_hours]
                
                if not long_periods.empty:
                    group_results[server] = long_periods
            
            if group_results:
                high_usage_results[group_name] = group_results
                
                # Plot summary of high usage for this group
                plt.figure(figsize=(16, 8))
                
                server_names = []
                durations = []
                max_usages = []
                
                for server, periods in group_results.items():
                    for _, period in periods.iterrows():
                        server_names.append(server)
                        durations.append(period['duration_hours'])
                        max_usages.append(period['max_usage'])
                
                if server_names:  # Check if there's data to plot
                    plt.scatter(durations, max_usages, c=self.group_colors[group_name], alpha=0.7)
                    
                    plt.title(f'High System Usage Periods for {group_name} (>{threshold}%)', fontsize=18)
                    plt.xlabel('Duration (Hours)', fontsize=14)
                    plt.ylabel('Maximum CPU Usage (%)', fontsize=14)
                    plt.grid(True, linestyle='--', alpha=0.5)
                    
                    plt.tight_layout()
                    plt.savefig(f'{output_dir}/{group_name}_high_usage_periods.png', dpi=300)
                    plt.close()
        
        return high_usage_results
    
    def generate_heatmap(self, output_dir='figures'):
        """Generate heatmap showing system usage patterns"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cached data
        all_data = self.load_all_data()
        
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            print(f"Generating heatmap for {group_name}...")
            
            # Combine all servers' data
            all_values = []
            for server, df in group_data.items():
                df_copy = df.copy()
                # Extract day of week and hour
                df_copy['day'] = df_copy['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
                df_copy['hour'] = df_copy['timestamp'].dt.hour
                all_values.append(df_copy)
            
            if not all_values:
                continue
                
            combined_df = pd.concat(all_values)
            
            # Create pivot table for heatmap: hours vs days of week
            heatmap_data = combined_df.pivot_table(
                values='value', 
                index='day', 
                columns='hour', 
                aggfunc='mean'
            )
            
            # Create heatmap
            plt.figure(figsize=(20, 10))
            ax = sns.heatmap(
                heatmap_data, 
                cmap='YlOrRd',
                annot=True, 
                fmt=".1f",
                linewidths=0.5,
                cbar_kws={'label': 'Average CPU Usage (%)'}
            )
            
            # Set labels
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ax.set_yticklabels(days)
            
            plt.title(f'System CPU Usage Heatmap for {group_name}', fontsize=18)
            plt.xlabel('Hour of Day', fontsize=14)
            plt.ylabel('Day of Week', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/{group_name}_heatmap.png', dpi=300)
            plt.close()
    
    def compare_all_groups(self, output_dir='figures'):
        """Generate comprehensive comparison of all server groups"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Use cached data
        all_data = self.load_all_data()
        
        # Calculate average usage across all server groups
        group_avgs = {}
        
        for group_name, group_data in all_data.items():
            if not group_data:
                continue
                
            # Combine all servers' data
            all_values = []
            for server, df in group_data.items():
                all_values.extend(df['value'].tolist())
            
            if all_values:
                group_avgs[group_name] = {
                    'mean': np.mean(all_values),
                    'median': np.median(all_values),
                    'std': np.std(all_values),
                    'min': np.min(all_values),
                    'max': np.max(all_values),
                    'servers': len(group_data)
                }
        
        if not group_avgs:
            print("No data available for comparison")
            return
            
        # Create bar chart for average system usage
        plt.figure(figsize=(18, 10))
        
        groups = list(group_avgs.keys())
        means = [group_avgs[g]['mean'] for g in groups]
        errors = [group_avgs[g]['std'] for g in groups]
        colors = [self.group_colors[g] for g in groups]
        
        bars = plt.bar(groups, means, yerr=errors, color=colors, alpha=0.8)
        
        # Add server count annotation above each bar
        for i, bar in enumerate(bars):
            server_count = group_avgs[groups[i]]['servers']
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + errors[i] + 2,
                f'{server_count} servers',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        plt.title('Average System CPU Usage Comparison Across Server Groups', fontsize=18)
        plt.xlabel('Server Group', fontsize=14)
        plt.ylabel('Average CPU Usage (%)', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/all_groups_comparison.png', dpi=300)
        plt.close()
        
        # Create detailed table with statistics
        stats_df = pd.DataFrame(group_avgs).T
        stats_df = stats_df[['servers', 'mean', 'median', 'std', 'min', 'max']]
        stats_df = stats_df.sort_values('mean', ascending=False)
        
        # Round values for better display
        stats_df = stats_df.round(2)
        
        # Save statistics to CSV
        stats_df.to_csv(f'{output_dir}/system_usage_statistics.csv')
        
        return stats_df


def main():
    """Main function to run the analysis"""
    # Get the script directory and construct paths relative to it
    script_dir = Path(__file__).parent.resolve()
    # Go up to 01_HPC_Research directory
    hpc_research_dir = script_dir.parent.parent.parent.parent

    # Set the Excel file path (in Stage00_HPC_raw_data directory)
    excel_file = hpc_research_dir / "Stage00_HPC_raw_data" / "prometheus_metrics_data_CPU使用率_系统使用率（%）_20250221_104358.xlsx"

    # Create output directory for figures (in the same directory as the script)
    output_dir = script_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize analyzer
    analyzer = SystemUsageAnalyzer(str(excel_file))
    
    # Run analyses
    print("\n1. Analyzing monthly patterns...")
    analyzer.compare_monthly_patterns(str(output_dir))

    print("\n2. Analyzing weekly patterns...")
    analyzer.compare_weekly_patterns(str(output_dir))

    print("\n3. Analyzing daily patterns...")
    analyzer.analyze_daily_patterns(str(output_dir))

    print("\n4. Finding high usage periods...")
    analyzer.find_high_usage_periods(threshold=80, duration_hours=2, output_dir=str(output_dir))

    print("\n5. Generating heatmaps...")
    analyzer.generate_heatmap(str(output_dir))

    print("\n6. Comparing all server groups...")
    stats = analyzer.compare_all_groups(str(output_dir))
    
    print("\nAnalysis complete! Results saved to:", output_dir)
    
    if stats is not None:
        print("\nSummary statistics:")
        print(stats)


if __name__ == "__main__":
    main()
