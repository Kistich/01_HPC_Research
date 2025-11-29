import pandas as pd
import os
from collections import defaultdict
import subprocess
import mmap
import io
import gc  # 添加gc模块用于主动内存管理

def analyze_csv(file_path):
    try:
        # 确定最佳的chunk大小
        # 假设每行平均100字节，预留2GB系统内存，每个chunk最多使用1GB内存
        available_mem = 1 * 1024 * 1024 * 1024  # 1GB for chunk
        avg_row_size = 100  # 假设每行平均100字节
        chunk_size = min(500000, int(available_mem / avg_row_size))
        print(f"使用chunk大小: {chunk_size}")

        # 读取CSV文件的前几行来获取列信息和数据类型
        print("分析数据类型...")
        df_sample = pd.read_csv(file_path, nrows=1000, low_memory=False)
        columns = df_sample.columns.tolist()
        
        # 优化数据类型以减少内存使用
        dtypes = {}
        column_categories = {
            'key_analysis': [],    # 关键分析列
            'resource': [],        # 资源相关列
            'status': [],         # 状态和类型列
            'host': [],           # 主机相关列
            'id_time': [],        # ID和时间列
            'other': []           # 其他列
        }
        
        # 初始化列分类
        for col in columns:
            if col in ['job_name', 'command', 'application', 'res_req']:
                column_categories['key_analysis'].append(col)
            elif col in ['max_mem', 'gpu_mem', 'num_processors', 'gpu_num']:
                column_categories['resource'].append(col)
            elif col in ['jstatus', 'exit_status', 'gpu_types', 'queue', 'job_status_str']:
                column_categories['status'].append(col)
            elif col in ['from_host', 'first_exec_host', 'exec_hosts']:
                column_categories['host'].append(col)
            elif col in ['job_id', 'user_id', 'submit_time', 'start_time', 'end_time']:
                column_categories['id_time'].append(col)
            else:
                column_categories['other'].append(col)
            
            if df_sample[col].dtype == 'object':
                dtypes[col] = 'category'
            elif df_sample[col].dtype == 'float64':
                dtypes[col] = 'float32'
            elif df_sample[col].dtype == 'int64':
                dtypes[col] = 'int32'
        
        print("\n=== 列名称和数据类型 ===")
        for col in columns:
            print(f"{col}: {dtypes.get(col, df_sample[col].dtype)}")
        
        # 初始化每列的值计数器和唯一值集合
        value_counts = {col: defaultdict(int) for col in columns}
        unique_values = {col: set() for col in columns}
        column_stats = {col: {'total_count': 0, 'null_count': 0} for col in columns}
        
        # 使用迭代器模式读取CSV
        print("\n正在分析数据分布，请稍候...")
        processed_rows = 0
        
        df_iterator = pd.read_csv(file_path, chunksize=chunk_size, dtype=dtypes)
        
        for chunk in df_iterator:
            chunk_size = len(chunk)
            processed_rows += chunk_size
            print(f"\r已处理行数: {processed_rows}", end="", flush=True)
            
            # 批量处理每一列
            for col in columns:
                column_stats[col]['total_count'] += len(chunk[col])
                column_stats[col]['null_count'] += chunk[col].isnull().sum()
                
                # 更新唯一值集合
                unique_values[col].update(chunk[col].dropna().unique())
                
                # 处理非空值的分布
                value_counts_temp = chunk[col].value_counts()
                for value, count in value_counts_temp.items():
                    value_counts[col][value] += count
            
            # 主动清理内存
            del chunk
            gc.collect()

        print(f"\n\n总行数: {processed_rows}")
        
        # 分析结果并生成建议的限制值
        suggested_limits = {}
        print("\n=== 列统计和建议的限制值 ===")
        for category, cols in column_categories.items():
            print(f"\n{category.upper()} 类别:")
            for col in cols:
                unique_count = len(unique_values[col])
                null_percentage = (column_stats[col]['null_count'] / column_stats[col]['total_count']) * 100
                
                # 根据不同类别和实际唯一值数量生成建议限制
                if category == 'key_analysis':
                    limit = None  # 保留所有唯一值
                elif category == 'status':
                    limit = None if unique_count < 100 else unique_count  # 状态列通常保留所有值
                elif category == 'resource':
                    limit = min(unique_count, 2000)  # 资源列保留较多唯一值
                elif category == 'host':
                    limit = min(unique_count, 1000)  # 主机列保留中等数量
                elif category == 'id_time':
                    limit = min(unique_count, 1000)  # ID和时间列可以限制
                else:
                    limit = min(unique_count, 500)  # 其他列使用默认限制
                
                suggested_limits[col] = limit
                
                print(f"{col}:")
                print(f"  唯一值数量: {unique_count}")
                print(f"  空值比例: {null_percentage:.2f}%")
                print(f"  建议的限制值: {limit if limit else '无限制'}")
                
                # 显示最常见的值的分布
                top_values = sorted(value_counts[col].items(), key=lambda x: (-x[1], x[0]))[:5]
                print("  最常见的值:")
                for value, count in top_values:
                    percentage = (count / column_stats[col]['total_count']) * 100
                    print(f"    {value}: {count} ({percentage:.2f}%)")
        
        # 保存建议的限制值到文件
        output_file = file_path.replace('.csv', '_column_limits.txt')
        with open(output_file, 'w') as f:
            f.write("# 建议的列唯一值限制配置\n")
            f.write("unique_value_limits = {\n")
            for category, cols in column_categories.items():
                f.write(f"    # {category.upper()} 类别\n")
                for col in cols:
                    limit = suggested_limits[col]
                    f.write(f"    '{col}': {limit if limit else 'None'},\n")
            f.write("}\n")
        
        print(f"\n建议的限制值配置已保存到: {output_file}")
        
        # 返回分析结果
        return {
            'columns': columns,
            'dtypes': dtypes,
            'value_counts': value_counts,
            'unique_counts': {col: len(values) for col, values in unique_values.items()},
            'column_stats': column_stats,
            'suggested_limits': suggested_limits,
            'total_rows': processed_rows
        }

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    file_path = "/mnt/raid/liuhongbin/job_analysis/job_analysis/raw_data/jobinfo_20250224_113534.csv"
    analyze_csv(file_path)