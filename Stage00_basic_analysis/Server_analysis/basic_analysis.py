import pandas as pd
import openpyxl
import argparse
import os
import time
import multiprocessing
from tabulate import tabulate
from tqdm import tqdm
import sys
from pathlib import Path

def analyze_sheet(args):
    """
    分析单个工作表（用于并行处理）
    """
    file_path, sheet_name = args
    try:
        # 仅读取少量行以快速分析
        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=1000)
        
        # 获取行数和列数 - 使用更可靠的方法
        # 获取真实的总行数（这需要读取整个工作表，但只读取第一列来提高效率）
        try:
            # 尝试仅读取第一列以获取总行数
            rows_df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[0])
            rows = len(rows_df)
        except Exception:
            # 如果失败，则用已经加载的数据估算行数
            rows = len(df)

        cols = len(df.columns)
        
        # 获取列名列表
        column_names = list(df.columns)
        
        # 获取数据类型分布
        dtypes = df.dtypes.value_counts().to_dict()
        dtype_str = ", ".join([f"{k}: {v}" for k, v in dtypes.items()])
        
        # 获取样本数据的非空值比例
        non_null_ratio = df.count().sum() / (len(df) * cols) * 100 if len(df) * cols > 0 else 0
        
        return {
            "工作表名": sheet_name,
            "行数": rows,
            "列数": cols,
            "数据完整度": f"{non_null_ratio:.2f}%",
            "数据类型": dtype_str,
            "列名前5个": ", ".join(column_names[:5]) + ("..." if len(column_names) > 5 else "")
        }
    except Exception as e:
        return {
            "工作表名": sheet_name,
            "行数": "错误",
            "列数": "错误",
            "数据完整度": "错误",
            "数据类型": f"读取错误: {str(e)}",
            "列名前5个": "无法获取"
        }

def analyze_excel(file_path):
    """
    分析Excel文件并返回基本统计信息，将结果保存到文件中
    
    参数:
        file_path: Excel文件的路径
    """
    start_time = time.time()
    
    # 创建输出字符串
    output = []
    output.append(f"\n分析文件: {os.path.basename(file_path)}")
    output.append(f"文件路径: {os.path.abspath(file_path)}")
    output.append("-" * 80)
    
    try:
        # 使用openpyxl仅获取工作表名称（read_only=True提高性能）
        print(f"正在读取工作表列表: {os.path.basename(file_path)}...")
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        sheet_names = workbook.sheetnames
        output.append(f"工作表总数: {len(sheet_names)}")
        
        # 关闭workbook释放内存
        workbook.close()
        
        # 确定最佳进程数（使用CPU核心数的一半）
        num_cores = multiprocessing.cpu_count()
        num_processes = max(1, min(num_cores // 2, len(sheet_names), 16))  # 最多16个进程，避免过度并行
        print(f"使用 {num_processes} 个并行进程进行分析（共有 {num_cores} 个CPU核心）")
        
        # 准备参数列表
        args_list = [(file_path, sheet_name) for sheet_name in sheet_names]
        
        # 使用多进程并行处理工作表
        sheet_stats = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用tqdm显示进度
            for result in tqdm(pool.imap(analyze_sheet, args_list), 
                              total=len(sheet_names), 
                              desc="分析工作表",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                sheet_stats.append(result)
        
        # 使用tabulate生成统计信息表格
        headers = sheet_stats[0].keys()
        table_data = [stat.values() for stat in sheet_stats]
        table_output = tabulate(table_data, headers=headers, tablefmt="grid")
        output.append(table_output)
        
        # 添加数据概要
        output.append("\n数据概要:")
        output.append(f"- 总工作表数: {len(sheet_names)}")
        total_rows = sum(stat['行数'] for stat in sheet_stats if isinstance(stat['行数'], int))
        output.append(f"- 总行数: {total_rows:,}")
        
        try:
            max_rows_sheet = max(sheet_stats, key=lambda x: x['行数'] if isinstance(x['行数'], int) else 0)
            output.append(f"- 最大行数工作表: {max_rows_sheet['工作表名']} ({max_rows_sheet['行数']:,}行)")
        except:
            output.append("- 无法确定最大行数工作表")
        
        # 添加执行时间
        elapsed_time = time.time() - start_time
        output.append(f"\n分析完成！总执行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        
        # 保存输出到文件
        output_dir = os.path.dirname(os.path.abspath(file_path))
        output_file = os.path.join(output_dir, "basic_info.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output))
        
        print(f"\n分析完成！详细结果已保存到: {output_file}")
        print(f"总执行时间: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
        
        return sheet_stats
    
    except Exception as e:
        error_msg = f"分析文件时发生错误: {str(e)}"
        print(error_msg)
        output.append(error_msg)
        
        # 尝试保存已有的输出到文件
        try:
            output_dir = os.path.dirname(os.path.abspath(file_path))
            output_file = os.path.join(output_dir, "basic_info.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(output))
            print(f"部分结果已保存到: {output_file}")
        except:
            print("无法保存结果到文件")
        
        return None

def main():
    parser = argparse.ArgumentParser(description='分析Excel文件的基本信息')
    parser.add_argument('file_path', help='Excel文件的路径')
    args = parser.parse_args()
    
    analyze_excel(args.file_path)

if __name__ == "__main__":
    main()
