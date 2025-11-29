import pandas as pd
import os

# 定义文件路径
input_file = '/data1/liuhongbin/job_analysis/job_prediction/2_Generation_analysis/second_generation_jobs.csv'
output_dir = '/data1/liuhongbin/job_analysis/job_analysis/User_analysis/User_behavior_analysis/results'
output_file = os.path.join(output_dir, 'job_type_analysis_results.txt')

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(input_file, engine='python')

# 作业类型分类函数
def classify_job_type(row):
    exec_hosts = str(row['exec_hosts']).lower()
    gpu_num = row['gpu_num']
    num_processors = row['num_processors']
    
    if 'gpu' not in exec_hosts:
        return 'CPU'
    else:
        if gpu_num > 0:
            if num_processors > gpu_num:
                return 'Hybrid'
            else:
                return 'GPU'
        else:
            return 'CPU'

# 应用分类函数
df['job_type'] = df.apply(classify_job_type, axis=1)

# 计算各类作业的数量和占比
total_jobs = len(df)
job_type_counts = df['job_type'].value_counts()
job_type_percentages = df['job_type'].value_counts(normalize=True) * 100

# 创建结果字符串
result_str = '作业类型分析结果\n'
result_str += '================\n\n'
result_str += f'总作业数量: {total_jobs}\n\n'
result_str += '各类作业数量:\n'
for job_type, count in job_type_counts.items():
    result_str += f'{job_type}: {count}\n'
result_str += '\n各类作业占比:\n'
for job_type, percentage in job_type_percentages.items():
    result_str += f'{job_type}: {percentage:.2f}%\n'

# 打印结果
print(result_str)

# 保存结果到文件
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(result_str)

print(f'结果已保存至 {output_file}')
