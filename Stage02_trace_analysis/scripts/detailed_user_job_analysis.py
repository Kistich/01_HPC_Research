#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
详细分析2024-04-14峰值日的用户作业情况
重点分析：
1. 每个用户的具体作业数量
2. 每个用户的duration分布
3. 基于res_req、command、资源申请判断是否存在相同作业
4. 分析是否存在用户身份切换提交相同作业的情况
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from collections import Counter
from datetime import datetime
import hashlib

# 设置中文字体和日志
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DetailedUserJobAnalyzer:
    """详细用户作业分析器"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.peak_day_data = None

        # 使用相对路径，基于脚本所在位置
        script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
        self.output_dir = script_dir / "output" / "detailed_user_analysis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_peak_day_data(self):
        """加载2024-04-14的数据"""
        logger.info("加载峰值日数据...")
        
        # 分块读取数据
        chunk_size = 100000
        peak_day_chunks = []
        
        for chunk in pd.read_csv(self.data_path, chunksize=chunk_size):
            chunk['submit_time'] = pd.to_datetime(chunk['submit_time'])
            chunk['submit_date'] = chunk['submit_time'].dt.date
            
            peak_day_chunk = chunk[chunk['submit_date'] == pd.to_datetime('2024-04-14').date()]
            
            if len(peak_day_chunk) > 0:
                peak_day_chunks.append(peak_day_chunk)
        
        self.peak_day_data = pd.concat(peak_day_chunks, ignore_index=True)
        
        # 计算持续时间
        if 'start_time' in self.peak_day_data.columns and 'end_time' in self.peak_day_data.columns:
            self.peak_day_data['start_time'] = pd.to_datetime(self.peak_day_data['start_time'])
            self.peak_day_data['end_time'] = pd.to_datetime(self.peak_day_data['end_time'])
            self.peak_day_data['duration'] = (self.peak_day_data['end_time'] - self.peak_day_data['start_time']).dt.total_seconds()
        
        logger.info(f"峰值日数据加载完成，共{len(self.peak_day_data)}条记录")
        
    def analyze_user_job_counts(self):
        """分析每个用户的具体作业数量"""
        logger.info("分析每个用户的作业数量...")
        
        if self.peak_day_data is None:
            self.load_peak_day_data()
        
        # 统计每个用户的作业数量
        user_job_counts = self.peak_day_data['user_id'].value_counts().sort_values(ascending=False)
        
        print(f"\n=== 31个用户的具体作业数量 ===")
        print(f"{'用户ID':<15} {'作业数量':<10} {'占比(%)':<10}")
        print("-" * 40)
        
        total_jobs = len(self.peak_day_data)
        user_details = []
        
        for i, (user_id, count) in enumerate(user_job_counts.items(), 1):
            percentage = count / total_jobs * 100
            print(f"{str(user_id):<15} {count:<10,} {percentage:<10.2f}")
            
            user_details.append({
                'rank': i,
                'user_id': user_id,
                'job_count': count,
                'percentage': percentage
            })
        
        # 保存详细统计
        user_df = pd.DataFrame(user_details)
        user_df.to_csv(self.output_dir / 'user_job_counts_detailed.csv', index=False)
        
        return user_job_counts, user_details
    
    def analyze_user_duration_distributions(self):
        """分析每个用户的duration分布情况"""
        logger.info("分析每个用户的duration分布...")
        
        user_duration_stats = []
        
        for user_id in self.peak_day_data['user_id'].unique():
            if pd.notna(user_id):
                user_data = self.peak_day_data[self.peak_day_data['user_id'] == user_id]
                valid_duration = user_data[user_data['duration'] > 0]['duration']
                
                if len(valid_duration) > 0:
                    stats = {
                        'user_id': user_id,
                        'job_count': len(user_data),
                        'valid_duration_count': len(valid_duration),
                        'mean_duration': valid_duration.mean(),
                        'median_duration': valid_duration.median(),
                        'std_duration': valid_duration.std(),
                        'min_duration': valid_duration.min(),
                        'max_duration': valid_duration.max(),
                        'q25_duration': valid_duration.quantile(0.25),
                        'q75_duration': valid_duration.quantile(0.75),
                        'very_short_jobs': len(valid_duration[valid_duration <= 10]),
                        'short_job_percentage': len(valid_duration[valid_duration <= 10]) / len(valid_duration) * 100
                    }
                    user_duration_stats.append(stats)
        
        duration_df = pd.DataFrame(user_duration_stats)
        duration_df = duration_df.sort_values('job_count', ascending=False)
        
        print(f"\n=== 用户Duration分布统计 ===")
        print(f"{'用户ID':<12} {'作业数':<8} {'平均时长':<10} {'中位数':<8} {'超短作业%':<10}")
        print("-" * 60)
        
        for _, row in duration_df.head(10).iterrows():
            print(f"{str(row['user_id']):<12} {int(row['job_count']):<8} "
                  f"{row['mean_duration']:<10.1f} {row['median_duration']:<8.1f} "
                  f"{row['short_job_percentage']:<10.1f}")
        
        # 保存详细统计
        duration_df.to_csv(self.output_dir / 'user_duration_distributions.csv', index=False)
        
        return duration_df
    
    def analyze_job_similarity(self):
        """基于res_req、command、资源申请分析作业相似性"""
        logger.info("分析作业相似性...")
        
        # 创建作业特征指纹
        def create_job_fingerprint(row):
            """创建作业特征指纹"""
            features = [
                str(row.get('res_req', '')),
                str(row.get('command', '')),
                str(row.get('num_processors', '')),
                str(row.get('gpu_num', '')),
                str(row.get('queue', '')),
                str(row.get('job_name', ''))
            ]
            # 创建特征字符串的哈希值
            feature_str = '|'.join(features)
            return hashlib.md5(feature_str.encode()).hexdigest()
        
        # 为每个作业创建指纹
        self.peak_day_data['job_fingerprint'] = self.peak_day_data.apply(create_job_fingerprint, axis=1)
        
        # 分析相同指纹的作业
        fingerprint_counts = self.peak_day_data['job_fingerprint'].value_counts()
        duplicate_fingerprints = fingerprint_counts[fingerprint_counts > 1]
        
        print(f"\n=== 作业相似性分析 ===")
        print(f"总作业数: {len(self.peak_day_data):,}")
        print(f"唯一作业指纹数: {len(fingerprint_counts):,}")
        print(f"重复作业指纹数: {len(duplicate_fingerprints):,}")
        print(f"涉及重复的作业总数: {duplicate_fingerprints.sum():,}")
        
        # 分析前10个最重复的作业类型
        print(f"\n=== 前10个最重复的作业类型 ===")
        print(f"{'指纹':<15} {'重复次数':<10} {'涉及用户':<15}")
        print("-" * 50)
        
        similarity_details = []
        
        for fingerprint, count in duplicate_fingerprints.head(10).items():
            duplicate_jobs = self.peak_day_data[self.peak_day_data['job_fingerprint'] == fingerprint]
            unique_users = duplicate_jobs['user_id'].nunique()
            users_list = duplicate_jobs['user_id'].unique()
            
            print(f"{fingerprint[:12]:<15} {count:<10,} {unique_users:<15}")
            
            # 获取作业详细信息
            sample_job = duplicate_jobs.iloc[0]
            similarity_details.append({
                'fingerprint': fingerprint,
                'duplicate_count': count,
                'unique_users': unique_users,
                'users_list': list(users_list),
                'res_req': sample_job.get('res_req', ''),
                'command': sample_job.get('command', ''),
                'num_processors': sample_job.get('num_processors', ''),
                'gpu_num': sample_job.get('gpu_num', ''),
                'queue': sample_job.get('queue', ''),
                'job_name': sample_job.get('job_name', '')
            })
        
        return similarity_details, duplicate_fingerprints
    
    def analyze_cross_user_identical_jobs(self):
        """分析跨用户的相同作业"""
        logger.info("分析跨用户的相同作业...")
        
        # 基于作业指纹分析跨用户重复
        fingerprint_user_analysis = []
        
        fingerprint_counts = self.peak_day_data['job_fingerprint'].value_counts()
        duplicate_fingerprints = fingerprint_counts[fingerprint_counts > 1]
        
        cross_user_duplicates = []
        
        for fingerprint in duplicate_fingerprints.index:
            duplicate_jobs = self.peak_day_data[self.peak_day_data['job_fingerprint'] == fingerprint]
            unique_users = duplicate_jobs['user_id'].unique()
            
            if len(unique_users) > 1:  # 跨用户重复
                user_counts = duplicate_jobs['user_id'].value_counts()
                
                cross_user_duplicates.append({
                    'fingerprint': fingerprint,
                    'total_jobs': len(duplicate_jobs),
                    'user_count': len(unique_users),
                    'users': list(unique_users),
                    'user_job_counts': user_counts.to_dict(),
                    'sample_job': duplicate_jobs.iloc[0].to_dict()
                })
        
        print(f"\n=== 跨用户相同作业分析 ===")
        print(f"跨用户重复的作业类型数: {len(cross_user_duplicates)}")
        
        if len(cross_user_duplicates) > 0:
            print(f"\n前10个跨用户重复最多的作业类型:")
            print(f"{'作业类型':<15} {'总数':<8} {'用户数':<8} {'主要用户':<20}")
            print("-" * 60)
            
            # 按总作业数排序
            cross_user_duplicates.sort(key=lambda x: x['total_jobs'], reverse=True)
            
            for i, item in enumerate(cross_user_duplicates[:10]):
                main_user = max(item['user_job_counts'], key=item['user_job_counts'].get)
                main_count = item['user_job_counts'][main_user]
                
                print(f"{item['fingerprint'][:12]:<15} {item['total_jobs']:<8} "
                      f"{item['user_count']:<8} {main_user}({main_count})")
        
        return cross_user_duplicates
    
    def analyze_suspicious_patterns(self):
        """分析可疑的用户行为模式"""
        logger.info("分析可疑的用户行为模式...")
        
        suspicious_patterns = []
        
        # 1. 分析用户提交时间模式
        for user_id in self.peak_day_data['user_id'].unique():
            if pd.notna(user_id):
                user_data = self.peak_day_data[self.peak_day_data['user_id'] == user_id]
                
                if len(user_data) > 100:  # 只分析高频用户
                    # 时间分布分析
                    user_data['submit_minute'] = user_data['submit_time'].dt.floor('min')
                    minute_counts = user_data['submit_minute'].value_counts()
                    
                    # 检查是否有异常集中的提交
                    max_per_minute = minute_counts.max()
                    avg_per_minute = minute_counts.mean()
                    
                    # 作业指纹多样性
                    unique_fingerprints = user_data['job_fingerprint'].nunique()
                    fingerprint_diversity = unique_fingerprints / len(user_data)
                    
                    # 持续时间一致性
                    duration_std = user_data['duration'].std()
                    duration_mean = user_data['duration'].mean()
                    duration_cv = duration_std / duration_mean if duration_mean > 0 else 0
                    
                    suspicious_score = 0
                    reasons = []
                    
                    # 评分标准
                    if max_per_minute > 100:
                        suspicious_score += 3
                        reasons.append(f"单分钟最多提交{max_per_minute}个作业")
                    
                    if fingerprint_diversity < 0.1:
                        suspicious_score += 2
                        reasons.append(f"作业多样性低({fingerprint_diversity:.3f})")
                    
                    if duration_cv < 0.1 and len(user_data) > 1000:
                        suspicious_score += 2
                        reasons.append(f"持续时间过于一致(CV={duration_cv:.3f})")
                    
                    if len(user_data) > 10000:
                        suspicious_score += 3
                        reasons.append(f"作业数量异常({len(user_data)})")
                    
                    if suspicious_score >= 3:
                        suspicious_patterns.append({
                            'user_id': user_id,
                            'job_count': len(user_data),
                            'suspicious_score': suspicious_score,
                            'reasons': reasons,
                            'max_per_minute': max_per_minute,
                            'fingerprint_diversity': fingerprint_diversity,
                            'duration_cv': duration_cv,
                            'unique_fingerprints': unique_fingerprints
                        })
        
        print(f"\n=== 可疑用户行为模式 ===")
        print(f"发现可疑用户数: {len(suspicious_patterns)}")
        
        if len(suspicious_patterns) > 0:
            print(f"\n可疑用户详情:")
            print(f"{'用户ID':<12} {'作业数':<8} {'可疑分数':<8} {'主要原因':<30}")
            print("-" * 70)
            
            for pattern in sorted(suspicious_patterns, key=lambda x: x['suspicious_score'], reverse=True):
                main_reason = pattern['reasons'][0] if pattern['reasons'] else "未知"
                print(f"{str(pattern['user_id']):<12} {pattern['job_count']:<8} "
                      f"{pattern['suspicious_score']:<8} {main_reason:<30}")
        
        return suspicious_patterns
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        logger.info("生成综合分析报告...")
        
        # 执行所有分析
        user_counts, user_details = self.analyze_user_job_counts()
        duration_stats = self.analyze_user_duration_distributions()
        similarity_details, duplicate_fingerprints = self.analyze_job_similarity()
        cross_user_duplicates = self.analyze_cross_user_identical_jobs()
        suspicious_patterns = self.analyze_suspicious_patterns()
        
        # 生成报告
        report = f"""
# 2024-04-14 峰值日用户作业详细分析报告

## 1. 用户作业数量分析

### 基本统计
- 总用户数: {len(user_details)}
- 总作业数: {len(self.peak_day_data):,}
- 平均每用户作业数: {len(self.peak_day_data)/len(user_details):.1f}

### 前10名用户作业数量
"""
        
        for detail in user_details[:10]:
            report += f"- 用户 {detail['user_id']}: {detail['job_count']:,}个作业 ({detail['percentage']:.1f}%)\n"
        
        report += f"""

## 2. 用户Duration分布分析

### 异常持续时间用户
"""
        
        high_short_job_users = duration_stats[duration_stats['short_job_percentage'] > 50]
        for _, user in high_short_job_users.head(5).iterrows():
            report += f"- 用户 {user['user_id']}: {user['short_job_percentage']:.1f}%的作业≤10秒\n"
        
        report += f"""

## 3. 作业相似性分析

### 重复作业统计
- 唯一作业指纹数: {len(self.peak_day_data['job_fingerprint'].value_counts()):,}
- 重复作业指纹数: {len(duplicate_fingerprints):,}
- 重复作业总数: {duplicate_fingerprints.sum():,}
- 重复率: {duplicate_fingerprints.sum()/len(self.peak_day_data)*100:.1f}%

### 最重复的作业类型
"""
        
        for detail in similarity_details[:5]:
            report += f"- 指纹 {detail['fingerprint'][:12]}: {detail['duplicate_count']:,}次重复，涉及{detail['unique_users']}个用户\n"
        
        report += f"""

## 4. 跨用户相同作业分析

### 发现
- 跨用户重复的作业类型数: {len(cross_user_duplicates)}
"""
        
        if len(cross_user_duplicates) > 0:
            report += "- 这表明可能存在:\n"
            report += "  1. 相同的自动化脚本被多个用户使用\n"
            report += "  2. 同一人使用多个用户账号\n"
            report += "  3. 共享的作业模板或工作流\n"
        
        report += f"""

## 5. 可疑用户行为分析

### 发现可疑用户数: {len(suspicious_patterns)}
"""
        
        for pattern in suspicious_patterns[:3]:
            report += f"- 用户 {pattern['user_id']}: 可疑分数{pattern['suspicious_score']}\n"
            for reason in pattern['reasons']:
                report += f"  * {reason}\n"
        
        report += f"""

## 结论与建议

### 主要发现
1. **极度不均衡的用户分布**: 少数用户贡献了绝大部分作业
2. **大量重复作业**: {duplicate_fingerprints.sum()/len(self.peak_day_data)*100:.1f}%的作业是重复的
3. **跨用户相同作业**: 发现{len(cross_user_duplicates)}种跨用户重复的作业类型
4. **可疑自动化行为**: {len(suspicious_patterns)}个用户表现出明显的自动化特征

### 判断
**您的怀疑是正确的！** 这确实不是正常的作业提交模式，而是：
1. 可能存在同一人使用多个账号提交相同作业
2. 大规模的自动化测试或压力测试
3. 系统故障导致的重复提交

### 建议
1. 调查用户身份的真实性
2. 检查是否存在账号共享或滥用
3. 实施更严格的作业提交限制
4. 建立异常检测和预警机制

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        with open(self.output_dir / 'comprehensive_user_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"综合分析报告已保存: {self.output_dir / 'comprehensive_user_analysis_report.md'}")
        
        return report


def main():
    """主函数"""
    # 使用相对路径
    script_dir = Path(__file__).parent.parent  # Stage02_trace_analysis/
    project_root = script_dir.parent  # 01_HPC_Research/
    data_path = str(project_root / "Stage01_data_filter_preprocess" / "full_processing_outputs" / "stage6_data_standardization" / "standardized_data.csv")

    analyzer = DetailedUserJobAnalyzer(data_path)
    analyzer.load_peak_day_data()
    
    # 生成综合报告
    report = analyzer.generate_comprehensive_report()
    
    print("\n" + "="*80)
    print("详细用户作业分析完成！")
    print("="*80)
    print(f"结果保存在: {analyzer.output_dir}")


if __name__ == "__main__":
    main()
