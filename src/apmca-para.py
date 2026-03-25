import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from collections import deque
import math
import warnings
import concurrent.futures
import threading
from threading import Lock
import time
import psutil
import os
import random

warnings.filterwarnings('ignore')

def set_global_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)  # 让 python 的哈希也固定
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 让 cudnn 也尽量确定性（如果你用到了 GPU）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # 没装 torch 就忽略
        pass
class APMCAAlgorithmParallel:
    def __init__(self, k=3, e=2, S=2, variance_threshold=0.1, max_workers=None):
        """
        初始化并行APMCA算法

        Parameters:
        k: k-匿名性参数
        e: 敏感属性差异阈值
        S: 每个子集的目标聚类数
        variance_threshold: 方差阈值
        max_workers: 最大线程数，None表示自动检测
        """
        self.k = k
        self.e = e
        self.S = S
        self.variance_threshold = variance_threshold

        # 自动检测最优线程数
        if max_workers is None:
            # 考虑CPU核心数和内存限制
            cpu_count = os.cpu_count()
            available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
            # 根据内存调整线程数（每个线程约需要1GB内存）
            memory_based_workers = max(1, int(available_memory // 2))
            self.max_workers = min(cpu_count, memory_based_workers, 8)  # 最大限制为8个线程
        else:
            self.max_workers = max_workers

        print(f"初始化并行APMCA算法，使用 {self.max_workers} 个线程")

        self.anonymized_data = []
        self.final_clusters = []
        self.lock = Lock()  # 线程锁用于保护共享资源

    def calculate_sensitive_difference(self, data, sensitive_attr='y'):
        """计算敏感属性差异度"""
        if len(data) == 0:
            return 0
        return data[sensitive_attr].nunique()

    def calculate_variance(self, data, numeric_columns=None):
        """计算数据集方差"""
        if numeric_columns is None:
            numeric_columns = self._get_numeric_columns(data)

        if len(data) == 0 or len(numeric_columns) == 0:
            return 0

        total_variance = 0
        for col in numeric_columns:
            if col in data.columns:
                variance = data[col].var()
                if not np.isnan(variance):
                    total_variance += variance

        return total_variance

    def _get_numeric_columns(self, data):
        # 只把真正连续数值当 numeric（按你的 QI）
        numeric_candidates = ['age', 'duration', 'campaign']
        return [
            c for c in numeric_candidates
            if c in data.columns and pd.api.types.is_numeric_dtype(data[c])
        ]

    def select_attribute_for_perturbation(self, data, variance_info, numeric_columns):
        """基于方差选择划分属性"""
        max_variance = -1
        selected_attribute = None

        for col in numeric_columns:
            if col in data.columns:
                col_variance = data[col].var()
                if not np.isnan(col_variance) and col_variance > max_variance:
                    max_variance = col_variance
                    selected_attribute = col

        return selected_attribute if selected_attribute else self.randomly_select_attribute(data, numeric_columns)

    def randomly_select_attribute(self, data, numeric_columns):
        """随机选择划分属性"""
        if len(numeric_columns) == 0:
            return None
        return np.random.choice(numeric_columns)

    def calculate_median(self, data, attribute):
        """计算属性中位数"""
        if attribute not in data.columns:
            return 0
        return data[attribute].median()

    def split_subset(self, data, attribute, median):
        """划分数据集"""
        if attribute not in data.columns:
            return data, pd.DataFrame()

        subset1 = data[data[attribute] <= median].copy()
        subset2 = data[data[attribute] > median].copy()

        return subset1, subset2

    def parallel_mondrian_processing(self, initial_data, sensitive_attr='y'):
        """
        并行Mondrian预处理
        """
        print("开始并行Mondrian预处理...")

        numeric_columns = self._get_numeric_columns(initial_data)
        results = deque()
        total_subsets_processed = 0

        def process_subset(subset, depth=0):
            """处理单个子集的函数"""
            nonlocal total_subsets_processed

            if depth > 20:  # 防止递归过深
                return [subset]

            n = len(subset)
            sensitive_diff = self.calculate_sensitive_difference(subset, sensitive_attr)

            # 情况1: 子集大小合适且满足e-差异性
            if self.k <= n <= 2 * self.k and sensitive_diff >= self.e:
                return [subset]

            # 情况2: 子集太大，需要划分
            elif n > 2 * self.k:
                variance = self.calculate_variance(subset, numeric_columns)

                if variance > self.variance_threshold:
                    attribute = self.select_attribute_for_perturbation(subset, variance, numeric_columns)
                else:
                    attribute = self.randomly_select_attribute(subset, numeric_columns)

                if attribute is None:
                    return [subset]

                median = self.calculate_median(subset, attribute)
                subset1, subset2 = self.split_subset(subset, attribute, median)

                results = []
                if len(subset1) > 0:
                    results.extend(process_subset(subset1, depth + 1))
                if len(subset2) > 0:
                    results.extend(process_subset(subset2, depth + 1))

                return results

            # 情况3: 子集大小合适但不满足e-差异性，暂时保留
            elif n >= self.k:
                return [subset]

            # 情况4: 子集太小
            else:
                return [subset] if n > 0 else []

        # 使用线程池并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 初始任务
            future = executor.submit(process_subset, initial_data)
            all_futures = [future]
            all_results = []

            # 收集所有结果
            while all_futures:
                done, not_done = concurrent.futures.wait(all_futures, timeout=1,
                                                         return_when=concurrent.futures.FIRST_COMPLETED)

                for future in done:
                    try:
                        result = future.result()
                        if result:
                            all_results.extend(result)
                            total_subsets_processed += len(result)
                    except Exception as e:
                        print(f"子集处理错误: {e}")

                all_futures = list(not_done)

                # 显示进度
                if total_subsets_processed % 100 == 0:
                    print(f"已处理 {total_subsets_processed} 个子集...")

        print(f"并行Mondrian预处理完成，生成 {len(all_results)} 个子集")
        return all_results

    def determine_num_clusters(self, subset, S):
        """确定聚类数量"""
        n = len(subset)
        if n <= 3 * self.k:
            return max(1, min(S, n // self.k))
        else:
            return min(S * 2, n // self.k)

    def k_means_clustering(self, data, num_clusters, numeric_columns):
        """K-means聚类"""
        if len(data) < num_clusters:
            return [data]

        if len(data) == 0:
            return []

        clustering_data = data[numeric_columns].values

        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(clustering_data)

            result_clusters = []
            for cluster_id in range(num_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if len(cluster_indices) > 0:
                    cluster_data = data.iloc[cluster_indices].copy()
                    result_clusters.append(cluster_data)

            return result_clusters
        except Exception as e:
            print(f"K-means聚类出错: {e}")
            return [data]

    def parallel_clustering(self, subsets, numeric_columns, sensitive_attr='y'):
        """
        并行聚类处理
        """
        print("开始并行聚类处理...")

        def process_single_subset(subset):
            """处理单个子集的聚类"""
            if len(subset) < self.k:
                # 小子集直接返回
                clusters = [subset]
                generalized_clusters = [self.generalize_cluster(cluster, [], sensitive_attr)
                                        for cluster in clusters if len(cluster) > 0]
                return generalized_clusters, clusters
            else:
                # 确定聚类数量并聚类
                num_clusters = self.determine_num_clusters(subset, self.S)
                clusters = self.k_means_clustering(subset, num_clusters, numeric_columns)

                # 泛化每个簇
                # generalized_clusters = []
                # valid_clusters = []
                # for cluster in clusters:
                #     if len(cluster) >= self.k:  # 确保满足k-匿名性
                #         generalized = self.generalize_cluster(cluster, [], sensitive_attr)
                #         generalized_clusters.append(generalized)
                #         valid_clusters.append(cluster)
                #
                # return generalized_clusters, valid_clusters
                # --- PATCH: 不丢记录：将 <k 的小簇合并进最近的大簇 ---
                big_clusters = [c for c in clusters if len(c) >= self.k]
                small_clusters = [c for c in clusters if 0 < len(c) < self.k]

                # 极端情况：如果全部都是小簇（或 KMeans 异常），直接把整个 subset 当成一个簇
                # （通常 Mondrian 子集大小 >= k，所以这样仍满足 k）
                if len(big_clusters) == 0:
                    big_clusters = [subset]

                # 预先计算大簇的“质心”（用数值特征均值）
                def _centroid(df):
                    # 注意：numeric_columns 是你传进来的用于聚类的列（如 age/duration/campaign）
                    return df[numeric_columns].mean().to_numpy(dtype=float)

                centroids = [_centroid(c) for c in big_clusters]

                # 把每个小簇合并到最近的大簇
                for sc in small_clusters:
                    sc_cent = _centroid(sc)

                    # 找最近的大簇（欧氏距离）
                    dists = [float(((cent - sc_cent) ** 2).sum()) for cent in centroids]
                    j = int(min(range(len(dists)), key=lambda i: dists[i]))

                    # 合并
                    big_clusters[j] = pd.concat([big_clusters[j], sc], axis=0)
                    # 更新该大簇质心（避免后续合并误差累积）
                    centroids[j] = _centroid(big_clusters[j])

                # 现在 big_clusters 覆盖了 subset 的所有记录，且每簇 >= k
                generalized_clusters = [self.generalize_cluster(c, [], sensitive_attr) for c in big_clusters if
                                        len(c) > 0]
                valid_clusters = big_clusters

                return generalized_clusters, valid_clusters
                # --- END PATCH ---

        all_generalized = []
        all_clusters = []
        completed = 0
        total = len(subsets)

        # 使用线程池并行处理所有子集
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_subset = {executor.submit(process_single_subset, subset): subset
                                for subset in subsets}

            # 处理完成的任务
            for future in concurrent.futures.as_completed(future_to_subset):
                try:
                    generalized_clusters, clusters = future.result()
                    with self.lock:
                        all_generalized.extend(generalized_clusters)
                        all_clusters.extend(clusters)
                    completed += 1

                    if completed % 50 == 0:
                        print(f"聚类进度: {completed}/{total} ({completed / total * 100:.1f}%)")

                except Exception as e:
                    print(f"子集聚类处理错误: {e}")
                    completed += 1

        print(f"并行聚类完成，生成 {len(all_generalized)} 个聚类")
        return all_generalized, all_clusters

    def generalize_cluster(self, cluster_data, quasi_identifiers, sensitive_attr='y'):
        """泛化聚类数据"""
        if len(cluster_data) == 0:
            return {}

        generalized_record = {}

        # 处理数值型属性
        numeric_columns = self._get_numeric_columns(cluster_data)
        for col in numeric_columns:
            min_val = cluster_data[col].min()
            max_val = cluster_data[col].max()
            if min_val == max_val:
                generalized_record[col] = f"[{min_val}]"
            else:
                generalized_record[col] = f"[{min_val},{max_val}]"

        # 处理分类属性（按你的 QI）
        categorical_columns = ['marital_status', 'education', 'contact']
        categorical_columns = [col for col in categorical_columns if col in cluster_data.columns]

        for col in categorical_columns:
            unique_values = cluster_data[col].unique()
            if len(unique_values) == 1:
                generalized_record[col] = str(unique_values[0])
            else:
                values_list = sorted([str(v).strip().strip("'\"") for v in unique_values])
                generalized_record[col] = '\\'.join(values_list)

        generalized_record['cluster_size'] = len(cluster_data)
        # 敏感属性统计
        if sensitive_attr in cluster_data.columns:
            sensitive_stats = cluster_data[sensitive_attr].value_counts().to_dict()
            generalized_record[sensitive_attr] = sensitive_stats


        return generalized_record

    def anonymize(self, data, quasi_identifiers, sensitive_attr='y'):
        """执行并行APMCA匿名化算法"""
        print(f"开始并行APMCA匿名化处理")
        print(f"配置: k={self.k}, e={self.e}, S={self.S}, 线程数={self.max_workers}")
        print(f"原始数据记录数: {len(data)}")

        start_time = time.time()

        # 识别数值型列
        numeric_columns = self._get_numeric_columns(data)
        print(f"使用的数值型特征: {numeric_columns}")

        if not numeric_columns:
            print("错误: 没有找到数值型特征列")
            return []

        # 步骤1: 并行Mondrian预处理
        mondrian_start = time.time()
        D_subsets = self.parallel_mondrian_processing(data, sensitive_attr)
        mondrian_time = time.time() - mondrian_start
        print(f"Mondrian预处理完成，耗时: {mondrian_time:.2f}秒")

        # 步骤2: 并行聚类和泛化
        clustering_start = time.time()
        self.anonymized_data, self.final_clusters = self.parallel_clustering(
            D_subsets, numeric_columns, sensitive_attr)
        clustering_time = time.time() - clustering_start

        total_time = time.time() - start_time
        print(f"聚类完成，耗时: {clustering_time:.2f}秒")
        print(f"总执行时间: {total_time:.2f}秒")
        print(f"并行APMCA匿名化完成，生成 {len(self.anonymized_data)} 个聚类")
        # ===== 在这里计算并打印平均熵 H̅ 和平均 p_max =====
        if self.anonymized_data and sensitive_attr in self.anonymized_data[0]:
            p_list = []
            H_list = []

            for rec in self.anonymized_data:
                sens_dist = rec.get(sensitive_attr)  # dict: {敏感值: 计数}
                size = rec.get("cluster_size", 0)  # 当前簇大小
                if not sens_dist or not size:
                    continue

                # 转成概率分布
                probs = [cnt / size for cnt in sens_dist.values() if cnt > 0]
                if not probs:
                    continue

                # p_max
                p_max = max(probs)
                p_list.append(p_max)

                # 熵（以 2 为底）
                H = -sum(p * math.log2(p) for p in probs)
                H_list.append(H)

            if p_list:
                mean_pmax = sum(p_list) / len(p_list)
                print(f"[APMCA] 平均 p_max = {mean_pmax:.6f}")

            if H_list:
                mean_H = sum(H_list) / len(H_list)
                print(f"[APMCA] 平均熵 H̅ = {mean_H:.6f}")
        return self.anonymized_data

    def calculate_utility_metrics(self, original_data, anonymized_data, quasi_identifiers):
        """计算数据效用指标"""
        print("\n计算数据效用指标...")

        if len(self.final_clusters) == 0:
            return {
                'average_information_loss': 0,
                'min_cluster_size': 0,
                'max_cluster_size': 0,
                'avg_cluster_size': 0,
                'total_clusters': 0,
                'efficiency_improvement': 0
            }

        # 计算信息损失
        total_loss = 0
        numeric_columns = self._get_numeric_columns(original_data)
        valid_comparisons = 0

        for col in numeric_columns:
            original_range = original_data[col].max() - original_data[col].min()
            if original_range > 0:
                for cluster in self.final_clusters:
                    if col in cluster.columns:
                        cluster_range = cluster[col].max() - cluster[col].min()
                        loss = cluster_range / original_range
                        total_loss += loss
                        valid_comparisons += 1

        avg_loss = total_loss / valid_comparisons if valid_comparisons > 0 else 0

        # 计算聚类大小统计
        cluster_sizes = [len(cluster) for cluster in self.final_clusters if len(cluster) > 0]

        if cluster_sizes:
            min_size = min(cluster_sizes)
            max_size = max(cluster_sizes)
            avg_size = np.mean(cluster_sizes)

            print(f"平均信息损失: {avg_loss:.4f}")
            print(f"聚类大小统计 - 最小: {min_size}, 最大: {max_size}, 平均: {avg_size:.2f}")
        else:
            min_size = max_size = avg_size = 0
            print("没有有效的聚类")

        return {
            'average_information_loss': avg_loss,
            'min_cluster_size': min_size,
            'max_cluster_size': max_size,
            'avg_cluster_size': avg_size,
            'total_clusters': len(self.final_clusters),
            'efficiency_improvement': self.max_workers  # 理论改进倍数
        }

    def performance_benchmark(self, data, quasi_identifiers, sensitive_attr='y'):
        """
        性能基准测试：比较并行和串行版本的性能
        """
        print("\n" + "=" * 60)
        print("性能基准测试")
        print("=" * 60)

        # 测试并行版本
        print("测试并行版本...")
        parallel_start = time.time()
        parallel_result = self.anonymize(data, quasi_identifiers, sensitive_attr)
        parallel_time = time.time() - parallel_start

        # 测试串行版本（使用1个线程）
        print("\n测试串行版本...")
        serial_algorithm = APMCAAlgorithmParallel(
            k=self.k, e=self.e, S=self.S,
            variance_threshold=self.variance_threshold, max_workers=1
        )
        serial_start = time.time()
        serial_result = serial_algorithm.anonymize(data, quasi_identifiers, sensitive_attr)
        serial_time = time.time() - serial_start

        # 计算加速比
        speedup = serial_time / parallel_time if parallel_time > 0 else 1

        print(f"\n性能对比结果:")
        print(f"并行版本时间: {parallel_time:.2f}秒")
        print(f"串行版本时间: {serial_time:.2f}秒")
        print(f"加速比: {speedup:.2f}x")
        print(f"效率: {(speedup / self.max_workers * 100):.1f}%")

        return {
            'parallel_time': parallel_time,
            'serial_time': serial_time,
            'speedup': speedup,
            'efficiency': speedup / self.max_workers * 100
        }


def load_bank_data():
    """加载银行数据"""
    try:
        '''
        if os.path.exists('bank-full.csv'):
            data = pd.read_csv('bank-full.csv', sep=';')
            print("从bank-full.csv加载数据成功")
        elif os.path.exists('bank.csv'):
            data = pd.read_csv('bank.csv', sep=';')
            print("从bank.csv加载数据成功")
        else:
            # 生成模拟数据
            print("使用模拟数据进行测试")
            np.random.seed(42)
            n_samples = 10000  # 更大的数据集用于测试性能

            data = pd.DataFrame({
                'age': np.random.randint(18, 80, n_samples),
                'job': np.random.choice(['admin.', 'technician', 'services', 'management', 'blue-collar'], n_samples),
                'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
                'education': np.random.choice(['secondary', 'primary', 'tertiary', 'unknown'], n_samples),
                'balance': np.random.randint(-1000, 20000, n_samples),
                'duration': np.random.randint(0, 3000, n_samples),
                'campaign': np.random.randint(1, 15, n_samples),
                'y': np.random.choice(['yes', 'no'], n_samples, p=[0.12, 0.88])
            })
        '''
        excel_path = "a_head10000.xlsx"
        if os.path.exists(excel_path):
            #data = pd.read_csv('bank-full.csv', sep=';')
            data = pd.read_excel(excel_path, sheet_name="Sheet1")
            print("从本地文件加载excel数据集成功")
        print(f"数据集形状: {data.shape}")
        return data

    except Exception as e:
        print(f"数据加载失败: {e}")
        # 返回小的测试数据集
        np.random.seed(42)
        return pd.DataFrame({
            'age': np.random.randint(18, 80, 1000),
            'balance': np.random.randint(-1000, 10000, 1000),
            'duration': np.random.randint(0, 2000, 1000),
            'y': np.random.choice(['yes', 'no'], 1000)
        })


def main():
    """主函数"""
    print("=" * 60)
    print("并行APMCA算法测试 - Intel Core i5 13500H优化版")
    print("=" * 60)

    # 显示系统信息
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"可用内存: {psutil.virtual_memory().available / (1024 ** 3):.1f} GB")

    # 加载数据
    data = load_bank_data()
    print(f"数据集预览:")
    print(data.head())

    # 使用适当大小的数据集
    if len(data) > 20000:
        sample_size = min(20000, len(data) // 2)
        data = data.sample(sample_size, random_state=42)
        print(f"使用 {sample_size} 条记录的子集进行测试")

    # 定义准标识符和敏感属性

    quasi_identifiers = ['age', 'marital_status', 'education', 'contact', 'duration', 'campaign']
    sensitive_attr = 'job categorical'

    # 初始化并行算法
    apmca_parallel = APMCAAlgorithmParallel(
        k=8,
        e=3,
        S=6,
        variance_threshold=1000,
        max_workers=None  # 自动检测
    )

    # 执行匿名化
    anonymized_data = apmca_parallel.anonymize(data, quasi_identifiers, sensitive_attr)

    # 显示结果
    print("\n匿名化结果预览:")
    for i, record in enumerate(anonymized_data[:2]):
        print(f"\n聚类 {i + 1}:")
        for key, value in list(record.items())[:4]:
            print(f"  {key}: {value}")
        if sensitive_attr in record:
            print(f"  敏感属性分布: {record[sensitive_attr]}")

    # 计算效用指标
    metrics = apmca_parallel.calculate_utility_metrics(data, anonymized_data, quasi_identifiers)

    # 性能基准测试（可选，需要较长时间）
    if len(data) <= 100:  # 只在数据量较小时进行基准测试
        print("\n进行性能基准测试...")
        benchmark_results = apmca_parallel.performance_benchmark(
            data.sample(2000, random_state=42),  # 使用更小的数据集进行基准测试
            quasi_identifiers,
            sensitive_attr
        )
        metrics.update(benchmark_results)

    print("\n" + "=" * 60)
    print("并行APMCA算法测试完成!")
    print("=" * 60)

    return data, anonymized_data, metrics


if __name__ == "__main__":
    # 运行测试
    set_global_seed(42)
    original_data, anonymized_data, metrics = main()

    # 保存结果
    if anonymized_data:
        try:
            anonymized_df = pd.DataFrame(anonymized_data)

            # 处理字典类型的列
            for col in anonymized_df.columns:
                if anonymized_df[col].apply(lambda x: isinstance(x, dict) if pd.notna(x) else False).any():
                    anonymized_df[col] = anonymized_df[col].astype(str)

            output_file = 'apmca_parallel_result_Adult.xlsx'
            anonymized_df.to_excel(output_file, index=False)
            print(f"匿名化结果已保存到 '{output_file}'")

            # 保存性能指标
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel('apmca_parallel_metrics_Adult.xlsx', index=False)
            print("性能指标已保存到 'apmca_parallel_metrics_Adult.xlsx'")


        except Exception as e:
            print(f"保存结果时出错: {e}")
