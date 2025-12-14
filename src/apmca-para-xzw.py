import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from queue import PriorityQueue  # 使用优先队列模拟 W.extract_min()
import math


class APMCAAlgorithm:
    def __init__(self, k=3, e=2, S=2, variance_threshold=0.1):
        """
        初始化 APMCA 算法（严格遵循伪代码）

        Parameters:
        k: k-匿名性参数
        e: 敏感属性差异阈值（e-差异性）
        S: 每个子集的目标聚类数
        variance_threshold: 方差阈值，用于选择划分属性
        """
        self.k = k
        self.e = e
        self.S = S
        self.variance_threshold = variance_threshold
        self.anonymized_data = []
        self.final_clusters = []

    def _get_numeric_columns(self, data):
        """获取数值型列"""
        numeric_candidates = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
        return [col for col in numeric_candidates if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

    def calculate_sensitive_difference(self, data, sensitive_attr='y'):
        """计算敏感属性差异度（不同取值数量）"""
        if len(data) == 0:
            return 0
        return data[sensitive_attr].nunique()

    def calculate_variance(self, data, numeric_columns=None):
        """计算数据集方差（所有数值列方差之和）"""
        if numeric_columns is None:
            numeric_columns = self._get_numeric_columns(data)
        if len(data) == 0 or len(numeric_columns) == 0:
            return 0
        total_variance = sum(data[col].var() for col in numeric_columns if col in data.columns)
        return total_variance

    def select_attribute_for_perturbation(self, data, variance_info, numeric_columns):
        """基于最大方差选择划分属性"""
        max_variance = -1
        selected_attribute = None
        for col in numeric_columns:
            if col in data.columns:
                var_val = data[col].var()
                if not np.isnan(var_val) and var_val > max_variance:
                    max_variance = var_val
                    selected_attribute = col
        return selected_attribute if selected_attribute else self.randomly_select_attribute(data, numeric_columns)

    def randomly_select_attribute(self, data, numeric_columns):
        """随机选择一个属性"""
        available = [col for col in numeric_columns if col in data.columns]
        return np.random.choice(available) if available else None

    def calculate_median(self, data, attribute):
        """计算中位数"""
        return data[attribute].median()

    def split_subset(self, data, attribute, median):
        """按中位数划分"""
        subset1 = data[data[attribute] <= median].copy()
        subset2 = data[data[attribute] > median].copy()
        return subset1, subset2

    def determine_num_clusters(self, subset):
        """确定聚类数量"""
        n = len(subset)
        if n <= 3 * self.k:
            return max(1, min(self.S, n // self.k))
        else:
            return min(self.S * 2, n // self.k)

    def k_means_clustering(self, data, num_clusters, numeric_columns):
        """K-means 聚类"""
        if len(data) < num_clusters:
            return [data]
        if len(data) == 0:
            return []
        X = data[numeric_columns].values
        try:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            clusters = []
            for i in range(num_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) > 0:
                    clusters.append(data.iloc[cluster_indices].copy())
            return clusters
        except:
            return [data]  # 失败则返回原始数据作为一个簇

    def generalize_cluster(self, cluster_data, sensitive_attr='y'):
        """泛化单个簇"""
        if len(cluster_data) == 0:
            return {}
        generalized = {}

        # 数值属性：区间表示
        numeric_cols = self._get_numeric_columns(cluster_data)
        for col in numeric_cols:
            min_val, max_val = cluster_data[col].min(), cluster_data[col].max()
            generalized[col] = f"[{min_val},{max_val}]" if min_val != max_val else f"[{min_val}]"

        # 分类属性：合并取值
        cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
        cat_cols = [c for c in cat_cols if c in cluster_data.columns]
        for col in cat_cols:
            vals = sorted(cluster_data[col].unique())
            generalized[col] = '\\'.join(map(str, vals))

        # 敏感属性分布
        if sensitive_attr in cluster_data.columns:
            generalized[sensitive_attr] = dict(cluster_data[sensitive_attr].value_counts())
            generalized['cluster_size'] = len(cluster_data)

        return generalized

    def anonymize(self, data, quasi_identifiers, sensitive_attr='y'):
        """
        执行 APMCA 匿名化（严格遵循伪代码）
        """
        print("开始执行 APMCA 算法（串行、优先队列版本）...")
        print(f"参数: k={self.k}, e={self.e}, S={self.S}, variance_threshold={self.variance_threshold}")
        print(f"数据量: {len(data)}")

        numeric_columns = self._get_numeric_columns(data)
        if not numeric_columns:
            raise ValueError("未找到有效的数值型准标识符")

        # Step 1: 初始化 W 和 D
        W = PriorityQueue()  # 优先队列，按子集大小排序（可改为其他优先级）
        D = []  # 存储满足条件的子集

        # 将整个数据集加入 W
        W.put((len(data), id(data), data))  # (size, unique_id, subset) 避免不可比较

        # Step 2: Mondrian 风格划分阶段
        while not W.empty():
            _, _, subset = W.get()  # extract_min() —— 这里 size 最小优先
            n = len(subset)
            sensitive_diff = self.calculate_sensitive_difference(subset, sensitive_attr)

            # 情况1: 满足 k ≤ n ≤ 2k 且 sensitive_diff ≥ e
            if self.k <= n <= 2 * self.k and sensitive_diff >= self.e:
                D.append(subset)

            # 情况2: n > 2k，需要划分
            elif n > 2 * self.k:
                variance = self.calculate_variance(subset, numeric_columns)
                if variance > self.variance_threshold:
                    attribute = self.select_attribute_for_perturbation(subset, variance, numeric_columns)
                else:
                    attribute = self.randomly_select_attribute(subset, numeric_columns)

                if attribute is None:
                    D.append(subset)  # 无法划分，直接加入 D
                else:
                    median = self.calculate_median(subset, attribute)
                    subset1, subset2 = self.split_subset(subset, attribute, median)

                    # 将两个子集加入 W（即使为空也加入，后续判断）
                    if len(subset1) > 0:
                        W.put((len(subset1), id(subset1), subset1))
                    if len(subset2) > 0:
                        W.put((len(subset2), id(subset2), subset2))

            # 情况3: k ≤ n < k 不成立？伪代码此处可能有歧义
            # 原文第21-22行：else if n >= k then W.add(subset);
            # 但 n >= k 已在前面处理，这里可能是 n < k 的情况？
            # 根据上下文，我们理解为：若不满足前两个条件但 n >= k，则重新入队（罕见）
            elif n >= self.k:
                # 可能是 sensitive_diff < e 但大小合适，重新入队等待进一步划分
                W.put((n, id(subset), subset))

            # 注意：n < k 的情况未在伪代码中明确，我们暂不处理（丢弃或报错需根据需求）

        print(f"Mondrian 划分完成，生成 {len(D)} 个最终子集")

        # Step 3: 对每个子集进行聚类与泛化
        self.anonymized_data = []
        self.final_clusters = []

        for subset in D:
            num_clusters = self.determine_num_clusters(subset)
            clusters = self.k_means_clustering(subset, num_clusters, numeric_columns)

            for cluster in clusters:
                if len(cluster) >= self.k:  # 确保满足 k-匿名
                    generalized = self.generalize_cluster(cluster, sensitive_attr)
                    self.anonymized_data.append(generalized)
                    self.final_clusters.append(cluster)

        print(f"聚类与泛化完成，生成 {len(self.anonymized_data)} 个匿名簇")

        # ===== 计算并打印平均熵 H̅ 和平均 p_max =====
        if self.anonymized_data and sensitive_attr in self.anonymized_data[0]:
            p_list = []
            H_list = []

            for rec in self.anonymized_data:
                sens_dist = rec.get(sensitive_attr)
                size = rec.get("cluster_size", 0)
                if not sens_dist or not size:
                    continue

                probs = [cnt / size for cnt in sens_dist.values() if cnt > 0]
                if not probs:
                    continue

                p_max = max(probs)
                p_list.append(p_max)

                H = -sum(p * math.log2(p) for p in probs)
                H_list.append(H)

            if p_list:
                mean_pmax = sum(p_list) / len(p_list)
                print(f"[APMCA-serial] 平均 p_max = {mean_pmax:.6f}")

            if H_list:
                mean_H = sum(H_list) / len(H_list)
                print(f"[APMCA-serial] 平均熵 H̅ = {mean_H:.6f}")
        # ===== 代码到此结束 =====

        return self.anonymized_data
