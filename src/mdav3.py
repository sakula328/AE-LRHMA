import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings
import time
import math


warnings.filterwarnings('ignore')


class StrictKEMDAV:
    def __init__(self, k, e, quasi_identifiers, sensitive_attribute):
        """
        (k, e)-MDAV 算法 - 严格按论文实现

        参数:
        k: 每个等价类的最小记录数
        e: 敏感属性差异阈值
        quasi_identifiers: 准标识符列名列表
        sensitive_attribute: 敏感属性列名
        """
        self.k = k
        self.e = e
        self.quasi_identifiers = quasi_identifiers
        self.sensitive_attribute = sensitive_attribute
        self.scaler = StandardScaler()

    def Calculate_Sensitive_Difference(self, data):  #计算当前数据集敏感属性的“差异度”指标
        """计算数据集中敏感属性的差异 - 算法步骤2的条件判断"""
        if len(data) == 0:
            return 0

        sensitive_values = data[self.sensitive_attribute].values
        unique_count = len(np.unique(sensitive_values))
        return unique_count

    def Find_Farthest_Pair(self, data, normalized_data):  #在标准化后的准标识符空间里找最远的一对记录 R,S
        """找到T中最远的一对记录R和S - 算法步骤3"""
        n = len(data)
        if n < 2:
            return 0, 0  # 如果没有足够记录，返回默认值

        # 计算所有记录之间的欧氏距离
        distances = euclidean_distances(normalized_data)
        np.fill_diagonal(distances, -1)  # 将对角线设为-1，避免自己与自己比较

        # 找到最大距离对应的索引
        max_idx = np.unravel_index(np.argmax(distances), distances.shape)
        return max_idx[0], max_idx[1]

    def Find_Nearest_K(self, data, normalized_data, center_idx, k): #以某个中心点找最近的 k 条记录
        """从T中选择k个最近记录形成初始分组"""
        if len(data) <= k:
            return list(range(len(data)))

        center_point = normalized_data[center_idx]
        distances = []

        for i in range(len(data)):
            if i != center_idx:
                distance = np.linalg.norm(normalized_data[i] - center_point)
                distances.append((i, distance))

        # 按距离排序并返回最近的k-1个（加上中心点共k个）
        distances.sort(key=lambda x: x[1])
        nearest_indices = [center_idx] + [idx for idx, _ in distances[:k - 1]]
        return nearest_indices

    def Ensure_e_Diversity(self, group_indices, data):  #检查组内敏感值种类数，不够 e 就尝试替换记录
        """确保分组满足e-差异性"""
        if len(group_indices) == 0:
            return group_indices

        sensitive_values = [data.iloc[idx][self.sensitive_attribute] for idx in group_indices]
        unique_count = len(set(sensitive_values))

        # 如果当前分组不满足e-差异性
        if unique_count < self.e:
            print(f"    e-多样性检查: {unique_count} < {self.e}, 需要调整分组")

            # 获取所有可用记录
            all_indices = set(range(len(data)))
            current_group_set = set(group_indices)
            available_indices = list(all_indices - current_group_set)

            current_sensitive_set = set(sensitive_values)

            # 尝试替换非中心记录来增加多样性
            # 保持第一个记录（中心点）不变
            for i in range(1, len(group_indices)):
                if len(available_indices) == 0:
                    break

                replace_idx = group_indices[i]

                # 寻找能增加敏感值多样性的记录
                found_replacement = False
                for new_idx in available_indices:
                    new_sensitive = data.iloc[new_idx][self.sensitive_attribute]

                    if new_sensitive not in current_sensitive_set:
                        # 替换记录
                        group_indices[i] = new_idx
                        available_indices.remove(new_idx)
                        current_sensitive_set.add(new_sensitive)
                        unique_count = len(current_sensitive_set)
                        found_replacement = True
                        print(f"      替换记录 {replace_idx} -> {new_idx}, 敏感值: {new_sensitive}")
                        break

                if unique_count >= self.e:
                    break

            print(f"    调整后e-多样性: {unique_count}")

        return group_indices

    def Form_Group_With_e_Diversity(self, data, normalized_data, center_idx, k, e):
        """以中心点为中心，从T中选择k个最近记录形成初始分组，并确保满足e-差异性"""
        # 步骤1: 找到最近的k个记录
        group_indices = self.Find_Nearest_K(data, normalized_data, center_idx, k)

        # 步骤2: 确保e-多样性
        group_indices = self.Ensure_e_Diversity(group_indices, data)

        return group_indices

    def Generalize(self, group_indices, data):
        """泛化操作 - 对分组中的准标识符进行泛化"""
        if len(group_indices) == 0:
            return None

        group_data = data.iloc[group_indices]
        generalized_record = {}

        # 对每个准标识符进行泛化
        for col in self.quasi_identifiers:
            if pd.api.types.is_numeric_dtype(data[col]):
                # 数值型属性：使用[min, max]区间泛化
                min_val = group_data[col].min()
                max_val = group_data[col].max()
                generalized_record[col] = [min_val, max_val]
            else:
                # 分类型属性：使用所有可能值的列表
                unique_values = sorted(list(group_data[col].unique()))
                generalized_record[col] = unique_values

        # 敏感属性保持原值（在实际应用中可能需要进一步处理）
        generalized_record[self.sensitive_attribute] = group_data[self.sensitive_attribute].iloc[0]

        return generalized_record

    def Distribute_And_Generalize_Remaining(self, T, Anonymized_Data, data, normalized_data):
        """将剩余记录分配到Anonymized_Data中已有的分组，并泛化所有分组"""
        print("执行Distribute_And_Generalize_Remaining操作")

        if len(T) == 0:
            return Anonymized_Data

        # 分配剩余记录到已有分组
        for i in range(len(T)):
            remaining_idx = T.index[i]
            remaining_point = normalized_data[data.index.get_loc(remaining_idx)]

            min_distance = float('inf')
            best_group_index = -1

            # 找到最适合的分组
            for j, group in enumerate(Anonymized_Data):
                if 'group_indices' not in group:
                    continue

                # 计算到分组中心的距离
                group_indices = group['group_indices']
                group_points = []

                for idx in group_indices:
                    original_idx = data.index.get_loc(idx)
                    group_points.append(normalized_data[original_idx])

                if group_points:
                    group_center = np.mean(group_points, axis=0)
                    distance = np.linalg.norm(remaining_point - group_center)

                    if distance < min_distance:
                        min_distance = distance
                        best_group_index = j

            # 将记录添加到最佳分组
            if best_group_index != -1:
                Anonymized_Data[best_group_index]['group_indices'].append(remaining_idx)
                print(f"  剩余记录 {remaining_idx} 分配到分组 {best_group_index}")

        # 重新泛化所有分组
        print("重新泛化所有分组...")
        generalized_groups = []

        for group in Anonymized_Data:
            if 'group_indices' in group:
                # 原始索引标签（如 0,1,2... 或 DataFrame 的 index）
                idx_labels = group['group_indices']

                # 转成在 data 里的“位置索引”，用于 Generalize
                idx_positions = [data.index.get_loc(idx) for idx in idx_labels]

                generalized_group = self.Generalize(idx_positions, data)
                if generalized_group:
                    generalized_group['group_size'] = len(idx_labels)
                    # 关键：把 group_indices 保留下来，供后续计算熵和 p_max 使用
                    generalized_group['group_indices'] = list(idx_labels)
                    generalized_groups.append(generalized_group)

        return generalized_groups

    def load_data_from_excel(self, file_path, sheet_name=0):
        """从Excel文件加载数据"""
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"成功加载数据，形状: {data.shape}")
            print("数据列:", data.columns.tolist())
            return data
        except Exception as e:
            print(f"加载Excel文件失败: {e}")
            return None

    def preprocess_data(self, data):
        """数据预处理"""
        required_columns = self.quasi_identifiers + [self.sensitive_attribute]
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            print(f"错误: 缺少必要的列: {missing_columns}")
            return None

        # 处理缺失值
        data_clean = data.dropna(subset=required_columns)
        if len(data_clean) < len(data):
            print(f"警告: 删除了 {len(data) - len(data_clean)} 条包含缺失值的记录")

        data_clean = data_clean.reset_index(drop=True)
        return data_clean

    def normalize_quasi_identifiers(self, data):
        """标准化准标识符"""
        quasi_data = data[self.quasi_identifiers]

        # 分离数值型和分类型列
        numeric_columns = [col for col in self.quasi_identifiers
                           if pd.api.types.is_numeric_dtype(quasi_data[col])]
        categorical_columns = [col for col in self.quasi_identifiers
                               if col not in numeric_columns]

        normalized_data = np.zeros((len(data), len(self.quasi_identifiers)))

        # 标准化数值型列
        if numeric_columns:
            numeric_values = self.scaler.fit_transform(quasi_data[numeric_columns])
            for i, col in enumerate(self.quasi_identifiers):
                if col in numeric_columns:
                    col_idx = numeric_columns.index(col)
                    normalized_data[:, i] = numeric_values[:, col_idx]

        # 编码分类型列
        for i, col in enumerate(self.quasi_identifiers):
            if col in categorical_columns:
                encoded = pd.Categorical(quasi_data[col]).codes
                normalized_data[:, i] = encoded

        return normalized_data

    def kemdav_algorithm(self, data):   #整个 (k,e)-MDAV 主流程
        """严格按论文实现的(k,e)-MDAV算法主函数"""
        print("开始执行(k,e)-MDAV算法...")
        print(f"算法参数: k={self.k}, e={self.e}")
        print(f"准标识符: {self.quasi_identifiers}")
        print(f"敏感属性: {self.sensitive_attribute}")

        # 步骤1: 初始化
        T = data.copy()
        normalized_T = self.normalize_quasi_identifiers(T)
        Anonymized_Data = []

        print(f"初始数据集大小: {len(T)}")

        # 步骤2-8: 主循环
        iteration = 0
        while len(T) > 2 * self.k and self.Calculate_Sensitive_Difference(T) >= self.e:
            iteration += 1
            print(f"\n=== 迭代 {iteration} ===")
            print(f"剩余记录数: {len(T)}")
            print(f"敏感属性差异: {self.Calculate_Sensitive_Difference(T)}")

            # 步骤3: 找到最远的一对记录R和S
            R_idx, S_idx = self.Find_Farthest_Pair(T, normalized_T)
            print(f"找到最远点对: R={R_idx}, S={S_idx}")

            # 步骤4: 以R为中心形成分组
            print("形成以R为中心的分组...")
            GroupR = self.Form_Group_With_e_Diversity(T, normalized_T, R_idx, self.k, self.e)
            print(f"GroupR大小: {len(GroupR)}")

            # 步骤5: 以S为中心形成分组
            print("形成以S为中心的分组...")
            GroupS = self.Form_Group_With_e_Diversity(T, normalized_T, S_idx, self.k, self.e)
            print(f"GroupS大小: {len(GroupS)}")

            # 步骤6: 从T中移除分组
            indices_to_remove = GroupR + GroupS
            T_indices_to_remove = T.index[indices_to_remove]
            T = T.drop(T_indices_to_remove)
            normalized_T = np.delete(normalized_T, indices_to_remove, axis=0)

            # 步骤7: 泛化并添加到Anonymized_Data
            Generalized_GroupR = self.Generalize(GroupR, data)
            Generalized_GroupS = self.Generalize(GroupS, data)

            if Generalized_GroupR:
                Generalized_GroupR['group_indices'] = [data.index[i] for i in GroupR]
                Generalized_GroupR['group_size'] = len(GroupR)
                Anonymized_Data.append(Generalized_GroupR)

            if Generalized_GroupS:
                Generalized_GroupS['group_indices'] = [data.index[i] for i in GroupS]
                Generalized_GroupS['group_size'] = len(GroupS)
                Anonymized_Data.append(Generalized_GroupS)

            print(f"移除分组后剩余记录数: {len(T)}")

        # 步骤9-12: 处理剩余记录
        print(f"\n=== 处理剩余记录 ===")
        print(f"剩余记录数: {len(T)}")

        if len(T) >= self.k and self.Calculate_Sensitive_Difference(T) >= self.e:
            print("条件满足: |T| >= k 且 敏感属性差异 >= e")
            print("直接泛化剩余记录...")
            remaining_indices = list(range(len(T)))
            Generalized_T = self.Generalize(remaining_indices, data)
            if Generalized_T:
                Generalized_T['group_indices'] = list(T.index)
                Generalized_T['group_size'] = len(T)
                Anonymized_Data.append(Generalized_T)
        else:
            print("条件不满足，执行Distribute_And_Generalize_Remaining...")
            Anonymized_Data = self.Distribute_And_Generalize_Remaining(T, Anonymized_Data, data,
                                                                       self.normalize_quasi_identifiers(data))

        # ===== 在创建最终数据集之前，计算平均熵 H̅ 和平均 p_max =====
        p_list = []
        H_list = []

        for group in Anonymized_Data:
            idxs = group.get('group_indices')
            if not idxs:
                continue

            # 从原始（预处理后）数据中取出这个等价类的敏感属性
            sens_series = data.loc[idxs, self.sensitive_attribute]
            counts = sens_series.value_counts()
            size = counts.sum()
            if size == 0:
                continue

            probs = counts / size

            # p_max
            p_max = probs.max()
            p_list.append(p_max)

            # 熵（以 2 为底）
            H = -sum(p * math.log2(p) for p in probs)
            H_list.append(H)

        if p_list:
            mean_pmax = sum(p_list) / len(p_list)
            print(f"[MDAV] 平均 p_max = {mean_pmax:.6f}")

        if H_list:
            mean_H = sum(H_list) / len(H_list)
            print(f"[MDAV] 平均熵 H̅ = {mean_H:.6f}")
        # 创建最终匿名化数据集
        print("\n=== 创建最终匿名化数据集 ===")
        final_records = []
        for group in Anonymized_Data:
            if 'group_size' not in group:
                raise ValueError("Missing group_size in some generalized group. Fix group_size assignment.")
            group_size = int(group['group_size'])
            for i in range(group_size):
                record = group.copy()
                if 'group_indices' in record:
                    del record['group_indices']
                final_records.append(record)

        anonymized_df = pd.DataFrame(final_records)

        print(f"匿名化完成! 生成 {len(Anonymized_Data)} 个等价类")
        print(f"最终数据集大小: {len(anonymized_df)}")

        return anonymized_df

    def run_from_excel(self, file_path, sheet_name=0):
        """从Excel文件运行完整算法"""
        # 加载数据
        raw_data = self.load_data_from_excel(file_path, sheet_name)
        if raw_data is None:
            return None

        # 预处理数据
        data = self.preprocess_data(raw_data)
        if data is None:
            return None

        print(f"预处理后数据形状: {data.shape}")
        print(data.head())

        # 执行算法
        result = self.kemdav_algorithm(data)
        return result


# 测试代码
if __name__ == "__main__":
    start_time = time.time()
    # 定义算法参数
    quasi_identifiers = ['age', 'marital_status', 'education', 'contact', 'duration', 'campaign']
    sensitive_attribute = 'job categorical'

    k = 8
    e = 3

    # 创建算法实例
    kemdav = StrictKEMDAV(k, e, quasi_identifiers, sensitive_attribute)

    # 从Excel文件运行算法
    excel_file = "head10000.xlsx"
    sheet_name = "Sheet1"

    print("=" * 60)
    print("(k,e)-MDAV算法测试")
    print("=" * 60)

    result = kemdav.run_from_excel(excel_file, sheet_name)

    processing_time = time.time() - start_time

    if result is not None:
        print("\n" + "=" * 60)
        print("匿名化结果:")
        print("=" * 60)
        print(result)

        # 保存结果
        output_file = excel_file + "-strict_kemdav_result.xlsx"
        result.to_excel(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")

        # 显示统计信息
        print("\n统计信息:")
        print(f"原始记录数: 4")
        print(f"匿名化后记录数: {len(result)}")
        print(f"生成的等价类数: {len(result) // k}")
        print(f"每个等价类大小: {k}")
        print(f"处理时间: {processing_time:.2f} 秒")

    else:
        print("算法执行失败!")