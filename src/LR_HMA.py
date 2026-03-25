import pandas as pd
import numpy as np
import warnings
import time
import math
import os
import random

from sklearn.preprocessing import StandardScaler, OneHotEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# =========================
# 自编码器定义
# =========================
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dims=None):
        super(AutoEncoder, self).__init__()

        if hidden_dims is None:
            h = max(16, 2 * embedding_dim)
            hidden_dims = [h]

        # encoder
        layers_enc = []
        prev = input_dim
        for h_dim in hidden_dims:
            layers_enc.append(nn.Linear(prev, h_dim))
            layers_enc.append(nn.ReLU())
            prev = h_dim
        layers_enc.append(nn.Linear(prev, embedding_dim))
        self.encoder = nn.Sequential(*layers_enc)

        # decoder
        layers_dec = []
        prev = embedding_dim
        for h_dim in reversed(hidden_dims):
            layers_dec.append(nn.Linear(prev, h_dim))
            layers_dec.append(nn.ReLU())
            prev = h_dim
        layers_dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*layers_dec)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return z


# =========================
# LR-HMA 主类
# =========================
class LRHMA:
    """
    LR-HMA: Latent Representation-based Hybrid Microaggregation (Autoencoder + 隐私增强版)
    """

    def __init__(
        self,
        k,
        e,
        quasi_identifiers,
        sensitive_attribute,
        variance_threshold=0.05,
        embedding_dim=None,
        max_group_size=None,
        alpha_max=0.8,        # 单一敏感值最大占比
        entropy_min=0.0,      # 组内敏感熵下界
        ae_hidden_dims=None,
        ae_epochs=50,
        ae_batch_size=128,
        ae_lr=1e-3,
        random_state=42,
        device=None,
        encoding_mode="integer",  #（修改）
        cat_perm_seed = None,     #修改
    ):
        self.k = int(k)
        self.e = int(e)
        self.quasi_identifiers = list(quasi_identifiers)
        self.sensitive_attribute = sensitive_attribute
        self.variance_threshold = variance_threshold
        self.embedding_dim = embedding_dim
        self.random_state = random_state

        self.max_group_size = max_group_size if max_group_size is not None else 3 * self.k

        #增加三个参数 + 存 encoder（修改）
        self.encoding_mode = encoding_mode
        self.cat_perm_seed = cat_perm_seed

        self.onehot = None
        self.cat_categories_ = {}  # 存每个类别列的 categories（保证 fit/transform 一致）

        # 隐私增强参数
        self.alpha_max = alpha_max
        self.entropy_min = entropy_min

        # AE 参数
        self.ae_hidden_dims = ae_hidden_dims
        self.ae_epochs = ae_epochs
        self.ae_batch_size = ae_batch_size
        self.ae_lr = ae_lr

        # 特征处理
        self.scaler = StandardScaler()
        self.numeric_columns = None
        self.categorical_columns = None

        # AE & 表示
        self.ae = None
        self.data_ = None
        self.Z_ = None

        # 统计
        self.last_groups_ = None
        self.last_original_n_ = None

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    # ========= 读数 & 预处理 =========
    def load_data_from_excel(self, file_path, sheet_name=0):
        try:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            print(f"成功加载数据，形状: {data.shape}")
            print("数据列:", data.columns.tolist())
            return data
        except Exception as e:
            print(f"加载Excel文件失败: {e}")
            return None

    def preprocess_data(self, data):
        required_columns = self.quasi_identifiers + [self.sensitive_attribute]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            print(f"错误: 缺少必要的列: {missing_columns}")
            return None

        data_clean = data.dropna(subset=required_columns)
        if len(data_clean) < len(data):
            print(f"警告: 删除了 {len(data) - len(data_clean)} 条包含缺失值的记录")

        data_clean = data_clean.reset_index(drop=True)
        return data_clean

    # ========= 特征矩阵 =========
    def _prepare_features(self, data, fit=False):#（修改）
        quasi_data = data[self.quasi_identifiers]

        # 1) split numeric / categorical
        if fit:
            numeric_columns = [c for c in self.quasi_identifiers
                               if pd.api.types.is_numeric_dtype(quasi_data[c])]
            categorical_columns = [c for c in self.quasi_identifiers if c not in numeric_columns]
            self.numeric_columns = numeric_columns
            self.categorical_columns = categorical_columns
        else:
            numeric_columns = self.numeric_columns
            categorical_columns = self.categorical_columns
            if numeric_columns is None:
                raise RuntimeError("必须先调用 _prepare_features(..., fit=True)")

        # 2) numeric part
        X_num = None
        if numeric_columns:
            if fit:
                X_num = self.scaler.fit_transform(quasi_data[numeric_columns])
            else:
                X_num = self.scaler.transform(quasi_data[numeric_columns])

        # 3) categorical part
        if self.encoding_mode.lower() == "onehot":
            if fit:
                try:
                    self.onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                except TypeError:
                    self.onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
                X_cat = self.onehot.fit_transform(quasi_data[categorical_columns]) if categorical_columns else None
            else:
                X_cat = self.onehot.transform(quasi_data[categorical_columns]) if categorical_columns else None

            # concat
            if X_num is None and X_cat is None:
                return np.zeros((len(data), 0), dtype=float)
            if X_num is None:
                return X_cat.astype(float)
            if X_cat is None:
                return X_num.astype(float)
            return np.hstack([X_num, X_cat]).astype(float)

        # ------------- integer mode (default) -------------
        # 4) integer encoding with optional permutation + stable categories
        X_cat_int = None
        if categorical_columns:
            if fit:
                rng = np.random.RandomState(self.cat_perm_seed) if self.cat_perm_seed is not None else None
                codes_list = []
                for col in categorical_columns:
                    cats = pd.Series(quasi_data[col]).astype("category").cat.categories.tolist()
                    # optional permutation
                    if rng is not None:
                        cats = list(cats)
                        rng.shuffle(cats)
                    self.cat_categories_[col] = cats

                    # stable coding
                    codes = pd.Categorical(quasi_data[col], categories=cats).codes.astype(float).reshape(-1, 1)
                    codes_list.append(codes)
                X_cat_int = np.hstack(codes_list) if codes_list else None
            else:
                codes_list = []
                for col in categorical_columns:
                    cats = self.cat_categories_.get(col, None)
                    if cats is None:
                        # fallback (shouldn't happen if fit=True ran)
                        cats = pd.Series(quasi_data[col]).astype("category").cat.categories.tolist()
                    codes = pd.Categorical(quasi_data[col], categories=cats).codes.astype(float).reshape(-1, 1)
                    codes_list.append(codes)
                X_cat_int = np.hstack(codes_list) if codes_list else None

        # 5) concat numeric + integer-cat
        if X_num is None and X_cat_int is None:
            return np.zeros((len(data), 0), dtype=float)
        if X_num is None:
            return X_cat_int.astype(float)
        if X_cat_int is None:
            return X_num.astype(float)
        return np.hstack([X_num, X_cat_int]).astype(float)

    # ========= Phase 0：自编码器表示学习 =========
    def _train_representation_encoder(self):
        print("  [AE] 构造特征矩阵并训练自编码器...")
        X = self._prepare_features(self.data_, fit=True)
        X_tensor = torch.from_numpy(X).float()

        n_samples, input_dim = X.shape
        if self.embedding_dim is None:
            z_dim = min(min(input_dim, 8), max(2, input_dim))
        else:
            z_dim = min(self.embedding_dim, input_dim)

        self.ae = AutoEncoder(input_dim, z_dim, self.ae_hidden_dims).to(self.device)

        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=self.ae_batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.ae.parameters(), lr=self.ae_lr)
        criterion = nn.MSELoss()

        self.ae.train()
        for epoch in range(self.ae_epochs):
            epoch_loss = 0.0
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.ae(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)
            epoch_loss /= n_samples
            if (epoch + 1) % max(1, self.ae_epochs // 5) == 0:
                print(f"    [AE] Epoch {epoch+1}/{self.ae_epochs}, loss={epoch_loss:.6f}")
        print("  [AE] 训练完成。")
        print("[DEBUG] encoding_mode =", self.encoding_mode, "cat_perm_seed =", self.cat_perm_seed)
        print("[DEBUG] numeric_columns =", self.numeric_columns)
        print("[DEBUG] categorical_columns =", self.categorical_columns)
        print("[DEBUG] X shape =", X.shape)

    def _encode_to_latent(self):
        X = self._prepare_features(self.data_, fit=False)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        self.ae.eval()
        with torch.no_grad():
            Z_tensor = self.ae.encode(X_tensor)
        self.Z_ = Z_tensor.cpu().numpy()
        print(f"  [AE] 潜在表示维度: {self.Z_.shape[1]}")

    # ========= Phase 1：Z 中 Mondrian 划分 =========
    def _mondrian_partition(self):
        n = len(self.data_)
        all_indices = list(range(n))
        return self._mondrian_partition_on_indices(all_indices, tag="全局")

    def _mondrian_partition_on_indices(self, indices, tag="局部"):
        queue = [list(indices)]
        subsets = []
        remainder = []

        print(f"\n[Phase 1-{tag}] 在 {len(indices)} 条记录上进行 Mondrian 划分...")

        while queue:
            subset = queue.pop(0)
            m = len(subset)
            if m == 0:
                continue

            sens_values = self.data_.iloc[subset][self.sensitive_attribute]
            sens_unique = sens_values.nunique()

            if self.k <= m <= 2 * self.k and sens_unique >= self.e:
                subsets.append(subset)
                print(f"  子集大小 {m}，敏感值种类 {sens_unique}，加入有效子集 D")
                continue

            if m > 2 * self.k:
                Z_sub = self.Z_[subset, :]
                var = np.var(Z_sub, axis=0)
                split_dim = int(np.argmax(var))
                max_var = var[split_dim]

                if max_var <= self.variance_threshold:
                    print(f"  子集大小 {m}，最大方差 {max_var:.4f} <= 阈值，丢入剩余集合")
                    remainder.extend(subset)
                    continue

                dim_values = Z_sub[:, split_dim]
                median = np.median(dim_values)

                left = [idx for idx in subset if self.Z_[idx, split_dim] <= median]
                right = [idx for idx in subset if self.Z_[idx, split_dim] > median]

                if len(left) < self.k or len(right) < self.k:
                    print(
                        f"  子集大小 {m}，划分后 left={len(left)}, right={len(right)} < k，丢入剩余集合"
                    )
                    remainder.extend(subset)
                else:
                    print(
                        f"  子集大小 {m}，按维度 {split_dim} 中位数划分 -> left={len(left)}, right={len(right)}"
                    )
                    queue.append(left)
                    queue.append(right)
            else:
                print(
                    f"  子集大小 {m} 或敏感值种类 {sens_unique} 不满足条件，丢入剩余集合"
                )
                remainder.extend(subset)

        print(
            f"[Phase 1-{tag}] 完成：生成 {len(subsets)} 个有效子集，剩余记录数 {len(remainder)}"
        )
        return subsets, remainder

    # ========= Phase 2：局部 MDAV =========
    def _sensitive_diversity(self, indices):
        if not indices:
            return 0
        svals = self.data_.iloc[indices][self.sensitive_attribute]
        return svals.nunique()

    def _sensitive_stats(self, indices):
        if not indices:
            return 0.0, 0.0
        s = self.data_.iloc[indices][self.sensitive_attribute]
        counts = s.value_counts()
        total = len(s)
        if total == 0:
            return 0.0, 0.0
        p_max = counts.max() / total
        H = 0.0
        for c in counts:
            p = c / total
            H -= p * math.log(p + 1e-12)
        return p_max, H

    def _farthest_from_centroid(self, indices):
        Z_sub = self.Z_[indices, :]
        center = Z_sub.mean(axis=0)
        diffs = Z_sub - center
        dists = np.linalg.norm(diffs, axis=1)
        pos = int(np.argmax(dists))
        return indices[pos]

    def _ensure_e_diversity(self, group_indices, candidate_pool, max_iter=20):
        """
        尝试让 group_indices 满足：
          - 种类数 ≥ e
          - p_max ≤ alpha_max
          - entropy ≥ entropy_min
        """
        if len(group_indices) == 0:
            return group_indices

        data = self.data_
        group_indices = list(group_indices)
        candidate_pool = list(candidate_pool)

        def get_sens(idx_list):
            return data.iloc[idx_list][self.sensitive_attribute]

        # 先保证 distinct_count ≥ e
        sens_vals = list(get_sens(group_indices))
        unique_vals = set(sens_vals)

        if len(unique_vals) < self.e:
            freq = {}
            for v in sens_vals:
                freq[v] = freq.get(v, 0) + 1
            majority_val = max(freq.items(), key=lambda x: x[1])[0]

            for cand_idx in list(candidate_pool):
                cand_s = data.iloc[cand_idx][self.sensitive_attribute]
                if cand_s in unique_vals:
                    continue

                replace_pos = None
                for i, idx in enumerate(group_indices):
                    if (
                        data.iloc[idx][self.sensitive_attribute] == majority_val
                        and freq[majority_val] > 1
                    ):
                        replace_pos = i
                        break
                if replace_pos is None:
                    continue

                old_idx = group_indices[replace_pos]
                group_indices[replace_pos] = cand_idx

                freq[majority_val] -= 1
                freq[cand_s] = freq.get(cand_s, 0) + 1
                unique_vals.add(cand_s)
                candidate_pool.remove(cand_idx)

                if len(unique_vals) >= self.e:
                    break

        # 再尝试优化 p_max / 熵
        for _ in range(max_iter):
            p_max, H = self._sensitive_stats(group_indices)
            if p_max <= self.alpha_max and H >= self.entropy_min:
                break

            svals = get_sens(group_indices)
            freq = svals.value_counts().to_dict()
            majority_val = max(freq.items(), key=lambda x: x[1])[0]

            improved = False
            for cand_idx in list(candidate_pool):
                cand_s = data.iloc[cand_idx][self.sensitive_attribute]
                if cand_s == majority_val:
                    continue

                replace_pos = None
                for i, idx in enumerate(group_indices):
                    if (
                        data.iloc[idx][self.sensitive_attribute] == majority_val
                        and freq[majority_val] > 1
                    ):
                        replace_pos = i
                        break
                if replace_pos is None:
                    continue

                old_idx = group_indices[replace_pos]
                new_group = list(group_indices)
                new_group[replace_pos] = cand_idx
                new_p_max, new_H = self._sensitive_stats(new_group)

                if (new_p_max < p_max) or (new_H > H):
                    group_indices = new_group
                    candidate_pool.remove(cand_idx)
                    improved = True
                    break

            if not improved:
                break

        return group_indices

    def _form_group_with_constraints(self, U, center_idx):
        if center_idx not in U:
            return []

        if len(U) <= self.k:
            return list(U)

        center_z = self.Z_[center_idx]
        candidate_indices = [idx for idx in U if idx != center_idx]

        dists = np.linalg.norm(self.Z_[candidate_indices] - center_z, axis=1)
        order = np.argsort(dists)
        nearest_k_minus_1 = [candidate_indices[i] for i in order[: self.k - 1]]

        group = [center_idx] + nearest_k_minus_1
        remaining_pool = [idx for idx in U if idx not in group]
        group = self._ensure_e_diversity(group, remaining_pool)

        return group

    def _local_mdav(self, subset_indices):
        U = list(subset_indices)
        groups = []
        remainder = []

        print(
            f"\n  [Phase 2-子集] 开始在子集大小 {len(U)} 上执行局部 MDAV 式微聚集..."
        )

        while len(U) >= 2 * self.k:
            if self._sensitive_diversity(U) < self.e:
                print("    子集敏感值多样性不足，停止继续分组。")
                break

            center1 = self._farthest_from_centroid(U)
            group1 = self._form_group_with_constraints(U, center1)
            for idx in group1:
                if idx in U:
                    U.remove(idx)
            groups.append(group1)
            print(f"    形成分组 Group1，大小={len(group1)}")

            if len(U) < self.k or self._sensitive_diversity(U) < self.e:
                break

            center2 = self._farthest_from_centroid(U)
            group2 = self._form_group_with_constraints(U, center2)
            for idx in group2:
                if idx in U:
                    U.remove(idx)
            groups.append(group2)
            print(f"    形成分组 Group2，大小={len(group2)}")

        if len(U) >= self.k and self._sensitive_diversity(U) >= self.e:
            print(f"    剩余记录 {len(U)} 形成一个完整分组。")
            groups.append(list(U))
        else:
            if len(U) > 0:
                print(f"    剩余记录 {len(U)} 丢入全局剩余集合。")
            remainder.extend(U)

        print(f"  [Phase 2-子集] 完成，生成 {len(groups)} 个分组。")
        return groups, remainder

    # ========= Phase 3：remainder & 大组 =========
    def _secondary_partition_for_remainder(self, remainder_indices, groups):
        remainder_indices = list(remainder_indices)
        if not remainder_indices:
            return groups, []

        if len(remainder_indices) < self.max_group_size:
            print(
                f"\n[Phase 3-1] remainder 共 {len(remainder_indices)} 条 (< max_group_size)，跳过二次划分。"
            )
            return groups, remainder_indices

        print(
            f"\n[Phase 3-1] 对巨大 remainder (大小={len(remainder_indices)}) 进行二次划分..."
        )

        subsets2, rem2 = self._mondrian_partition_on_indices(
            remainder_indices, tag="remainder"
        )

        remainder_tail = list(rem2)

        for i, sb in enumerate(subsets2):
            print(f"\n  [Phase 3-1] 处理 remainder 子集 {i + 1}/{len(subsets2)}，大小={len(sb)}")
            lg, lr = self._local_mdav(sb)
            groups.extend(lg)
            remainder_tail.extend(lr)

        print(
            f"[Phase 3-1] remainder 二次划分完成：新生成分组 {len(subsets2)} 批，剩余尾巴 {len(remainder_tail)} 条。"
        )
        return groups, remainder_tail

    def _distribute_remainder_to_groups(self, remainder_indices, groups, verbose=True):
        """
        Phase 3-2：把剩余记录分配回已有分组，同时尽量不把任何组塞到超过 max_group_size。
        如果所有组都满员（>= max_group_size），则把无法分配的记录先放到缓冲区，
        缓冲区够 k 条时单独跑一轮局部 MDAV，否则才在无奈情况下打破 max_group_size 限制。
        """
        if not remainder_indices:
            if verbose:
                print("\n[Phase 3-2] 无剩余记录需要分配。")
            return

        if verbose:
            print(
                f"\n[Phase 3-2] 开始分配剩余记录，共 {len(remainder_indices)} 条，当前分组数 {len(groups)}。"
            )

        # 先算一遍初始质心
        def compute_centroids(grps):
            centroids_ = []
            for g in grps:
                Z_g = self.Z_[g, :]
                centroids_.append(Z_g.mean(axis=0))
            return np.vstack(centroids_)

        centroids = compute_centroids(groups)

        # 缓冲区：如果所有组都满了，先把记录丢这里
        buffer = []

        for idx in remainder_indices:
            # 找出仍有“空位”的组
            candidate_ids = [gi for gi, g in enumerate(groups) if len(g) < self.max_group_size]

            if candidate_ids:
                # 还有没塞满的组，就在这些组里找最近的
                z = self.Z_[idx]
                cand_centroids = centroids[candidate_ids]
                dists = np.linalg.norm(cand_centroids - z, axis=1)
                best_local = int(np.argmin(dists))
                best_group = candidate_ids[best_local]
                groups[best_group].append(idx)
                if verbose:
                    print(f"  剩余记录 {idx} 分配到分组 {best_group}（当前大小={len(groups[best_group])}）")
            else:
                # 所有组都已经 >= max_group_size，先暂存，后面单独处理
                buffer.append(idx)
                if verbose:
                    print(f"  所有分组已达到 max_group_size，记录 {idx} 先放入缓冲区。")

        # 处理缓冲区里的“无处可去”的记录
        if buffer:
            if verbose:
                print(f"[Phase 3-2] 缓冲区中共有 {len(buffer)} 条记录需要单独处理。")

            if len(buffer) >= self.k:
                # 缓冲区够大，可以单独跑一轮局部 MDAV
                print(f"  对缓冲区运行局部 MDAV，形成新的小分组...")
                lg, lr = self._local_mdav(buffer)
                groups.extend(lg)

                # 更新质心，方便后面可能的尾巴分配
                centroids = compute_centroids(groups)

                if lr:
                    # 仍有少量尾巴，再无奈地就近分配（这里允许轻微突破 max_group_size）
                    print(f"  缓冲区局部 MDAV 后仍有 {len(lr)} 条尾巴，忽略 max_group_size 做就近分配。")
                    for idx in lr:
                        z = self.Z_[idx]
                        dists = np.linalg.norm(centroids - z, axis=1)
                        best_group = int(np.argmin(dists))
                        groups[best_group].append(idx)
                        if verbose:
                            print(f"    尾巴记录 {idx} 分配到分组 {best_group}（当前大小={len(groups[best_group])}）")
            else:
                # 缓冲区太小 (<k)，无法单独成组，只好在少数情况下打破 max_group_size 限制
                print(
                    f"  缓冲区记录数 {len(buffer)} < k={self.k}，无法单独成组，"
                    f"在少数情况下允许突破 max_group_size 就近分配。"
                )
                centroids = compute_centroids(groups)
                for idx in buffer:
                    z = self.Z_[idx]
                    dists = np.linalg.norm(centroids - z, axis=1)
                    best_group = int(np.argmin(dists))
                    groups[best_group].append(idx)
                    if verbose:
                        print(f"    缓冲区记录 {idx} 分配到分组 {best_group}（当前大小={len(groups[best_group])}）")

        if verbose:
            print("[Phase 3-2] 剩余记录分配完成。")

    def _refine_large_groups(self, groups):
        print(
            f"\n[Phase 3-3] 检查并细化超过 max_group_size={self.max_group_size} 的大组..."
        )

        refined = []
        leftover = []

        for g in groups:
            if len(g) <= self.max_group_size:
                refined.append(g)
            else:
                print(f"  发现大组，大小={len(g)}，进行局部细化...")
                subsets_b, rem_b = self._mondrian_partition_on_indices(
                    g, tag="big-group"
                )
                leftover.extend(rem_b)

                for sb in subsets_b:
                    lg, lr = self._local_mdav(sb)
                    refined.extend(lg)
                    leftover.extend(lr)

        if leftover:
            print(
                f"[Phase 3-3] 细化后仍有 {len(leftover)} 条记录未成组，分配到最近的分组..."
            )
            self._distribute_remainder_to_groups(leftover, refined, verbose=False)

        print(f"[Phase 3-3] 大组细化完成，当前分组数 {len(refined)}")
        return refined

    # ========= 输出构造 =========
    def _build_anonymized_dataframe(self, groups):
        records = []
        data = self.data_

        for g in groups:
            g_sorted = sorted(g)
            group_df = data.iloc[g_sorted]

            generalized_record = {}

            for col in self.quasi_identifiers:
                col_series = group_df[col]
                if pd.api.types.is_numeric_dtype(col_series):
                    min_val = col_series.min()
                    max_val = col_series.max()
                    generalized_record[col] = [min_val, max_val]
                else:
                    generalized_record[col] = sorted(
                        col_series.dropna().unique().tolist()
                    )

            generalized_record[self.sensitive_attribute] = group_df[
                self.sensitive_attribute
            ].iloc[0]

            generalized_record["group_size"] = len(g_sorted)

            p_max, H = self._sensitive_stats(g_sorted)
            generalized_record["p_max"] = p_max
            generalized_record["sens_entropy"] = H

            records.append(generalized_record)

        anonymized_df = pd.DataFrame(records)
        return anonymized_df

    # ========= 主流程 =========
    def lrhma_algorithm(self, data: pd.DataFrame):
        print("开始执行自编码器 + 隐私增强版 LR-HMA 算法...")
        print(f"算法参数: k={self.k}, e={self.e}, max_group_size={self.max_group_size}")
        print(f"隐私参数: alpha_max={self.alpha_max}, entropy_min={self.entropy_min}")
        print(f"准标识符: {self.quasi_identifiers}")
        print(f"敏感属性: {self.sensitive_attribute}")
        print(f"设备: {self.device}")

        self.data_ = data.reset_index(drop=True).copy()

        # Phase 0
        print("\n=== Phase 0: 自编码器表示学习 ===")
        self._train_representation_encoder()
        self._encode_to_latent()

        # Phase 1
        subsets, remainder_p1 = self._mondrian_partition()

        # Phase 2
        print("\n=== Phase 2: 各子集内局部 (k,e) + 分布约束的 MDAV 微聚集 ===")
        all_groups = []
        remainder_p2 = []

        for i, subset in enumerate(subsets):
            print(f"\n--- 处理子集 {i + 1}/{len(subsets)}，大小={len(subset)} ---")
            local_groups, local_rem = self._local_mdav(subset)
            all_groups.extend(local_groups)
            remainder_p2.extend(local_rem)

        total_remainder = list(remainder_p1)
        total_remainder.extend(remainder_p2)
        print(
            f"\nPhase 1+2 结束：共生成 {len(all_groups)} 个初始分组，剩余记录 {len(total_remainder)} 条。"
        )

        # Phase 3-1
        all_groups, remainder_tail = self._secondary_partition_for_remainder(
            total_remainder, all_groups
        )

        # Phase 3-2
        self._distribute_remainder_to_groups(remainder_tail, all_groups)

        # Phase 3-3
        all_groups = self._refine_large_groups(all_groups)

        # 输出
        print("\n=== 创建最终匿名化数据集 ===")
        anonymized_df = self._build_anonymized_dataframe(all_groups)
        print(f"LR-HMA 完成! 生成 {len(all_groups)} 个等价类")
        print(f"最终数据集大小(行数=等价类数): {len(anonymized_df)}")

        self.last_groups_ = all_groups
        self.last_original_n_ = len(data)

        return anonymized_df, all_groups

    def run_from_excel(self, file_path, sheet_name=0):
        raw_data = self.load_data_from_excel(file_path, sheet_name)
        if raw_data is None:
            return None

        data = self.preprocess_data(raw_data)
        if data is None:
            return None

        print(f"预处理后数据形状: {data.shape}")
        print(data.head())

        result, groups = self.lrhma_algorithm(data)

        self.last_groups_ = groups
        self.last_original_n_ = len(data)
        return result


# ============  主函数  ============
if __name__ == "__main__":
    start_time = time.time()

    # ===== 1. 参数设置（按你的数据情况改） =====
    quasi_identifiers = [ 'age','marital_status','education','contact','duration','campaign']
    sensitive_attribute = 'job categorical'   # 注意要和表里的列名一致
    k = 8
    e = 3
    # 原始数据文件（放在和 LR-HMA.py 同一文件夹下）
    excel_file = "head10000.xlsx"
    sheet_name = "Sheet1"

    # ===== 2. 创建算法实例 =====
    lrhma = LRHMA(
        k=k,
        e=e,
        quasi_identifiers=quasi_identifiers,
        sensitive_attribute=sensitive_attribute,
        variance_threshold=0.05,
        embedding_dim=3,          # 潜在表示维度
        max_group_size=3 * k,     # 最大组大小，可调整成 2*k、4*k
        alpha_max=0.8,            # 单一敏感值最大占比
        entropy_min=0.0,          # 想加强隐私可改成 0.5 / 1.0 等
        ae_hidden_dims=[32],      # 自编码器隐藏层
        ae_epochs=50,             # 自编码器训练轮数
        ae_batch_size=128,
        ae_lr=1e-3,
        random_state=42,
        device="cpu",
    )

    print("=" * 60)
    print("自编码器 + LR-HMA 算法测试")
    print("=" * 60)
    print(f"当前工作路径: {os.getcwd()}")
    print(f"输入文件: {excel_file}")

    # ===== 3. 运行算法 =====
    result = lrhma.run_from_excel(excel_file, sheet_name)

    processing_time = time.time() - start_time

    # ===== 4. 输出与保存 =====
    if result is not None:
        print("\n" + "=" * 60)
        print("匿名化结果 (前5行):")
        print("=" * 60)
        print(result.head())

        # 结果文件名：head1000_lrhma_ae_improved.xlsx
        base_name = os.path.splitext(excel_file)[0]
        output_file = base_name + "_lrhma_ae_improved.xlsx"
        result.to_excel(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")

        # ===== 5. 一些统计信息，对标原代码风格 =====
        print("\n统计信息:")
        original_n = lrhma.last_original_n_ if lrhma.last_original_n_ is not None else "未知"
        num_groups = len(lrhma.last_groups_) if lrhma.last_groups_ is not None else 0

        print(f"原始记录数: {original_n}")
        print(f"匿名化后记录数(等价类数): {len(result)}")
        print(f"生成的等价类数: {num_groups}")
        if isinstance(original_n, int) and num_groups > 0:
            avg_size = original_n / num_groups
            print(f"平均等价类大小: {avg_size:.2f}")

        # 进一步的隐私指标：平均 p_max 和平均熵
        if "p_max" in result.columns and "sens_entropy" in result.columns:
            mean_p_max = result["p_max"].mean()
            max_p_max = result["p_max"].max()
            mean_entropy = result["sens_entropy"].mean()
            print(f"平均最大敏感值占比 p_max: {mean_p_max:.4f}")
            print(f"最大组的 p_max: {max_p_max:.4f}")
            print(f"平均敏感熵: {mean_entropy:.4f}")

        if "group_size" in result.columns:
            max_group_size = result["group_size"].max()
            mean_group_size = result["group_size"].mean()
            min_group_size = result["group_size"].min()
            print("\n等价类大小分布：")
            print(f"  最大组大小: {max_group_size}")
            print(f"  平均组大小: {mean_group_size:.2f}")
            print(f"  最小组大小: {min_group_size}")
        print(f"总处理时间: {processing_time:.2f} 秒")
    else:
        print("算法执行失败！")


