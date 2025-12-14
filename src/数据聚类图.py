import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.cluster import KMeans
from LR_HMA import LRHMA   # 确保文件名是 LR_HMA.py，类名是 LRHMA


def get_axis_limits(Z, margin=0.5, symmetric=False):
    """
    根据 latent 表示 Z 计算 3 个坐标轴的范围，用于统一两张图的坐标。
    symmetric=True 时以 0 为中心对称；False 时保持真实 min/max。
    """
    x_min, x_max = Z[:, 0].min(), Z[:, 0].max()
    y_min, y_max = Z[:, 1].min(), Z[:, 1].max()
    z_min, z_max = Z[:, 2].min(), Z[:, 2].max()

    if symmetric:
        x_abs = max(abs(x_min), abs(x_max))
        y_abs = max(abs(y_min), abs(y_max))
        z_abs = max(abs(z_min), abs(z_max))
        x_min, x_max = -x_abs, x_abs
        y_min, y_max = -y_abs, y_abs
        z_min, z_max = -z_abs, z_abs

    # 加一点边距，避免点贴在边框上
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    z_min -= margin
    z_max += margin

    return x_min, x_max, y_min, y_max, z_min, z_max


def main():
    # ===== 1. 与你平时实验一致的参数 =====
    excel_file = "head10000.xlsx"   # 换成你实际使用的 Adult / Bank 文件
    sheet_name = "Sheet1"

    quasi_identifiers = [
        'age', 'marital_status', 'education',
        'contact', 'duration', 'campaign'
    ]
    sensitive_attribute = 'job categorical'

    k = 8
    e = 3

    lrhma = LRHMA(
        k=k,
        e=e,
        quasi_identifiers=quasi_identifiers,
        sensitive_attribute=sensitive_attribute,
        variance_threshold=0.05,
        embedding_dim=3,          # 3 维 latent，方便 3D 画图
        max_group_size=3 * k,
        alpha_max=0.8,
        entropy_min=0.0,
        ae_hidden_dims=[32],
        ae_epochs=50,
        ae_batch_size=128,
        ae_lr=1e-3,
        random_state=42,
    )

    # ===== 2. 运行 AE-LRHMA 算法 =====
    result = lrhma.run_from_excel(excel_file, sheet_name)
    if result is None:
        print("LR-HMA 运行失败，无法可视化。")
        return

    # 自编码器得到的潜在表示（所有记录）
    Z = lrhma.Z_  # shape: (n_samples, 3)
    groups = lrhma.last_groups_  # 等价类索引列表

    n_samples = Z.shape[0]

    # ===== A. 给每个样本一个等价类 ID（group_id） =====
    group_id = np.full(n_samples, -1, dtype=int)
    for gid, idx_list in enumerate(groups):
        for idx in idx_list:
            group_id[idx] = gid

    # 有些可能没被分到 group，过滤掉
    mask = group_id >= 0
    Z = Z[mask]
    group_id = group_id[mask]

    # ---- 在这里计算统一的坐标范围（两张图都会用） ----
    x_min, x_max, y_min, y_max, z_min, z_max = get_axis_limits(
        Z, margin=0.5, symmetric=False
    )

    # ===== B. 计算每个等价类在 latent 空间的中心 =====
    group_centers = []
    gid_list = []
    for gid, idx_list in enumerate(groups):
        idx_list = np.array(idx_list, dtype=int)
        if len(idx_list) == 0:
            continue

        # 只保留在 mask 内的索引（原始索引）
        real_idx = [i for i in idx_list if i < n_samples and mask[i]]
        if len(real_idx) == 0:
            continue

        # 将原始索引转换成 Z（已 mask 后）的行号
        mask_indices = np.where(mask)[0]  # 原始索引 -> Z 行号 的映射数组
        pos_list = [np.where(mask_indices == i)[0][0] for i in real_idx]
        center = Z[pos_list].mean(axis=0)

        group_centers.append(center)
        gid_list.append(gid)

    group_centers = np.vstack(group_centers)

    # ===== C. 在“等价类中心”上做 KMeans，再合成 10 个大簇 =====
    n_meta_clusters = 10
    kmeans = KMeans(n_clusters=n_meta_clusters, random_state=42)
    meta_labels_for_groups = kmeans.fit_predict(group_centers)

    # 建立 group_id -> meta_cluster 的映射
    gid_to_meta = {gid: meta for gid, meta in zip(gid_list, meta_labels_for_groups)}

    # 给每条记录一个 meta_cluster ID
    meta_id = np.full(n_samples, -1, dtype=int)
    for gid, idx_list in enumerate(groups):
        meta = gid_to_meta.get(gid, -1)
        if meta == -1:
            continue
        for idx in idx_list:
            if idx < n_samples:
                meta_id[idx] = meta

    # 和上面的 mask 对齐
    meta_id = meta_id[mask]

    # 统一字体（英文 + 数学符号友好）
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # ===== D. 图1：原始 latent 分布（可以抽样） =====
    max_points_before = 8000
    if Z.shape[0] > max_points_before:
        idx_before = np.random.choice(Z.shape[0], max_points_before, replace=False)
        Z_before = Z[idx_before]
    else:
        Z_before = Z

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.scatter(
        Z_before[:, 0],
        Z_before[:, 1],
        Z_before[:, 2],
        c="#888888",
        s=3,
        alpha=0.5,
        edgecolors='none'
    )

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_zlabel("Latent dim 3")
    ax.set_title("AE latent space before anonymization")

    # 统一坐标范围 + 视角
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=20, azim=-60)

    plt.tight_layout()
    plt.savefig("bank_ae_latent_before.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已保存图像：bank_ae_latent_before.png")

    # ===== E. 图2：AE-LRHMA groups（10 个 meta-clusters，每个颜色很多点） =====
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    cmap = plt.get_cmap("tab10")

    for c in range(n_meta_clusters):
        m = (meta_id == c)
        if not m.any():
            continue

        ax.scatter(
            Z[m, 0],
            Z[m, 1],
            Z[m, 2],
            c=[cmap(c)],
            s=5,
            alpha=0.9,
            edgecolors='none',
            label=f"Cluster {c + 1}",
        )

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_zlabel("Latent dim 3")
    ax.set_title("AE-LRHMA groups in latent space")

    # 和图1完全一致的坐标范围与视角
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.view_init(elev=20, azim=-60)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        fontsize=8,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig("bank_ae_lrhma_groups.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("已保存图像：bank_ae_lrhma_groups.png")


if __name__ == "__main__":
    main()

