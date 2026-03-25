import pandas as pd
import numpy as np
import math
import ast
import re

# ============================================================
#  解析区间 / 类别
# ============================================================

def parse_interval(val):
    """
    把 '[np.int64(19), np.int64(38)]' 或 '[22,26]' 等字符串
    解析成 (lo, hi) 浮点数；单值 '[35]' 或 '35' 解析成 (35,35)
    """
    if isinstance(val, (int, float, np.number)):
        v = float(val)
        return v, v
    if not isinstance(val, str):
        return float("nan"), float("nan")

    s = val.strip()
    try:
        # 把 'np.int64' 去掉，再 literal_eval
        cleaned = s.replace("np.int64", "")
        obj = ast.literal_eval(cleaned)
        if isinstance(obj, (list, tuple)):
            nums = [float(x) for x in obj]
        else:
            nums = [float(obj)]
    except Exception:
        # 兜底：直接抓所有数字
        nums = [float(x) for x in re.findall(r"-?\d+", s)]
        if not nums:
            return float("nan"), float("nan")

    return min(nums), max(nums)


def parse_cat_set(val):
    """
    解析类别集合：
    - '0\\1\\2' -> {'0','1','2'}
    - '0,1,2'   -> {'0','1','2'}
    - '[np.int64(0), np.int64(4)]' -> {'0','1','2','3','4'}
    """
    if isinstance(val, (int, float, np.integer)):
        return {str(int(val))}
    if not isinstance(val, str):
        return set()

    s = val.strip()
    # 区间形式，转成多个类别
    if "[" in s and "np.int64" in s:
        lo, hi = parse_interval(s)
        if math.isnan(lo):
            return set()
        return {str(i) for i in range(int(lo), int(hi) + 1)}

    if "\\" in s:
        parts = [p for p in s.split("\\") if p != ""]
    else:
        parts = [p for p in s.split(",") if p != ""]

    return set(parts)


# ============================================================
#  统一 QI 列定义（关键：同一数据集、所有方法用同一组列）
# ============================================================

# Adult 上用于 NCP 的 QI
ADULT_QI_NUMERIC = ["age", "duration", "campaign"]
ADULT_QI_CATEGORICAL = ["marital_status", "education", "contact", "job categorical"]

# Bank 上用于 NCP 的 QI
BANK_QI_NUMERIC = ["age", "duration", "campaign"]
BANK_QI_CATEGORICAL = ["education", "contact", "job categorical"]

def filter_existing_qi(df, qi_numeric, qi_categorical, tag=""):
    """
    确保只使用在 df 中真实存在的列，避免某些方法缺列报错。
    同时打印实际参与 NCP 计算的列，方便你检查“是否一致”。
    """
    num_cols = [c for c in qi_numeric if c in df.columns]
    cat_cols = [c for c in qi_categorical if c in df.columns]

    missing = [c for c in qi_numeric + qi_categorical if c not in df.columns]
    if missing:
        print(f"[{tag}] 下列列在结果表中不存在，将不会参与 NCP 计算：{missing}")
    print(f"[{tag}] NCP 使用的数值型列: {num_cols}")
    print(f"[{tag}] NCP 使用的类别型列: {cat_cols}")

    return num_cols, cat_cols


# ============================================================
#  NCP 主函数
# ============================================================

def compute_ncp_dataset(anon_df,
                        qi_numeric,
                        qi_categorical,
                        cluster_size_col,
                        orig_df=None,
                        tag=""):
    """
    计算一个匿名化数据集的 NCP。

    anon_df:           等价类结果 DataFrame（每一行代表一个等价类）
    qi_numeric:        数值型 QI 列名列表
    qi_categorical:    类别型 QI 列名列表
    cluster_size_col:  等价类大小所在列名（MDAV/AE-LRHMA 是 group_size，APMCA 是 cluster_size）
    orig_df:           用来确定全局取值范围的“基准数据集”（统一用原始数据）
    tag:               打印用的小标签（Adult-APMCA 等）
    """

    import math

    # 统一过滤：只保留真实存在的 QI 列
    qi_numeric, qi_categorical = filter_existing_qi(
        anon_df, qi_numeric, qi_categorical, tag=tag
    )

    # 1) 选择做全局范围的基准数据：优先用原始数据
    if orig_df is None:
        base_df = anon_df
    else:
        base_df = orig_df

    # 2) 数值属性的全局 [gmin, gmax]
    num_ranges = {}
    for col in qi_numeric:
        lows, highs = [], []
        for v in base_df[col]:
            lo, hi = parse_interval(v)
            if not math.isnan(lo):
                lows.append(lo)
                highs.append(hi)
        if not lows:
            num_ranges[col] = (0.0, 1.0)
            continue
        gmin, gmax = min(lows), max(highs)
        if gmax == gmin:
            gmax = gmin + 1.0
        num_ranges[col] = (gmin, gmax)

    # 3) 类别属性的全局取值集合
    cat_domains = {}
    for col in qi_categorical:
        dom = set()
        for v in base_df[col]:
            dom |= parse_cat_set(v)
        cat_domains[col] = dom

    # 4) 按等价类加权的 NCP
    N = float(anon_df[cluster_size_col].sum())
    total_ncp = 0.0
    attr_count = len(qi_numeric) + len(qi_categorical)

    for _, row in anon_df.iterrows():
        size = float(row[cluster_size_col])
        if size <= 0 or attr_count == 0:
            continue

        group_ncp_sum = 0.0

        # 数值型 QI
        for col in qi_numeric:
            lo, hi = parse_interval(row[col])
            gmin, gmax = num_ranges[col]
            total_len = gmax - gmin
            if total_len <= 0:
                ncp_attr = 0.0
            else:
                ncp_attr = (hi - lo) / total_len
                # 强制限定在 [0,1]
                ncp_attr = max(0.0, min(1.0, ncp_attr))
            group_ncp_sum += ncp_attr

        # 类别型 QI
        for col in qi_categorical:
            cats = parse_cat_set(row[col])
            dom = cat_domains[col]
            if not dom:
                ncp_attr = 0.0
            else:
                ncp_attr = len(cats) / len(dom)
                ncp_attr = max(0.0, min(1.0, ncp_attr))
            group_ncp_sum += ncp_attr

        # 每个等价类的 NCP ∈ [0,1]
        group_ncp = group_ncp_sum / attr_count

        # 按等价类大小加权
        total_ncp += (size / N) * group_ncp

    return total_ncp


# ============================================================
#  主程序：保证「同一数据集内，三种方法用同一组列」
# ============================================================

if __name__ == "__main__":
    # ========= Adult: 读取原始数据 =========
    # 用原始数据 a_head10000.xlsx 作为全局范围的基准
    orig_adult = pd.read_excel("a_head10000.xlsx")

    # ========= Adult: APMCA =========
    apmca_adult = pd.read_excel("apmca_parallel_result_Adult.xlsx")
    print("Adult APMCA 列名：", apmca_adult.columns.tolist())
    ncp_apmca_adult = compute_ncp_dataset(
        anon_df=apmca_adult,
        qi_numeric=ADULT_QI_NUMERIC,
        qi_categorical=ADULT_QI_CATEGORICAL,
        cluster_size_col="cluster_size",
        orig_df=orig_adult,
        tag="Adult-APMCA",
    )
    print(f"APMCA Adult NCP = {ncp_apmca_adult:.4f}")

    # ========= Adult: MDAV =========
    mdav_adult = pd.read_excel("a_head10000.xlsx-strict_kemdav_result.xlsx")
    print("Adult MDAV 列名：", mdav_adult.columns.tolist())

    if "group_size" not in mdav_adult.columns:
        # 如果没有组大小列，可以根据你的实现补上（这里暂时假设 k=8）
        mdav_adult["group_size"] = 8

    ncp_mdav_adult = compute_ncp_dataset(
        anon_df=mdav_adult,
        qi_numeric=ADULT_QI_NUMERIC,
        qi_categorical=ADULT_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_adult,
        tag="Adult-MDAV",
    )
    print(f"MDAV Adult  NCP = {ncp_mdav_adult:.4f}")

    # ========= Adult: AE-LRHMA =========
    ael_adult = pd.read_excel("a_head10000_lrhma_ae_improved.xlsx")
    print("Adult AE-LRHMA 列名：", ael_adult.columns.tolist())

    ncp_ael_adult = compute_ncp_dataset(
        anon_df=ael_adult,
        qi_numeric=ADULT_QI_NUMERIC,
        qi_categorical=ADULT_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_adult,
        tag="Adult-AE-LRHMA",
    )
    print(f"AE-LRHMA Adult NCP = {ncp_ael_adult:.4f}")

    # ========= Bank: 读取原始数据 =========
    orig_bank = pd.read_excel("head10000.xlsx")

    # ========= Bank: APMCA =========
    apmca_bank = pd.read_excel("apmca_parallel_result_bank.xlsx")
    print("Bank APMCA 列名：", apmca_bank.columns.tolist())

    ncp_apmca_bank = compute_ncp_dataset(
        anon_df=apmca_bank,
        qi_numeric=BANK_QI_NUMERIC,
        qi_categorical=BANK_QI_CATEGORICAL,
        cluster_size_col="cluster_size",  # APMCA 的等价类大小列
        orig_df=orig_bank,
        tag="Bank-APMCA",
    )
    print(f"APMCA Bank NCP = {ncp_apmca_bank:.4f}")

    # ========= Bank: MDAV =========
    mdav_bank = pd.read_excel("head10000.xlsx-strict_kemdav_result.xlsx")
    print("Bank MDAV 列名：", mdav_bank.columns.tolist())

    if "group_size" not in mdav_bank.columns:
        mdav_bank["group_size"] = 8  # 没有的话同样补一列

    ncp_mdav_bank = compute_ncp_dataset(
        anon_df=mdav_bank,
        qi_numeric=BANK_QI_NUMERIC,
        qi_categorical=BANK_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_bank,
        tag="Bank-MDAV",
    )
    print(f"MDAV Bank NCP = {ncp_mdav_bank:.4f}")

    # ========= Bank: AE-LRHMA =========
    ael_bank = pd.read_excel("head10000_lrhma_ae_improved.xlsx")
    print("Bank AE-LRHMA 列名：", ael_bank.columns.tolist())

    ncp_ael_bank = compute_ncp_dataset(
        anon_df=ael_bank,
        qi_numeric=BANK_QI_NUMERIC,
        qi_categorical=BANK_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_bank,
        tag="Bank-AE-LRHMA",
    )
    print(f"AE-LRHMA Bank NCP = {ncp_ael_bank:.4f}")

    # Adult 原始数据
    orig_adult = pd.read_excel("a_head10000.xlsx")

    # Adult w/o-diversity
    adult_wodiv = pd.read_excel("Adult-w_o-diversity_lrhma.xlsx")  # 你实际保存的名字
    ncp_adult_wodiv = compute_ncp_dataset(
        anon_df=adult_wodiv,
        qi_numeric=ADULT_QI_NUMERIC,
        qi_categorical=ADULT_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_adult,
        tag="Adult-w/o-diversity",
    )
    print(f"Adult w/o-diversity NCP = {ncp_adult_wodiv:.4f}")

    # Adult w/o-AE
    adult_woae = pd.read_excel("Adult-w_o-AE_lrhma.xlsx")
    ncp_adult_woae = compute_ncp_dataset(
        anon_df=adult_woae,
        qi_numeric=ADULT_QI_NUMERIC,
        qi_categorical=ADULT_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_adult,
        tag="Adult-w/o-AE",
    )
    print(f"Adult w/o-AE NCP = {ncp_adult_woae:.4f}")

    # Bank 原始数据
    orig_bank = pd.read_excel("head10000.xlsx")

    # Bank w/o-diversity
    bank_wodiv = pd.read_excel("Bank-w_o-diversity_lrhma.xlsx")
    ncp_bank_wodiv = compute_ncp_dataset(
        anon_df=bank_wodiv,
        qi_numeric=BANK_QI_NUMERIC,
        qi_categorical=BANK_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_bank,
        tag="Bank-w/o-diversity",
    )
    print(f"Bank w/o-diversity NCP = {ncp_bank_wodiv:.4f}")

    # Bank w/o-AE
    bank_woae = pd.read_excel("Bank-w_o-AE_lrhma.xlsx")
    ncp_bank_woae = compute_ncp_dataset(
        anon_df=bank_woae,
        qi_numeric=BANK_QI_NUMERIC,
        qi_categorical=BANK_QI_CATEGORICAL,
        cluster_size_col="group_size",
        orig_df=orig_bank,
        tag="Bank-w/o-AE",
    )
    print(f"Bank w/o-AE NCP = {ncp_bank_woae:.4f}")




