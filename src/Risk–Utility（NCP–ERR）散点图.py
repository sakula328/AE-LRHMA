import pandas as pd
import matplotlib.pyplot as plt

# ====== 1) 改成你自己的文件名（如果不一样就改这里） ======
BANK_XLSX = "bank_stability.xlsx"
ADULT_XLSX = "Adult_stability.xlsx"

# ====== 2) 选择用哪个 ext_frac 的攻击结果画图 ======
# 建议用 0.10（论文里也最好默认 10% 外部表），或者你也可以改成 0.20
TARGET_EXT_FRAC = 0.10

# ====== 3) 你的 NCP（从 compute_ncp.py 输出里抄的） ======
# Adult
NCP_ADULT = {
    "AE-LRHMA": 0.2808,
    "APMCA": 0.3861,
    "MDAV": 0.4431,
}
# Bank
NCP_BANK = {
    "AE-LRHMA": 0.3373,
    "APMCA": 0.4087,
    "MDAV": 0.4516,
}


def load_err_mean_std(xlsx_path: str, ext_frac: float) -> pd.DataFrame:
    """
    Read ERR_mean / ERR_std from the 'agg_mean_std' sheet.
    Return DataFrame with columns: method, ERR_mean, ERR_std
    """
    df = pd.read_excel(xlsx_path, sheet_name="agg_mean_std")
    # 兼容你表里 ext_frac 列可能叫 ext_frac 或 ext_frac_
    if "ext_frac" not in df.columns:
        raise ValueError(f"{xlsx_path} missing 'ext_frac' column. columns={list(df.columns)}")

    sub = df[df["ext_frac"].astype(float).round(2) == round(ext_frac, 2)].copy()
    if sub.empty:
        raise ValueError(f"{xlsx_path}: no rows found for ext_frac={ext_frac}. "
                         f"available ext_frac={sorted(df['ext_frac'].unique())}")

    need_cols = ["method", "ERR_mean", "ERR_std"]
    for c in need_cols:
        if c not in sub.columns:
            raise ValueError(f"{xlsx_path} missing column '{c}' in agg_mean_std. columns={list(sub.columns)}")

    return sub[need_cols].copy()


def build_points(err_df: pd.DataFrame, ncp_map: dict, dataset_tag: str) -> pd.DataFrame:
    """
    Merge ERR and NCP into one table: method, dataset, NCP, ERR_mean, ERR_std
    """
    rows = []
    for _, r in err_df.iterrows():
        method = str(r["method"])
        if method not in ncp_map:
            raise ValueError(f"{dataset_tag}: NCP not provided for method '{method}'. "
                             f"provided keys={list(ncp_map.keys())}")

        rows.append({
            "dataset": dataset_tag,
            "method": method,
            "NCP": float(ncp_map[method]),
            "ERR_mean": float(r["ERR_mean"]),
            "ERR_std": float(r["ERR_std"]),
        })
    return pd.DataFrame(rows)


def main():
    # Load ERR mean/std from xlsx
    bank_err = load_err_mean_std(BANK_XLSX, TARGET_EXT_FRAC)
    adult_err = load_err_mean_std(ADULT_XLSX, TARGET_EXT_FRAC)

    bank_pts = build_points(bank_err, NCP_BANK, "Bank")
    adult_pts = build_points(adult_err, NCP_ADULT, "Adult")

    pts = pd.concat([adult_pts, bank_pts], ignore_index=True)

    # Plot
    plt.figure(figsize=(10, 6))

    # 用不同 marker 区分数据集（不指定颜色，让 matplotlib 自动分配）
    marker_map = {"Adult": "o", "Bank": "s"}

    for dataset in ["Adult", "Bank"]:
        sub = pts[pts["dataset"] == dataset].copy()

        # scatter
        plt.scatter(sub["NCP"], sub["ERR_mean"], marker=marker_map[dataset], s=180, label=dataset)

        # error bar（可选：ERR_std）
        plt.errorbar(sub["NCP"], sub["ERR_mean"], yerr=sub["ERR_std"],
                     fmt="none", capsize=3)

        # annotate
        for _, r in sub.iterrows():
            short = "A" if dataset == "Adult" else "B"
            label = f"{r['method']} ({short})"
            plt.text(r["NCP"] + 0.003, r["ERR_mean"] + 0.002, label, fontsize=11)

    plt.title("Risk–Utility Scatter (Adult & Bank)")
    plt.xlabel("NCP (utility loss; lower is better)")
    plt.ylabel("ERR (linkage risk; lower is better)")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(title="Dataset")

    # 保存
    out_png = "risk_utility_adult_bank.png"
    out_pdf = "risk_utility_adult_bank.pdf"
    plt.tight_layout()
    plt.savefig(out_png, dpi=1000)
    plt.savefig(out_pdf)
    # plt.show()

    print(f"[OK] Saved: {out_png}, {out_pdf}")
    print("\nPoints used:")
    print(pts.sort_values(["dataset", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()

