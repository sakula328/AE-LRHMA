import os
import time
import pandas as pd
import numpy as np

from LR_HMA import LRHMA  # 你的类在 LR_HMA.py 里

# --- metrics helpers ---
from compute_ncp import (
    compute_ncp_dataset,
    filter_existing_qi,
    ADULT_QI_NUMERIC, ADULT_QI_CATEGORICAL,
    BANK_QI_NUMERIC, BANK_QI_CATEGORICAL,
)
from linkage_attack_eval import (
    build_ec_table_ae,
    compute_metrics,
    infer_orig_dtypes,
)

# ---------------------------
# 你需要在这里配置：数据 + QI + SA
# ---------------------------
DATASETS = {
    "Adult": {
        "input_file": "a_head10000.xlsx",
        "sheet": 0,
        "qi": ["age", "marital_status", "education", "contact", "duration", "campaign"],
        "sa": "job categorical",
        "categorical_qi": ["marital_status", "education", "contact"],
    },
    "Bank": {
        "input_file": "head10000.xlsx",
        "sheet": 0,
        "qi": ["age", "marital_status", "education", "contact", "duration", "campaign"],
        "sa": "job categorical",
        "categorical_qi": ["marital_status", "education", "contact"],
    }
}

# linkage-attack 里用的 ext_frac（你图里 3 个值）
EXT_FRACS = [0.05, 0.10, 0.20]
# linkage-attack 多随机种子（mean ± std）
ATTACK_SEEDS = [0, 1, 2, 3, 4]

# EXT_FRACS = [0.10]
# ATTACK_SEEDS = [0]   # 或 [0,1] 先小跑


# LRHMA 参数（按你论文/主实验设置）
LRHMA_PARAMS = dict(
    k=8,
    e=3,
    variance_threshold=0.05,
    embedding_dim=3,
    max_group_size=None,     # None = 3*k（你类里默认）
    alpha_max=0.8,
    entropy_min=0.0,
    ae_hidden_dims=[32],
    ae_epochs=50,
    ae_batch_size=128,
    ae_lr=1e-3,
    random_state=42,
    device="cpu",
)

OUTDIR = "review_encoding_runs"


def _compute_ncp_for_dataset(dataset_name: str, anon_xlsx_path: str, orig_df: pd.DataFrame) -> float:
    """Compute NCP using your compute_ncp.py implementation."""
    anon_df = pd.read_excel(anon_xlsx_path)

    if dataset_name.lower() == "adult":
        qi_num, qi_cat = ADULT_QI_NUMERIC, ADULT_QI_CATEGORICAL
    else:
        qi_num, qi_cat = BANK_QI_NUMERIC, BANK_QI_CATEGORICAL

    # ensure we only use existing columns (robust to method outputs)
    qi_num, qi_cat = filter_existing_qi(anon_df, qi_num, qi_cat, tag=f"{dataset_name}/NCP")

    ncp = compute_ncp_dataset(
        anon_df=anon_df,
        qi_numeric=qi_num,
        qi_categorical=qi_cat,
        cluster_size_col="group_size",
        orig_df=orig_df,
        tag=f"{dataset_name}/NCP"
    )
    return float(ncp)


def _compute_linkage_err_umr(orig_df: pd.DataFrame, anon_xlsx_path: str, qi_cols: list) -> dict:
    """Compute linkage-attack ERR/UMR for each ext_frac, aggregated over ATTACK_SEEDS."""
    # linkage_attack_eval expects no NaN in QI columns
    orig = orig_df.dropna(subset=qi_cols).reset_index(drop=True)
    orig_dtypes = infer_orig_dtypes(orig, qi_cols)

    ec = build_ec_table_ae(anon_xlsx_path, qi_cols)

    out = {}
    for ext_frac in EXT_FRACS:
        ext_n = max(1, int(len(orig) * float(ext_frac)))
        print(f"[LINKAGE] ext_frac={ext_frac:.2f} ext_n={ext_n} start", flush=True)#新增进度条

        per_seed = []
        for seed in ATTACK_SEEDS:
            t1 = time.time()#新增进度显示
            print(f"[LINKAGE]   seed={seed} ...", end="", flush=True)#新增进度显示

            external = orig.sample(n=ext_n, random_state=int(seed)).reset_index(drop=True)
            m = compute_metrics(ec, external, qi_cols, orig_dtypes)
            per_seed.append(m)

            print(f" done in {time.time() - t1:.1f}s  ERR={m['ERR']:.4f} UMR={m['UMR']:.4f}", flush=True)#新增进度显示

        # aggregate mean/std over seeds
        err_vals = [x["ERR"] for x in per_seed]
        umr_vals = [x["UMR"] for x in per_seed]

        key = f"{ext_frac:.2f}"
        out[f"ERR_{key}_mean"] = float(np.mean(err_vals))
        out[f"ERR_{key}_std"] = float(np.std(err_vals, ddof=1)) if len(err_vals) > 1 else 0.0
        out[f"UMR_{key}_mean"] = float(np.mean(umr_vals))
        out[f"UMR_{key}_std"] = float(np.std(umr_vals, ddof=1)) if len(umr_vals) > 1 else 0.0

    # keep a convenient default ERR/UMR = ext_frac 0.10 mean (as your previous summary columns)
    out["ERR"] = out["ERR_0.10_mean"]
    out["UMR"] = out["UMR_0.10_mean"]
    return out


def run_one(dataset_name, cfg, variant_name, encoding_mode, cat_perm_seed):
    os.makedirs(OUTDIR, exist_ok=True)

    # 1) 读数据
    df = pd.read_excel(cfg["input_file"], sheet_name=cfg["sheet"])
    df = df.dropna(subset=cfg["qi"] + [cfg["sa"]]).reset_index(drop=True)

    #修改
    # Only force categorical QIs to str when we truly need categorical handling (onehot)
    if encoding_mode == "onehot":
        for c in cfg.get("categorical_qi", []):
            if c in df.columns:
                df[c] = df[c].astype(str)

    # 2) 运行 LRHMA
    t0 = time.time()
    model = LRHMA(
        quasi_identifiers=cfg["qi"],
        sensitive_attribute=cfg["sa"],
        encoding_mode=encoding_mode,
        cat_perm_seed=cat_perm_seed,
        **LRHMA_PARAMS
    )
    anon_df, groups = model.lrhma_algorithm(df)
    runtime = time.time() - t0

    # 3) 保存匿名化结果
    out_xlsx = os.path.join(OUTDIR, f"{dataset_name}__{variant_name}.xlsx")
    anon_df.to_excel(out_xlsx, index=False)

    # 4) groups stats
    n = len(df)
    g = len(groups)
    avg_size = n / g if g > 0 else np.nan

    # 5) 自动算指标：NCP + linkage ERR/UMR
    ncp = _compute_ncp_for_dataset(dataset_name, out_xlsx, orig_df=df)

    linkage = _compute_linkage_err_umr(orig_df=df, anon_xlsx_path=out_xlsx, qi_cols=cfg["qi"])

    # 统一返回结构（summary 直接写 xlsx）
    metrics = {
        "dataset": dataset_name,
        "variant": variant_name,
        "encoding_mode": encoding_mode,
        "cat_perm_seed": cat_perm_seed,
        "n_records": n,
        "n_groups": g,
        "avg_group_size": avg_size,
        "runtime_sec": runtime,
        "anon_file": out_xlsx,
        "NCP": ncp,
        **linkage
    }

    return metrics


def main():
    all_rows = []

    for dname, cfg in DATASETS.items():
        # A) 编码对比：integer vs onehot
        all_rows.append(run_one(dname, cfg, "AE_LRHMA_integer", "integer", None))
        all_rows.append(run_one(dname, cfg, "AE_LRHMA_onehot", "onehot", None))

        # B) integer 置换敏感性：5次
        for s in range(5):
            all_rows.append(run_one(dname, cfg, f"AE_LRHMA_intperm{s}", "integer", s))

    summary = pd.DataFrame(all_rows)
    summary_file = os.path.join(OUTDIR, "summary_metrics.xlsx")
    summary.to_excel(summary_file, index=False)
    print(f"[OK] Saved summary to: {summary_file}")

    # 额外：为“置换敏感性”生成 mean±std 表（包含 NCP + 全部 ERR/UMR 列）
    perm = summary[summary["variant"].str.contains("intperm", na=False)].copy()
    if len(perm) > 0:
        metric_cols = ["NCP", "ERR", "UMR"] + [c for c in perm.columns if c.startswith("ERR_") or c.startswith("UMR_")]
        metric_cols = [c for c in metric_cols if c in perm.columns]

        agg = perm.groupby("dataset")[metric_cols].agg(["mean","std"])
        agg_file = os.path.join(OUTDIR, "integer_permutation_mean_std.xlsx")
        agg.to_excel(agg_file)
        print(f"[OK] Saved permutation stats to: {agg_file}")


if __name__ == "__main__":
    main()
