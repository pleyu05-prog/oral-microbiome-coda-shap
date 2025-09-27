# scripts/prepare_hmp_oral.py
# 适配 Kaggle: antaresnyc/human-metagenomics 的 abundance.csv
import pandas as pd, numpy as np, os
from pathlib import Path
ORAL_KEYS = ["oral","mouth","saliva","tongue","buccal","plaque","gingiva","keratinized","hard palate","tonsil","subgingival","supragingival"]

def clr_transform(rel):
    rel = rel + 1e-9
    log_rel = np.log(rel)
    gm = log_rel.mean(axis=1).values.reshape(-1,1)
    return log_rel - gm

def main():
    data_dir = Path("data")
    # 递归寻找 abundance.csv（行=样本，列=物种/特征 + 若干元数据列）
    cand = [p for p in data_dir.rglob("abundance.csv")]
    if not cand:
        raise SystemExit("未找到 data/.../abundance.csv，请先下载数据集 human-metagenomics。")
    p = cand[0]
    print(f"[INFO] using {p}")

    # 先读前几行看列名
    head = pd.read_csv(p, nrows=5)
    cols = [c.lower() for c in head.columns]

    # 典型元数据列名猜测
    site_col = None
    for k in ["body_site","bodysite","site","subsite","body_subsite","bodysubsite"]:
        if k in cols: site_col = head.columns[cols.index(k)]; break

    # 读全表
    df = pd.read_csv(p)

    # 若有样本ID列，优先用；否则用 DataFrame 索引
    sid_col = None
    for k in ["sample","sample_id","sid","run","id"]:
        if k in cols: sid_col = head.columns[cols.index(k)]; break
    if sid_col is None:
        df.index.name = "sample_id"
    else:
        df = df.set_index(sid_col)

    # 识别特征列：保留数值型且非常见元数据列
    non_feat = set()
    if site_col: non_feat.add(site_col)
    for k in ["dataset","study","country","disease","cohort","subject","age","sex"]:
        if k in cols:
            non_feat.add(head.columns[cols.index(k)])
    num_mask = df.apply(lambda s: np.issubdtype(s.dtype, np.number)).rename("is_num")
    feat_cols = [c for c in df.columns if num_mask[c] and c not in non_feat]

    # 仅保留“计数/丰度”特征；去掉全零与出现率<1%的稀疏列
    X = df[feat_cols].copy()
    present = (X > 0).mean(0)
    X = X.loc[:, present >= 0.01]

    # 相对丰度 → CLR
    X = X.div(X.sum(1).replace(0, np.nan), axis=0).fillna(0)
    X = pd.DataFrame(clr_transform(X), index=X.index, columns=X.columns)

    # 生成标签：body_site/subsite
    if site_col is None:
        raise SystemExit("没有发现 body_site/subsite 等列，无法筛口腔样本。")
    y = df[site_col].astype(str)

    # 只保留“口腔”相关样本
    is_oral = y.str.lower().apply(lambda s: any(k in s for k in ORAL_KEYS))
    X_oral, y_oral = X.loc[is_oral], y.loc[is_oral]
    assert len(X_oral) > 0, "筛选后没有口腔样本，请检查 site 列取值。"

    out = Path("data/omics"); out.mkdir(parents=True, exist_ok=True)
    X_oral.to_csv(out / "features_clr.csv")
    y_oral.rename("site").to_csv(out / "labels.csv", header=True)
    print(f"[OK] 保存: {out/'features_clr.csv'}, {out/'labels.csv'}")
    print("[HINT] 口腔位点计数(Top10):")
    print(y_oral.value_counts().head(10))

if __name__ == "__main__":
    main()
