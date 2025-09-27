# scripts/train_microbiome_clf.py
# 多类别（口腔不同位点）分类：LogReg(L1) + XGBoost + SHAP 可解释
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ASSETS = Path("assets")

def load_xy():
    X = pd.read_csv("data/omics/features_clr.csv", index_col=0)
    y = pd.read_csv("data/omics/labels.csv", index_col=0)["site"].astype(str)
    X = X.loc[y.index]
    return X, y

def eval_and_plot(clf, Xtr, Xte, ytr, yte, name):
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    f1  = f1_score(yte, yp, average="macro")
    print(f"{name}: acc={acc:.3f}, macroF1={f1:.3f}")
        # 混淆矩阵（把下划线换成换行、加大画布、旋转 x 轴标签）
    labels = sorted(yte.unique())
    disp_labels = [s.replace("_", "\n") for s in labels]

    cm = confusion_matrix(yte, yp, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))  # ← 原来是 (5,5)
    ConfusionMatrixDisplay(cm, display_labels=disp_labels).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.tick_params(axis="x", labelrotation=30, labelsize=9)  # ← 旋转+减小字号
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(f"{name} Confusion Matrix")
    fig.subplots_adjust(bottom=0.28)  # ← 给 x 轴标签留空间
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / f"{name}_confusion.png", dpi=150)
    plt.close(fig)

    return acc, f1

def shap_bar(values, feat_names, k, out_png, title):
    vals = values
    if isinstance(vals, list):         # 多分类可能返回 list[(n,d)]*C
        vals = np.mean([np.abs(v) for v in vals], axis=0)
    vals = np.asarray(vals)
    if vals.ndim == 3:                 # (n,C,d)
        vals = np.mean(np.abs(vals), axis=(0,1))
    else:                              # (n,d)
        vals = np.mean(np.abs(vals), axis=0)
    idx = np.argsort(vals)[-k:]
    plt.figure(figsize=(6,4))
    plt.barh(range(len(idx)), vals[idx])
    plt.yticks(range(len(idx)), np.array(feat_names)[idx])
    plt.xlabel("mean |SHAP|"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def main():
    X, y = load_xy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 1) 逻辑回归（移除 multi_class 参数，避免未来警告）
    logreg = LogisticRegression(penalty="l1", solver="saga", max_iter=5000, n_jobs=-1, C=1.0)
    acc1, f11 = eval_and_plot(logreg, Xtr, Xte, ytr, yte, "LogRegL1")

    # 2) XGBoost（需要整数标签）
    le = LabelEncoder()
    ytr_enc = le.fit_transform(ytr)
    yte_enc = le.transform(yte)

    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.7,
        tree_method="hist", eval_metric="mlogloss", n_jobs=4
    )
    xgb.fit(Xtr, ytr_enc)
    pred_enc = xgb.predict(Xte)
    pred = le.inverse_transform(pred_enc)

    acc2 = accuracy_score(yte, pred)
    f12  = f1_score(yte, pred, average="macro")
    print(f"XGBoost: acc={acc2:.3f}, macroF1={f12:.3f}")

    labels = sorted(yte.unique())
    disp_labels = [s.replace("_", "\n") for s in labels]

    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=disp_labels).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.tick_params(axis="x", labelrotation=30, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title("XGBoost Confusion Matrix")
    fig.subplots_adjust(bottom=0.28)
    fig.tight_layout()
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / "XGBoost_confusion.png", dpi=150)
    plt.close(fig)


    # === SHAP ===
    try:
        expl_lr = shap.Explainer(logreg, Xtr, feature_names=X.columns)
        sv_lr = expl_lr(Xte).values
        shap_bar(sv_lr, X.columns, 20, ASSETS/"shap_top20_linear.png", "Top-20 SHAP (LogReg)")
    except Exception as e:
        print(f"[WARN] SHAP for LogReg failed: {e}. Fallback to coef_.")
        coef = np.mean(np.abs(logreg.coef_), axis=0)
        idx = np.argsort(coef)[-20:]
        plt.figure(figsize=(6,4))
        plt.barh(range(len(idx)), coef[idx])
        plt.yticks(range(len(idx)), np.array(X.columns)[idx])
        plt.xlabel("|coef|"); plt.title("Top-20 (LogReg coef fallback)")
        plt.tight_layout(); plt.savefig(ASSETS/"shap_top20_linear.png", dpi=150); plt.close()

    try:
        expl_xgb = shap.TreeExplainer(xgb)
        sv_xgb = expl_xgb.shap_values(Xte)
        shap_bar(sv_xgb, X.columns, 20, ASSETS/"shap_top20_xgb.png", "Top-20 SHAP (XGB)")
    except Exception as e:
        print(f"[WARN] SHAP for XGB failed: {e}. Skipped.")

    with open(ASSETS/"results.txt","w",encoding="utf-8") as f:
        f.write(f"LogRegL1 acc={acc1:.3f} macroF1={f11:.3f}\n")
        f.write(f"XGBoost  acc={acc2:.3f} macroF1={f12:.3f}\n")
    print("[OK] 保存图表到 assets/，结果到 assets/results.txt")

if __name__ == "__main__":
    main()
