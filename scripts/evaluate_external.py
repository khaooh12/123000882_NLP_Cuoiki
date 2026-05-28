"""
Đánh giá ngoại sinh TF-IDF + KMeans dựa trên nhãn category thực tế.
Chỉ số: NMI, ARI, Purity, Homogeneity, Completeness, V-measure, Confusion matrix
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH   = "data/processed/cleaned_articles.csv"
RESULTS_DIR = "results"
K_MIN, K_MAX = 3, 10
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.85


def purity_score(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    return float(cm.max(axis=0).sum()) / len(y_true)


def run_external_eval(k, X, y_true_encoded):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        normalized_mutual_info_score, adjusted_rand_score,
        homogeneity_completeness_v_measure,
    )

    km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300)
    labels = km.fit_predict(X)

    nmi  = normalized_mutual_info_score(y_true_encoded, labels, average_method="arithmetic")
    ari  = adjusted_rand_score(y_true_encoded, labels)
    purity = purity_score(y_true_encoded, labels)
    hom, com, vms = homogeneity_completeness_v_measure(y_true_encoded, labels)

    return labels, {
        "k":            k,
        "NMI":          round(nmi, 4),
        "ARI":          round(ari, 4),
        "Purity":       round(purity, 4),
        "Homogeneity":  round(hom, 4),
        "Completeness": round(com, 4),
        "V-measure":    round(vms, 4),
    }


def confusion_table(y_true, labels, label_names, k):
    """Trả về DataFrame: hàng = nhãn thực, cột = cụm KMeans."""
    n_labels = len(label_names)
    cm = np.zeros((n_labels, k), dtype=int)
    for true_id, cluster_id in zip(y_true, labels):
        cm[true_id, cluster_id] += 1
    df = pd.DataFrame(cm, index=label_names, columns=[f"C{i}" for i in range(k)])
    df["TOTAL"] = df.sum(axis=1)
    df.loc["TOTAL"] = df.sum(axis=0)
    return df


def dominant_label_per_cluster(y_true, labels, label_names, k):
    rows = []
    for ci in range(k):
        mask = labels == ci
        if not mask.any():
            rows.append({"Cụm": ci, "Nhãn chủ đạo": "—", "Số bài nhãn đó": 0, "Tổng cụm": 0, "Độ thuần (%)": 0})
            continue
        vals = y_true[mask]
        dominant_idx = np.bincount(vals).argmax()
        dominant_name = label_names[dominant_idx]
        dom_count = int((vals == dominant_idx).sum())
        total = int(mask.sum())
        rows.append({
            "Cụm": ci,
            "Nhãn chủ đạo": dominant_name,
            "Số bài nhãn đó": dom_count,
            "Tổng cụm": total,
            "Độ thuần (%)": round(dom_count / total * 100, 1),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[Data] Đọc dữ liệu...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df = df[df["cleaned_text"].notna() & (df["cleaned_text"].str.strip().str.len() >= 10)].copy()
    df = df[df["category"].notna()].copy()

    label_names = sorted(df["category"].unique())
    label2id = {l: i for i, l in enumerate(label_names)}
    y_true = df["category"].map(label2id).values

    docs = df["cleaned_text"].astype(str).tolist()
    print(f"  {len(docs):,} bài hợp lệ, {len(label_names)} nhãn: {label_names}")

    print("\n[TF-IDF] Vectorize...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF, max_df=TFIDF_MAX_DF)
    X = vec.fit_transform(docs)
    print(f"  Ma trận: {X.shape}")

    all_metrics = []
    all_labels  = {}

    print(f"\n{'='*65}")
    print(f"  Đánh giá ngoại sinh TF-IDF + KMeans  (k={K_MIN}..{K_MAX})")
    print(f"{'='*65}")

    for k in range(K_MIN, K_MAX + 1):
        labels, metrics = run_external_eval(k, X, y_true)
        all_labels[k]   = labels
        all_metrics.append(metrics)
        print(f"\n[k={k}]")
        for key, val in metrics.items():
            if key != "k":
                print(f"  {key:<14}: {val:.4f}")

    # Lưu CSV chỉ số ngoại sinh
    ext_csv = os.path.join(RESULTS_DIR, "external_eval_results.csv")
    pd.DataFrame(all_metrics).to_csv(ext_csv, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {ext_csv}")

    # In bảng tổng kết
    print(f"\n{'='*65}")
    print("  BẢNG TỔNG HỢP ĐÁNH GIÁ NGOẠI SINH")
    print(f"{'='*65}")
    hdr = f"{'k':>3}  {'NMI':>6}  {'ARI':>6}  {'Purity':>7}  {'Homog.':>7}  {'Complet.':>9}  {'V-meas.':>8}"
    print(hdr)
    print("-" * len(hdr))
    for m in all_metrics:
        print(f"{m['k']:>3}  {m['NMI']:>6.4f}  {m['ARI']:>6.4f}  {m['Purity']:>7.4f}  "
              f"{m['Homogeneity']:>7.4f}  {m['Completeness']:>9.4f}  {m['V-measure']:>8.4f}")

    # Confusion matrix & độ thuần cụm cho k=8 (bằng số nhãn thực)
    for k_show in [7, 8, 9]:
        labels = all_labels[k_show]
        print(f"\n--- Nhãn chủ đạo mỗi cụm (k={k_show}) ---")
        dom_df = dominant_label_per_cluster(y_true, labels, label_names, k_show)
        print(dom_df.to_string(index=False))

        conf_df = confusion_table(y_true, labels, label_names, k_show)
        conf_path = os.path.join(RESULTS_DIR, f"confusion_matrix_k{k_show}.csv")
        conf_df.to_csv(conf_path, encoding="utf-8-sig")
        print(f"[Saved] {conf_path}")

    # Lưu JSON đầy đủ
    json_path = os.path.join(RESULTS_DIR, "external_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
