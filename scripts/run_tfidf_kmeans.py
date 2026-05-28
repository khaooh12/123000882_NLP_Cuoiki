"""
Thực nghiệm TF-IDF + KMeans
- Chạy với dải k từ K_MIN đến K_MAX
- Tính Silhouette score (sklearn) và Coherence c_v (gensim)
- Lưu kết quả ra results/tfidf_kmeans_results.json và results/tfidf_kmeans_results.csv
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Cấu hình thực nghiệm ──────────────────────────────────────────────────────
DATA_PATH    = "data/processed/cleaned_articles.csv"
RESULTS_DIR  = "results"
K_MIN, K_MAX = 3, 10          # dải số cụm cần thử
TFIDF_MAX_FEATURES = 5000     # số đặc trưng TF-IDF
TFIDF_MIN_DF = 3              # bỏ từ xuất hiện < 3 doc
TFIDF_MAX_DF = 0.85           # bỏ từ xuất hiện > 85% doc
TOP_WORDS    = 15             # số từ đại diện mỗi cụm để tính Coherence
SAMPLE_SIZE  = None           # None = dùng toàn bộ; số nguyên để lấy mẫu nhanh
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path, sample_size=None):
    print(f"[Data] Đọc {path} ...")
    df = pd.read_csv(path, encoding="utf-8-sig")
    docs = df["cleaned_text"].fillna("").astype(str).tolist()
    docs = [d for d in docs if len(d.strip()) >= 10]
    if sample_size and sample_size < len(docs):
        import random; random.seed(42)
        docs = random.sample(docs, sample_size)
    print(f"  -> {len(docs):,} văn bản hợp lệ")
    return docs


def build_tfidf(docs):
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(f"\n[TF-IDF] Vectorize (max_features={TFIDF_MAX_FEATURES}) ...")
    t0 = time.time()
    vec = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
    )
    X = vec.fit_transform(docs)
    print(f"  -> Ma trận TF-IDF: {X.shape}  ({time.time()-t0:.1f}s)")
    return X, vec


def get_top_words_per_cluster(labels, docs, feature_names, k, top_n=TOP_WORDS):
    """Trích top TF-IDF words của từng cụm để tính Coherence."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    cluster_top_words = []
    for ci in range(k):
        cluster_docs = [docs[i] for i in range(len(docs)) if labels[i] == ci]
        if not cluster_docs:
            cluster_top_words.append([])
            continue
        try:
            cvec = TfidfVectorizer(vocabulary=feature_names, min_df=1)
            mat  = cvec.fit_transform(cluster_docs)
            scores = np.asarray(mat.mean(axis=0)).flatten()
            top_idx = scores.argsort()[::-1][:top_n]
            cluster_top_words.append([feature_names[i] for i in top_idx])
        except Exception:
            cluster_top_words.append([])
    return cluster_top_words


def calc_coherence(cluster_top_words, tokenized_docs):
    """Tính Coherence c_v với gensim CoherenceModel."""
    try:
        from gensim import corpora
        from gensim.models.coherencemodel import CoherenceModel

        # Lọc cụm có từ
        valid_topics = [tw for tw in cluster_top_words if tw]
        if not valid_topics:
            return None

        dictionary = corpora.Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=2, no_above=0.85)

        cm = CoherenceModel(
            topics=valid_topics,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
        )
        return float(cm.get_coherence())
    except Exception as e:
        print(f"    [Coherence] Lỗi gensim: {e}")
        return None


def run_experiment(k, X, docs, feature_names, tokenized_docs):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    print(f"\n[KMeans] k={k} ...")
    t0 = time.time()

    km = KMeans(n_clusters=k, random_state=42, n_init="auto", max_iter=300)
    labels = km.fit_predict(X)
    elapsed_km = time.time() - t0

    # Silhouette — tính trên tập con nếu quá lớn (tránh OOM)
    n_sample = min(len(docs), 10000)
    idx_sample = np.random.choice(len(docs), n_sample, replace=False) if len(docs) > n_sample else np.arange(len(docs))
    sil = float(silhouette_score(X[idx_sample], labels[idx_sample], metric="cosine", sample_size=None))

    # Inertia (WCSS)
    inertia = float(km.inertia_)

    # Top words mỗi cụm
    cluster_top_words = get_top_words_per_cluster(labels, docs, feature_names, k)

    # Coherence
    t1 = time.time()
    coherence = calc_coherence(cluster_top_words, tokenized_docs)
    elapsed_coh = time.time() - t1

    # Phân bố cụm
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}

    print(f"  Silhouette  : {sil:.4f}")
    print(f"  Coherence   : {coherence:.4f}" if coherence is not None else "  Coherence   : N/A (gensim không có)")
    print(f"  Inertia     : {inertia:,.0f}")
    print(f"  Time KMeans : {elapsed_km:.1f}s  |  Coherence: {elapsed_coh:.1f}s")
    print(f"  Phân bố cụm : {dist}")

    row = {
        "k":               k,
        "silhouette":      round(sil, 6),
        "coherence_cv":    round(coherence, 6) if coherence is not None else None,
        "inertia":         round(inertia, 2),
        "time_kmeans_s":   round(elapsed_km, 2),
        "time_coherence_s": round(elapsed_coh, 2),
        "cluster_dist":    dist,
        "top_words":       {str(i): cluster_top_words[i] for i in range(k)},
    }
    return row


def main():
    np.random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Tải dữ liệu
    docs = load_data(DATA_PATH, SAMPLE_SIZE)
    tokenized_docs = [d.split() for d in docs]

    # 2. TF-IDF
    X, vec = build_tfidf(docs)
    feature_names = list(vec.get_feature_names_out())

    # 3. Thực nghiệm với từng k
    all_rows = []
    print(f"\n{'='*60}")
    print(f"  Thực nghiệm TF-IDF + KMeans  (k={K_MIN}..{K_MAX})")
    print(f"{'='*60}")

    for k in range(K_MIN, K_MAX + 1):
        row = run_experiment(k, X, docs, feature_names, tokenized_docs)
        all_rows.append(row)

    # 4. Lưu JSON (đầy đủ, kể cả top_words)
    json_path = os.path.join(RESULTS_DIR, "tfidf_kmeans_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {json_path}")

    # 5. Lưu CSV (chỉ điểm số, không kèm top_words)
    summary_rows = [
        {k2: v for k2, v in row.items() if k2 not in ("cluster_dist", "top_words")}
        for row in all_rows
    ]
    csv_path = os.path.join(RESULTS_DIR, "tfidf_kmeans_results.csv")
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")

    # 6. In bảng tổng kết
    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ THỰC NGHIỆM TF-IDF + KMeans")
    print(f"{'='*60}")
    header = f"{'k':>4}  {'Silhouette':>12}  {'Coherence(c_v)':>14}  {'Inertia':>14}"
    print(header)
    print("-" * len(header))
    for row in all_rows:
        coh = f"{row['coherence_cv']:.4f}" if row["coherence_cv"] is not None else "    N/A   "
        print(f"{row['k']:>4}  {row['silhouette']:>12.4f}  {coh:>14}  {row['inertia']:>14,.0f}")

    # 7. Chọn k tốt nhất
    valid = [r for r in all_rows if r["silhouette"] is not None]
    best_sil = max(valid, key=lambda r: r["silhouette"])
    print(f"\n  >> k tốt nhất theo Silhouette: k={best_sil['k']}  (sil={best_sil['silhouette']:.4f})")
    if any(r["coherence_cv"] for r in all_rows):
        best_coh = max((r for r in all_rows if r["coherence_cv"]), key=lambda r: r["coherence_cv"])
        print(f"  >> k tốt nhất theo Coherence : k={best_coh['k']}  (c_v={best_coh['coherence_cv']:.4f})")


if __name__ == "__main__":
    main()
