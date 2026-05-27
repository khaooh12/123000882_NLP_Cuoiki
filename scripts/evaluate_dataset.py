"""
evaluate_dataset.py — Đánh giá toàn diện dataset (tất cả nguồn)
Chạy: python evaluate_dataset.py
"""
import pandas as pd
import numpy as np
import os
from collections import Counter

SEP  = "=" * 64
SEP2 = "-" * 50

# ── Load tất cả file ────────────────────────────────────────
vne  = pd.read_csv("data/raw/vnexpress_articles.csv")
vntc = pd.read_csv("data/raw/vntc_articles.csv")
news = pd.read_csv("data/raw/newsgroups_articles.csv")
all_ = pd.read_csv("data/raw/all_articles.csv")
cln  = pd.read_csv("data/processed/cleaned_articles.csv")
emb  = np.load("data/processed/embeddings.npy")

print(SEP)
print("  DANH GIA DATASET TONG HOP")
print(SEP)

# ============================================================
# 1. TONG QUAN TUNG NGUON
# ============================================================
print("\n[1] TONG QUAN THEO NGUON")
print(SEP2)
src_info = [
    ("VnExpress (scraping)",   vne,  "vi"),
    ("VNTC-10Topics (HF)",     vntc, "vi"),
    ("20 Newsgroups (Kaggle)", news, "en"),
]
for label, df, lang in src_info:
    n     = len(df)
    t_len = df["title"].str.len().fillna(0)
    d_len = df["description"].str.len().fillna(0)
    miss  = df["description"].isna().sum() + (df["description"].fillna("") == "").sum()
    cats  = df["category"].nunique()
    t_avg = t_len.mean()
    d_avg = d_len[d_len > 0].mean() if (d_len > 0).any() else 0
    print(f"  {label}")
    print(f"    Bai viet       : {n:>5}  |  Ngon ngu: {lang}")
    print(f"    Chuyen muc     : {cats} nhom")
    miss_pct = miss / n * 100
    print(f"    Thieu mo ta    : {miss} ({miss_pct:.1f}%)")
    print(f"    Title len TB   : {t_avg:.0f} ky tu")
    print(f"    Desc  len TB   : {d_avg:.0f} ky tu")
    print()

vi_count = all_[all_["language"] == "vi"].shape[0] if "language" in all_.columns else len(all_)
en_count = all_[all_["language"] == "en"].shape[0] if "language" in all_.columns else 0
print(f"  TONG CONG (raw) : {len(all_)} bai  (vi={vi_count} / en={en_count})")

# ============================================================
# 2. PHAN BO CHUYEN MUC
# ============================================================
print("\n[2] PHAN BO CHUYEN MUC — all_articles.csv")
print(SEP2)
cat_counts = all_["category"].value_counts()
max_cnt    = cat_counts.max()
print(f"  {'Chuyen muc':<20}  {'Tong':>5}  {'VI':>5}  {'EN':>5}  {'%':>6}  Bar")
print(f"  {'-'*20}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  ---")
for cat, cnt in cat_counts.items():
    if "language" in all_.columns:
        vi = int(all_[(all_["category"] == cat) & (all_["language"] == "vi")].shape[0])
        en = int(cnt - vi)
    else:
        vi, en = cnt, 0
    pct = cnt / len(all_) * 100
    bar = "#" * int(cnt / max_cnt * 25)
    print(f"  {cat:<20}  {cnt:>5}  {vi:>5}  {en:>5}  {pct:>5.1f}%  {bar}")

# ============================================================
# 3. KIEM TRA TRUNG LAP
# ============================================================
print("\n[3] KIEM TRA TRUNG LAP — all_articles.csv")
print(SEP2)
dup_url   = all_.duplicated(subset=["url"]).sum()
dup_title = all_.duplicated(subset=["title"]).sum()
dup_desc  = all_.duplicated(subset=["description"]).sum()
print(f"  URL trung         : {dup_url}  {'OK' if dup_url == 0 else '[!]'}")
print(f"  Tieu de trung     : {dup_title}  {'OK' if dup_title == 0 else '[!] (do bien the tieu de)'}")
print(f"  Mo ta trung       : {dup_desc}  {'OK' if dup_desc == 0 else '[!] (binh thuong voi template data)'}")

# ============================================================
# 4. CHAT LUONG NOI DUNG
# ============================================================
print("\n[4] CHAT LUONG NOI DUNG — all_articles.csv")
print(SEP2)
all_["title"]       = all_["title"].fillna("")
all_["description"] = all_["description"].fillna("")
all_["combined"]    = (all_["title"] + " " + all_["description"]).str.strip()
all_["word_count"]  = all_["combined"].str.split().str.len().fillna(0)
all_["char_count"]  = all_["combined"].str.len().fillna(0)

groups = [("ALL", all_)]
if "language" in all_.columns:
    groups += [("VI ", all_[all_["language"] == "vi"]),
               ("EN ", all_[all_["language"] == "en"])]

for tag, sub in groups:
    if sub.empty:
        continue
    wc    = sub["word_count"]
    cc    = sub["char_count"]
    short = int((wc < 10).sum())
    print(f"  [{tag}] {len(sub):>5} bai")
    print(f"    Tu/bai   : min={wc.min():.0f}  max={wc.max():.0f}  "
          f"mean={wc.mean():.1f}  median={wc.median():.0f}")
    print(f"    Ky tu    : min={cc.min():.0f}  max={cc.max():.0f}  "
          f"mean={cc.mean():.1f}  median={cc.median():.0f}")
    pct_short = short / len(sub) * 100
    print(f"    Qua ngan (<10 tu): {short} ({pct_short:.1f}%)")

# ============================================================
# 5. VOCABULARY — tieng Viet
# ============================================================
print("\n[5] VOCABULARY — chi tieng Viet")
print(SEP2)
vi_df  = all_[all_["language"] == "vi"] if "language" in all_.columns else all_
vi_tok = []
for t in vi_df["combined"]:
    vi_tok.extend(str(t).lower().split())
vocab_vi = set(vi_tok)
freq     = Counter(vi_tok)
avg_tok  = len(vi_tok) / len(vi_df) if len(vi_df) > 0 else 0
print(f"  Tong token         : {len(vi_tok):,}")
print(f"  Tu vung rieng biet : {len(vocab_vi):,}")
print(f"  TB token/bai       : {avg_tok:.1f}")
print(f"  Top-10 tu pho bien :")
for w, c in freq.most_common(10):
    pct = c / len(vi_tok) * 100
    print(f"    {w:<25}: {c:>5}  ({pct:.2f}%)")

# ============================================================
# 6. PROCESSED DATA (cleaned / embeddings)
# ============================================================
print("\n[6] TINH TRANG PROCESSED DATA")
print(SEP2)
cln_src = cln["source"].value_counts().to_dict() if "source" in cln.columns else {"(unknown)": len(cln)}
print(f"  cleaned_articles.csv : {len(cln)} bai")
for k, v in cln_src.items():
    print(f"    source={k}: {v}")
print(f"  embeddings.npy       : shape={emb.shape}  dtype={emb.dtype}")
nan_count  = int(np.isnan(emb).sum())
zero_rows  = int((np.abs(emb).sum(axis=1) == 0).sum())
norms      = np.linalg.norm(emb, axis=1)
print(f"  NaN / Zero-vector    : {nan_count} / {zero_rows}  {'OK' if nan_count == 0 and zero_rows == 0 else '[!]'}")
print(f"  L2 Norm TB           : {norms.mean():.4f}")

gap = len(all_) - len(cln)
print(f"\n  *** can re-run preprocess.py va embedding.py ***")
print(f"  *** cleaned con thieu {gap} bai chua xu ly ***")

# ============================================================
# 7. CAN BANG DU LIEU
# ============================================================
print("\n[7] CAN BANG DU LIEU")
print(SEP2)
cat_s  = cat_counts.std()
cat_m  = cat_counts.mean()
cv     = cat_s / cat_m
ratio  = cat_counts.max() / cat_counts.min()
print(f"  So chuyen muc     : {len(cat_counts)}")
print(f"  Nhieu nhat        : {cat_counts.idxmax()} ({cat_counts.max()} bai)")
print(f"  It nhat           : {cat_counts.idxmin()} ({cat_counts.min()} bai)")
print(f"  Ti le max/min     : {ratio:.1f}x")
print(f"  He so bien thien  : {cv:.2f}  {'(can bang)' if cv < 0.5 else '(mat can bang!)'}")

vi_cats  = vi_df["category"].value_counts()
vi_ratio = vi_cats.max() / vi_cats.min()
print(f"\n  [Chi tieng Viet — {len(vi_df)} bai]")
for cat, cnt in vi_cats.items():
    bar = "#" * int(cnt / vi_cats.max() * 20)
    pct = cnt / len(vi_df) * 100
    print(f"    {cat:<20}: {cnt:>4} ({pct:.1f}%)  {bar}")
print(f"  Ti le max/min VI  : {vi_ratio:.1f}x  {'OK' if vi_ratio < 4 else '[!] Mat can bang'}")

# ============================================================
# 8. TOM TAT & KHUYEN NGHI
# ============================================================
print(f"\n{SEP}")
print("  TOM TAT & KHUYEN NGHI")
print(SEP)

issues = []
if dup_title > 0:
    issues.append(f"  [!] {dup_title} tieu de trung lap (do bien the template — chap nhan duoc)")
if dup_desc > 0:
    issues.append(f"  [!] {dup_desc} mo ta trung (VnExpress dummy + VNTC co the lap) — nen tang scrape that")
if gap > 0:
    issues.append(f"  [!] cleaned/embeddings cu ({len(cln)} bai) — can re-run preprocess + embedding")
vi_min_cat = vi_cats.min()
vi_min_name = vi_cats.idxmin()
if vi_min_cat < 100:
    issues.append(f"  [!] '{vi_min_name}' chi co {vi_min_cat} bai VI — nen bo sung")

print(f"""
  RAW DATASET TONG HOP:
    VnExpress (scraping)  :   700 bai  [vi]
    VNTC-10Topics (HF)    : 1.500 bai  [vi]
    20 Newsgroups (Kaggle): 3.000 bai  [en]
    ─────────────────────────────────────
    Tong all_articles.csv : 5.200 bai
    Tieng Viet (VI)       : 2.200 bai  <- dung cho topic modeling chinh
    Tieng Anh  (EN)       : 3.000 bai  <- dung de test/validate pipeline

  DANH GIA MO HINH:
    BERTopic (HDBSCAN)    : 2.200 VI  ->  DU (khuyen nghi min 500)
    LDA baseline          : 2.200 VI  ->  DU
    Coherence c_v         : Se tinh sau khi retrain
    Silhouette Score      : Se tinh sau khi retrain

  CAC VAN DE HIEN TAI:""")

if issues:
    for issue in issues:
        print(issue)
else:
    print("  Khong co van de nghiem trong.")

print(f"""
  LENH CAN CHAY TIEP:
    # Xu ly tieng Viet (khuyen nghi):
    python src/preprocess.py --lang vi

    # Hoac xu ly toan bo (5.200 bai):
    python src/preprocess.py

    # Sau do:
    python src/embedding.py
    python src/topic_model.py
""")
print(SEP)
