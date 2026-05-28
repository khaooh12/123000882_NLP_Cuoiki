"""
data_loader.py — Hợp nhất và kiểm tra trùng lặp URL trên các tập dữ liệu thô.

Hai chức năng chính:
  1. merge_all_sources()  — Gộp tất cả file nguồn thành all_articles.csv
  2. check_duplicates()   — Báo cáo / làm sạch trùng lặp URL trong từng file
"""

import os
import glob
import argparse
import pandas as pd
import sys
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# File bị loại khỏi quá trình gộp
MERGE_EXCLUDE = {"all_articles.csv", "test_articles.csv", "newsgroups_articles.csv"}

# Thứ tự ưu tiên nguồn khi gộp (nguồn chất lượng cao ưu tiên giữ khi trùng URL)
SOURCE_PRIORITY = [
    "vntc", "znews", "wiki-dump", "wiki-mini", "wiki-qa",
    "wiki_dump", "wiki_mini", "wiki_qa",
    "vnexpress", "dantri", "tienphong", "nhandan",
    "thanhnien", "tuoitre", "vietnamnet",
]

STANDARD_COLS = ["url", "title", "description", "category", "publish_date", "source", "language"]


# ──────────────────────────────────────────────────────────────────────────────
# 1. MERGE ALL SOURCES
# ──────────────────────────────────────────────────────────────────────────────

def merge_all_sources(
    raw_dir: str = "data/raw",
    output_path: str = "data/raw/all_articles.csv",
    lang_filter: str = "vi",
) -> pd.DataFrame:
    """
    Gộp tất cả file *_articles.csv trong raw_dir (trừ MERGE_EXCLUDE) thành
    một DataFrame duy nhất, lọc theo ngôn ngữ, khử trùng URL, rồi ghi ra output_path.

    Tham số:
        raw_dir      — thư mục chứa các file nguồn
        output_path  — đường dẫn file đầu ra (all_articles.csv)
        lang_filter  — 'vi' / 'en' / 'all'

    Trả về: DataFrame sau khi hợp nhất và khử trùng.
    """
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    source_files = [f for f in csv_files if os.path.basename(f) not in MERGE_EXCLUDE]

    if not source_files:
        print(f"[!] Không tìm thấy file nguồn nào trong {raw_dir}")
        return pd.DataFrame()

    print("=" * 80)
    print("  BƯỚC 1 — ĐỌC & CHUẨN HOÁ TỪNG NGUỒN")
    print("=" * 80)

    frames = []
    for fpath in source_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, dtype=str)
        except Exception as e:
            print(f"  [Lỗi] Không đọc được {fname}: {e}")
            continue

        # Đảm bảo đủ cột chuẩn
        for col in STANDARD_COLS:
            if col not in df.columns:
                df[col] = ""

        df = df[STANDARD_COLS].copy()

        # Chuẩn hoá kiểu chuỗi, tránh float NaN
        for col in STANDARD_COLS:
            df[col] = df[col].fillna("").astype(str).str.strip()

        # Lọc ngôn ngữ
        if lang_filter != "all":
            before = len(df)
            df = df[df["language"] == lang_filter]
            if len(df) == 0:
                print(f"  ⚠️  {fname:<35}: 0 bài sau lọc lang='{lang_filter}' (bỏ qua)")
                continue

        desc_fill = int((df["description"] != "").sum())
        print(
            f"  📁 {fname:<35}: {len(df):>7,} bài  "
            f"desc={desc_fill/max(len(df),1):.0%}"
        )
        frames.append(df)

    if not frames:
        print("[!] Không có dữ liệu nào để gộp.")
        return pd.DataFrame()

    print()
    print("=" * 80)
    print("  BƯỚC 2 — HỢP NHẤT VÀ KHỬ TRÙNG URL")
    print("=" * 80)

    combined = pd.concat(frames, ignore_index=True)
    total_raw = len(combined)
    print(f"  Tổng sau concat    : {total_raw:>8,} bài")

    # Bỏ hàng không có URL
    combined = combined[combined["url"] != ""]
    print(f"  Sau lọc URL rỗng   : {len(combined):>8,} bài")

    # Sắp xếp theo độ ưu tiên nguồn trước khi drop_duplicates
    # → keep='first' sẽ giữ bản có nguồn chất lượng cao nhất
    priority_map = {src: i for i, src in enumerate(SOURCE_PRIORITY)}
    combined["_prio"] = combined["source"].map(priority_map).fillna(len(SOURCE_PRIORITY))
    combined = combined.sort_values("_prio").drop(columns=["_prio"])

    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    dupes_removed = before_dedup - len(combined)
    print(f"  Sau khử trùng URL  : {len(combined):>8,} bài  (loại {dupes_removed:,} URL trùng)")

    print()
    print("=" * 80)
    print("  PHÂN BỐ NGUỒN SAU HỢP NHẤT")
    print("=" * 80)
    for src, cnt in combined["source"].value_counts().items():
        desc_rate = (combined.loc[combined["source"] == src, "description"] != "").mean()
        print(f"  {src:<20}: {cnt:>7,} bài  desc={desc_rate:.0%}")

    print()
    print("=" * 80)
    print("  PHÂN BỐ CHUYÊN MỤC SAU HỢP NHẤT")
    print("=" * 80)
    for cat, cnt in combined["category"].value_counts().head(20).items():
        pct = cnt / len(combined) * 100
        print(f"  {cat:<25}: {cnt:>7,} ({pct:.1f}%)")

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    print()
    print(f"  ✅ Đã ghi {len(combined):,} bài → {output_path}")

    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 2. CHECK DUPLICATES (giữ nguyên từ phiên bản cũ)
# ──────────────────────────────────────────────────────────────────────────────

def check_duplicates(clean_files: bool):
    raw_dir = "data/raw"
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    target_files = [f for f in csv_files if os.path.basename(f) not in {"newsgroups_articles.csv"}]

    if not target_files:
        print(f"[!] Không tìm thấy file CSV nào trong {raw_dir}")
        return

    print("=" * 80)
    print("  BÁO CÁO TRÙNG LẶP URL NỘI BỘ")
    print("=" * 80)

    url_to_sources = defaultdict(list)
    file_stats = []

    for fpath in sorted(target_files):
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath)
            total_rows = len(df)
            if "url" not in df.columns:
                print(f"  [Cảnh báo] {fname} không có cột 'url'. Bỏ qua.")
                continue
            unique_urls = df["url"].nunique()
            duplicates = total_rows - unique_urls
            file_stats.append({"file": fname, "path": fpath, "total": total_rows,
                                "unique": unique_urls, "duplicates": duplicates, "df": df})
            for url in df["url"].dropna():
                url_to_sources[url].append(fname)
            dup_pct = duplicates / total_rows * 100 if total_rows > 0 else 0
            print(f"  📁 {fname:<30}: {total_rows:>7,} | Duy nhất: {unique_urls:>7,} | Trùng: {duplicates:>5,} ({dup_pct:.1f}%)")
        except Exception as e:
            print(f"  [Lỗi] {fname}: {e}")

    print("\n" + "=" * 80)
    print("  TRÙNG LẶP CHÉO GIỮA CÁC NGUỒN")
    print("=" * 80)

    overlap_count = 0
    overlap_pairs = defaultdict(int)
    for url, sources in url_to_sources.items():
        if len(sources) > 1:
            overlap_count += 1
            unique_sources = sorted(set(sources))
            for i in range(len(unique_sources)):
                for j in range(i + 1, len(unique_sources)):
                    overlap_pairs[(unique_sources[i], unique_sources[j])] += 1

    if overlap_count == 0:
        print("  ✅ Không có URL trùng chéo giữa các nguồn.")
    else:
        print(f"  ⚠️  {overlap_count:,} URL xuất hiện ở nhiều nguồn:")
        for (s1, s2), cnt in sorted(overlap_pairs.items(), key=lambda x: -x[1]):
            print(f"    {s1} <-> {s2} : {cnt:,} bài")

    if clean_files:
        print("\n" + "=" * 80)
        print("  DỌN DẸP TRÙNG LẶP NỘI BỘ")
        print("=" * 80)
        for stat in file_stats:
            if stat["duplicates"] > 0:
                df_clean = stat["df"].drop_duplicates(subset=["url"], keep="first")
                try:
                    df_clean.to_csv(stat["path"], index=False, encoding="utf-8-sig")
                    print(f"  🧹 {stat['file']}: loại {stat['duplicates']:,} bài → còn {len(df_clean):,}")
                except Exception as e:
                    print(f"  [Lỗi] {stat['file']}: {e}")
            else:
                print(f"  ✅ {stat['file']}: sạch sẽ.")
        print("\n  [Hoàn tất]")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset Loader & Merger cho NLP Cuối Khoá")
    parser.add_argument("--merge", action="store_true",
                        help="Gộp tất cả file nguồn thành all_articles.csv")
    parser.add_argument("--lang", choices=["vi", "en", "all"], default="vi",
                        help="Lọc ngôn ngữ khi gộp (default: vi)")
    parser.add_argument("--check", action="store_true",
                        help="Kiểm tra trùng lặp URL trong từng file")
    parser.add_argument("--clean", action="store_true",
                        help="Kết hợp với --check: dọn trùng lặp nội bộ")
    args = parser.parse_args()

    if args.merge:
        merge_all_sources(lang_filter=args.lang)

    if args.check:
        check_duplicates(clean_files=args.clean)

    if not args.merge and not args.check:
        parser.print_help()


if __name__ == "__main__":
    main()
