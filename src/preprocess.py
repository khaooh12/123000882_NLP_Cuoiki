import os
import re
import unicodedata
import pandas as pd
from underthesea import word_tokenize

# List of common Vietnamese stopwords
VIETNAMESE_STOPWORDS = {
    "và", "là", "của", "được", "trong", "có", "những", "cho", "với", "như", "các", "đã", "từ", "ở", "ra", 
    "này", "khi", "lại", "đến", "để", "bằng", "vào", "về", "nhưng", "thì", "sự", "nhà", "nước", "ngày", "năm", 
    "sau", "trên", "trước", "cả", "hay", "cũng", "đều", "họ", "chỉ", "vẫn", "đang", "do", "nơi", "nhiều", 
    "khác", "mới", "cần", "muốn", "biết", "qua", "theo", "nói", "cho_biết", "thêm", "cụ_thể", "ví_dụ", 
    "trường_hợp", "thời_gian", "nội_dung", "phần", "việc", "sẽ", "chưa", "hơn", "nhất", "như_vậy", "tự", 
    "lớn", "nhỏ", "nêu", "gặp", "loại", "bộ", "chủ_yếu", "đơn_vị", "tại", "một_số", "tổng_số", "đối_tượng", 
    "phương_tiện", "hoạt_động", "thực_hiện", "tiến_hành", "liên_quan", "chúng_ta", "chúng_tôi", "ông", "bà",
    "anh", "chị", "em", "nó", "chúng", "tôi", "ta", "mình", "cậu", "tớ", "nào", "gì", "ai", "đâu", "sao", 
    "thế", "nào", "bao_nhiêu", "vậy", "nhé", "nha", "ừ", "vâng", "dạ", "a", "à", "ơi", "thế_nào", "thay_vì",
    "dưới", "ngoài", "giữa", "bên", "trong_đó", "bên_cạnh", "ngoài_ra", "đặc_biệt", "đồng_thời", "trực_tiếp"
}

def clean_html_and_urls(text):
    if not isinstance(text, str):
        return ""
    # Remove URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML Tags
    text = re.sub(r'<.*?>', '', text)
    return text

def normalize_vietnamese_unicode(text):
    # Normalize unicode to NFC form
    return unicodedata.normalize("NFC", text)

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
        
    # 1. Clean HTML, URLs
    text = clean_html_and_urls(text)
    
    # 2. Normalize Unicode
    text = normalize_vietnamese_unicode(text)
    
    # 3. Lowercase
    text = text.lower()
    
    # 4. Remove special characters and numbers (keep letters and spaces)
    # Keeping Vietnamese accents character range
    text = re.sub(r'[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 5. Word Segmentation using underthesea
    try:
        tokenized = word_tokenize(text, format="text")
    except Exception as e:
        # Fallback to simple split if word_tokenize fails
        print(f"[Cảnh báo] Lỗi tách từ underthesea: {e}. Sử dụng tách từ cơ bản.")
        tokenized = text
        
    # 6. Stopwords removal
    tokens = tokenized.split()
    filtered_tokens = [tok for tok in tokens if tok not in VIETNAMESE_STOPWORDS and not tok.isdigit()]
    
    return " ".join(filtered_tokens)

def main():
    import argparse, time

    # ── Tự động chọn file input ưu tiên all_articles.csv nếu có ──
    default_input = (
        "data/raw/all_articles.csv"
        if os.path.exists("data/raw/all_articles.csv")
        else "data/raw/vnexpress_articles.csv"
    )

    parser = argparse.ArgumentParser(description="Tiền xử lý văn bản tiếng Việt cho Topic Modeling")
    parser.add_argument(
        "--input", type=str, default=default_input,
        help=f"File CSV đầu vào (default: {default_input})",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed/cleaned_articles.csv",
        help="File CSV đầu ra (default: data/processed/cleaned_articles.csv)",
    )
    parser.add_argument(
        "--lang", choices=["all", "vi", "en"], default="all",
        help="Lọc theo ngôn ngữ: all / vi (chỉ tiếng Việt) / en (chỉ tiếng Anh). "
             "Default: all (xử lý tất cả)",
    )
    args = parser.parse_args()

    print(f"\n[Preprocessor] Bắt đầu tiền xử lý dữ liệu...")
    print(f"  Input : {args.input}")
    print(f"  Output: {args.output}")
    if args.lang != "all":
        print(f"  Lọc ngôn ngữ: {args.lang}")

    if not os.path.exists(args.input):
        print(f"[Lỗi] Không tìm thấy file: {args.input}")
        print("Hãy chạy 'python src/scraper.py' hoặc 'python src/data_loader.py' trước.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    df = pd.read_csv(args.input, skipinitialspace=True)
    print(f"  Đã đọc {len(df)} bài viết từ {args.input}")

    # ── Lọc ngôn ngữ (nếu có cột language) ──────────────────
    if args.lang != "all" and "language" in df.columns:
        before = len(df)
        df = df[df["language"] == args.lang].copy()
        print(f"  Lọc ngôn ngữ '{args.lang}': còn {len(df)}/{before} bài.")

    # ── Gộp tiêu đề + mô tả ──────────────────────────────────
    df["title"]       = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    combined_texts    = (df["title"] + " " + df["description"]).str.strip()

    print("  Đang chạy tiền xử lý và tách từ (quá trình này có thể mất vài phút)...")
    start_time = time.time()

    cleaned_texts = []
    log_step = max(50, len(combined_texts) // 20)  # Log ~20 lần
    for idx, text in enumerate(combined_texts):
        cleaned = preprocess_text(text)
        cleaned_texts.append(cleaned)
        if (idx + 1) % log_step == 0 or (idx + 1) == len(combined_texts):
            pct = (idx + 1) / len(combined_texts) * 100
            print(f"    -> {idx + 1}/{len(combined_texts)} bài ({pct:.0f}%) ...")

    df["cleaned_text"] = cleaned_texts

    # ── Loại bài quá ngắn sau làm sạch ───────────────────────
    initial_len = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() >= 10].copy()
    removed = initial_len - len(df)
    if removed:
        print(f"  Đã lọc bỏ {removed} bài quá ngắn sau tiền xử lý.")

    df.to_csv(args.output, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_time
    print(f"\n[Preprocessor Hoàn tất] Đã lưu {len(df)} dòng → {args.output}  ({elapsed:.1f} giây)")

    # Thống kê nhanh
    if "source" in df.columns:
        print("  Phân bố theo nguồn:")
        for src, cnt in df["source"].value_counts().items():
            print(f"    {src:<20}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")
    if "category" in df.columns:
        print("  Phân bố theo chuyên mục:")
        for cat, cnt in df["category"].value_counts().items():
            print(f"    {cat:<20}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()
