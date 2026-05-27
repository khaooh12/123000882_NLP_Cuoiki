"""
data_loader.py — Bổ sung dataset từ nhiều nguồn:

  1. Kaggle        : crawford/20-newsgroups    (tiếng Anh — test pipeline)
  2. HuggingFace   : kornwtp/VNTC-10Topics     (tiếng Việt — 84k bài, 10 chủ đề)
  3. HuggingFace   : vntc/WikiQA-84k           (tiếng Việt — 84k Q&A từ Wikipedia)
  4. HuggingFace   : vntc/wiki-mini-corpus     (tiếng Việt — 23k bài Wikipedia ngắn)
  5. HuggingFace   : vntc/wiki-dump-cleaned    (tiếng Việt — 1.28M bài Wikipedia)

Kết quả:
  data/raw/newsgroups_articles.csv   — 20 Newsgroups
  data/raw/vntc_articles.csv         — VNTC-10Topics
  data/raw/wiki_qa_articles.csv      — WikiQA-84k
  data/raw/wiki_mini_articles.csv    — wiki-mini-corpus
  data/raw/wiki_dump_articles.csv    — wiki-dump-cleaned (mẫu)
  data/raw/all_articles.csv          — Gộp toàn bộ

Cách dùng:
  python src/data_loader.py                            # vntc + newsgroups (mặc định)
  python src/data_loader.py --source all               # tất cả nguồn
  python src/data_loader.py --source all-vi            # chỉ tiếng Việt (vntc+wiki*)
  python src/data_loader.py --source all-wiki          # 3 nguồn wiki
  python src/data_loader.py --source wiki-qa,wiki-mini # chỉ hai nguồn này
  python src/data_loader.py --source vntc --limit 200  # 200 bài/nhóm
  python src/data_loader.py --no-merge                 # không tạo all_articles.csv
"""

import os
import re
import time
import random
import argparse
import textwrap
import pandas as pd

# ===========================================================
# CATEGORY MAPPINGS
# ===========================================================

# VNTC labels (không dấu) → tên chuyên mục dự án
VNTC_CAT_MAP: dict[str, str] = {
    "The thao":          "Thể thao",
    "Chinh tri Xa hoi":  "Thời sự",
    "Phap luat":         "Pháp luật",
    "Suc khoe":          "Sức khỏe",
    "Doi song":          "Sức khỏe",       # Đời sống → gần Sức khỏe nhất
    "Van hoa":           "Du lịch",        # Văn hóa → gần Du lịch/Văn hóa
    "The gioi":          "Thời sự",        # Thế giới → Thời sự
    "Kinh doanh":        "Kinh doanh",
    "Vi tinh":           "Khoa học-CN",    # Vi tính → Khoa học Công nghệ
    "Khoa hoc":          "Khoa học-CN",
}

# 20 Newsgroups categories → tên chuyên mục dự án
NEWS20_CAT_MAP: dict[str, str] = {
    "rec.sport.baseball":       "Thể thao",
    "rec.sport.hockey":         "Thể thao",
    "rec.autos":                "Thể thao",
    "rec.motorcycles":          "Thể thao",
    "sci.med":                  "Sức khỏe",
    "sci.space":                "Khoa học-CN",
    "sci.electronics":          "Khoa học-CN",
    "sci.crypt":                "Khoa học-CN",
    "comp.graphics":            "Khoa học-CN",
    "comp.os.ms-windows.misc":  "Khoa học-CN",
    "comp.sys.ibm.pc.hardware": "Khoa học-CN",
    "comp.sys.mac.hardware":    "Khoa học-CN",
    "comp.windows.x":           "Khoa học-CN",
    "talk.politics.guns":       "Thời sự",
    "talk.politics.mideast":    "Thời sự",
    "talk.politics.misc":       "Thời sự",
    "talk.religion.misc":       "Thời sự",
    "soc.religion.christian":   "Thời sự",
    "alt.atheism":              "Thời sự",
    "misc.forsale":             "Kinh doanh",
}


# ===========================================================
# KEYWORD MAP — suy ra chuyên mục cho bài Wikipedia
# ===========================================================
# Mỗi key là chuyên mục, value là danh sách từ khoá tiếng Việt (không dấu lẫn có dấu)
WIKI_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Thể thao": [
        "bóng đá", "cầu thủ", "huấn luyện viên", "trận đấu", "thi đấu",
        "câu lạc bộ", "vô địch", "huy chương", "olympic", "thể thao",
        "tennis", "bóng rổ", "bơi lội", "boxing", "cầu lông", "golf",
        "đội tuyển", "v-league", "aff cup", "world cup", "sea games",
        "vận động viên", "giải vô địch", "sân vận động",
    ],
    "Sức khỏe": [
        "bệnh viện", "điều trị", "thuốc", "bác sĩ", "y tế", "dịch bệnh",
        "virus", "vi khuẩn", "vaccine", "phẫu thuật", "ung thư", "tiểu đường",
        "tim mạch", "huyết áp", "dinh dưỡng", "sức khỏe", "bệnh nhân",
        "y học", "sinh học", "dược phẩm", "kháng sinh",
    ],
    "Kinh doanh": [
        "kinh tế", "doanh nghiệp", "công ty", "cổ phiếu", "ngân hàng",
        "đầu tư", "xuất khẩu", "nhập khẩu", "thị trường", "lạm phát",
        "gdp", "tài chính", "chứng khoán", "thương mại", "sản xuất",
        "doanh thu", "lợi nhuận", "vốn", "dự án", "kinh doanh",
    ],
    "Thời sự": [
        "chính phủ", "quốc hội", "thủ tướng", "chủ tịch", "bộ trưởng",
        "chính trị", "bầu cử", "hiến pháp", "luật", "quốc gia", "lịch sử",
        "chiến tranh", "hòa bình", "đảng", "ngoại giao", "liên hợp quốc",
        "cách mạng", "độc lập", "thống nhất", "nhà nước", "tổng thống",
    ],
    "Giáo dục": [
        # --- Cơ sở giáo dục ---
        "trường đại học", "trường học", "trường phổ thông", "trường trung học",
        "trường tiểu học", "trường mầm non", "trường cao đẳng", "học viện",
        "đại học", "cao đẳng", "trung cấp",
        # --- Người học & người dạy ---
        "học sinh", "sinh viên", "học viên", "giáo viên", "giảng viên",
        "giáo sư", "phó giáo sư", "tiến sĩ", "thạc sĩ", "hiệu trưởng",
        "hiệu phó", "thầy giáo", "cô giáo",
        # --- Hoạt động giáo dục ---
        "giáo dục", "giảng dạy", "đào tạo", "học tập", "dạy học",
        "giáo dục đại học", "giáo dục phổ thông", "giáo dục mầm non",
        "chương trình học", "chương trình đào tạo", "giáo trình",
        "sách giáo khoa", "bài giảng", "bài tập", "kiến thức",
        # --- Thi cử & đánh giá ---
        "thi cử", "kỳ thi", "tuyển sinh", "xét tuyển", "nhập học",
        "điểm chuẩn", "điểm số", "học bổng", "ôn thi", "thi tốt nghiệp",
        "thi đại học", "đề thi", "kiểm tra", "đánh giá",
        # --- Bằng cấp & chứng chỉ ---
        "bằng cấp", "chứng chỉ", "cử nhân", "tốt nghiệp", "bằng tốt nghiệp",
        "bằng đại học", "bằng thạc sĩ", "bằng tiến sĩ",
        # --- Chính sách & quản lý GD ---
        "bộ giáo dục", "sở giáo dục", "bộ gd-đt", "học phí", "học kỳ",
        "năm học", "cải cách giáo dục", "chính sách giáo dục",
        # --- Nghiên cứu khoa học ---
        "nghiên cứu", "công trình nghiên cứu", "luận văn", "luận án",
        "hội thảo khoa học", "tạp chí khoa học",
    ],
    "Du lịch": [
        "du lịch", "thành phố", "di sản", "văn hóa", "lễ hội", "ẩm thực",
        "địa danh", "phong tục", "đền", "chùa", "bãi biển", "núi",
        "khách sạn", "resort", "danh thắng", "kiến trúc", "nghệ thuật",
        "phim", "âm nhạc", "hội họa", "điêu khắc",
    ],
    "Pháp luật": [
        "tòa án", "luật sư", "bị cáo", "hình phạt", "điều khoản",
        "vi phạm", "hợp đồng", "quyền lợi", "nghĩa vụ", "tội phạm",
        "cảnh sát", "điều tra", "truy tố", "xét xử", "án phạt",
        "pháp luật", "hiến pháp", "quyền con người",
    ],
    "Khoa học-CN": [
        "máy tính", "phần mềm", "internet", "trí tuệ nhân tạo", "robot",
        "vũ trụ", "vật lý", "hóa học", "toán học", "công nghệ", "kỹ thuật",
        "điện tử", "vi mạch", "lập trình", "dữ liệu", "thuật toán",
        "thiên văn", "phân tử", "nguyên tử", "năng lượng", "khí hậu",
        "sinh học", "di truyền", "tiến hóa",
    ],
}

# Ngưỡng keyword tối thiểu riêng theo category.
# Giáo dục thiếu dữ liệu nghiêm trọng → hạ ngưỡng xuống 1 để giữ nhiều bài hơn.
CATEGORY_MIN_KEYWORDS: dict[str, int] = {
    "Giáo dục": 1,
}
_DEFAULT_MIN_KEYWORDS = 2   # ngưỡng mặc định cho tất cả category còn lại

# Prefix các trang meta Wikipedia cần bỏ qua
WIKI_SKIP_TITLE_PREFIXES = (
    "Wikipedia:", "Thảo luận:", "Thể loại:", "Tập tin:", "Người dùng:",
    "MediaWiki:", "Bản mẫu:", "Trợ giúp:", "Cổng thông tin:",
    "Talk:", "Category:", "File:", "User:", "Template:", "Help:",
)


# ===========================================================
# TIỆN ÍCH
# ===========================================================

def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _clean_wiki_text(text: str, max_chars: int = 500) -> str:
    """
    Làm sạch văn bản Wikipedia: giải mã HTML entities,
    loại bỏ MediaWiki markup và template noise.
    """
    import html as html_lib
    if not isinstance(text, str):
        return ""
    # Giải mã HTML entities: &lt; → <, &amp; → &, ...
    text = html_lib.unescape(text)
    # Loại HTML tags còn sót
    text = re.sub(r"<[^>]+>", " ", text)
    # Loại MediaWiki directives: __NOEDITSECTION__ , __TOC__ ...
    text = re.sub(r"__[A-Z_]+__", " ", text)
    # Loại template blocks {{ ... }}
    text = re.sub(r"\{\{[^}]*\}\}", " ", text)
    # Loại wiki link [[File:...]] hoặc [[Category:...]]
    text = re.sub(r"\[\[(File|Category|Tập tin|Thể loại):[^\]]*\]\]", " ", text, flags=re.IGNORECASE)
    # Chuyển [[link|display]] → display  ;  [[link]] → link
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    # Loại URL thô
    text = re.sub(r"https?://\S+", " ", text)
    # Loại dòng chỉ có ký tự đặc biệt (dòng phân cách, v.v.)
    lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 10]
    text = " ".join(lines)
    # Chuẩn hoá khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _infer_wiki_category(title: str, text: str) -> str:
    """
    Suy ra chuyên mục cho bài Wikipedia bằng đếm từ khoá.
    Trả về 'Bách khoa' nếu không khớp đủ mạnh.

    Ngưỡng tối thiểu lấy từ CATEGORY_MIN_KEYWORDS (mặc định 2).
    Riêng 'Giáo dục' dùng ngưỡng 1 để bù đắp sự thiếu hụt dữ liệu.
    """
    combined = (title + " " + text).lower()
    scores: dict[str, int] = {}
    for cat, keywords in WIKI_CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[cat] = score
    if not scores:
        return "Bách khoa"
    best_cat = max(scores, key=lambda c: scores[c])
    min_kw   = CATEGORY_MIN_KEYWORDS.get(best_cat, _DEFAULT_MIN_KEYWORDS)
    return best_cat if scores[best_cat] >= min_kw else "Bách khoa"


def _split_vntc_text(full_text: str, title_max: int = 100, desc_max: int = 400) -> tuple[str, str]:
    """
    Tách văn bản VNTC (không có tiêu đề riêng) thành title + description.
    - title : câu đầu tiên hoặc title_max ký tự đầu (đến ranh giới từ)
    - description : đoạn tiếp theo tối đa desc_max ký tự
    """
    text = _clean(full_text)
    if not text:
        return "", ""

    # Thử tách theo câu (dấu chấm/chấm hỏi/chấm than + khoảng trắng)
    sentence_end = re.search(r"[.!?]\s+", text[:title_max + 40])
    if sentence_end and sentence_end.end() <= title_max + 40:
        cut = sentence_end.end()
        title = text[:cut].strip(". !?")
        desc  = text[cut:cut + desc_max].strip()
    else:
        # Cắt ở ranh giới từ gần title_max nhất
        cut = title_max
        space = text.rfind(" ", 0, cut + 1)
        cut = space if space > 20 else cut
        title = text[:cut].strip()
        desc  = text[cut:cut + desc_max].strip()

    return _clean(title), _clean(desc)


def _parse_newsgroup_post(raw_text: str) -> tuple[str, str]:
    """
    Tách Subject → title và đoạn body đầu tiên → description từ bài viết newsgroup thô.
    """
    lines = raw_text.splitlines()

    title = ""
    body_lines: list[str] = []
    in_body = False

    for line in lines:
        if not in_body:
            # Headers kết thúc khi gặp dòng trống
            if line.strip() == "":
                in_body = True
                continue
            m = re.match(r"^[Ss]ubject:\s*(.+)$", line)
            if m:
                subject = m.group(1).strip()
                # Bỏ "Re:" prefix
                subject = re.sub(r"^[Rr][Ee]:\s*", "", subject).strip()
                title = subject
        else:
            stripped = line.strip()
            # Bỏ dòng trích dẫn (bắt đầu bằng ">"), dòng rỗng ở đầu, dòng chữ ký
            if stripped.startswith(">"):
                continue
            if re.match(r"^[-=_]{3,}$", stripped):
                break  # Gặp đường kẻ phân cách → dừng
            body_lines.append(stripped)
            if sum(len(l) for l in body_lines) >= 500:
                break

    # Ghép và cắt body thành description (~300 ký tự)
    body = " ".join(l for l in body_lines if l)
    body = re.sub(r"\s+", " ", body).strip()
    description = body[:350].rsplit(" ", 1)[0] if len(body) > 350 else body

    return _clean(title), _clean(description)


# ===========================================================
# LOADER 1: 20 NEWSGROUPS
# ===========================================================

def _load_newsgroups_sklearn(max_per_cat: int) -> pd.DataFrame:
    """Tải 20 Newsgroups qua sklearn (không cần xác thực Kaggle)."""
    from sklearn.datasets import fetch_20newsgroups

    print("  [sklearn] Đang tải 20 Newsgroups (subset='all', giữ headers)...")
    bunch = fetch_20newsgroups(
        subset="all",
        remove=(),          # Giữ headers để parse Subject
        random_state=42,
    )

    rng = random.Random(42)
    records: list[dict] = []
    cat_counts: dict[str, int] = {}
    skipped = 0

    for idx, (text, target_id) in enumerate(zip(bunch.data, bunch.target)):
        cat_raw = bunch.target_names[target_id]
        mapped_cat = NEWS20_CAT_MAP.get(cat_raw, cat_raw)

        if cat_counts.get(cat_raw, 0) >= max_per_cat:
            continue

        title, description = _parse_newsgroup_post(text)

        if len(title) < 5 or len(description) < 20:
            skipped += 1
            continue

        cat_counts[cat_raw] = cat_counts.get(cat_raw, 0) + 1
        day   = rng.randint(1, 28)
        month = rng.randint(1, 12)
        year  = rng.randint(1995, 2000)   # 20 Newsgroups data từ những năm 90s

        records.append({
            "url":          f"news20://{cat_raw}/{idx:06d}",
            "title":        title,
            "description":  description,
            "category":     mapped_cat,
            "publish_date": f"{day:02d}/{month:02d}/{year}",
            "source":       "20newsgroups",
            "language":     "en",
        })

    df = pd.DataFrame(records)
    print(f"  [sklearn] Đã tải {len(df)} bài (bỏ qua {skipped} bài không đủ nội dung).")
    return df


def _load_newsgroups_kaggle(max_per_cat: int) -> pd.DataFrame:
    """
    Tải 20 Newsgroups qua kagglehub (crawler/20-newsgroups).
    Cần có ~/.kaggle/kaggle.json hoặc biến môi trường KAGGLE_USERNAME + KAGGLE_KEY.
    """
    import kagglehub

    print("  [kaggle] Đang tải dataset crawford/20-newsgroups...")
    path = kagglehub.dataset_download("crawford/20-newsgroups")
    print(f"  [kaggle] Đã tải về: {path}")

    rng = random.Random(42)
    records: list[dict] = []
    cat_counts: dict[str, int] = {}

    # Walk qua thư mục, mỗi subfolder là một category
    for root, dirs, files in os.walk(path):
        folder_name = os.path.basename(root)
        # Chỉ lấy các folder là category hợp lệ
        if folder_name not in NEWS20_CAT_MAP:
            continue

        cat_raw = folder_name
        mapped_cat = NEWS20_CAT_MAP[cat_raw]

        file_list = [f for f in files if not f.endswith((".json", ".csv", ".md"))]
        rng.shuffle(file_list)

        for fname in file_list:
            if cat_counts.get(cat_raw, 0) >= max_per_cat:
                break

            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    raw_text = fh.read()
            except OSError:
                continue

            title, description = _parse_newsgroup_post(raw_text)
            if len(title) < 5 or len(description) < 20:
                continue

            cat_counts[cat_raw] = cat_counts.get(cat_raw, 0) + 1
            day   = rng.randint(1, 28)
            month = rng.randint(1, 12)
            year  = rng.randint(1995, 2000)

            records.append({
                "url":          f"news20://{cat_raw}/{fname}",
                "title":        title,
                "description":  description,
                "category":     mapped_cat,
                "publish_date": f"{day:02d}/{month:02d}/{year}",
                "source":       "20newsgroups",
                "language":     "en",
            })

    df = pd.DataFrame(records)
    print(f"  [kaggle] Đã tải {len(df)} bài từ raw files.")
    return df


def load_20newsgroups(max_per_cat: int = 150, prefer_kaggle: bool = True) -> pd.DataFrame:
    """
    Tải 20 Newsgroups.
    Thử kagglehub trước (cần API key), fallback sang sklearn.

    Args:
        max_per_cat   : Số bài tối đa mỗi category (default 150)
        prefer_kaggle : Nếu True, thử kaggle trước rồi mới dùng sklearn

    Returns:
        DataFrame với columns: url, title, description, category,
                                publish_date, source, language
    """
    print("\n[DataLoader] Nguồn 1: 20 Newsgroups (tiếng Anh)")

    if prefer_kaggle:
        try:
            return _load_newsgroups_kaggle(max_per_cat)
        except Exception as exc:
            print(f"  [Cảnh báo] kagglehub thất bại: {exc}")
            print("  -> Chuyển sang sklearn...")

    return _load_newsgroups_sklearn(max_per_cat)


# ===========================================================
# LOADER 2: VNTC-10Topics (HuggingFace)
# ===========================================================

def load_vntc(max_per_cat: int = 150) -> pd.DataFrame:
    """
    Tải VNTC-10Topics từ HuggingFace (kornwtp/VNTC-10Topics).

    Dataset gồm ~84.000 bài báo tiếng Việt phân thành 10 chủ đề.
    Sử dụng cả hai split train + test, lấy tối đa max_per_cat bài/chủ đề.

    Args:
        max_per_cat : Số bài tối đa mỗi label (default 150)

    Returns:
        DataFrame với columns: url, title, description, category,
                                publish_date, source, language
    """
    from datasets import load_dataset

    print("\n[DataLoader] Nguồn 2: VNTC-10Topics (HuggingFace — tiếng Việt)")
    print("  Đang tải dataset kornwtp/VNTC-10Topics ...")

    ds = load_dataset("kornwtp/VNTC-10Topics")
    print(f"  Splits: { {k: len(v) for k, v in ds.items()} }")

    # Gộp tất cả splits
    all_texts:  list[str] = []
    all_labels: list[str] = []
    for split_name, split_data in ds.items():
        all_texts.extend(split_data["text"])
        all_labels.extend(split_data["label"])

    print(f"  Tổng bài gộp: {len(all_texts):,}")

    # Xây dựng mapping: label → danh sách index (shuffle trước)
    from collections import defaultdict
    label_to_idxs: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(all_labels):
        label_to_idxs[lbl].append(i)

    rng = random.Random(42)
    for lbl in label_to_idxs:
        rng.shuffle(label_to_idxs[lbl])

    print(f"  Các label: {list(label_to_idxs.keys())}")
    print(f"  Lấy tối đa {max_per_cat} bài/label\n")

    records: list[dict] = []
    skipped = 0
    art_id  = 500_000  # ID bắt đầu để không trùng với dummy data

    for lbl, idxs in label_to_idxs.items():
        mapped_cat = VNTC_CAT_MAP.get(lbl, lbl)
        taken = 0

        for idx in idxs:
            if taken >= max_per_cat:
                break

            full_text = all_texts[idx]
            title, description = _split_vntc_text(full_text)

            if len(title) < 5 or len(description) < 15:
                skipped += 1
                continue

            day   = rng.randint(1, 28)
            month = rng.randint(1, 12)
            year  = rng.randint(2010, 2020)

            records.append({
                "url":          f"vntc://article/{art_id}",
                "title":        title,
                "description":  description,
                "category":     mapped_cat,
                "publish_date": f"{day:02d}/{month:02d}/{year}",
                "source":       "vntc",
                "language":     "vi",
            })
            art_id += 1
            taken  += 1

    df = pd.DataFrame(records)
    print(f"  [VNTC] Đã tải {len(df)} bài (bỏ qua {skipped} bài quá ngắn).")
    return df


# ===========================================================
# LOADER 3: WikiQA-84k (vntc/WikiQA-84k)
# ===========================================================

def load_wikiqa(max_docs: int = 2000) -> pd.DataFrame:
    """
    Tải vntc/WikiQA-84k — 84.000 cặp (context, question, answer) từ Wikipedia tiếng Việt.

    Chuyển đổi:
      title       = question  (câu hỏi tự nhiên về chủ đề)
      description = context   (đoạn Wikipedia tương ứng — văn bản sạch)
      category    = suy ra từ nội dung bằng từ khoá

    Args:
        max_docs : Tổng số bài lấy (không theo category — shuffle ngẫu nhiên)
    """
    from datasets import load_dataset

    print(f"\n[DataLoader] Nguồn 3: WikiQA-84k  (max={max_docs})")
    print("  Đang tải vntc/WikiQA-84k ...")

    ds = load_dataset("vntc/WikiQA-84k")
    split0 = list(ds.keys())[0]
    data   = ds[split0]
    print(f"  Tổng mẫu: {len(data):,}")

    rng     = random.Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    records: list[dict] = []
    skipped = 0
    art_id  = 1_000_000

    for idx in indices:
        if len(records) >= max_docs:
            break

        sample  = data[idx]
        title   = _clean(sample.get("question", ""))
        context = _clean(sample.get("context", ""))

        if len(title) < 10 or len(context) < 30:
            skipped += 1
            continue

        description = context[:450]
        category    = _infer_wiki_category(title, description)

        day   = rng.randint(1, 28)
        month = rng.randint(1, 12)
        year  = rng.randint(2010, 2023)

        records.append({
            "url":          f"wiki-qa://article/{art_id}",
            "title":        title,
            "description":  description,
            "category":     category,
            "publish_date": f"{day:02d}/{month:02d}/{year}",
            "source":       "wiki-qa",
            "language":     "vi",
        })
        art_id += 1

    df = pd.DataFrame(records)
    print(f"  [WikiQA] Đã tải {len(df):,} bài (bỏ qua {skipped}).")
    if not df.empty:
        print(f"  Phân bố category: { df['category'].value_counts().to_dict() }")
    return df


# ===========================================================
# LOADER 4: wiki-mini-corpus (vntc/wiki-mini-corpus)
# ===========================================================

def load_wiki_mini(max_docs: int = 2000) -> pd.DataFrame:
    """
    Tải vntc/wiki-mini-corpus — 23.039 bài Wikipedia tiếng Việt ngắn, văn bản sạch.

    Cấu trúc mỗi mẫu:
      id       : int
      passage  : "Title: <tên bài>\\n<nội dung>"
      metadata : {'split': int, 'title': str, 'token_count': int}

    Chuyển đổi:
      title       = metadata['title']
      description = nội dung passage (bỏ dòng "Title: ...")
      category    = suy ra từ từ khoá

    Args:
        max_docs : Tổng số bài lấy (shuffle ngẫu nhiên)
    """
    from datasets import load_dataset

    print(f"\n[DataLoader] Nguồn 4: wiki-mini-corpus  (max={max_docs})")
    print("  Đang tải vntc/wiki-mini-corpus ...")

    ds     = load_dataset("vntc/wiki-mini-corpus")
    split0 = list(ds.keys())[0]
    data   = ds[split0]
    print(f"  Tổng mẫu: {len(data):,}")

    rng     = random.Random(42)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    records: list[dict] = []
    skipped = 0
    art_id  = 2_000_000

    for idx in indices:
        if len(records) >= max_docs:
            break

        sample   = data[idx]
        passage  = sample.get("passage", "")
        metadata = sample.get("metadata", {})

        # Lấy title từ metadata (ưu tiên) hoặc dòng đầu của passage
        title = metadata.get("title", "") if isinstance(metadata, dict) else ""
        if not title:
            first_line = passage.split("\n")[0]
            title = re.sub(r"^Title:\s*", "", first_line).strip()

        # Bỏ dòng "Title: ..." khỏi passage
        body_lines = [l for l in passage.splitlines() if not l.strip().startswith("Title:")]
        body = " ".join(l.strip() for l in body_lines if l.strip())
        description = _clean_wiki_text(body, max_chars=450)

        title = _clean(title)
        if len(title) < 3 or len(description) < 30:
            skipped += 1
            continue

        # Bỏ qua trang meta Wikipedia
        if title.startswith(WIKI_SKIP_TITLE_PREFIXES):
            skipped += 1
            continue

        category = _infer_wiki_category(title, description)
        day   = rng.randint(1, 28)
        month = rng.randint(1, 12)
        year  = rng.randint(2010, 2023)

        records.append({
            "url":          f"wiki-mini://article/{art_id}",
            "title":        title,
            "description":  description,
            "category":     category,
            "publish_date": f"{day:02d}/{month:02d}/{year}",
            "source":       "wiki-mini",
            "language":     "vi",
        })
        art_id += 1

    df = pd.DataFrame(records)
    print(f"  [wiki-mini] Đã tải {len(df):,} bài (bỏ qua {skipped}).")
    if not df.empty:
        print(f"  Phân bố category: { df['category'].value_counts().to_dict() }")
    return df


# ===========================================================
# LOADER 5: wiki-dump-cleaned (vntc/wiki-dump-cleaned)
# ===========================================================

def load_wiki_dump(max_docs: int = 2000) -> pd.DataFrame:
    """
    Tải mẫu từ vntc/wiki-dump-cleaned — 1.28M bài Wikipedia tiếng Việt.

    Cấu trúc mỗi mẫu:
      id, revid, url   : metadata
      title            : tên bài Wikipedia (sạch)
      text             : nội dung bài (có HTML entities, cần làm sạch)

    Chuyển đổi:
      title       = title (giữ nguyên)
      description = text đã làm sạch HTML (450 ký tự đầu)
      category    = suy ra từ từ khoá
      url         = url (Wikipedia URL thật)

    Args:
        max_docs : Tổng số bài lấy (streaming — không tải toàn bộ 1.28M)
    """
    from datasets import load_dataset

    print(f"\n[DataLoader] Nguồn 5: wiki-dump-cleaned  (max={max_docs})")
    print("  Đang tải vntc/wiki-dump-cleaned (streaming=True) ...")

    # Dùng streaming để không load toàn bộ 1.28M vào RAM
    ds_stream = load_dataset("vntc/wiki-dump-cleaned", streaming=True, split="train")

    records: list[dict] = []
    skipped = 0
    seen    = 0

    for sample in ds_stream:
        seen += 1
        if len(records) >= max_docs:
            break

        title = _clean(sample.get("title", ""))
        text  = sample.get("text", "")
        url   = sample.get("url", f"wiki-dump://article/{seen}")

        # Bỏ qua trang meta
        if not title or title.startswith(WIKI_SKIP_TITLE_PREFIXES):
            skipped += 1
            continue

        description = _clean_wiki_text(text, max_chars=450)
        if len(description) < 40:
            skipped += 1
            continue

        category = _infer_wiki_category(title, description)

        records.append({
            "url":          url,
            "title":        title,
            "description":  description,
            "category":     category,
            "publish_date": "01/01/2021",   # Wikipedia dump không có ngày cụ thể
            "source":       "wiki-dump",
            "language":     "vi",
        })

        if seen % 5000 == 0:
            print(f"  ... đã duyệt {seen:,} bài, lấy được {len(records):,}")

    df = pd.DataFrame(records)
    print(f"  [wiki-dump] Đã tải {len(df):,} bài (duyệt {seen:,}, bỏ qua {skipped}).")
    if not df.empty:
        print(f"  Phân bố category: { df['category'].value_counts().to_dict() }")
    return df


# ===========================================================
# MERGE & SAVE
# ===========================================================

def _print_stats(df: pd.DataFrame, label: str) -> None:
    print(f"\n  ── {label} ({len(df)} bài) ──")
    for col in ["source", "language", "category"]:
        if col in df.columns:
            val_counts = df[col].value_counts()
            for val, cnt in val_counts.items():
                print(f"    {col}={val!r:<25}: {cnt:>5} ({cnt/len(df)*100:.1f}%)")


def reclassify_bach_khoa(df: pd.DataFrame) -> pd.DataFrame:
    """
    Duyệt lại toàn bộ bài có category='Bách khoa' và thử phân loại lại
    bằng _infer_wiki_category() (đã dùng ngưỡng thấp hơn cho Giáo dục).

    Chỉ áp dụng cho các nguồn wiki (wiki-qa, wiki-mini, wiki-dump) vì
    đó là nơi phát sinh "Bách khoa" — vnexpress/vntc không bị ảnh hưởng.
    """
    wiki_sources = {"wiki-qa", "wiki-mini", "wiki-dump"}
    mask = (df["category"] == "Bách khoa") & (
        df["source"].isin(wiki_sources) if "source" in df.columns
        else True
    )
    indices = df.index[mask].tolist()

    changed_counts: dict[str, int] = {}
    for idx in indices:
        title = str(df.at[idx, "title"])
        desc  = str(df.at[idx, "description"])
        new_cat = _infer_wiki_category(title, desc)
        if new_cat != "Bách khoa":
            df.at[idx, "category"] = new_cat
            changed_counts[new_cat] = changed_counts.get(new_cat, 0) + 1

    total_changed = sum(changed_counts.values())
    print(f"  [Reclassify] Phân loại lại {total_changed}/{len(indices)} bài Bách khoa:")
    for cat, cnt in sorted(changed_counts.items(), key=lambda x: -x[1]):
        print(f"    → {cat:<20}: +{cnt}")
    return df


def balance_dataset(
    df: pd.DataFrame,
    max_per_cat: int = 2500,
    min_per_cat: int = 1800,
    oversample_seed: int = 42,
) -> pd.DataFrame:
    """
    Cân bằng dataset theo category:
    - Undersample  : category > max_per_cat  → lấy ngẫu nhiên max_per_cat bài
    - Oversample   : category < min_per_cat  → lặp lại (with replacement) lên min_per_cat
    - Giữ nguyên   : nằm trong [min_per_cat, max_per_cat]

    Sau khi balance toàn bộ dữ liệu được shuffle lại.
    """
    parts: list[pd.DataFrame] = []
    for cat, group in df.groupby("category", sort=False):
        n = len(group)
        if n > max_per_cat:
            sampled = group.sample(n=max_per_cat, random_state=oversample_seed)
            parts.append(sampled)
            print(f"  [Balance] {cat:<20}: {n:>5} → {max_per_cat:>5}  (undersample -{n - max_per_cat})")
        elif n < min_per_cat:
            needed = min_per_cat - n
            extra  = group.sample(n=needed, replace=True, random_state=oversample_seed)
            parts.append(pd.concat([group, extra], ignore_index=True))
            print(f"  [Balance] {cat:<20}: {n:>5} → {min_per_cat:>5}  (oversample  +{needed})")
        else:
            parts.append(group)
            print(f"  [Balance] {cat:<20}: {n:>5} → {n:>5}  (giữ nguyên)")

    result = pd.concat(parts, ignore_index=True)
    result = result.sample(frac=1, random_state=oversample_seed).reset_index(drop=True)
    return result


def save_and_merge(
    dfs: dict[str, pd.DataFrame],
    vnexpress_path: str = "data/raw/vnexpress_articles.csv",
    output_dir: str = "data/raw",
    create_merged: bool = True,
    do_reclassify: bool = True,
    do_balance: bool = False,
    max_per_cat: int = 2500,
    min_per_cat: int = 1800,
) -> pd.DataFrame:
    """
    Lưu từng nguồn ra file riêng, sau đó gộp tất cả thành all_articles.csv.

    Args:
        dfs            : dict tên → DataFrame (vd: {"vntc": df_vntc, ...})
        vnexpress_path : Đường dẫn file VnExpress hiện có
        output_dir     : Thư mục lưu output
        create_merged  : Có tạo all_articles.csv không
        do_reclassify  : Phân loại lại bài 'Bách khoa' bằng keyword mới (default True)
        do_balance     : Cân bằng dataset (undersample lớn / oversample nhỏ)
        max_per_cat    : Ngưỡng undersample (default 2500)
        min_per_cat    : Ngưỡng oversample  (default 1800)

    Returns:
        DataFrame gộp (hoặc rỗng nếu create_merged=False)
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_dfs: list[pd.DataFrame] = []

    # ── Lưu từng nguồn bổ sung ────────────────────────────────
    for name, df in dfs.items():
        if df is None or df.empty:
            print(f"  [!] Bỏ qua '{name}' (rỗng).")
            continue
        path = os.path.join(output_dir, f"{name}_articles.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  Đã lưu {len(df):>5} bài → {path}")
        saved_dfs.append(df)

    if not create_merged:
        return pd.DataFrame()

    # ── Tải VnExpress nếu có ─────────────────────────────────
    all_parts: list[pd.DataFrame] = []
    if os.path.exists(vnexpress_path):
        vne_df = pd.read_csv(vnexpress_path)
        if "source" not in vne_df.columns:
            vne_df["source"] = "vnexpress"
        if "language" not in vne_df.columns:
            vne_df["language"] = "vi"
        all_parts.append(vne_df)
        print(f"  Đã đọc  {len(vne_df):>5} bài ← {vnexpress_path}")
    else:
        print(f"  [Cảnh báo] Không tìm thấy {vnexpress_path} — bỏ qua VnExpress.")

    all_parts.extend(saved_dfs)

    if not all_parts:
        print("  [Lỗi] Không có nguồn dữ liệu nào để gộp.")
        return pd.DataFrame()

    merged = pd.concat(all_parts, ignore_index=True)

    # Đảm bảo các cột cần thiết tồn tại
    for col in ["url", "title", "description", "category", "publish_date", "source", "language"]:
        if col not in merged.columns:
            merged[col] = ""

    # Loại trùng lặp theo url
    before = len(merged)
    merged = merged.drop_duplicates(subset=["url"], keep="first")
    if before - len(merged) > 0:
        print(f"  Đã loại {before - len(merged)} bài trùng URL.")

    # ── Phân loại lại bài Bách khoa → category phù hợp hơn ──
    if do_reclassify:
        print(f"\n  Đang phân loại lại bài 'Bách khoa' (ngưỡng Giáo dục=1)...")
        merged = reclassify_bach_khoa(merged)

    # ── Cân bằng dataset ─────────────────────────────────────
    if do_balance:
        print(f"\n  Đang cân bằng dataset (max={max_per_cat}, min={min_per_cat})...")
        merged = balance_dataset(merged, max_per_cat=max_per_cat, min_per_cat=min_per_cat)

    merged_path = os.path.join(output_dir, "all_articles.csv")
    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    print(f"\n  Đã lưu  {len(merged):>5} bài (gộp) → {merged_path}")

    return merged


# ===========================================================
# MAIN
# ===========================================================

# Tên nguồn hợp lệ và alias nhóm
_VALID_SOURCES = {"vntc", "newsgroups", "wiki-qa", "wiki-mini", "wiki-dump"}
_SOURCE_GROUPS = {
    "all":      _VALID_SOURCES,
    "all-vi":   {"vntc", "wiki-qa", "wiki-mini", "wiki-dump"},
    "all-wiki": {"wiki-qa", "wiki-mini", "wiki-dump"},
}


def _resolve_sources(source_arg: str) -> set[str]:
    """Phân giải --source thành tập tên nguồn cụ thể."""
    if source_arg in _SOURCE_GROUPS:
        return set(_SOURCE_GROUPS[source_arg])
    # Comma-separated: "vntc,wiki-qa"
    parts = {s.strip() for s in source_arg.split(",")}
    unknown = parts - _VALID_SOURCES
    if unknown:
        print(f"  [Cảnh báo] Nguồn không hợp lệ bị bỏ qua: {unknown}")
    return parts & _VALID_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bổ sung dataset cho NLP Topic Modeling từ nhiều nguồn",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent("""
        Nguồn khả dụng (dùng riêng lẻ hoặc kết hợp bằng dấu phẩy):
          vntc        — kornwtp/VNTC-10Topics   (84k bài báo VI, 10 chủ đề)
          newsgroups  — 20 Newsgroups           (18k bài EN, 20 nhóm)
          wiki-qa     — vntc/WikiQA-84k         (84k Q&A Wikipedia VI)
          wiki-mini   — vntc/wiki-mini-corpus   (23k bài Wikipedia VI ngắn)
          wiki-dump   — vntc/wiki-dump-cleaned  (1.28M bài Wikipedia VI, dùng streaming)

        Alias nhóm:
          all         — tất cả 5 nguồn trên
          all-vi      — chỉ tiếng Việt (vntc + wiki-qa + wiki-mini + wiki-dump)
          all-wiki    — 3 nguồn wiki

        Ví dụ:
          python src/data_loader.py --source all-vi --limit 200
          python src/data_loader.py --source vntc,wiki-qa --limit 300
          python src/data_loader.py --source wiki-dump --limit 3000
          python src/data_loader.py --source all --no-kaggle
        """),
    )
    parser.add_argument(
        "--source", type=str, default="vntc,newsgroups",
        help="Nguồn cần tải (mặc định: vntc,newsgroups). Xem alias ở trên.",
    )
    parser.add_argument(
        "--limit", type=int, default=150,
        help="Bài tối đa mỗi category/label (VNTC) hoặc tổng số bài (wiki-*). Default: 150",
    )
    parser.add_argument(
        "--no-kaggle", action="store_true",
        help="Dùng sklearn thay vì kagglehub cho 20 Newsgroups",
    )
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Không tạo all_articles.csv",
    )
    parser.add_argument(
        "--no-reclassify", action="store_true",
        help="Bỏ qua bước phân loại lại bài 'Bách khoa' (default: luôn chạy)",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Cân bằng dataset: undersample category lớn, oversample 'Giáo dục' và category nhỏ",
    )
    parser.add_argument(
        "--max-per-cat", type=int, default=2500,
        help="Ngưỡng undersample tối đa bài/chuyên mục khi --balance (default: 2500)",
    )
    parser.add_argument(
        "--min-per-cat", type=int, default=1800,
        help="Ngưỡng oversample tối thiểu bài/chuyên mục khi --balance (default: 1800)",
    )
    args = parser.parse_args()

    SEP = "=" * 62
    print(SEP)
    print("  BO SUNG DATASET — NLP Cuoi Khoa")
    print(SEP)

    active_sources = _resolve_sources(args.source)
    if not active_sources:
        print("  [Lỗi] Không có nguồn nào được chọn. Thoát.")
        return
    print(f"  Nguồn sẽ tải: {sorted(active_sources)}")
    print(f"  Giới hạn    : {args.limit} bài/nhóm (hoặc /tổng với wiki-*)\n")

    loaded: dict[str, pd.DataFrame] = {}

    # ── Tải từng nguồn ───────────────────────────────────────
    if "newsgroups" in active_sources:
        loaded["newsgroups"] = load_20newsgroups(
            max_per_cat=args.limit,
            prefer_kaggle=not args.no_kaggle,
        )

    if "vntc" in active_sources:
        loaded["vntc"] = load_vntc(max_per_cat=args.limit)

    if "wiki-qa" in active_sources:
        loaded["wiki_qa"] = load_wikiqa(max_docs=args.limit)

    if "wiki-mini" in active_sources:
        loaded["wiki_mini"] = load_wiki_mini(max_docs=args.limit)

    if "wiki-dump" in active_sources:
        loaded["wiki_dump"] = load_wiki_dump(max_docs=args.limit)

    # ── Thống kê từng nguồn ──────────────────────────────────
    print(f"\n{SEP}")
    print("  KET QUA TAI VE")
    print(SEP)
    for name, df in loaded.items():
        if df is not None and not df.empty:
            _print_stats(df, name.upper())

    # ── Lưu file riêng + gộp ─────────────────────────────────
    print(f"\n{SEP}")
    print("  LUU FILE")
    print(SEP)
    merged = save_and_merge(
        dfs=loaded,
        create_merged=not args.no_merge,
        do_reclassify=not args.no_reclassify,
        do_balance=args.balance,
        max_per_cat=args.max_per_cat,
        min_per_cat=args.min_per_cat,
    )

    # ── Thống kê merged ───────────────────────────────────────
    if not merged.empty:
        print(f"\n{SEP}")
        print("  THONG KE all_articles.csv")
        print(SEP)
        _print_stats(merged, "MERGED")
        vi  = int((merged["language"] == "vi").sum()) if "language" in merged.columns else len(merged)
        en  = len(merged) - vi
        print(f"\n  Tong: {len(merged)} bai  (VI={vi} / EN={en})")
        print(f"\n  Cac buoc tiep theo:")
        print(f"    python src/preprocess.py --lang vi   # chi tieng Viet")
        print(f"    python src/embedding.py")
        print(f"    python src/topic_model.py")

    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
