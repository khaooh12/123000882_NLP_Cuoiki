import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup

VIETNAMNET_CATEGORIES = {
    "Thời sự": "https://vietnamnet.vn/thoi-su",
    "Kinh doanh": "https://vietnamnet.vn/kinh-doanh",
    "Thể thao": "https://vietnamnet.vn/the-thao",
    "Giáo dục": "https://vietnamnet.vn/giao-duc",
    "Sức khỏe": "https://vietnamnet.vn/suc-khoe",
    "Pháp luật": "https://vietnamnet.vn/phap-luat",
    "Du lịch": "https://vietnamnet.vn/du-lich",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

def clean_text(text):
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def scrape_vietnamnet_category(session, cat_name, url, existing_urls, existing_titles, max_articles=20):
    print(f"\n[Scraper Test] Đang cào chuyên mục '{cat_name}' từ VietnamNet...")
    articles = []
    
    try:
        resp = session.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            print(f"  [Lỗi HTTP {resp.status_code}] Không thể tải trang {url}")
            return articles
            
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # VietnamNet uses 'verticalPost' and 'horizontalPost' classes for articles
        posts = soup.find_all(class_=lambda c: c and any(x in c for x in ["verticalPost", "horizontalPost"]))
        print(f"  Tìm thấy {len(posts)} phần tử bài viết trên trang.")
        
        for post in posts:
            if len(articles) >= max_articles:
                break
                
            title_a = post.find("a", href=True)
            if not title_a:
                continue
                
            title = clean_text(title_a.get_text())
            if not title:
                title = clean_text(title_a.get("title", ""))
                
            href = title_a["href"].strip()
            if not href or not href.endswith(".html"):
                continue
                
            # Convert relative path to absolute
            abs_url = href if href.startswith("http") else f"https://vietnamnet.vn{href}"
            
            # Check duplication
            if abs_url in existing_urls or title.lower() in existing_titles:
                continue
                
            # Get description
            desc_el = post.find(class_=lambda c: c and any(x in c for x in ["desc", "sapo", "summary", "description"]))
            description = clean_text(desc_el.get_text()) if desc_el else ""
            
            if len(title) < 10 or len(description) < 15:
                continue
                
            articles.append({
                "url": abs_url,
                "title": title,
                "description": description,
                "category": cat_name,
                "publish_date": time.strftime("%d/%m/%Y"),
                "source": "vietnamnet",
                "language": "vi"
            })
            
        print(f"  -> Đã thu thập thành công {len(articles)} bài viết hợp lệ không trùng lặp.")
        
    except Exception as e:
        print(f"  [Lỗi] Gặp ngoại lệ khi cào '{cat_name}': {e}")
        
    return articles

def main():
    print("="*60)
    print("  THU THẬP TẬP KIỂM THỬ (TEST SET) TỪ VIETNAMNET")
    print("="*60)
    
    train_path = "data/raw/all_articles.csv"
    output_path = "data/raw/test_articles.csv"
    
    # 1. Load existing URLs and titles for duplicate checking
    existing_urls = set()
    existing_titles = set()
    
    if os.path.exists(train_path):
        print(f"Đọc tệp huấn luyện: {train_path} để kiểm tra trùng lặp...")
        train_df = pd.read_csv(train_path)
        if "url" in train_df.columns:
            existing_urls = set(train_df["url"].dropna().tolist())
        if "title" in train_df.columns:
            existing_titles = set(train_df["title"].dropna().str.lower().str.strip().tolist())
        print(f"  Đã tải {len(existing_urls):,} URLs và {len(existing_titles):,} Tiêu đề vào bộ nhớ.")
    else:
        print(f"[Cảnh báo] Không tìm thấy file {train_path}. Sẽ cào mà không kiểm tra trùng lặp.")

    session = requests.Session()
    all_test_articles = []
    
    # 2. Scrape each category
    for cat_name, url in VIETNAMNET_CATEGORIES.items():
        cat_articles = scrape_vietnamnet_category(
            session=session,
            cat_name=cat_name,
            url=url,
            existing_urls=existing_urls,
            existing_titles=existing_titles,
            max_articles=20
        )
        all_test_articles.extend(cat_articles)
        time.sleep(1.0) # rate limit
        
    if not all_test_articles:
        print("[Lỗi] Không cào được bài viết nào từ VietnamNet. Kiểm tra kết nối mạng hoặc cấu trúc trang.")
        return
        
    # 3. Save to file
    test_df = pd.DataFrame(all_test_articles)
    test_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("\n" + "="*60)
    print(f"[Hoàn tất] Đã lưu {len(test_df)} bài viết kiểm thử từ VietnamNet vào: {output_path}")
    print("="*60)

if __name__ == "__main__":
    main()
