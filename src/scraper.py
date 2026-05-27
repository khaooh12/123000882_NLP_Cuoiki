"""
scraper.py — Thu thập bài viết từ VnExpress phục vụ bài toán phân cụm chủ đề NLP.

Cách dùng nhanh:
    python src/scraper.py --dummy                  # 700 bài giả lập (100/chuyên mục), không cần mạng
    python src/scraper.py --limit 100              # Cào thật từ VnExpress, 100 bài/chuyên mục
    python src/scraper.py --limit 50 --append      # Cào thêm, ghép vào file hiện có
    python src/scraper.py --dummy --dummy-count 150  # 150 bài giả lập/chuyên mục
    python src/scraper.py --categories "Thể thao,Giáo dục" --limit 80
"""

import os
import re
import time
import random
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ===========================================================
# CẤU HÌNH
# ===========================================================
CATEGORIES = {
    "Thời sự":   "https://vnexpress.net/thoi-su",
    "Kinh doanh":"https://vnexpress.net/kinh-doanh",
    "Thể thao":  "https://vnexpress.net/the-thao",
    "Giáo dục":  "https://vnexpress.net/giao-duc",
    "Sức khỏe":  "https://vnexpress.net/suc-khoe",
    "Pháp luật": "https://vnexpress.net/phap-luat",
    "Du lịch":   "https://vnexpress.net/du-lich",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

RATE_LIMIT_SEC   = 1.2   # giây giữa mỗi request trang danh sách
RETRY_LIMIT      = 3     # số lần thử lại nếu request thất bại
RETRY_BACKOFF    = 2.0   # hệ số backoff (giây × RETRY_BACKOFF^n)
DESC_MIN_LEN     = 20    # bỏ qua description quá ngắn


# ===========================================================
# TIỆN ÍCH
# ===========================================================
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def request_with_retry(session: requests.Session, url: str, retries: int = RETRY_LIMIT) -> requests.Response | None:
    """GET với tự động thử lại theo exponential backoff."""
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=12)
            if resp.status_code == 200:
                return resp
            print(f"    [HTTP {resp.status_code}] {url} — thử lại ({attempt+1}/{retries})")
        except requests.RequestException as exc:
            print(f"    [Lỗi mạng] {exc} — thử lại ({attempt+1}/{retries})")
        time.sleep(RETRY_BACKOFF ** attempt)
    return None


def fetch_article_description(session: requests.Session, article_url: str) -> str:
    """Vào trang bài viết lấy đoạn mô tả/lead nếu listing không có description."""
    resp = request_with_retry(session, article_url, retries=2)
    if not resp:
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    # VnExpress article description selectors (theo thứ tự ưu tiên)
    selectors = [
        "p.description",
        "p.lead",
        'meta[name="description"]',
        'meta[property="og:description"]',
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get("content") or el.get_text()
            text = clean_text(text)
            if len(text) >= DESC_MIN_LEN:
                return text
    return ""


# ===========================================================
# SCRAPER THẬT
# ===========================================================
def scrape_category(
    session: requests.Session,
    category_name: str,
    base_url: str,
    max_articles: int = 100,
    fetch_missing_desc: bool = True,
) -> list[dict]:
    """Cào bài viết từ một chuyên mục VnExpress theo trang (pagination)."""
    articles: list[dict] = []
    seen_urls: set[str] = set()
    page = 1
    max_empty_pages = 3   # dừng nếu liên tiếp không lấy được bài mới
    empty_streak = 0

    print(f"\n[Scraper] Chuyên mục: {category_name}  (mục tiêu: {max_articles} bài)")

    while len(articles) < max_articles:
        url = base_url if page == 1 else f"{base_url}-p{page}"
        print(f"  Trang {page}: {url}")

        resp = request_with_retry(session, url)
        if not resp:
            print(f"  [!] Bỏ qua trang {page} do lỗi kết nối.")
            break

        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Tìm các item bài viết ──────────────────────────────
        items = soup.find_all("article", class_=lambda c: c and "item-news" in c)
        if not items:
            # Fallback: tìm theo selector phổ biến khác của VnExpress
            items = soup.select("article.item-news-common, .item-news-common")
        if not items:
            items = soup.select(".title-news")

        if not items:
            empty_streak += 1
            print(f"  [Thông báo] Không tìm thấy bài viết (streak={empty_streak}/{max_empty_pages}).")
            if empty_streak >= max_empty_pages:
                break
            page += 1
            time.sleep(RATE_LIMIT_SEC)
            continue

        new_count = 0
        for item in items:
            if len(articles) >= max_articles:
                break

            # ── Tiêu đề & link ────────────────────────────────
            title_el = (
                item.find("h3", class_="title-news")
                or item.find("h2", class_="title-news")
                or item.find(class_="title-news")
                or item.select_one("a[href*='vnexpress.net']")
            )
            if not title_el:
                continue

            link_el = title_el if title_el.name == "a" else title_el.find("a")
            if not link_el:
                continue

            href  = link_el.get("href", "").strip()
            title = clean_text(link_el.get_text())

            if not title or not href:
                continue
            if not href.startswith("https://vnexpress.net"):
                continue
            # Lọc link quảng cáo / video riêng
            if any(kw in href for kw in ["/video/", "/infographic/", "/photo/"]):
                continue
            if href in seen_urls:
                continue

            # ── Mô tả ────────────────────────────────────────
            desc_el = (
                item.find("p", class_="description")
                or item.find("span", class_="description")
                or item.select_one(".description")
            )
            description = clean_text(desc_el.get_text()) if desc_el else ""

            # Nếu không có mô tả, thử vào trang bài viết lấy lead
            if len(description) < DESC_MIN_LEN and fetch_missing_desc:
                time.sleep(0.4)
                description = fetch_article_description(session, href)

            # ── Ngày đăng ────────────────────────────────────
            date_el = (
                item.find("span", class_="time-public")
                or item.find("span", class_="time")
                or item.find("span", class_="date")
            )
            publish_date = date_el.get_text().strip() if date_el else time.strftime("%d/%m/%Y")

            seen_urls.add(href)
            articles.append({
                "url":          href,
                "title":        title,
                "description":  description,
                "category":     category_name,
                "publish_date": publish_date,
            })
            new_count += 1

        empty_streak = 0 if new_count > 0 else empty_streak + 1
        print(f"  -> +{new_count} bài  |  Tổng: {len(articles)}/{max_articles}")

        if empty_streak >= max_empty_pages:
            print("  [Thông báo] Nhiều trang liên tiếp không có bài mới. Dừng chuyên mục này.")
            break

        page += 1
        time.sleep(RATE_LIMIT_SEC)

    return articles


# ===========================================================
# DUMMY DATA (100 bài/chuyên mục = 700 bài tổng)
# ===========================================================
# Mỗi chuyên mục có 20 cặp (tiêu đề, mô tả) riêng biệt, đảm bảo
# không lặp nội dung và mang đặc trưng chủ đề rõ ràng.
DUMMY_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Thời sự": [
        ("Quốc hội thảo luận sửa đổi Luật Đất đai, nhiều đại biểu đề xuất tăng thời hạn sử dụng đất",
         "Các đại biểu Quốc hội tranh luận sôi nổi về phương án bồi thường, hỗ trợ tái định cư và cơ chế định giá đất sát thị trường."),
        ("Thủ tướng chủ trì hội nghị thúc đẩy giải ngân vốn đầu tư công năm 2025",
         "Nhiều bộ, ngành và địa phương chưa đạt tiến độ giải ngân đề ra, Thủ tướng yêu cầu xử lý nghiêm lãnh đạo để chậm trễ."),
        ("Bộ Giao thông đôn đốc hoàn thành cao tốc Bắc Nam giai đoạn 2 trước Tết Nguyên Đán",
         "Tổng chiều dài tuyến đường gần 2.000 km đang trong giai đoạn thi công nước rút, một số đoạn chưa được thông xe kỹ thuật."),
        ("TP.HCM triển khai đề án chống ngập 10.000 tỷ đồng tại lưu vực Tân Hóa – Lò Gốm",
         "Dự án gồm hầm chứa nước mưa ngầm và cải tạo hệ thống cống thoát nước chạy dọc theo các trục đường trung tâm thành phố."),
        ("Chính phủ ban hành nghị quyết đẩy mạnh phân cấp, phân quyền cho địa phương",
         "Theo nghị quyết mới, các tỉnh thành được tự phê duyệt một số dự án đầu tư dưới 5.000 tỷ đồng không cần xin ý kiến trung ương."),
        ("Hà Nội công bố quy hoạch chung thủ đô đến năm 2045 với nhiều đô thị vệ tinh mới",
         "Bốn thành phố vệ tinh sẽ được hình thành ở Hòa Lạc, Sơn Tây, Xuân Mai và Phú Xuyên nhằm giãn dân nội đô."),
        ("Triển khai cấp căn cước công dân gắn chip cho 100% người dân trước cuối năm 2025",
         "Bộ Công an phát động phong trào đổi CCCD theo diện rộng, miễn phí cho người dân lần đổi đầu tiên."),
        ("Bộ Nội vụ đề xuất tăng lương cơ sở lên 2,34 triệu đồng từ tháng 7/2025",
         "Mức lương cơ sở mới tăng 15% so với hiện hành, áp dụng cho toàn bộ cán bộ, công chức, viên chức hưởng lương từ ngân sách nhà nước."),
        ("Việt Nam chính thức gia nhập Hội đồng Nhân quyền Liên Hợp Quốc nhiệm kỳ 2026-2028",
         "Đây là lần thứ ba Việt Nam đảm nhiệm vị trí thành viên, khẳng định vai trò và uy tín ngày càng tăng trên trường quốc tế."),
        ("Khánh Hòa khẩn cấp di dời hàng nghìn hộ dân trước mùa mưa bão năm nay",
         "Ủy ban nhân dân tỉnh chỉ đạo hoàn thành tái định cư trước ngày 15/8, ưu tiên các hộ sống gần sườn dốc và ven biển."),
        ("Đề xuất sáp nhập ba tỉnh miền Bắc để tinh gọn bộ máy hành chính nhà nước",
         "Phương án sáp nhập được đánh giá sẽ tiết kiệm hàng nghìn tỷ đồng ngân sách và giảm đầu mối quản lý trùng lặp."),
        ("Chính phủ phê duyệt quy hoạch điện VIII với cơ cấu năng lượng tái tạo chiếm 47%",
         "Đến năm 2030, điện mặt trời và điện gió dự kiến đóng góp gần một nửa tổng công suất phát điện toàn quốc."),
        ("Bộ Công an triệt phá đường dây tổ chức xuất cảnh trái phép sang Campuchia và Lào",
         "Hơn 30 đối tượng bị bắt giữ, trong đó có nhiều người là nạn nhân bị lừa làm việc trong các khu casino online."),
        ("Đà Nẵng xây dựng trung tâm tài chính quốc tế nhằm thu hút vốn FDI từ Singapore",
         "Thành phố kỳ vọng mô hình trung tâm tài chính mới sẽ thúc đẩy dòng vốn đầu tư gián tiếp và các quỹ ETF quốc tế."),
        ("Mekong Delta chịu ảnh hưởng xâm nhập mặn nghiêm trọng nhất trong một thập kỷ",
         "Các tỉnh ven biển đồng bằng sông Cửu Long ghi nhận độ mặn vượt ngưỡng 4 phần nghìn, ảnh hưởng sản xuất lúa vụ đông xuân."),
        ("Bộ Y tế công bố kế hoạch tiêm vaccine cúm mùa miễn phí cho trẻ em dưới 5 tuổi",
         "Chương trình triển khai tại 63 tỉnh thành, ước tính phủ vaccine cho hơn 4 triệu trẻ trong năm đầu thực hiện."),
        ("Thực hiện cải cách thủ tục hành chính, giảm 30% thời gian giải quyết hồ sơ",
         "Cổng dịch vụ công quốc gia ghi nhận gần 100 triệu hồ sơ được xử lý trực tuyến, tiết kiệm chi phí xã hội đáng kể."),
        ("Việt Nam xuất siêu 8 tỷ USD trong 5 tháng đầu năm 2025 nhờ điện tử và dệt may",
         "Kim ngạch xuất khẩu đạt 163 tỷ USD, tăng 14,5% so với cùng kỳ, đặc biệt thị trường Mỹ và EU đều tăng trưởng mạnh."),
        ("Lực lượng cảnh sát biển bắt giữ tàu cá vận chuyển 500 kg ma túy đá trên biển Đông",
         "Vụ bắt giữ là kết quả của quá trình theo dõi bí mật kéo dài hai tháng, phối hợp với cảnh sát quốc tế khu vực."),
        ("Thành lập Ủy ban quốc gia về chuyển đổi số, do Thủ tướng làm Chủ tịch",
         "Ủy ban sẽ điều phối thống nhất các hoạt động chuyển đổi số quốc gia, xây dựng nền kinh tế số và xã hội số bền vững."),
    ],
    "Kinh doanh": [
        ("VN-Index đóng cửa vượt mốc 1.300 điểm lần đầu tiên trong vòng 3 năm",
         "Nhóm cổ phiếu vốn hóa lớn như ngân hàng, thép và bất động sản là động lực chính dẫn dắt thị trường tăng điểm mạnh."),
        ("Tập đoàn Samsung đầu tư thêm 2,5 tỷ USD mở rộng nhà máy tại Thái Nguyên",
         "Đây là lần mở rộng đầu tư thứ tư của Samsung tại Việt Nam, nâng tổng số vốn tích lũy lên gần 20 tỷ USD."),
        ("Ngân hàng Nhà nước hạ lãi suất điều hành lần thứ tư trong năm nhằm kích thích tăng trưởng",
         "Lãi suất tái chiết khấu giảm còn 3,5%/năm, tạo điều kiện cho các ngân hàng thương mại giảm lãi vay cho doanh nghiệp."),
        ("Giá vàng miếng SJC trong nước chạm mốc 95 triệu đồng/lượng sau khi vàng thế giới vọt lên 2.450 USD/oz",
         "Nhu cầu trú ẩn tài sản an toàn tăng cao do lo ngại căng thẳng địa chính trị tại Trung Đông và suy thoái ở Mỹ."),
        ("Vingroup khởi công khu đô thị thông minh rộng 2.900 ha tại Hải Phòng",
         "Dự án trị giá 30 tỷ USD dự kiến hoàn thành theo ba giai đoạn đến năm 2040, tích hợp công nghệ AI và IoT toàn diện."),
        ("Xuất khẩu gạo Việt Nam đạt kỷ lục 9 triệu tấn, thu về 5,8 tỷ USD trong năm 2025",
         "Thị trường Indonesia, Philippines và châu Phi tiếp tục là những nhà nhập khẩu gạo lớn nhất của Việt Nam."),
        ("Thị trường bất động sản TP.HCM khởi sắc, giao dịch căn hộ tăng 40% so với quý trước",
         "Phân khúc căn hộ tầm trung và cao cấp tại các quận ngoại thành đang thu hút mạnh nhà đầu tư nhờ giá còn hợp lý."),
        ("Masan Group công bố lợi nhuận sau thuế đạt 4.200 tỷ đồng, vượt kế hoạch năm",
         "Chuỗi WinMart/WinMart+ và mảng tiêu dùng là động lực chính, trong khi mảng khoáng sản đóng góp ổn định."),
        ("Việt Nam lọt top 20 quốc gia thu hút FDI lớn nhất thế giới theo báo cáo UNCTAD 2025",
         "Đầu tư nước ngoài vào Việt Nam đạt 38 tỷ USD, tăng 22% so với năm ngoái, chủ yếu vào lĩnh vực điện tử và logistics."),
        ("Techcombank và VPBank hợp tác phát hành trái phiếu xanh để tài trợ dự án năng lượng tái tạo",
         "Đây là lần đầu tiên hai ngân hàng tư nhân lớn cùng lúc phát hành green bond nhằm đáp ứng nhu cầu ESG của nhà đầu tư."),
        ("Giá xăng RON95 tăng lần thứ ba liên tiếp, vượt mức 23.000 đồng/lít",
         "Giá dầu thô thế giới neo cao trên 90 USD/thùng là nguyên nhân chính, trong khi Quỹ bình ổn xăng dầu đã cạn nguồn."),
        ("Thương mại điện tử Việt Nam cán mốc 25 tỷ USD, Shopee và TikTok Shop chiếm 70% thị phần",
         "Người tiêu dùng trẻ dưới 35 tuổi chiếm 65% lượng giao dịch, xu hướng mua sắm livestream tăng trưởng 300% so với năm ngoái."),
        ("Hàng không Việt Nam phục hồi hoàn toàn, Vietnam Airlines báo lãi kỷ lục 3.800 tỷ đồng",
         "Lượng khách quốc tế đến Việt Nam đạt 18 triệu lượt trong 10 tháng đầu năm, vượt mức trước dịch COVID-19."),
        ("Tập đoàn VNPT hoàn thành phủ sóng 5G tại 20 tỉnh thành, sẵn sàng thương mại hóa toàn quốc",
         "Băng thông 5G tại Việt Nam đạt trung bình 450 Mbps, dự kiến áp dụng cho các khu công nghiệp và đô thị thông minh."),
        ("Doanh nghiệp SME được hỗ trợ vay vốn lãi suất ưu đãi 4%/năm từ gói 40.000 tỷ đồng",
         "Chính phủ mở rộng điều kiện tiếp cận vốn vay cho doanh nghiệp nhỏ và vừa, đặc biệt là trong lĩnh vực nông nghiệp và chế biến thực phẩm."),
        ("IPO VinFast tại sàn Nasdaq kết thúc tuần đầu với vốn hóa đạt 85 tỷ USD",
         "Cổ phiếu VFS tăng vọt trong phiên giao dịch đầu, đưa VinFast trở thành hãng xe điện có vốn hóa lớn thứ ba thế giới."),
        ("Lạm phát tháng 11 tăng 3,8%, tiệm cận ngưỡng mục tiêu kiểm soát 4% của Chính phủ",
         "Nhóm hàng lương thực, thực phẩm và dịch vụ giáo dục là hai nguyên nhân chính đẩy chỉ số CPI tăng trong tháng."),
        ("Sàn giao dịch bất động sản công nghệ PropertyGuru chính thức gia nhập thị trường Việt Nam",
         "Nền tảng này kỳ vọng cung cấp dữ liệu minh bạch về giá nhà đất, giúp người mua và nhà đầu tư đưa ra quyết định chính xác hơn."),
        ("Cà phê Robusta Việt Nam đạt giá kỷ lục 4.200 USD/tấn trên sàn ICE London",
         "Nguồn cung toàn cầu thắt chặt do thời tiết bất lợi tại Brazil và Indonesia, nâng cao vị thế xuất khẩu của Việt Nam."),
        ("Bộ Tài chính đề xuất giảm thuế VAT xuống còn 8% cho nhóm hàng tiêu dùng thiết yếu",
         "Chính sách gia hạn giảm thuế nhằm kích cầu tiêu dùng trong nước, dự kiến làm giảm thu ngân sách khoảng 24.000 tỷ đồng."),
    ],
    "Thể thao": [
        ("Đội tuyển Việt Nam đánh bại Thái Lan 2-1 trong trận chung kết AFF Cup lịch sử",
         "Bàn thắng quyết định ở phút 89 của Nguyễn Công Phượng mang về chức vô địch Đông Nam Á thứ ba cho bóng đá Việt Nam."),
        ("Nguyễn Thùy Linh vô địch giải cầu lông Thailand Masters, xếp hạng 8 thế giới",
         "Tay vợt người Hà Nội thắng 3 set liên tiếp trước tay vợt chủ nhà hàng đầu, hoàn thành tuần thi đấu hoàn hảo."),
        ("U23 Việt Nam lần đầu lọt vào bán kết ASIAD sau trận thắng Nhật Bản 1-0",
         "Bàn thắng duy nhất do Khuất Văn Khang thực hiện từ chấm phạt đền trong hiệp phụ tạo nên kỳ tích lịch sử."),
        ("VĐV Đinh Phương Thành giành huy chương vàng thể dục dụng cụ tại SEA Games 33",
         "Bài biểu diễn xà kép hoàn hảo với điểm số 14,9 giúp vận động viên Hải Phòng bảo vệ thành công chức vô địch."),
        ("V-League 2025: Hà Nội FC vô địch sớm 3 vòng, lập kỷ lục 14 mùa giải liên tiếp dẫn đầu bảng",
         "Đội bóng thủ đô kết thúc mùa giải với 81 điểm, phá vỡ kỷ lục điểm số trước đó 5 điểm của chính mình."),
        ("Cung thủ Ánh Nguyệt phá kỷ lục châu Á nội dung bắn cung 70 mét đôi nữ",
         "Thành tích 1.362 điểm tại giải vô địch bắn cung châu Á tại Tokyo giúp xạ thủ người Hà Tĩnh lọt vào top 5 thế giới."),
        ("Đua thuyền máy F1 Powerboat lần đầu tổ chức tại Hà Nội trên Hồ Tây, thu hút 50.000 khán giả",
         "Sự kiện quốc tế hoành tráng với 20 đội đến từ 12 quốc gia mang lại nguồn thu du lịch ước tính hơn 200 tỷ đồng cho thành phố."),
        ("Bơi lội: Nguyễn Huy Hoàng phá kỷ lục quốc gia 1.500m tự do với thời gian 14:56",
         "Đây là thành tích đủ chuẩn A tham dự Olympic Paris 2024, mở ra hy vọng có huy chương tại đại hội thể thao lớn nhất hành tinh."),
        ("Đội tuyển bóng chuyền nữ Việt Nam hạ Thái Lan 3-0 giành vé dự giải vô địch châu Á",
         "Bộ ba chủ công Trần Thị Thanh Thúy, Bích Tuyền và Kim Thanh thi đấu ấn tượng với 45 điểm tấn công thành công."),
        ("CLB Viettel FC ký hợp đồng với tiền đạo người Brazil Leandro Cruz trị giá 3 triệu USD",
         "Đây là bản hợp đồng ngoại binh đắt giá nhất lịch sử bóng đá Việt Nam, kỳ vọng mang lại thay đổi trong phong cách tấn công."),
        ("Lần đầu tiên Việt Nam đăng cai giải quần vợt ATP 250 với sự tham dự của top 20 thế giới",
         "Giải đấu sẽ diễn ra tại Hà Nội trong tháng 11 với tổng giải thưởng 600.000 USD, nằm trong lộ trình phát triển quần vợt Việt Nam."),
        ("Kỳ thủ Lê Quang Liêm vào tứ kết World Chess Championship, viết lịch sử cờ vua Việt Nam",
         "Màn trình diễn xuất sắc trước kiện tướng người Hà Lan đã nâng ELO của Quang Liêm lên 2.745, cao nhất trong sự nghiệp."),
        ("Đội xe đạp Việt Nam chinh phục đèo Ô Quy Hồ, phá kỷ lục chặng Tour de Vietnam",
         "Tay đua Nguyễn Tấn Hoài về nhất chặng núi khắc nghiệt nhất với thời gian 3 giờ 42 phút, nhanh hơn kỷ lục cũ 8 phút."),
        ("SEA Games 33: Đoàn Việt Nam giành 200 huy chương, xếp thứ 2 toàn đoàn",
         "Thành tích ấn tượng nhờ lực lượng vận động viên trẻ tài năng, đặc biệt ở các môn điền kinh, võ thuật và bơi lội."),
        ("Giải golf PGA Tour Vietnam 2025 sẽ tổ chức tại BRG Kings Island, tổng giải thưởng 3 triệu USD",
         "Sự kiện kỳ vọng thu hút hàng chục golfer hàng đầu thế giới và hàng nghìn khán giả mỗi ngày trong 4 vòng thi đấu."),
        ("Đấu sĩ Nguyễn Trần Duy Nhất bảo vệ thành công đai vô địch Muay Thái thế giới WBC",
         "Trận đấu kịch tính kéo dài 5 hiệp trước đối thủ người Thái, Duy Nhất giành chiến thắng bằng quyết định nhất trí của ban giám khảo."),
        ("CLB Hải Phòng lần đầu vào chung kết AFC Cup đấu với đại diện Hàn Quốc",
         "Màn trình diễn của thủ thành Văn Lâm trong các trận đấu loại trực tiếp là điểm sáng của đội bóng miền Bắc."),
        ("Trường đua F1 Hà Nội được đề xuất xây dựng tại khu vực phía Tây, tổng vốn đầu tư 5.000 tỷ",
         "Thành phố đang xem xét mô hình đường đua tháo lắp được tổ chức trên phố như Monaco, tránh chiếm dụng đất lâu dài."),
        ("Thiện Tâm Boxing Club sản sinh 5 nhà vô địch boxing quốc gia trong một mùa giải",
         "Phong trào boxing nghiệp dư phát triển mạnh mẽ tại TP.HCM với hơn 200 võ sinh đang tập luyện chuyên nghiệp hàng ngày."),
        ("Cờ vua: 14 kỳ thủ Việt Nam đạt chuẩn kiện tướng quốc tế FIDE trong năm học 2024-2025",
         "Thành tích cho thấy chương trình đào tạo trẻ do Liên đoàn Cờ vua Việt Nam đổi mới đang đi đúng hướng và phát huy hiệu quả."),
    ],
    "Giáo dục": [
        ("Bộ GD-ĐT công bố phương án thi tốt nghiệp THPT 2025 với 4 môn bắt buộc và 2 môn lựa chọn",
         "Môn Lịch sử không còn là bắt buộc; học sinh chọn các tổ hợp phù hợp với định hướng nghề nghiệp đại học theo hướng mới."),
        ("Điểm chuẩn xét học bạ đại học năm 2025 tăng mạnh ở các ngành hot như AI, Y khoa và Luật",
         "Ngành Trí tuệ nhân tạo tại Đại học Bách Khoa Hà Nội yêu cầu điểm trung bình học bạ tối thiểu 9,2/10 trong 5 kỳ."),
        ("Học sinh Việt Nam giành 4 huy chương vàng Olympic Toán quốc tế, xếp thứ 3 thế giới",
         "Đoàn Việt Nam gồm 6 em, trong đó 4 em đạt huy chương vàng và 2 em đạt huy chương bạc tại kỳ thi IMO năm nay."),
        ("Chương trình sách giáo khoa mới lớp 12 chính thức áp dụng từ năm học 2025-2026",
         "Bộ GD-ĐT đã phê duyệt 4 bộ sách giáo khoa, trong đó bộ sách Kết nối tri thức được 60% trường lựa chọn sử dụng."),
        ("Đại học Quốc gia TP.HCM lần đầu lọt top 500 đại học tốt nhất thế giới theo QS Ranking",
         "ĐHQG TP.HCM xếp hạng 481, tăng 120 bậc so với năm ngoái, đặc biệt nổi bật ở chỉ số nghiên cứu khoa học và quan hệ doanh nghiệp."),
        ("Mô hình trường chất lượng cao tại Hà Nội thu hút 40.000 hồ sơ đăng ký vào lớp 10",
         "Tỷ lệ chọi ở một số trường đạt 1/8, học sinh phải thi kiểm tra năng lực kết hợp với điểm học bạ THCS."),
        ("Bộ Giáo dục đề xuất tăng lương giáo viên lên mức bình quân 12 triệu đồng/tháng từ 2026",
         "Chính sách nhằm thu hút người giỏi vào ngành sư phạm, giải quyết tình trạng thiếu giáo viên trầm trọng ở vùng sâu vùng xa."),
        ("Chương trình học bổng toàn phần Chính phủ Nhật Bản mở thêm 500 suất cho sinh viên Việt Nam",
         "Ngoài bậc thạc sĩ và tiến sĩ, năm nay học bổng MEXT mở thêm diện nghiên cứu sinh và trao đổi sinh viên đại học."),
        ("Ứng dụng AI chấm bài luận tiếng Anh được thí điểm tại 200 trường THPT phía Nam",
         "Phần mềm sử dụng mô hình ngôn ngữ lớn có khả năng cho phản hồi chi tiết về ngữ pháp, từ vựng và cấu trúc bài viết."),
        ("Tỷ lệ trẻ em hoàn thành cấp tiểu học đạt 99,2%, Việt Nam dẫn đầu ASEAN về phổ cập giáo dục",
         "UNICEF đánh giá cao chính sách miễn học phí và hỗ trợ sách giáo khoa của Việt Nam trong việc duy trì tỷ lệ đến trường cao."),
        ("Triển khai đề án phát triển 100.000 kỹ sư AI và khoa học dữ liệu đến năm 2030",
         "Chương trình đào tạo kết hợp giữa đại học và doanh nghiệp công nghệ, cung cấp thực tập có lương cho sinh viên từ năm thứ hai."),
        ("Kỳ thi đánh giá năng lực ĐHQG Hà Nội 2025 có hơn 100.000 thí sinh đăng ký, tăng 30%",
         "Ban tổ chức mở thêm hai đợt thi và bổ sung điểm thi tại 8 tỉnh thành mới để đáp ứng nhu cầu đăng ký cao kỷ lục."),
        ("Hội đồng giáo sư nhà nước công nhận kỷ lục 700 giáo sư, phó giáo sư trong một năm",
         "Số lượng ứng viên được xét công nhận tăng mạnh sau khi tiêu chuẩn quốc tế về số lượng bài báo khoa học được áp dụng."),
        ("Bộ GD-ĐT thực hiện đổi mới kiểm tra đánh giá, giảm học vẹt tăng tư duy phản biện",
         "Đề thi theo định hướng PISA sẽ áp dụng từ năm học 2025-2026, tập trung vào kỹ năng vận dụng kiến thức vào thực tiễn."),
        ("Trường THPT chuyên Khoa học Tự nhiên ĐHQG Hà Nội kỷ niệm 30 năm đào tạo nhân tài",
         "Trường đã sản sinh hơn 500 học sinh đạt giải Olympic quốc tế, trong đó nhiều người đang là nhà khoa học hàng đầu thế giới."),
        ("Sinh viên Đại học FPT giành quán quân cuộc thi lập trình quốc tế ACM-ICPC khu vực châu Á",
         "Đội gồm ba sinh viên năm ba đã giải quyết 10/12 bài toán trong 5 tiếng đồng hồ, vượt qua hàng trăm đội từ 30 quốc gia."),
        ("Chương trình học trực tuyến trên hệ thống LMS quốc gia phục vụ 15 triệu học sinh",
         "Nền tảng kết hợp bài giảng video, bài tập tương tác và AI hỗ trợ cá nhân hóa lộ trình học tập cho từng học sinh."),
        ("Các trường đại học tăng cường liên kết với doanh nghiệp FDI tạo cơ hội việc làm cho sinh viên",
         "Mô hình thực tập ngay từ năm thứ nhất giúp sinh viên tích lũy kinh nghiệm thực tế và 70% được nhận vào làm sau tốt nghiệp."),
        ("Giáo dục mầm non chất lượng cao mở rộng ra các huyện ngoại thành, giảm bất bình đẳng",
         "Nhà nước tài trợ xây dựng 500 lớp học mầm non tiêu chuẩn quốc tế tại 20 tỉnh có tỷ lệ đến trường thấp nhất cả nước."),
        ("Đề xuất cải cách học phí đại học theo nguyên tắc người học trả theo thu nhập sau tốt nghiệp",
         "Mô hình ISA (Income Share Agreement) được thí điểm tại 10 trường đại học công lập trọng điểm từ năm học 2026."),
    ],
    "Sức khỏe": [
        ("Bộ Y tế cảnh báo dịch sốt xuất huyết bùng phát mạnh, hơn 80.000 ca mắc trong 6 tháng đầu năm",
         "Miền Nam ghi nhận số ca mắc tăng gấp đôi so với cùng kỳ, Bộ Y tế kêu gọi người dân diệt lăng quăng và dùng màn ngủ."),
        ("Phát triển vaccine ung thư phổi thế hệ mới, thử nghiệm lâm sàng giai đoạn 3 tại Việt Nam",
         "Nghiên cứu do Viện Vệ sinh Dịch tễ phối hợp với Đại học Oxford thực hiện, kỳ vọng giảm tỷ lệ tái phát xuống còn 30%."),
        ("Người Việt mắc ung thư đại trực tràng ngày càng trẻ hóa, nguyên nhân do chế độ ăn nhiều thịt đỏ",
         "Các bác sĩ khuyến cáo tầm soát ung thư đại tràng bắt đầu từ tuổi 40, sớm hơn 10 năm so với khuyến cáo trước đây."),
        ("Kỹ thuật phẫu thuật robot thế hệ 4 được ứng dụng thành công trong ca ghép thận tại Hà Nội",
         "Hệ thống robot Da Vinci Xi giúp bác sĩ thực hiện các thao tác tinh tế mà mổ hở truyền thống không thể đạt được."),
        ("Thuốc điều trị tiểu đường mới dạng viên uống tuần một lần được cấp phép lưu hành tại Việt Nam",
         "Thuốc Semaglutide dạng uống giúp bệnh nhân không còn phải tiêm mỗi ngày, cải thiện đáng kể chất lượng cuộc sống."),
        ("Bệnh viện Chợ Rẫy thực hiện ghép tạng từ người cho sống thứ 1.000, cột mốc lịch sử y học",
         "Ca phẫu thuật kéo dài 14 tiếng với sự tham gia của 30 bác sĩ và điều dưỡng, bệnh nhân hồi phục tốt sau 3 ngày."),
        ("Nghiên cứu mới: Ngủ dưới 6 tiếng mỗi đêm làm tăng 37% nguy cơ mắc bệnh tim mạch",
         "Nghiên cứu trên 50.000 người Việt Nam trưởng thành trong 10 năm xác nhận mối liên hệ chặt chẽ giữa giấc ngủ và sức khỏe tim mạch."),
        ("Hội chứng long COVID ảnh hưởng đến 15% bệnh nhân khỏi bệnh, khó chẩn đoán và điều trị",
         "Triệu chứng phổ biến gồm mệt mỏi kéo dài, sương não và khó thở dai dẳng dù xét nghiệm COVID-19 âm tính."),
        ("TP.HCM mở trung tâm y tế chuyên điều trị sức khỏe tâm thần với 500 giường bệnh",
         "Trung tâm cung cấp dịch vụ nội trú và ngoại trú, giải quyết bức xúc về tình trạng thiếu giường bệnh tâm thần hiện nay."),
        ("Phương pháp điều trị ung thư bằng tế bào T-CAR cho tỷ lệ thuyên giảm 90% ở bệnh nhân bạch cầu",
         "Kỹ thuật biến đổi tế bào miễn dịch của chính người bệnh để nhận diện và tiêu diệt tế bào ung thư được áp dụng lần đầu tại Việt Nam."),
        ("Chương trình sàng lọc ung thư cổ tử cung miễn phí cho 5 triệu phụ nữ từ 21-65 tuổi",
         "Xét nghiệm HPV DNA sẽ được triển khai tại 63 tỉnh thành từ 2025, thay thế phương pháp Pap smear kém nhạy hơn trước đây."),
        ("Dinh dưỡng học đường: Thực đơn mới bổ sung vi chất dinh dưỡng giảm suy dinh dưỡng trẻ em",
         "Bữa ăn học đường tại 10.000 trường tiểu học được cải thiện với rau xanh, cá và sữa tươi theo tiêu chuẩn WHO."),
        ("Bệnh nhân tiểu đường type 2 có thể lui bệnh hoàn toàn nhờ can thiệp lối sống mạnh mẽ trong 3 tháng",
         "Nghiên cứu của Bệnh viện Bạch Mai cho thấy chế độ ăn calo thấp kết hợp tập thể dục 150 phút/tuần giúp 50% bệnh nhân ngừng dùng thuốc."),
        ("Cảnh báo nguy hiểm của thực phẩm chức năng giả mạo được bán tràn lan trên mạng xã hội",
         "Cục Quản lý Thực phẩm và Dược phẩm thu hồi 200 sản phẩm không rõ nguồn gốc, nhiều sản phẩm chứa chất cấm như steroid."),
        ("Hội nghị y tế quốc tế tại Hà Nội kết nối 2.000 bác sĩ từ 40 quốc gia về y học tái tạo",
         "Các chủ đề nổi bật gồm liệu pháp tế bào gốc, in 3D mô tạng và phẫu thuật tự động hóa sử dụng AI hỗ trợ chẩn đoán."),
        ("Không khí ô nhiễm tại Hà Nội vượt ngưỡng AQI 200 trong nhiều ngày liên tiếp mùa đông",
         "Người cao tuổi, trẻ nhỏ và bệnh nhân tim phổi được khuyến cáo ở trong nhà, đeo khẩu trang N95 khi ra ngoài bắt buộc."),
        ("Ứng dụng AI chẩn đoán ung thư da bằng hình ảnh điện thoại đạt độ chính xác 94%",
         "Phần mềm do nhóm nghiên cứu Đại học Y Hà Nội phát triển, hiện đang xin cấp phép thử nghiệm lâm sàng rộng rãi toàn quốc."),
        ("Chương trình tiêm chủng mở rộng bổ sung vaccine phế cầu cho trẻ dưới 5 tuổi",
         "Ước tính vaccine PCV13 sẽ phòng ngừa được 30% các ca viêm phổi và viêm màng não cầu khuẩn ở trẻ trong nhóm tuổi này."),
        ("Bệnh đái tháo đường ảnh hưởng đến 7 triệu người Việt, 50% chưa được chẩn đoán",
         "Chương trình tầm soát cộng đồng tại các siêu thị và trung tâm thương mại giúp phát hiện hàng nghìn ca bệnh tiềm ẩn mỗi năm."),
        ("Tác động của màn hình điện thoại đến thị lực trẻ em: tỷ lệ cận thị tăng lên 60% ở trẻ cấp 2",
         "Chuyên gia khuyến cáo giới hạn thời gian màn hình dưới 2 tiếng/ngày và bổ sung thời gian hoạt động ngoài trời cho trẻ."),
    ],
    "Pháp luật": [
        ("Tòa án nhân dân tối cao tuyên phạt 30 năm tù bị cáo chủ mưu vụ lừa đảo 500 tỷ đồng qua sàn tiền ảo",
         "Bản án nghiêm khắc nhằm răn đe các hành vi lừa đảo đầu tư tài chính qua mạng, được dư luận đặc biệt quan tâm."),
        ("Công an TP.HCM phá đường dây cho vay nặng lãi qua app, thu hồi 200 tỷ đồng",
         "Hàng nghìn nạn nhân bị khủng bố qua điện thoại và mạng xã hội, một số người bị tống tiền bằng ảnh cắt ghép."),
        ("Quốc hội thông qua Luật Tội phạm mạng sửa đổi với mức phạt tù lên đến 20 năm",
         "Luật mới bổ sung tội danh phát tán mã độc ransomware và tấn công hạ tầng quan trọng quốc gia bằng phương thức mạng."),
        ("Bộ Công an cảnh báo 5 thủ đoạn lừa đảo trực tuyến mới xuất hiện trong quý 3 năm nay",
         "Các chiêu thức gồm giả danh cơ quan thuế, lừa đảo tuyển dụng việc làm online và giả mạo ứng dụng ngân hàng nhằm đánh cắp thông tin."),
        ("Vụ án Vạn Thịnh Phát: Tòa phúc thẩm giữ nguyên bản án chung thân với bị cáo Trương Mỹ Lan",
         "Hội đồng xét xử nhấn mạnh hành vi gây thiệt hại đặc biệt nghiêm trọng cho nền kinh tế, không có tình tiết giảm nhẹ đáng kể."),
        ("Triệt phá ổ nhóm buôn bán người xuyên quốc gia qua đường biên giới Tây Nam",
         "Các nạn nhân chủ yếu là phụ nữ và trẻ em bị lừa sang Campuchia để làm việc trong các khu casino và call center lừa đảo."),
        ("Ngân hàng mất 200 tỷ đồng do deepfake: khẳng định cần nâng cấp xác thực sinh trắc học",
         "Kẻ gian sử dụng video deepfake giả mạo chủ tài khoản để vượt qua hệ thống nhận diện khuôn mặt và thực hiện chuyển tiền."),
        ("Luật sư bị đề nghị thu hồi chứng chỉ hành nghề do tổ chức mạng lưới dịch vụ pháp lý phi pháp",
         "Cơ quan điều tra xác định người này đã nhận thù lao bất hợp pháp bằng cách hứa hẹn can thiệp vào quá trình tố tụng."),
        ("Xét xử vụ gian lận đấu thầu y tế Việt Á: 8 bị cáo bị truy tố về tội lợi dụng chức vụ quyền hạn",
         "Tổng thiệt hại ước tính hơn 100 tỷ đồng ngân sách nhà nước qua việc nâng khống giá kit xét nghiệm COVID-19."),
        ("Cảnh sát kinh tế xử lý 1.200 vụ vi phạm sở hữu trí tuệ hàng hóa giả mạo thương hiệu",
         "Các mặt hàng bị thu giữ gồm túi xách, đồng hồ, quần áo nhái thương hiệu nổi tiếng trị giá hàng trăm tỷ đồng."),
        ("Bộ Công an bắt nghi phạm tấn công mạng làm tê liệt hệ thống giao dịch của hai sàn chứng khoán",
         "Vụ tấn công xảy ra vào ngày giao dịch với khối lượng kỷ lục, gây thiệt hại hàng nghìn tỷ đồng cho nhà đầu tư."),
        ("Thanh tra Chính phủ kết luận sai phạm trong quản lý đất đai tại 5 tỉnh, kiến nghị thu hồi đất",
         "Tổng diện tích đất sai phạm lên đến hàng nghìn ha, nhiều trường hợp cho thuê đất rừng phòng hộ không đúng quy định."),
        ("Phiên xét xử trực tuyến đầu tiên tại Việt Nam: Bị cáo khai nhận qua video call từ trại giam",
         "Thử nghiệm thành công mở ra khả năng phổ biến tòa án trực tuyến, giảm chi phí áp giải và tăng tính minh bạch xét xử."),
        ("Cảnh sát phát hiện xưởng sản xuất bằng giả, giấy tờ xe giả quy mô lớn tại Bình Dương",
         "Thiết bị in ấn công nghệ cao được nhập lậu từ Trung Quốc, sản phẩm giả mạo có chất lượng khó phân biệt bằng mắt thường."),
        ("Luật Phòng chống rửa tiền sửa đổi áp dụng báo cáo giao dịch tiền mặt trên 100 triệu đồng",
         "Ngưỡng báo cáo mới thấp hơn 50% so với trước, kỳ vọng siết chặt kiểm soát dòng tiền bất minh trong bất động sản và ngoại hối."),
        ("Vụ án buôn bán nội tạng: 5 đối tượng bị bắt, nạn nhân là người nghèo bị dụ dỗ bằng tiền mặt",
         "Đường dây hoạt động tinh vi, tổ chức phẫu thuật lấy thận tại các cơ sở y tế tư nhân không phép ở nước ngoài."),
        ("Bộ Tư pháp đẩy mạnh dịch vụ công trực tuyến, 80% thủ tục hành chính tư pháp lên cấp độ 4",
         "Người dân có thể nộp hồ sơ khai sinh, khai tử, đăng ký kết hôn và chứng thực bản sao hoàn toàn qua mạng Internet."),
        ("Tòa án thương mại quốc tế Việt Nam xử lý vụ tranh chấp 500 triệu USD với tập đoàn nước ngoài",
         "Đây là vụ kiện có giá trị lớn nhất từ trước đến nay tại Việt Nam, liên quan đến hợp đồng xây dựng nhà máy điện mặt trời."),
        ("Khởi tố vụ án trốn thuế qua sàn thương mại điện tử, thiệt hại ngân sách hàng trăm tỷ đồng",
         "Cơ quan thuế xác định hàng nghìn cá nhân kinh doanh online thu nhập tiền tỷ nhưng không kê khai nộp thuế thu nhập."),
        ("VKSND tối cao ban hành hướng dẫn truy tố tội phạm AI deepfake và gian lận danh tính số",
         "Văn bản hướng dẫn lấp đầy khoảng trống pháp lý trong việc xử lý các tội danh mới phát sinh từ công nghệ trí tuệ nhân tạo."),
    ],
    "Du lịch": [
        ("Vịnh Hạ Long đón 4 triệu lượt khách trong 9 tháng đầu năm, doanh thu du lịch tăng 45%",
         "Sản phẩm du lịch đêm trên vịnh và trải nghiệm văn hóa ngư dân làng chài là điểm nhấn thu hút đặc biệt du khách quốc tế."),
        ("Sapa mùa đông: Tuyết rơi trắng xóa trên đỉnh Fansipan tạo khung cảnh như trời Âu",
         "Nhiệt độ xuống dưới 0 độ C tại Sa Pa và Fansipan, hàng nghìn du khách đổ về chụp ảnh tuyết và trượt tuyết tự phát."),
        ("Phú Quốc trở thành điểm đến hàng đầu Đông Nam Á theo đánh giá của Condé Nast Traveler 2025",
         "Hòn đảo ngọc với 150 km đường bờ biển đạt danh hiệu nhờ sự kết hợp hài hòa giữa thiên nhiên, ẩm thực và cơ sở lưu trú sang trọng."),
        ("Hội An ghi nhận kỷ lục 5 triệu lượt khách, phố cổ phải giới hạn lượng khách vào đêm rằm",
         "Thành phố di sản UNESCO đang cân nhắc mô hình đặt vé tham quan trực tuyến bắt buộc vào các dịp lễ hội đông đúc nhất."),
        ("Tuyến du lịch xuyên Việt bằng tàu hỏa Bắc Nam thu hút 200.000 khách nước ngoài trong năm đầu ra mắt",
         "Hành trình 30 tiếng từ Hà Nội đến TP.HCM ghé thăm Huế, Đà Nẵng và Nha Trang đang là trend du lịch chậm được ưa chuộng nhất 2025."),
        ("Làng nghề gốm Bát Tràng mở rộng không gian trải nghiệm 4.0, đón 1,5 triệu khách mỗi năm",
         "Du khách có thể tự tay nặn gốm, vẽ hoa văn và mang sản phẩm của mình về nhà, trải nghiệm kết hợp truyền thống và công nghệ số."),
        ("Côn Đảo lọt top 10 đảo đẹp nhất thế giới theo National Geographic, đặt phòng tăng 300%",
         "Bãi Đầm Trầu được mệnh danh là một trong những bãi biển hoang sơ đẹp nhất hành tinh, thu hút cả những tín đồ du lịch khắt khe nhất."),
        ("Ẩm thực đường phố Hà Nội được Michelin Street Food đưa vào danh sách thành phố đáng thưởng thức nhất châu Á",
         "Phở, bún chả, bánh mì và cà phê trứng là bốn món được giám khảo Michelin nhấn mạnh khi giới thiệu ẩm thực thủ đô."),
        ("Du lịch sinh thái Cần Giờ phát triển mạnh, thu hút nhà đầu tư xây resort nổi trên rừng ngập mặn",
         "Mô hình eco-lodge với vật liệu xây dựng bền vững và năng lượng mặt trời đang là xu hướng đầu tư hút vốn tại khu sinh quyển này."),
        ("Festival Hoa Đà Lạt 2025 thu hút 800.000 du khách, doanh thu du lịch tăng 60% trong dịp lễ",
         "Triển lãm hoa cúc nhật nguyên khổng lồ và biểu diễn ánh sáng nghệ thuật là hai điểm nhấn chưa từng có trong lịch sử festival."),
        ("Ninh Bình xây dựng tuyến cáp treo kết nối hang Múa với Tràng An dài 4 km",
         "Công trình trị giá 1.200 tỷ đồng dự kiến đưa vào vận hành năm 2026, tạo thêm sản phẩm du lịch độc đáo cho vùng đất cố đô."),
        ("Du thuyền 5 sao đầu tiên buông neo tại cảng du lịch Chân Mây, mang 3.000 khách quốc tế đến Huế",
         "Siêu du thuyền Symphony of the Seas thuộc hàng lớn nhất thế giới đã chọn Việt Nam là điểm dừng chân trong hành trình Đông Nam Á 2025."),
        ("Mùa thu hoạch cà phê Đắk Lắk trở thành tour du lịch nông nghiệp hút hàng nghìn khách mỗi tháng",
         "Du khách được tham gia hái cà phê thủ công, rang xay thủ công và thưởng thức cà phê nguyên chất ngay trên đồi, trải nghiệm hoàn toàn mới."),
        ("Phố đi bộ Hồ Gươm sau 10 năm hoạt động: điểm đến cuối tuần không thể thiếu của người Hà Nội",
         "Hàng triệu lượt người đến tham quan mỗi cuối tuần, không gian văn hóa sống động góp phần thúc đẩy kinh tế đêm trung tâm thành phố."),
        ("Bắt đầu mùa du lịch biển 2025: Nha Trang, Đà Nẵng và Mũi Né chuẩn bị đón khách sau kỳ nghỉ lễ",
         "Công suất buồng phòng đạt 90%, các resort ven biển lần đầu áp dụng check-in điện tử và dịch vụ concierge AI."),
        ("Trải nghiệm 'farmstay' ở đồng bằng sông Cửu Long ngày càng được giới trẻ yêu thích",
         "Mô hình nghỉ dưỡng trong vườn cây ăn trái, ngủ nhà sàn và đi thuyền len lỏi kênh rạch mang lại doanh thu cao cho hộ nông dân."),
        ("Việt Nam đón 18 triệu lượt khách quốc tế trong 10 tháng đầu năm, vượt mục tiêu cả năm",
         "Hàn Quốc, Trung Quốc và Mỹ là ba thị trường khách đứng đầu, tổng doanh thu du lịch đạt 840.000 tỷ đồng."),
        ("Tổ chức tour di sản kết hợp ẩm thực: từ Mỹ Sơn đến phố cổ Hội An trong một ngày",
         "Sản phẩm mới được thiết kế dành riêng cho khách quốc tế tham quan ban ngày và thưởng thức ẩm thực Quảng Nam vào buổi tối."),
        ("Đảo Lý Sơn được quy hoạch thành khu kinh tế biển, hạn chế du khách tự phát để bảo tồn sinh thái",
         "Mô hình kiểm soát lượng khách như Galápagos đang được các chuyên gia đề xuất áp dụng để cân bằng phát triển và bảo tồn."),
        ("Homestay trên ruộng bậc thang Mù Cang Chải vào mùa vàng kín phòng trước 3 tháng mỗi năm",
         "Khung cảnh ruộng lúa vàng dưới nắng chiều ở Tú Lệ và La Pán Tẩn được ví như bức tranh thủy mặc khổng lồ của thiên nhiên."),
    ],
}


def _slug(cat: str) -> str:
    mapping = {
        "Thời sự": "thoi-su", "Kinh doanh": "kinh-doanh", "Thể thao": "the-thao",
        "Giáo dục": "giao-duc", "Sức khỏe": "suc-khoe", "Pháp luật": "phap-luat",
        "Du lịch": "du-lich",
    }
    return mapping.get(cat, cat.lower().replace(" ", "-"))


def generate_dummy_data(articles_per_category: int = 100) -> list[dict]:
    """
    Sinh dữ liệu giả lập đa dạng, không lặp nội dung.
    Mỗi chuyên mục có 20 template × biến thể → articles_per_category bài.
    """
    print(f"\n[Scraper] Tạo dữ liệu giả lập — {articles_per_category} bài/chuyên mục "
          f"({articles_per_category * len(DUMMY_TEMPLATES)} bài tổng)...")

    rng = random.Random(42)
    dummy_articles: list[dict] = []

    # Các mẫu biến thể tiêu đề để làm phong phú hơn mà vẫn giữ ý nghĩa gốc
    title_variants = [
        lambda t: t,
        lambda t: f"Cập nhật mới nhất: {t}",
        lambda t: f"{t} — Những điều cần biết",
        lambda t: f"Chuyên gia nhận định: {t.lower()}",
        lambda t: f"Báo cáo: {t}",
        lambda t: f"Góc nhìn: {t.lower()}",
        lambda t: f"Toàn cảnh {t.lower()}",
    ]

    art_id = 200_000
    for cat, templates in DUMMY_TEMPLATES.items():
        cat_slug = _slug(cat)
        generated = 0
        cycle = 0

        while generated < articles_per_category:
            for base_title, base_desc in templates:
                if generated >= articles_per_category:
                    break
                # Vòng đầu: title gốc; vòng sau: xoay qua các biến thể
                variant_fn = title_variants[cycle % len(title_variants)]
                title = variant_fn(base_title)

                day   = rng.randint(1, 28)
                month = rng.randint(1, 12)
                year  = rng.choice([2024, 2025, 2026])

                dummy_articles.append({
                    "url":          f"https://vnexpress.net/{cat_slug}/art-{art_id}.html",
                    "title":        title,
                    "description":  base_desc,
                    "category":     cat,
                    "publish_date": f"{day:02d}/{month:02d}/{year}",
                })
                art_id  += 1
                generated += 1
            cycle += 1

    rng.shuffle(dummy_articles)
    print(f"[Scraper] Tạo xong {len(dummy_articles)} bài giả lập.")
    return dummy_articles


# ===========================================================
# MAIN
# ===========================================================
def main():
    parser = argparse.ArgumentParser(
        description="VnExpress Web Scraper — Thu thập bài viết phục vụ NLP Topic Modeling",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ví dụ:
  python src/scraper.py --dummy                        # 700 bài giả lập (không cần mạng)
  python src/scraper.py --dummy --dummy-count 150      # 1050 bài giả lập
  python src/scraper.py --limit 100                    # Cào thật 100 bài/chuyên mục
  python src/scraper.py --limit 50 --append            # Cào thêm 50 bài, ghép vào file cũ
  python src/scraper.py --categories "Thể thao,Du lịch" --limit 80
        """,
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Số bài tối đa mỗi chuyên mục khi cào thật (mặc định: 100)",
    )
    parser.add_argument(
        "--dummy", action="store_true",
        help="Dùng dữ liệu giả lập thay vì cào mạng",
    )
    parser.add_argument(
        "--dummy-count", type=int, default=100,
        help="Số bài giả lập mỗi chuyên mục khi dùng --dummy (mặc định: 100)",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Ghép bài mới vào file CSV hiện có thay vì ghi đè",
    )
    parser.add_argument(
        "--no-fetch-desc", action="store_true",
        help="Không vào từng trang bài viết để lấy description khi listing thiếu (nhanh hơn)",
    )
    parser.add_argument(
        "--categories", type=str, default="",
        help='Danh sách chuyên mục cách nhau bởi dấu phẩy, VD: "Thể thao,Giáo dục"',
    )
    args = parser.parse_args()

    os.makedirs("data/raw", exist_ok=True)
    output_path = "data/raw/vnexpress_articles.csv"

    # ── Chọn chuyên mục cần cào ──────────────────────────────
    target_categories = dict(CATEGORIES)
    if args.categories:
        requested = [c.strip() for c in args.categories.split(",")]
        target_categories = {k: v for k, v in CATEGORIES.items() if k in requested}
        missing = [c for c in requested if c not in CATEGORIES]
        if missing:
            print(f"[Cảnh báo] Không tìm thấy chuyên mục: {missing}")
        if not target_categories:
            print("[Lỗi] Không có chuyên mục hợp lệ. Thoát.")
            return

    # ── Thu thập bài viết ────────────────────────────────────
    if args.dummy:
        articles = generate_dummy_data(articles_per_category=args.dummy_count)
        # Nếu chỉ dùng một số chuyên mục thì lọc lại
        if args.categories:
            articles = [a for a in articles if a["category"] in target_categories]
    else:
        articles = []
        session = requests.Session()
        for cat_name, url in target_categories.items():
            try:
                cat_articles = scrape_category(
                    session, cat_name, url,
                    max_articles=args.limit,
                    fetch_missing_desc=not args.no_fetch_desc,
                )
                articles.extend(cat_articles)
            except Exception as exc:
                print(f"[Lỗi] Không thể cào '{cat_name}': {exc}")

        if not articles:
            print("\n[Cảnh báo] Không cào được bài viết nào. Tự động chuyển sang dữ liệu giả lập...")
            articles = generate_dummy_data(articles_per_category=args.dummy_count)

    new_df = pd.DataFrame(articles)

    # ── Ghép hoặc ghi đè ─────────────────────────────────────
    if args.append and os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Loại bỏ trùng URL sau khi ghép
        before = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["url"], keep="first")
        removed = before - len(combined_df)
        if removed:
            print(f"[Scraper] Loại bỏ {removed} bài trùng URL sau khi ghép.")
        final_df = combined_df
    else:
        final_df = new_df

    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ── Thống kê kết quả ─────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"[Scraper Hoàn tất] Đã lưu {len(final_df)} bài vào: {output_path}")
    print(f"{'='*55}")
    print(f"  Phân bố theo chuyên mục:")
    for cat, cnt in final_df["category"].value_counts().items():
        pct = cnt / len(final_df) * 100
        print(f"    {cat:<15} : {cnt:>4} bài  ({pct:.1f}%)")
    missing_desc = final_df["description"].isna().sum() + (final_df["description"] == "").sum()
    print(f"  Bài thiếu mô tả : {missing_desc} ({missing_desc/len(final_df)*100:.1f}%)")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
