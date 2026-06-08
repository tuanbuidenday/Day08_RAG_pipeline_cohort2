"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://cuoi.tuoitre.vn/chu-bin-bi-tam-giu-vi-lien-quan-den-mai-thi-thuy-20240606194643806.htm",
    "https://plo.vn/ca-si-chi-dan-an-tay-va-nhung-nghe-si-danh-mat-su-nghiep-vi-ma-tuy-post819930.html",
    "https://vov.vn/phap-luat/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-co-tien-tu-thien-truc-phuong-post1280703.vov",
    "https://danviet.vn/dien-vien-huu-tin-nguoi-vua-bi-phat-hien-su-dung-ma-tuy-la-ai-202206122002482-print799621.html",
    "https://vnexpress.net/dien-vien-le-hang-bi-dieu-tra-mua-ban-ma-tuy-4597048.html",
    "https://plo.vn/bat-khan-cap-ca-si-chau-viet-cuong-post473865.html",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        title = result.metadata.get("title") if result.metadata else None
        return {
            "url": url,
            "title": title or "Unknown",
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": result.markdown or "",
        }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
