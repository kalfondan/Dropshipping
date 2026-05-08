import cloudscraper
from bs4 import BeautifulSoup
import re
import json

class IsraeliSiteScraper:
    def __init__(self):
        # יצירת סקרייפר שמדמה דפדפן כרום אמיתי על חלונות
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        # חבילת ההסוואה: הכותרות האלו אומרות לסופר-פארם "אני אדם אמיתי שגולש מכרום, הגעתי מגוגל ואני מבין עברית"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "he-IL,he;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0"
        }

    def _extract_price_from_json_ld(self, soup: BeautifulSoup) -> float:
        """
        השיטה המתקדמת ביותר: קריאת נתונים מובנים (Structured Data).
        עובד בצורה מושלמת על אתרי Shopify, WooCommerce ועוד.
        """
        scripts = soup.find_all('script', type='application/ld+json')
        
        for script in scripts:
            try:
                data = json.loads(script.text)
                
                if isinstance(data, list):
                    for item in data:
                        if item.get('@type') == 'Product' and 'offers' in item:
                            return self._parse_offer(item['offers'])
                            
                elif isinstance(data, dict):
                    if data.get('@type') == 'Product' and 'offers' in data:
                        return self._parse_offer(data['offers'])
            except:
                continue 
                
        return 0.0

    def _parse_offer(self, offers) -> float:
        """
        פונקציית עזר לחילוץ המחיר מתוך אובייקט ההצעה של JSON-LD.
        """
        try:
            if isinstance(offers, list) and len(offers) > 0:
                offers = offers[0]
                
            if 'price' in offers:
                return float(offers['price'])
        except:
            pass
        return 0.0

    def _extract_price_html_fallback(self, soup: BeautifulSoup) -> float:
        """
        לוגיקת הגיבוי: אם אין JSON-LD, מחפשים לפי CSS.
        הוספנו סלקטורים ספציפיים ל-Shopify.
        """
        price_selectors = [
            '.price-item--regular', '.price-item--sale', 
            'span.money', 
            '.woocommerce-Price-amount', 
            '.price',
            '.product-price',
            '.current-price'
        ]
        
        for selector in price_selectors:
            elements = soup.select(selector)
            for element in elements:
                raw_text = element.get_text()
                clean_text = raw_text.replace('₪', '').replace('ש"ח', '').replace(',', '').strip()
                match = re.search(r'\d+(\.\d+)?', clean_text)
                if match:
                    try:
                        price_val = float(match.group())
                        if price_val > 0:
                            return price_val
                    except ValueError:
                        continue
        return 0.0

    def scrape_product_page(self, url: str) -> dict:
        try:
            print(f"[Scraper] Attempting to bypass protections and fetch: {url}")
            
            # --- השינוי הקריטי מתבצע כאן: הוספנו את ה-headers שמונעים שגיאת 491 ---
            response = self.scraper.get(url, headers=self.headers, timeout=15)
            
            if response.status_code != 200:
                return {"error": f"Failed to retrieve page. Status: {response.status_code}", "status": "failed"}
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- 1. חילוץ כותרת ---
            title = soup.find('h1')
            title_text = title.text.strip() if title else ""
            if not title_text and soup.title:
                title_text = soup.title.text.strip()
            
            # --- 2. חילוץ מחיר ---
            price_number = self._extract_price_from_json_ld(soup)
            
            if price_number == 0.0:
                price_number = self._extract_price_html_fallback(soup)
            
            # --- 3. חילוץ תמונה ---
            image_url = None
            img_tag = soup.find("meta", property="og:image")
            if img_tag:
                image_url = img_tag["content"]
            
            if not image_url:
                img = soup.find("img")
                if img:
                    image_url = img.get("src")

            if image_url and image_url.startswith("//"):
                image_url = "https:" + image_url

            return {
                "url": url,
                "title": title_text,
                "price_ils": price_number,
                "image_url": image_url,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "status": "failed"}

# --- בלוק הבדיקה ---
if __name__ == "__main__":
    scraper = IsraeliSiteScraper()
    # נבדוק את סופר-פארם ישירות כדי לראות שעקפנו את ה-491
    test_url = "https://shop.super-pharm.co.il/"
    
    print("-" * 60)
    result = scraper.scrape_product_page(test_url)
    
    if result.get("status") == "success":
        print("Scraping Successful! ✅ bypassed the Firewall!")
        print(f"Product Title: {result['title']}")
        print(f"Price Extracted: {result['price_ils']} ILS") 
        print(f"Image URL: {result['image_url']}")
    else:
        print(f"Scraping Failed ❌: {result['error']}")
    print("-" * 60)