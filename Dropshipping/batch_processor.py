import os
import time
import tempfile
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

# ייבוא המודולים שלך מהפרויקט
from src.scraping_module.site_scraper import IsraeliSiteScraper
from src.scraping_module.image_searcher import DropshippingImageSearcher
from src.main import analyze_product_dynamic
from src.vision_module.vision_pipeline import DropshippingVisionPipeline

# 🔑 המפתח שלך (הועתק מ-app.py)
SERPAPI_KEY = "ac97d95421f90a5426adfd830a8574a663e406eb1dfb30bd6354e07d24ea6853"

# ==========================================
# פונקציות עזר (הועתקו מ-app.py)
# ==========================================
def download_to_temp(url):
    if not url: return None
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(response.content)
                return tmp.name
    except:
        return None
    return None

def fetch_price_fallback(url: str) -> float:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        }
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            amz_price = soup.find('span', class_='a-price-whole')
            if not amz_price: amz_price = soup.find('span', id='priceblock_ourprice')
            if amz_price: return float(re.sub(r'[^\d.]', '', amz_price.text))
            wmt_price = soup.find('span', itemprop='price')
            if not wmt_price:
                wmt_div = soup.find('div', attrs={"data-testid": "add-to-cart-section"})
                if wmt_div: wmt_price = wmt_div.find('span', class_='w_iUH7')
            if wmt_price: return float(re.sub(r'[^\d.]', '', wmt_price.text))
            text = soup.get_text()
            match = re.search(r'\$\s*([0-9,.]+)', text)
            if match: return float(match.group(1).replace(',', ''))
    except Exception:
        pass
    return 0.0

# ==========================================
# ה"מוח" המרכזי שמריץ מוצר בודד (בלי UI)
# ==========================================
def process_single_url(target_url):
    temp_target = None
    temp_db = None
    
    try:
        # 1. סריקת האתר הישראלי
        scraper = IsraeliSiteScraper()
        site_data = scraper.scrape_product_page(target_url)
        
        if site_data['status'] == 'failed':
            raise Exception(f"Scraping failed: {site_data.get('error')}")
        
        he_title = site_data['title']
        he_price = site_data['price_ils']
        he_img_url = site_data.get('image_url')
        
        if he_price <= 0.0: he_price = 149.0
        if not he_img_url: raise Exception("No image found on target site")

        # 2. חיפוש בגוגל לנס
        searcher = DropshippingImageSearcher(api_key=SERPAPI_KEY)
        search_results = searcher.find_supplier_match(he_img_url)
        
        if search_results.get("status") == "error":
            raise Exception(f"API Error: {search_results.get('error')}")

        if not search_results.get("match_found", False):
            raise Exception("No visual matches found globally")

        # 3. אימות ויזואלי (מעל 0.35)
        temp_target = download_to_temp(he_img_url)
        verified_matches = []
        vision_engine = DropshippingVisionPipeline()
        
        all_matches = search_results.get('all_matches', [])
        if not all_matches and search_results.get('best_match'):
            all_matches = [search_results['best_match']]
            
        for match in all_matches:
            match_thumb = match.get('thumbnail')
            if match_thumb:
                temp_match_img = download_to_temp(match_thumb)
                if temp_match_img:
                    vis_res = vision_engine.check_if_image_is_dropshipped(temp_target, [temp_match_img])
                    match_score = vis_res['highest_similarity_score']
                    os.remove(temp_match_img)
                    if match_score >= 0.35:
                        verified_matches.append(match)

        if not verified_matches:
            raise Exception("Matches found but visual similarity < 0.35")

        # 4. חילוץ מחירי שוק גלובליים
        global_prices_list = []
        for match in verified_matches:
            price_val = 0.0
            try:
                price_str = str(match.get('price', '0')).replace('$', '').replace(',', '').strip()
                if price_str and price_str.replace('.', '', 1).isdigit():
                    price_val = float(price_str)
            except: pass
                
            if price_val <= 0.0 and match.get('link'):
                fallback_val = fetch_price_fallback(match['link'])
                if fallback_val > 0: price_val = fallback_val
                    
            if price_val > 0:
                global_prices_list.append(price_val)

        # 5. קבלת החלטה סופית ברשת הנוירונים (MLP)
        best_match = verified_matches[0]
        en_title = best_match.get('title', 'Unknown')
        ch_img_url = best_match.get('thumbnail', he_img_url)
        temp_db = download_to_temp(ch_img_url)

        final_results = analyze_product_dynamic(
            temp_target,
            temp_db if temp_db else temp_target,
            he_title,
            en_title,
            "7 days", 
            he_price,
            global_prices_list 
        )
        
        # שמירת שם המוצר כדי שנוכל לכתוב אותו באקסל
        final_results['product_name'] = he_title
        return final_results

    finally:
        # ניקיון זיכרון חשוב!
        if temp_target and os.path.exists(temp_target): os.remove(temp_target)
        if temp_db and os.path.exists(temp_db): os.remove(temp_db)

# ==========================================
# לולאת הריצה הגדולה של האקסל
# ==========================================
def run_bulk_analysis(input_csv: str, output_csv: str):
    print(f"📦 מתחיל סריקת לינקים מהקובץ: {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        print(f"❌ שגיאה בקריאת הקובץ: {e}")
        return
    
    processed_urls = []
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        processed_urls = existing_df['URL'].tolist()
        print(f"🔄 ממשיך סריקה ומדלג על {len(processed_urls)} לינקים שכבר עובדו...")

    with open(output_csv, 'a', encoding='utf-8-sig') as f:
        if not processed_urls:
            f.write("URL,Product_Name,Is_Dropship,Confidence,Z_Score,Error\n")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing Products"):
            current_url = str(row['url']).strip()
            
            # מסנן אוטומטית לינקים שאינם מוצרים כדי למנוע קריסות של ה-Scraper
            if "/p/" not in current_url:
                continue 
            
            if current_url in processed_urls:
                continue
            
            try:
                # הרצת התהליך המלא על לינק בודד
                results = process_single_url(current_url)
                
                name = results.get('product_name', 'Unknown')
                is_drop = "Yes" if results['is_dropshipping_confirmed'] else "No"
                conf = f"{results['final_confidence_score']*100:.2f}%"
                z = results['breakdown']['price_anomaly_z']
                
                clean_name = str(name).replace(',', '.')
                f.write(f"\"{current_url}\",\"{clean_name}\",{is_drop},{conf},{z},None\n")
                f.flush()

            except Exception as e:
                error_msg = str(e).replace(',', '.')
                f.write(f"\"{current_url}\",Unknown,ERROR,0.0,0.0,\"{error_msg}\"\n")
                f.flush()
            
            # השהייה אנושית של 5 שניות למניעת חסימת IP מול סופר-פארם
            time.sleep(5) 

    print(f"\n✅ סריקת האצווה הסתיימה! התוצאות נשמרו ב: {output_csv}")

if __name__ == "__main__":
    INPUT_FILE = "urls_database.csv"
    OUTPUT_FILE = "final_200_results.csv"
    run_bulk_analysis(INPUT_FILE, OUTPUT_FILE)