import streamlit as st
import os
import tempfile
import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

from src.scraping_module.site_scraper import IsraeliSiteScraper
from src.scraping_module.image_searcher import DropshippingImageSearcher
from src.main import analyze_product_dynamic
from src.vision_module.vision_pipeline import DropshippingVisionPipeline

st.set_page_config(page_title="Dropshipping AI V2.0 - Final", page_icon="🤖", layout="wide")

# מפתח ה-API של SerpApi
SERPAPI_KEY = "ac97d95421f90a5426adfd830a8574a663e406eb1dfb30bd6354e07d24ea6853" 

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
    """
    מתחזה לסורק הרשמי של גוגל (Googlebot) כדי לעקוף חסימות Anti-Bot 
    ולשלוף את המחיר מאתרים קשוחים כמו אמזון ווולמארט.
    """
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
            
            # --- חיפוש במבנה של אמזון ---
            amz_price = soup.find('span', class_='a-price-whole')
            if not amz_price:
                amz_price = soup.find('span', id='priceblock_ourprice')
            if amz_price:
                return float(re.sub(r'[^\d.]', '', amz_price.text))
                
            # --- חיפוש במבנה של וולמארט ---
            wmt_price = soup.find('span', itemprop='price')
            if not wmt_price:
                wmt_div = soup.find('div', attrs={"data-testid": "add-to-cart-section"})
                if wmt_div:
                    wmt_price = wmt_div.find('span', class_='w_iUH7')
            if wmt_price:
                return float(re.sub(r'[^\d.]', '', wmt_price.text))
                
            # --- גיבוי אחרון: חיפוש תבנית דולרית גנרית בדף ---
            text = soup.get_text()
            match = re.search(r'\$\s*([0-9,.]+)', text)
            if match:
                return float(match.group(1).replace(',', ''))
    except Exception:
        pass
    
    return 0.0

st.markdown("""
    <dir dir="rtl" style="text-align: right;">
        <h1>🤖 Dropshipping Detector V2.0 - סריקה סטטיסטית</h1>
        <p>המערכת מחפשת מקור בסין ומנתחת אנומליות מחיר (Z-Score) בעזרת AI</p>
    </dir>
""", unsafe_allow_html=True)

target_url = st.text_input("הדבק כאן לינק למוצר לבדיקה:", placeholder="https://www.gadgetshop.co.il/...")

if st.button("בצע ניתוח אוטומטי מלא 🚀"):
    if not target_url:
        st.error("אנא הזן לינק תקין.")
    else:
        with st.spinner('מבצע סריקה וניתוח (זה עשוי לקחת כ-15-30 שניות)...'):
            
            scraper = IsraeliSiteScraper()
            site_data = scraper.scrape_product_page(target_url)
            
            if site_data['status'] == 'failed':
                st.error(f"שגיאה בסריקת האתר: {site_data['error']}")
                st.stop()
            
            he_title = site_data['title']
            he_price = site_data['price_ils']
            he_img_url = site_data.get('image_url')
            
            if he_price <= 0.0:
                he_price = 149.0
            
            st.info(f"📍 נמצא באתר הישראלי: {he_title} | מחיר: {he_price}₪")
            if he_img_url:
                st.image(he_img_url, width=200, caption="התמונה שזוהתה באתר")

            searcher = DropshippingImageSearcher(api_key=SERPAPI_KEY)
            search_results = searcher.find_supplier_match(he_img_url if he_img_url else "")
            
            if search_results.get("status") == "error":
                st.error(f"🚨 שגיאת התחברות ל-API של גוגל לנס: {search_results.get('error')}")
                st.stop()

            found_match = search_results.get("match_found", False)
            verified_matches = []
            temp_target = download_to_temp(he_img_url)

            # --- 3. סינון ויזואלי מחמיר ---
            if found_match and temp_target:
                vision_engine = DropshippingVisionPipeline()
                MIN_VISUAL_SCORE = 0.35 
                
                all_matches = search_results.get('all_matches', [])
                if not all_matches and search_results.get('best_match'):
                    all_matches = [search_results['best_match']]
                
                with st.spinner('מאמת ויזואלית את התוצאות מגוגל מול המוצר המקורי...'):
                    for match in all_matches:
                        match_thumb = match.get('thumbnail')
                        match['debug_score'] = 0.0 
                        
                        if match_thumb:
                            temp_match_img = download_to_temp(match_thumb)
                            if temp_match_img:
                                vis_res = vision_engine.check_if_image_is_dropshipped(temp_target, [temp_match_img])
                                match_score = vis_res['highest_similarity_score']
                                match['debug_score'] = match_score 
                                os.remove(temp_match_img)
                                
                                if match_score >= MIN_VISUAL_SCORE:
                                    verified_matches.append(match)

            if not verified_matches:
                st.success("✅ לא נמצאה התאמה ויזואלית ודאית באתרי הספקים. המערכת עוצרת את הניתוח.")
                if temp_target and os.path.exists(temp_target): os.remove(temp_target)
                st.stop()

            # --- 4. איסוף נתונים סטטיסטיים מכל הספקים שאושרו ---
            st.warning(f"⚠️ התאמה ויזואלית אומתה בהצלחה! אוסף נתוני שוק לחישוב סטטיסטי (Z-Score)...")
            
            global_prices_list = []
            
            # חילוץ מחירים לפני ההגעה לפונקציה הראשית
            for match in verified_matches:
                price_val = 0.0
                try:
                    price_str = str(match.get('price', '0')).replace('$', '').replace(',', '').strip()
                    if price_str and price_str.replace('.', '', 1).isdigit():
                        price_val = float(price_str)
                except Exception:
                    pass
                    
                if price_val <= 0.0 and match.get('link'):
                    fallback_val = fetch_price_fallback(match['link'])
                    if fallback_val > 0:
                        price_val = fallback_val
                        
                if price_val > 0:
                    global_prices_list.append(price_val)
                    match['extracted_price'] = price_val 

            best_match = verified_matches[0]
            en_title = best_match['title']
            ch_img_url = best_match.get('thumbnail', he_img_url)
            temp_db = download_to_temp(ch_img_url)

            # --- 5. הפעלת מנוע ה-AI המשולב ---
            try:
                final_results = analyze_product_dynamic(
                    temp_target,
                    temp_db if temp_db else temp_target,
                    he_title,
                    en_title,
                    "7 days", 
                    he_price,
                    global_prices_list # כאן אנחנו מעבירים את רשימת המחירים המלאה!
                )

               # --- 6. הצגת התוצאות הסופיות ---
                confidence = final_results['final_confidence_score'] * 100
                st.divider()
                st.subheader("📊 סיכום ניתוח AI וסטטיסטיקה")
                
                # תמיד נציג את הפאנל הסטטיסטי, בלי קשר לתוצאה הסופית!
                stats = final_results['breakdown'].get('statistics', {})
                if stats.get('method') == 'z_score':
                    st.info("💡 ניתוח כלכלי מבוסס סטטיסטיקה (Z-Score Outlier Detection)")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("מחיר ישראל", f"₪{he_price:.2f}")
                    col2.metric("ממוצע גלובלי", f"₪{stats.get('mean', 0) * 3.7:.2f}")
                    
                    z_val = stats.get('z_score', 0)
                    col3.metric("ציון תקן (Z-Score)", f"{z_val:.2f}σ", delta="אנומליה חריגה!" if z_val > 1.5 else "תקין", delta_color="inverse")
                    st.divider()

                # הודעת הסטטוס הסופית (דרופשיפינג או לגיטימי)
                if final_results['is_dropshipping_confirmed']:
                    st.error(f"🚨 חשד גבוה לדרופשיפינג: {confidence:.2f}%")
                    st.progress(final_results['final_confidence_score'])
                    
                    st.subheader("🛒 מצאנו את המוצר אצל הספקים הבאים:")
                    
                    sorted_matches = sorted(verified_matches, key=lambda x: x.get('debug_score', 0.0), reverse=True)
                    table_data = []
                    
                    for match in sorted_matches:
                        price_val = match.get('extracted_price', 0.0)
                        price_display = f"${price_val:.2f}" if price_val > 0 else "לא זמין ב-API"
                            
                        table_data.append({
                            "ספק": match.get('source', 'ספק גלובלי'),
                            "מחיר משוער": price_display,
                            "דמיון ויזואלי": match.get('debug_score', 0.0),
                            "קישור": match.get('link', '#')
                        })

                    if table_data:
                        top_match = table_data[0]
                        st.success(f"🏆 ההתאמה הוויזואלית הגבוהה ביותר: {top_match['ספק']} (ציון: {top_match['דמיון ויזואלי']:.2f})")
                        st.link_button("לחץ למעבר לחנות המומלצת 🛒", top_match['קישור'], type="primary", use_container_width=True)
                        
                        df = pd.DataFrame(table_data)
                        st.dataframe(
                            df,
                            column_config={
                                "ספק": st.column_config.TextColumn("ספק מוכר"),
                                "מחיר משוער": st.column_config.TextColumn("מחיר משוער"),
                                "דמיון ויזואלי": st.column_config.NumberColumn("דמיון ויזואלי", format="%.2f"),
                                "קישור": st.column_config.LinkColumn("קישור לחנות", display_text="מעבר לחנות 🔗")
                            },
                            hide_index=True,
                            use_container_width=True
                        )

                else:
                    st.success(f"✅ המוצר נראה לגיטימי: {confidence:.2f}%")
                    st.progress(final_results['final_confidence_score'])
                
                with st.expander("ראה פירוט טכני (Breakdown)"):
                    st.json(final_results['breakdown'])

            finally:
                if temp_target and os.path.exists(temp_target): os.remove(temp_target)
                if temp_db and os.path.exists(temp_db): os.remove(temp_db)