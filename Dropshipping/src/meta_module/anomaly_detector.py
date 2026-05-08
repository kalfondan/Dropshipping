import re
import numpy as np

class AnomalyDetector:
    # מודול זה אחראי על ניתוח המטא-דאטה וההיגיון העסקי של האתר.
    # שודרג לזיהוי חריגות סטטיסטיות (Z-Score) להבטחת דיוק מדעי.

    def __init__(self, max_legit_shipping_days=10, max_safe_markup=2.5):
        # הגדרת ספים עסקיים וסטטיסטיים
        self.max_legit_shipping_days = max_legit_shipping_days
        self.max_safe_markup = max_safe_markup
        self.z_score_threshold = 1.2 # מעל 1.5 סטיות תקן נחשב לאנומליה

    def analyze_shipping(self, shipping_text: str) -> dict:
        # חילוץ מספר ימי העסקים מתוך טקסט חופשי באמצעות Regex
        days_found = re.findall(r'\d+', shipping_text)
        if not days_found:
            return {"score": 0.5, "reason": "No shipping time specified"}
        
        max_days = max([int(d) for d in days_found])
        
        is_suspicious = max_days > self.max_legit_shipping_days
        score = 1.0 if is_suspicious else 0.0
        
        return {
            "days": max_days,
            "is_suspicious": is_suspicious,
            "score": score,
            "reason": f"Shipping time: {max_days} days"
        }

    def analyze_price(self, israeli_price: float, global_prices, usd_to_ils: float = 3.7) -> dict:
        """
        מקבל מחיר ישראלי ו-global_prices (שיכול להיות מחיר יחיד או רשימה של מחירים).
        מפעיל חישוב Heuristic או Z-Score בהתאם לנתונים.
        """
        # מקרה 1: Fallback Method (Heuristic Markup) - אם קיבלנו רק מחיר אחד או רשימה קטנה
        if isinstance(global_prices, (int, float)) or (isinstance(global_prices, list) and len(global_prices) < 2):
            single_usd_price = global_prices if isinstance(global_prices, (int, float)) else (global_prices[0] if global_prices else 5.0)
            chinese_price_ils = single_usd_price * usd_to_ils
            markup = israeli_price / chinese_price_ils if chinese_price_ils > 0 else 0
            
            is_suspicious = markup > self.max_safe_markup
            score = 1.0 if is_suspicious else 0.0
            
            return {
                "markup": round(markup, 2),
                "is_suspicious": is_suspicious,
                "score": score,
                "method": "heuristic",
                "z_score": 0.0,
                "reason": f"Price is {markup:.1f}x higher than global average"
            }

        # מקרה 2: Statistical Method (Z-Score) - שדרוג ה-Data Science
        # ממירים את כל מחירי העולם לשקלים
        global_prices_ils = [p * usd_to_ils for p in global_prices]
        
        mean_price = np.mean(global_prices_ils)
        std_dev = np.std(global_prices_ils)
        
        if std_dev == 0:
            std_dev = 1.0 # מניעת חלוקה באפס
            
        # חישוב ציון התקן
        z_score = (israeli_price - mean_price) / std_dev
        
        is_suspicious = z_score > self.z_score_threshold
        
        # נרמול הציון לטווח של 0 עד 1 (מעל 3 סטיות תקן יקבל את הציון המקסימלי 1.0)
        score = min(1.0, max(0.0, z_score / 3.0)) if is_suspicious else 0.0
        markup = israeli_price / mean_price if mean_price > 0 else 0

        return {
            "markup": round(markup, 2),
            "is_suspicious": bool(is_suspicious),
            "score": float(round(score, 2)),
            "method": "z_score",
            "z_score": float(round(z_score, 2)),
            "mean": float(round(mean_price / usd_to_ils, 2)), 
            "reason": f"Anomaly: Z-Score of {z_score:.2f}"
        }

    def check_contact_email(self, email: str) -> dict:
        is_private = bool(re.search(r'@(gmail|hotmail|yahoo|outlook)\.com', email.lower()))
        score = 0.8 if is_private else 0.0
        
        return {
            "is_private_email": is_private,
            "score": score,
            "reason": "Uses private email provider (Gmail/Hotmail)" if is_private else "Uses professional domain"
        }

# בלוק בדיקה (Main)
if __name__ == "__main__":
    detector = AnomalyDetector()
    
    print("Testing Metadata Anomaly Detection...\n" + "-"*50)
    
    ship_info = detector.analyze_shipping("משלוח מהיר עד הבית: 14-21 ימי עסקים")
    email_info = detector.check_contact_email("support.israelsale@gmail.com")
    
    # בדיקת שיטה ישנה (מחיר בודד)
    print("Testing Heuristic (Old) Method:")
    price_info_old = detector.analyze_price(israeli_price=299, global_prices=15.0) 
    print(f"Result: {price_info_old['reason']} -> Suspicious: {price_info_old['is_suspicious']}\n")
    
    # בדיקת שיטה חדשה (רשימת מחירים - Z-Score)
    print("Testing Z-Score (Data Science) Method:")
    israeli_price_test = 299.0
    global_prices_test = [15.0, 16.5, 14.0, 18.0, 15.5] # מחירים בדולרים מאמזון/וולמארט
    price_info_new = detector.analyze_price(israeli_price=israeli_price_test, global_prices=global_prices_test)
    
    print(f"Result: {price_info_new['reason']} -> Suspicious: {price_info_new['is_suspicious']}")
    print(f"Calculated Z-Score: {price_info_new['z_score']}")
    print(f"Anomaly Score (0-1): {price_info_new['score']:.2f}")
    print("-" * 50)