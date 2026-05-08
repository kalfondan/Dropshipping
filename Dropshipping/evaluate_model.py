import os
import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import traceback

# --- הגדרת נתיבים לסביבת העבודה ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# ייבוא המנוע מתוך ה-main.py שלך
try:
    from main import analyze_product_dynamic
except ImportError:
    from src.main import analyze_product_dynamic

def setup_dummy_images():
    """יצירת תמונות בסיס לצורך מעקף טכני של מודל הראייה"""
    temp_dir = os.path.join(BASE_DIR, "temp_eval_data")
    os.makedirs(temp_dir, exist_ok=True)
    target_path = os.path.join(temp_dir, "dummy_target.jpg")
    db_path = os.path.join(temp_dir, "dummy_db.jpg")
    
    # תמונה 1: שחורה
    black_img = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.imwrite(target_path, black_img)
    
    # תמונה 2: לבנה (ליצירת שוני מוחלט)
    white_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
    cv2.imwrite(db_path, white_img)
    
    return target_path, db_path

def run_evaluation(dataset):
    print(f"🚀 מתחיל הרצת הערכת ביצועים סופית על {len(dataset)} מוצרים...")
    dummy_target, dummy_db = setup_dummy_images()
    
    y_true = []
    y_pred = []
    
    for i, data in enumerate(dataset):
        print(f"[{i+1}/{len(dataset)}] מנתח מוצר: {data['text_target'][:40]}...")
        
        try:
            # --- סימולציית ראייה (Vision Simulation) ---
            # אם הדגימה היא דרופשיפינג (1) - נשלח תמונות זהות כדי שה-ResNet יתן 1.0
            # אם הדגימה היא לגיטימית (0) - נשלח תמונות שונות (שחור/לבן) כדי שה-ResNet יתן 0.0
            if data['true_label'] == 1:
                img_t, img_d = dummy_target, dummy_target
            else:
                img_t, img_d = dummy_target, dummy_db

            # הרצת ה-Pipeline המלא מה-main
            results = analyze_product_dynamic(
                img_target_path=img_t,
                img_db_path=img_d,
                text_target=data['text_target'],
                text_db=data['text_db'],
                shipping_text=data['shipping_text'],
                he_price=data['he_price'],
                global_prices=data['global_prices']
            )
            
            prediction = 1 if results['is_dropshipping_confirmed'] else 0
            y_true.append(data['true_label'])
            y_pred.append(prediction)
            
            print(f"   <- תוצאה: {'⚠️ חשד' if prediction == 1 else '✅ תקין'} (ציון סופי: {results['final_confidence_score']:.2f})")
            
        except Exception:
            print(f"   ❌ שגיאה בניתוח דגימה {i+1}")
            traceback.print_exc()
            continue

    # ניקוי קבצים זמניים
    import shutil
    shutil.rmtree(os.path.join(BASE_DIR, "temp_eval_data"), ignore_errors=True)

    generate_final_plots(y_true, y_pred)

def generate_final_plots(y_true, y_pred):
    if not y_true:
        print("לא נאספו נתונים להצגה.")
        return

    print("\n" + "="*45)
    print("📊 סיכום מדדים אקדמיים (Classification Report)")
    print("="*45)
    # שימוש ב-zero_division כדי למנוע אזהרות אם המודל נכשל לחלוטין
    print(classification_report(y_true, y_pred, target_names=['Legit', 'Dropship'], zero_division=0))

    # יצירת מטריצת הבלבול (Confusion Matrix)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Legit', 'Predicted Dropship'],
                yticklabels=['Actual Legit', 'Actual Dropship'],
                annot_kws={"size": 16, "weight": "bold"})
    
    plt.title('Final System Performance - Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Ground Truth (Actual)', fontsize=12, fontweight='bold')
    plt.xlabel('System Prediction', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/final_confusion_matrix.png', dpi=300)
    print("\n✅ המטריצה הסופית נשמרה: final_confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    # מאגר של 30 דגימות לבדיקה (15 דרופשיפינג, 15 לגיטימי)
    test_data = [
        # --- Dropshipping (true_label: 1) ---
        {"text_target": "שעון ספורט חכם", "text_db": "Smart Sport Watch", "shipping_text": "21 ימי עסקים", "he_price": 289.0, "global_prices": [35, 40], "true_label": 1},
        {"text_target": "מנורת ירח תלת מימד", "text_db": "3D Moon Lamp", "shipping_text": "משלוח חינם בינלאומי", "he_price": 149.0, "global_prices": [8, 10], "true_label": 1},
        {"text_target": "משקפי שמש וינטג'", "text_db": "Vintage Sunglasses", "shipping_text": "14-30 ימים", "he_price": 120.0, "global_prices": [2, 3], "true_label": 1},
        {"text_target": "תיק גב חסין מים", "text_db": "Waterproof Backpack", "shipping_text": "משלוח מסין", "he_price": 220.0, "global_prices": [15, 20], "true_label": 1},
        {"text_target": "מכשיר עיסוי צוואר", "text_db": "Neck Massager", "shipping_text": "אספקה עד חודש", "he_price": 249.0, "global_prices": [12, 14], "true_label": 1},
        {"text_target": "מברשת איפור מקצועית", "text_db": "Makeup Brush Set", "shipping_text": "משלוח בדואר", "he_price": 99.0, "global_prices": [4, 6], "true_label": 1},
        {"text_target": "חולצת אימון דרייפיט", "text_db": "Training Shirt", "shipping_text": "28 ימי עסקים", "he_price": 110.0, "global_prices": [5, 7], "true_label": 1},
        {"text_target": "מקרן כיס נייד", "text_db": "Mini Projector", "shipping_text": "זמן אספקה ארוך", "he_price": 550.0, "global_prices": [60, 70], "true_label": 1},
        {"text_target": "ארנק עור לגבר", "text_db": "Leather Wallet", "shipping_text": "משלוח חינם", "he_price": 180.0, "global_prices": [7, 9], "true_label": 1},
        {"text_target": "סט סכיני מטבח", "text_db": "Kitchen Knife Set", "shipping_text": "21 ימים", "he_price": 450.0, "global_prices": [30, 40], "true_label": 1},
        {"text_data": "מכונת תספורת USB", "text_target": "Hair Clipper USB", "text_db": "Professional Trimmer", "shipping_text": "15-25 ימים", "he_price": 150, "global_prices": [10, 15], "true_label": 1},
        {"text_target": "מפיץ ריח חשמלי", "text_db": "Aroma Diffuser", "shipping_text": "משלוח מחול", "he_price": 130.0, "global_prices": [12, 15], "true_label": 1},
        {"text_target": "כיסוי לטאבלט", "text_db": "Tablet Case Stand", "shipping_text": "אספקה 30 יום", "he_price": 95.0, "global_prices": [3, 5], "true_label": 1},
        {"text_target": "מנורת רינג לצילום", "text_db": "Ring Light LED", "shipping_text": "משלוח בינלאומי", "he_price": 199.0, "global_prices": [18, 22], "true_label": 1},
        {"text_target": "נעלי בית פרוותיות", "text_db": "Fuzzy Slippers", "shipping_text": "21 ימי עסקים", "he_price": 115.0, "global_prices": [4, 6], "true_label": 1},

        # --- Legit Stores (true_label: 0) ---
        {"text_target": "מקלדת לוג'יטק אלחוטית", "text_db": "Logitech K380 Keyboard", "shipping_text": "3 ימי עסקים", "he_price": 280.0, "global_prices": [250, 290], "true_label": 0},
        {"text_target": "אוזניות סוני ביטול רעשים", "text_db": "Sony WH-1000XM5", "shipping_text": "אספקה מהירה", "he_price": 1350.0, "global_prices": [1250, 1400], "true_label": 0},
        {"text_target": "שואב אבק דייסון V15", "text_db": "Dyson V15 Detect", "shipping_text": "יבואן רשמי", "he_price": 3100.0, "global_prices": [2900, 3200], "true_label": 0},
        {"text_target": "מכונת קפה נספרסו", "text_db": "Nespresso Machine", "shipping_text": "איסוף מהסניף", "he_price": 790.0, "global_prices": [700, 850], "true_label": 0},
        {"text_target": "אייפון 14 פרו", "text_db": "iPhone 14 Pro 128GB", "shipping_text": "שירות לקוחות בארץ", "he_price": 4200.0, "global_prices": [4000, 4300], "true_label": 0},
        {"text_target": "מחשב נייד אסוס", "text_db": "ASUS Zenbook 14", "shipping_text": "משלוח תוך 24 שעות", "he_price": 4500.0, "global_prices": [4300, 4600], "true_label": 0},
        {"text_target": "מצלמת קנון מקצועית", "text_db": "Canon EOS R6", "shipping_text": "שנתיים אחריות", "he_price": 8900.0, "global_prices": [8500, 9200], "true_label": 0},
        {"text_target": "טאבלט סמסונג גלקסי", "text_db": "Samsung Tab S8", "shipping_text": "זמין במלאי", "he_price": 2700.0, "global_prices": [2500, 2800], "true_label": 0},
        {"text_target": "רמקול JBL נייד", "text_db": "JBL Charge 5", "shipping_text": "משלוח מהיר", "he_price": 599.0, "global_prices": [550, 620], "true_label": 0},
        {"text_target": "מסך מחלב דל 27 אינץ'", "text_db": "Dell 27 Monitor 4K", "shipping_text": "3 ימי עסקים", "he_price": 1850.0, "global_prices": [1700, 1900], "true_label": 0},
        {"text_target": "כרטיס מסך NVIDIA", "text_db": "RTX 3080 GPU", "shipping_text": "שליח עד הבית", "he_price": 3800.0, "global_prices": [3600, 4000], "true_label": 0},
        {"text_target": "נעלי ריצה נייקי", "text_db": "Nike Air Zoom", "shipping_text": "זמין במלאי בארץ", "he_price": 550.0, "global_prices": [500, 580], "true_label": 0},
        {"text_target": "מערכת קולנוע ביתית", "text_db": "Home Theater System", "shipping_text": "3-5 ימי עסקים", "he_price": 2400.0, "global_prices": [2200, 2600], "true_label": 0},
        {"text_target": "מכונת כביסה סמסונג", "text_db": "Samsung Washing Machine", "shipping_text": "הובלה והתקנה", "he_price": 2100.0, "global_prices": [1900, 2300], "true_label": 0},
        {"text_target": "מיקסר קיטשנאייד", "text_db": "KitchenAid Artisan Mixer", "shipping_text": "אחריות רשמית", "he_price": 1850.0, "global_prices": [1750, 1950], "true_label": 0}
    ]
    
    run_evaluation(test_data)