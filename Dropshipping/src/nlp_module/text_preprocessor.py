import re
from deep_translator import GoogleTranslator

class TextPreprocessor:
    # מחלקה זו אחראית להכין וליישר את הטקסטים לפני שהם מוזנים למודל הבינה המלאכותית
    
    def __init__(self):
        # אתחול המתרגם: שפת מקור עברית (iw/he), שפת יעד אנגלית (en)
        self.translator = GoogleTranslator(source='iw', target='en')
        
        # רשימה בסיסית של "מילות עצירה" (Stop-words) באנגלית שאינן תורמות למשמעות המוצר
        self.stop_words = {'the', 'is', 'in', 'at', 'of', 'on', 'and', 'a', 'to', 'for', 'with'}

    def clean_text(self, text: str) -> str:
        # פונקציית העזר שמבצעת את "מכונת הכביסה" על הטקסט האנגלי
        
        # 1. הפיכה לאותיות קטנות (Lowercase) כדי ש-Watch ו-watch ייחשבו כאותו דבר
        text = text.lower()
        
        # 2. ניקוי רעשים: מחיקת כל מה שאינו אות באנגלית, מספר או רווח (מעלים אימוג'ים וסימני פיסוק)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # 3. מחיקת רווחים כפולים שנוצרו בעקבות המחיקות הקודמות
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. הסרת מילות עצירה (Stop-words)
        words = text.split()
        clean_words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(clean_words)

    def process_israeli_target(self, hebrew_text: str) -> str:
        # פונקציה ראשית 1: מטפלת בטקסט מהאתר הישראלי (כוללת תרגום)
        print(f"Original Hebrew: '{hebrew_text}'")
        
        translated = self.translator.translate(hebrew_text)
        print(f"Translated to English: '{translated}'")
        
        final_clean = self.clean_text(translated)
        return final_clean

    def process_chinese_db(self, english_text: str) -> str:
        # פונקציה ראשית 2: מטפלת בטקסט מהאתר הסיני (רק ניקוי, ללא תרגום)
        print(f"Original AliExpress: '{english_text}'")
        
        final_clean = self.clean_text(english_text)
        return final_clean

# בלוק בדיקה: ירוץ רק אם נפעיל את הקובץ הזה ישירות
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    print("Testing NLP Preprocessing Pipeline...\n" + "-"*40)
    
    # טקסט דמה מאתר ישראלי (עם סמיילים וסימני קריאה)
    israeli_text = "שעון ספורט עמיד במים לגבר ב-50% הנחה 🔥!!!"
    
    # טקסט דמה מעלי אקספרס
    aliexpress_text = "Waterproof sport watch for men at 50% discount [SALE]"
    
    print("--- Processing Israeli Site ---")
    clean_target = preprocessor.process_israeli_target(israeli_text)
    print(f"FINAL CLEAN TARGET: '{clean_target}'\n")
    
    print("--- Processing AliExpress Site ---")
    clean_db = preprocessor.process_chinese_db(aliexpress_text)
    print(f"FINAL CLEAN DB: '{clean_db}'\n")