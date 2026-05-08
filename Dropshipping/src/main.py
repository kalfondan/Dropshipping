import os
import sys

# הוספת נתיב ה-src למערכת כדי שפייתון יזהה את המודולים הפנימיים בצורה תקינה
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision_module.vision_pipeline import DropshippingVisionPipeline
from nlp_module.text_preprocessor import TextPreprocessor
from nlp_module.bert_embedder import DropshippingBertEmbedder
from nlp_module.record_linkage import RecordLinkageEngine
from meta_module.anomaly_detector import AnomalyDetector
from aggregator import DropshippingAggregator

def analyze_product_dynamic(img_target_path: str, img_db_path: str, text_target: str, text_db: str, shipping_text: str, he_price: float, global_prices: list):
    """
    מנוע הניתוח המרכזי (Orchestrator).
    מבצע את כל ה-Pipeline: תרגום, חילוץ מאפייני AI, חישוב סטטיסטי והחלטה ע"י מודל ML.
    """
    
    # ---------------------------------------------------------
    # 1. ניתוח ויזואלי (ResNet50 + SSIM)
    # ---------------------------------------------------------
    print("\n[1/4] Starting Vision Analysis...")
    vision_engine = DropshippingVisionPipeline()
    vis_res = vision_engine.check_if_image_is_dropshipped(img_target_path, [img_db_path])
    v_score = vis_res['highest_similarity_score']

    # ---------------------------------------------------------
    # 2. ניתוח טקסטואלי היברידי (Translation + BERT + Record Linkage)
    # ---------------------------------------------------------
    print("[2/4] Starting NLP Analysis (with Translation)...")
    
    # שלב א': תרגום מעברית לאנגלית וניקוי (התיקון הקריטי!)
    preprocessor = TextPreprocessor()
    english_target = preprocessor.process_israeli_target(text_target)
    english_db = preprocessor.process_chinese_db(text_db)
    
    # שלב ב': הפקת Embeddings וחישוב דמיון סמנטי
    nlp_engine = DropshippingBertEmbedder()
    vec_target = nlp_engine.get_sentence_embedding(english_target)
    vec_db = nlp_engine.get_sentence_embedding(english_db)
    bert_score = nlp_engine.calculate_similarity(vec_target, vec_db)
    
    # שלב ג': שקלול Record Linkage (Jaccard + Levenshtein)
    linkage_engine = RecordLinkageEngine()
    hybrid_nlp_results = linkage_engine.get_hybrid_nlp_score(english_target, english_db, bert_score)
    n_score = hybrid_nlp_results['hybrid_score']

    # ---------------------------------------------------------
    # 3. ניתוח מטא-דאטה וסטטיסטיקה (Z-Score)
    # ---------------------------------------------------------
    print("[3/4] Running Statistical Anomaly Detection...")
    meta_analyzer = AnomalyDetector()
    ship_info = meta_analyzer.analyze_shipping(shipping_text)
    
    # חישוב אנומליה במחיר - מחזיר Z-Score גולמי למודל ה-ML
    price_info = meta_analyzer.analyze_price(he_price, global_prices)
    raw_z_score = price_info.get('z_score', 0.0)

    # # ---------------------------------------------------------
    # # 4. היתוך סופי (Machine Learning Aggregator)
    # # ---------------------------------------------------------
    # print("[4/4] Executing Final ML Decision (Logistic Regression)...")
    # aggregator = DropshippingAggregator()
    
    # # המודל מקבל את 3 הציונים ומחזיר פסק דין
    # fusion_result = aggregator.fuse_results(v_score, n_score, raw_z_score)


    # ---------------------------------------------------------
    # 4. היתוך סופי (Machine Learning Aggregator)
    # ---------------------------------------------------------
    print("[4/4] Executing Final ML Decision (Neural Network)...")
    
    # שליפת ציון ה-Jaccard (Linkage) מתוך התוצאות ההיברידיות
    linkage_score = hybrid_nlp_results['metrics']['jaccard_word_score']
    
    aggregator = DropshippingAggregator()
    
    # שים לב! עכשיו אנחנו שולחים 4 פרמטרים במקום 3
    fusion_result = aggregator.fuse_results(v_score, n_score, linkage_score, raw_z_score)


    return {
        "is_dropshipping_confirmed": fusion_result["is_dropshipping_confirmed"],
        "final_confidence_score": fusion_result["final_confidence_score"],
        "breakdown": {
            "vision_score": float(v_score),
            "nlp_score": float(n_score),
            "price_anomaly_z": float(raw_z_score),
            "nlp_metrics": hybrid_nlp_results['metrics'],
            "price_stats": price_info,
            "shipping_info": ship_info
        }
    }

def run_test():
    """
    פונקציה להרצה ידנית של ה-Pipeline לצורך בדיקת תקינות הקוד (Sanity Check)
    """
    # הגדרת נתיבים יחסיים
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_img = os.path.join(base_dir, "data", "israeli_target_images", "suspected_watch.jpg")
    db_img = os.path.join(base_dir, "data", "chinese_db_images", "aliexpress_watch_match.jpg")
    
    # נתוני דמה לבדיקה
    results = analyze_product_dynamic(
        img_target_path=target_img,
        img_db_path=db_img,
        text_target="שעון ספורט חכם עמיד במים",
        text_db="Smart waterproof sports watch for men",
        shipping_text="משלוח תוך 14-21 ימי עסקים",
        he_price=350.0,
        global_prices=[15.0, 18.0, 14.5, 20.0]
    )
    
    print("\n" + "="*50)
    print(f"FINAL RESULT: {'⚠️ DROPSHIPPING' if results['is_dropshipping_confirmed'] else '✅ LEGIT'}")
    print(f"CONFIDENCE: {results['final_confidence_score']*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    run_test()