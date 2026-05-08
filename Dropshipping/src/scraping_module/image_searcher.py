import requests

class DropshippingImageSearcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search.json"
        
        # רשת דיג רחבה: אנחנו מחפשים מילות מפתח בתוך הלינק או שם האתר
        self.supplier_keywords = [
            "aliexpress",   
            "temu",         
            "alibaba",      
            "1688",         
            "shein",        
            "dhgate",       
            "banggood",
            "made-in-china",
            "cjdropshipping",
            "gearbest",
            "amazon", 
            "walmart"
        ]

    def find_supplier_match(self, image_url: str) -> dict:
        print(f"🔍 Searching Google Lens for: {image_url}")
        
        # --- שכבת ההגנה: בדיקה שהסקרייפר אכן מצא תמונה לפני שניגשים ל-API ---
        if not image_url or image_url.strip() == "":
            print("[!] Error: No image URL provided. Skipping Google Lens search to avoid 400 Bad Request.")
            return {"status": "error", "error": "No image extracted from target site."}

        params = {"engine": "google_lens", "url": image_url, "api_key": self.api_key}

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            visual_matches = data.get("visual_matches", [])

            # הדפסה לדיבאג: אילו אתרים גוגל מצא?
            print("\n--- [DEBUG] TOP 10 GOOGLE LENS RESULTS ---")
            for i, match in enumerate(visual_matches[:10]):
                print(f"{i+1}. {match.get('source')} -> {match.get('link')[:60]}...")
            print("-------------------------------------------\n")

            supplier_results = []
            for match in visual_matches:
                link = match.get("link", "").lower()
                source = match.get("source", "").lower()
                
                # אם אחת מהמילים ברשימה נמצאת בלינק או בשם המקור
                if any(kw in link or kw in source for kw in self.supplier_keywords):
                    supplier_results.append({
                        "title": match.get("title"),
                        "link": match.get("link"),
                        "source": match.get("source"),
                        "price": match.get("price", {}).get("extracted_value"),
                        "thumbnail": match.get("thumbnail")
                    })
            
            if supplier_results:
                return {
                    "status": "success", 
                    "match_found": True, 
                    "best_match": supplier_results[0],
                    "all_matches": supplier_results, # מחזירים את כל ההתאמות ל-UI
                    "total_supplier_matches": len(supplier_results)
                }
            
            # אם לא מצאנו כלום, נחזיר את ה-5 הראשונים מדיבאג
            return {
                "status": "success", 
                "match_found": False,
                "top_other_matches": visual_matches[:5]
            }
            
        except Exception as e:
            print(f"[!] API Request Failed: {e}")
            return {"status": "error", "error": str(e)}

# --- בלוק בדיקה ---
if __name__ == "__main__":
    MY_API_KEY = "ac97d95421f90a5426adfd830a8574a663e406eb1dfb30bd6354e07d24ea6853" 
    searcher = DropshippingImageSearcher(api_key=MY_API_KEY)
    test_image_url = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/items/poke-ball.png" 
    
    print("-" * 60)
    result = searcher.find_supplier_match(test_image_url)
    
    if result.get("status") == "success":
        if result["match_found"]:
            print(f"✅ Found {result['total_supplier_matches']} match(es) on Supplier Sites!")
            print(f"Title: {result['best_match']['title']}")
        else:
            print("❌ No dropshipping supplier matches found.")
    else:
        print(f"⚠️ Error: {result.get('error')}")
    print("-" * 60)