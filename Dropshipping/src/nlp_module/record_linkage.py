import re

class RecordLinkageEngine:
    # מנוע זה מוודא התאמה מדויקת של תווים ומילים (לזיהוי מספרי דגמים ומותגים)
    # המטרה: לפצות על ה"עיוורון" של מודלי שפה עמוקים (BERT) לדקויות של אותיות ומספרים.

    @staticmethod
    def clean_text(text: str) -> str:
        # ניקוי טקסט בסיסי: אותיות קטנות והסרת סימני פיסוק
        if not text:
            return ""
        text = text.lower()
        return re.sub(r'[^\w\s]', '', text)

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        מחשב את מדד ג'קארד (Jaccard Index) - חיתוך חלקי איחוד של מילים.
        מצוין לבדיקה אם שתי הכותרות מכילות את אותן מילות מפתח, ללא קשר לסדר שלהן.
        """
        set1 = set(self.clean_text(text1).split())
        set2 = set(self.clean_text(text2).split())
        
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union

    def levenshtein_ratio(self, s1: str, s2: str) -> float:
        """
        מימוש קלאסי של אלגוריתם Levenshtein Distance (מרחק עריכה) מבוסס תכנות דינמי.
        מודד כמה פעולות (הכנסה, מחיקה, החלפה) נדרשות כדי להפוך מחרוזת אחת לשנייה.
        """
        s1, s2 = self.clean_text(s1), self.clean_text(s2)
        if len(s1) < len(s2):
            return self.levenshtein_ratio(s2, s1)
        if len(s2) == 0:
            return 0.0

        # יצירת מטריצת המרחקים של תכנות דינמי (Dynamic Programming)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        max_len = max(len(s1), len(s2))
        distance = previous_row[-1]
        
        # נרמול הציון לטווח של 0 עד 1 (1.0 = התאמה מושלמת)
        return 1.0 - (distance / max_len)

    def get_hybrid_nlp_score(self, target_text: str, db_text: str, bert_score: float) -> dict:
        """
        משקלל את ההבנה הסמנטית של BERT יחד עם ההתאמה המדויקת של Record Linkage.
        """
        jaccard = self.jaccard_similarity(target_text, db_text)
        levenshtein = self.levenshtein_ratio(target_text, db_text)
        
        # נוסחת שקלול היברידית:
        # 50% משמעות וסמנטיקה (BERT)
        # 30% חפיפת מילים מדויקות (Jaccard)
        # 20% חפיפת תווים מדויקים - לזיהוי דגמים (Levenshtein)
        hybrid_score = (bert_score * 0.5) + (jaccard * 0.3) + (levenshtein * 0.2)
        
        return {
            "hybrid_score": float(hybrid_score),
            "metrics": {
                "bert_semantic_score": float(round(bert_score, 4)),
                "jaccard_word_score": float(round(jaccard, 4)),
                "levenshtein_char_score": float(round(levenshtein, 4))
            }
        }