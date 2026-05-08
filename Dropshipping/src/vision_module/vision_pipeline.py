import torch
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from .feature_extractor import VisionFeatureExtractor

class DropshippingVisionPipeline:
    def __init__(self):
        self.extractor = VisionFeatureExtractor()

    def _calculate_ssim(self, img_path1, img_path2):
        # קריאת התמונות והמרתן לאפור 
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
        
        # הגנת קריסה במקרה של תמונה פגומה
        if img1 is None or img2 is None:
            return 0.0
            
        # --- שדרוג: Data Augmentation לשוויון איכויות ---
        
        # 1. נרמול רזולוציה: הקטנת שתי התמונות לגודל ריבועי אחיד
        # זה מונע את העיוות שנוצר כשהשווינו HD ל-Thumbnail
        target_size = (256, 256)
        img1_res = cv2.resize(img1, target_size, interpolation=cv2.INTER_AREA)
        img2_res = cv2.resize(img2, target_size, interpolation=cv2.INTER_AREA)
        
        # 2. טשטוש גאוסיאני (Gaussian Blur)
        # מעלים את ה"רעש" הדיגיטלי ואת הפיקסול מהתמונות המוקטנות של גוגל
        img1_aug = cv2.GaussianBlur(img1_res, (5, 5), 0)
        img2_aug = cv2.GaussianBlur(img2_res, (5, 5), 0)
        
        # חישוב המדד המבני על התמונות שעברו אוגמנטציה
        score, _ = ssim(img1_aug, img2_aug, full=True)
        return score

    def check_if_image_is_dropshipped(self, target_image_path, db_images_paths, threshold=0.85):
        target_features = self.extractor.extract_features(target_image_path)
        
        highest_score = 0
        best_match_path = None
        
        for db_path in db_images_paths:
            db_features = self.extractor.extract_features(db_path)
            
            # 1. דמיון AI עמוק (Cosine Similarity של ResNet)
            ai_sim = torch.nn.functional.cosine_similarity(target_features, db_features, dim=0).item()
            
            # 2. דמיון מבני עם אוגמנטציה (SSIM)
            struct_sim = self._calculate_ssim(target_image_path, db_path)
            
            # 3. שקלול סופי (50% AI, 50% SSIM)
            combined_score = (ai_sim * 0.5) + (struct_sim * 0.5)
            
            if combined_score > highest_score:
                highest_score = combined_score
                best_match_path = db_path
        
        return {
            "is_dropshipping_image": highest_score >= threshold,
            "highest_similarity_score": round(highest_score, 4),
            "matched_with_image": best_match_path,
            "threshold_used": threshold
        }