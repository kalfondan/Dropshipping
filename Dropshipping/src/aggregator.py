import os
import torch
import torch.nn as nn

# ---------------------------------------------------------
# 1. הגדרת רשת הנוירונים (PyTorch MLP)
# ---------------------------------------------------------
class DropshippingMLPHead(nn.Module):
    def __init__(self, input_dim=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1),
            nn.Sigmoid()  # פולט הסתברות בין 0 ל-1
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. מחלקת ה-Aggregator 
# ---------------------------------------------------------
class DropshippingAggregator:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, 'models', 'mlp_head.pt')
        self.threshold = 0.60
        
        # אתחול רשת הנוירונים (מצפה ל-4 משתנים)
        self.model = DropshippingMLPHead(input_dim=4)
        
        # התיקון הקריטי: מעביר את הרשת למצב חיזוי באופן קבוע כדי למנוע קריסת BatchNorm עם פריט בודד
        self.model.eval() 
        
        self.is_trained = False
        
        # טעינת המשקולות המאומנות אם הקובץ קיים
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            self.is_trained = True
            print("[V] Trained PyTorch MLP Model loaded successfully into Aggregator.")
        else:
            print("[!] MLP Model not found. Please run train_mlp.py first.")

    def fuse_results(self, vision_score, nlp_score, linkage_score, z_score):
        """
        מקבל 4 ציונים מהמודולים השונים ומעביר אותם דרך רשת הנוירונים.
        """
        # נרמול ה-Z-Score לטווח של 0-1 (חשוב ליציבות של רשתות נוירונים)
        z_norm = min(1.0, max(0.0, z_score / 3.0))
        
        # בניית וקטור הקלט (Tensor) - מכניסים מוצר אחד (Batch size 1) עם 4 תכונות
        features = torch.tensor([[vision_score, nlp_score, linkage_score, z_norm]], dtype=torch.float32)
        
        # הרצת ה-Inference (חיזוי) ללא חישוב גרדיאנטים
        with torch.no_grad():
            probability = self.model(features).item()
            
        is_dropshipping = probability >= self.threshold
        
        return {
            "final_confidence_score": float(probability),
            "is_dropshipping_confirmed": is_dropshipping,
            "breakdown": {
                "vision_score": float(vision_score),
                "nlp_score": float(nlp_score),
                "linkage_score": float(linkage_score),
                "price_anomaly_z": float(z_score)
            }
        }