import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.aggregator import DropshippingMLPHead

def create_synthetic_dataset():
    X_train = []
    y_train = []

    # --- הכנסת הדוגמאות האמיתיות שאספנו מהרשת ---
    
    # 1. כרית סופר-פארם יקרה (229 ש"ח) -> דרופשיפינג / הפקעה חריגה (הנתונים המקוריים)
    X_train.append([0.95, 0.20, 0.00, 1.00]) 
    y_train.append([1.0]) 

    # 2. כרית שופרסל ביניים (139 ש"ח) -> לגיטימי (יבוא סביר)
    X_train.append([0.95, 0.35, 0.10, 0.40]) 
    y_train.append([0.0]) 

    # 3. כרית סופר-פארם זולה (80 ש"ח) -> לגיטימי לחלוטין
    X_train.append([0.90, 0.40, 0.15, 0.10]) 
    y_train.append([0.0])

    # 4. **המקרה החדש!** כרית סופר-פארם יקרה מאוד (229 ש"ח) מהאפליקציה!
    # [vision: 0.6694, bert: 0.7831, jaccard: 0.2667, z-score: 192 (מנורמל ל-1.0)]
    X_train.append([0.6694, 0.7831, 0.2667, 1.00]) 
    y_train.append([1.0]) # התווית: 1.0 (דרופשיפינג / אנומליה)

    # --- שכפול אוטומטי (Tabular Data Augmentation) ---
    # כדי שהרשת לא תלמד רק 4 דוגמאות בודדות, נשכפל אותן עם רעש אקראי
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X_train)):
        base_x = X_train[i]
        base_y = y_train[i]
        
        # יוצרים 20 עותקים לכל מקרה אמיתי
        for _ in range(20):
            noisy_x = [
                min(1.0, max(0.0, base_x[0] + random.uniform(-0.05, 0.05))), # הוספת רעש לראייה
                min(1.0, max(0.0, base_x[1] + random.uniform(-0.05, 0.05))), # הוספת רעש ל-BERT
                base_x[2], # Jaccard לרוב נשאר קבוע כי הוא סופר תווים מדויקים
                min(1.0, max(0.0, base_x[3] + random.uniform(-0.1, 0.1)))    # הוספת רעש ל-Z-Score
            ]
            augmented_X.append(noisy_x)
            augmented_y.append(base_y)

    # מחברים את 4 הדוגמאות המקוריות יחד עם 80 המשוכפלות
    final_X = X_train + augmented_X
    final_y = y_train + augmented_y

    return torch.tensor(final_X, dtype=torch.float32), torch.tensor(final_y, dtype=torch.float32)

def train_network():
    print("🚀 Starting MLP Neural Network Training...")
    
    # טעינת המבנה של הרשת
    model = DropshippingMLPHead(input_dim=4)
    X, y = create_synthetic_dataset()
    
    # הגדרת אופטימיזטור ופונקציית שגיאה
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # הרצת האימון
    model.train()
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {loss.item():.4f}")

    # שמירת המודל המאומן לדריסת הקובץ הישן
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'src', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    save_path = os.path.join(models_dir, 'mlp_head.pt')
    torch.save(model.state_dict(), save_path)
    
    print(f"\n✅ Training Complete! Model saved successfully to:")
    print(save_path)

if __name__ == "__main__":
    train_network()