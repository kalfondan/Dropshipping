import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import numpy as np

class DropshippingBertEmbedder:
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing BERT Environment...")
        print(f"Selected Compute Device: {self.device.type.upper()}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)

    def get_sentence_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * attention_mask, 1)
        sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # מחזיר טנזור כדי שנוכל לבצע עליו חישובים מתמטיים ישירות ב-GPU (יותר מהיר)
        return mean_pooled

    # הוספת חישוב הדמיון הסמנטי
    def calculate_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        # חישוב Cosine Similarity ישירות על ה-GPU לפני ההעברה ל-CPU
        cos_sim = F.cosine_similarity(emb1, emb2)
        return cos_sim.item()

    def visualize_embeddings(self, emb_target: torch.Tensor, emb_db: torch.Tensor, text_target: str, text_db: str):
        # המרה ל-Numpy רק לצורך הציור הגרפי
        target_np = emb_target.cpu().numpy()[0]
        db_np = emb_db.cpu().numpy()[0]
        
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.canvas.manager.set_window_title("BERT Semantic Space Visualization")
        
        data_to_plot = np.vstack([target_np[:50], db_np[:50]])
        
        cax = ax.imshow(data_to_plot, aspect='auto', cmap='viridis')
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Israeli Target', 'Global DB'])
        ax.set_xlabel("Vector Dimensions (First 50 of 768)")
        ax.set_title("Visual Comparison of Semantic Vectors (Heatmap)")
        fig.colorbar(cax, orientation='horizontal', pad=0.3, label="Activation Value")
        
        plt.tight_layout()
        plt.show()

# בלוק הרצה ראשי (Main Execution)
if __name__ == "__main__":
    embedder = DropshippingBertEmbedder()
    
    clean_target = "waterproof sports watch men 50 discount"
    clean_db = "waterproof sport watch men 50 discount sale"
    
    print("-" * 50)
    print(f"Generating 768D Vector for Target: '{clean_target}'...")
    target_vector = embedder.get_sentence_embedding(clean_target)
    
    print(f"Generating 768D Vector for DB: '{clean_db}'...")
    db_vector = embedder.get_sentence_embedding(clean_db)
    
    # הוספת חישוב הדמיון הסמנטי
    similarity_score = embedder.calculate_similarity(target_vector, db_vector)
    print("-" * 50)
    print(f"✅ Semantic Similarity Score: {similarity_score * 100:.2f}%")
    print("-" * 50)
    
    print("Opening Python/Matplotlib Visual Diagram...")
    embedder.visualize_embeddings(target_vector, db_vector, clean_target, clean_db)