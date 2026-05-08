import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class VisionFeatureExtractor:
    # מחלקה זו אחראית על טעינת מודל ה-CNN (ResNet50)
    # וחילוץ וקטור המאפיינים (Embedding) מהתמונה.
    
    def __init__(self):
        # טעינת מודל מאומן מראש וחיתוך שכבת הסיווג האחרונה
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
        # העברה למצב הסקה (Evaluation) ללא שינוי משקולות
        self.model.eval()
        
        # צינור העיבוד המקדים: שינוי גודל, חיתוך, המרה לטנזור ונרמול צבעים
        self.preprocess = transforms.Compose([
            transforms.Resize(256),                 
            transforms.CenterCrop(224),             
            transforms.ToTensor(),                  
            transforms.Normalize(                   
                mean=[0.485, 0.456, 0.406],         
                std=[0.229, 0.224, 0.225]           
            )
        ])

    def extract_features(self, image_path: str) -> torch.Tensor:
        # קבלת נתיב תמונה והחזרת וקטור מתמטי שטוח באורך 2048
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.preprocess(img)
            img_tensor = img_tensor.unsqueeze(0) # הוספת מימד אצווה
            
            with torch.no_grad(): # כיבוי חישוב גרדיאנטים לחיסכון במשאבים
                features = self.model(img_tensor)
                
            return torch.flatten(features)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None