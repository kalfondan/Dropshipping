import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. קריאת הקובץ המעודכן (אחרי שהוספת לו את עמודת האמת)
try:
    df = pd.read_csv('final_200_results.csv')
except FileNotFoundError:
    print("שגיאה: הקובץ לא נמצא.")
    exit()

# מוודאים שהעמודה שיצרת באמת קיימת
if 'Actual_Is_Dropship' not in df.columns:
    print("שגיאה: חסרה עמודת 'Actual_Is_Dropship'. הוסף אותה באקסל לפני ההרצה.")
    exit()

# 2. סינון שגיאות (ERROR) - אנחנו רוצים לבדוק רק מקרים שהמודל נתן להם חיזוי
df_clean = df[df['Is_Dropship'].isin(['Yes', 'No'])].copy()

# 3. חילוץ התשובות של המודל מול התשובות האמיתיות
y_true = df_clean['Actual_Is_Dropship']  # האמת (Ground Truth)
y_pred = df_clean['Is_Dropship']         # החיזוי של המודל (Prediction)

# 4. חישוב מטריצת הבלבול בעזרת scikit-learn
cm = confusion_matrix(y_true, y_pred, labels=['Yes', 'No'])

# 5. ציור ויזואלי של המטריצה
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, 
    display_labels=['Dropshipping (Yes)', 'Legit (No)']
)

# הגדרת עיצוב הגרף
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)

plt.title('Confusion Matrix - 200 Real World Products', fontsize=16, pad=20)
plt.xlabel('Predicted Label (What the AI said)', fontsize=12)
plt.ylabel('True Label (What it actually is)', fontsize=12)

# הצגת הגרף על המסך
plt.show()