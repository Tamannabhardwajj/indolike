# fraud_detection.py

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 2. Load Data
# Load training and test CSVs from data/ folder
df_train = pd.read_csv("data/fraudTrain.csv")
df_test = pd.read_csv("data/fraudTest.csv")

# Combine datasets
df = pd.concat([df_train, df_test], ignore_index=True)

print("\nDataset Shape:", df.shape)
print(df.head())

# 3. Check Class Distribution
print("\nFraud cases in dataset:")
if 'is_fraud' in df.columns:  # Many Kaggle fraud datasets use 'is_fraud' instead of 'Class'
    target_col = 'is_fraud'
else:
    target_col = 'Class'

print(df[target_col].value_counts())

sns.countplot(x=target_col, data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.savefig("outputs/class_distribution.jpg")
plt.show()

# 4. Split Data
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.jpg")
plt.show()

# 8. Save Model
joblib.dump(model, "fraud_detection_model.pkl")
print("\nModel saved as 'fraud_detection_model.pkl'")
