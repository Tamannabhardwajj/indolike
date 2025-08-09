# sms_spam_detection.py

# 1. Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# 2. Load Data
# Make sure spam.csv is in the 'data' folder
data_path = os.path.join("data", "spam.csv")
df = pd.read_csv(data_path, encoding='latin-1')

# Keep only the needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("\nDataset Shape:", df.shape)
print(df.head())

# 3. Check Class Distribution
print("\nClass distribution:")
print(df['label'].value_counts())

sns.countplot(x='label', data=df)
plt.title("Ham vs Spam Messages")
plt.savefig("outputs/sms_class_distribution.jpg")
plt.show()

# 4. Split Data
X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Text Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Predictions
y_pred = model.predict(X_test_vec)

# 8. Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.savefig("outputs/sms_confusion_matrix.jpg")
plt.show()

# 9. Save Model and Vectorizer
joblib.dump(model, "sms_spam_model.pkl")
joblib.dump(vectorizer, "sms_vectorizer.pkl")
print("\nModel saved as 'sms_spam_model.pkl'")
print("Vectorizer saved as 'sms_vectorizer.pkl'")
