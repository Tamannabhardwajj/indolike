# House Price Prediction - Linear Regression
# Author: Tamanna Bhardwajj

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Load Dataset
train_df = pd.read_csv("data/train.csv")

print("Data Loaded ✅")
print(train_df.head())

# 3. Select Features
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]
target = "SalePrice"

X = train_df[features]
y = train_df[target]

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# 8. Visualization
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# 9. Save the Model
import joblib
joblib.dump(model, "house_price_model.pkl")
print("\nModel Saved as 'house_price_model.pkl'")
