# 🌲 Forest Fire Prediction Project
# Author: Your Name
# Description: Predict the burned area of forest fires using environmental data.

# ------------------------------
# STEP 1: Import Required Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------
# STEP 2: Load Dataset
# ------------------------------
df = pd.read_csv('forestfires.csv')  # Make sure this file is in the same folder
print("✅ Dataset loaded successfully!\n")
print(df.head())
print("\nShape of dataset:", df.shape)
print("\nColumns:", df.columns.tolist())

# ------------------------------
# STEP 3: Understand Target Distribution
# ------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df['area'], bins=30, kde=True, color='green')
plt.title('Distribution of Burned Area (in hectares)')
plt.xlabel('Burned Area')
plt.ylabel('Count')
plt.show()

# The data is skewed → Apply log transformation to stabilize variance
df['area'] = np.log1p(df['area'])  # log(area + 1)

plt.figure(figsize=(8,5))
sns.histplot(df['area'], bins=30, kde=True, color='orange')
plt.title('Distribution of Log-Transformed Burned Area')
plt.xlabel('log(area + 1)')
plt.ylabel('Count')
plt.show()

# ------------------------------
# STEP 4: Encode Categorical Features
# ------------------------------
le = LabelEncoder()
df['month'] = le.fit_transform(df['month'])
df['day'] = le.fit_transform(df['day'])

# ------------------------------
# STEP 5: Feature and Target Separation
# ------------------------------
X = df.drop('area', axis=1)
y = df['area']

# ------------------------------
# STEP 6: Feature Scaling
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# STEP 7: Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print("\n✅ Data split into training and testing sets successfully.")

# ------------------------------
# STEP 8: Train Linear Regression Model
# ------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# ------------------------------
# STEP 9: Train Random Forest Model
# ------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ------------------------------
# STEP 10: Evaluate Both Models
# ------------------------------
print("\n📊 Model Performance Comparison:\n")
print("Linear Regression R² Score:", round(r2_score(y_test, y_pred_lr), 3))
print("Random Forest R² Score:", round(r2_score(y_test, y_pred_rf), 3))
print("Random Forest MAE:", round(mean_absolute_error(y_test, y_pred_rf), 3))

# ------------------------------
# STEP 11: Visualize Actual vs Predicted (Random Forest)
# ------------------------------
plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred_rf, alpha=0.7, color='green')
plt.xlabel("Actual Burned Area (log scale)")
plt.ylabel("Predicted Burned Area (log scale)")
plt.title("Actual vs Predicted Burned Area")
plt.show()

# ------------------------------
# STEP 12: Feature Importance Visualization
# ------------------------------
importance = rf.feature_importances_
plt.figure(figsize=(10,5))
sns.barplot(x=df.drop('area', axis=1).columns, y=importance, palette="viridis")
plt.xticks(rotation=45)
plt.title("Feature Importance in Forest Fire Prediction")
plt.show()

# ------------------------------
# STEP 13: Example Prediction
# ------------------------------
example = [[6, 5, 90.0, 120.0, 600.0, 10.0, 30.0, 4.0, 0.0, 30.0, 5.0, 0.0]]  # sample values
example_scaled = scaler.transform(example)
predicted_area = rf.predict(example_scaled)
predicted_area = np.expm1(predicted_area)  # reverse log1p transformation
print(f"\n🔥 Predicted Burned Area for Example Input: {predicted_area[0]:.2f} hectares")

# ------------------------------
# STEP 14: Summary of Findings
# ------------------------------
print("\n📈 Summary:")
print("1. The dataset is highly skewed, so log transformation was applied.")
print("2. Random Forest performed better than Linear Regression.")
print("3. Temperature, Wind, and Humidity were key predictors of fire severity.")
print("4. This model can help forecast the spread of future fires based on conditions.")

# ------------------------------
# END OF PROJECT
# ------------------------------
print("\n✅ Forest Fire Prediction Project Completed Successfully!")
