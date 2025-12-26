import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Load dataset
# Adjusted to the workspace file name
df = pd.read_csv('Housing.csv')  # or your dataset name

# Basic exploration
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDataset info:")
print(df.info())

# Handle missing values if any
df = df.dropna()  # or use df.fillna()

# Separate features and target (assuming 'price' is your target column)
if 'price' not in df.columns:
    raise ValueError("Expected target column 'price' not found in dataset. Update model.py to use the correct target column.")

X = df.drop('price', axis=1)  # Change 'price' to your target column name
y = df['price']

# Handle categorical variables if present
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor
print("\nTraining model...")
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    min_samples_split=5
)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
print("\n=== Model Performance ===")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error: ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"Root Mean Squared Error: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Save model, scaler, and feature names
print("\nSaving model...")
with open('house_price_model.pkl', 'wb') as f:
    pickle.dump(model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("Model, scaler, and feature names saved successfully!")
