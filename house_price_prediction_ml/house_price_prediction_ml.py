import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import kagglehub

# Download latest version
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

print("Path to dataset files:", path)

zip_path = r'C:\Users\sakshi\Downloads\archive.zip'
extract_to = r'C:\Users\sakshi\Downloads\archive'

import zipfile

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

csv_path = r'C:\Users\sakshi\Downloads\archive\Housing.csv'
prev_data = pd.read_csv(csv_path)
print(prev_data)

Data = prev_data.drop(['bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea'], axis=1)
print(Data)

# Confirm the structure of the DataFrame
print(Data.columns)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

Data['furnishingstatus'] = le.fit_transform(Data['furnishingstatus'])
print(Data['furnishingstatus'].unique())  # Should show only numeric values

sns.lineplot(x = "area", y ="price", data = Data, hue = "furnishingstatus")
plt.show()

from sklearn.model_selection import train_test_split
X = Data.drop('price', axis=1) 
y = Data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print("Model coefficients:", model.coef_)
print("Model intercept:", model.intercept_)
y_pred = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

import joblib  

# Save the model
joblib.dump(model, "house_price_model.pkl")
print("Model saved successfully!")

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Take user input
area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter number of bedrooms: "))
parking = int(input("Enter number of parking spaces: "))
furnishing_status = input("Enter furnishing status (Furnished, Semi-Furnished, Unfurnished): ")

# Convert furnishing_status into numerical values (same encoding as used during training)
furnishing_map = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
furnishing_status_encoded = furnishing_map.get(furnishing_status, 0)  # Default to 0 if invalid input

# Convert input into a DataFrame with proper column names
user_input = pd.DataFrame([[area, bedrooms, parking, furnishing_status_encoded]], 
                          columns=['area', 'bedrooms', 'parking', 'furnishingstatus'])

# Predict the price
predicted_price = model.predict(user_input)

# Display the prediction
print(f"Estimated House Price: {predicted_price[0]:,.2f}")