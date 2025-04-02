import xgboost as xgb
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class XGModel:
    def __init__(self, data):
        # Sample dataset (replace with your horse racing data)
        processedData = data.processed_data()
        X = processedData[0]
        y = processedData[1]
    
        # 6️⃣ Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 7️⃣ Train XGBoost Model
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500, learning_rate=0.05)
        model.fit(X_train, y_train)

        # 8️⃣ Predict Horse Positions
        predictions = model.predict(X_test)
        # predictions = np.maximum(predictions.round().astype(int), 1)
        # Output Predictions
        print("Predicted Positions:", predictions)
        print(len(predictions))

         # 9️⃣ Calculate Accuracy (percentage of correct predictions)
        correct_predictions = (predictions == y_test).sum()
        total_predictions = len(y_test)
        accuracy = (correct_predictions / total_predictions) * 100  # Percentage of correct predictions

        # Output Predictions and Accuracy
        print("Predicted Positions:", predictions)
        print(f"Accuracy: {accuracy:.2f}%")

        # Optionally print number of correct predictions
        print(f"Correct Predictions: {correct_predictions}/{total_predictions}")

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R²: {r2:.2f}")