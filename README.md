 Project Title:
Prediction of Car Selling Price Using Machine Learning

 Introduction:
This project aims to develop a machine learning model that predicts the selling price of used cars based on features such as manufacturing year, present price, kilometers driven, fuel type, transmission type, and ownership. The Random Forest Regressor algorithm was chosen for its robustness and high accuracy when handling mixed data types (numerical and categorical).

Project Steps:
1. Data Loading:
The dataset was loaded using the pandas library from the file car data.csv.

2. Exploratory Data Analysis:
Total records: 301
Total columns: 9
No missing values were found.

3. Data Preprocessing:
The Car_Name column was dropped as it doesn’t impact the prediction.
A new column Car_Age was created by subtracting the Year from 2025.
Categorical variables like Fuel_Type, Transmission, and Selling_type were encoded using LabelEncoder.

4. Data Splitting:
Features (X) and target (y - Selling_Price) were separated.

Data was split into 80% training and 20% testing using train_test_split.

5. Model Training:
A Random Forest Regressor model was trained on the dataset.

The model was fitted using the training set.

6. Model Evaluation:
R² Score: 0.9601
Mean Squared Error (MSE): 0.9191
These metrics reflect a very high model performance, explaining over 96% of the variance in car selling price predictions.

 Visualization :
A scatter plot comparing actual vs. predicted values was created to visualize the model performance.

plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Prices")
plt.grid(True)
plt.show()
result:
The model successfully predicts car selling prices with high accuracy. It can be integrated into web platforms or dealership tools to assist in pricing decisions. The use of RandomForestRegressor provided stability and accurate results.

 Used Libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score








