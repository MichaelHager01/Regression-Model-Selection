# Michael Hager
# UIN: 430006723
# A2: Regression Model Selection in Predicting Wear Rate of Mechanical Component



import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt



# Define a custom RMSE scorer
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



# Read and separate into features (inputs) and response (output)
raw_data = pd.read_csv("train_dataset.csv")
scaler = MinMaxScaler()
scaler.fit(raw_data)
raw_data = scaler.transform(raw_data)
features = raw_data[:, :3]
response = raw_data[:, 3]


# Define the polynomial degrees
degrees = [1, 2, 3, 4, 5, 6]


# Make the RMSE scorer using make_scorer
rmse = make_scorer(rmse_scorer)


# Initialize an empty list to store RMSE for each model
average_rmse = []


# Loop through each degree and perform cross-validation
for degree in degrees:
    
    # Initialize the polynomial features transformer
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(features)
    
    # Initialize the linear regression model
    reg = LinearRegression()
        
    # Perform 5-fold cross-validation
    scores = cross_val_score(reg, poly_features, response, cv=5, scoring=rmse)

    # Compute the average RMSE and append it to the list
    average_rmse.append(np.mean(scores))



# Plot the relationship between polynomial order and average RMSE
plt.plot(degrees, average_rmse, marker='o')
plt.title('Polynomial Degree vs Average RMSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average RMSE')
plt.grid(True)
plt.show()


# Find the index of the lowest cross-validation score
best_degree_index = np.argmin(average_rmse)


# Select the best polynomial degree
best_degree = degrees[best_degree_index]


# Initialize the polynomial features transformer with the best degree
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
features_poly = poly.fit_transform(features)


# Initialize the linear regression model
reg = LinearRegression()


# Retrain the model using the entire training dataset
reg.fit(features_poly, response)


# Save the retrained model to a file
dump(reg, 'best_model.joblib')


