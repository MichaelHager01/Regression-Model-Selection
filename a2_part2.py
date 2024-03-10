# Michael Hager
# UIN: 430006723
# A2: Regression Model Selection in Predicting Wear Rate of Mechanical Component



import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import make_pipeline
from joblib import dump



# Define a custom RMSE scorer
def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))



# Read data from csv and separate into features (inputs) and response (output)
raw_data = pd.read_csv("train_dataset.csv")
features = raw_data[['Rotational speed (RPM)','Load on the bearing (Newton)', 'Hardness of the material (HB)']].to_numpy()
response = raw_data['Wear rate'].to_numpy()


# Define the polynomial degrees and penalty coefficients
degrees = [1, 2, 3, 4, 5, 6]
alphas = [0.001, 0.01, 0.1, 1]


# Make the RMSE scorer using make_scorer
rmse = make_scorer(rmse_scorer)


# Initialize an empty dictionary to store average RMSE for each combination
average_rmse = {}


# Loop through each combination of polynomial order and penalty coefficient
for degree in degrees:
    for alpha in alphas:
        
        # Generate polynomial features based on the RPM, Load, and Hardness
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # Initialize Ridge-regularized linear regression model with pipeline
        model = make_pipeline(MinMaxScaler(), poly, Ridge(alpha=alpha))
        
        # Perform 5-fold cross-validation and compute RMSE
        scores = cross_val_score(model, features, response, cv=5, scoring=rmse)
        
        # Compute the average RMSE
        average_rmse[(degree, alpha)] = np.mean(scores)



# Find the combination with the lowest average RMSE
best_degree, best_alpha = min(average_rmse, key=average_rmse.get)


# Retrain the Ridge-regularized model using the entire training dataset with the best combination
poly = PolynomialFeatures(degree=best_degree, include_bias=False)
model = make_pipeline(MinMaxScaler(), poly, Ridge(alpha=best_alpha))
model.fit(features, response)


# Save the retrained model to a file
dump(model, 'best_ridge_model.joblib')


