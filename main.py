import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
df = pd.read_csv(
    "gams_preprocessed_hourly.csv",
    parse_dates=["ts"],
    index_col=["ts"],
)
pollutants=['co2', 'humidity', 'pm10', 'pm25', 'temperature', 'voc']

def lstm_regression(X_train, y_train, X_test, y_test,feature):
    # Reshape training and testing data to be 3D for LSTM
    X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define model architecture for LSTM
    model = Sequential()
    model.add(LSTM(1024, activation='relu', input_shape=(1, X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=1024, verbose=1,use_multiprocessing=True,workers=-1)

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = model.predict(X_train).flatten()
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = model.predict(X_test).flatten()
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)

     # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'LSTM Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_Lstm.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"LSTM Training MSE: {mse_train:.9f}")
    print(f"LSTM Training MAPE: {mape_train:.9f}%")
    print(f"LSTM Testing MSE: {mse_test:.9f}")
    print(f"LSTM Testing MAPE: {mape_test:.9f}%")

    return (mse_train, mse_test, mape_train, mape_test)

def lasso_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for Lasso Regression
    model = Lasso()
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'fit_intercept': [True, False]
    }

    # Perform grid search with cross-validation
    print("Performing grid search for Lasso Regression...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=3)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)

    # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Lasso Regression Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_lasso.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Lasso Regression Training MSE: {mse_train:.9f}")
    print(f"Lasso Regression Training MAPE: {mape_train:.9f}%")
    print(f"Lasso Regression Testing MSE: {mse_test:.9f}")
    print(f"Lasso Regression Testing MAPE: {mape_test:.9f}%")
    
    return (mse_train, mse_test, mape_train, mape_test)

def decision_tree_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for Decision Tree
    model = DecisionTreeRegressor()
    param_grid = {
        'max_depth': [3, 5, 7,13,17 ,None],
        'min_samples_split': [2, 5, 10,20],
        'min_samples_leaf': [1, 2, 4,8,12,14,16]
    }

    # Perform grid search with cross-validation
    print("Performing grid search for Decision Tree...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
     # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Decisionn Tree vs Actual Values (Hourly)')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_DTR.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Decisionn Tree Training MSE: {mse_train:.9f}")
    print(f"Decisionn Tree Training MAPE: {mape_train:.9f}%")
    print(f"Decisionn Tree Testing MSE: {mse_test:.9f}")
    print(f"Decisionn Tree Testing MAPE: {mape_test:.9f}%")
    return (mse_train, mse_test, mape_train, mape_test)

def svr_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for Support Vector Regressor (SVR)
    model = SVR()
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1]
    }

    # Perform grid search with cross-validation
    print("Performing grid search for SVR...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
     # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Support Vector Regressor Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_SVR.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Support Vector Regressor Training MSE: {mse_train:.9f}")
    print(f"Support Vector Regressor Training MAPE: {mape_train:.9f}%")
    print(f"Support Vector Regressor Testing MSE: {mse_test:.9f}")
    print(f"Support Vector Regressor Testing MAPE: {mape_test:.9f}%")
    return (mse_train, mse_test, mape_train, mape_test)

def knn_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for KNN
    model = KNeighborsRegressor()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11,15,21],
        'weights': ['uniform', 'distance'],
        'p': [1, 2,5]
    }

    # Perform grid search with cross-validation
    print("Performing grid search for KNN...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
    # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'KNN Regressor Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_KNN.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"KNN Regressor Training MSE: {mse_train:.9f}")
    print(f"KNN Regressor Training MAPE: {mape_train:.9f}%")
    print(f"KNN Regressor Testing MSE: {mse_test:.9f}")
    print(f"KNN Regressor Testing MAPE: {mape_test:.9f}%")
    return (mse_train, mse_test, mape_train, mape_test)


def random_forest_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for Random Forest
    model = RandomForestRegressor()
    param_grid = {
        'n_estimators': [150,300,450,600,750,1000],
        'min_samples_leaf': [1, 2, 4,8,12,14,16],
        'max_depth': [ 40, 60,100,200]
        
    }
    
    # Perform grid search with cross-validation
    print("Performing grid search for Random Forest...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)
     # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'Random Forest Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_randomforest.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Random Forest Training MSE: {mse_train:.9f}")
    print(f"Random Forest Training MAPE: {mape_train:.9f}%")
    print(f"Random Forest Testing MSE: {mse_test:.9f}")
    print(f"Random Forest Testing MAPE: {mape_test:.9f}%")
    return (mse_train, mse_test, mape_train, mape_test)


def xgboost_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for XGBoost
    model = xgb.XGBRegressor()
    param_grid = {
        'max_depth': [7,9, 11,13],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [150,300,450,700],
    
        
    }

    # Perform grid search with cross-validation
    print("Performing grid search for XGBoost...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred)

    # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'XGBoost Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_XGBoost.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"LightGBM Training MSE: {mse_train:.9f}")
    print(f"LightGBM Training MAPE: {mape_train:.9f}%")
    print(f"LightGBM Testing MSE: {mse_test:.9f}")
    print(f"LightGBM Testing MAPE: {mape_test:.9f}%")
    return (mse_train, mse_test, mape_train, mape_test)

def lightgbm_regression(X_train, y_train, X_test, y_test,feature):
    # Define model and parameter grid for LightGBM
    model = lgb.LGBMRegressor()
    param_grid = {
        'num_leaves': [70,90,120,150,300,500,700],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [150,300,450,600,800,1000],
    }

    # Perform grid search with cross-validation
    print("Performing grid search for LightGBM...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                               cv=5, n_jobs=-1, scoring='neg_mean_squared_error',verbose=3)
    grid_search.fit(X_train, y_train)
    print(f"Best hyperparameters found: {grid_search.best_params_}")

    # Make predictions on train and test data and calculate MSE and MAPE
    y_train_pred = grid_search.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    y_pred = grid_search.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_pred) 

    # Plot predictions vs actual values on hourly timeline
    fig = plt.figure(figsize=(8,6))
    plt.plot(y_test.index, y_test.values)
    plt.plot(y_test.index, y_pred)
    plt.legend(['Actual', 'Predicted'])
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(f'LightGBM Predictions vs Actual Values (Hourly) for {feature}')
    plt.rcParams['figure.dpi'] = 300
    fig.savefig(f"./Plots/{feature}_lightgbm.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"LightGBM Training MSE: {mse_train:.9f}")
    print(f"LightGBM Training MAPE: {mape_train:.9f}%")
    print(f"LightGBM Testing MSE: {mse_test:.9f}")
    print(f"LightGBM Testing MAPE: {mape_test:.9f}%")

    return (mse_train, mse_test, mape_train, mape_test)

def return_train_data(data,feature):
    X_train = data[data.index <= "2017-03-01"]
    X_test = data[data.index > "2017-03-01"]
    y_train = X_train[feature].copy()
    y_test = X_test[feature].copy()

    # remove raw time series from predictors set
    X_train = X_train.drop(feature, axis=1)
    X_test = X_test.drop(feature, axis=1)
    return (X_train, y_train, X_test, y_test,feature)


import pandas as pd
import json

# Define a list of regression models to evaluate
regression_models = [
    "LightGBM",
    "XGBoost",
    "RandomForest",
    "KNN",
    "SVR",
    "DecisionTree",
    "Lasso",
    "LSTM"
]

# Create an empty dictionary to hold the results
results_dict = {}
for model in regression_models:
    results_dict[model] = {}

# Loop over all features
for val in pollutants:
    # Get training and testing data for target feature
    data_values = return_train_data(df, val)
    X_train, y_train, X_test, y_test, feature_name = data_values
    
    # Evaluate each regression model on the selected feature
    mse_train, mse_test, mape_train, mape_test = lightgbm_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["LightGBM"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}

    mse_train, mse_test, mape_train, mape_test = xgboost_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["XGBoost"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}

    mse_train, mse_test, mape_train, mape_test = random_forest_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["RandomForest"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}
    
    mse_train, mse_test, mape_train, mape_test = knn_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["KNN"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}
    
    mse_train, mse_test, mape_train, mape_test = svr_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["SVR"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}
    
    mse_train, mse_test, mape_train, mape_test = decision_tree_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["DecisionTree"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}
    
    mse_train, mse_test, mape_train, mape_test = lasso_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["Lasso"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}

    mse_train, mse_test, mape_train, mape_test = lstm_regression(X_train, y_train, X_test, y_test, feature_name)
    results_dict["LSTM"][feature_name] = {"mse_train": mse_train, "mse_test": mse_test, "mape_train": mape_train, "mape_test": mape_test}

# Write results dictionary to JSON file
with open('results.json', 'w') as f:
    json.dump(results_dict, f)

print("Results written to results.json")
