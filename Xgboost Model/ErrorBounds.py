import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d

# File and folder setup
excel_file = r'synthetic_s_aureus_growth_curves.xlsx'
output_folder = 'S.AUREUS-LB-FAKE'
os.makedirs(output_folder, exist_ok=True)

xls = pd.ExcelFile(excel_file)

def rename_columns(dataframe):
    "Rename columns of the dataframe."
    new_columns = ['time (h)'] + [f'OD{i+2}' for i in range(len(dataframe.columns) - 1)]
    dataframe.columns = new_columns
    return dataframe

def train_evaluate_model(X_train, y_train, X_test, y_test):
    "Train the XGBoost model and evaluate it."
    param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'alpha': [0, 0.1, 1],
        'lambda': [1, 1.5, 2]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred_test = best_model.predict(X_test)

    return best_model, y_pred_test

def update_initial_data(initial_data, new_data):
    "Update initial data with new data."
    return pd.concat([initial_data, new_data], ignore_index=True)

def save_predictions_to_csv(time, y_pred_upper, y_pred_lower, output_csv):
    "Save error bounds to a CSV file."
    plot_data = pd.DataFrame({
        'time (h)': time,
        'Upper Bound': y_pred_upper,
        'Lower Bound': y_pred_lower
    })

    plot_data.to_csv(output_csv, index=False)
    print(f'Error bounds saved to {output_csv}')

def plot_error_bounds(time, y_pred_upper, y_pred_lower, sheet_name):
    "Plot the error bounds without actual or predicted values."
    plt.figure(figsize=(10, 6))
    
    plt.fill_between(time, y_pred_lower, y_pred_upper, color='gray', alpha=0.5, label='Error Bounds')
    plt.title(f'Error Bounds for {sheet_name}')
    plt.xlabel('Time (h)')
    plt.ylabel('OD Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Load all sheets
sheet_names = xls.sheet_names

# Initialize an empty DataFrame for the initial data
df_train_initial = pd.DataFrame()

# Iteratively train and update the model
for i in range(len(sheet_names)):
    df_train = pd.read_excel(xls, sheet_name=sheet_names[i])
    df_train = rename_columns(df_train)

    if i == len(sheet_names) - 1:
        df_test = df_train.copy()  # Use the last sheet for both training and testing
    else:
        df_test = pd.read_excel(xls, sheet_name=sheet_names[i + 1])
        df_test = rename_columns(df_test)

    od_columns = [col for col in df_train.columns if col.startswith('OD')]

    X_train = df_train.drop(columns=['time (h)'])
    y_train = df_train[od_columns]
    X_test = df_test.drop(columns=['time (h)'])
    y_test = df_test[od_columns]

    y_pred_aggregated = np.zeros_like(y_test.values)

    for col_idx, col in enumerate(od_columns):
        print(f'\nTraining model for column: {col} using sheet "{sheet_names[i]}" as training data.')
        X_train_col = X_train
        y_train_col = y_train[col]
        X_test_col = X_test
        y_test_col = y_test[col]

        model, y_pred_col = train_evaluate_model(X_train_col, y_train_col, X_test_col, y_test_col)
        y_pred_aggregated[:, col_idx] = y_pred_col

    y_pred_aggregated = np.mean(y_pred_aggregated, axis=1)

    # Calculate residuals and residuals range
    residuals = df_test[od_columns] - y_pred_aggregated[:, np.newaxis]
    residuals_max = residuals.max(axis=1)
    residuals_min = residuals.min(axis=1)
    residuals_range = residuals_max - residuals_min

    # Percentile-based residuals range
    error_percentile = 90
    residuals_range_percentile = np.percentile(residuals_range, error_percentile)

    # Smooth the percentile residuals range using Gaussian filter
    smooth_window = 100
    smoothed_residuals_range = gaussian_filter1d(residuals_range, sigma=smooth_window)

    # Define time-dependent scaling factor
    time_dependent_factor = np.sqrt(df_test['time (h)'] / df_test['time (h)'].max())
    error_scaling_factor = 1.5
    scaled_error = error_scaling_factor * smoothed_residuals_range * time_dependent_factor

    # Calculate error bounds
    y_pred_upper = y_pred_aggregated + scaled_error
    y_pred_lower = y_pred_aggregated - scaled_error

    # Clip lower bounds to ensure they don't go below zero
    y_pred_lower = np.maximum(y_pred_lower, 0)

    # Extract time values for plotting
    time = df_test['time (h)'][len(df_test['time (h)']) - len(y_pred_aggregated):]

    output_csv = os.path.join(output_folder, f'{sheet_names[i]}_error_bounds.csv')
    save_predictions_to_csv(time, y_pred_upper, y_pred_lower, output_csv)

    # Update the training data for the next iteration
    df_train_initial = update_initial_data(df_train_initial, df_train)

    print(f'Model updated with sheet "{sheet_names[i]}" as training data.\n')

print("Final model training complete.")
