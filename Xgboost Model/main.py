import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter1d
import time

# Set the Matplotlib backend to 'Agg' for non-interactive plotting
import matplotlib
matplotlib.use('Agg')

# File and folder setup
excel_file = r'synthetic_s_aureus_growth_curves.xlsx'
output_folder = 'S.AUREUS-LB-FAKE'
plot_folder = 'Synthetic-SA-Plots'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

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

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', verbosity=1)
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred_test = best_model.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f'Best parameters: {grid_search.best_params_}')
    print('\nTesting set evaluation:')
    print(f'Mean Squared Error: {mse_test:.2f}')
    print(f'R^2 Score: {r2_test:.2f}')

    return best_model, y_pred_test

def update_initial_data(initial_data, new_data):
    "Update initial data with new data."
    return pd.concat([initial_data, new_data], ignore_index=True)

def save_predictions_to_csv(time, y_pred, y_pred_upper, y_pred_lower, output_csv):
    "Save OD predictions and error bounds to a CSV file."
    plot_data = pd.DataFrame({
        'time (h)': time,
        'Predicted': y_pred,
        'Upper Bound': y_pred_upper,
        'Lower Bound': y_pred_lower
    })

    plot_data.to_csv(output_csv, index=False)
    print(f'Predictions and error bounds saved to {output_csv}')

def plot_predictions(time, y_test, y_pred, y_pred_upper, y_pred_lower, od_columns, sheet_name):
    "Plot the actual vs predicted data with error bounds."
    plt.figure(figsize=(10, 6))
    
    for col_idx, col in enumerate(od_columns):
        plt.plot(time, y_test[col], label=f'Actual {col}', marker='o', linestyle='--')
        plt.plot(time, y_pred, label=f'Predicted {col}', marker='x')

    plt.fill_between(time, y_pred_lower, y_pred_upper, color='gray', alpha=0.2, label='Error Bounds')
    plt.title(f'OD Predictions for {sheet_name}')
    plt.xlabel('Time (h)')
    plt.ylabel('OD Values')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_folder, f'{sheet_name}_plot.png'))
    plt.close()

# Load all sheets
sheet_names = xls.sheet_names

# Initialize an empty DataFrame for the initial data
df_train_initial = pd.DataFrame()

# Record the start time
start_time = time.time()

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

    # Calculate initial actual values and predictions for each OD column
    initial_actuals = df_test[od_columns].iloc[0].values  # Initial actual values for each column
    initial_predictions = y_pred_aggregated[0]  # Initial predictions are already aggregated

    # Calculate residuals for each column and then compute the range
    residuals = (df_test[od_columns] - initial_actuals) - (y_pred_aggregated[:, np.newaxis] - initial_predictions)
    residuals_max = residuals.max(axis=1)
    residuals_min = residuals.min(axis=1)
    residuals_range = residuals_max - residuals_min

    # Use a percentile-based error range
    error_percentile = 90  # Use the 90th percentile of residuals range
    residuals_range_percentile = np.percentile(residuals_range, error_percentile)

    # Smooth the percentile residuals range using Gaussian filter to reduce jaggedness
    smooth_window = 100  # Adjust window size for more or less smoothing
    smoothed_residuals_range = gaussian_filter1d(residuals_range, sigma=smooth_window)

    # Define a non-linear time-dependent error scaling factor
    time_dependent_factor = np.sqrt(df_test['time (h)'] / df_test['time (h)'].max())  # Non-linear increase
    error_scaling_factor = 1.5  # Adjust base error scaling factor
    scaled_error = error_scaling_factor * smoothed_residuals_range * time_dependent_factor

    # Ensure that error bounds encompass all possible residuals
    y_pred_upper = y_pred_aggregated + scaled_error
    y_pred_lower = y_pred_aggregated - scaled_error

    # Clip lower bounds to ensure they don't go below zero
    y_pred_lower = np.maximum(y_pred_lower, 0)

    time = df_test['time (h)'][len(df_test['time (h)']) - len(y_pred_aggregated):]

    output_csv = os.path.join(output_folder, f'{sheet_names[i]}_predictions.csv')
    save_predictions_to_csv(time, y_pred_aggregated, y_pred_upper, y_pred_lower, output_csv)

    # Save the plot to a file
    plot_predictions(time, y_test, y_pred_aggregated, y_pred_upper, y_pred_lower, od_columns, sheet_names[i])

    # Update the training data for the next iteration
    df_train_initial = update_initial_data(df_train_initial, df_train)

    print(f'Model updated with sheet "{sheet_names[i]}" as training data.\n')

# Record the end time and calculate the total training time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Final model training complete in {elapsed_time / 60:.2f} minutes.")
