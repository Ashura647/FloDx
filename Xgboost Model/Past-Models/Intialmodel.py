import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

excel_file = r'KHK growth curves_LB.xlsx'
xls = pd.ExcelFile(excel_file)

def rename_columns(dataframe):
    new_columns = ['time (h)'] + [f'OD{i+2}' for i in range(len(dataframe.columns) - 1)]
    dataframe.columns = new_columns
    return dataframe

def create_lag_features(data, target_cols, lag=1):
    lagged_data = data.copy()
    for col in target_cols:
        for i in range(1, lag + 1):
            lagged_data[f'lag_{i}_{col}'] = lagged_data[col].shift(i)
    lagged_data.dropna(inplace=True)  
    return lagged_data

def train_evaluate_model(X_train, y_train, X_test, y_test):
    if X_train.empty or y_train.empty:
        print("Error: Training data is empty. Model cannot be trained.")
        return None, None

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }

    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    print(f'Best parameters: {grid_search.best_params_}')
    print('Training set evaluation:')
    print(f'Mean Squared Error: {mse_train:.2f}')
    print(f'R^2 Score: {r2_train:.2f}')
    print('\nTesting set evaluation:')
    print(f'Mean Squared Error: {mse_test:.2f}')
    print(f'R^2 Score: {r2_test:.2f}')

    return best_model, y_pred_test

def update_initial_data(initial_data, new_data):
    if initial_data.empty:
        return new_data
    return pd.concat([initial_data, new_data], ignore_index=True)

def plot_od_predictions(df, y_pred, sheet_name):
    plt.figure(figsize=(12, 8))
    plt.plot(df['time (h)'], df['OD2'], label='Original Data (OD2)', linestyle='-', color='blue', alpha=0.8)
    plt.plot(df['time (h)'][len(df['time (h)']) - len(y_pred):], y_pred, label='Predicted', linestyle='--')
    plt.xlabel('Time (hours)')
    plt.ylabel('OD')
    plt.title(f'OD Predictions vs Original Data for {sheet_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

def print_data_head(df):
    print(df.head())

sheet_name_train_initial = 'No.1 (n=12)'
df_train_initial = pd.read_excel(xls, sheet_name=sheet_name_train_initial)
sheet_name_test_initial = 'No.12 (n=12)'
df_test_initial = pd.read_excel(xls, sheet_name=sheet_name_test_initial)

print(f'Data from sheet "{sheet_name_train_initial}":')
print_data_head(df_train_initial)
print(f'\nData from sheet "{sheet_name_test_initial}":')
print_data_head(df_test_initial)

df_train_initial = rename_columns(df_train_initial)
df_test_initial = rename_columns(df_test_initial)

od_columns = [col for col in df_train_initial.columns if col.startswith('OD')]

X_train_initial = create_lag_features(df_train_initial, od_columns).drop(columns=['time (h)'] + od_columns)
y_train_initial = create_lag_features(df_train_initial, od_columns)[od_columns]
X_test_initial = create_lag_features(df_test_initial, od_columns).drop(columns=['time (h)'] + od_columns)
y_test_initial = create_lag_features(df_test_initial, od_columns)[od_columns]

y_pred_aggregated_initial = np.zeros_like(y_test_initial.values)

for col_idx, col in enumerate(od_columns):
    print(f'\nTraining model for column: {col}')
    X_train_col = X_train_initial
    y_train_col = y_train_initial[col]
    X_test_col = X_test_initial
    y_test_col = y_test_initial[col]
    
    model, y_pred_col = train_evaluate_model(X_train_col, y_train_col, X_test_col, y_test_col)
    y_pred_aggregated_initial[:, col_idx] = y_pred_col

y_pred_aggregated_initial = np.mean(y_pred_aggregated_initial, axis=1)

plot_od_predictions(df_test_initial, y_pred_aggregated_initial, sheet_name_test_initial)

update_model = input("Do you want to update the model with new training data? (yes/no): ").strip().lower()

if update_model == 'yes':
    new_sheet_name_train = sheet_name_test_initial
    df_train_new = pd.read_excel(xls, sheet_name=new_sheet_name_train)
    
    print(f'\nUpdating model with new training data from sheet "{new_sheet_name_train}":')
    print_data_head(df_train_new)
    
    df_train_updated = update_initial_data(df_train_initial, rename_columns(df_train_new))
    
    X_train_updated = create_lag_features(df_train_updated, od_columns).drop(columns=['time (h)'] + od_columns)
    y_train_updated = create_lag_features(df_train_updated, od_columns)[od_columns]
    
    y_pred_aggregated_updated = np.zeros_like(y_test_initial.values)

    for col_idx, col in enumerate(od_columns):
        print(f'\nTraining model for column: {col}')
        X_train_col = X_train_updated
        y_train_col = y_train_updated[col]
        
        model, y_pred_col = train_evaluate_model(X_train_col, y_train_col, X_test_initial, y_test_initial[col])
        y_pred_aggregated_updated[:, col_idx] = y_pred_col

    y_pred_aggregated_updated = np.mean(y_pred_aggregated_updated, axis=1)

    df_train_initial = df_train_updated

    plot_od_predictions(df_test_initial, y_pred_aggregated_updated, sheet_name_test_initial)

elif update_model == 'no':
    print("\nModel remains unchanged.")
else:
    print("\nInvalid input. Model remains unchanged.")
