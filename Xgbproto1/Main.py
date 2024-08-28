import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Gompertz function
def gompertz(t, y0, C, mu, lambda_, beta):
    return y0 + C * np.exp(-np.exp(beta * (lambda_ - t)))

# Parameters
y0 = 0.5    # Initial log count or absorbance
ymax = 2.5  # Final log count or absorbance
C_default = ymax - y0  # Increase in log count or absorbance from y0 to ymax
mu = 0.1    # Maximum growth rate
lambda_ = 1.5  # Lag time
beta = 1.0  # Model coefficient

# Time range for initial data and prediction
t_initial = np.linspace(0, 10, 100)
t_prediction = np.linspace(0, 12, 100)

# Generate y values using the Gompertz function for initial data
y_initial = gompertz(t_initial, y0, C_default, mu, lambda_, beta)

# Create a DataFrame for initial data
data = pd.DataFrame({'time': t_initial, 'log_count': y_initial})

# Plotting the generated data
plt.figure(figsize=(10, 6))
plt.plot(t_initial, y_initial, label='Gompertz Curve (Initial Data)')
plt.xlabel('Time (hours)')
plt.ylabel('Log Count or Absorbance')
plt.title('Gompertz Growth Curve for E. coli')
plt.legend()
plt.grid(True)

# Train an XGBoost model on initial data
X_train = data[['time']]
y_train = data['log_count']

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Make predictions for the extended time range
y_pred = model.predict(pd.DataFrame({'time': t_prediction}))

# Plot the predicted growth
plt.scatter(t_prediction, y_pred, color='red', label='Predicted growth (XGBoost)')
plt.legend()

# Calculate R-squared value for a third-order polynomial fit to predictions
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(t_prediction.reshape(-1, 1))
poly_model = LinearRegression()
poly_model.fit(X_poly, y_pred)
y_pred_poly = poly_model.predict(X_poly)

r2 = r2_score(y_pred, y_pred_poly)
print(f"R-squared value (Polynomial 3rd order): {r2:.4f}")

# Plot the polynomial fit
plt.plot(t_prediction, y_pred_poly, color='green', linestyle='--', label=f'Polynomial Fit (R²={r2:.2f})')
plt.legend()

plt.show()

# Function to load new data from Excel file and update the model
def load_and_update_model(file_path, sheet_names, model, data):
    all_data = []

    for sheet_name in sheet_names:
        data_new = pd.read_excel(file_path, sheet_name=sheet_name)
        data_new.columns = data_new.columns.str.strip()
        data_new['Time'] = pd.to_datetime(data_new['time (h)'], unit='h')
        data_new.set_index('Time', inplace=True)
        all_data.append(data_new)

    new_data = pd.concat(all_data, axis=0)
    updated_data = pd.concat([data, new_data])

    # Features and target
    X_updated = updated_data[['time']]
    y_updated = updated_data['log_count']

    # Retrain the model with updated data
    model.fit(X_updated, y_updated)

    return model, updated_data

# Function to update Gompertz parameter C
def update_gompertz_C(model, t_range):
    y_pred_new = model.predict(pd.DataFrame({'time': t_range}))
    C_updated = max(y_pred_new) - y0
    return C_updated

# Function to let user choose between default and updated parameters
def choose_parameters(use_default, C_default, C_updated):
    if use_default:
        return C_default
    else:
        return C_updated

# Simulate loading new data from Excel file and updating the model
file_path = r'5918608\KHK growth curves_LB.xlsx'
sheet_names = ['No.{} (n=12)'.format(i) for i in range(1, 37)]  # Sheets No.1 to No.36

model, data = load_and_update_model(file_path, sheet_names, model, data)

# Update the Gompertz parameter C
C_updated = update_gompertz_C(model, t_prediction)
print(f"Updated C value: {C_updated}")

# User chooses whether to use default or updated parameters
use_default = False  # Change this to True to use default parameters

# Choose Gompertz parameter C
C_final = choose_parameters(use_default, C_default, C_updated)
print(f"Final C value: {C_final}")

# Plot the final predicted growth with the chosen Gompertz parameter C
y_final = gompertz(t_prediction, y0, C_final, mu, lambda_, beta)

plt.figure(figsize=(10, 6))
plt.plot(t_prediction, y_final, label='Final Gompertz Curve')
plt.xlabel('Time (hours)')
plt.ylabel('Log Count or Absorbance')
plt.title('Final Gompertz Growth Curve with Chosen Parameters')
plt.legend()
plt.grid(True)
plt.show()
