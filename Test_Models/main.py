import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
import time

# Function to load and process the data from a specific sheet
def load_and_process_data(file_path, sheet_name):
    print(f"Loading data from sheet: {sheet_name}")
    start_time = time.time()
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    columns = ['time (h)'] + [f'OD{i}' for i in range(1, len(df.columns))]
    df.columns = columns
    end_time = time.time()
    print(f"Data loaded in {end_time - start_time:.2f} seconds.")
    return df

# Richards' model
def richards(x, A, v, k, T):
    return A * np.power(1 + (v * np.exp(k * (T - x))) / (1 + np.exp(k * (T - x))), -1 / v)

# Improved initial guess estimation
def estimate_initial_params(x, y):
    print("Estimating initial parameters...")
    start_time = time.time()
    A = np.max(y)
    k = (np.log(9) / (np.percentile(x, 75) - np.percentile(x, 25)))  # Growth estimation
    T = x[np.argmax(np.diff(y))] # Inflection point at max slope
    v = 1  # Initial guess for v
    end_time = time.time()
    print(f"Initial parameters estimated in {end_time - start_time:.2f} seconds.")
    print(T, k, A, v)
    return A, v, k, T

# Function to fit and plot the Richards' model
def fit_and_plot_richards(test_df, od_column, training_df, combined_df=None):
    print(f"Fitting and plotting Richards' model for OD column: {od_column}")
    start_time = time.time()

    x_test = test_df['time (h)'].values
    y_test = test_df[od_column].values
    x_train = training_df['time (h)'].values
    y_train = training_df[od_column].values

    if combined_df is not None:
        x_combined = combined_df['time (h)'].values
        y_combined = combined_df[od_column].values
    else:
        x_combined = None
        y_combined = None

    plt.figure(figsize=(12, 8))
    plt.scatter(x_test, y_test, label='Test Data', color='black')

    try:
        # Estimate initial parameters
        print("Estimating initial parameters...")
        initial_guess_start_time = time.time()
        initial_guess = estimate_initial_params(x_train, y_train)
        initial_guess_end_time = time.time()
        print(f"Initial parameter estimation took {initial_guess_end_time - initial_guess_start_time:.2f} seconds.")

        print("Fitting Richards' model to training data...")
        fit_start_time = time.time()
        popt, pcov = curve_fit(richards, x_train, y_train, p0=initial_guess, maxfev=10000, method='trf')
        fit_end_time = time.time()
        print(f"Richards' model fitting took {fit_end_time - fit_start_time:.2f} seconds.")
        
        if combined_df is not None:
            try:
                print("Fitting Richards' model to combined data...")
                fit_combined_start_time = time.time()
                popt_combined, _ = curve_fit(richards, x_combined, y_combined, p0=popt, maxfev=10000, method='trf')
                fit_combined_end_time = time.time()
                print(f"Richards' model fitting with combined data took {fit_combined_end_time - fit_combined_start_time:.2f} seconds.")
            except Exception as e:
                print(f"Error fitting the Richards' model with combined data: {e}")
                popt_combined = popt
        else:
            popt_combined = popt

        x_fit = np.linspace(min(x_test), max(x_test), 1000)
        y_fit = richards(x_fit, *popt_combined)

        # Calculate error bounds
        print("Calculating error bounds...")
        bounds_start_time = time.time()
        alpha = 0.05
        z = norm.ppf(1 - alpha / 2)
        inflation_factor = 6.6666  # Increase this to inflate the error bounds
        perr = inflation_factor * z * np.sqrt(np.diag(pcov))
        y_fit_upper = richards(x_fit, *(popt_combined + perr))
        y_fit_lower = richards(x_fit, *(popt_combined - perr))
        bounds_end_time = time.time()
        print(f"Error bounds calculation took {bounds_end_time - bounds_start_time:.2f} seconds.")

        plt.plot(x_fit, y_fit, color='orange', label='Richards\' fit')
        plt.fill_between(x_fit, y_fit_lower, y_fit_upper, color='orange', alpha=0.4, label='Increased Error Area')

    except Exception as e:
        print(f"Error fitting the Richards' model: {e}")

    plt.title(f'Richards\' Fit for {od_column}')
    plt.xlabel('Time (h)')
    plt.ylabel(od_column)
    plt.legend()
    plt.show()

    end_time = time.time()
    print(f"Fitting and plotting completed in {end_time - start_time:.2f} seconds.")

# Main function to execute the steps
def main():
    file_path = r'growthdashbtr.xlsx'
    xls = pd.ExcelFile(file_path)
    
    print("Available sheets:")
    for i, sheet in enumerate(xls.sheet_names):
        print(f"{i + 1}. {sheet}")
    
    try:
        test_sheet_number = int(input("Enter the number corresponding to the test sheet: ")) - 1
        train_sheet_number = int(input("Enter the number corresponding to the training sheet: ")) - 1
        test_sheet = xls.sheet_names[test_sheet_number]
        train_sheet = xls.sheet_names[train_sheet_number]
    except (ValueError, IndexError):
        print("Invalid selection. Please enter valid numbers.")
        return
    
    test_df = load_and_process_data(file_path, test_sheet)
    training_df = load_and_process_data(file_path, train_sheet)
    
    od_columns = test_df.columns[1:]
    print("Available OD columns:")
    for i, col in enumerate(od_columns):
        print(f"{i + 1}. {col}")
    
    try:
        choice = int(input("Enter the number corresponding to the OD column you want to plot: "))
        selected_od = od_columns[choice - 1]
    except (ValueError, IndexError):
        print("Invalid selection. Please enter a valid number.")
        return
    
    # Plot Richards' model with only training data
    fit_and_plot_richards(test_df, selected_od, training_df)

    # Ask if the user wants to include the test data in the model fitting
    include_test_data = input("Would you like to update the model with test data and plot the improved model? (yes/no): ").strip().lower()
    
    if include_test_data == 'yes':
        combined_df = pd.concat([training_df, test_df], axis=0)
        fit_and_plot_richards(test_df, selected_od, training_df, combined_df)
    else:
        print("Model fitting completed with only training data.")

if __name__ == "__main__":
    main()
