import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import os
import sys

# Parameters for the logistic growth model
OD_max = 2  # Maximum Optical Density (carrying capacity)
r = 1.4       # Growth rate
time_max = 24  # Maximum time for the simulation in hours
interval = 500  # Time between frames in milliseconds
fallback_check_time = 1  # Time in hours before fallback check
max_user_prompts = 2  # Number of times to prompt the user before checking other folders

# Time points for the logistic growth simulation
time_points = np.linspace(0, time_max, num=100)

# Logistic growth function scaled for OD
def logistic_growth(t, OD_max, r, t_0):
    return OD_max / (1 + np.exp(-r * (t - t_0)))

def find_matching_bounds_in_folder(bounds_folder):
    """Find the matching bounds from the given folder based on simulation data."""
    try:
        for filename in os.listdir(bounds_folder):
            if filename.endswith(".csv"):
                bounds_file = os.path.join(bounds_folder, filename)
                df_bounds = pd.read_csv(bounds_file)
                if set(df_bounds.columns) == {'time (h)', 'Upper Bound', 'Lower Bound'}:
                    return df_bounds
    except PermissionError:
        log_event(f"Permission denied accessing folder: {bounds_folder}")
        sys.exit(1)
    raise ValueError("No matching bounds file found in the folder.")

def find_matching_prediction_in_folder(bounds_folder):
    """Find a matching prediction file in the given folder based on the naming pattern."""
    try:
        for filename in os.listdir(bounds_folder):
            if filename.endswith("_predictions.csv"):
                pred_file = os.path.join(bounds_folder, filename)
                df_pred = pd.read_csv(pred_file)
                if set(df_pred.columns) == {'time (h)', 'Predicted', 'Upper Bound', 'Lower Bound'}:
                    return df_pred
    except PermissionError:
        log_event(f"Permission denied accessing folder: {bounds_folder}")
        sys.exit(1)
    raise ValueError("No matching prediction file found in the folder.")

def adjust_bounds(time, current_OD, y_pred_upper, y_pred_lower):
    """Adjust the bounds if they cross the simulation line."""
    if len(current_OD) > 0:
        # Interpolate the bounds to match the length of the simulation data
        new_y_pred_upper = np.interp(time, np.linspace(0, time_max, len(y_pred_upper)), y_pred_upper)
        new_y_pred_lower = np.interp(time, np.linspace(0, time_max, len(y_pred_lower)), y_pred_lower)
        
        # Update the bounds based on current simulation data
        new_y_pred_upper = np.maximum(new_y_pred_upper, current_OD)
        new_y_pred_lower = np.minimum(new_y_pred_lower, current_OD)
        return new_y_pred_upper, new_y_pred_lower
    
    return y_pred_upper, y_pred_lower

def plot_from_csv_with_simulation(csv_file, lb_models_folder, title, t_0, save_as_gif=False, suspected_bacteria='n/a'):
    """Read data from a CSV file and plot the predictions with error bounds, including a logistic growth simulation.
       Optionally save the animation as a GIF."""
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Check for required columns
    required_columns = ['time (h)', 'Predicted', 'Upper Bound', 'Lower Bound']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV file is missing required column: {col}")

    # Extract data from the DataFrame
    time = df['time (h)'].values
    y_pred = df['Predicted'].values
    y_pred_upper = df['Upper Bound'].values
    y_pred_lower = df['Lower Bound'].values

    # Ensure data length consistency
    if not (len(time) == len(y_pred) == len(y_pred_upper) == len(y_pred_lower)):
        raise ValueError("Length of time and prediction columns must be the same.")

    # Determine which folders to scan
    if suspected_bacteria == 'n/a':
        folders_to_scan = [os.path.join(lb_models_folder, subdir) for subdir in os.listdir(lb_models_folder) if os.path.isdir(os.path.join(lb_models_folder, subdir))]
    else:
        folder_path = os.path.join(lb_models_folder, suspected_bacteria)
        if not os.path.exists(folder_path):
            print(f"Warning: Specified bacteria folder '{suspected_bacteria}' not found. Scanning all folders instead.")
            folders_to_scan = [os.path.join(lb_models_folder, subdir) for subdir in os.listdir(lb_models_folder) if os.path.isdir(os.path.join(lb_models_folder, subdir))]
        else:
            folders_to_scan = [folder_path]

    # Add fallback folder
    fallback_folder = os.path.join(lb_models_folder, r'S.AUREUS-LB-FAKE')
    if os.path.isdir(fallback_folder):
        folders_to_scan.append(fallback_folder)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the prediction data with initial error bounds
    pred_line, = ax.plot(time, y_pred, label='Predicted', linestyle='--', color='red')
    upper_bound_line, = ax.plot(time, y_pred_upper, linestyle='--', color='gray', label='Upper Bound')
    lower_bound_line, = ax.plot(time, y_pred_lower, linestyle='--', color='gray', label='Lower Bound')

    # Initialize fill_between for error margin
    error_margin = ax.fill_between(time, y_pred_lower, y_pred_upper, color='gray', alpha=0.3, label='Error Margin')

    # Plot settings
    ax.set_xlim(0, max(time))
    ax.set_ylim(0, OD_max * 1.1)  # Adjust Y-limit to OD_max
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Optical Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    # Add logistic growth simulation line
    line, = ax.plot([], [], lw=2, color='blue', label='Logistic Growth Simulation')

    # Initialize the simulation line
    def init():
        line.set_data([], [])
        return [line]

    # Update the simulation line
    def update(frame):
        nonlocal prev_crossing, pred_line, upper_bound_line, lower_bound_line, y_pred, y_pred_upper, y_pred_lower

        if frame >= len(time_points):
            return [line]

        # Ensure index is within bounds
        frame = min(frame, len(time_points) - 1)
        current_time = time_points[:frame + 1]  # Ensure we include the current frame
        current_OD = logistic_growth(current_time, OD_max, r, t_0)
        line.set_data(current_time, current_OD)

        # Check for crossing events
        if len(current_time) > 1:
            current_last_time = current_time[-1]
            current_last_OD = current_OD[-1]

            # Find the closest index for the current time
            closest_idx = np.searchsorted(time, current_last_time, side='left')
            if closest_idx >= len(y_pred_upper):
                closest_idx = len(y_pred_upper) - 1
            if closest_idx < 0:
                closest_idx = 0

            upper_bound = y_pred_upper[closest_idx] if closest_idx < len(y_pred_upper) else y_pred_upper[-1]
            lower_bound = y_pred_lower[closest_idx] if closest_idx < len(y_pred_lower) else y_pred_lower[-1]

            # Debugging prints
            print(f"Current Time: {current_last_time:.2f}")
            print(f"Current OD: {current_last_OD:.2f}")
            print(f"Upper Bound: {upper_bound:.2f}")
            print(f"Lower Bound: {lower_bound:.2f}")

            # Check if the simulation line is within the bounds
            within_upper_bound = current_last_OD <= upper_bound
            within_lower_bound = current_last_OD >= lower_bound

            if prev_crossing is None:
                prev_crossing = (within_upper_bound, within_lower_bound)
            else:
                prev_within_upper, prev_within_lower = prev_crossing
                current_within_upper = within_upper_bound
                current_within_lower = within_lower_bound

                if prev_within_upper and not current_within_upper:
                    log_event(f"Crossed above upper bound at time {current_last_time:.2f} hours.")
                    # Attempt to update prediction data
                    if attempt_update_prediction(folders_to_scan, max_user_prompts):
                        # If the user agreed to update, update the plot
                        pred_line.set_ydata(y_pred)
                        upper_bound_line.set_ydata(y_pred_upper)
                        lower_bound_line.set_ydata(y_pred_lower)
                    else:
                        # If user declined and we have checked all folders
                        if not recheck_other_folders(folders_to_scan, fallback_folder):
                            log_event("No valid prediction data found.")
                            return [line]

                    prev_crossing = (current_within_upper, current_within_lower)
                elif prev_within_lower and not current_within_lower:
                    log_event(f"Crossed below lower bound at time {current_last_time:.2f} hours.")
                    # Attempt to update prediction data
                    if attempt_update_prediction(folders_to_scan, max_user_prompts):
                        # If the user agreed to update, update the plot
                        pred_line.set_ydata(y_pred)
                        upper_bound_line.set_ydata(y_pred_upper)
                        lower_bound_line.set_ydata(y_pred_lower)
                    else:
                        # If user declined and we have checked all folders
                        if not recheck_other_folders(folders_to_scan, fallback_folder):
                            log_event("No valid prediction data found.")
                            return [line]

                    prev_crossing = (current_within_upper, current_within_lower)
                else:
                    prev_crossing = (current_within_upper, current_within_lower)

        return [line, pred_line, upper_bound_line, lower_bound_line, error_margin]

    def attempt_update_prediction(folders_to_scan, max_prompts):
        """Attempt to update the prediction by asking the user."""
        prompts = 0
        while prompts < max_prompts:
            update = input("A new prediction line was found. Do you want to update the plot with this data? (yes/no): ").strip().lower()
            if update == 'yes':
                # Update the prediction from the current folder
                return True
            elif update == 'no':
                prompts += 1
                if prompts >= max_prompts:
                    return False
            else:
                print("Please enter 'yes' or 'no'.")
                continue

        return False

    def recheck_other_folders(folders_to_scan, fallback_folder):
        """Check other folders for valid prediction data after user declines update."""
        log_event("User declined to update prediction. Checking other folders.")
        for folder in folders_to_scan:
            if folder == fallback_folder:
                continue
            try:
                df_new_pred = find_matching_prediction_in_folder(folder)
                if df_new_pred is not None:
                    new_time = df_new_pred['time (h)'].values
                    new_y_pred = df_new_pred['Predicted'].values
                    new_y_pred_upper = df_new_pred['Upper Bound'].values
                    new_y_pred_lower = df_new_pred['Lower Bound'].values
                    # Update the plot with new prediction data
                    pred_line.set_ydata(new_y_pred)
                    upper_bound_line.set_ydata(new_y_pred_upper)
                    lower_bound_line.set_ydata(new_y_pred_lower)
                    return True
            except ValueError as ve:
                log_event(str(ve))
                continue

        # Check the fallback folder if no new valid prediction was found
        try:
            df_fallback_pred = find_matching_prediction_in_folder(fallback_folder)
            if df_fallback_pred is not None:
                new_time = df_fallback_pred['time (h)'].values
                new_y_pred = df_fallback_pred['Predicted'].values
                new_y_pred_upper = df_fallback_pred['Upper Bound'].values
                new_y_pred_lower = df_fallback_pred['Lower Bound'].values
                # Update the plot with fallback prediction data
                pred_line.set_ydata(new_y_pred)
                upper_bound_line.set_ydata(new_y_pred_upper)
                lower_bound_line.set_ydata(new_y_pred_lower)
                return True
        except ValueError as ve:
            log_event(str(ve))
            return False

        return False

    # Initialize previous crossing status
    prev_crossing = None

    # Set up the animation
    ani = animation.FuncAnimation(fig, update, frames=len(time_points), init_func=init, blit=True, interval=interval)

    # Save the animation as a GIF if required
    if save_as_gif:
        ani.save('animation.gif', writer='imagemagick')

    plt.show()

def log_event(message):
    """Log events to a file."""
    logging.basicConfig(filename='simulation.log', level=logging.INFO)
    logging.info(message)

# Example usage
csv_file = r'LB-Models\E-Coli-LB\No.2 (n=12)_predictions.csv'
lb_models_folder = r'LB-Models'
title = 'Bacteria Growth Simulation'
t_0 = float(input("Enter the initial time (t_0) for the simulation: "))
save_as_gif = input("Do you want to save the animation as a GIF? (yes/no): ").strip().lower() == 'yes'
suspected_bacteria = input("Enter the suspected bacteria type (e.g., E-Coli, S-Aureus) or 'n/a' to scan all folders: ").strip()

plot_from_csv_with_simulation(csv_file, lb_models_folder, title, t_0, save_as_gif, suspected_bacteria)
