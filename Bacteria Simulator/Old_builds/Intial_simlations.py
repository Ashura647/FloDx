import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import os

# Parameters for the logistic growth model
OD_max = 1  # Maximum Optical Density (carrying capacity)
r = 1.4       # Growth rate
time_max = 24  # Maximum time for the simulation in hours
interval = 500  # Time between frames in milliseconds

# Time points for the logistic growth simulation
time_points = np.linspace(0, time_max, num=100)

# Logistic growth function scaled for OD
def logistic_growth(t, OD_max, r, t_0):
    return OD_max / (1 + np.exp(-r * (t - t_0)))

def find_matching_bounds(bounds_folder):
    """Find the matching bounds from the given folder based on simulation data."""
    for filename in os.listdir(bounds_folder):
        if filename.endswith(".csv"):
            bounds_file = os.path.join(bounds_folder, filename)
            df_bounds = pd.read_csv(bounds_file)
            if set(df_bounds.columns) == {'time (h)', 'Upper Bound', 'Lower Bound'}:
                return df_bounds
    raise ValueError("No matching bounds file found in the folder.")

def find_matching_prediction(bounds_folder):
    """Find a matching prediction file in the given folder based on the naming pattern."""
    for filename in os.listdir(bounds_folder):
        if filename.endswith("_predictions.csv"):
            pred_file = os.path.join(bounds_folder, filename)
            df_pred = pd.read_csv(pred_file)
            if set(df_pred.columns) == {'time (h)', 'Predicted', 'Upper Bound', 'Lower Bound'}:
                return df_pred
    raise ValueError("No matching prediction file found in the folder.")

def adjust_bounds(time, current_OD, y_pred_upper, y_pred_lower):
    """Adjust the bounds if they cross the simulation line."""
    time_min = time.min()
    time_max = time.max()
    
    # Adjust bounds based on the current simulation line
    if len(current_OD) > 0:
        # New upper and lower bounds
        new_y_pred_upper = np.maximum(y_pred_upper, current_OD)
        new_y_pred_lower = np.minimum(y_pred_lower, current_OD)
        return new_y_pred_upper, new_y_pred_lower
    
    return y_pred_upper, y_pred_lower

def setup_logger():
    """Set up the logger for real-time logging to both console and file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create file handler for logging
    file_handler = logging.FileHandler('simulator.log')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Create console handler for logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_event(message):
    """Log the event to the file and print it to the console."""
    logging.info(message)

def plot_from_csv_with_simulation(csv_file, bounds_folder, title, t_0, save_as_gif=False):
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

    # Scan the folder for matching bounds
    bounds_df = find_matching_bounds(bounds_folder)
    bounds_time = bounds_df['time (h)'].values
    bounds_upper = bounds_df['Upper Bound'].values
    bounds_lower = bounds_df['Lower Bound'].values

    # Ensure bounds data length consistency
    if not (len(bounds_time) == len(bounds_upper) == len(bounds_lower)):
        raise ValueError("Length of bounds data columns must be the same.")

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the prediction data with initial error bounds
    pred_line, = ax.plot(time, y_pred, label='Predicted', linestyle='--', color='red')
    upper_bound_line, = ax.plot(time, y_pred_upper, linestyle='--', color='gray', label='Upper Bound')
    lower_bound_line, = ax.plot(time, y_pred_lower, linestyle='--', color='gray', label='Lower Bound')

    # Initialize fill_between for error margin
    error_margin = ax.fill_between(time, y_pred_lower, y_pred_upper, color='gray', alpha=0.3, label='Error Margin')

    # Add logistic growth simulation line
    line, = ax.plot([], [], lw=2, color='blue', label='Logistic Growth Simulation')

    # Initialize plot settings
    ax.set_xlim(0, max(time))
    ax.set_ylim(0, max(max(y_pred_upper), OD_max) * 1.1)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Optical Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

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

        # Check for crossing events and approach warnings
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

            # Thresholds for warnings
            threshold_warning = 0.1  # Define a threshold for approaching the bounds

            # Check if the simulation line is approaching the bounds
            approaching_upper = upper_bound - threshold_warning <= current_last_OD <= upper_bound
            approaching_lower = lower_bound <= current_last_OD <= lower_bound + threshold_warning

            if approaching_upper:
                log_event(f"Warning: Simulation line is approaching the upper bound at time {current_last_time:.2f} hours.")
            if approaching_lower:
                log_event(f"Warning: Simulation line is approaching the lower bound at time {current_last_time:.2f} hours.")

            # Threshold to determine if the simulation line is well past the bounds
            threshold = 0.2  # Define a threshold to determine "well past" the bounds
            well_past_upper = current_last_OD > (upper_bound + threshold)
            well_past_lower = current_last_OD < (lower_bound - threshold)

            if prev_crossing is None:
                prev_crossing = (well_past_upper, well_past_lower)
            else:
                prev_above_upper, prev_below_lower = prev_crossing
                current_above_upper = well_past_upper
                current_below_lower = well_past_lower

                if prev_above_upper and not current_above_upper:
                    log_event(f"Crossed below upper bound at time {current_last_time:.2f} hours.")
                    # Find matching prediction file
                    try:
                        new_pred_df = find_matching_prediction(bounds_folder)
                        new_pred_time = new_pred_df['time (h)'].values
                        new_y_pred = new_pred_df['Predicted'].values
                        new_y_pred_upper = new_pred_df['Upper Bound'].values
                        new_y_pred_lower = new_pred_df['Lower Bound'].values
                        
                        # Prompt user for update
                        user_choice = input("A new prediction line was found. Do you want to update the plot with this data? (yes/no): ").strip().lower()
                        if user_choice == 'yes':
                            y_pred = new_y_pred
                            y_pred_upper = new_y_pred_upper
                            y_pred_lower = new_y_pred_lower
                            pred_line.set_ydata(y_pred)
                            upper_bound_line.set_ydata(y_pred_upper)
                            lower_bound_line.set_ydata(y_pred_lower)
                        else:
                            # Adjust bounds if user opts not to update prediction line
                            y_pred_upper, y_pred_lower = adjust_bounds(time, current_OD, y_pred_upper, y_pred_lower)
                            upper_bound_line.set_ydata(y_pred_upper)
                            lower_bound_line.set_ydata(y_pred_lower)

                    except ValueError:
                        log_event("Unable to update prediction data.")
                    
                if prev_below_lower and not current_below_lower:
                    log_event(f"Crossed above lower bound at time {current_last_time:.2f} hours.")
                    # Find matching prediction file
                    try:
                        new_pred_df = find_matching_prediction(bounds_folder)
                        new_pred_time = new_pred_df['time (h)'].values
                        new_y_pred = new_pred_df['Predicted'].values
                        new_y_pred_upper = new_pred_df['Upper Bound'].values
                        new_y_pred_lower = new_pred_df['Lower Bound'].values
                        
                        # Prompt user for update
                        user_choice = input("A new prediction line was found. Do you want to update the plot with this data? (yes/no): ").strip().lower()
                        if user_choice == 'yes':
                            y_pred = new_y_pred
                            y_pred_upper = new_y_pred_upper
                            y_pred_lower = new_y_pred_lower
                            pred_line.set_ydata(y_pred)
                            upper_bound_line.set_ydata(y_pred_upper)
                            lower_bound_line.set_ydata(y_pred_lower)
                        else:
                            # Adjust bounds if user opts not to update prediction line
                            y_pred_upper, y_pred_lower = adjust_bounds(time, current_OD, y_pred_upper, y_pred_lower)
                            upper_bound_line.set_ydata(y_pred_upper)
                            lower_bound_line.set_ydata(y_pred_lower)

                    except ValueError:
                        log_event("Unable to update prediction data.")

                prev_crossing = (current_above_upper, current_below_lower)

        return [line, pred_line, upper_bound_line, lower_bound_line, error_margin]

    # Initialize previous crossing state
    prev_crossing = None

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(time_points),
        init_func=init, interval=interval, blit=True,
        repeat=False  # Ensure the animation does not loop
    )

    # Show or save the plot with the animation
    if save_as_gif:
        gif_filename = 'simulation_animation.gif'
        ani.save(gif_filename, writer='pillow', fps=1000 / interval)
        print(f"Animation saved as {gif_filename}")
    else:
        plt.show()

def main():
    setup_logger()  # Initialize logger
    
    # Step 1: Ask the user to select the bacterium
    bacterium_choice = input("Choose the bacterium (Ecoli, Salmonella, or Bacillus): ").strip().lower()

    # Step 2: Ask the user to select the medium
    if bacterium_choice == 'ecoli':
        medium_choice = input("Choose the medium for E. coli (LB, MAA, or M63): ").strip().lower()
    elif bacterium_choice == 'staph.a':
        medium_choice = input("Choose the medium for Salmonella (LB, MAA, or M63): ").strip().lower()
    else:
        print("Invalid bacterium choice! Please choose either 'Ecoli', 'Staph. A'.")
        return

    # Step 3: Select the corresponding file paths and parameters based on the choices
    if bacterium_choice == 'ecoli':
        if medium_choice == 'lb':
            csv_file_updated = r'LB-Models\E-Coli-LB\No.2 (n=12)_predictions.csv'
            bounds_folder = r'LB-Models\E-Coli-LB'
            title = 'OD Predictions vs Original Data with Simulation (E. coli in LB Medium)'
            t_0 = float(input("Enter the value for t_0 (suggested: 6): "))
        elif medium_choice == 'maa':
            csv_file_updated = r'MAA-2-Model\No.2 (n=12)_predictions.csv'
            bounds_folder = r'MAA-2-Model'
            title = 'OD Predictions vs Original Data with Simulation (E. coli in MAA Medium)'
            t_0 = float(input("Enter the value for t_0 (suggested: 10): "))
        elif medium_choice == 'm63':
            csv_file_updated = r'M63-Model\No.3 (n=12)_predictions.csv'
            bounds_folder = r'M63-Model'
            title = 'OD Predictions vs Original Data with Simulation (E. coli in M63 Medium)'
            t_0 = float(input("Enter the value for t_0 (suggested: 14): "))
        else:
            print("Invalid medium choice for E. coli! Please choose either 'LB', 'MAA', or 'M63'.")
            return

    elif bacterium_choice == 'staph.a':
        if medium_choice == 'lb':
            csv_file_updated = r'LB-Models\General\S.Aureus_LB_predictions.csv'
            bounds_folder = r'LB-Models\S.AUREUS-LB-FAKE'
            title = 'OD Predictions vs Original Data with Simulation (Staph Aureus in LB Medium)'
            t_0 = float(input("Enter the value for t_0 (suggested: 3): "))
        
        else:
            print("Invalid medium choice for Staph. A! Please choose 'LB'")
            return


    else:
        print("Invalid bacterium choice!")
        return

    # Step 4: Call the plotting function with the selected parameters
    plot_from_csv_with_simulation(csv_file_updated, bounds_folder, title, t_0)

# Run the main function
if __name__ == "__main__":
    main()
