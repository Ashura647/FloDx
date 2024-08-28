import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
import os

file_path = r"Device 2 - Copy.xlsx"
sheets = ["100%", "50%", "25%", "12%", "1%", "0.78%", "0.39%", "0.5%", "0%"]

def Varience():
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    plancks = 6.626e-34

    best_combinations_dict = {sheet_name: {} for sheet_name in sheets}
    variance_dict = {sheet_name: {} for sheet_name in sheets}

    for sheet_name in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        df.sort_values(by=["sensor", "time (s)"], inplace=True)

        unique_sensor_ids = df['sensor'].unique()

        for sensor_id in unique_sensor_ids:
            sensor_data = df[df['sensor'] == sensor_id]

            F0 = 1 / sensor_data['Clear']

        # List of available channels (wavelengths)
            channels = [f'F{i}' for i in range(1, 9) if f'F{i}' in sensor_data.columns]

            best_r2 = -np.inf
            best_combination = None

        # Evaluate all combinations of channels
            for r in range(1, min(len(channels), 4) + 1):  # Maximum combination size limited to 4 channels
                for combo in combinations(channels, r):
                    X = sensor_data[list(combo)].values
                    y = sensor_data['Clear'].values
                    model = LinearRegression()
                    model.fit(X, y)
                    r2 = model.score(X, y)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_combination = combo

        # Store the best combination for this sensor
            best_combinations_dict[sheet_name][(sensor_id, best_combination)] = best_r2

        # Calculate standard deviation between channels in the best combination
            if best_combination:
                std_devs = {}
                for i, channel1 in enumerate(best_combination):
                    for j, channel2 in enumerate(best_combination):
                        if i < j:
                            std_dev_channel1 = np.std(sensor_data[channel1])
                            std_dev_channel2 = np.std(sensor_data[channel2])
                            variance = np.abs(std_dev_channel1 - std_dev_channel2)
                            std_devs[(channel1, channel2)] = variance

                variance_dict[sheet_name][(sensor_id, best_combination)] = std_devs

    rows_variances = []

    for sheet_name, variances in variance_dict.items():
        for (sensor_id, combination), channel_variances in variances.items():
            for (channel1, channel2), std_dev_variance in channel_variances.items():
                rows_variances.append([sheet_name, sensor_id, channel1, channel2, std_dev_variance])

    columns_variances = ['Sheet', 'Sensor ID', 'Channel 1', 'Channel 2', 'Std Deviation Variance']
    results_variances_df = pd.DataFrame(rows_variances, columns=columns_variances)

    excel_file_variances = os.path.join(output_folder, 'channel_variances.xlsx')
    results_variances_df.to_excel(excel_file_variances, index=False)

    print(f"Channel standard deviation variances exported to {excel_file_variances}")
