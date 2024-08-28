import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from itertools import combinations
import os

file_path = r"Device 2 - Copy.xlsx"
sheets = ["100%", "50%", "25%", "12%", "1%", "0.78%", "0.39%", "0.5%", "0%"]
def Combinations():
    sheets = ["100%", "50%", "25%", "12%", "1%", "0.78%", "0.39%", "0.5%", "0%"]

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    plancks = 6.626e-34

    best_combinations_dict = {sheet_name: {} for sheet_name in sheets}
    variance_dict = {sheet_name: {} for sheet_name in sheets}

    for sheet_name in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=[df.columns[2], df.columns[3]])  # Drop columns 3 and 4
        df.sort_values(by=["sensor", "time (s)"], inplace=True)

        unique_sensor_ids = df['sensor'].unique()

    for sensor_id in unique_sensor_ids:
        sensor_data = df[df['sensor'] == sensor_id]

        F0 = 1 / sensor_data['Clear']

        channels = [f'F{i}' for i in range(1, 9) if f'F{i}' in sensor_data.columns and i != 3 and i != 4]

        best_r2 = -np.inf
        best_combination = None

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

        best_combinations_dict[sheet_name][(sensor_id, best_combination)] = best_r2

        if best_combination:
            for i, channel1 in enumerate(best_combination):
                for j, channel2 in enumerate(best_combination):
                    if i < j:
                        var_channel1 = np.var(sensor_data[channel1])
                        var_channel2 = np.var(sensor_data[channel2])
                        variance = np.abs(var_channel1 - var_channel2)
                        variance_dict[sheet_name][(sensor_id, channel1, channel2)] = variance

    rows_combinations = []
    rows_variances = []

    for sheet_name, sensor_combos in best_combinations_dict.items():
        for (sensor_id, combination), r2 in sensor_combos.items():
            combination_row = [sheet_name, sensor_id] + list(combination) + [r2]
            rows_combinations.append(combination_row)

        if (sensor_id, combination) in variance_dict[sheet_name]:
            for (channel1, channel2), variance in variance_dict[sheet_name][(sensor_id, combination)].items():
                variance_row = [sheet_name, sensor_id, channel1, channel2, variance]
                rows_variances.append(variance_row)

    max_channels = max(len(row) - 3 for row in rows_combinations)  # Exclude 'Sheet', 'Sensor ID', and 'R-squared' columns
    columns_combinations = ['Sheet', 'Sensor ID'] + [f'F{i}' for i in range(1, max_channels + 1)] + ['R-squared']

    results_combinations_df = pd.DataFrame(rows_combinations, columns=columns_combinations)
    results_variances_df = pd.DataFrame(rows_variances, columns=['Sheet', 'Sensor ID', 'Channel 1', 'Channel 2', 'Variance'])

    excel_file_combinations = os.path.join(output_folder, 'best_combinations.xlsx')
    excel_file_variances = os.path.join(output_folder, 'channel_variances.xlsx')

    results_combinations_df.to_excel(excel_file_combinations, index=False)
    results_variances_df.to_excel(excel_file_variances, index=False)

    print(f"Best combinations (excluding F3 and F4) exported to {excel_file_combinations}")
    print(f"Channel variances exported to {excel_file_variances}")
