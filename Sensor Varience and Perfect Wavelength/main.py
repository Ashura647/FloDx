import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tabulate import tabulate
import os
from sklearn.linear_model import LinearRegression
from itertools import combinations

file_path = r"Device 1 - Copy.xlsx"
sheets = ["100%", "50%", "25%", "12%", "1%", "0.78%", "0.39%", "0.5%", "0%"]


def Combinations():
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

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

            channels = [f'F{i}' for i in range(1, 9) if f'F{i}' in sensor_data.columns]

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
    
def ODvsTPlot(file_path, sheets):
    output_folder = 'plots'
    os.makedirs(output_folder, exist_ok=True)

    plancks = 6.626e-34

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'yellow', 'black', 'gray']

    for sheet_name in sheets:
        plt.figure(figsize=(14, 7))
        plt.title(f'OD against Time for {sheet_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('OD')
        plt.grid(True)

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=[df.columns[4], df.columns[5]])  

        df.sort_values(by=["sensor", "time (s)"], inplace=True)

        unique_sensor_ids = df['sensor'].unique()

        color_index = 0  

        for sensor_id in unique_sensor_ids:
            sensor_data = df[df['sensor'] == sensor_id]

            F0 = 1 / sensor_data['Clear']

            od_table = []

            for i in range(1, 9):
                column_name = f'F{i}'
                if column_name in sensor_data.columns:
                    F = 1 / sensor_data[column_name]

                    I = plancks * F
                    I0 = plancks * F0

                    sensor_data[f'OD_{i}'] = -np.log10(I / I0)

                    time_column = sensor_data['time (s)']
                    od_column = sensor_data[f'OD_{i}']

                    plt.plot(time_column, od_column, label=f'Channel_{i}', color=colors[color_index])

                    m, b = np.polyfit(time_column, od_column, 1)

                    plt.plot(time_column, m * time_column + b, linestyle='dashed')

                    r_squared = r2_score(od_column, m * time_column + b)

                    plt.text(time_column.iloc[-1], m * time_column.iloc[-1] + b, f'R² = {r_squared:.4f}', fontsize=9, verticalalignment='bottom')

                    for t, od in zip(time_column, od_column):
                        od_table.append([f'F{i}', t, od])

                    print(f'R-squared for {sheet_name} F{i}:', r_squared)

                    color_index = (color_index + 1) % len(colors)

            print(f"\nOD values for {sheet_name} and sensor ID {sensor_id}:")
            print(tabulate(od_table, headers=['Frequency', 'Time (s)', 'OD'], tablefmt='grid'))

            plt.legend()
            plt.title(f'OD against Time for {sheet_name} - Sensor ID {sensor_id}')

            plot_filename = f"{output_folder}/{sheet_name}_SensorID_{sensor_id}_plot.png"
            plt.savefig(plot_filename)
            print(f"Plot saved as {plot_filename}")

            plt.clf()

        plt.close()

    print("All plots saved successfully.")

def average_data(file_path, sheets):
    rows_avg_freq = []

    for sheet_name in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=[df.columns[4], df.columns[5]])  # Assuming columns 4 and 5 are dropped

        df.sort_values(by=["sensor", "time (s)"], inplace=True)

        unique_sensor_ids = df['sensor'].unique()

        for sensor_id in unique_sensor_ids:
            sensor_data = df[df['sensor'] == sensor_id]

            for i in range(1, 9):
                column_name = f'F{i}'
                if column_name in sensor_data.columns:
                    F = sensor_data[column_name]

                    avg_frequency = np.average(F)
                    sensor_data[f'AVG_F_{i}'] = avg_frequency

                    rows_avg_freq.append([sheet_name, sensor_id, column_name, avg_frequency])

            print(f"Average Frequencies for Sheet '{sheet_name}', Sensor ID '{sensor_id}':")
            print(pd.DataFrame(rows_avg_freq, columns=['Sheet', 'Sensor ID', 'Channel', 'Average Frequency']))
            print()

    # Create a DataFrame from the list of rows
    avg_freq_df = pd.DataFrame(rows_avg_freq, columns=['Sheet', 'Sensor ID', 'Channel', 'Average Frequency'])

    # Save to CSV
    #avg_freq_csv = 'average_frequencies.csv'
    #avg_freq_df.to_csv(avg_freq_csv, index=False)
    #print(f"Average frequencies saved to {avg_freq_csv}")
    return avg_freq_df

def extract_concentration(sheet_name): #Stripping the Percentage symbol to get sheets
    try:
        return float(sheet_name.rstrip('%'))
    except ValueError:
        return float('inf')

def sort_sheets_by_concentration(sheets):
    return sorted(sheets, key=extract_concentration)

def ODvsCPlot(file_path, sheets):
    plancks = 6.626e-34

    sorted_sheets = sort_sheets_by_concentration(sheets)

    results = []

    # Read '0%' sheet to get F0 values
    df_0 = pd.read_excel(file_path, sheet_name='0%')
    df_0.columns = df_0.columns.str.strip()
    df_0 = df_0.drop(columns=[df_0.columns[4], df_0.columns[5]], errors='ignore')
    df_0.sort_values(by=["sensor", "time (s)"], inplace=True)

    unique_sensor_ids_0 = df_0['sensor'].unique()

    # Iterate through each channel (F1 to F8)
    for i in range(1, 9):
        column_name = 'F' + str(i)

        # Check if column exists in '0%' sheet
        if column_name in df_0.columns:
            F0 = df_0[column_name]

            # Now process the other sheets
            for sheet_name in sorted_sheets:
                if sheet_name == '0%':
                    continue  # Skip the '0%' sheet as we already have F0 values

                print(f"Processing sheet: {sheet_name}")

                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df.columns = df.columns.str.strip()
                df = df.drop(columns=[df.columns[4], df.columns[5]], errors='ignore') # Dropping these columns due to poor linear regression over time
                df.sort_values(by=["sensor", "time (s)"], inplace=True)

                unique_sensor_ids = df['sensor'].unique()

                for sensor_id in unique_sensor_ids: 
                    print(f"Processing sensor {sensor_id}, concentration {sheet_name}")

                    sensor_data = df[df['sensor'] == sensor_id]

                    if column_name in sensor_data.columns:
                        F = sensor_data[column_name]

                        # Check for division by zero
                        if np.any(F == 0):
                            print(f"Division by zero encountered in sensor {sensor_id}, channel {column_name}. Skipping...")
                            continue

                        I = plancks * F
                        I0 = plancks * F0

                        # Calculate Optical Density (OD)
                        OD = np.log10(I0 / I)  # Calculate OD 
                        print(f"OD values for sensor {sensor_id}, concentration {sheet_name}, channel {i}:\n{OD}")

                        sensor_data['OD_' + str(i)] = np.abs(OD)

                        # Taking the Average and the Standard Deviation. Outputted Seperately
                        od_column = sensor_data['OD_' + str(i)]
                        avg_od = od_column.mean()
                        std_od = od_column.std()

                        print(f"Avg OD for sensor {sensor_id}, concentration {sheet_name}, channel {i}: {avg_od}")

                        results.append({
                            'Sensor': sensor_id,
                            'Concentration': sheet_name,
                            'Channel_' + str(i) + '_avg': avg_od,
                            'Channel_' + str(i) + '_std': std_od
                        })

    results_df = pd.DataFrame(results)

    results_df['Concentration'] = pd.Categorical(results_df['Concentration'], categories=sorted_sheets, ordered=True)

    # Create a new DataFrame for average OD values per concentration per sensor
    avg_results = results_df.groupby(['Sensor', 'Concentration']).mean().reset_index()

    avg_output_file = "ODConcDat.xlsx"
    avg_results.to_excel(avg_output_file, index=False)
    print(f"Calculations saved to {avg_output_file}")


#Input here:
ODvsCPlot(file_path, sheets)
