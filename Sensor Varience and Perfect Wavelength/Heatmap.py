import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from tabulate import tabulate
import os
from itertools import combinations

file_path = r"Device 2 - Copy.xlsx"
sheets = ["100%", "50%", "25%", "12%", "1%", "0.78%", "0.39%", "0.5%", "0%"]

def Heatmap():
    output_folder = 'plots'
    os.makedirs(output_folder, exist_ok=True)

    plancks = 6.626e-34

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']

    r_squared_dict = {sheet_name: {} for sheet_name in sheets}
    top_combinations_dict = {sheet_name: {} for sheet_name in sheets}

    for sheet_name in sheets:
        plt.figure(figsize=(14, 7))
        plt.title(f'OD against Time for {sheet_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('OD')
        plt.grid(True)

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[df.columns[4], df.columns[5]])  # Drop columns 3 and 4
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

                if sensor_id not in r_squared_dict[sheet_name]:
                    r_squared_dict[sheet_name][sensor_id] = {}
                r_squared_dict[sheet_name][sensor_id][f'F{i}'] = r_squared

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

# Prepare data for heatmap
    heatmap_data = []

    for sheet_name, sensors in r_squared_dict.items():
        for sensor_id, freqs in sensors.items():
            for freq, r_squared in freqs.items():
                heatmap_data.append([sheet_name, sensor_id, freq, r_squared])

    heatmap_df = pd.DataFrame(heatmap_data, columns=['Sheet', 'Sensor', 'Frequency', 'R_squared'])
    heatmap_pivot = heatmap_df.pivot_table(index=['Sheet', 'Sensor'], columns='Frequency', values='R_squared')

# Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'R-squared'})
    plt.title('R-squared Values Heatmap')
    plt.xlabel('Frequency Channels')
    plt.ylabel('Sheet and Sensor')
    heatmap_filename = f"{output_folder}/R_squared_heatmap.png"
    plt.savefig(heatmap_filename)
    plt.show()
    print(f"Heatmap saved as {heatmap_filename}")

    print("Analysis complete.")
