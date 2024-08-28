import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_concentration(sheet_name):
    try:
        return float(sheet_name.rstrip('%'))
    except ValueError:
        return float('inf')

def sort_sheets_by_concentration(sheets):
    return sorted(sheets, key=extract_concentration)

def calculate_r_squared(x, y, degree=3):
    coeffs = np.polyfit(x, y, degree)
    p = np.poly1d(coeffs)
    y_pred = p(x)
    y_mean = np.mean(y)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def ODvsCPlot(file_path, sheets):
    plancks = 6.626e-34

    sorted_sheets = sort_sheets_by_concentration(sheets)

    results = []

    for sheet_name in sorted_sheets:
        print(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.columns = df.columns.str.strip()
        df = df.drop(columns=[df.columns[4], df.columns[5]], errors='ignore')

        df.sort_values(by=["sensor", "time (s)"], inplace=True)

        unique_sensor_ids = df['sensor'].unique()

        for sensor_id in unique_sensor_ids:
            print(f"Processing sensor {sensor_id}, concentration {sheet_name}")

            sensor_data = df[df['sensor'] == sensor_id]

            F0 =sensor_data['Clear']

            for i in range(1, 9):
                column_name = 'F' + str(i)
                if column_name in sensor_data.columns:
                    F =  sensor_data[column_name]

                    # Check for division by zero
                    if np.any(F == 0) or np.any(F0 == 0):
                        print(f"Division by zero encountered in sensor {sensor_id}, channel {column_name}. Skipping...")
                        continue

                    I = plancks * F
                    I0 = plancks * F0

                    # Check for zero or negative values in intensity calculation
                    if np.any(I <= 0) or np.any(I0 <= 0):
                        print(f"Invalid intensity values in sensor {sensor_id}, channel {column_name}. Skipping...")
                        continue

                    # Calculate Optical Density (OD)
                    OD = -np.log10(I / I0)
                    print(f"OD values for sensor {sensor_id}, concentration {sheet_name}, channel {i}:\n{OD}")

                    sensor_data['OD_' + str(i)] = np.abs(OD)

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

    avg_output_file = "AveragesOD.xlsx"
    avg_results.to_excel(avg_output_file, index=False)
    print(f"Averages saved to {avg_output_file}")

    unique_sensor_ids = avg_results['Sensor'].unique()
    channels = ['Channel_' + str(i) for i in range(1, 9)]

ODvsCPlot(file_path, sheets)
