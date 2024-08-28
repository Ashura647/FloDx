"""
import pandas as pd

# Load the Excel file
file_path = r'growthdash.xlsx'  # Replace with the path to your file
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Drop 12 columns after the first column
df.drop(df.columns[2:14], axis=1, inplace=True)

# Rename the remaining columns starting from the second column
new_column_names = ['OD' + str(i+1) for i in range(len(df.columns)-1)]
df.columns = [df.columns[0]] + new_column_names

# Save the modified DataFrame to a new Excel file
output_path = 'cleaned_data.xlsx'  # Replace with your desired output file path
df.to_excel(output_path, index=False)
"""

"""
import pandas as pd

# Load the dataset
file_path = 'gcplyr-dat.xlsx'
data = pd.read_excel(file_path)

# Convert time from minutes to hours
data['time_hours'] = data['time'] / 60

# Extract the time (in hours) and OD columns
subset_data = data[['time_hours', 'OD']]

# Determine the points where the data should be split
split_points = subset_data[subset_data['time_hours'] == 0].index.tolist() + [len(subset_data)]

# Split the dataset into segments based on split points
segments = [subset_data.iloc[split_points[i]:split_points[i+1]] for i in range(len(split_points)-1)]

# Save each segment to a different sheet in the same Excel file
output_file_path = 'sorted_data.xlsx'
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    for i, segment in enumerate(segments):
        sheet_name = f'Segment_{i+1}'
        segment.to_excel(writer, sheet_name=sheet_name, index=False)

print("Data has been saved successfully.")
"""

"""
import pandas as pd

def fuse_and_reorder_columns(input_file, output_file):
    # Read the Excel file
    xls = pd.ExcelFile(input_file)
    
    # Initialize lists to hold data
    time_hours_list = []
    od_data_list = []
    
    # Loop through all sheets
    for sheet_name in xls.sheet_names:
        # Read each sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Check if 'time_hours' column exists
        if 'time_hours' not in df.columns:
            print(f"'time_hours' column not found in sheet: {sheet_name}")
            continue
        
        # Collect 'time_hours' and OD columns
        time_hours_list.append(df[['time_hours']])
        
        # Collect OD columns, ignoring 'time_hours'
        od_columns = [col for col in df.columns if col != 'time_hours']
        if od_columns:
            od_data_list.append(df[od_columns])
    
    # Concatenate all time_hours columns side-by-side
    time_hours_df = pd.concat(time_hours_list, axis=0).drop_duplicates().reset_index(drop=True)
    
    # Concatenate all OD columns side-by-side
    od_data_df = pd.concat(od_data_list, axis=1)
    
    # Ensure time_hours is aligned with OD data
    if len(time_hours_df) != len(od_data_df):
        print("Warning: The number of rows in 'time_hours' does not match the number of rows in OD data.")
    
    # Combine 'time_hours' with OD data
    final_df = pd.concat([time_hours_df, od_data_df], axis=1)
    
    # Reorder columns
    od_column_names = [col for col in final_df.columns if 'OD' in col]
    final_df = pd.concat([final_df[['time_hours']], final_df[od_column_names]], axis=1)
    
    # Write the combined DataFrame to a new Excel file
    final_df.to_excel(output_file, index=False)
    print(f"Columns have been fused and reordered and saved to {output_file}")




# Usage
file_path = r'gcplyr-dat-sorted.xlsx'  # Replace with your input file path
output_path = 'output.xlsx'  # Replace with your desired output file path
fuse_and_reorder_columns(file_path, output_path)

"""


import pandas as pd
import numpy as np

# Function to determine if two columns are similar within the given tolerance
def are_columns_similar(col1, col2, tolerance=0.02):
    return np.all(np.abs(col1 - col2) <= tolerance)

# Function to process the Excel file
def process_excel_file(input_file, output_file, tolerance=0.02):
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Ensure 'time(h)' is a column in the DataFrame
    if 'time (h)' not in df.columns:
        raise ValueError("'time(h)' column not found in the input file.")
    
    # Separate 'time(h)' column from the rest
    time_column = df[['time (h)']]
    data_columns = df.drop(columns='time(h)')
    
    # List to keep track of processed columns
    processed_columns = set()
    
    # Create a dictionary to hold columns for each sheet
    sheets = {}
    
    # Iterate over columns in the DataFrame
    for i, col1 in enumerate(data_columns.columns):
        if col1 in processed_columns:
            continue
        
        # Create a new sheet for the current column
        sheet_name = f"Sheet_{len(sheets) + 1}"
        sheets[sheet_name] = ['time (h)'] + [col1]
        
        # Mark the current column as processed
        processed_columns.add(col1)
        
        # Compare with all other columns to find similar ones
        for j, col2 in enumerate(data_columns.columns):
            if i != j and col2 not in processed_columns:
                if are_columns_similar(data_columns[col1], data_columns[col2], tolerance):
                    sheets[sheet_name].append(col2)
                    processed_columns.add(col2)
    
    # Write the results to a new Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, columns in sheets.items():
            df_to_save = df[['time (h)'] + columns]
            df_to_save.to_excel(writer, sheet_name=sheet_name, index=False)



# Example usage
input_file = 'gcpllyr-predat.xlsx'  # Path to the input Excel file
output_file = 'your_output_file.xlsx'  # Path to the output Excel file

process_excel_file(input_file, output_file)
