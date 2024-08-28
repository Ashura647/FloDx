import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Provided "blank" data
time_points = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24])
blank_od600 = np.array([0.0, 0.2, 1.5, 2.0, 2.1, 2.2, 2.2, 2.2, 2.1, 2.1, 2.1, 2.1])

# Number of synthetic curves to generate
num_curves = 100

# Time points every 30 minutes up to 23 hours
new_time_points = np.arange(0, 23.5, 0.5)

# Define different variations
variations = {
    'Low_Variation': (0.05, 0.05),    # Low noise for original
    'Medium_Low_Variation': (0.05, 0.1), # Medium-low noise
    'Medium_Variation': (0.1, 0.1),    # Medium noise
    'High_Variation': (0.1, 0.15),    # High noise
    'Extreme_Variation': (0.15, 0.2)   # Extreme high noise
}

# Create DataFrames for each variation
data_frames = {}

for name, (original_noise_std, high_noise_std) in variations.items():
    # Data structures to hold the synthetic data
    synthetic_data = {}

    for i in range(num_curves):
        # Generate synthetic curve
        noise = np.random.normal(0, original_noise_std, len(time_points))
        synthetic_curve = blank_od600 + noise
        synthetic_curve = np.clip(synthetic_curve, 0, None)  # Ensure no negative OD600 values
        synthetic_data[f'Curve_{i+1}'] = synthetic_curve

    # Interpolate synthetic curves at 30-minute intervals
    interpolated_data = {}
    for curve in synthetic_data:
        interp_func = interp1d(time_points, synthetic_data[curve], kind='linear')
        interpolated_data[curve] = interp_func(new_time_points)

    # Create DataFrame for the current variation
    df_variation = pd.DataFrame(interpolated_data, index=new_time_points)
    df_variation.index.name = 'Time (hours)'

    # Add DataFrame to the dictionary
    data_frames[name] = df_variation

# Save all datasets to an Excel file with separate sheets
with pd.ExcelWriter('synthetic_s_aureus_growth_curves.xlsx', mode='w') as writer:
    for name, df in data_frames.items():
        df.to_excel(writer, sheet_name=name)

print("Synthetic growth curves with different variations saved to 'synthetic_s_aureus_growth_curves.xlsx'.")
