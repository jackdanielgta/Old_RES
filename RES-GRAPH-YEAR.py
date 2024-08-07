import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')

# Convert CLOSEDATE to datetime and filter for 2023 and 2024
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold = sold[sold['CLOSEDATE'].dt.year.isin([2023, 2024])]

# Calculate P/SQFT
sold['P/SQFT'] = sold['CLOSEPRICE'] / sold['ABGSQFT']

# Filter data from 1700 onwards
sold_filtered = sold[sold['YEARBUILT'] >= 1700].copy()

# Create 5-year bins
sold_filtered['YearBin'] = pd.cut(sold_filtered['YEARBUILT'], 
                                  bins=range(1700, int(sold_filtered['YEARBUILT'].max()) + 6, 5),
                                  labels=range(1702, int(sold_filtered['YEARBUILT'].max()) + 3, 5))

# Function to calculate IQR statistics and count
def calculate_iqr_and_count(x):
    return pd.Series({
        'median': x.median(),
        'q1': x.quantile(0.25),
        'q3': x.quantile(0.75),
        'count': x.count()
    })

# Group by the bins and calculate statistics
grouped = sold_filtered.groupby('YearBin')['P/SQFT'].apply(calculate_iqr_and_count).reset_index()

# Reshape the data from long to wide format
grouped_wide = grouped.pivot(index='YearBin', columns='level_1', values='P/SQFT').reset_index()

# Convert YearBin to numeric
grouped_wide['YearBin'] = grouped_wide['YearBin'].astype(int)

# Filter for ranges with count >= 5
grouped_wide = grouped_wide[grouped_wide['count'] >= 5]

# Create the IQR chart
fig, ax = plt.subplots(figsize=(20, 8))

# Plot the IQR as vertical lines
ax.vlines(grouped_wide['YearBin'], grouped_wide['q1'], grouped_wide['q3'], color='blue', linewidth=2, label='IQR')

# Plot the median as points
ax.scatter(grouped_wide['YearBin'], grouped_wide['median'], color='red', s=30, zorder=3, label='Median')

# Customize the plot
plt.title('Price per Square Foot IQR by 5-Year Intervals (1700 onwards, Sales from 2023-2024, Count >= 5)')
plt.xlabel('Year Built (5-year intervals)')
plt.ylabel('Price per Square Foot ($)')
plt.xticks(rotation=90)

# Show every 10th x-tick to avoid overcrowding
ax.xaxis.set_major_locator(plt.MultipleLocator(10))

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# Print the data
print(grouped_wide)