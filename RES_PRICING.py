import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')

# Calculate P/SQFT
sold['P/SQFT'] = sold['CLOSEPRICE'] / sold['ABGSQFT']

# Filter data from 1900 onwards
sold_filtered = sold[sold['YEARBUILT'] >= 1900].copy()

# Create 5-year bins
sold_filtered['YearBin'] = pd.cut(sold_filtered['YEARBUILT'], 
                                  bins=range(1900, int(sold_filtered['YEARBUILT'].max()) + 6, 5),
                                  labels=range(1902, int(sold_filtered['YEARBUILT'].max()) + 3, 5))

# Function to calculate median of top and bottom 60%
def median_top_bottom_60(x):
    n = len(x)
    top_60 = x.nlargest(int(n * 0.6))
    bottom_60 = x.nsmallest(int(n * 0.6))
    return pd.Series({
        'median': x.median(),
        'q1': x.quantile(0.25),
        'q3': x.quantile(0.75),
        'top_60_median': top_60.median(),
        'bottom_60_median': bottom_60.median()
    })

# Group by the bins and calculate statistics
grouped = sold_filtered.groupby('YearBin')['P/SQFT'].apply(median_top_bottom_60).reset_index()

# Reshape the data from long to wide format
grouped_wide = grouped.pivot(index='YearBin', columns='level_1', values='P/SQFT').reset_index()

# Convert YearBin to numeric
grouped_wide['YearBin'] = grouped_wide['YearBin'].astype(int)

# Create the candlestick chart
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the "candles"
ax.vlines(grouped_wide['YearBin'], grouped_wide['bottom_60_median'], grouped_wide['top_60_median'], color='black', linewidth=1)
ax.vlines(grouped_wide['YearBin'], grouped_wide['q1'], grouped_wide['q3'], color='black', linewidth=4)

# Plot the median as a horizontal line with a different color and increased width
ax.hlines(grouped_wide['median'], grouped_wide['YearBin']-0.5, grouped_wide['YearBin']+0.5, color='red', linewidth=2)

# Customize the plot
plt.title('Price per Square Foot Distribution by 5-Year Intervals (1900 onwards)')
plt.xlabel('Year Built (5-year intervals)')
plt.ylabel('Price per Square Foot ($)')
plt.xticks(rotation=90)

# Show every 5th x-tick to avoid overcrowding
ax.xaxis.set_major_locator(plt.MultipleLocator(5))

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(['Median', 'IQR', '60% Range'], loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()