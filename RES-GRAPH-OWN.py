import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')

# Convert CLOSEDATE to datetime and filter for 2023 and 2024
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold = sold[sold['CLOSEDATE'].dt.year.isin([2023, 2024])]

# Calculate P/SQFT and OWNERSHIP_%
sold['P/SQFT'] = sold['CLOSEPRICE'] / sold['ABGSQFT']
sold['OWNERSHIP_%'] = sold['OWNOCCA21'] / sold['TOTHSGA21']

# Create bins for OWNERSHIP_%
sold['OWNERSHIP_BIN'] = pd.cut(sold['OWNERSHIP_%'], 
                               bins=np.arange(0, 1.05, 0.05), 
                               labels=[f'{i:.2f}-{i+0.05:.2f}' for i in np.arange(0, 1, 0.05)])

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

# Group by OWNERSHIP_BIN and calculate statistics
grouped = sold.groupby('OWNERSHIP_BIN')['P/SQFT'].apply(median_top_bottom_60).reset_index()

# Reshape the data from long to wide format
grouped_wide = grouped.pivot(index='OWNERSHIP_BIN', columns='level_1', values='P/SQFT').reset_index()

# Sort by OWNERSHIP_BIN
grouped_wide = grouped_wide.sort_values('OWNERSHIP_BIN')

# Create the candlestick chart
fig, ax = plt.subplots(figsize=(15, 6))

# Plot the 60% range "candles" in green, slightly offset to the right
ax.vlines([x + 0.1 for x in range(len(grouped_wide))], grouped_wide['bottom_60_median'], grouped_wide['top_60_median'], color='green', linewidth=2, label='60% Range')

# Plot the IQR in black
ax.vlines(range(len(grouped_wide)), grouped_wide['q1'], grouped_wide['q3'], color='black', linewidth=4, label='IQR')

# Plot the median as a horizontal line with a red color and increased width
ax.hlines(grouped_wide['median'], [x-0.25 for x in range(len(grouped_wide))], [x+0.25 for x in range(len(grouped_wide))], color='red', linewidth=2, label='Median')

# Customize the plot
plt.title('Price per Square Foot Distribution by Ownership Percentage (Sales from 2023-2024)')
plt.xlabel('Ownership Percentage')
plt.ylabel('Price per Square Foot ($)')
plt.xticks(range(len(grouped_wide)), grouped_wide['OWNERSHIP_BIN'], rotation=90)

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()