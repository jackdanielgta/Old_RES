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

# Function to choose income
def choose_income(row):
    return row["MHIA21"] if 1 < row["MHIA21"] <= 250000 else row["AHIA21"]

# Choose income using the provided function
sold['INCOME'] = sold.apply(choose_income, axis=1)

# Create bins for INCOME every 10k
income_bins = list(range(0, 260000, 10000)) + [float('inf')]
income_labels = [f'{i/1000:.0f}k-{(i+10000)/1000:.0f}k' for i in range(0, 250000, 10000)] + ['250k+']
sold['INCOME_BIN'] = pd.cut(sold['INCOME'], bins=income_bins, labels=income_labels, include_lowest=True)

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

# Group by INCOME_BIN and calculate statistics
grouped = sold.groupby('INCOME_BIN')['P/SQFT'].apply(median_top_bottom_60).reset_index()

# Reshape the data from long to wide format
grouped_wide = grouped.pivot(index='INCOME_BIN', columns='level_1', values='P/SQFT').reset_index()

# Sort by INCOME_BIN
grouped_wide = grouped_wide.sort_values('INCOME_BIN')

# Create the candlestick chart
fig, ax = plt.subplots(figsize=(20, 8))  # Increased figure size for better readability

# Plot the 60% range "candles" in green, slightly offset to the right
ax.vlines([x + 0.1 for x in range(len(grouped_wide))], grouped_wide['bottom_60_median'], grouped_wide['top_60_median'], color='green', linewidth=2, label='60% Range')

# Plot the IQR in black
ax.vlines(range(len(grouped_wide)), grouped_wide['q1'], grouped_wide['q3'], color='black', linewidth=4, label='IQR')

# Plot the median as a horizontal line with a red color and increased width
ax.hlines(grouped_wide['median'], [x-0.25 for x in range(len(grouped_wide))], [x+0.25 for x in range(len(grouped_wide))], color='red', linewidth=2, label='Median')

# Customize the plot
plt.title('Price per Square Foot Distribution by Income (Sales from 2023-2024)')
plt.xlabel('Income Range')
plt.ylabel('Price per Square Foot ($)')
plt.xticks(range(len(grouped_wide)), grouped_wide['INCOME_BIN'], rotation=90)

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()

# Print the data
print(grouped_wide)