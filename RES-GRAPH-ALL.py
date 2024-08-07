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

# Define income bins
income_bins = [0, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 160000, 210000, 240000, 250000, float('inf')]
income_labels = ['<30k', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100', '100-160', '160-210', '210-240', '240-250', '>250']

# Assign income bins
sold['INCOME_BIN'] = pd.cut(sold['INCOME'], bins=income_bins, labels=income_labels, include_lowest=True)

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
grouped = sold_filtered.groupby(['YearBin', 'INCOME_BIN'])['P/SQFT'].apply(median_top_bottom_60).reset_index()

# Reshape the data from long to wide format
grouped_wide = grouped.pivot(index=['YearBin', 'INCOME_BIN'], columns='level_2', values='P/SQFT').reset_index()

# Convert YearBin to numeric
grouped_wide['YearBin'] = grouped_wide['YearBin'].astype(int)

# Create the candlestick chart
fig, ax = plt.subplots(figsize=(20, 10))

# Define colors for income bins
color_map = plt.cm.get_cmap('viridis')
colors = color_map(np.linspace(0, 1, len(income_labels)))

# Plot data for each income bin
for idx, income_bin in enumerate(income_labels):
    data = grouped_wide[grouped_wide['INCOME_BIN'] == income_bin]
    
    # Plot the 60% range "candles"
    ax.vlines(data['YearBin'] + 0.1*idx, data['bottom_60_median'], data['top_60_median'], color=colors[idx], linewidth=2, alpha=0.7)
    
    # Plot the IQR
    ax.vlines(data['YearBin'] + 0.1*idx, data['q1'], data['q3'], color=colors[idx], linewidth=4, alpha=0.7)
    
    # Plot the median as a horizontal line
    ax.hlines(data['median'], data['YearBin'] + 0.1*idx - 0.05, data['YearBin'] + 0.1*idx + 0.05, color=colors[idx], linewidth=2, alpha=0.7)

# Customize the plot
plt.title('Price per Square Foot Distribution by 5-Year Intervals and Income Bins (1900 onwards, Sales from 2023-2024)')
plt.xlabel('Year Built (5-year intervals)')
plt.ylabel('Price per Square Foot ($)')
plt.xticks(rotation=90)

# Show every 5th x-tick to avoid overcrowding
ax.xaxis.set_major_locator(plt.MultipleLocator(5))

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
legend_elements = [plt.Line2D([0], [0], color=colors[i], lw=4, label=income_bin) for i, income_bin in enumerate(income_labels)]
plt.legend(handles=legend_elements, title='Income Bins', loc='upper left', bbox_to_anchor=(1, 1))

# Adjust layout to prevent cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print the data
print(grouped_wide)