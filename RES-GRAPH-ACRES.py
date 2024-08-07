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

# Filter out any negative or zero ACRES values
sold_filtered = sold[sold['ACRES'] > 0].copy()

# Create bins for ACRES
sold_filtered['AcreBin'] = pd.cut(sold_filtered['ACRES'], 
                                  bins=[0, 0.25, 0.5, 1, 2, 5, 10, 20, 50, 100, float('inf')],
                                  labels=['0-0.25', '0.25-0.5', '0.5-1', '1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100+'])

# Function to calculate IQR statistics and count
def calculate_iqr_and_count(x):
    return pd.Series({
        'median': x.median(),
        'q1': x.quantile(0.25),
        'q3': x.quantile(0.75),
        'count': x.count()
    })

# Group by the bins and calculate statistics
grouped = sold_filtered.groupby('AcreBin')['P/SQFT'].apply(calculate_iqr_and_count).reset_index()

# Reshape the data from long to wide format
grouped_wide = grouped.pivot(index='AcreBin', columns='level_1', values='P/SQFT').reset_index()

# Filter for ranges with count >= 5
grouped_wide = grouped_wide[grouped_wide['count'] >= 5]

# Create the IQR chart
fig, ax = plt.subplots(figsize=(20, 8))

# Plot the IQR as vertical lines
ax.vlines(range(len(grouped_wide)), grouped_wide['q1'], grouped_wide['q3'], color='blue', linewidth=2, label='IQR')

# Plot the median as points
ax.scatter(range(len(grouped_wide)), grouped_wide['median'], color='red', s=30, zorder=3, label='Median')

# Customize the plot
plt.title('Price per Square Foot IQR by Acre Range (Sales from 2023-2024, Count >= 5)')
plt.xlabel('Acres')
plt.ylabel('Price per Square Foot ($)')
plt.xticks(range(len(grouped_wide)), grouped_wide['AcreBin'], rotation=45, ha='right')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
plt.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()

# Print the data
print(grouped_wide)