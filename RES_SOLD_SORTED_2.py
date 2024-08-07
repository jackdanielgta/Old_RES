import pandas as pd
import numpy as np

# Load the data
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')

# Select relevant columns
sold = sold[["MLS1", "GEOID", "PROPTYPE", "CLOSEPRICE", "CLOSEDATE", "CITY", "DIRECT", "STYLE", "PROPTAX"]]

# Update 'PROPTYPE' based on 'STYLE'
sold.loc[sold['STYLE'].str.contains('Townhouse', na=False) & (sold['PROPTYPE'] == 'CO'), 'PROPTYPE'] = 'TH'
sold.loc[sold['STYLE'].str.contains('Mobile Home', na=False) & (sold['PROPTYPE'] == 'SF'), 'PROPTYPE'] = 'MH'
sold.loc[sold['STYLE'].str.contains('Cape Cod', na=False) & (sold['PROPTYPE'] == 'SF'), 'PROPTYPE'] = 'SC'

# Convert 'CLOSEDATE' to datetime format and extract the year
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold['CLOSEYEAR'] = sold['CLOSEDATE'].dt.year.fillna(0).astype(int)

# Define tax bins and labels
tax_bins = [
    -np.inf, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
    5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000, 10500, 11000, 11500, 12000, 12500,
    13000, 13500, 14000, 14500, 15000, 15500, 16000, 16500, 17000, 17500, 18000, 18500, 19000, 19500, 20000, 21000, 22000,
    23000, 24000, 25000, np.inf
]

tax_bin_labels = [
    '<500', '500-1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000',
    '3000-3500', '3500-4000', '4000-4500', '4500-5000', '5000-5500',
    '5500-6000', '6000-6500', '6500-7000', '7000-7500', '7500-8000',
    '8000-8500', '8500-9000', '9000-9500', '9500-10000', '10000-10500',
    '10500-11000', '11000-11500', '11500-12000', '12000-12500', '12500-13000',
    '13000-13500', '13500-14000', '14000-14500', '14500-15000', '15000-15500',
    '15500-16000', '16000-16500', '16500-17000', '17000-17500', '17500-18000',
    '18000-18500', '18500-19000', '19000-19500', '19500-20000', '20000-21000',
    '21000-22000', '22000-23000', '23000-24000', '24000-25000', ">25000"
]

# Filter out properties with taxes in the specified range
excluded_values = [-np.inf, np.inf, 999999999, 99999999, 9999999, 999999, 99999, 0, 1, 100000000]
sold = sold[~sold['PROPTAX'].isin(excluded_values)]

# Create a new column 'TAX_BIN' with the binned data
sold['TAX_BIN'] = pd.cut(sold['PROPTAX'], bins=tax_bins, labels=tax_bin_labels).astype(str)

# Sort the DataFrame by the required columns
sold = sold.sort_values(by=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN'])

# Group by 'TAX_BIN' and include additional columns using first()
grouped = sold.groupby(['CLOSEYEAR', 'CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN'], observed=True).agg(
    median_closeprice=('CLOSEPRICE', 'median'),
    count=('CLOSEPRICE', 'size'),
).reset_index()

# Filter groups where the count is at least 5
filtered = grouped[grouped['count'] >= 5].copy()

# Convert TAX_BIN to ordered categorical type for proper sorting
filtered['TAX_BIN'] = pd.Categorical(filtered['TAX_BIN'], categories=tax_bin_labels, ordered=True)

# Sort the filtered DataFrame by the grouping columns and CLOSEYEAR
filtered = filtered.sort_values(by=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'CLOSEYEAR'])

# Save the updated dataframe to a new CSV file
filtered.to_csv(r'C:\cctaddr\RES_SOLD_SORTED_UPDATED.csv', index=False)

print(f"Updated file saved to {r'C:\cctaddr\RES_SOLD_SORTED_UPDATED.csv'}")
