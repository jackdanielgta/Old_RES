import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import percentileofscore

rental = pd.read_csv(r"C:\Users\jackd\Desktop\Python RE\RENTAL_DATA_OFFICIAL_2024_05_07.csv", low_memory=False)

# Select relevant columns
rental = rental[["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CLOSEDATE", "CITY", "COUNTY", "MHIA21", "AHIA21", "ACRES", "STYLE", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "BATHSTOTAL"]]

# Update 'PROPTYPE' based on 'STYLE'
rental.loc[rental['STYLE'].str.contains('Townhouse', na=False), 'PROPTYPE'] = 'TH'
rental.loc[rental['STYLE'].str.contains('Mobile Home', na=False), 'PROPTYPE'] = 'MH'
rental.loc[rental['STYLE'].str.contains('Cape Cod', na=False), 'PROPTYPE'] = 'SFCC'

# Convert 'CLOSEDATE' to datetime format and extract the year
rental['CLOSEDATE'] = pd.to_datetime(rental['CLOSEDATE'], errors='coerce')
rental['CLOSEYEAR'] = rental['CLOSEDATE'].dt.year

# Create P_SQFT
rental['P_SQFT'] = rental['CLOSEPRICE'] / rental['ABGSQFT']

# Filter for 'CLOSEYEAR' 2023 or 2024
rental = rental[rental['CLOSEYEAR'].isin([2023, 2024])]

# Define function to choose income
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]
    
rental["INCOME"] = rental.apply(choose_income, axis=1)

# Define bins for 'INCOME'
income_bins = [-np.inf, 10000, 20000, 30000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000, 80000, 90000, 100000, 150000, 200000, 250000, np.inf]
income_bin_labels = ['<10000', '10000-20000', '20000-30000', '30000-40000', '40000-45000', '45000-50000', '50000-55000', '55000-60000', '60000-65000', '65000-70000', '70000-75000', '75000-80000', '80000-90000', '90000-100000', '100000-150000', '150000-200000', '200000-250000', '>250000']

# Create a new column 'INCOME_BIN' with the binned data
rental['INCOME_BIN'] = pd.cut(rental['INCOME'], bins=income_bins, labels=income_bin_labels)

# Group by 'PROPTYPE', 'DIRECT', 'SQFT_BIN', and 'INCOME_BIN' and calculate the median 'CLOSEPRICE' and count of each group
grouped = rental.groupby(['CLOSEYEAR', 'COUNTY', 'CITY', 'PROPTYPE', 'DIRECT', 'INCOME_BIN']).agg(
    median_closeprice=('P_SQFT', 'median'),
    count=('P_SQFT', 'size')
).reset_index()

# Filter groups where the count is at least 8
filtered = grouped[grouped['count'] >= 5]

# Print the result
print(filtered)

# Save the result to a CSV file
filtered.to_csv(r'C:\cctaddr\RES_RENTAL_SORTED.csv', index=False)