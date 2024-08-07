import pandas as pd
from datetime import datetime

# Read the CSV files with specified encoding
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", encoding='ISO-8859-1', low_memory=False)
listing = pd.read_csv(r"C:\Users\jackd\Downloads\res-2024-05-11.csv", encoding='ISO-8859-1', low_memory=False)
land = pd.read_csv(r"C:\Users\jackd\Downloads\Land_2024-05-15.csv", encoding='ISO-8859-1', low_memory=False)

# Set appropriate columns for listing and sold dataframes, include 'DIRECT' and 'STYLE'
listing = listing[["MLS1", "CITY", "GEOID", "PROPTYPE", "ABGSQFT", "ACRES", "YEARBUILT", "DIRECT", "STYLE", "CURRPRICE", "MHIA21", "AHIA21", "PROPTAX"]]
listing.columns = ["MLS1", "CITY", "GEOID", "PROPTYPE", "L_ABGSQFT", "L_ACRES", "L_YEARBUILT", "L_DIRECT", "STYLE", "CURRPRICE", "MHIA21", "AHIA21", "L_PROPTAX"]
sold = sold[["CITY", "GEOID", "PROPTYPE", "ABGSQFT", "ACRES", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE", "MHIA21", "AHIA21", "PROPTAX", "CLOSEDATE"]]
sold.columns = ["CITY", "GEOID", "PROPTYPE", "S_ABGSQFT", "S_ACRES", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "STYLE", "MHIA21", "AHIA21", "S_PROPTAX", "CLOSEDATE"]

# Update 'PROPTYPE' based on 'STYLE'
sold.loc[sold['STYLE'].str.contains('Townhouse', na=False), 'PROPTYPE'] = 'TH'
sold.loc[sold['STYLE'].str.contains('Mobile Home', na=False), 'PROPTYPE'] = 'MH'
sold.loc[sold['STYLE'].str.contains('Cape Cod', na=False), 'PROPTYPE'] = 'SC'
listing.loc[listing['STYLE'].str.contains('Townhouse', na=False), 'PROPTYPE'] = 'TH'
listing.loc[listing['STYLE'].str.contains('Mobile Home', na=False), 'PROPTYPE'] = 'MH'
listing.loc[listing['STYLE'].str.contains('Cape Cod', na=False), 'PROPTYPE'] = 'SC'

# Convert 'CLOSEDATE' to datetime format and extract the year
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold['CLOSEYEAR'] = sold['CLOSEDATE'].dt.year

# Filter for 'CLOSEYEAR' 2023 or 2024
sold = sold[sold['CLOSEYEAR'].isin([2023, 2024])]

# Merge dataframes on columns 'CITY' and 'PROPTYPE'
merged_df = pd.merge(listing, sold, on=["GEOID", "PROPTYPE"])

# Define the current year and the threshold year
current_year = datetime.now().year
threshold_year = current_year - 15

# Define function to choose income
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]

# Apply function to calculate income
sold["S_INCOME"] = sold.apply(choose_income, axis=1)
listing["L_INCOME"] = listing.apply(choose_income, axis=1)

def filter_logic(row):
    criteria_match = ((row['S_ABGSQFT'] <= 1.1 * row['L_ABGSQFT']) & 
                      (row['S_ACRES'] <= 1.1 * row['L_ACRES']))
    
    if row["L_DIRECT"] == "N":
        if row['L_YEARBUILT'] < threshold_year:
            # All conditions including criteria match and year built match
            return (criteria_match &
                    (row["S_DIRECT"] == "N") &
                    (row['S_YEARBUILT'] < threshold_year))
        else:
            # Only criteria match
            return (criteria_match &
                    (row["S_DIRECT"] == "N"))
    else:
        if row['L_YEARBUILT'] < threshold_year:
            # All conditions including criteria match and year built match
            return (criteria_match &
                    (row['S_YEARBUILT'] < threshold_year))
        else:
            # Only criteria match
            return criteria_match

# Apply filter logic
filtered_df = merged_df[merged_df.apply(filter_logic, axis=1)]

# Find the row with the maximum 'CLOSEPRICE' for each 'MLS1', keeping 'MLS_COMP'
max_indices = filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
max_values_df = filtered_df.loc[max_indices]

# Ensure 'ABGSQFT' is correctly included and use the correct column from listing if necessary
max_values_df = filtered_df.loc[max_indices, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'L_ABGSQFT']]  # Change as necessary
max_values_df.rename(columns={'L_ABGSQFT': 'ABGSQFT'}, inplace=True)

# Calculate the 'PROFIT' as the difference between 'CLOSEPRICE' and 'CURRPRICE'
max_values_df['PROFIT'] = max_values_df['CLOSEPRICE'] - max_values_df['CURRPRICE']

# Calculate 'CASH_ON_CASH'
investment = 50000 * (max_values_df['ABGSQFT'] / 1000)
max_values_df['CASH_ON_CASH'] = (max_values_df['PROFIT'] - investment) / (max_values_df['CURRPRICE'] * 0.2)

# Select desired columns
final_df = max_values_df[['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH']]

# Reset index to clean up the DataFrame
final_df.reset_index(drop=True, inplace=True)

print(final_df)
final_df.to_csv(r"C:\cctaddr\RES_HIGHEST_COMP.csv", index=False)