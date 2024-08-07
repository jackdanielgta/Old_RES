import pandas as pd
from datetime import datetime

# Read the CSV files
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')
listing = pd.read_csv(r"C:\Users\jackd\Desktop\RES-MF_LISTINGS.csv", low_memory=False, encoding='latin1')

# Set appropriate columns for listing and sold dataframes, include 'DIRECT' and 'STYLE'
listing = listing[["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", "YEARBUILT", "DIRECT", "STYLE", "CURRPRICE"]]
listing.columns = ["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", "L_YEARBUILT", "L_DIRECT", "L_STYLE", "CURRPRICE"]
sold = sold[["GEOID", "PROPTYPE", "ABGSQFT", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE"]]
sold.columns = ["GEOID", "PROPTYPE", "ABGSQFT", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "S_STYLE"]

# Merge dataframes on columns 'GEOID' and 'PROPTYPE'
merged_df = pd.merge(listing, sold, on=["GEOID", "PROPTYPE"])

# Define the current year and the threshold year
current_year = datetime.now().year
threshold_year = current_year - 15

# Define the complex logic as per Google Sheets structure
def filter_logic(row):
    if row["L_DIRECT"] == "N":
        if row['L_YEARBUILT'] < threshold_year:
            if row['L_STYLE'] == "Cape Cod":
                # All conditions + Cape Cod in both
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row["S_DIRECT"] == "N") &
                        (row['S_YEARBUILT'] < threshold_year) & 
                        (row['S_STYLE'] == "Cape Cod"))
            else:
                # All conditions but without Cape Cod style match
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row["S_DIRECT"] == "N") &
                        (row['S_YEARBUILT'] < threshold_year))
        else:
            if row['L_STYLE'] == "Cape Cod":
                # Just the size match + Cape Cod in both
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row["S_DIRECT"] == "N") &
                        (row['S_STYLE'] == "Cape Cod"))
            else:
                # Just the size match
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row["S_DIRECT"] == "N"))
    else:
        if row['L_YEARBUILT'] < threshold_year:
            if row['L_STYLE'] == "Cape Cod":
                # All conditions + Cape Cod in both
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row['S_YEARBUILT'] < threshold_year) & 
                        (row['S_STYLE'] == "Cape Cod"))
            else:
                # All conditions but without Cape Cod style match
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row['S_YEARBUILT'] < threshold_year))
        else:
            if row['L_STYLE'] == "Cape Cod":
                # Just the size match + Cape Cod in both
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']) &
                        (row['S_STYLE'] == "Cape Cod"))
            else:
                # Just the size match
                return ((row['ABGSQFT_y'] >= 0.9 * row['ABGSQFT_x']) & 
                        (row['ABGSQFT_y'] <= 1.1 * row['ABGSQFT_x']))

# Apply filter logic
filtered_df = merged_df[merged_df.apply(filter_logic, axis=1)]

# Find the row with the maximum 'CLOSEPRICE' for each 'MLS1', keeping 'MLS_COMP'
max_indices = filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
max_values_df = filtered_df.loc[max_indices]

# Ensure 'ABGSQFT' is correctly included and use the correct column from listing if necessary
max_values_df = filtered_df.loc[max_indices, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'ABGSQFT_x']]  # Change as necessary
max_values_df.rename(columns={'ABGSQFT_x': 'ABGSQFT'}, inplace=True)

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