import pandas as pd
from datetime import datetime

# Read the CSV files with a specified encoding
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", encoding='latin1', low_memory=False)
land = pd.read_csv(r"C:\cctaddr\Landdd.csv", encoding='latin1', low_memory=False)

income_spread = 5000
current_year = datetime.now().year

# Set appropriate columns for land and sold dataframes, include 'DIRECT' and 'STYLE'
land = land[["MLS1", "CITY", "GEOID", "ACRES", "YEARBUILT", "DIRECT", "CURRPRICE", "MHIA21", "AHIA21"]]
land.columns = ["MLS1", "CITY", "GEOID", "ACRES", "L_YEARBUILT", "L_DIRECT", "CURRPRICE", "MHIA21", "AHIA21"]
sold = sold[["CITY", "GEOID", "PROPTYPE", "ACRES", "ABGSQFT", "CLOSEDATE", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE", "MHIA21", "AHIA21"]]
sold.columns = ["CITY", "GEOID", "PROPTYPE", "ACRES", "ABGSQFT", "CLOSEDATE", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "S_STYLE", "MHIA21", "AHIA21"]

# Select rows where both 'S_YEARBUILT' and 'ABGSQFT' are not NaN
not_missing = ~sold['S_YEARBUILT'].isna() & ~sold['ABGSQFT'].isna()
sold = sold[not_missing]

# Merge dataframes on columns 'CITY' and 'PROPTYPE'
merged_df = pd.merge(land, sold, on=["GEOID"])

# Define the complex logic as per Google Sheets structure
def filter_logic(row):
    if row["L_DIRECT"] == "N":
        return (
            pd.notnull(row['ACRES_x']) and pd.notnull(row['ACRES_y']) and
            (row['ACRES_y'] <= 1.1 * row['ACRES_x']) and
            (row["S_DIRECT"] == "N")
            )
    else:
        return (
            pd.notnull(row['ACRES_x']) and pd.notnull(row['ACRES_y']) and
            (0.9 * row['ACRES_x'] <= row['ACRES_y'] <= 1.1 * row['ACRES_x'])
            )

# Apply filter logic
filtered_df = merged_df[merged_df.apply(filter_logic, axis=1)]

# Filter out rows with NaN in 'UPDATED_SOLDPRICE' before finding the row with the maximum 'UPDATED_SOLDPRICE' for each 'MLS1'
filtered_df = filtered_df.dropna(subset=["CLOSEPRICE"])

# Find the row with the maximum 'UPDATED_SOLDPRICE' for each 'MLS1', keeping 'MLS_COMP'
max_indices = filtered_df.groupby('MLS1')["CLOSEPRICE"].idxmax()

# Ensure 'ABGSQFT' is correctly included and use the correct column from land if necessary
max_values_df = filtered_df.loc[max_indices, ['MLS1', "CLOSEPRICE", 'MLS_COMP', 'CURRPRICE', 'ACRES_x', 'ABGSQFT']]
max_values_df.rename(columns={'ACRES_x': 'ACRES'}, inplace=True)

# Calculate the 'PROFIT' as the difference between 'UPDATED_SOLDPRICE' and 'CURRPRICE'
max_values_df['PROFIT'] = max_values_df["CLOSEPRICE"] - max_values_df['CURRPRICE']

# Calculate 'CASH_ON_CASH'
investment = 50000 * (max_values_df['ABGSQFT'] / 1000)
max_values_df['CASH_ON_CASH'] = (max_values_df['PROFIT'] - investment) / (max_values_df['CURRPRICE'] * 0.2)

# Select desired columns
final_df = max_values_df[['MLS1', "CLOSEPRICE", 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH']]

# Reset index to clean up the DataFrame
final_df.reset_index(drop=True, inplace=True)

print(final_df)
final_df.to_csv(r"C:\cctaddr\LAND_HIGHEST_COMP.csv", index=False)