import pandas as pd
from datetime import datetime

# Read the CSV files
sold = pd.read_csv(r"C:\Users\jackd\Desktop\Python RE\RES_SOLD_OFFICIAL_11-25-2021_05-17-2024.csv", low_memory=False)
land = pd.read_csv(r"C:\Users\jackd\Downloads\Land_2024-05-15.csv", low_memory=False)

income_spread = 5000

# Set appropriate columns for land and sold dataframes, include 'DIRECT' and 'STYLE'
land = land[["MLS1", "CITY", "GEOID", "ACRES", "YEARBUILT", "WTRFRONT", "CURRPRICE", "MHIA21", "AHIA21"]]
land.columns = ["MLS1", "CITY", "GEOID", "ACRES", "L_YEARBUILT", "WTRFRONT", "CURRPRICE", "MHIA21", "AHIA21"]
sold = sold[["CITY", "GEOID", "PROPTYPE", "ACRES", "ABGSQFT", "CLOSEDATE", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE", "MHIA21", "AHIA21"]]
sold.columns = ["CITY", "GEOID", "PROPTYPE", "ACRES", "ABGSQFT", "CLOSEDATE", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "S_STYLE", "MHIA21", "AHIA21"]

# NEW*****
sold["P_ABGSQFT"] = sold["CLOSEPRICE"] / sold["ABGSQFT"]

def get_year(date):
    date = pd.Timestamp(date)
    return f"{date.year}"

# Create the 'CLOSEYEAR' column and fill it with the formatted year
sold['CLOSEYEAR'] = sold['CLOSEDATE'].apply(get_year)

age_bins = [0, 15, 30, 50, 100, float('inf')]
size_bins = [0, 1000, 2000, 3000, 4000, float('inf')]

sold["AGE"] = pd.cut(sold["S_YEARBUILT"], bins=age_bins, labels=False)
sold["SIZE"] = pd.cut(sold["ABGSQFT"], bins=size_bins, labels=False)

# Debugging step: Print the columns to ensure 'AGE' and 'SIZE' exist
print("Columns in 'sold' DataFrame after creating 'AGE' and 'SIZE':")
print(sold.columns)

# Calculate the median CLOSEPRICE per group and assign it to a new column
grouped_medians = sold.groupby(["GEOID", "PROPTYPE", "CLOSEYEAR", "AGE", "SIZE"], observed=True)["CLOSEPRICE"].median().reset_index()
grouped_medians.rename(columns={"CLOSEPRICE": "UPDATED_CLOSEPRICE"}, inplace=True)

# Merge back to the original sold dataframe
sold = sold.merge(grouped_medians, on=["GEOID", "PROPTYPE", "CLOSEYEAR", "AGE", "SIZE"], how="left")

# Calculate the percentage growth for each year against the current year for each group of the same GEOID and PROPTYPE
current_year = datetime.now().year
current_year_medians = grouped_medians[grouped_medians["CLOSEYEAR"] == str(current_year)]

grouped_medians = grouped_medians.merge(
    current_year_medians[["GEOID", "PROPTYPE", "UPDATED_CLOSEPRICE", "AGE", "SIZE"]],
    on=["GEOID", "PROPTYPE", "AGE", "SIZE"],
    suffixes=("", "_CURRENT"),
    how="left"
)

grouped_medians["PERCENTAGE_GROWTH"] = (
    (grouped_medians["UPDATED_CLOSEPRICE"] - grouped_medians["UPDATED_CLOSEPRICE_CURRENT"]) /
    grouped_medians["UPDATED_CLOSEPRICE_CURRENT"]
) * 100

# Select columns for output
updated_sold = grouped_medians[["GEOID", "PROPTYPE", "CLOSEYEAR", "AGE", "SIZE", "UPDATED_CLOSEPRICE", "PERCENTAGE_GROWTH"]]

# Debugging step: Print the columns to ensure 'AGE' and 'SIZE' exist in 'updated_sold'
print("Columns in 'updated_sold' DataFrame:")
print(updated_sold.columns)

final_sold = pd.merge(sold, updated_sold, on=["GEOID", "PROPTYPE", "CLOSEYEAR", "AGE", "SIZE"], how="left")

final_sold["UPDATED_SOLDPRICE"] = final_sold["CLOSEPRICE"] + (final_sold["PERCENTAGE_GROWTH"] / 100 * final_sold["CLOSEPRICE"])

print(final_sold)
final_sold.to_csv(r"C:\cctaddr\LAND_HIGHEST_COMP_SOLD.csv", index=False)
# NEW*****

# Define function to choose income
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]

# Apply function to calculate income
sold["S_INCOME"] = sold.apply(choose_income, axis=1)
land["L_INCOME"] = land.apply(choose_income, axis=1)

# Merge dataframes on columns 'CITY' and 'PROPTYPE'
merged_df = pd.merge(land, sold, on=["CITY"])

# Define the complex logic as per Google Sheets structure
def filter_logic(row):
    if row["WTRFRONT"] == "N":
        return (
            pd.notnull(row['ACRES_x']) and pd.notnull(row['ACRES_y']) and
            pd.notnull(row['S_INCOME']) and pd.notnull(row['L_INCOME']) and
            (row['ACRES_y'] <= 1.1 * row['ACRES_x']) and
            (row['L_INCOME'] - income_spread <= row['S_INCOME'] <= row['L_INCOME'] + income_spread) and
            (row["S_DIRECT"] == "N")
            )
    else:
        return (
            pd.notnull(row['ACRES_x']) and pd.notnull(row['ACRES_y']) and
            pd.notnull(row['S_INCOME']) and pd.notnull(row['L_INCOME']) and
            (0.9 * row['ACRES_x'] <= row['ACRES_y'] <= 1.1 * row['ACRES_x']) and
            (row['L_INCOME'] - income_spread <= row['S_INCOME'] <= row['L_INCOME'] + income_spread)
            )

# Apply filter logic
filtered_df = merged_df[merged_df.apply(filter_logic, axis=1)]

# Find the row with the maximum 'CLOSEPRICE' for each 'MLS1', keeping 'MLS_COMP'
max_indices = filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
max_values_df = filtered_df.loc[max_indices]

# Ensure 'ABGSQFT' is correctly included and use the correct column from land if necessary
max_values_df = filtered_df.loc[max_indices, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'ACRES_x', 'ABGSQFT']]
max_values_df.rename(columns={'ACRES_x': 'ACRES'}, inplace=True)

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
final_df.to_csv(r"C:\cctaddr\LAND_HIGHEST_COMP.csv", index=False)
