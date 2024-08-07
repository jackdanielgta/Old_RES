import pandas as pd
from datetime import datetime

# Read the CSV files
sold = pd.read_csv(r"C:\Users\jackd\Downloads\RES_SOLD_RESCUE.csv", low_memory=False)
land = pd.read_csv(r"C:\Users\jackd\Downloads\Land_2024-05-15.csv", low_memory=False)

income_spread = 5000
current_year = datetime.now().year

# Set appropriate columns for land and sold dataframes, include 'DIRECT' and 'STYLE'
land = land[["MLS1", "CITY", "GEOID", "ACRES", "YEARBUILT", "WTRFRONT", "CURRPRICE", "MHIA21", "AHIA21"]]
land.columns = ["MLS1", "CITY", "GEOID", "ACRES", "L_YEARBUILT", "WTRFRONT", "CURRPRICE", "MHIA21", "AHIA21"]
sold = sold[["CITY", "GEOID", "PROPTYPE", "ACRES", "ABGSQFT", "CLOSEDATE", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE", "MHIA21", "AHIA21"]]
sold.columns = ["CITY", "GEOID", "PROPTYPE", "ACRES", "ABGSQFT", "CLOSEDATE", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "S_STYLE", "MHIA21", "AHIA21"]

# Select rows where both 'S_YEARBUILT' and 'ABGSQFT' are not NaN
not_missing = ~sold['S_YEARBUILT'].isna() & ~sold['ABGSQFT'].isna()
sold = sold[not_missing]

def get_year(date):
    date = pd.Timestamp(date)
    return f"{date.year}"

# Create the 'CLOSEYEAR' column and fill it with the formatted year
sold['CLOSEYEAR'] = sold['CLOSEDATE'].apply(get_year)

# Define function to choose income
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]

# Apply function to calculate income
sold["S_INCOME"] = sold.apply(choose_income, axis=1)
land["L_INCOME"] = land.apply(choose_income, axis=1)

def age_round(value):
    rounded_value = (round((current_year - value) / 10)) * 10
    return min(rounded_value, 20)

def sqft_round(value):
    rounded_value = (round(value / 2000)) * 2000
    return min(rounded_value, 4000)

sold['ROUNDED_AGE'] = sold['S_YEARBUILT'].apply(age_round)
sold['ROUNDED_SQFT'] = sold["ABGSQFT"].apply(sqft_round)

sold['P_SQFT'] = sold['CLOSEPRICE'] / sold['ABGSQFT']

# Calculate the median P_ABGSQFT per group and assign it to a new column
grouped_medians = sold.groupby(["GEOID", "PROPTYPE", "CLOSEYEAR", 'ROUNDED_AGE', 'ROUNDED_SQFT'])['P_SQFT'].median().reset_index()
grouped_medians.rename(columns={'P_SQFT': "MEDIAN_P_SQFT"}, inplace=True)

print(grouped_medians.head(10))
grouped_medians.to_csv(r"C:\cctaddr\landtest.csv", index=False)

# Calculate the percentage growth for each year against the current year for each group of the same GEOID and PROPTYPE
current_year_medians = grouped_medians[grouped_medians["CLOSEYEAR"] == str(current_year)]

grouped_medians = grouped_medians.merge(
    current_year_medians[["GEOID", "PROPTYPE", 'ROUNDED_AGE', 'ROUNDED_SQFT', "MEDIAN_P_SQFT"]],
    on=["GEOID", "PROPTYPE", 'ROUNDED_AGE', 'ROUNDED_SQFT'],
    suffixes=("", "_CURRENT"),
    how="left"
)

grouped_medians["PERCENTAGE_GROWTH"] = (
    (grouped_medians["MEDIAN_P_SQFT_CURRENT"] - grouped_medians["MEDIAN_P_SQFT"]) /
    grouped_medians["MEDIAN_P_SQFT"]
)

# Select columns for output
updated_sold = grouped_medians[["GEOID", "PROPTYPE", "CLOSEYEAR", "MEDIAN_P_SQFT", 'ROUNDED_AGE', 'ROUNDED_SQFT', "PERCENTAGE_GROWTH"]]

final_sold = pd.merge(sold, updated_sold, on=["GEOID", "PROPTYPE", "CLOSEYEAR", 'ROUNDED_AGE', 'ROUNDED_SQFT'], how="left")

final_sold["UPDATED_SOLDPRICE"] = final_sold["CLOSEPRICE"] + (final_sold["PERCENTAGE_GROWTH"] * final_sold["CLOSEPRICE"])

print(final_sold)
final_sold.to_csv(r"C:\cctaddr\LAND_HIGHEST_COMP_SOLD.csv", index=False)

# Merge dataframes on columns 'CITY' and 'PROPTYPE'
merged_df = pd.merge(land, final_sold, on=["CITY"])

# Define the complex logic as per Google Sheets structure
def filter_logic(row):
    if row["WTRFRONT"] == "N":
        return (
            pd.notnull(row['ACRES_x']) and pd.notnull(row['ACRES_y']) and
            pd.notnull(row['S_INCOME']) and pd.notnull(row['L_INCOME']) and
            (row['ACRES_y'] <= 1.1 * row['ACRES_x']) and
            (row['L_INCOME'] - income_spread <= row['S_INCOME'] <= row['L_INCOME'] + income_spread) &
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

# Filter out rows with NaN in 'UPDATED_SOLDPRICE' before finding the row with the maximum 'UPDATED_SOLDPRICE' for each 'MLS1'
filtered_df = filtered_df.dropna(subset=["UPDATED_SOLDPRICE"])

# Find the row with the maximum 'UPDATED_SOLDPRICE' for each 'MLS1', keeping 'MLS_COMP'
max_indices = filtered_df.groupby('MLS1')["UPDATED_SOLDPRICE"].idxmax()

# Ensure 'ABGSQFT' is correctly included and use the correct column from land if necessary
max_values_df = filtered_df.loc[max_indices, ['MLS1', "UPDATED_SOLDPRICE", 'MLS_COMP', 'CURRPRICE', 'ACRES_x', 'ABGSQFT']]
max_values_df.rename(columns={'ACRES_x': 'ACRES'}, inplace=True)

# Calculate the 'PROFIT' as the difference between 'UPDATED_SOLDPRICE' and 'CURRPRICE'
max_values_df['PROFIT'] = max_values_df["UPDATED_SOLDPRICE"] - max_values_df['CURRPRICE']

# Calculate 'CASH_ON_CASH'
investment = 50000 * (max_values_df['ABGSQFT'] / 1000)
max_values_df['CASH_ON_CASH'] = (max_values_df['PROFIT'] - investment) / (max_values_df['CURRPRICE'] * 0.2)

# Select desired columns
final_df = max_values_df[['MLS1', "UPDATED_SOLDPRICE", 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH']]

# Reset index to clean up the DataFrame
final_df.reset_index(drop=True, inplace=True)

print(final_df)
final_df.to_csv(r"C:\cctaddr\LAND_HIGHEST_COMP.csv", index=False)