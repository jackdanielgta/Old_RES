import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import percentileofscore

income_spread = 5000
downpayment = .2
interest_rate = 0.0766
Amoritization = 30
Distance_Latitude = 41.0089370679199
Distance_Longitude = -73.6482077977137
year_age = 30

# Load CSV files
sold = pd.read_csv(r"C:\Users\jackd\Desktop\Python RE\RES_SOLD_SHORT_OFFICIAL_11-25-2021_05-17-2024.csv", low_memory=False)
listing = pd.read_csv(r"C:\Users\jackd\Downloads\RES-NEW-2024-06-28.csv", low_memory=False, encoding='latin1')
rental = pd.read_csv(r"C:\Users\jackd\Desktop\Python RE\RENTAL_DATA_OFFICIAL_2024_05_07.csv", low_memory=False)
land = pd.read_csv(r"C:\Users\jackd\Downloads\Land_2024-05-15.csv", low_memory=False)

# Select and rename columns
listing = listing[["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CURRPRICE", "CITY", "MHIA21", "AHIA21", "ACRES", "STYLE", "BEDSTOTAL", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "VACANTA21", "HHA21", "BATHSTOTAL", "TOTAL_UNIT"]]
sold = sold[["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CITY", "MHIA21", "AHIA21", "ACRES", "STYLE", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "BATHSTOTAL"]]
rental = rental[["MLS1", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CITY", "MHIA21", "AHIA21", "BEDSTOTAL", "BATHSTOTAL"]]

# Remove prefix from rental PROPTYPE
rental['PROPTYPE'] = rental['PROPTYPE'].str.replace('RN/', '')
#There are other rental 'PROPTYPE' that are not lined up with listing's

# Replace placeholder values and infinite values with NaN
placeholder_values = [999999999.00, 99999999.00, 9999999.00, 999999.00, 99999.00, 0, 1, 100000000]
sold['PROPTAX'].replace([np.inf, -np.inf] + placeholder_values, np.nan, inplace=True)

# Bin the TOT_SQFT into quartiles
sold['TOT_SQFT_BINNED'] = pd.qcut(sold['TOT_SQFT'], q=4, labels=False)

# Define a function to impute based on the median within each combined group
def impute_with_combined_group_median(df, target_column, group_columns):
    # Compute the median within each combined group
    group_medians = df.groupby(group_columns)[target_column].transform('median')
    # Fill NaN values with the group's median
    df[target_column].fillna(group_medians, inplace=True)

# Impute using the combined groups of GEOID and binned TOT_SQFT
impute_with_combined_group_median(sold, 'PROPTAX', ['GEOID', 'TOT_SQFT_BINNED'])

# Verify if there are any remaining NaN or infinite values
print(sold['PROPTAX'].isna().sum())
print(np.isinf(sold['PROPTAX']).sum())

# Drop the binned column if it's no longer needed
sold.drop(columns=['TOT_SQFT_BINNED'], inplace=True)

# Define function to choose income
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]

# Apply function to calculate income
sold["INCOME"] = sold.apply(choose_income, axis=1)
listing["INCOME"] = listing.apply(choose_income, axis=1)
rental["INCOME"] = rental.apply(choose_income, axis=1)

def calculate_carry_costs(df, price_col):
    conditions = (df['PROPTAX'] > 0) & (df['PROPTAX'] <= 99999)
    df['CARRY_COSTS'] = np.where(
        conditions,
        (df[price_col] / 1000) * 4.2 / 12 + df['HOA'] + (df['PROPTAX'] / 12),
        np.nan
    )

# Apply function to calculate carry_costs
calculate_carry_costs(sold, 'CLOSEPRICE')
calculate_carry_costs(listing, 'CURRPRICE')

listing["OWNERSHIP_%"] = listing["OWNOCCA21"] / listing["TOTHSGA21"]

listing["POP_DENSITY"] = (listing["TOTPOPA21"]/listing["ALAND"] * 0.386102)*1000

listing["MORT_PAY"] = (listing["CURRPRICE"] * (1 - downpayment)) * ((interest_rate / 12) * (1 + interest_rate / 12) ** (12 * Amoritization)) / ((1 + interest_rate / 12) ** (12 * Amoritization) - 1)

listing["COMPLETE_PAYMENT"] = listing["MORT_PAY"] + listing["CARRY_COSTS"]

listing["DTI_RATIO"] = listing["COMPLETE_PAYMENT"] / (listing["INCOME"] / 12)

listing['VACANCY'] = listing["VACANTA21"] / listing["HHA21"]

# Calculate distance using the Haversine formula
listing["DISTANCE"] = 3959 * np.arccos(
    np.cos(np.radians(Distance_Latitude)) * np.cos(np.radians(listing["LATITUDE"])) *
    np.cos(np.radians(listing["LONGITUDE"]) - np.radians(Distance_Longitude)) +
    np.sin(np.radians(Distance_Latitude)) * np.sin(np.radians(listing["LATITUDE"]))
)

# Calculate price per ABGSQFT for sold properties
rental["P_ABGSQFT"] = rental["CLOSEPRICE"] / rental["ABGSQFT"]

# Calculate price per ABGSQFT for sold properties
sold["P_ABGSQFT"] = sold["CLOSEPRICE"] / sold["ABGSQFT"]

# Calculate price per acre
sold["P_ACRE"] = sold["CLOSEPRICE"] / sold["ACRES"]
listing["P_ACRE"] = listing["CURRPRICE"] / listing["ACRES"]

# Clean up "STYLE" descriptions
sold["STYLE"] = sold["STYLE"].str.replace("Cape Cod", "", case=False, regex=True)
listing["STYLE"] = listing["STYLE"].str.replace("Cape Cod", "", case=False, regex=True)

# Calculate price per square foot, adjust for Cape Cod
sold["P_SQFT"] = sold["CLOSEPRICE"] / sold["ABGSQFT"]
cape_cod_mask = sold["STYLE"].str.contains("Cape Cod", case=False, na=False)
sold.loc[cape_cod_mask, "P_SQFT"] /= 0.8

listing["P_SQFT"] = listing["CURRPRICE"] / listing["ABGSQFT"]
cape_cod_mask = listing["STYLE"].str.contains("Cape Cod", case=False, na=False)
listing.loc[cape_cod_mask, "P_SQFT"] /= 0.8

# Adjust property built year price ratio
sold["AGE_DEP"] = (year_age + 2) - (sold["YEARBUILT"] - (datetime.now().year - year_age)).clip(lower=1)
listing["AGE_DEP"] = (year_age + 2) - (listing["YEARBUILT"] - (datetime.now().year - year_age)).clip(lower=1)

# Create an income range for comparison
listing['LOWER_BOUND'] = listing['INCOME'] - income_spread
listing['UPPER_BOUND'] = listing['INCOME'] + income_spread

def calculate_stats(row, sold_df):
    # Filter sold DataFrame based on conditions defined by the listing row
    filtered_sold = sold_df[
        (sold_df['CITY'] == row['CITY']) &
        (sold_df['PROPTYPE'] == row['PROPTYPE']) &
        (sold_df['INCOME'] >= row['LOWER_BOUND']) &
        (sold_df['INCOME'] <= row['UPPER_BOUND'])
    ]
    
    # If no entries match the filter, return None or default values
    if filtered_sold.empty:
        return pd.Series([None, None, None, None, None, None, None, None, None, None, None, None, 0], index=['CLOSE_PRICE', 'CLOSE_AGE', 'CLOSE_CARRY_COSTS', 'CLOSE_P_ACRE', 'CLOSE_P_SQFT', 'MED_P_ACRE', 'MED_P_SQFT', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'MED_DOM', 'SIZE_COUNT'])

    # Compute the aggregation statistics
    close_prices = filtered_sold['CLOSEPRICE']
    close_age = filtered_sold['AGE_DEP']
    close_carry_costs = filtered_sold['CARRY_COSTS']
    close_p_acre = filtered_sold["P_ACRE"]
    close_p_sqft = filtered_sold["P_SQFT"]
    price_acre = filtered_sold["P_ACRE"].median()
    price_sqft = filtered_sold["P_SQFT"].median()
    min_price = filtered_sold['CLOSEPRICE'].min()
    max_price = filtered_sold['CLOSEPRICE'].max()
    min_psqft = filtered_sold['P_ABGSQFT'].nsmallest(2).mean()
    max_psqft = filtered_sold['P_ABGSQFT'].nlargest(2).mean()
    dom = filtered_sold["DOM"].median()
    size_count = filtered_sold['P_ABGSQFT'].size
    
    return pd.Series([close_prices, close_age, close_carry_costs, close_p_acre, close_p_sqft, price_acre, price_sqft, min_price, max_price, min_psqft, max_psqft, dom, size_count], index=['CLOSE_PRICE', 'CLOSE_AGE', 'CLOSE_CARRY_COSTS', 'CLOSE_P_ACRE', 'CLOSE_P_SQFT', 'MED_P_ACRE', 'MED_P_SQFT', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'MED_DOM', 'SIZE_COUNT'])

def calculate_stats_r(row, rental_df):
    # Filter sold DataFrame based on conditions defined by the listing row
    filtered_rental = rental_df[
        (rental_df['CITY'] == row['CITY']) &
        (rental_df['PROPTYPE'] == row['PROPTYPE']) &
        (rental_df['BEDSTOTAL'] == row['BEDSTOTAL']) &
        (rental_df['INCOME'] >= row['LOWER_BOUND']) &
        (rental_df['INCOME'] <= row['UPPER_BOUND'])
    ]
    
    # If no entries match the filter, return None or default values
    if filtered_rental.empty:
        return pd.Series([None, None, 0], index=['RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT'])

    # Compute the aggregation statistics
    rent_price = filtered_rental["CLOSEPRICE"].median()
    price_sqft = filtered_rental["P_ABGSQFT"].median()
    size_count = filtered_rental['P_ABGSQFT'].size
    
    return pd.Series([rent_price, price_sqft, size_count], index=['RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT'])

# Apply the custom function to each row in the listing DataFrame
stats = listing.apply(lambda row: calculate_stats(row, sold), axis=1)

# Apply the custom function to each row in the listing DataFrame
stats_r = listing.apply(lambda row: calculate_stats_r(row, rental), axis=1)

# Combine the results with the listing DataFrame
final_result = pd.concat([listing, stats, stats_r], axis=1)

# Calculate cash flow after merging the rental statistics back
final_result["CASHFLOW"] = (final_result["RENT_PRICE"] * np.where(final_result["TOTAL_UNIT"] == 0, 1, final_result["TOTAL_UNIT"])) - final_result["COMPLETE_PAYMENT"]

def calculate_percentile(row, score_column, value_column):
    try:
        return percentileofscore(row[score_column], row[value_column]) / 100
    except Exception:
        return None

# Define columns for percentile calculations
percentile_columns = [
    ('CLOSE_PRICE', 'CURRPRICE', '%_RANK_PRICE'),
    ('CLOSE_AGE', 'AGE_DEP', '%_RANK_AGE'),
    ('CLOSE_CARRY_COSTS', 'CARRY_COSTS', '%_RANK_CARRY_COSTS'), #Avoid so many NaN's
    ('CLOSE_P_SQFT', 'P_SQFT', '%_RANK_P_SQFT'),
    ('CLOSE_P_ACRE', 'P_ACRE', '%_RANK_P_ACRE')
]

# Apply the percentile calculation for each defined column set
for score_col, value_col, result_col in percentile_columns:
    final_result[result_col] = final_result.apply(lambda row: calculate_percentile(row, score_col, value_col), axis=1)
#Other final calculations
final_result['MAX/MIN_PSQFT'] = final_result.apply(lambda row: row['MAX_PSQFT'] / row['MIN_PSQFT'] if row['MIN_PSQFT'] > 0 else None, axis=1)
final_result['PROP_SQFT'] = listing["P_SQFT"] / final_result['MED_P_SQFT']
final_result['PROP_ACRE'] = listing["P_ACRE"] / final_result['MED_P_ACRE']

# Output the result to the console
print(final_result[['MLS1', 'CITY', 'PROPTYPE', 'LOWER_BOUND', 'UPPER_BOUND', 'MED_P_ACRE', 'PROP_ACRE', 'MED_P_SQFT', 'PROP_SQFT', 'MED_DOM', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'SIZE_COUNT', 'MAX/MIN_PSQFT', "OWNERSHIP_%", "POP_DENSITY", "MORT_PAY", "CARRY_COSTS", "COMPLETE_PAYMENT", "INCOME", "DTI_RATIO", "DISTANCE", 'RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT', "INCOME", '%_RANK_PRICE', '%_RANK_AGE', '%_RANK_CARRY_COSTS', '%_RANK_P_SQFT', '%_RANK_P_ACRE', 'VACANCY', 'CURRPRICE', 'BATHSTOTAL', 'BEDSTOTAL', 'DOM']])

# Optionally, save the result to a CSV file
final_result[['MLS1', 'CITY', 'PROPTYPE', 'LOWER_BOUND', 'UPPER_BOUND', 'MED_P_ACRE', 'PROP_ACRE', 'MED_P_SQFT', 'PROP_SQFT', 'MED_DOM', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'SIZE_COUNT', 'MAX/MIN_PSQFT', "OWNERSHIP_%", "POP_DENSITY", "MORT_PAY", "CARRY_COSTS", "COMPLETE_PAYMENT", "INCOME", "DTI_RATIO", "DISTANCE", 'RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT', "INCOME", '%_RANK_PRICE', '%_RANK_AGE', '%_RANK_CARRY_COSTS', '%_RANK_P_SQFT', '%_RANK_P_ACRE', 'VACANCY', 'CURRPRICE', 'BATHSTOTAL', 'BEDSTOTAL', 'DOM', "CASHFLOW"]].to_csv(r'C:\cctaddr\RES-CALC-APPRECIATION-ALTERNATIVE.csv', index=False)
final_result.to_csv(r'C:\cctaddr\RES-STATS.csv', index=False)