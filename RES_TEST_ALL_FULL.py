import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import percentileofscore

#FOR MED_PRICE ADD NEXT SMALLER TAX RANGE IF NaN

# Define constants
tax_spread = 250
income_spread = 5000
downpayment = 0.2
interest_rate = 0.0766
Amoritization = 30
Distance_Latitude = 41.0089370679199
Distance_Longitude = -73.6482077977137
year_age = 30

# Load CSV files
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')
listing = pd.read_csv(r"C:\Users\jackd\Desktop\RES-MF_LISTINGS.csv", low_memory=False, encoding='latin1')
rental = pd.read_csv(r"C:\Users\jackd\Desktop\Python RE\RENTAL_DATA_OFFICIAL_2024_05_07.csv", low_memory=False)
land = pd.read_csv(r"C:\Users\jackd\Downloads\Land_2024-05-15.csv", low_memory=False)

# Select and rename columns
columns_listing = ["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CURRPRICE", "CITY", "MHIA21", "AHIA21", "ACRES", "STYLE", "BEDSTOTAL", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "VACANTA21", "HHA21", "BATHSTOTAL", "TOTAL_UNIT"]
columns_sold = ["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CITY", "MHIA21", "AHIA21", "ACRES", "STYLE", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "BATHSTOTAL", 'CLOSEDATE']
columns_rental = ["MLS1", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CITY", "MHIA21", "AHIA21", "BEDSTOTAL", "BATHSTOTAL", "DIRECT", "STYLE"]

listing = listing[columns_listing]
sold = sold[columns_sold]
rental = rental[columns_rental]

# Data cleaning and transformation
rental['PROPTYPE'] = rental['PROPTYPE'].str.replace('RN/', '')

property_types = {'Townhouse': 'TH', 'Mobile Home': 'MH', 'Cape Cod': 'SC'}
for style, ptype in property_types.items():
    mask = sold['STYLE'].str.contains(style, na=False)
    sold.loc[mask, 'PROPTYPE'] = ptype
    listing.loc[mask, 'PROPTYPE'] = ptype
    rental.loc[mask, 'PROPTYPE'] = ptype

sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold = sold[sold['CLOSEDATE'].dt.year.isin([2023, 2024])]

placeholder_values = [np.inf, -np.inf, 999999999.00, 99999999.00, 9999999.00, 999999.00, 99999.00, 0, 1, 100000000]
sold['PROPTAX'].replace(placeholder_values, np.nan, inplace=True)

bin_edges = [0, 1000, 2000, 3000, 4000, 5500, np.inf]
bin_labels = [0, 1, 2, 3, 4, 5]
sold['TOT_SQFT_BINNED'] = pd.cut(sold['TOT_SQFT'], bins=bin_edges, labels=bin_labels, right=False)
rental['TOT_SQFT_BINNED'] = pd.cut(rental['TOT_SQFT'], bins=bin_edges, labels=bin_labels, right=False)

def impute_with_combined_group_median(df, target_column, group_columns):
    group_medians = df.groupby(group_columns)[target_column].transform('median')
    df[target_column].fillna(group_medians, inplace=True)

impute_with_combined_group_median(sold, 'PROPTAX', ['GEOID', 'TOT_SQFT_BINNED'])

# Function to choose income
def choose_income(row):
    return row["MHIA21"] if 1 < row["MHIA21"] <= 250000 else row["AHIA21"]

for df in [sold, listing, rental]:
    df["INCOME"] = df.apply(choose_income, axis=1)

def calculate_carry_costs(df, price_col):
    df['HOA'].fillna(0, inplace=True)
    df['CARRY_COSTS'] = (df[price_col] / 1000) * 4.2 / 12 + df['HOA'] + (df['PROPTAX'] / 12)

calculate_carry_costs(sold, 'CLOSEPRICE')
calculate_carry_costs(listing, 'CURRPRICE')

listing["OWNERSHIP_%"] = listing["OWNOCCA21"] / listing["TOTHSGA21"]
listing["POP_DENSITY"] = (listing["TOTPOPA21"] / listing["ALAND"] * 0.386102) * 1000

monthly_interest_rate = interest_rate / 12
listing["MORT_PAY"] = (listing["CURRPRICE"] * (1 - downpayment)) * (monthly_interest_rate * (1 + monthly_interest_rate) ** (12 * Amoritization)) / ((1 + monthly_interest_rate) ** (12 * Amoritization) - 1)
listing["COMPLETE_PAYMENT"] = listing["MORT_PAY"] + listing["CARRY_COSTS"]
listing["DTI_RATIO"] = listing["COMPLETE_PAYMENT"] / (listing["INCOME"] / 12)
listing['VACANCY'] = listing["VACANTA21"] / listing["HHA21"]

# Calculate distance using the Haversine formula
listing["DISTANCE"] = 3959 * np.arccos(
    np.cos(np.radians(Distance_Latitude)) * np.cos(np.radians(listing["LATITUDE"])) *
    np.cos(np.radians(listing["LONGITUDE"]) - np.radians(Distance_Longitude)) +
    np.sin(np.radians(Distance_Latitude)) * np.sin(np.radians(listing["LATITUDE"]))
)

# Price calculations
for df in [sold, rental]:
    df["P_ABGSQFT"] = df["CLOSEPRICE"] / df["ABGSQFT"]

sold["P_SQFT"] = sold["CLOSEPRICE"] / sold["TOT_SQFT"]
listing["P_SQFT"] = listing["CURRPRICE"] / listing["TOT_SQFT"]

sold["P_ACRE"] = sold["CLOSEPRICE"] / sold["ACRES"]
listing["P_ACRE"] = listing["CURRPRICE"] / listing["ACRES"]

# Adjust property built year price ratio
current_year = datetime.now().year
sold["AGE_DEP"] = (year_age + 2) - (sold["YEARBUILT"] - (current_year - year_age)).clip(lower=1)
listing["AGE_DEP"] = (year_age + 2) - (listing["YEARBUILT"] - (current_year - year_age)).clip(lower=1)

listing['TAX_LOWER_BOUND'] = listing['PROPTAX'] - tax_spread
listing['TAX_UPPER_BOUND'] = listing['PROPTAX'] + tax_spread
listing['INCOME_LOWER_BOUND'] = listing['INCOME'] - income_spread
listing['INCOME_UPPER_BOUND'] = listing['INCOME'] + income_spread

def calculate_tax_stats(row, sold_df):
    filtered_sold = sold_df[
        (sold_df['CITY'] == row['CITY']) &
        (sold_df['PROPTYPE'] == row['PROPTYPE']) &
        (sold_df['DIRECT'] == row['DIRECT']) &
        (sold_df['PROPTAX'] >= row['TAX_LOWER_BOUND']) &
        (sold_df['PROPTAX'] <= row['TAX_UPPER_BOUND'])
    ]
    
    if filtered_sold.empty:
        return pd.Series([None, 0], index=['MED_PRICE', 'TAX_COUNT'])

    median_price = filtered_sold['CLOSEPRICE'].median()
    tax_count = filtered_sold['CLOSEPRICE'].size
    return pd.Series([median_price, tax_count], index=['MED_PRICE', 'TAX_COUNT'])

def calculate_stats(row, sold_df):
    filtered_sold = sold_df[
        (sold_df['CITY'] == row['CITY']) &
        (sold_df['PROPTYPE'] == row['PROPTYPE']) &
        (sold_df['DIRECT'] == row['DIRECT']) &
        (sold_df['GEOID'] == row['GEOID'])
    ]
    
    if filtered_sold.empty:
        return pd.Series([None, None, None, None, None, None, None, None, None, None, None, None, 0], index=['CLOSE_PRICE', 'CLOSE_AGE', 'CLOSE_CARRY_COSTS', 'CLOSE_P_ACRE', 'CLOSE_P_SQFT', 'MED_P_ACRE', 'MED_P_SQFT', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'MED_DOM', 'SIZE_COUNT'])

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

def calculate_rent_stats(row, rental_df):
    bin_label = pd.cut([row['TOT_SQFT']], bins=bin_edges, labels=bin_labels, right=False)[0]
    criteria_match = ((rental_df['CITY'] == row['CITY']) &
                      (rental_df['BEDSTOTAL'] == row['BEDSTOTAL']) &
                      (rental_df['INCOME'] >= row['INCOME_LOWER_BOUND']) &
                      (rental_df['INCOME'] <= row['INCOME_UPPER_BOUND']) &
                      (rental_df['TOT_SQFT_BINNED'] == bin_label))

    if row['DIRECT'] == 'N':
        if row['PROPTYPE'] == 'SF':
            filtered_rental = rental_df[criteria_match & (rental_df['PROPTYPE'] == 'SF') & (rental_df['DIRECT'] == 'N')]
        else:
            filtered_rental = rental_df[criteria_match & (rental_df['PROPTYPE'] != 'SF') & (rental_df['DIRECT'] == 'N')]
    else:
        if row['PROPTYPE'] == 'SF':
            filtered_rental = rental_df[criteria_match & (rental_df['PROPTYPE'] == 'SF')]
        else:
            filtered_rental = rental_df[criteria_match & (rental_df['PROPTYPE'] != 'SF')]

    if filtered_rental.empty:
        return pd.Series([None, None, 0], index=['RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT'])

    rent_price = filtered_rental["CLOSEPRICE"].median()
    price_sqft = filtered_rental["P_ABGSQFT"].median()
    size_count = filtered_rental['P_ABGSQFT'].size
    
    return pd.Series([rent_price, price_sqft, size_count], index=['RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT'])

tax_stats = listing.apply(lambda row: calculate_tax_stats(row, sold), axis=1)
stats = listing.apply(lambda row: calculate_stats(row, sold), axis=1)
rent_stats = listing.apply(lambda row: calculate_rent_stats(row, rental), axis=1)

final_result = pd.concat([listing, tax_stats, stats, rent_stats], axis=1)
final_result["CASHFLOW"] = (final_result["RENT_PRICE"] * np.where(final_result['PROPTYPE'].isin(['2F', '3F', '4F', 'MF']), final_result["TOTAL_UNIT"], 1)) - final_result["COMPLETE_PAYMENT"]

def calculate_percentile(row, score_column, value_column):
    try:
        return percentileofscore(row[score_column], row[value_column]) / 100
    except Exception:
        return None

percentile_columns = [
    ('CLOSE_PRICE', 'CURRPRICE', '%_RANK_PRICE'),
    ('CLOSE_AGE', 'AGE_DEP', '%_RANK_AGE'),
    ('CLOSE_CARRY_COSTS', 'CARRY_COSTS', '%_RANK_CARRY_COSTS'), 
    ('CLOSE_P_SQFT', 'P_SQFT', '%_RANK_P_SQFT'),
    ('CLOSE_P_ACRE', 'P_ACRE', '%_RANK_P_ACRE')
]

for score_col, value_col, result_col in percentile_columns:
    final_result[result_col] = final_result.apply(lambda row: calculate_percentile(row, score_col, value_col), axis=1)

final_result['MAX/MIN_PSQFT'] = final_result.apply(lambda row: row['MAX_PSQFT'] / row['MIN_PSQFT'] if row['MIN_PSQFT'] > 0 else None, axis=1)
final_result['PROP_SQFT'] = listing["P_SQFT"] / final_result['MED_P_SQFT']
final_result['PROP_ACRE'] = listing["P_ACRE"] / final_result['MED_P_ACRE']
final_result['PROFIT_MED'] = final_result["MED_PRICE"] - listing["CURRPRICE"]
final_result['PROFIT_%_OF_LIST'] = final_result['PROFIT_MED'] / listing["CURRPRICE"]

listing2 = listing[["MLS1", "CITY", "GEOID", "PROPTYPE", "ABGSQFT", "ACRES", "YEARBUILT", "DIRECT", "STYLE", "CURRPRICE", "MHIA21", "AHIA21", "PROPTAX"]]
listing2.columns = ["MLS1", "CITY", "GEOID", "PROPTYPE", "L_ABGSQFT", "L_ACRES", "L_YEARBUILT", "L_DIRECT", "STYLE", "CURRPRICE", "MHIA21", "AHIA21", "L_PROPTAX"]
sold2 = sold[["CITY", "GEOID", "PROPTYPE", "ABGSQFT", "ACRES", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE", "MHIA21", "AHIA21", "PROPTAX", "CLOSEDATE"]]
sold2.columns = ["CITY", "GEOID", "PROPTYPE", "S_ABGSQFT", "S_ACRES", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "STYLE", "MHIA21", "AHIA21", "S_PROPTAX", "CLOSEDATE"]

merged_df = pd.merge(listing2, sold2, on=["GEOID", "PROPTYPE"])

current_year = datetime.now().year
threshold_year = current_year - 15

def filter_logic(row):
    criteria_match = ((row['S_ABGSQFT'] <= 1.1 * row['L_ABGSQFT']) & 
                      (row['S_ACRES'] <= 1.1 * row['L_ACRES']))
    
    if row["L_DIRECT"] == "N":
        if row['L_YEARBUILT'] < threshold_year:
            return (criteria_match & (row["S_DIRECT"] == "N") & (row['S_YEARBUILT'] < threshold_year))
        else:
            return (criteria_match & (row["S_DIRECT"] == "N"))
    else:
        if row['L_YEARBUILT'] < threshold_year:
            return (criteria_match & (row['S_YEARBUILT'] < threshold_year))
        else:
            return criteria_match

filtered_df = merged_df[merged_df.apply(filter_logic, axis=1)]
max_indices = filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
max_values_df = filtered_df.loc[max_indices]
max_values_df = filtered_df.loc[max_indices, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'L_ABGSQFT']]
max_values_df.rename(columns={'L_ABGSQFT': 'ABGSQFT'}, inplace=True)

max_values_df['PROFIT'] = max_values_df['CLOSEPRICE'] - max_values_df['CURRPRICE']
investment = 50000 * (max_values_df['ABGSQFT'] / 1000)
max_values_df['CASH_ON_CASH'] = (max_values_df['PROFIT'] - investment) / (max_values_df['CURRPRICE'] * 0.2)
final_df = max_values_df[['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH']]

final_merge = pd.merge(final_result, final_df, on="MLS1", how='left')

final_columns = [
    'MLS1', 'CITY', 'PROPTYPE', 'PROPTAX', 'INCOME_LOWER_BOUND', 'INCOME_UPPER_BOUND', 'MED_P_ACRE', 'PROP_ACRE', 
    'MED_P_SQFT', 'PROP_SQFT', 'MED_DOM', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'SIZE_COUNT', 
    'MAX/MIN_PSQFT', "OWNERSHIP_%", "POP_DENSITY", "MORT_PAY", "CARRY_COSTS", "COMPLETE_PAYMENT", "INCOME", 
    "DTI_RATIO", "DISTANCE", 'RENT_PRICE', 'RENT_SQFT', 'RENT_COUNT', '%_RANK_PRICE', '%_RANK_AGE', 
    '%_RANK_CARRY_COSTS', '%_RANK_P_SQFT', '%_RANK_P_ACRE', 'VACANCY', 'CURRPRICE', 'BATHSTOTAL', 'BEDSTOTAL', 
    'DOM', "CASHFLOW", 'TAX_COUNT', 'PROFIT_MED', 'PROFIT_%_OF_LIST', 'MED_PRICE', 'TAX_LOWER_BOUND', 'TAX_UPPER_BOUND', 
    'CLOSEPRICE', 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH'
]

print(final_merge[final_columns])
final_merge[final_columns].to_csv(r'C:\cctaddr\RES-STATS.csv', index=False)
final_merge.to_csv(r'C:\cctaddr\RES_FULL_DATA.csv', index=False)