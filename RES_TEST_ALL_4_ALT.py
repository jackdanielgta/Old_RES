import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import percentileofscore

# Define constants
TAX_SPREAD = 250
INCOME_SPREAD = 5000
DOWNPAYMENT = 0.2
INTEREST_RATE = 0.0766
AMORTIZATION = 30
DISTANCE_LATITUDE = 41.0089370679199
DISTANCE_LONGITUDE = -73.6482077977137
YEAR_AGE = 30
CURRENT_YEAR = datetime.now().year

# Helper functions
def read_csv_efficient(file_path, columns, **kwargs):
    return pd.read_csv(file_path, usecols=columns, low_memory=False, **kwargs)

def clean_and_transform(df):
    df['PROPTYPE'] = df['PROPTYPE'].str.replace('RN/', '')
    multi_family_types = {"MF", "2F", "3F", "4F"}
    mask = df['PROPTYPE'].isin(multi_family_types)
    df.loc[mask, ["BEDSTOTAL", "ABGSQFT"]] = df.loc[mask, ["BEDSTOTAL", "ABGSQFT"]].div(df.loc[mask, "TOTAL_UNIT"]).round(0)
    return df

def update_proptype(df):
    style_mapping = {'Townhouse': 'TH', 'Mobile Home': 'MH', 'Cape Cod': 'SC'}
    for style, prop_type in style_mapping.items():
        mask = df['STYLE'].str.contains(style, na=False) & (df['PROPTYPE'].isin(['SF', 'CO']))
        df.loc[mask, 'PROPTYPE'] = prop_type
    return df

def choose_income(row):
    return row["MHIA21"] if 1 < row["MHIA21"] <= 250000 else row["AHIA21"]

def calculate_carry_costs(df, price_col):
    df['HOA'] = df['HOA'].fillna(0)
    df['CARRY_COSTS'] = (df[price_col] / 1000) * 4.2 / 12 + df['HOA'] + (df['PROPTAX'] / 12)
    return df

def calculate_distance(df):
    return 3959 * np.arccos(
        np.cos(np.radians(DISTANCE_LATITUDE)) * np.cos(np.radians(df["LATITUDE"])) *
        np.cos(np.radians(df["LONGITUDE"]) - np.radians(DISTANCE_LONGITUDE)) +
        np.sin(np.radians(DISTANCE_LATITUDE)) * np.sin(np.radians(df["LATITUDE"]))
    )

def calculate_percentile(row, score_column, value_column):
    try:
        return percentileofscore(row[score_column], row[value_column]) / 100
    except Exception:
        return None

# Load and preprocess data
columns_sold = ["MLS1", "GEOID", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CITY", "MHIA21", "AHIA21", "ACRES", "STYLE", "YEARBUILT", "DIRECT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "BATHSTOTAL", 'CLOSEDATE', "TOTAL_UNIT", "COUNTY", "BANK_OWNED", "FLOOD_ZN"]
columns_listing = columns_sold + ["CURRPRICE"]
columns_rental = ["MLS1", "PROPTYPE", "ABGSQFT", 'TOT_SQFT', "CLOSEPRICE", "CITY", "MHIA21", "AHIA21", "BEDSTOTAL", "BATHSTOTAL", "DIRECT", "STYLE", "COUNTY"]

sold = read_csv_efficient(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", columns_sold, encoding='latin1')
listing = read_csv_efficient(r"C:\Users\jackd\Downloads\trala.csv", columns_listing, encoding='latin1')
rental = read_csv_efficient(r"C:\Users\jackd\Desktop\Python RE\RENTAL_DATA_OFFICIAL_2024_05_07.csv", columns_rental)

for df in [sold, listing, rental]:
    df = clean_and_transform(df)
    df = update_proptype(df)

sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
sold = sold[sold['CLOSEDATE'].dt.year.isin([2023, 2024])]

placeholder_values = [np.inf, -np.inf, 999999999, 99999999, 9999999, 999999, 99999, 0, 1, 100000000]
sold['PROPTAX'] = sold['PROPTAX'].replace(placeholder_values, np.nan)

bin_edges = [0, 900, 1200, 1700, 2200, 2700, 3300, 4200, np.inf]
bin_labels = range(8)
for df in [sold, rental]:
    df['ABGSQFT_BINNED'] = pd.cut(df['ABGSQFT'], bins=bin_edges, labels=bin_labels, right=False)

# Impute missing values
group_medians = sold.groupby(['GEOID', 'ABGSQFT_BINNED'])['PROPTAX'].transform('median')
sold['PROPTAX'] = sold['PROPTAX'].fillna(group_medians)

for df in [sold, listing, rental]:
    df["INCOME"] = df.apply(choose_income, axis=1)

bin_edges_income = [0, 40000, 60000, 80000, 100000, 120000, 150000, 200000, np.inf]
bin_labels_income = range(8)
for df in [sold, rental]:
    df['TOT_INCOME_BINNED'] = pd.cut(df['INCOME'], bins=bin_edges_income, labels=bin_labels_income, right=False)

sold = calculate_carry_costs(sold, 'CLOSEPRICE')
listing = calculate_carry_costs(listing, 'CURRPRICE')

# Calculate additional fields for listing
monthly_interest_rate = INTEREST_RATE / 12
listing["MORT_PAY"] = (listing["CURRPRICE"] * (1 - DOWNPAYMENT)) * (monthly_interest_rate * (1 + monthly_interest_rate) ** (12 * AMORTIZATION)) / ((1 + monthly_interest_rate) ** (12 * AMORTIZATION) - 1)
listing["COMPLETE_PAYMENT"] = listing["MORT_PAY"] + listing["CARRY_COSTS"]
listing["DTI_RATIO"] = listing["COMPLETE_PAYMENT"] / (listing["INCOME"] / 12)
listing['VACANCY'] = listing["VACANTA21"] / listing["HHA21"]
listing["DISTANCE"] = calculate_distance(listing)
listing["OWNERSHIP_%"] = listing["OWNOCCA21"] / listing["TOTHSGA21"]
listing["POP_DENSITY"] = (listing["TOTPOPA21"] / listing["ALAND"] * 0.386102) * 1000

# Price calculations
for df in [sold, rental]:
    df["P_ABGSQFT"] = df["CLOSEPRICE"] / df["ABGSQFT"]

for df in [sold, listing]:
    df["P_SQFT"] = df["CLOSEPRICE" if "CLOSEPRICE" in df.columns else "CURRPRICE"] / df["TOT_SQFT"]
    df["P_ACRE"] = df["CLOSEPRICE" if "CLOSEPRICE" in df.columns else "CURRPRICE"] / df["ACRES"]
    df["AGE_DEP"] = (YEAR_AGE + 2) - (df["YEARBUILT"] - (CURRENT_YEAR - YEAR_AGE)).clip(lower=1)

listing['TAX_LOWER_BOUND'] = listing['PROPTAX'] - TAX_SPREAD
listing['TAX_UPPER_BOUND'] = listing['PROPTAX'] + TAX_SPREAD
listing['INCOME_LOWER_BOUND'] = listing['INCOME'] - INCOME_SPREAD
listing['INCOME_UPPER_BOUND'] = listing['INCOME'] + INCOME_SPREAD

# Define functions for calculating statistics
def calculate_tax_stats(row, sold_df):
    tax_spread_increment = 250
    attempts = 3
    lower_bound, upper_bound = row['TAX_LOWER_BOUND'], row['TAX_UPPER_BOUND']
    
    for _ in range(attempts):
        filtered_sold = sold_df[
            (sold_df['CITY'] == row['CITY']) &
            (sold_df['PROPTYPE'] == row['PROPTYPE']) &
            (sold_df['DIRECT'] == row['DIRECT']) &
            (sold_df['PROPTAX'].between(lower_bound, upper_bound))
        ]
        
        tax_count = filtered_sold['CLOSEPRICE'].size
        if tax_count >= 3:
            return pd.Series({'MED_PRICE': filtered_sold['CLOSEPRICE'].median(), 'TAX_COUNT': tax_count})
        
        lower_bound -= tax_spread_increment
        upper_bound += tax_spread_increment
    
    return pd.Series({'MED_PRICE': filtered_sold['CLOSEPRICE'].median() if not filtered_sold.empty else None, 'TAX_COUNT': tax_count})

def calculate_stats(row, sold_df):
    filtered_sold = sold_df[
        (sold_df['CITY'] == row['CITY']) &
        (sold_df['PROPTYPE'] == row['PROPTYPE']) &
        (sold_df['DIRECT'] == row['DIRECT']) &
        (sold_df['GEOID'] == row['GEOID'])
    ]
    
    if filtered_sold.empty:
        return pd.Series({col: None for col in ['CLOSE_PRICE', 'CLOSE_AGE', 'CLOSE_CARRY_COSTS', 'CLOSE_P_ACRE', 'CLOSE_P_SQFT', 'MED_P_ACRE', 'MED_P_SQFT', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'MED_DOM', 'SIZE_COUNT']})

    return pd.Series({
        'CLOSE_PRICE': filtered_sold['CLOSEPRICE'],
        'CLOSE_AGE': filtered_sold['AGE_DEP'],
        'CLOSE_CARRY_COSTS': filtered_sold['CARRY_COSTS'],
        'CLOSE_P_ACRE': filtered_sold["P_ACRE"],
        'CLOSE_P_SQFT': filtered_sold["P_SQFT"],
        'MED_P_ACRE': filtered_sold["P_ACRE"].median(),
        'MED_P_SQFT': filtered_sold["P_SQFT"].median(),
        'MIN_PRICE': filtered_sold['CLOSEPRICE'].min(),
        'MAX_PRICE': filtered_sold['CLOSEPRICE'].max(),
        'MIN_PSQFT': filtered_sold['P_ABGSQFT'].nsmallest(2).mean(),
        'MAX_PSQFT': filtered_sold['P_ABGSQFT'].nlargest(2).mean(),
        'MED_DOM': filtered_sold["DOM"].median(),
        'SIZE_COUNT': filtered_sold['P_ABGSQFT'].size
    })

def calculate_rent_stats(row, rental_df):
    def filter_rentals(city_or_county, sqft_bin_label, income_bin_label):
        criteria_match = (
            (rental_df[city_or_county] == row[city_or_county]) &
            (rental_df['BEDSTOTAL'] == row['BEDSTOTAL']) &
            (rental_df['ABGSQFT_BINNED'] == sqft_bin_label) &
            (rental_df['TOT_INCOME_BINNED'] == income_bin_label)
        )
        return rental_df[criteria_match & (rental_df['DIRECT'] == 'N')] if row['DIRECT'] == 'N' else rental_df[criteria_match]

    sqft_bin_label = pd.cut([row['ABGSQFT']], bins=bin_edges, labels=bin_labels, right=False)[0]
    income_bin_label = pd.cut([row['INCOME']], bins=bin_edges_income, labels=bin_labels_income, right=False)[0]
    
    filtered_rental = filter_rentals('CITY', sqft_bin_label, income_bin_label)
    if filtered_rental.empty:
        filtered_rental = filter_rentals('COUNTY', sqft_bin_label, income_bin_label)
    
    if filtered_rental.empty:
        return pd.Series({'RENT_PRICE': None, 'RENT_COUNT': 0})
    
    return pd.Series({'RENT_PRICE': filtered_rental["CLOSEPRICE"].median(), 'RENT_COUNT': filtered_rental['CLOSEPRICE'].size})

# Calculate statistics
tax_stats = listing.apply(lambda row: calculate_tax_stats(row, sold), axis=1)
stats = listing.apply(lambda row: calculate_stats(row, sold), axis=1)
rent_stats = listing.apply(lambda row: calculate_rent_stats(row, rental), axis=1)

final_result = pd.concat([listing, tax_stats, stats, rent_stats], axis=1)

# Calculate additional fields
final_result["CASHFLOW"] = (final_result["RENT_PRICE"] * np.where(final_result['PROPTYPE'].isin(['2F', '3F', '4F', 'MF']), final_result["TOTAL_UNIT"], 1)) - final_result["COMPLETE_PAYMENT"]
final_result["CAP_RATE"] = ((final_result["CASHFLOW"] * 12) / final_result["CURRPRICE"]) * 100

# Calculate percentile ranks
percentile_columns = [
    ('CLOSE_PRICE', 'CURRPRICE', '%_RANK_PRICE'),
    ('CLOSE_AGE', 'AGE_DEP', '%_RANK_AGE'),
    ('CLOSE_CARRY_COSTS', 'CARRY_COSTS', '%_RANK_CARRY_COSTS'), 
    ('CLOSE_P_SQFT', 'P_SQFT', '%_RANK_P_SQFT'),
    ('CLOSE_P_ACRE', 'P_ACRE', '%_RANK_P_ACRE')
]

for score_col, value_col, result_col in percentile_columns:
    final_result[result_col] = final_result.apply(lambda row: calculate_percentile(row, score_col, value_col), axis=1)

# Calculate additional ratios and differences
final_result['MAX/MIN_PSQFT'] = final_result.apply(lambda row: row['MAX_PSQFT'] / row['MIN_PSQFT'] if row['MIN_PSQFT'] > 0 else None, axis=1)
final_result['PROP_SQFT'] = final_result["P_SQFT"] / final_result['MED_P_SQFT']
final_result['PROP_ACRE'] = final_result["P_ACRE"] / final_result['MED_P_ACRE']
final_result['PROFIT_MED'] = final_result["MED_PRICE"] - final_result["CURRPRICE"]
# Continuing from where we left off...

final_result['PROFIT_%_OF_LIST'] = final_result['PROFIT_MED'] / final_result["CURRPRICE"]

# Prepare data for final merge
listing2 = listing[["MLS1", "CITY", "GEOID", "PROPTYPE", "ABGSQFT", "ACRES", "YEARBUILT", "DIRECT", "STYLE", "CURRPRICE", "MHIA21", "AHIA21", "PROPTAX"]]
listing2.columns = ["MLS1", "CITY", "GEOID", "PROPTYPE", "L_ABGSQFT", "L_ACRES", "L_YEARBUILT", "L_DIRECT", "STYLE", "CURRPRICE", "MHIA21", "AHIA21", "L_PROPTAX"]

sold2 = sold[["CITY", "GEOID", "PROPTYPE", "ABGSQFT", "ACRES", "YEARBUILT", "CLOSEPRICE", "MLS1", "DIRECT", "STYLE", "MHIA21", "AHIA21", "PROPTAX", "CLOSEDATE"]]
sold2.columns = ["CITY", "GEOID", "PROPTYPE", "S_ABGSQFT", "S_ACRES", "S_YEARBUILT", "CLOSEPRICE", "MLS_COMP", "S_DIRECT", "STYLE", "MHIA21", "AHIA21", "S_PROPTAX", "CLOSEDATE"]

# Merge all "PROPTYPE" values excluding "LN/RS"
merged_df = pd.merge(listing2[listing2['PROPTYPE'] != 'LN/RS'], sold2, on=["GEOID", "PROPTYPE"])

threshold_year = CURRENT_YEAR - 15

def filter_logic(row):
    criteria_match = ((row['S_ABGSQFT'] <= 1.25 * row['L_ABGSQFT']) & 
                      (row['S_ACRES'] <= 1.25 * row['L_ACRES']))
    
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

# Handle "LN/RS" separately
ln_rs_listing = listing2[listing2['PROPTYPE'] == 'LN/RS']
if not ln_rs_listing.empty:
    ln_rs_merged_df = pd.merge(ln_rs_listing, sold2, on=["GEOID"])
    
    def filter_logic_ln_rs(row):
        if row["L_DIRECT"] == "N":
            return (
                pd.notnull(row['L_ACRES']) and pd.notnull(row['S_ACRES']) and
                (row['S_ACRES'] <= 1.25 * row['L_ACRES']) and
                (row["S_DIRECT"] == "N")
                )
        else:
            return (
                pd.notnull(row['L_ACRES']) and pd.notnull(row['S_ACRES']) and
                (row['S_ACRES'] <= 1.25 * row['L_ACRES'])
                )
    
    ln_rs_filtered_df = ln_rs_merged_df[ln_rs_merged_df.apply(filter_logic_ln_rs, axis=1)]

    max_indices = filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
    max_values_df = filtered_df.loc[max_indices, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'L_ABGSQFT']]
    max_values_df = max_values_df.rename(columns={'L_ABGSQFT': 'ABGSQFT'})
    
    max_indices_land = ln_rs_filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
    max_values_df_land = ln_rs_filtered_df.loc[max_indices_land, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'L_ABGSQFT']]
    max_values_df_land = max_values_df_land.rename(columns={'L_ABGSQFT': 'ABGSQFT'})
    
    max_values_df = pd.concat([max_values_df, max_values_df_land])
else:
    max_indices = filtered_df.groupby('MLS1')['CLOSEPRICE'].idxmax()
    max_values_df = filtered_df.loc[max_indices, ['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'CURRPRICE', 'L_ABGSQFT']]
    max_values_df = max_values_df.rename(columns={'L_ABGSQFT': 'ABGSQFT'})

max_values_df['PROFIT'] = max_values_df['CLOSEPRICE'] - max_values_df['CURRPRICE']
investment = 50000 * (max_values_df['ABGSQFT'] / 1000)
max_values_df['CASH_ON_CASH'] = (max_values_df['PROFIT'] - investment) / (max_values_df['CURRPRICE'] * 0.2)
final_df = max_values_df[['MLS1', 'CLOSEPRICE', 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH']]

final_merge = pd.merge(final_result, final_df, on="MLS1", how='left')

final_columns = [
    'MLS1', 'CITY', 'PROPTYPE', 'PROPTAX', 'PROP_ACRE', 
    'PROP_SQFT', 'MED_DOM', 'MIN_PRICE', 'MAX_PRICE', 'MIN_PSQFT', 'MAX_PSQFT', 'SIZE_COUNT', 
    'MAX/MIN_PSQFT', "OWNERSHIP_%", "POP_DENSITY", "MORT_PAY", "CARRY_COSTS", "COMPLETE_PAYMENT", "INCOME", 
    "DTI_RATIO", "DISTANCE", 'RENT_PRICE', 'RENT_COUNT', '%_RANK_PRICE', '%_RANK_AGE', 
    '%_RANK_CARRY_COSTS', '%_RANK_P_SQFT', '%_RANK_P_ACRE', 'VACANCY', 'CURRPRICE', 'BATHSTOTAL', 'BEDSTOTAL', 
    'DOM', "CASHFLOW", 'TAX_COUNT', 'PROFIT_MED', 'PROFIT_%_OF_LIST', 'MED_PRICE', 
    'CLOSEPRICE', 'MLS_COMP', 'PROFIT', 'CASH_ON_CASH', "BANK_OWNED", "FLOOD_ZN", "CAP_RATE", "ABGSQFT", "LATITUDE", "LONGITUDE"
]

# Print and save results
print(final_merge[final_columns])
final_merge[final_columns].to_csv(r'C:\cctaddr\RES-STATS.csv', index=False)
final_merge.to_csv(r'C:\cctaddr\RES_FULL_DATA.csv', index=False)