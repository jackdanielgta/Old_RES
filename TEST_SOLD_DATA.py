import pandas as pd
import numpy as np
from datetime import datetime

# Read the CSV file
sold = pd.read_csv(r"C:\Users\jackd\Downloads\RES_SOLD_RESCUE.csv", low_memory=False)

# Replace placeholder values and infinite values with NaN
placeholder_values = [999999999.00, 99999999.00, 9999999.00, 999999.00, 99999.00]
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

sold["OWNERSHIP_%"] = sold["OWNOCCA21"] / sold["TOTHSGA21"]

sold["POP_DENSITY"] = (sold["TOTPOPA21"]/sold["ALAND"] * 0.386102)*1000

# Categorize INCOME
def categorize_income(income):
    if income <= 50000:
        return '0-50000'
    elif 50000 < income <= 60000:
        return '50000-60000'
    elif 60000 < income <= 70000:
        return '60000-70000'
    elif 70000 < income <= 100000:
        return '70000-100000'
    elif 100000 < income <= 125000:
        return '100000-125000'
    elif 125000 < income <= 150000:
        return '125000-150000'
    else:
        return 'over 150000'

sold['INCOME_CATEGORY'] = sold['INCOME'].apply(categorize_income)

# Calculate per stats
sold['P_SQFT'] = sold['CLOSEPRICE'] / sold['ABGSQFT']
sold['P_TSQFT'] = sold['CLOSEPRICE'] / sold['TOT_SQFT']
sold['P_ACRE'] = sold['CLOSEPRICE'] / sold['ACRES']
sold['T_SQFT'] = sold['PROPTAX'] / sold['ABGSQFT']
sold['T_TSQFT'] = sold['PROPTAX'] / sold['TOT_SQFT']
sold['T_ACRE'] = sold['PROPTAX'] / sold['ACRES']

# Convert 'CLOSEDATE' to datetime
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'])

# Create a new column 'YEAR' to label the year in which a property closes
sold['YEAR'] = sold['CLOSEDATE'].dt.year

# Calculate the current year
current_year = datetime.now().year

# Calculate the age of the property
sold['AGE'] = current_year - sold['YEARBUILT']

# Categorize the density of the property
def categorize_density(density):
    if density <= .7:
        return 'Low Density'
    else:
        return 'High Density'

sold["POP_DENSITY_CATEGORY"] = sold["POP_DENSITY"].apply(categorize_density)

# Categorize the ownership of the property
def categorize_ownership(ownership):
    if ownership <= .5:
        return 'Low Ownership'
    else:
        return 'High Ownership'

sold["OWNERSHIP_CATEGORY"] = sold["OWNERSHIP_%"].apply(categorize_ownership)

# Categorize the sale of the property
def categorize_sale(sale):
    if sale <= 14:
        return 'Quick'
    elif 14 < sale <= 45:
        return 'Moderate'
    else:
        return 'Long'

sold["SALE_CATEGORY"] = sold["DOM"].apply(categorize_sale)

# Categorize STYLE
def categorize_style(row):
    style = row['STYLE']
    proptype = row['PROPTYPE']
    if isinstance(style, str):
        style = style.lower()
        if 'cape cod' in style:
            return 'Cape Cod'
        elif 'colonial' in style:
            return 'Colonial'
        elif 'townhouse' in style:
            return 'Townhouse'
        elif 'mobile home' in style:
            return 'Mobile Home'
        elif 'ranch' in style:
            if 'CO' in proptype or any(x in style for x in ['mid rise', 'high rise', 'apartment']):
                return 'Condo Apartment'
            return 'Ranch'
    return 'other'

sold['STYLE_CATEGORY'] = sold.apply(categorize_style, axis=1)

# Categorize HEATTYPE
def categorize_heattype(heattype):
    if isinstance(heattype, str):
        heattype = heattype.lower()
        if 'hot air' in heattype:
            return 'Hot Air'
        elif 'heat pump' in heattype:
            return 'Heat Pump'
    return 'other'

sold['HEATTYPE_CATEGORY'] = sold['HEATTYPE'].apply(categorize_heattype)

# Categorize AGE and CONTEMPORARY STYLE
def categorize_age_and_style(row):
    age = row['AGE']
    style = row['STYLE']
    if age <= 15 or (isinstance(style, str) and 'contempor' in style.lower()):
        return 'New'
    return 'Old'

sold['AGE_STYLE_CATEGORY'] = sold.apply(categorize_age_and_style, axis=1)

# Custom aggregation functions
def median_of_bottom(series):
    return series.nsmallest(3).median()

def median_of_top(series):
    return series.nlargest(3).median()

# Group by 'CITY', 'INCOME_CATEGORY', 'PROPTYPE', 'YEAR', and apply the custom aggregations
aggregations = {
    'P_SQFT': [median_of_bottom, median_of_top, 'count']
}

# Apply the aggregations
tally = sold.groupby(["CITY", 'INCOME_CATEGORY', "PROPTYPE", "YEAR"]).agg(aggregations).reset_index()

# Flatten the column names after aggregation
tally.columns = ['_'.join(col).strip() if col[1] else col[0] for col in tally.columns.values]

# Display the first 10 rows of the result
print(tally.head(10))
# Save the merged data with projected P_SQFT to a CSV file
tally.to_csv(r"C:\cctaddr\SOLD_DATA_TEST2.csv", index=False)

# Calculate the current year's min and max P_SQFT for each group
current_year_tally = tally[tally['YEAR'] == current_year]

# Merge the tally with the original sold data to get the min and max P_SQFT for each group
merged = pd.merge(sold, tally, on=["CITY", 'INCOME_CATEGORY', "PROPTYPE", "YEAR"], suffixes=('', '_tally'))

# Calculate the normalized score
merged['P_SQFT_normalized'] = (merged['P_SQFT'] - merged['P_SQFT_median_of_bottom']) / (merged['P_SQFT_median_of_top'] - merged['P_SQFT_median_of_bottom'])

# Merge with the current year's tally to get the min and max P_SQFT for the current year
merged = pd.merge(merged, current_year_tally[["CITY", 'INCOME_CATEGORY', 'PROPTYPE', 'P_SQFT_median_of_bottom', 'P_SQFT_median_of_top', 'P_SQFT_count']], on=["CITY", 'INCOME_CATEGORY', "PROPTYPE"], suffixes=('', '_current'))

# Calculate the projected P_SQFT for the current year with a condition on the count
merged['P_SQFT_projected'] = merged.apply(
    lambda row: None if row['P_SQFT_count_current'] < 6 or row['P_SQFT_count'] < 6 else row['P_SQFT_median_of_bottom_current'] + row['P_SQFT_normalized'] * (row['P_SQFT_median_of_top_current'] - row['P_SQFT_median_of_bottom_current']),
    axis=1)

merged['NEW_CLOSEPRICE'] = merged['P_SQFT_projected'] * merged['ABGSQFT']

# Display the first 10 rows of the result
print(merged.head(10))

# Save the merged data with projected P_SQFT to a CSV file
merged.to_csv(r"C:\cctaddr\SOLD_DATA_TEST3.csv", index=False)
