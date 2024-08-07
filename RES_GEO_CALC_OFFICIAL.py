import pandas as pd
from datetime import datetime

# Load data
sold = pd.read_csv(r"C:\Users\jackd\Desktop\Python RE\RES_SOLD_SHORT_OFFICIAL_11-25-2021_05-17-2024.csv", low_memory=False)
listing = pd.read_csv(r"C:\Users\jackd\Downloads\RES_2024-05-15.csv", low_memory=False)

# Define and convert numeric columns to numeric types, coercing errors
numeric_columns_sold = ["CURRPRICE", "ABGSQFT", "ACRES", "YEARBUILT"]
sold[numeric_columns_sold] = sold[numeric_columns_sold].apply(pd.to_numeric, errors='coerce')

# Clean up "STYLE" descriptions
sold["STYLE"] = sold["STYLE"].str.replace("Cape Cod", "", case=False, regex=True)

# Calculate price per square foot, adjust for Cape Cod
sold["P_SQFT"] = sold["CURRPRICE"] / sold["ABGSQFT"]
cape_cod_mask = sold["STYLE"].str.contains("Cape Cod", case=False, na=False)
sold.loc[cape_cod_mask, "P_SQFT"] /= 0.8

# Adjust property built year price ratio
current_year = datetime.now().year
threshold_year = current_year - 100
sold["P_YRBT"] = (sold["YEARBUILT"] - threshold_year).clip(lower=1)
sold["P_YRBT"] = sold["CURRPRICE"] / sold["P_YRBT"]

# Calculate price per acre
sold["P_ACRE"] = sold["CURRPRICE"] / sold["ACRES"]

# Exclude non-numeric columns for median calculation
numeric_columns_for_median = ["P_SQFT", "P_YRBT", "P_ACRE", "DOM"]  # Add other numeric columns as needed

# Filter and calculate medians for groups with more than one entry
grouped = sold.groupby(["GEOID", "PROPTYPE"])
counts = grouped.size()
valid_groups = counts[counts > 1].index
medians = sold.loc[sold.set_index(["GEOID", "PROPTYPE"]).index.isin(valid_groups)].groupby(["GEOID", "PROPTYPE"])[numeric_columns_for_median].median().reset_index()

# Merge data
Geo_List = listing[["MLS1", "GEOID", "PROPTYPE"]]
merged_data = pd.merge(Geo_List, medians, on=["GEOID", "PROPTYPE"], how="left")

print(merged_data)
merged_data.to_csv(r"C:\cctaddr\Listings_Per_Filter.csv", index=False)