import pandas as pd
import numpy as np
from scipy import stats

# Load data
sold_data = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')
listing_data = pd.read_csv(r"C:\Users\jackd\Downloads\righthererightnownow.csv", low_memory=False, encoding='latin1')

# Choose income
def choose_income(row):
    return row["MHIA21"] if 1 < row["MHIA21"] <= 250000 else row["AHIA21"]

for df in [sold_data, listing_data]:
    df["INCOME"] = df.apply(choose_income, axis=1)

# Calculate P_SQFT if it doesn't exist
if 'P_SQFT' not in sold_data.columns:
    sold_data['P_SQFT'] = sold_data['CLOSEPRICE'] / sold_data['TOT_SQFT']
if 'P_SQFT' not in listing_data.columns:
    listing_data['P_SQFT'] = listing_data['CURRPRICE'] / listing_data['TOT_SQFT']

# Create income bins
bin_edges_income = [0, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 160000, 210000, 240000, 250000, np.inf]
bin_labels_income = ['0-40k', '40k-50k', '50k-60k', '60k-70k', '70k-80k', '80k-90k', '90k-100k', '100k-160k', '160k-210k', '210k-240k', '240k-250k', '250k+']
sold_data['income_bin'] = pd.cut(sold_data['INCOME'], bins=bin_edges_income, labels=bin_labels_income)

# Calculate MAD for P_SQFT
def mad(x):
    return np.median(np.abs(x - np.median(x)))

bin_stats = sold_data.groupby('income_bin').agg({
    'P_SQFT': ['median', mad, 'min', 'max']
}).reset_index()
bin_stats.columns = ['income_bin', 'P_SQFT_median', 'P_SQFT_MAD', 'P_SQFT_min', 'P_SQFT_max']

# Calculate z-score using MAD and normalize
def normalized_z_score_mad(row):
    z_score = 0.6745 * (row['P_SQFT'] - row['P_SQFT_median']) / row['P_SQFT_MAD']
    if row['P_SQFT'] < row['P_SQFT_min']:
        return 0
    elif row['P_SQFT'] > row['P_SQFT_max']:
        return 100
    else:
        return stats.norm.cdf(z_score) * 100

# Assign income bins to listing data
listing_data['income_bin'] = pd.cut(listing_data['INCOME'], bins=bin_edges_income, labels=bin_labels_income)

# Calculate normalized z-scores for listing data
listing_data = listing_data.merge(bin_stats, on='income_bin', how='left')
listing_data['P_SQFT_normalized_z_score'] = listing_data.apply(normalized_z_score_mad, axis=1)

# Display results
print(bin_stats)
print(listing_data[['MLS1', 'INCOME', 'income_bin', 'P_SQFT', 'P_SQFT_normalized_z_score']].head())
listing_data[['MLS1', 'INCOME', 'income_bin', 'P_SQFT', 'P_SQFT_normalized_z_score']].to_csv(r'C:\cctaddr\RES_MAD_INCOME.csv', index=False)