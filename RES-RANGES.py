import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import re
from scipy import stats

# Load only the sold data
sold = pd.read_csv(r"C:\cctaddr\OFFICIAL_FILTERED_SOLD.csv", low_memory=False, encoding='latin1')

# Helper functions
def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def mad_score(points):
    median = np.median(points)
    mad = np.median(np.abs(points - median))
    modified_z_scores = 0.6745 * (points - median) / mad
    return modified_z_scores

def cap_outliers(df, column, threshold=3.5):
    z_scores = mad_score(df[column])
    df[column] = df[column].mask(z_scores > threshold, df[column].quantile(0.95))
    return df

def choose_income(row):
    return min(row["MHIA21"], 250000) if pd.notna(row["MHIA21"]) else min(row["AHIA21"], 250000)

# Calculate new columns efficiently
sold['VACANCY'] = np.minimum(safe_divide(sold["VACANTA21"], sold["HHA21"]), 1)
sold["INCOME"] = sold.apply(choose_income, axis=1)
sold["POP_DENSITY"] = np.minimum((sold["TOTPOPA21"] / sold["ALAND"] * 0.386102) * 1000, 1.5)  # 2% cap (20,000 per sq mile)
sold["OWNERSHIP_%"] = safe_divide(sold["OWNOCCA21"], sold["TOTHSGA21"])
sold['PRICE_SQFT'] = safe_divide(sold['CLOSEPRICE'], sold['ABGSQFT'])
sold['ABGSQFT'] = np.minimum(sold['ABGSQFT'], 5000)
sold['ACRES'] = np.minimum(sold['ACRES'], 5)

# Cap outliers using MAD score
sold = cap_outliers(sold, 'PRICE_SQFT')

# List of variables to analyze
variables = ["ABGSQFT", "VACANCY", "INCOME", "POP_DENSITY", "OWNERSHIP_%", "ACRES", "PRICE_SQFT"]

# Create bins for each variable against PRICE_SQFT
for var in variables:
    if var != "PRICE_SQFT":
        x = sold[var].dropna()
        y = sold["PRICE_SQFT"].dropna()
        
        # Only use data points where both x and y are valid
        mask = ~(np.isnan(x) | np.isnan(y))
        x = x[mask]
        y = y[mask]
        
        # Create percentile-based bins for the main variable
        x_bins = np.percentile(x, np.linspace(0, 100, 21))  # 20 bins
        
        # Create equal-width bins for PRICE_SQFT
        y_bins = np.linspace(y.min(), y.max(), 21)  # 20 bins
        
        # Plot the result
        plt.figure(figsize=(12, 10))
        plt.hist2d(x, y, bins=(x_bins, y_bins), cmap='viridis', norm=colors.LogNorm())
        plt.colorbar(label='Frequency')
        plt.xlabel(var)
        plt.ylabel('Price per Square Foot')
        plt.title(f'{var} vs Price per Square Foot')
        plt.savefig(f'C:/cctaddr/{var}_vs_PRICE_SQFT.png')
        plt.close()
        
        # Save bin edges to CSV
        pd.DataFrame({
            f'{var}_bins': x_bins[:-1],
            f'{var}_bins_upper': x_bins[1:],
            'PRICE_SQFT_bins': y_bins[:-1],
            'PRICE_SQFT_bins_upper': y_bins[1:]
        }).to_csv(f'C:/cctaddr/{var}_PRICE_SQFT_bins.csv', index=False)

print("Analysis complete. Check the output files in C:/cctaddr/")