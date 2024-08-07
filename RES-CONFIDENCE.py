import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

# Define constants
today = datetime.now()
one_year_ago = today - timedelta(days=365)
current_year = datetime.now().year

# Load CSV file
sold = pd.read_csv(r"C:\cctaddr\OFFICIAL_FILTERED_SOLD.csv", low_memory=False, encoding='latin1')

# Create AGE_BINNED for sold DataFrame
newbuild_threshold = current_year - 15
age_bin_edges = [-np.inf, 1935, newbuild_threshold, np.inf]
age_bin_labels = ['Pre-1935', '1935-2009', '2010+']
sold['AGE_BINNED'] = pd.cut(sold['YEARBUILT'], bins=age_bin_edges, labels=age_bin_labels, right=False)

income_bin_edges = [0, 40000, 60000, 90000, 210000, np.inf]
income_bin_labels = ['0-40k', '40k-60k', '60k-90k', '90k-210k', '210k+']
sold['TOT_INCOME_BINNED'] = pd.cut(sold['INCOME'], bins=income_bin_edges, labels=income_bin_labels, right=False)

abgsqft_bin_edges = [0, 800, 1200, 1800, 2600, np.inf]
abgsqft_bin_labels = ['0-800', '801-1200', '1201-1800', '1801-2600', '2601+']
sold['ABGSQFT_BINNED'] = pd.cut(sold['ABGSQFT'], bins=abgsqft_bin_edges, labels=abgsqft_bin_labels, right=False)

acres_bin_edges = [0, .25, 20, 100, np.inf]
acres_bin_labels = ['0-0.25', '0.25-20', '20.1-100', '100+']
sold['ACRES_BINNED'] = pd.cut(sold['ACRES'], bins=acres_bin_edges, labels=acres_bin_labels, right=False)

own_bin_edges = [0, .2, .45, .7, np.inf]
own_bin_labels = ['0-20%', '20-45%', '45-70%', '70%+']
sold['OWN_BINNED'] = pd.cut(sold['OWNERSHIP_%'], bins=own_bin_edges, labels=own_bin_labels, right=False)

vac_bin_edges = [0, .15, .45, np.inf]
vac_bin_labels = ['0-15%', '15-45%', '45%+']
sold['VAC_BINNED'] = pd.cut(sold['VACANCY'], bins=vac_bin_edges, labels=vac_bin_labels, right=False)

popden_bin_edges = [0, .5, .9, 1.9, np.inf]
popden_bin_labels = ['0-0.5', '0.5-0.9', '0.9-1.9', '1.9+']
sold['POPDEN_BINNED'] = pd.cut(sold['POP_DENSITY'], bins=popden_bin_edges, labels=popden_bin_labels, right=False)

# Function to calculate MAD confidence interval
def mad_confidence_interval(group, confidence=0.95):
    median = group.median()
    mad = np.median(np.abs(group - median))
    z_score = scipy_stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * mad / np.sqrt(len(group))
    lower_bound = median - margin_of_error
    upper_bound = median + margin_of_error
    return pd.Series({'median': median, 'lower_ci': lower_bound, 'upper_ci': upper_bound, 'count': len(group)})

# Group by all specified columns
groupby_columns = ["CITY", "PROPTYPE", "TOT_INCOME_BINNED", "ABGSQFT_BINNED", "VAC_BINNED"] 
                   #"COUNTY", "AGE_BINNED", "ACRES_BINNED", "ZONING", "POPDEN_BINNED", "OWN_BINNED", "DIRECT"

# Calculate MAD confidence intervals
mad_ci = sold.groupby(groupby_columns)['CURRPRICE'].apply(mad_confidence_interval).reset_index()
mad_ci.columns = groupby_columns + ['stat', 'value']
mad_ci = mad_ci.pivot(index=groupby_columns, columns='stat', values='value')

# Filter out rows where count is less than 3
mad_ci = mad_ci[mad_ci['count'] >= 3]

mad_ci = mad_ci.sort_values(groupby_columns + ['median'], ascending=[True] * len(groupby_columns) + [False])

# Print results and save to CSV
with open(r'C:\cctaddr\RES_CONFIDENCE_SOLD_DETAILED_CURRPRICE.csv', 'w') as f:
    print("MAD Confidence Intervals for CURRPRICE by all specified columns:")
    print(mad_ci)
    f.write("MAD Confidence Intervals for CURRPRICE by all specified columns:\n")
    f.write(mad_ci.to_string())

# Save results to Excel
mad_ci.to_excel(r'C:\cctaddr\RES_CONFIDENCE_SOLD_DETAILED_CURRPRICE.xlsx')

print("Results saved to C:\\cctaddr\\RES_CONFIDENCE_SOLD_DETAILED_CURRPRICE.csv and C:\\cctaddr\\RES_CONFIDENCE_SOLD_DETAILED_CURRPRICE.xlsx")