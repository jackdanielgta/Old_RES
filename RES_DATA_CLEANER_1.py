import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from scipy import stats as scipy_stats

# Define constants
today = datetime.now()
one_year_ago = today - timedelta(days=365)
current_year = datetime.now().year

# Load CSV file
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')

# Data cleaning and transformation
sold['CLOSEDATE'] = pd.to_datetime(sold['CLOSEDATE'], errors='coerce')
placeholder_values = [np.inf, -np.inf, 999999999, 99999999, 9999999, 999999, 99999, 0, 1, 100000000]
sold['PROPTAX'].replace(placeholder_values, np.nan, inplace=True)

# Create AGE_BINNED for sold DataFrame
newbuild_threshold = current_year - 15
age_bin_edges = [-np.inf, 1935, newbuild_threshold, np.inf]
age_bin_labels = [0, 1, 2]
sold['AGE_BINNED'] = pd.cut(sold['YEARBUILT'], bins=age_bin_edges, labels=age_bin_labels, right=False)

# Update 'PROPTYPE' based on 'STYLE'
sold.loc[sold['STYLE'].str.contains('Townhouse', na=False) & (sold['PROPTYPE'] == 'CO'), 'PROPTYPE'] = 'TH'
sold.loc[sold['STYLE'].str.contains('Mobile Home', na=False) & (sold['PROPTYPE'] == 'SF'), 'PROPTYPE'] = 'MH'
sold.loc[sold['STYLE'].str.contains('Cape Cod', na=False) & (sold['PROPTYPE'] == 'SF'), 'PROPTYPE'] = 'SC'

# Function to categorize zoning
def categorize_zoning(zone):
    zone = str(zone).upper().replace('-', '').replace(' ', '').replace('_', '')
    
    # Residential Zones
    if re.match(r'^(R|RA|RB|RC|RD|RE|RF|RG|RH|RR|RS|RU|RAA|RAR|RES|RESA|RESB)', zone):
        return zone[:2] if zone.startswith('R') else zone[:3]
    
    # Multi-Family Residential
    elif re.match(r'^(RM|RMF|MFDD|MFR|APT)', zone):
        return zone[:3] if zone.startswith('RM') else zone[:4]
    
    # Residential by Lot Size/Density
    elif re.match(r'^R\d+', zone):
        match = re.match(r'^R(\d+)', zone)
        return f'R-{match.group(1)}'
    
    # Age-Restricted/Special Residential
    elif any(zone.startswith(s) for s in ['ARHD', 'PRD', 'PUD', 'HOD']):
        return zone[:3] if zone.startswith('PUD') else zone[:4]
    
    # Commercial Zones
    elif re.match(r'^(C|CB|CN|CG|CA|CL|CT|B|BA|BB|BC|BD|BG|BR|BUS)', zone):
        return zone[:2] if len(zone) > 2 else zone
    
    # Mixed-Use
    elif re.match(r'^(MX|MU|MUP)', zone):
        return zone[:2] if zone.startswith('M') else zone[:3]
    
    # Industrial Zones
    elif re.match(r'^(I|IL|IND|IP|LI)', zone):
        return zone[:2] if zone.startswith('I') else 'LI'
    
    # Agricultural Zones
    elif re.match(r'^(A|AG|AAA)', zone):
        return 'A' if zone.startswith('A') else zone[:3]
    
    # Open Space/Conservation
    elif re.match(r'^(OS|RROS|CONS)', zone):
        return zone[:2] if zone.startswith('OS') else zone[:4]
    
    # Planned Development
    elif re.match(r'^(PD|PDD|PAD|PRD)', zone):
        return zone[:3]
    
    # Village/Town Center
    elif re.match(r'^(VC|TC|TCD)', zone):
        return zone[:2] if zone.startswith('V') or zone.startswith('TC') else zone[:3]
    
    # Overlay Zones
    elif re.match(r'^(OSD|HOD|WD)', zone):
        return zone[:3]
    
    # Waterfront
    elif re.match(r'^(WF|WD)', zone):
        return zone[:2]
    
    # Historic
    elif re.match(r'^(HD|HIS)', zone):
        return zone[:2] if zone.startswith('HD') else zone[:3]
    
    # Floating Zones
    elif re.match(r'^(FZ|FLEX)', zone):
        return zone[:2] if zone.startswith('FZ') else zone[:4]
    
    # Special Districts
    elif re.match(r'^(SD|SDD)', zone):
        return zone[:2] if zone.startswith('SD') else zone[:3]
    
    # Design Districts
    elif re.match(r'^(DD|DT)', zone):
        return zone[:2]
    
    # Transition Zones
    elif re.match(r'^(T|TR)', zone):
        return zone[:1] if zone.startswith('T') else zone[:2]
    
    # For any other cases
    return 'Other'

# Apply the categorization to your dataframe
sold['ZONING'] = sold['ZONING'].apply(categorize_zoning)

# Function to calculate price per square foot safely
def safe_price_per_sqft(price, sqft):
    return np.where(sqft > 0, price / sqft, np.nan)

# Add price per square foot column
sold['PRICE_SQFT'] = safe_price_per_sqft(sold['CLOSEPRICE'], sold['ABGSQFT'])

# Function to calculate statistics and threshold
def calculate_stats_and_threshold(group):
    p_sqft = group['PRICE_SQFT'].dropna()
    if len(p_sqft) == 0:
        return pd.Series({
            'MEAN_P_SQFT': np.nan,
            'MAD_P_SQFT': np.nan,
            'MIN_P_SQFT': np.nan,
            'SIZE_COUNT': 0,
            'THRESHOLD': np.nan
        })
    stats_data = pd.Series({
        'MEAN_P_SQFT': np.mean(p_sqft),
        'MAD_P_SQFT': np.median(np.abs(p_sqft - np.mean(p_sqft))),
        'MIN_P_SQFT': np.min(p_sqft),
        'SIZE_COUNT': len(p_sqft),
        'THRESHOLD': np.percentile(p_sqft, 5) if len(p_sqft) > 1 else np.nan
    })
    return stats_data

# Function to recursively calculate stats and filter data
def recursive_stats_and_filter(group, end_date, min_count=2):
    start_date = end_date - timedelta(days=365)
    current_year_data = group[(group['CLOSEDATE'] >= start_date) & (group['CLOSEDATE'] < end_date)]
    
    stats_data = calculate_stats_and_threshold(current_year_data)
    
    filtered_current_year = current_year_data
    filtered_older_data = pd.DataFrame()
    
    if pd.notnull(stats_data['THRESHOLD']):
        older_data = group[group['CLOSEDATE'] < start_date]
        filtered_older_data = older_data[older_data['PRICE_SQFT'] > stats_data['THRESHOLD']]
    
    combined_data = pd.concat([filtered_current_year, filtered_older_data])
    
    if combined_data.shape[0] >= min_count or start_date <= group['CLOSEDATE'].min():
        return combined_data, stats_data, start_date
    else:
        return recursive_stats_and_filter(group, start_date, min_count)

# Calculate stats and threshold for each group
group_stats = {}
filtered_data = []

for (city, prop_type, direct, geoid, age_bin), group in sold.groupby(['CITY', 'PROPTYPE', 'DIRECT', 'GEOID', 'AGE_BINNED'], observed=True):
    filtered_group, stats_data, start_date = recursive_stats_and_filter(group, one_year_ago)
    group_stats[(city, prop_type, direct, geoid, age_bin)] = stats_data
    filtered_data.append(filtered_group)

# Combine all filtered data
filtered_sold = pd.concat(filtered_data, ignore_index=True)

# Convert group_stats to DataFrame for easier merging
group_stats_df = pd.DataFrame.from_dict(group_stats, orient='index')
group_stats_df.index.names = ['CITY', 'PROPTYPE', 'DIRECT', 'GEOID', 'AGE_BINNED']
group_stats_df.reset_index(inplace=True)

# Merge group stats back to the filtered_sold DataFrame
filtered_sold = filtered_sold.merge(group_stats_df, on=['CITY', 'PROPTYPE', 'DIRECT', 'GEOID', 'AGE_BINNED'], how='left')

# Calculate MAD_THRESHOLD for each row
def calculate_mad_threshold(row):
    if np.isnan(row['PRICE_SQFT']) or np.isnan(row['MAD_P_SQFT']) or np.isnan(row['MEAN_P_SQFT']):
        return np.nan
    if row['MAD_P_SQFT'] == 0:
        return 50 if row['PRICE_SQFT'] == row['MEAN_P_SQFT'] else (0 if row['PRICE_SQFT'] < row['MEAN_P_SQFT'] else 100)
    else:
        z_score = (row['PRICE_SQFT'] - row['MEAN_P_SQFT']) / (1.4826 * row['MAD_P_SQFT'])
        return scipy_stats.norm.cdf(z_score) * 100

filtered_sold['MAD_THRESHOLD'] = filtered_sold.apply(calculate_mad_threshold, axis=1)

# Remove rows with NaN values in critical columns
filtered_sold = filtered_sold.dropna(subset=['CLOSEPRICE', 'ABGSQFT', 'MAD_THRESHOLD'])

# Print the first few rows of the filtered_sold DataFrame
print(filtered_sold.head())
# Save the filtered_sold DataFrame to a CSV file
filtered_sold.to_csv(r'C:\cctaddr\OFFICIAL_FILTERED_SOLD.csv', index=False)

# Group by the new ZONING
sdf = filtered_sold.groupby("ZONING").size().reset_index(name='COUNT')
print(sdf)
sdf.to_csv(r'C:\cctaddr\RES_ZONING_SOLD.csv', index=False)