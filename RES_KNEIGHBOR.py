import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, median_absolute_error

downpayment = 0.2
interest_rate = 0.0766
Amoritization = 30

# Load the data
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')
listing = pd.read_csv(r"C:\Users\jackd\Downloads\newnewnew1.csv", low_memory=False, encoding='latin1')

# Update and round BEDSTOTAL & ABGSQFT for specific property types
multi_family_types = ["MF", "2F", "3F", "4F"]
for df in [sold, listing]:
    mask = df['PROPTYPE'].isin(multi_family_types)
    df.loc[mask, "BEDSTOTAL"] = (df.loc[mask, "BEDSTOTAL"] / df.loc[mask, "TOTAL_UNIT"]).round(0)
    df.loc[mask, "ABGSQFT"] = (df.loc[mask, "ABGSQFT"] / df.loc[mask, "TOTAL_UNIT"]).round(0)

# Update 'PROPTYPE' based on 'STYLE'
for df in [sold, listing]:
    df.loc[df['STYLE'].str.contains('Townhouse', na=False) & (df['PROPTYPE'] == 'CO'), 'PROPTYPE'] = 'TH'
    df.loc[df['STYLE'].str.contains('Mobile Home', na=False) & (df['PROPTYPE'] == 'SF'), 'PROPTYPE'] = 'MH'
    df.loc[df['STYLE'].str.contains('Cape Cod', na=False) & (df['PROPTYPE'] == 'SF'), 'PROPTYPE'] = 'SC'

# Function to choose income
def choose_income(row):
    return row["MHIA21"] if 1 < row["MHIA21"] <= 250000 else row["AHIA21"]

for df in [sold, listing]:
    df["INCOME"] = df.apply(choose_income, axis=1)

def calculate_carry_costs(df, price_col):
    df['HOA'].fillna(0, inplace=True)
    df['CARRY_COSTS'] = (df[price_col] / 1000) * 4.2 / 12 + df['HOA'] + (df['PROPTAX'] / 12)

calculate_carry_costs(sold, 'CLOSEPRICE')
calculate_carry_costs(listing, 'CURRPRICE')

monthly_interest_rate = interest_rate / 12

# Define imputation function
def impute_with_combined_group_median(dataframe, target_column, group_columns):
    group_medians = dataframe.groupby(group_columns, observed=True)[target_column].transform('median')
    dataframe[target_column].fillna(group_medians, inplace=True)

# Calculate additional features for both sold and listing DataFrames
for df in [sold, listing]:
    df["OWNERSHIP_%"] = df["OWNOCCA21"] / df["TOTHSGA21"]
    df["POP_DENSITY"] = (df["TOTPOPA21"] / df["ALAND"] * 0.386102) * 1000
    df['VACANCY'] = df["VACANTA21"] / df["HHA21"]
    
    price_col = 'CLOSEPRICE' if 'CLOSEPRICE' in df.columns else 'CURRPRICE'
    df["MORT_PAY"] = (df[price_col] * (1 - downpayment)) * (monthly_interest_rate * (1 + monthly_interest_rate) ** (12 * Amoritization)) / ((1 + monthly_interest_rate) ** (12 * Amoritization) - 1)
    df["COMPLETE_PAYMENT"] = df["MORT_PAY"] + df["CARRY_COSTS"]
    df["DTI_RATIO"] = df["COMPLETE_PAYMENT"] / (df["INCOME"] / 12)

    # Replace placeholder values with NaN
    placeholder_values = [np.inf, -np.inf, 999999999, 99999999, 9999999, 999999, 99999, 0, 1, 100000000]
    df['PROPTAX'].replace(placeholder_values, np.nan, inplace=True)

    # Create ABGSQFT bins
    bin_edges = [0, 650, 850, 1050, 1300, 1600, 2150, 2600, 3300, 4300, np.inf]
    bin_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    df['ABGSQFT_BINNED'] = pd.cut(df['ABGSQFT'], bins=bin_edges, labels=bin_labels, right=False)

    # Apply imputation
    impute_with_combined_group_median(df, 'PROPTAX', ['GEOID', 'ABGSQFT_BINNED'])

# Separate numeric and categorical features
numeric_features = ["ABGSQFT", 'TOT_SQFT', "INCOME", "MHIA21", "AHIA21", 'CARRY_COSTS', "OWNERSHIP_%", "POP_DENSITY", "MORT_PAY", "COMPLETE_PAYMENT", "DTI_RATIO", 'VACANCY', "ACRES", "BEDSTOTAL", "YEARBUILT", "DOM", 'HOA', 'PROPTAX', "OWNOCCA21", "TOTHSGA21", "TOTPOPA21", "ALAND", "LATITUDE", "LONGITUDE", "VACANTA21", "HHA21", "BATHSTOTAL", "TOTAL_UNIT", "GARAGEN"]
categorical_features = ["GEOID", "PROPTYPE", "CITY", "STYLE", "DIRECT", "COUNTY", "BANK_OWNED", "FLOOD_ZN"]

features = numeric_features + categorical_features

# Function to clean numeric data
def clean_numeric_data(df, columns):
    for col in columns:
        if df[col].dtype in ['int64', 'float64']:
            # Replace infinity with NaN
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Clip values to 3 standard deviations from the mean
            mean = df[col].mean()
            std = df[col].std()
            df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)
    
    return df

# Function to convert categorical columns to strings
def convert_categorical_to_string(df, categorical_columns):
    for col in categorical_columns:
        df[col] = df[col].astype(str)
    return df

# Apply cleaning to both dataframes
sold = clean_numeric_data(sold, numeric_features)
listing = clean_numeric_data(listing, numeric_features)

# Convert categorical features to strings
sold = convert_categorical_to_string(sold, categorical_features)
listing = convert_categorical_to_string(listing, categorical_features)

# Function to clean and prepare data
def clean_data(df, is_listing=False):
    price_col = 'CURRPRICE' if is_listing else 'CLOSEPRICE'
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    
    columns_to_keep = ['MLS1'] + features + [price_col]
    df = df[columns_to_keep].copy()
    
    df = df.dropna(subset=features, how='all')
    
    return df

# Clean and prepare the data
sold = clean_data(sold)
listing = clean_data(listing, is_listing=True)

# Remove rows with NaN in CLOSEPRICE
sold = sold.dropna(subset=['CLOSEPRICE'])
y_sold = sold['CLOSEPRICE']

# Create preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a pipeline with preprocessing and KNN
knn = Pipeline(steps=[('preprocessor', preprocessor),
                      ('knn', KNeighborsRegressor(n_neighbors=5, weights='distance'))])

# Function to check column issues
def check_column_issues(df, col):
    if df[col].dtype in ['int64', 'float64']:
        return df[col].isnull().sum(), np.isinf(df[col]).sum()
    else:
        return df[col].isnull().sum(), 0  # Non-numeric columns can't have inf values

# Fit the pipeline
try:
    knn.fit(sold[features], y_sold)
except ValueError as e:
    print(f"Error during model fitting: {e}")
    print("Problematic columns:")
    for col in features:
        nan_count, inf_count = check_column_issues(sold, col)
        if nan_count > 0 or inf_count > 0:
            print(f"{col}: {nan_count} NaN values, {inf_count} infinite values")
    raise

# Predict prices for the listing data
predicted_prices = knn.predict(listing[features])

# Calculate the mean absolute percentage error (MAPE) of the model on the sold data
y_pred_sold = knn.predict(sold[features])
mape = np.mean(np.abs((y_sold - y_pred_sold) / y_sold)) * 100
print(f"MAPE on sold data: {mape}")

# Calculate MAPE for listing data (excluding NaN values)
valid_mask = ~listing['CURRPRICE'].isna()
y_true_listing = listing.loc[valid_mask, 'CURRPRICE']
y_pred_listing = predicted_prices[valid_mask]
mape_listing = np.mean(np.abs((y_true_listing - y_pred_listing) / y_true_listing)) * 100
print(f"MAPE on listing data: {mape_listing}")

# Create a price range based on the MAPE
listing['PREDICTED_PRICE'] = predicted_prices
listing['PRICE_MIN'] = listing['PREDICTED_PRICE'] * (1 - mape_listing/100)
listing['PRICE_MAX'] = listing['PREDICTED_PRICE'] * (1 + mape_listing/100)

# Reorder columns to have MLS1 first
columns_order = ['MLS1'] + [col for col in listing.columns if col != 'MLS1']
listing = listing[columns_order]

# Print the results
print("\nFinal results:")
print(listing[['MLS1', 'CURRPRICE', 'PREDICTED_PRICE', 'PRICE_MIN', 'PRICE_MAX']])

# Save the results to a CSV file
listing.to_csv('listing_with_predicted_prices.csv', index=False)

# Remove rows with NaN in CURRPRICE for statistics calculation
listing_valid = listing.dropna(subset=['CURRPRICE'])

# Print statistics
print("\nStatistics:")
print("MAPE on listing data:", mape_listing)
print("Correlation between CURRPRICE and PREDICTED_PRICE:", 
      listing_valid['CURRPRICE'].corr(listing_valid['PREDICTED_PRICE']))
print("Mean difference between CURRPRICE and PREDICTED_PRICE:", 
      (listing_valid['CURRPRICE'] - listing_valid['PREDICTED_PRICE']).mean())
print("Median difference between CURRPRICE and PREDICTED_PRICE:", 
      (listing_valid['CURRPRICE'] - listing_valid['PREDICTED_PRICE']).median())
print("R-squared score:", r2_score(listing_valid['CURRPRICE'], listing_valid['PREDICTED_PRICE']))
print("Mean Absolute Error:", mean_absolute_error(listing_valid['CURRPRICE'], listing_valid['PREDICTED_PRICE']))
print("Median Absolute Error:", median_absolute_error(listing_valid['CURRPRICE'], listing_valid['PREDICTED_PRICE']))