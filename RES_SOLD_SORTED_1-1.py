import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the provided CSV file
file_path = "C:\\cctaddr\\RES_SOLD_SORTED_UPDATED.csv"
df = pd.read_csv(file_path)

# Encoding categorical features
label_encoders = {}
categorical_cols = ['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Separate data into features and target
features = df[['CLOSEYEAR', 'CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'ABGSQFT', 'TOT_SQFT', 'ACRES', 'INCOME', 'YEAR_BUILT']]
target = df['median_closeprice']

# Create a combination of all unique values for CITY, PROPTYPE, DIRECT, and TAX_BIN
unique_combinations = df[['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN']].drop_duplicates()

# Determine the range of years in the dataset
min_year = int(df['CLOSEYEAR'].min())
max_year = int(df['CLOSEYEAR'].max())

# Create a DataFrame with all combinations of unique values and years
all_years = pd.DataFrame({'CLOSEYEAR': range(min_year, max_year + 1)})
all_combinations = unique_combinations.merge(all_years, how='cross')

# Merge with the original dataframe to identify missing combinations
merged_df = all_combinations.merge(df, on=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'CLOSEYEAR'], how='left')

# Identify rows with missing median_closeprice
missing_groups = merged_df[merged_df['median_closeprice'].isna()]

# Prepare training data using existing non-missing values
train_data = df[df['median_closeprice'].notna()]
predict_data = missing_groups

X_train = train_data[features.columns]
y_train = train_data['median_closeprice']
X_predict = predict_data[features.columns]

# Impute missing values in the feature set
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_predict_imputed = imputer.transform(X_predict)

# Train the RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train_imputed, y_train)

# Predict the missing values
predicted_closeprice = model.predict(X_predict_imputed)

# Fill in the predicted values in the original dataframe
missing_groups['median_closeprice'] = predicted_closeprice

# Merge the predicted missing groups back into the original dataframe
filled_df = pd.concat([df, missing_groups])

# Decode the encoded categorical columns back
for col in categorical_cols:
    le = label_encoders[col]
    filled_df[col] = le.inverse_transform(filled_df[col])

# Sort the DataFrame as per the specified columns
sorted_filled_df = filled_df.sort_values(by=['CITY', 'PROPTYPE', 'DIRECT', 'TAX_BIN', 'CLOSEYEAR'])

# Save the updated dataframe to a new CSV file
output_file_path_filled = 'C:\\cctaddr\\RES_SOLD_FILLED.csv'
sorted_filled_df.to_csv(output_file_path_filled, index=False)

print(f"Updated file saved to {output_file_path_filled}")