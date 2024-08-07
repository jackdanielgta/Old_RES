import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load the dataset
file_path = r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv"
data = pd.read_csv(file_path, encoding='latin1')

# Define columns
numerical_cols = data.select_dtypes(include=['number']).columns.drop('PROPTAX')
categorical_cols = data.select_dtypes(include=['object']).columns

# Create a preprocessing pipeline with imputation and one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ]
)

# Separate features and target variable
X = data.drop(columns=['PROPTAX'])
y = data['PROPTAX']

# Preprocess the data
X_encoded = preprocessor.fit_transform(X)

# Handle any missing values in the target column by dropping them
mask = ~y.isna()
X_encoded = X_encoded[mask]
y = y[mask]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Extract feature importances
feature_importances = model.feature_importances_

# Get the feature names after encoding
encoded_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)
all_feature_names = np.append(encoded_feature_names, numerical_cols)

# Create a DataFrame to display feature importances
feature_importances_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Display the top features
print(feature_importances_df.head(10))
