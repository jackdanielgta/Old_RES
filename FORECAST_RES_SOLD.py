import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Load the data
data = pd.read_csv(r"C:\Users\jackd\Downloads\RES_SOLD_RESCUE.csv", low_memory=False)

# Convert 'CLOSEDATE' to datetime format
data['CLOSEDATE'] = pd.to_datetime(data['CLOSEDATE'], errors='coerce')

# Function to choose income category
def choose_income(row):
    if 1 < row["MHIA21"] <= 250000:
        return row["MHIA21"]
    else:
        return row["AHIA21"]

data["INCOME"] = data.apply(choose_income, axis=1)

# Define income ranges
def income_category(income):
    if income <= 50000:
        return '0-50000'
    elif income <= 60000:
        return '50000-60000'
    elif income <= 70000:
        return '60000-70000'
    elif income <= 100000:
        return '70000-100000'
    elif income <= 125000:
        return '100000-125000'
    elif income <= 150000:
        return '125000-150000'
    else:
        return 'over 150000'

data['INCOME_CATEGORY'] = data['INCOME'].apply(income_category)

# Filter properties sold over 6 months ago
six_months_ago = datetime.now() - timedelta(days=6*30)
data_old = data[data['CLOSEDATE'] < six_months_ago]

# Group by 'GEOID' and forecast current prices for each region
forecasted_prices = []

# Function to forecast current price using ARIMA
def forecast_current_price(group):
    # Ensure the group is sorted by date
    group = group.sort_values(by='CLOSEDATE')
    
    # Remove duplicates in 'CLOSEDATE'
    group = group.loc[~group['CLOSEDATE'].duplicated(keep='first')]
    
    # Set 'CLOSEDATE' as index
    group = group.set_index('CLOSEDATE')
    
    # Ensure there are enough observations
    if len(group) < 10:  # You can adjust the threshold as needed
        return np.nan
    
    # Fill missing dates for the time series continuity
    group = group.asfreq('D').ffill()
    
    # Fit the ARIMA model
    try:
        model = ARIMA(group['CLOSEPRICE'], order=(5, 1, 0))
        model_fit = model.fit()
        
        # Forecast the price to the current date
        steps = (datetime.now() - group.index[-1]).days
        forecast = model_fit.forecast(steps=steps).iloc[-1] if steps > 0 else group['CLOSEPRICE'].iloc[-1]
    except Exception as e:
        forecast = np.nan
    
    return forecast

# Apply forecasting function with fallback grouping
for geoid, group in data_old.groupby('GEOID'):
    current_price_forecast = forecast_current_price(group)
    if np.isnan(current_price_forecast):
        # Fallback to INCOME and CITY grouping
        for (income, city), sub_group in group.groupby(['INCOME_CATEGORY', 'CITY']):
            sub_group_forecast = forecast_current_price(sub_group)
            forecasted_prices.extend([sub_group_forecast] * len(sub_group))
    else:
        forecasted_prices.extend([current_price_forecast] * len(group))

# Add the forecasted prices as a new column
data_old['FORECASTED_CLOSEPRICE'] = forecasted_prices

# Save the updated DataFrame with forecasted prices to a CSV file
data_old.to_csv(r"C:\cctaddr\SOLD_DATA_TEST4.csv", index=False)

print(data_old[['MLS1', 'GEOID', 'CITY', 'INCOME_CATEGORY', 'CLOSEDATE', 'CLOSEPRICE', 'FORECASTED_CLOSEPRICE']])