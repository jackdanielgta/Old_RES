import pandas as pd
import numpy as np
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from time import sleep
import re

# Initialize the geocoder
geolocator = ArcGIS(user_agent="geoapiExercises")

# Function to get latitude and longitude with retries
def get_lat_lon(address, max_retries=3):
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(f"{address}, USA", timeout=30)
            if location:
                return location.latitude, location.longitude
            else:
                print(f"No results found for address: {address}")
                return None, None
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            if attempt < max_retries - 1:
                print(f"Error geocoding address {address}: {e}. Retrying...")
                sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Error geocoding address {address}: {e}")
                return None, None
    return None, None

# Function to process each Excel file
def process_file(file_path, skiprows, file_type):
    # Load the data from the Excel file, skipping the specified rows
    df = pd.read_excel(file_path, skiprows=skiprows)

    # Select and rename columns based on file type
    if file_type == "Hubzu":
        df = df[["Property Address", "County", "List Price", "Listing status", "Property Type", "Beds", "Baths", "Area"]]
        df.columns = ["ADDRESS", "COUNTY", "CURRPRICE", "STATUS", "PROPTYPE", "BEDSTOTAL", "BATHSTOTAL", "ABGSQFT"]
        
        # Standardize address format and extract city, state, ZIP
        def standardize_address(address):
            parts = address.split(',')
            if len(parts) >= 2:
                street = parts[0].strip()
                city_state_zip = parts[-1].strip().split()
                if len(city_state_zip) >= 3:
                    city = ' '.join(city_state_zip[:-2])
                    state = city_state_zip[-2]
                    zip_code = city_state_zip[-1]
                    return f"{street}, {city}, {state} {zip_code}", city, state, zip_code
            return address, None, None, None

        df['ADDRESS'], df['CITY'], df['STATE'], df['ZIP'] = zip(*df['ADDRESS'].apply(standardize_address))

    elif file_type == "Xome":
        df = df[["Address Line", "City", "State or Provinces Code", "Postal Code", "County", "Current Bid", "Bedrooms", "Bathrooms", "Square Footage", "Lot Size", "Asset Type"]]
        df.columns = ["ADDRESS", "CITY", "STATE", "ZIP", "COUNTY", "CURRPRICE", "BEDSTOTAL", "BATHSTOTAL", "ABGSQFT", "LOTSIZE", "PROPTYPE"]
        # Convert ZIP to string and handle NaN values
        df['ZIP'] = df['ZIP'].astype(str).replace('nan', '')
        df['ADDRESS'] = df['ADDRESS'] + ' ' + df['CITY'] + ' ' + df['STATE'] + ' ' + df['ZIP']

    elif file_type == "Auction":
        df = df[["Property Address", "County", "State", "Zip", "Property Type", "Home Square Footage", "Bedrooms", "Baths", "Lot Size", "Starting Bid"]]
        df.columns = ["ADDRESS", "COUNTY", "STATE", "ZIP", "PROPTYPE", "ABGSQFT", "BEDSTOTAL", "BATHSTOTAL", "LOTSIZE", "CURRPRICE"]
        # Convert ZIP to string and handle NaN values
        df['ZIP'] = df['ZIP'].astype(str).replace('nan', '')
        
        # Extract city from address
        def extract_city(address):
            parts = address.split(',')
            if len(parts) >= 2:
                return parts[-2].strip()
            return None

        df['CITY'] = df['ADDRESS'].apply(extract_city)

    # Apply the function to each address and store results in new columns
    df['LATITUDE'] = np.nan
    df['LONGITUDE'] = np.nan

    for i, row in df.iterrows():
        address = row['ADDRESS']
        lat, lon = get_lat_lon(address)
        df.at[i, 'LATITUDE'] = lat
        df.at[i, 'LONGITUDE'] = lon
        # Sleep to avoid overloading the geocoding service
        sleep(1)

    # Add ACRES field
    if 'LOTSIZE' in df.columns:
        df['ACRES'] = df['LOTSIZE'].apply(lambda x: float(x.split()[0]) if isinstance(x, str) and 'acres' in x.lower() else np.nan)
    else:
        df['ACRES'] = np.nan

    # Add SOURCE field
    df['SOURCE'] = file_type

    return df

# Process each Excel file
hubzu_df = process_file(r"C:\Users\jackd\Desktop\AUCTION-LAND EXPORTS\Hubzu.xlsx", skiprows=3, file_type="Hubzu")
xome_df = process_file(r"C:\Users\jackd\Desktop\AUCTION-LAND EXPORTS\Xome.xlsx", skiprows=1, file_type="Xome")
auction_df = process_file(r"C:\Users\jackd\Desktop\AUCTION-LAND EXPORTS\auction.com.xlsx", skiprows=0, file_type="Auction")

# Merge all dataframes
merged_df = pd.concat([hubzu_df, xome_df, auction_df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(r'C:\cctaddr\MERGED_EXPORTS.csv', index=False)

print("Geocoding completed and merged CSV file saved.")