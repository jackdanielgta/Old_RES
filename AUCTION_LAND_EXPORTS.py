import pandas as pd
import numpy as np
from geopy.geocoders import ArcGIS
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from time import sleep

# Load the data from the Excel file, skipping the first three rows
hubzu = pd.read_excel(r"C:\Users\jackd\Desktop\AUCTION-LAND EXPORTS\Hubzu.xlsx", skiprows=3)

# Select specific columns from the DataFrame
hubzu = hubzu[["Property Address", "County", "List Price", "Listing status", "Property Type", "Beds", "Baths", "Area"]]
hubzu.columns = ["ADDRESS", "COUNTY", "CURRPRICE", "STATUS", "PROPTYPE", "BEDSTOTAL", "BATHSTOTAL", "ABGSQFT"]

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

# Apply the function to each address and store results in new columns
hubzu['LATITUDE'] = np.nan
hubzu['LONGITUDE'] = np.nan

for i, row in hubzu.iterrows():
    address = row['ADDRESS']
    lat, lon = get_lat_lon(address)
    hubzu.at[i, 'LATITUDE'] = lat
    hubzu.at[i, 'LONGITUDE'] = lon
    # Sleep to avoid overloading the geocoding service
    sleep(1)

# Save the updated DataFrame to a new CSV file
hubzu.to_csv(r'C:\cctaddr\ADDITIONAL_EXPORTS.csv', index=False)

print("Geocoding completed and CSV file saved.")