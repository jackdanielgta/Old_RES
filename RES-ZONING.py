import pandas as pd
import re
import numpy as np
from datetime import datetime
from scipy.stats import percentileofscore
from scipy import stats

# Load CSV files
sold = pd.read_csv(r"C:\Users\jackd\Desktop\IMPORTANT_RES_MF_09-21-2021-06-30-2024.csv", low_memory=False, encoding='latin1')
listing = pd.read_csv(r"C:\Users\jackd\Downloads\righthererightnownow.csv", low_memory=False, encoding='latin1')

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
sold['ZONING_CATEGORY'] = sold['ZONING'].apply(categorize_zoning)
listing['ZONING_CATEGORY'] = listing['ZONING'].apply(categorize_zoning)

# Group by the new category
sdf = sold.groupby("ZONING_CATEGORY").size().reset_index(name='COUNT')
ldf = listing.groupby("ZONING_CATEGORY").size().reset_index(name='COUNT')

print(sdf)
sdf.to_csv(r'C:\cctaddr\RES_ZONING_SOLD.csv', index=False)

print(ldf)
ldf.to_csv(r'C:\cctaddr\RES_ZONING_LISTING.csv', index=False)