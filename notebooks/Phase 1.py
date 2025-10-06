# PHASE 1: DATA PREPROCESSING FOR NASA EXOPLANET ARCHIVE DATASET

import pandas as pd
import numpy as np

data = pd.read_csv(
    "data/kepler_koi.csv", # can be replaced with another dataset (.csv) from the NASA Exoplanet Archive.
    skiprows=53, # you should change the header row. In this case, the header is at row 54.
    engine='python',
    on_bad_lines='skip'
)
mapping = {
    "CONFIRMED": 1,
    "FALSE POSITIVE": 0,
    "CANDIDATE": 2,
}

data["koi_disposition"] = data["koi_disposition"].map(mapping) # intended to avoid reading errors

data = data.dropna(thresh=int(0.7 * data.shape[1])) # threshold of 70% of data from each row by full columns
for col in data.select_dtypes(include=np.number).columns: 
    data[col] = data[col].fillna(data[col].median()) # fill missing values with median for numerical columns

# Feature engineering ratios
data["depth_duration_ratio"] = data["koi_depth"] / data["koi_duration"] # indicates how deep the transit is in relation to its duration.
data["radius_period_ratio"] = data["koi_prad"] / data["koi_period"] # indicates the scale of the planet in relation to its orbit.
data["impact_duration_ratio"] = data["koi_impact"] / data["koi_duration"] # indicates how the impact parameter relates to the transit duration.
data["insol_radius_ratio"] = data["koi_insol"] / data["koi_prad"] # indicates how the insolation flux relates to the planet radius.

data.to_csv("data/kepler_koi_processed.csv", index=False) # store clear dataset in the "data" folder
print("Processed columns:", data.columns)