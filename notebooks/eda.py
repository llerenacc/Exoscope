# EXPLORATORY DATA ANALYSIS (EDA) FOR NASA EXOPLANET ARCHIVE DATASET

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv( # stored in the "data" folder
    "data/kepler_koi.csv", # can be replaced with another dataset (.csv) from the NASA Exoplanet Archive.
    skiprows=53, # you should change the header row. In this case, the header is at row 54.
    engine='python',
    on_bad_lines='skip' # in case of bad lines, skip them.
)

# Sampling
print("First rows of dataset:")
print(df.head()) # displays the first 5 rows of the dataset.

# General information
print("\nDataset information:")
print(df.info())

# Basic statistics
print("\nDescriptive statistics:")
print(df.describe()) # provides basic statistics for numerical columns.

# Count of values ​​in the koi_disposition column
if "koi_disposition" in df.columns:
    print("\nClass distribution (koi_disposition):")
    print(df["koi_disposition"].value_counts())
    sns.countplot(data=df, x="koi_disposition")
    plt.title("Exoplanets distribution by disposition")
    plt.show()