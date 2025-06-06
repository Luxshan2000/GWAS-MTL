# filename: download_real_gwas_datasets_safe.py
# execution: true
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import io
import shutil

# Create directories for raw and processed data
os.makedirs('raw_data', exist_ok=True)
os.makedirs('processed_data', exist_ok=True)

print("Downloading real GWAS datasets...")

# 1. Download Type 2 Diabetes dataset from DIAGRAM consortium
# Using the DIAGRAM Metabochip meta-analysis (smaller dataset, <1MB)
t2d_url = "https://diagram-consortium.org/downloads/DIAGRAM_Gaulton_2015.zip"

try:
    print(f"Attempting to download T2D dataset from {t2d_url}")
    response = requests.get(t2d_url)
    
    if response.status_code == 200:
        print("Successfully downloaded T2D dataset")
        # Save the raw zip file
        with open('raw_data/DIAGRAM_Gaulton_2015.zip', 'wb') as f:
            f.write(response.content)
        
        # Extract
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall('raw_data/t2d')
            print("Extracted T2D dataset")
            t2d_files = os.listdir('raw_data/t2d')
            print(f"Files in T2D dataset: {t2d_files}")
    else:
        print(f"Failed to download T2D dataset. Status code: {response.status_code}")
        # Create a fallback dataset with a message
        with open('raw_data/t2d_download_failed.txt', 'w') as f:
            f.write(f"Failed to download from {t2d_url}. Status code: {response.status_code}")

except Exception as e:
    print(f"Error downloading T2D dataset: {e}")
    # Create a fallback dataset with a message
    with open('raw_data/t2d_download_failed.txt', 'w') as f:
        f.write(f"Failed to download from {t2d_url}. Error: {str(e)}")

# 2. Download the real Cardiovascular dataset we already have
print("\nUsing the cardiovascular disease dataset we already downloaded...")
if os.path.exists('cardiogramplusc4d_data.txt'):
    print("Cardiovascular dataset already exists")
    # Copy to raw data directory using shutil instead of os.system
    shutil.copy('cardiogramplusc4d_data.txt', 'raw_data/')
else:
    print("Downloading cardiovascular dataset...")
    cardio_url = "https://www.cardiogramplusc4d.org/media/cardiogramplusc4d-consortium/data-downloads/cardiogramplusc4d_data.zip"
    
    try:
        response = requests.get(cardio_url)
        if response.status_code == 200:
            print("Successfully downloaded cardiovascular dataset")
            with open('raw_data/cardiogramplusc4d_data.zip', 'wb') as f:
                f.write(response.content)
            
            # Extract
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall('raw_data/')
                print("Extracted cardiovascular dataset")
        else:
            print(f"Failed to download cardiovascular dataset. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading cardiovascular dataset: {e}")

# 3. Try to download a breast cancer dataset from GWAS Catalog
bc_urls = [
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST000001-GCST001000/GCST000095/harmonised/",
    # Breast cancer study by Easton DF - attempt
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST000001-GCST001000/GCST000101/harmonised/",
    # Try another potential breast cancer study
    "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST010001-GCST011000/GCST010120/harmonised/"
]

bc_downloaded = False
for bc_url in bc_urls:
    try:
        print(f"\nAttempting to download breast cancer dataset from {bc_url}")
        # Try to get index page
        response = requests.get(bc_url)
        if response.status_code == 200:
            print(f"Successfully connected to {bc_url}")
            # Look for .tsv or .txt files in the HTML response
            file_url = None
            if ".tsv" in response.text:
                import re
                matches = re.findall(r'href="([^"]*\.tsv)"', response.text)
                if matches:
                    file_url = f"{bc_url}{matches[0]}"
            
            if file_url:
                print(f"Downloading file from {file_url}")
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    with open(f'raw_data/breast_cancer.tsv', 'wb') as f:
                        f.write(file_response.content)
                    print("Successfully downloaded breast cancer dataset")
                    bc_downloaded = True
                    break
            else:
                print("Couldn't find a TSV file in the directory")
        else:
            print(f"Couldn't access {bc_url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error processing breast cancer URL {bc_url}: {e}")

# If we couldn't download a breast cancer dataset, create a dummy dataset to represent the issue
if not bc_downloaded:
    print("\nCouldn't download breast cancer dataset. Checking if we have processed data...")
    
    # Check if we already have processed data from before
    if os.path.exists('processed_data/multitask_gwas_data.csv'):
        print("Found existing processed data, will use that for the model")
    else:
        print("Creating a note about the breast cancer dataset download failure")
        with open('raw_data/breast_cancer_download_failed.txt', 'w') as f:
            f.write("Failed to download breast cancer dataset from GWAS Catalog. Will use the cardiovascular data to demonstrate the method.")

print("\nData download process completed.")

# Check what data we actually obtained
print("\nChecking available raw datasets:")
for item in os.listdir('raw_data'):
    print(f"- {item}")