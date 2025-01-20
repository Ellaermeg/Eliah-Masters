import os
import sys
sys.path.append("../Eliah-Masters")
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from API_cred import APICredentials  

# Initialize credentials
creds = APICredentials()

# API setup
API_BASE_URL = "https://api.bacdive.dsmz.de"
FIELDS_INFO_URL = f"{API_BASE_URL}/strain_fields_information"
HEADERS = {
    "Authorization": f"Basic {creds.EMAIL}:{creds.PASSWORD}",
    "Accept": "application/json"
}


# Output files
METADATA_FILE = 'bacdive_filtered_metadata.json'
CHECKPOINT_FILE = 'metadata_checkpoint.txt'

# Logging setup
logging.basicConfig(filename='metadata_download.log', level=logging.INFO, format='%(asctime)s %(message)s')

def get_checkpoint():
    """Retrieve the last processed BacDive ID."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                data = f.read().strip()
                if data:
                    return int(data)
    except ValueError as e:
        logging.warning(f"Invalid checkpoint value: {e}")
    return 1

def save_checkpoint(current_id):
    """Save the current BacDive ID."""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(current_id))

def fetch_fields_info():
    """Fetch the fields information from the BacDive API."""
    response = requests.get(FIELDS_INFO_URL, headers=HEADERS)
    try:
        response_data = response.json()
        return response_data
    except json.JSONDecodeError:
        logging.error(f"Response is not JSON: {response.status_code}, {response.text}")
        return None



def fetch_metadata(bacdive_id, retries=3):
    """Fetch metadata for a specific BacDive ID."""
    for attempt in range(retries):
        try:
            url = f"{API_BASE_URL}/bacteria/{bacdive_id}"
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logging.info(f"BacDive ID {bacdive_id} not found.")
                return None
        except requests.RequestException as e:
            logging.error(f"Attempt {attempt + 1} failed for BacDive ID {bacdive_id}: {e}")
            time.sleep(2)
    logging.error(f"All retries failed for BacDive ID {bacdive_id}")
    return None

def filter_metadata(strain, fields_info):
    """Filter metadata fields based on the API documentation."""
    filtered_data = {}
    for category, fields in fields_info.items():
        if category in strain:
            filtered_data[category] = {}
            for field in fields:
                filtered_data[category][field] = strain[category].get(field, None)
    return filtered_data

def download_metadata_by_id_parallel():
    """Download metadata in parallel using threading."""
    starting_id = get_checkpoint()
    MAX_RECORDS = 10  # Adjust this for your full run
    batch_data = []

    fields_info = fetch_fields_info()
    if not fields_info:
        logging.error("Failed to retrieve fields information.")
        return

    with ThreadPoolExecutor(max_workers=5) as executor:
        while starting_id <= MAX_RECORDS:
            future_to_id = {
                executor.submit(fetch_metadata, bacdive_id): bacdive_id
                for bacdive_id in range(starting_id, starting_id + 10)
            }
            for future in as_completed(future_to_id):
                bacdive_id = future_to_id[future]
                try:
                    strain = future.result()
                    if strain:
                        filtered_strain = filter_metadata(strain, fields_info)
                        filtered_strain['BacDive_ID'] = bacdive_id
                        batch_data.append(filtered_strain)
                except Exception as e:
                    logging.error(f"Error processing BacDive ID {bacdive_id}: {e}")

                save_checkpoint(bacdive_id)

            starting_id += 10

    if batch_data:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as jsonfile:
                try:
                    existing_data = json.load(jsonfile)
                except json.JSONDecodeError:
                    existing_data = []
        else:
            existing_data = []

        existing_data.extend(batch_data)

        with open(METADATA_FILE, 'w', encoding='utf-8') as jsonfile:
            json.dump(existing_data, jsonfile, indent=4)

    print("Metadata download complete.")
    logging.info("Metadata download complete.")

if __name__ == '__main__':
    download_metadata_by_id_parallel()
