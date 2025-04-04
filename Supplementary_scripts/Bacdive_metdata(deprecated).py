import sys
sys.path.append("../Eliah-Masters")
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from bacdive import BacdiveClient
from Supplementary_scripts.API_cred import APICredentials

# Initialize credentials from the class
creds = APICredentials()

# Set up logging
logging.basicConfig(filename='metadata_download.log', level=logging.INFO, format='%(asctime)s %(message)s')

# JSON file to store metadata
METADATA_FILE = 'bacdive_filtered_metadata.json'
CHECKPOINT_FILE = 'metadata_checkpoint.txt'

# Initialize BacDive client
try:
    client = BacdiveClient(creds.EMAIL, creds.PASSWORD)
    print("-- Authentication successful --")
except Exception as e:
    print(f"Error initializing BacDive client: {e}")
    exit(1)

def get_checkpoint():
    """Retrieve the last processed BacDive ID from the checkpoint file."""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                data = f.read().strip()
                if data:
                    return int(data)
    except ValueError as e:
        print(f"Warning: Invalid checkpoint value encountered: {e}")
        logging.warning(f"Invalid checkpoint value encountered: {e}")
    return 1  # Default to ID 1 if no valid checkpoint is found

def save_checkpoint(current_id):
    """Save the current BacDive ID to the checkpoint file."""
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(current_id))

def safe_get(data, keys, default=None):
    """Safely retrieve nested values from a dictionary."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data


def filter_metadata(strain):
    """Filter the metadata to only include the specified fields."""
    filtered_data = {}

    # Extract "Name and taxonomic classification"
    classification = safe_get(strain, ['Name and taxonomic classification'], {})
    filtered_data['Name and taxonomic classification'] = {
        'Species name': safe_get(classification, ['species']),
        'Strain designation': safe_get(classification, ['strain designation']),
        'Variant': safe_get(classification, ['variant']),
        'Type strain': safe_get(classification, ['type strain'])
    }

    # Extract "Morphology"
    morphology = safe_get(strain, ['Morphology'], {})
    filtered_data['Morphology'] = {
        'Colony color': safe_get(morphology, ['colony color']),
        'Colony shape': safe_get(morphology, ['colony shape']),
        'Cultivation medium used': safe_get(morphology, ['cultivation medium used']),
        'Medium Name': safe_get(morphology, ['medium name']),
        'Gram stain': safe_get(morphology, ['Gram stain'])  # Added Gram staining information
    }

    # Extract "Culture and growth conditions"
    culture_growth = safe_get(strain, ['Culture and growth conditions'], {})
    filtered_data['Culture and growth conditions'] = {
        'Culture medium': [safe_get(medium, ['name']) for medium in safe_get(culture_growth, ['culture medium'], [])],
        'Culture medium composition': [safe_get(medium, ['composition']) for medium in safe_get(culture_growth, ['culture medium'], [])],
        'Temperature': [safe_get(temp, ['temperature']) for temp in safe_get(culture_growth, ['culture temp'], [])],
        'pH': safe_get(culture_growth, ['pH'])
    }

    # Extract "Physiology and metabolism"
    physiology = safe_get(strain, ['Physiology and metabolism'], {})
    filtered_data['Physiology and metabolism'] = {
        'Ability of spore formation': safe_get(physiology, ['ability of spore formation']),
        'Name of produced compound': safe_get(physiology, ['name of produced compound']),
        'Halophily / tolerance level': safe_get(physiology, ['halophily / tolerance level']),
        'Salt': safe_get(physiology, ['salt']),
        'Salt conc.': safe_get(physiology, ['salt concentration']),
        'salt concentration unit': safe_get(physiology, ['salt concentration unit']),
        'Tested relation': safe_get(physiology, ['tested relation']),
        'Testresult (salt)': safe_get(physiology, ['testresult']),
        'Murein types': safe_get(physiology, ['murein types']),
        'Observation': safe_get(physiology, ['observation']),
        'Name of tolerated compound': safe_get(physiology, ['name of tolerated compound']),
        'Tolerance percentage': safe_get(physiology, ['tolerance percentage']),
        'Tolerated concentration': safe_get(physiology, ['tolerated concentration']),
        'Oxygen tolerance': safe_get(physiology, ['oxygen tolerance']),
        'Nutrition type': safe_get(physiology, ['nutrition type']),
        'Metabolite (utilization)': safe_get(physiology, ['metabolite utilization']),
        'Chebi ID': safe_get(physiology, ['Chebi-ID']),
        'Utilization activity': safe_get(physiology, ['utilization activity']),
        'Kind of utilization tested': safe_get(physiology, ['kind of utilization tested']),
        'Metabolite (antibiotic)': safe_get(physiology, ['metabolite antibiotic']),
        'Group ID of combined antibiotics': safe_get(physiology, ['group ID of combined antibiotics']),
        'has antibiotic function': safe_get(physiology, ['has antibiotic function']),
        'Antibiotic sensitivity': safe_get(physiology, ['antibiotic sensitivity']),
        'Antibiotic resistance': safe_get(physiology, ['antibiotic resistance']),
        'Metabolite (production)': safe_get(physiology, ['metabolite production']),
        'Production': safe_get(physiology, ['production']),
        'Enzyme': safe_get(physiology, ['enzymes', 'value']),
        'Enzyme activity': safe_get(physiology, ['enzymes', 'activity']),
        'EC number': safe_get(physiology, ['enzymes', 'ec'])
    }

    # Extract "Isolation, sampling and environmental information"
    isolation = safe_get(strain, ['Isolation, sampling and environmental information'], {})
    filtered_data['Isolation, sampling and environmental information'] = {
        '16S_seq_accession': safe_get(isolation, ['16S_seq_accession']),
        'SeqIdentity': safe_get(isolation, ['SeqIdentity']),
        'Host species': safe_get(isolation, ['host species']),
        'Country': safe_get(isolation, ['country'])
    }

    # Extract "Sequence information"
    sequence_info = safe_get(strain, ['Sequence information'], {})
    filtered_data['Sequence information'] = {
        '16S seq. accession number': [safe_get(seq, ['accession']) for seq in safe_get(sequence_info, ['16S sequences'], [])],
        'Genome seq. accession number': [safe_get(genome, ['accession']) for genome in safe_get(sequence_info, ['Genome sequences'], [])]
    }

    # Extract "Antibiotic susceptibility testing"
    antibiotic_testing = safe_get(strain, ['Antibiotic susceptibility testing'], {})
    filtered_data['Antibiotic susceptibility testing'] = antibiotic_testing

    # Extract "fatty acid profile"
    fatty_acid_profile = safe_get(strain, ['fatty acid profile'], {})
    filtered_data['fatty acid profile'] = fatty_acid_profile

    return filtered_data


def fetch_metadata(bacdive_id):
    """Fetch metadata for a given BacDive ID using the initialized BacDive client."""
    try:
        print(f'Fetching metadata for BacDive ID {bacdive_id}...')
        logging.info(f'Fetching metadata for BacDive ID {bacdive_id}...')

        # Use BacDiveClient to fetch metadata
        count = client.search(id=bacdive_id)
        if count == 0:
            print(f"No record found for BacDive ID {bacdive_id}.")
            logging.info(f"No record found for BacDive ID {bacdive_id}.")
            return None

        strain = next(client.retrieve())  
        print(json.dumps(strain, indent=4))  # Log the raw response for debugging


        # Filter out unwanted sections and keys
        filtered_strain = filter_metadata(strain)

        # Add BacDive_ID field separately for easy identification
        filtered_strain['BacDive_ID'] = bacdive_id

        return filtered_strain

    except Exception as e:
        print(f"Error retrieving strain for BacDive ID {bacdive_id}: {e}")
        logging.error(f"Error retrieving strain for BacDive ID {bacdive_id}: {e}")
        return None

def download_metadata_by_id_parallel():
    """Download metadata in parallel using multiple threads."""
    starting_id = get_checkpoint()
    MAX_RECORDS = 2  # Limiting to 100 IDs for a quick test
    batch_data = []

    with ThreadPoolExecutor(max_workers=5) as executor:  # Using 5 threads for testing
        while starting_id <= MAX_RECORDS:
            # Jump to 100000 if we reach 24873
            if 24873 <= starting_id < 100000:
                starting_id = 100000

            future_to_id = {executor.submit(fetch_metadata, bacdive_id): bacdive_id for bacdive_id in range(starting_id, min(starting_id + 10, MAX_RECORDS + 1))}
            for future in as_completed(future_to_id):
                bacdive_id = future_to_id[future]
                try:
                    data = future.result()
                    if data:
                        batch_data.append(data)
                except Exception as e:
                    print(f"Error processing BacDive ID {bacdive_id}: {e}")

                # Save the checkpoint
                save_checkpoint(bacdive_id)

            starting_id += 10

    # Write the batch data to JSON file
    if batch_data:
        if os.path.exists(METADATA_FILE):
            # If the file exists, load existing data and append the new batch
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
