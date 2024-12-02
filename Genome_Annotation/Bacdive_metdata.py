import sys
sys.path.append("../Eliah-Masters")
import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from bacdive import BacdiveClient
from API_cred import APICredentials

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

def filter_metadata(strain):
    """Filter the metadata to only include the specified fields."""
    filtered_data = {}

    # Extract "Name and taxonomic classification"
    classification = strain.get('Name and taxonomic classification', {})
    if isinstance(classification, dict):
        filtered_data['Name and taxonomic classification'] = {
            'Species name': classification.get('species'),
            'Strain designation': classification.get('strain designation'),
            'Variant': classification.get('variant'),
            'Type strain': classification.get('type strain')
        }

    # Extract "Morphology"
    morphology = strain.get('Morphology', {})
    if isinstance(morphology, dict):
        filtered_data['Morphology'] = {
            'Colony color': morphology.get('colony color'),
            'Colony shape': morphology.get('colony shape'),
            'Cultivation medium used': morphology.get('cultivation medium used'),
            'Medium Name': morphology.get('medium name'),
            'Gram stain': morphology.get('Gram stain')  # Added Gram staining information
        }

    # Extract "Culture and growth conditions"
    culture_growth = strain.get('Culture and growth conditions', {})
    if isinstance(culture_growth, dict):
        filtered_data['Culture and growth conditions'] = {
            'Culture medium': [medium.get('name') for medium in culture_growth.get('culture medium', [])],
            'Culture medium composition': [medium.get('composition') for medium in culture_growth.get('culture medium', [])],
            'Temperature': [temp.get('temperature') for temp in culture_growth.get('culture temp', [])],
            'pH': culture_growth.get('pH')
        }

    # Extract "Physiology and metabolism"
    physiology = strain.get('Physiology and metabolism', {})
    if isinstance(physiology, dict):
        filtered_data['Physiology and metabolism'] = {
            'Ability of spore formation': physiology.get('ability of spore formation'),
            'Name of produced compound': physiology.get('name of produced compound'),
            'Halophily / tolerance level': physiology.get('halophily / tolerance level'),
            'Salt': physiology.get('salt'),
            'Salt conc.': physiology.get('salt concentration'),
            'salt concentration unit': physiology.get('salt concentration unit'),
            'Tested relation': physiology.get('tested relation'),
            'Testresult (salt)': physiology.get('testresult'),
            'Murein types': physiology.get('murein types'),
            'Observation': physiology.get('observation'),
            'Name of tolerated compound': physiology.get('name of tolerated compound'),
            'Tolerance percentage': physiology.get('tolerance percentage'),
            'Tolerated concentration': physiology.get('tolerated concentration'),
            'Oxygen tolerance': physiology.get('oxygen tolerance'),
            'Nutrition type': physiology.get('nutrition type'),
            'Metabolite (utilization)': physiology.get('metabolite utilization'),
            'Chebi ID': physiology.get('Chebi-ID'),
            'Utilization activity': physiology.get('utilization activity'),
            'Kind of utilization tested': physiology.get('kind of utilization tested'),
            'Metabolite (antibiotic)': physiology.get('metabolite antibiotic'),
            'Group ID of combined antibiotics': physiology.get('group ID of combined antibiotics'),
            'has antibiotic function': physiology.get('has antibiotic function'),
            'Antibiotic sensitivity': physiology.get('antibiotic sensitivity'),
            'Antibiotic resistance': physiology.get('antibiotic resistance'),
            'Metabolite (production)': physiology.get('metabolite production'),
            'Production': physiology.get('production'),
            'Enzyme': physiology.get('enzymes', {}).get('value'),
            'Enzyme activity': physiology.get('enzymes', {}).get('activity'),
            'EC number': physiology.get('enzymes', {}).get('ec')
        }

    # Extract "Isolation, sampling and environmental information"
    isolation = strain.get('Isolation, sampling and environmental information', {})
    if isinstance(isolation, dict):
        filtered_data['Isolation, sampling and environmental information'] = {
            '16S_seq_accession': isolation.get('16S_seq_accession'),
            'SeqIdentity': isolation.get('SeqIdentity'),
            'Host species': isolation.get('host species'),
            'Country': isolation.get('country')
        }

    # Extract "Sequence information"
    sequence_info = strain.get('Sequence information', {})
    if isinstance(sequence_info, dict):
        filtered_data['Sequence information'] = {
            '16S seq. accession number': [seq.get('accession') for seq in sequence_info.get('16S sequences', [])],
            'Genome seq. accession number': [genome.get('accession') for genome in sequence_info.get('Genome sequences', [])]
        }

    # Extract "Antibiotic susceptibility testing"
    antibiotic_testing = strain.get('Antibiotic susceptibility testing', {})
    if isinstance(antibiotic_testing, dict):
        filtered_data['Antibiotic susceptibility testing'] = antibiotic_testing

    # Extract "fatty acid profile"
    fatty_acid_profile = strain.get('fatty acid profile', {})
    if isinstance(fatty_acid_profile, dict):
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
    MAX_RECORDS = 100  # Limiting to 100 IDs for a quick test
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
