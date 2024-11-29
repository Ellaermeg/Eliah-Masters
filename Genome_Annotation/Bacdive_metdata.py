import sys
sys.path.append("../Eliah-Masters")
import os
import csv
import time
from bacdive import BacdiveClient
from API_cred import APICredentials
import logging

# Initialize credentials from the class
creds = APICredentials()

# Initialize BacDive client
try:
    client = BacdiveClient(creds.EMAIL, creds.PASSWORD)
    print("-- Authentication successful --")
except Exception as e:
    print(f"Error initializing BacDive client: {e}")
    exit(1)

# Set up logging
logging.basicConfig(filename='metadata_download_test.log', level=logging.INFO, format='%(asctime)s %(message)s')

# CSV file to store metadata
METADATA_FILE = 'bacdive_metadata_test.csv'

# Initialize CSV file and write headers
with open(METADATA_FILE, 'w', newline='', encoding='utf8') as csvfile:
    fieldnames = [
        'BacDive_ID', 'Genus', 'Species', 'Strain_Designation',
        'NCBI_Tax_ID', '16S_Accession', 'Genome_Accession'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

CHECKPOINT_FILE = 'metadata_checkpoint.txt'

def get_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    return 1  # Start from ID 1 if no checkpoint exists

def save_checkpoint(current_id):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(current_id))

# Function to download metadata by iterating through IDs
def download_metadata_by_id():
    starting_id = 1  # Start from BacDive ID 1
    MAX_RECORDS = 97334  # Set a limit for testing purposes
    total_strains = 0
    max_consecutive_failures = 10000  # Maximum number of consecutive failures before stopping
    consecutive_failures = 0

    while total_strains < MAX_RECORDS and consecutive_failures < max_consecutive_failures:
        try:
            # Attempt to retrieve the strain data using BacDive ID
            print(f'Fetching metadata for BacDive ID {starting_id}...')
            logging.info(f'Fetching metadata for BacDive ID {starting_id}...')

            # Fetch metadata for the given ID
            count = client.search(id=starting_id)
        except Exception as e:
            print(f"Error during search for BacDive ID {starting_id}: {e}")
            logging.error(f"Error during search for BacDive ID {starting_id}: {e}")
            consecutive_failures += 1
            starting_id += 1  # Move to the next ID
            continue

        if count == 0:
            print(f"No record found for BacDive ID {starting_id}.")
            logging.info(f"No record found for BacDive ID {starting_id}.")
            consecutive_failures += 1
            starting_id += 1
            continue

        # Write metadata to CSV if data is found
        try:
            strain = next(client.retrieve())
            general_info = strain.get('General', {})
            taxonomy_info = strain.get('Name and taxonomic classification', {})
            sequence_info = strain.get('Sequence information', {})

            # Retrieve metadata fields
            bacdive_id = general_info.get('BacDive-ID', 'Unknown')
            genus = taxonomy_info.get('genus', 'Unknown')
            species = taxonomy_info.get('species', 'Unknown')
            strain_designation = taxonomy_info.get('strain designation', 'Unknown')
            ncbi_tax_id = general_info.get('NCBI tax id', {}).get('NCBI tax id', 'Unknown')

            # Retrieve accession numbers for 16S sequences and genome sequences
            sixteen_s_accession = None
            genome_accession = None

            if isinstance(sequence_info, dict):
                if '16S sequences' in sequence_info:
                    sixteen_s_data = sequence_info['16S sequences']
                    if isinstance(sixteen_s_data, list) and len(sixteen_s_data) > 0:
                        sixteen_s_accession = sixteen_s_data[0].get('accession')
                    elif isinstance(sixteen_s_data, dict):
                        sixteen_s_accession = sixteen_s_data.get('accession')

                if 'genome sequence' in sequence_info:
                    genome_seq_data = sequence_info['genome sequence']
                    if isinstance(genome_seq_data, list) and len(genome_seq_data) > 0:
                        genome_accession = genome_seq_data[0].get('accession')
                    elif isinstance(genome_seq_data, dict):
                        genome_accession = genome_seq_data.get('accession')

            # Write the row to CSV
            with open(METADATA_FILE, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    'BacDive_ID': bacdive_id,
                    'Genus': genus,
                    'Species': species,
                    'Strain_Designation': strain_designation,
                    'NCBI_Tax_ID': ncbi_tax_id,
                    '16S_Accession': sixteen_s_accession,
                    'Genome_Accession': genome_accession
                })

            total_strains += 1
            consecutive_failures = 0  # Reset consecutive failures since we successfully retrieved data
            print(f"Successfully fetched metadata for BacDive ID {bacdive_id}.")
            logging.info(f"Successfully fetched metadata for BacDive ID {bacdive_id}.")
        except Exception as e:
            print(f"Error retrieving strain for BacDive ID {starting_id}: {e}")
            logging.error(f"Error retrieving strain for BacDive ID {starting_id}: {e}")

        # Add a delay to prevent rate limiting
        time.sleep(0.5)

        # Increment ID for the next iteration
        starting_id += 1

    print(f"Metadata download complete. Total strains fetched: {total_strains}")
    logging.info(f"Metadata download complete. Total strains fetched: {total_strains}")

if __name__ == '__main__':
    download_metadata_by_id()
