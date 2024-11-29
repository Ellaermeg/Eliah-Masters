import sys
sys.path.append("../Eliah-Masters")
import os
import csv
import time
import logging
from bacdive import BacdiveClient
from API_cred import APICredentials

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
logging.basicConfig(filename='metadata_download.log', level=logging.INFO, format='%(asctime)s %(message)s')

# CSV file to store metadata
METADATA_FILE = 'bacdive_metadata.csv'

# Initialize CSV file and write headers if it doesn't exist yet
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w', newline='', encoding='utf8') as csvfile:
        fieldnames = [
            'BacDive_ID', 'Genus', 'Species', 'Strain_Designation',
            'NCBI_Tax_ID', '16S_Accession', 'Genome_Accession', 'Is_Type_Strain'
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
    starting_id = get_checkpoint()  # Start from the checkpoint if available
    MAX_RECORDS = 97334
    total_strains = 0
    max_consecutive_failures = 500
    consecutive_failures = 0
    batch_data = []

    while starting_id <= MAX_RECORDS and consecutive_failures < max_consecutive_failures:
        if starting_id == 24873:
            # Skip directly to ID 100000 if we reach the problematic range
            starting_id = 100000

        try:
            print(f'Fetching metadata for BacDive ID {starting_id}...')
            logging.info(f'Fetching metadata for BacDive ID {starting_id}...')

            # Fetch metadata for the given ID
            count = client.search(id=starting_id)
        except Exception as e:
            print(f"Error during search for BacDive ID {starting_id}: {e}")
            logging.error(f"Error during search for BacDive ID {starting_id}: {e}")
            consecutive_failures += 1
            starting_id += 1
            continue

        if count == 0:
            print(f"No record found for BacDive ID {starting_id}.")
            logging.info(f"No record found for BacDive ID {starting_id}.")
            consecutive_failures += 1
            starting_id += 1
            continue

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
            is_type_strain = taxonomy_info.get('is type strain', 'Unknown')  # Added `is_type_strain_header`

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

            # Add to batch
            batch_data.append({
                'BacDive_ID': bacdive_id,
                'Genus': genus,
                'Species': species,
                'Strain_Designation': strain_designation,
                'NCBI_Tax_ID': ncbi_tax_id,
                '16S_Accession': sixteen_s_accession,
                'Genome_Accession': genome_accession,
                'Is_Type_Strain': is_type_strain
            })

            total_strains += 1
            consecutive_failures = 0
            print(f"Successfully fetched metadata for BacDive ID {bacdive_id}.")
            logging.info(f"Successfully fetched metadata for BacDive ID {bacdive_id}.")

            # Save batch to CSV periodically
            if len(batch_data) >= 100:
                with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerows(batch_data)
                batch_data = []

            # Save the checkpoint after each successful retrieval
            save_checkpoint(starting_id)

        except Exception as e:
            print(f"Error retrieving strain for BacDive ID {starting_id}: {e}")
            logging.error(f"Error retrieving strain for BacDive ID {starting_id}: {e}")

        # Add a delay to prevent rate limiting
        time.sleep(0.5)

        # Increment ID for the next iteration
        starting_id += 1

    # Write any remaining data in the batch to the CSV
    if batch_data:
        with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(batch_data)

    print(f"Metadata download complete. Total strains fetched: {total_strains}")
    logging.info(f"Metadata download complete. Total strains fetched: {total_strains}")

if __name__ == '__main__':
    download_metadata_by_id()
