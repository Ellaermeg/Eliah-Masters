import sys
sys.path.append("../Eliah-Masters")
import os
import csv
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
logging.basicConfig(filename='metadata_download_parallel.log', level=logging.INFO, format='%(asctime)s %(message)s')

# CSV file to store metadata
METADATA_FILE = 'bacdive_metadata_expanded_parallel.csv'

# Initialize CSV file and write headers if it doesn't exist yet
fieldnames = [
    'BacDive_ID', 'Genus', 'Species', 'Strain_Designation', 'NCBI_Tax_ID',
    '16S_Accession', 'Genome_Accession', 'Is_Type_Strain', 'ID_Strains',
    'Gram_Stain', 'pH', 'Incubation_Period', 'Culture_Medium', 'Temperature',
    'Produced_Compound', 'Metabolite_Production', 'Production',
    'Oxygen_Tolerance', 'Nutrition_Type', 'Metabolite_Utilization',
    'Metabolite_Physiological', 'Assay', 'Enzyme', 'Enzyme_Activity_R',
    'Sample_Type_Isolated_From', 'Culture_Collection_No'
]

if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, 'w', newline='', encoding='utf8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

CHECKPOINT_FILE = 'metadata_checkpoint.txt'

def get_checkpoint():
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                data = f.read().strip()
                if data:  # Check if the file is not empty
                    return int(data)
    except ValueError as e:
        print(f"Warning: Invalid checkpoint value encountered: {e}")
        logging.warning(f"Invalid checkpoint value encountered: {e}")

    return 1  # Default to ID 1 if no valid checkpoint is found

def save_checkpoint(current_id):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(current_id))

def fetch_metadata(bacdive_id):
    """Fetch metadata for a given BacDive ID."""
    try:
        print(f'Fetching metadata for BacDive ID {bacdive_id}...')
        logging.info(f'Fetching metadata for BacDive ID {bacdive_id}...')

        # Fetch metadata for the given ID
        count = client.search(id=bacdive_id)
        if count == 0:
            print(f"No record found for BacDive ID {bacdive_id}.")
            logging.info(f"No record found for BacDive ID {bacdive_id}.")
            return None

        strain = next(client.retrieve())
        general_info = strain.get('General', {})
        taxonomy_info = strain.get('Name and taxonomic classification', {})
        culture_growth = strain.get('Culture and growth conditions', {})
        physiology = strain.get('Physiology and metabolism', {})
        isolation_info = strain.get('Isolation, sampling and environmental information', {})
        sequence_info = strain.get('Sequence information', {})
        external_links = strain.get('External links', {})

        # Retrieve metadata fields
        bacdive_id = general_info.get('BacDive-ID', 'Unknown')
        genus = taxonomy_info.get('genus', 'Unknown')
        species = taxonomy_info.get('species', 'Unknown')
        strain_designation = taxonomy_info.get('strain designation', 'Unknown')
        ncbi_tax_id = general_info.get('NCBI tax id', 'Unknown')
        is_type_strain = taxonomy_info.get('is type strain', 'Unknown')

        # Retrieve accession numbers
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

        # Extract other requested metadata fields
        id_strains = general_info.get('id strains', 'Unknown')
        gram_stain = physiology.get('Gram stain', 'Unknown')
        pH = culture_growth.get('pH', 'Unknown')
        incubation_period = culture_growth.get('incubation period', 'Unknown')
        culture_medium = culture_growth.get('culture medium', {}).get('name', 'Unknown') if isinstance(culture_growth.get('culture medium', {}), dict) else 'Unknown'
        temperature = culture_growth.get('temperature', 'Unknown')
        produced_compound = physiology.get('name of produced compound', 'Unknown')
        metabolite_production = physiology.get('Metabolite (production)', 'Unknown')
        production = physiology.get('Production', 'Unknown')
        oxygen_tolerance = physiology.get('oxygen tolerance', 'Unknown')
        nutrition_type = physiology.get('nutrition type', 'Unknown')
        metabolite_utilization = physiology.get('Metabolite (utilization)', 'Unknown')
        metabolite_physiological = physiology.get('Metabolite (physiological)', 'Unknown')
        assay = physiology.get('assay', 'Unknown')
        enzyme = physiology.get('Enzyme', 'Unknown')
        enzyme_activity_r = physiology.get('Enzyme activity R', 'Unknown')
        sample_type_isolated_from = isolation_info.get('sample type/isolated from', 'Unknown')
        culture_collection_no = external_links.get('culture collection no.', 'Unknown')

        return {
            'BacDive_ID': bacdive_id,
            'Genus': genus,
            'Species': species,
            'Strain_Designation': strain_designation,
            'NCBI_Tax_ID': ncbi_tax_id,
            '16S_Accession': sixteen_s_accession,
            'Genome_Accession': genome_accession,
            'Is_Type_Strain': is_type_strain,
            'ID_Strains': id_strains,
            'Gram_Stain': gram_stain,
            'pH': pH,
            'Incubation_Period': incubation_period,
            'Culture_Medium': culture_medium,
            'Temperature': temperature,
            'Produced_Compound': produced_compound,
            'Metabolite_Production': metabolite_production,
            'Production': production,
            'Oxygen_Tolerance': oxygen_tolerance,
            'Nutrition_Type': nutrition_type,
            'Metabolite_Utilization': metabolite_utilization,
            'Metabolite_Physiological': metabolite_physiological,
            'Assay': assay,
            'Enzyme': enzyme,
            'Enzyme_Activity_R': enzyme_activity_r,
            'Sample_Type_Isolated_From': sample_type_isolated_from,
            'Culture_Collection_No': culture_collection_no
        }

    except Exception as e:
        print(f"Error retrieving strain for BacDive ID {bacdive_id}: {e}")
        logging.error(f"Error retrieving strain for BacDive ID {bacdive_id}: {e}")
        return None

# Function to download metadata in parallel
def download_metadata_by_id_parallel():
    starting_id = get_checkpoint()
    MAX_RECORDS = 174334  # The highest known BacDive ID
    batch_data = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        while starting_id <= MAX_RECORDS:
            # Jump to 100000 if we reach 24873
            if 24873 <= starting_id < 100000:
                starting_id = 100000

            future_to_id = {executor.submit(fetch_metadata, bacdive_id): bacdive_id for bacdive_id in range(starting_id, min(starting_id + 500, MAX_RECORDS + 1))}
            for future in as_completed(future_to_id):
                bacdive_id = future_to_id[future]
                try:
                    data = future.result()
                    if data:
                        batch_data.append(data)
                except Exception as e:
                    print(f"Error processing BacDive ID {bacdive_id}: {e}")

                # Save batch to CSV periodically
                if len(batch_data) >= 500:
                    with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writerows(batch_data)
                    batch_data = []

                # Save the checkpoint
                save_checkpoint(bacdive_id)

            starting_id += 500

    # Write any remaining data in the batch to the CSV
    if batch_data:
        with open(METADATA_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerows(batch_data)

    print("Metadata download complete.")
    logging.info("Metadata download complete.")

if __name__ == '__main__':
    download_metadata_by_id_parallel()
