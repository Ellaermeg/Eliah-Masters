# Step 1: Python Script to Download Genomes from BacDive using API and fetch genome sequences from NCBI
# This script will query BacDive for bacterial strains and use metadata to fetch sequences from NCBI

import os
import time
import sys
import logging
import shutil
sys.path.append("../Eliah-Masters")
from bacdive import BacdiveClient
from Bio import Entrez
from API_cred import APICredentials

# Initialize credentials from the class
creds = APICredentials()

# Set up logging
logging.basicConfig(filename='download_genomes.log', level=logging.INFO, 
                    format='%(asctime)s %(message)s')

# Initialize BacDive client
try:
    client = BacdiveClient(creds.EMAIL, creds.PASSWORD)
    print("-- Authentication successful --")
    logging.info("Authentication successful")
except Exception as e:
    print(f"Error initializing BacDive client: {e}")
    logging.error(f"Error initializing BacDive client: {e}")
    exit(1)

# Directory to save downloaded genomes
OUTPUT_DIR = 'bacdive_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set NCBI Entrez email and API key parameters
Entrez.email = creds.EMAIL  # Use the email from APICredentials
Entrez.api_key = creds.NCBI_API_KEY  # Use the API key from APICredentials

MAX_GENOMES_TO_DOWNLOAD = 100000  # Set a maximum limit for the total genomes to download (optional)
CHECKPOINT_FILE = 'checkpoint.txt'

def fetch_genome_from_ncbi(accession_number):
    """Fetch genome sequence from NCBI given an accession number."""
    try:
        # Fetching the sequence using NCBI Entrez efetch
        handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="fasta", retmode="text", api_key=creds.NCBI_API_KEY)
        sequence_data = handle.read()
        handle.close()
        return sequence_data
    except Exception as e:
        print(f"Error fetching genome from NCBI for accession number {accession_number}: {e}")
        logging.error(f"Error fetching genome from NCBI for accession number {accession_number}: {e}")
        return None

def get_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return int(f.read().strip())
    return 0

def save_checkpoint(offset):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(offset))

def download_genomes():
    # Set a broad taxonomy search term to maximize the number of genomes retrieved
    taxonomy_query = 'Bacteria'  # Replace with a broad term that yields many results

    # Loop through all available strains using pagination
    offset = 0
    limit = 50  # Number of strains to retrieve per request
    total_strains = 0
    genomes_downloaded = 0  # Track the number of genomes downloaded

    # Remove the limit of 3 genomes and aim to download as many as possible
    while True:
        try:
            # Search for strains with a specific taxonomy query
            count = client.search(taxonomy=taxonomy_query, offset=offset, limit=limit)
            print(f'Fetching strains {offset} to {offset + limit}...')
        except Exception as e:
            print(f"Error during search: {e}")
            return

        # If no more strains are found, break the loop
        if count == 0:
            print("No more strains found.")
            break

        total_strains += count
        print(f'{count} strains found in this batch, total strains found so far: {total_strains}')

        # Iterate over each strain and use accession number to fetch genome data from NCBI
        try:
            for strain in client.retrieve():
                # Debugging: Print strain data to understand the structure
                print("Debug: Retrieved strain data:")
                print(strain)

                # Retrieve BacDive ID, genus, species, and strain designation for naming
                general_info = strain.get('General', {})
                taxonomy_info = strain.get('Name and taxonomic classification', {})
                
                bacdive_id = general_info.get('BacDive-ID', 'Unknown_ID')
                genus = taxonomy_info.get('genus', 'Unknown_Genus')
                species = taxonomy_info.get('species', 'Unknown_Species')
                strain_designation = taxonomy_info.get('strain designation', 'Unknown_Strain')

                # Construct a meaningful file name
                file_name = f"{genus}_{species}_{strain_designation}_BacDive_{bacdive_id}.fasta"
                # Replace spaces with underscores to make the filename filesystem-friendly
                file_name = file_name.replace(' ', '_')
                output_path = os.path.join(OUTPUT_DIR, file_name)

                accession_number = None

                # Check multiple potential keys for accession numbers
                # Attempt to get genome accession number from multiple sources
                sequence_info = strain.get('Sequence information', {})
                genome_based_predictions = strain.get('Genome-based predictions', {})

                # Check if accession number is available in Sequence information
                if '16S sequences' in sequence_info:
                    accession_number = sequence_info['16S sequences'].get('accession')
                elif 'genome sequence' in sequence_info:
                    accession_number = sequence_info['genome sequence'].get('accession')
                elif 'genome accession' in genome_based_predictions:
                    accession_number = genome_based_predictions.get('genome accession')
                elif strain.get('seq_acc_num'):
                    accession_number = strain.get('seq_acc_num')

                # If no accession number is found, log the available information for manual inspection
                if accession_number:
                    print(f"Fetching genome from NCBI for accession number {accession_number}")
                    genome_data = fetch_genome_from_ncbi(accession_number)

                    if genome_data:
                        try:
                            with open(output_path, 'w') as genome_file:
                                genome_file.write(genome_data)
                                print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')
                                genomes_downloaded += 1
                        except Exception as e:
                            print(f"Error writing genome file for BacDive ID {bacdive_id}: {e}")
                    else:
                        print(f"Could not fetch genome for BacDive ID {bacdive_id} (Accession: {accession_number})")
                else:
                    print(f"No accession number found for BacDive ID {bacdive_id}. Checking other keys for accession data.")

                    # Additional debugging - print the content of "Sequence information" and "Genome-based predictions"
                    if sequence_info:
                        print(f"Sequence information available for BacDive ID {bacdive_id}: {sequence_info}")
                    if genome_based_predictions:
                        print(f"Genome-based predictions available for BacDive ID {bacdive_id}: {genome_based_predictions}")

                # Add a delay to prevent NCBI rate limiting
                time.sleep(1)  # Add a 1-second delay between requests
        except KeyError as e:
            print(f"Error retrieving strains: {e}")

        # Update offset for the next batch
        offset += limit

    # After all genomes are downloaded, zip the OUTPUT_DIR
    zip_output_dir()

def zip_output_dir():
    zip_file_name = OUTPUT_DIR + '.zip'
    try:
        shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
        print(f"Successfully zipped the directory to {zip_file_name}")
    except Exception as e:
        print(f"Error zipping the directory: {e}")

if __name__ == '__main__':
    download_genomes()
