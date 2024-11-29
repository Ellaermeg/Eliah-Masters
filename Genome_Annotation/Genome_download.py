import sys
sys.path.append("../Eliah-Masters")
import csv
import os
import time
from Bio import Entrez
from API_cred import APICredentials
import logging

# Initialize credentials from the class
creds = APICredentials()

# Set NCBI Entrez email and API key parameters
Entrez.email = creds.EMAIL
Entrez.api_key = creds.NCBI_API_KEY

# Directory to save downloaded genomes
OUTPUT_DIR = 'ncbi_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(filename='genome_download.log', level=logging.INFO, format='%(asctime)s %(message)s')

# Function to download genome from NCBI
def fetch_genome_from_ncbi(accession_number):
    """Fetch genome sequence from NCBI using the accession number."""
    try:
        handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="fasta", retmode="text")
        sequence_data = handle.read()
        handle.close()
        return sequence_data
    except Exception as e:
        print(f"Error fetching genome from NCBI: {e}")
        logging.error(f"Error fetching genome from NCBI: {e}")
        return None

# Function to download genomes based on metadata
def download_genomes():
    genomes_downloaded = 0

    # Read metadata from CSV file
    with open('bacdive_metadata.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            bacdive_id = row['BacDive_ID']
            genus = row['Genus']
            species = row['Species']
            genome_accession = row['Genome_Accession']

            # Skip if accession number is not available
            if genome_accession == '' or genome_accession == 'None':
                print(f"No genome accession for BacDive ID {bacdive_id}. Skipping.")
                logging.info(f"No genome accession for BacDive ID {bacdive_id}. Skipping.")
                continue

            print(f"Fetching genome for BacDive ID {bacdive_id} with accession number {genome_accession}")
            logging.info(f"Fetching genome for BacDive ID {bacdive_id} with accession number {genome_accession}")

            genome_data = fetch_genome_from_ncbi(genome_accession)

            if genome_data:
                # Construct output file path
                file_name = f"{genus}_{species}_{bacdive_id}.fasta".replace(' ', '_')
                output_path = os.path.join(OUTPUT_DIR, file_name)

                # Save the genome sequence
                with open(output_path, 'w') as genome_file:
                    genome_file.write(genome_data)
                    print(f"Saved genome for BacDive ID {bacdive_id} to {output_path}")
                    logging.info(f"Saved genome for BacDive ID {bacdive_id} to {output_path}")
                    genomes_downloaded += 1

                # Add a delay to prevent NCBI rate limiting
                time.sleep(1)

    print(f"Genome download complete. Total genomes downloaded: {genomes_downloaded}")
    logging.info(f"Genome download complete. Total genomes downloaded: {genomes_downloaded}")

if __name__ == '__main__':
    download_genomes()