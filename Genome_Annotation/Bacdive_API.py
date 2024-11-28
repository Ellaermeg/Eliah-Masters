# Step 1: Python Script to Download Genomes from BacDive using API and fetch genome sequences from NCBI
# This script will query BacDive for bacterial strains and use metadata to fetch sequences from NCBI

import os
import time
from bacdive import BacdiveClient
from Bio import Entrez
import API_login

# Initialize BacDive client
try:
    client = BacdiveClient(API_login.EMAIL, API_login.PASSWORD)
    print("-- Authentication successful --")
except Exception as e:
    print(f"Error initializing BacDive client: {e}")
    exit(1)

# Directory to save downloaded genomes
OUTPUT_DIR = 'bacdive_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set NCBI Entrez email and API key parameters
Entrez.email = API_login.EMAIL  # Use the email from API_login
Entrez.api_key = API_login.NCBI_API_KEY  # Use the API key from API_login  

def fetch_genome_from_ncbi(accession_number):
    """Fetch genome sequence from NCBI given an accession number."""
    try:
        # Fetching the sequence using NCBI Entrez efetch
        handle = Entrez.efetch(db="nucleotide", id=accession_number, rettype="fasta", retmode="text", api_key=API_login.NCBI_API_KEY)
        sequence_data = handle.read()
        handle.close()
        return sequence_data
    except Exception as e:
        print(f"Error fetching genome from NCBI for accession number {accession_number}: {e}")
        return None

def download_genomes():
    # Set a broad taxonomy search term to maximize the number of genomes retrieved
    taxonomy_query = 'Bacillus'  # Replace with a broad term that yields many results

    # Loop through all available strains using pagination
    offset = 0
    limit = 50  # Number of strains to retrieve per request
    total_strains = 0

    while True:
        try:
            # Search for strains with pagination parameters and a taxonomy query
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

                bacdive_id = strain.get('bacdive_id')
                accession_number = strain.get('seq_acc_num')  # Assuming this key contains the accession number

                if accession_number:
                    print(f"Fetching genome from NCBI for accession number {accession_number}")
                    genome_data = fetch_genome_from_ncbi(accession_number)

                    if genome_data:
                        output_path = os.path.join(OUTPUT_DIR, f'{bacdive_id}.fasta')
                        try:
                            with open(output_path, 'w') as genome_file:
                                genome_file.write(genome_data)
                                print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')
                        except Exception as e:
                            print(f"Error writing genome file for BacDive ID {bacdive_id}: {e}")
                    else:
                        print(f"Could not fetch genome for BacDive ID {bacdive_id} (Accession: {accession_number})")
                else:
                    print(f"No accession number found for BacDive ID {bacdive_id}.")

                # Add a delay to prevent NCBI rate limiting
                time.sleep(1)  # Add a 1-second delay between requests
        except KeyError as e:
            print(f"Error retrieving strains: {e}")

        # Update offset for the next batch
        offset += limit

if __name__ == '__main__':
    download_genomes()
