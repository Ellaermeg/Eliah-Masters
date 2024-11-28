# Step 1: Python Script to Download Genomes from BacDive using API
# This script will query the BacDive API to fetch as many bacterial genome data as possible
# Requirements: bacdive, os

import os
from bacdive import BacDiveClient
import API_login

# Initialize BacDive client
try:
    client = BacDiveClient(EMAIL, PASSWORD)
    print("-- Authentication successful --")
except Exception as e:
    print(f"Error initializing BacDive client: {e}")
    exit(1)

# Directory to save downloaded genomes
OUTPUT_DIR = 'bacdive_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_genomes():
    # Specify a search parameter (e.g., search by taxonomy)
    taxonomy_query = 'Bacillus subtilis'  # Replace with your preferred search term

    try:
        # Search for strains matching the given taxonomy
        count = client.search(taxonomy=taxonomy_query)
    except Exception as e:
        print(f"Error during search: {e}")
        return
    
    print(f'{count} strains found.')

    # Iterate over each strain and save genome data if available
    try:
        for strain in client.retrieve():
            # Debugging: Print strain data to understand the structure
            print("Debug: Retrieved strain data:")
            print(strain)

            bacdive_id = strain.get('bacdive_id')
            genome_data = strain.get('genome_sequence')  # Adjust the key based on actual data structure

            if genome_data:
                output_path = os.path.join(OUTPUT_DIR, f'{bacdive_id}.fasta')
                try:
                    with open(output_path, 'w') as genome_file:
                        genome_file.write(genome_data)
                        print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')
                except Exception as e:
                    print(f"Error writing genome file for BacDive ID {bacdive_id}: {e}")
    except KeyError as e:
        print(f"Error retrieving strains: {e}")

if __name__ == '__main__':
    download_genomes()