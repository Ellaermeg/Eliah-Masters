# Python Script to Download Genomes from BacDive using API and fetch genome sequences from NCBI
import sys
sys.path.append("../Eliah-Masters")
import os
import time
from bacdive import BacdiveClient
from Bio import Entrez
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

# Directory to save downloaded genomes
OUTPUT_DIR = 'bacdive_genomes'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set NCBI Entrez email and API key parameters
Entrez.email = creds.EMAIL  # Use the email from APICredentials
Entrez.api_key = creds.NCBI_API_KEY  # Use the API key from APICredentials

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
        return None

def download_genomes():
    # Set a broad taxonomy search term to maximize the number of genomes retrieved
    taxonomy_query = 'Bacillus'  # Replace with a broad term that yields many results

    # Loop through all available strains using pagination
    offset = 0
    limit = 50  # Number of strains to retrieve per request
    total_strains = 0
    genomes_downloaded = 0  # Track the number of genomes downloaded

    while genomes_downloaded < 3:  # Stop after downloading 3 genomes
        try:
            # Search for strains with a specific taxonomy query
            count = client.search(taxonomy=taxonomy_query)
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

                bacdive_id = strain.get('General', {}).get('BacDive-ID', 'Unknown ID')  # Adjusted to use the General section
                
                # Check the "Sequence information" section for possible accession numbers
                sequence_info = strain.get('Sequence information', {})
                accession_number = None
                
                if '16S sequences' in sequence_info:
                    accession_number = sequence_info['16S sequences'].get('accession')
                elif 'genome sequence' in sequence_info:
                    accession_number = sequence_info['genome sequence'].get('accession')
                
                # If no accession number is found, continue looking for alternatives
                if accession_number:
                    print(f"Fetching genome from NCBI for accession number {accession_number}")
                    genome_data = fetch_genome_from_ncbi(accession_number)

                    if genome_data:
                        output_path = os.path.join(OUTPUT_DIR, f'{bacdive_id}.fasta')
                        try:
                            with open(output_path, 'w') as genome_file:
                                genome_file.write(genome_data)
                                print(f'Saved genome for BacDive ID {bacdive_id} to {output_path}')
                                genomes_downloaded += 1
                                if genomes_downloaded >= 3:
                                    print("Downloaded 3 genomes, stopping.")
                                    return
                        except Exception as e:
                            print(f"Error writing genome file for BacDive ID {bacdive_id}: {e}")
                    else:
                        print(f"Could not fetch genome for BacDive ID {bacdive_id} (Accession: {accession_number})")
                else:
                    print(f"No accession number found for BacDive ID {bacdive_id}. Checking other keys for accession data.")

                    # Additional debugging - print the content of "Sequence information"
                    if sequence_info:
                        print(f"Sequence information available for BacDive ID {bacdive_id}: {sequence_info}")
                    else:
                        print(f"No sequence information found for BacDive ID {bacdive_id}.")

                # Add a delay to prevent NCBI rate limiting
                time.sleep(1)  # Add a 1-second delay between requests
        except KeyError as e:
            print(f"Error retrieving strains: {e}")

        # Update offset for the next batch
        offset += limit

if __name__ == '__main__':
    download_genomes()